"""Adaptive arbitrary-threshold TCN localization for continuous SHO dynamics."""

from __future__ import annotations

import copy
import csv
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from torch import nn

import experiments.sho.run_stage1b_learned_continuous_c_localization as localization
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/tcn_adaptive_threshold_localization")
CHECKPOINT_PATH = OUTPUT_DIR / "best_validation_checkpoint.pt"
MAX_EPOCHS = 150
PATIENCE_EPOCHS = 20
BATCH_SIZE = localization.BATCH_SIZE
LEARNING_RATE = localization.LEARNING_RATE
TAUS_PER_TRAJECTORY = 8
TAU_GRID_COUNT = 65
TAU_GRID_OFFSET = 0.37
MAX_QUERIES = 12
LOW_CONFIDENCE = 0.1
HIGH_CONFIDENCE = 0.9
WIDTH_THRESHOLDS = (0.5, 0.25, 0.125, 0.0625, 0.03125)


class AdaptiveThresholdTCN(nn.Module):
    """Existing TCN feature extractor followed by a tau-conditioned head."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(
            *(localization.TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16])
        )
        self.comparator = nn.Sequential(
            nn.Linear(33, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def encode(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        return self.blocks(features).mean(dim=-1)

    def compare(self, features: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        if tau.shape[0] != features.shape[0]:
            raise ValueError("Feature and tau batch dimensions must match")
        expanded_features = features[:, None, :].expand(-1, tau.shape[1], -1)
        inputs = torch.cat((expanded_features, tau[:, :, None]), dim=2)
        batch_size, query_count, input_size = inputs.shape
        logits = self.comparator(inputs.reshape(-1, input_size))
        return logits.reshape(batch_size, query_count)

    def forward(self, prefix: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.compare(self.encode(prefix), tau)


def _stored_test_c_star() -> torch.Tensor:
    path = Path(
        "outputs/sho/stage1B/learned_continuous_c_localization/"
        "per_test_results.csv"
    )
    with path.open(encoding="utf-8", newline="") as file:
        rows = sorted(
            csv.DictReader(file), key=lambda row: int(row["trajectory_index"])
        )
    return torch.tensor([float(row["c_star"]) for row in rows], dtype=DTYPE)


def _tau_grid() -> torch.Tensor:
    unit = (
        torch.arange(TAU_GRID_COUNT, dtype=DTYPE) + TAU_GRID_OFFSET
    ) / TAU_GRID_COUNT
    return localization.LOWER + (localization.UPPER - localization.LOWER) * unit


def _targets(c_values: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return (c_values[:, None] < tau).to(DTYPE)


def _train_epoch(
    model: AdaptiveThresholdTCN,
    optimizer: torch.optim.Optimizer,
    positions: torch.Tensor,
    c_values: torch.Tensor,
    shuffle_generator: torch.Generator,
    tau_generator: torch.Generator,
) -> tuple[float, float]:
    model.train()
    permutation = torch.randperm(
        positions.shape[0], generator=shuffle_generator
    )
    total_loss = 0.0
    total_correct = 0
    total_decisions = 0
    for start in range(0, positions.shape[0], BATCH_SIZE):
        indices = permutation[start : start + BATCH_SIZE]
        tau = localization.LOWER + (
            localization.UPPER - localization.LOWER
        ) * torch.rand(
            indices.shape[0],
            TAUS_PER_TRAJECTORY,
            generator=tau_generator,
            dtype=DTYPE,
        )
        target = _targets(c_values[indices], tau)
        logits = model(positions[indices], tau)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * indices.shape[0]
        total_correct += int(
            ((logits.detach() >= 0.0) == target.to(torch.bool)).sum().item()
        )
        total_decisions += target.numel()
    return total_loss / positions.shape[0], total_correct / total_decisions


@torch.no_grad()
def _evaluate_grid(
    model: AdaptiveThresholdTCN,
    positions: torch.Tensor,
    c_values: torch.Tensor,
    tau_grid: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    total_loss = 0.0
    total_correct = 0
    total_decisions = 0
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        batch_tau = tau_grid[None, :].expand(stop - start, -1)
        target = _targets(c_values[start:stop], batch_tau)
        logits = model(positions[start:stop], batch_tau)
        logits_parts.append(logits)
        total_loss += float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="sum"
            ).item()
        )
        total_correct += int(
            ((logits >= 0.0) == target.to(torch.bool)).sum().item()
        )
        total_decisions += target.numel()
    logits = torch.cat(logits_parts, dim=0)
    return {
        "loss": total_loss / total_decisions,
        "accuracy": total_correct / total_decisions,
        "logits": logits,
        "probabilities": torch.sigmoid(logits),
    }


@torch.no_grad()
def _adaptive_intervals(
    model: AdaptiveThresholdTCN,
    positions: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    feature_parts = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        feature_parts.append(model.encode(positions[start : start + BATCH_SIZE]))
    features = torch.cat(feature_parts, dim=0)
    count = positions.shape[0]
    lower = torch.full((count,), localization.LOWER, dtype=DTYPE)
    upper = torch.full((count,), localization.UPPER, dtype=DTYPE)
    active = torch.ones(count, dtype=torch.bool)
    query_count = torch.zeros(count, dtype=torch.long)
    stop_reason = ["maximum_queries" for _ in range(count)]
    queried_taus: list[list[float]] = [[] for _ in range(count)]
    queried_probabilities: list[list[float]] = [[] for _ in range(count)]
    lower_history = [lower.clone()]
    upper_history = [upper.clone()]
    active_history = [active.clone()]

    for _ in range(MAX_QUERIES):
        indices = torch.nonzero(active, as_tuple=False).squeeze(1)
        if indices.numel() == 0:
            lower_history.append(lower.clone())
            upper_history.append(upper.clone())
            active_history.append(active.clone())
            continue
        tau = 0.5 * (lower[indices] + upper[indices])
        logits = model.compare(features[indices], tau).squeeze(1)
        probability = torch.sigmoid(logits)
        query_count[indices] += 1
        for local_index, trajectory_index in enumerate(indices.tolist()):
            queried_taus[trajectory_index].append(float(tau[local_index].item()))
            queried_probabilities[trajectory_index].append(
                float(probability[local_index].item())
            )
        high = probability >= HIGH_CONFIDENCE
        low = probability <= LOW_CONFIDENCE
        uncertain = ~(high | low)
        upper[indices[high]] = tau[high]
        lower[indices[low]] = tau[low]
        for trajectory_index in indices[uncertain].tolist():
            stop_reason[trajectory_index] = "uncertain_probability"
        active[indices[uncertain]] = False
        lower_history.append(lower.clone())
        upper_history.append(upper.clone())
        active_history.append(active.clone())

    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "query_count": query_count,
        "stop_reason": stop_reason,
        "queried_taus": queried_taus,
        "queried_probabilities": queried_probabilities,
        "lower_history": torch.stack(lower_history, dim=1),
        "upper_history": torch.stack(upper_history, dim=1),
        "active_history": torch.stack(active_history, dim=1),
    }


def _comparator_diagnostics(
    c_values: torch.Tensor,
    tau_grid: torch.Tensor,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    probabilities = evaluation["probabilities"]
    target = _targets(c_values, tau_grid[None, :]).to(torch.bool)
    high = probabilities >= HIGH_CONFIDENCE
    low = probabilities <= LOW_CONFIDENCE
    confident = high | low
    high_count = int(high.sum().item())
    low_count = int(low.sum().item())
    high_correct = target[high]
    low_correct = ~target[low]
    drops = torch.relu(probabilities[:, :-1] - probabilities[:, 1:])
    strict_violation_count = (drops > 0.0).sum(dim=1)
    meaningful_violation_count = (drops > 0.01).sum(dim=1)
    return {
        "binary_accuracy": evaluation["accuracy"],
        "bce": evaluation["loss"],
        "confident_fraction": float(confident.to(DTYPE).mean().item()),
        "confident_accuracy": float(
            (
                ((high & target) | (low & ~target))[confident]
                .to(DTYPE)
                .mean()
                .item()
            )
        ),
        "high_count": high_count,
        "high_accuracy": float(high_correct.to(DTYPE).mean().item()),
        "high_mean_probability": float(probabilities[high].mean().item()),
        "high_empirical_positive_rate": float(
            target[high].to(DTYPE).mean().item()
        ),
        "high_wrong_count": int((~high_correct).sum().item()),
        "low_count": low_count,
        "low_accuracy": float(low_correct.to(DTYPE).mean().item()),
        "low_mean_probability": float(probabilities[low].mean().item()),
        "low_empirical_positive_rate": float(
            target[low].to(DTYPE).mean().item()
        ),
        "low_wrong_count": int((~low_correct).sum().item()),
        "uncertain_fraction": float((~confident).to(DTYPE).mean().item()),
        "fraction_trajectories_any_strict_monotonicity_violation": float(
            (strict_violation_count > 0).to(DTYPE).mean().item()
        ),
        "mean_strict_monotonicity_violations_per_trajectory": float(
            strict_violation_count.to(DTYPE).mean().item()
        ),
        "fraction_trajectories_any_drop_above_0.01": float(
            (meaningful_violation_count > 0).to(DTYPE).mean().item()
        ),
        "worst_monotonicity_probability_drop": float(drops.max().item()),
    }


def _interval_contains(
    c_values: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    return (c_values >= lower - 1e-15) & (c_values <= upper + 1e-15)


def _subset_row(
    name: str,
    mask: torch.Tensor,
    c_values: torch.Tensor,
    intervals: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    count = int(mask.sum().item())
    contains = _interval_contains(
        c_values, intervals["lower"], intervals["upper"]
    )
    return {
        "row_type": "coverage_subset",
        "subset": name,
        "count": count,
        "coverage": float(contains[mask].to(DTYPE).mean().item())
        if count
        else float("nan"),
        "median_width": float(intervals["width"][mask].median().item())
        if count
        else float("nan"),
        "mean_width": float(intervals["width"][mask].mean().item())
        if count
        else float("nan"),
        "median_queries": float(
            intervals["query_count"][mask].to(DTYPE).median().item()
        )
        if count
        else float("nan"),
        **extra,
    }


def _summary_rows(
    best_epoch: int,
    epochs_run: int,
    stop_reason: str,
    best_validation_loss: float,
    validation_evaluation: dict[str, Any],
    test_evaluation: dict[str, Any],
    comparator: dict[str, Any],
    c_values: torch.Tensor,
    intervals: dict[str, Any],
    runtime_seconds: float,
) -> list[dict[str, Any]]:
    contains = _interval_contains(
        c_values, intervals["lower"], intervals["upper"]
    )
    width = intervals["width"]
    overall: dict[str, Any] = {
        "row_type": "overall",
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "stop_reason": stop_reason,
        "best_validation_bce": best_validation_loss,
        "validation_grid_accuracy": validation_evaluation["accuracy"],
        "test_grid_accuracy": comparator["binary_accuracy"],
        "test_grid_bce": comparator["bce"],
        "adaptive_test_coverage": float(contains.to(DTYPE).mean().item()),
        "median_interval_width": float(width.median().item()),
        "mean_interval_width": float(width.mean().item()),
        "worst_interval_width": float(width.max().item()),
        "median_query_count": float(
            intervals["query_count"].to(DTYPE).median().item()
        ),
        "mean_query_count": float(
            intervals["query_count"].to(DTYPE).mean().item()
        ),
        "fraction_stopped_by_uncertainty": sum(
            reason == "uncertain_probability"
            for reason in intervals["stop_reason"]
        )
        / len(intervals["stop_reason"]),
        "fraction_reaching_maximum_queries": sum(
            reason == "maximum_queries" for reason in intervals["stop_reason"]
        )
        / len(intervals["stop_reason"]),
        "runtime_seconds": runtime_seconds,
        **comparator,
    }
    for threshold in WIDTH_THRESHOLDS:
        overall[f"fraction_width_below_{threshold:g}"] = float(
            (width < threshold).to(DTYPE).mean().item()
        )
    rows: list[dict[str, Any]] = [overall]

    edges = torch.linspace(
        localization.LOWER,
        localization.UPPER,
        33,
        dtype=DTYPE,
    )
    regions = torch.bucketize(c_values, edges[1:-1])
    for region in range(32):
        rows.append(
            _subset_row(
                f"c_region_{region + 1}",
                regions == region,
                c_values,
                intervals,
                c_lower=float(edges[region].item()),
                c_upper=float(edges[region + 1].item()),
            )
        )
    rows.append(
        _subset_row(
            "previous_difficult_negative_regions_7_to_9",
            (regions >= 6) & (regions <= 8),
            c_values,
            intervals,
            c_lower=float(edges[6].item()),
            c_upper=float(edges[9].item()),
        )
    )
    rows.append(
        _subset_row(
            "negative_c",
            c_values < 0.0,
            c_values,
            intervals,
            c_lower=localization.LOWER,
            c_upper=0.0,
        )
    )
    rows.append(
        _subset_row(
            "nonnegative_c",
            c_values >= 0.0,
            c_values,
            intervals,
            c_lower=0.0,
            c_upper=localization.UPPER,
        )
    )
    for distance in (0.01, 0.05, 0.1, 0.2):
        rows.append(
            _subset_row(
                f"absolute_c_below_{distance:g}",
                torch.abs(c_values) < distance,
                c_values,
                intervals,
                maximum_absolute_c=distance,
            )
        )

    lower_history = intervals["lower_history"]
    upper_history = intervals["upper_history"]
    active_history = intervals["active_history"]
    for query_step in range(MAX_QUERIES + 1):
        query_width = upper_history[:, query_step] - lower_history[:, query_step]
        query_contains = _interval_contains(
            c_values,
            lower_history[:, query_step],
            upper_history[:, query_step],
        )
        rows.append(
            {
                "row_type": "query_step",
                "query_step": query_step,
                "coverage": float(query_contains.to(DTYPE).mean().item()),
                "median_width": float(query_width.median().item()),
                "mean_width": float(query_width.mean().item()),
                "fraction_still_active": float(
                    active_history[:, query_step].to(DTYPE).mean().item()
                ),
            }
        )

    probabilities = test_evaluation["probabilities"]
    targets = _targets(c_values, _tau_grid()[None, :]).to(torch.bool)
    calibration_edges = torch.linspace(0.0, 1.0, 11, dtype=DTYPE)
    for index in range(10):
        lower = calibration_edges[index]
        upper = calibration_edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities < upper if index < 9 else probabilities <= upper
        )
        selected_count = int(selected.sum().item())
        rows.append(
            {
                "row_type": "calibration_bin",
                "probability_lower": float(lower.item()),
                "probability_upper": float(upper.item()),
                "count": selected_count,
                "mean_predicted_probability": float(
                    probabilities[selected].mean().item()
                )
                if selected_count
                else float("nan"),
                "empirical_positive_fraction": float(
                    targets[selected].to(DTYPE).mean().item()
                )
                if selected_count
                else float("nan"),
            }
        )
    return rows


def _per_test_rows(
    c_values: torch.Tensor, intervals: dict[str, Any]
) -> list[dict[str, Any]]:
    contains = _interval_contains(
        c_values, intervals["lower"], intervals["upper"]
    )
    rows: list[dict[str, Any]] = []
    for index in range(c_values.shape[0]):
        rows.append(
            {
                "trajectory_index": index,
                "reference_c_star": float(c_values[index].item()),
                "interval_lower": float(intervals["lower"][index].item()),
                "interval_upper": float(intervals["upper"][index].item()),
                "interval_width": float(intervals["width"][index].item()),
                "contains_c_star": bool(contains[index].item()),
                "query_count": int(intervals["query_count"][index].item()),
                "stop_reason": intervals["stop_reason"][index],
                "queried_taus": ";".join(
                    f"{value:.17g}"
                    for value in intervals["queried_taus"][index]
                ),
                "queried_probabilities": ";".join(
                    f"{value:.17g}"
                    for value in intervals["queried_probabilities"][index]
                ),
            }
        )
    return rows


def _plot_width_vs_queries(intervals: dict[str, Any]) -> None:
    widths = intervals["upper_history"] - intervals["lower_history"]
    steps = torch.arange(MAX_QUERIES + 1)
    median = torch.quantile(widths, 0.5, dim=0)
    q10 = torch.quantile(widths, 0.1, dim=0)
    q90 = torch.quantile(widths, 0.9, dim=0)
    mean = widths.mean(dim=0)
    active_fraction = intervals["active_history"].to(DTYPE).mean(dim=0)
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.fill_between(
        steps.numpy(),
        q10.numpy(),
        q90.numpy(),
        alpha=0.2,
        color="tab:blue",
        label="10th–90th percentile width",
    )
    axis.plot(steps, median, marker="o", label="median width")
    axis.plot(steps, mean, linestyle="--", label="mean width")
    axis.set_xlabel("adaptive threshold queries")
    axis.set_ylabel("interval width")
    axis.set_xticks(steps.tolist())
    second = axis.twinx()
    second.plot(
        steps,
        active_fraction,
        color="tab:orange",
        marker="s",
        linewidth=1.2,
        label="fraction still active",
    )
    second.set_ylabel("fraction still active")
    second.set_ylim(-0.02, 1.02)
    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = second.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, loc="upper right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "interval_width_vs_queries.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, train_c = direct32._make_exact_labeled_split(
        localization.TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, validation_c = direct32._make_exact_labeled_split(
        localization.VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels_32, test_hidden_c = direct32._make_exact_labeled_split(
        localization.TEST_N, TEST_DATA_SEED + 1_810_300
    )
    direct32._verify_test_labels(test_labels_32)
    test_c_star = _stored_test_c_star()
    if float(torch.max(torch.abs(test_hidden_c - test_c_star)).item()) > 1e-12:
        raise RuntimeError("Stored test c* does not match deterministic split")

    tau_grid = _tau_grid()
    torch.manual_seed(localization.MODEL_SEED)
    model = AdaptiveThresholdTCN().to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    shuffle_generator = torch.Generator().manual_seed(
        localization.MODEL_SEED + 1
    )
    tau_generator = torch.Generator().manual_seed(localization.MODEL_SEED + 2)
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stop_reason = "maximum_epochs"
    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = _train_epoch(
            model,
            optimizer,
            train_positions,
            train_c,
            shuffle_generator,
            tau_generator,
        )
        validation = _evaluate_grid(
            model, validation_positions, validation_c, tau_grid
        )
        if float(validation["loss"]) < best_validation_loss:
            best_validation_loss = float(validation["loss"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_bce": best_validation_loss,
                    "validation_accuracy": validation["accuracy"],
                    "tau_grid": tau_grid,
                    "taus_per_trajectory": TAUS_PER_TRAJECTORY,
                    "model_seed": localization.MODEL_SEED,
                    "shuffle_seed": localization.MODEL_SEED + 1,
                    "tau_seed": localization.MODEL_SEED + 2,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                },
                CHECKPOINT_PATH,
            )
        else:
            stale_epochs += 1
        print(
            f"epoch={epoch:03d} train BCE={train_loss:.4e} "
            f"val BCE={validation['loss']:.4e} "
            f"train acc={train_accuracy:.4f} "
            f"val acc={validation['accuracy']:.4f} stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= PATIENCE_EPOCHS:
            stop_reason = "20_epochs_without_validation_bce_improvement"
            break

    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    validation_evaluation = _evaluate_grid(
        model, validation_positions, validation_c, tau_grid
    )
    test_evaluation = _evaluate_grid(
        model, test_positions, test_c_star, tau_grid
    )
    comparator = _comparator_diagnostics(
        test_c_star, tau_grid, test_evaluation
    )
    intervals = _adaptive_intervals(model, test_positions)
    runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        best_epoch,
        epoch,
        stop_reason,
        best_validation_loss,
        validation_evaluation,
        test_evaluation,
        comparator,
        test_c_star,
        intervals,
        runtime_seconds,
    )
    test_rows = _per_test_rows(test_c_star, intervals)
    _write_csv(OUTPUT_DIR / "adaptive_threshold_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_test_intervals.csv", test_rows)
    _plot_width_vs_queries(intervals)
    overall = summary_rows[0]
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "stop_reason": stop_reason,
            "test_grid_accuracy": overall["test_grid_accuracy"],
            "confident_accuracy": overall["confident_accuracy"],
            "adaptive_coverage": overall["adaptive_test_coverage"],
            "median_width": overall["median_interval_width"],
            "mean_width": overall["mean_interval_width"],
            "worst_width": overall["worst_interval_width"],
            "median_queries": overall["median_query_count"],
            "strict_monotonicity_violation_fraction": overall[
                "fraction_trajectories_any_strict_monotonicity_violation"
            ],
            "meaningful_monotonicity_violation_fraction": overall[
                "fraction_trajectories_any_drop_above_0.01"
            ],
            "runtime_seconds": runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
