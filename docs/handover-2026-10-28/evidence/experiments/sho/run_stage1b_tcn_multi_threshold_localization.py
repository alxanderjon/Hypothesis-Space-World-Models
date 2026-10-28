"""Ordered multi-threshold TCN localization for continuous SHO dynamics."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/tcn_multi_threshold_localization")
CHECKPOINT_PATH = OUTPUT_DIR / "best_validation_checkpoint.pt"
THRESHOLD_COUNT = 31
REGION_COUNT = THRESHOLD_COUNT + 1
MAX_EPOCHS = 150
PATIENCE_EPOCHS = 20
BATCH_SIZE = localization.BATCH_SIZE
LEARNING_RATE = localization.LEARNING_RATE
WIDTH_THRESHOLDS = (0.5, 0.25, 0.125, 0.0625)


def _edges() -> torch.Tensor:
    return torch.linspace(
        localization.LOWER,
        localization.UPPER,
        REGION_COUNT + 1,
        dtype=DTYPE,
    )


def _thresholds() -> torch.Tensor:
    return _edges()[1:-1]


def _targets(c_values: torch.Tensor) -> torch.Tensor:
    return (c_values[:, None] < _thresholds()[None, :]).to(DTYPE)


def _true_regions(c_values: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(c_values, _thresholds())


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


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    positions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    total_loss = 0.0
    total_correct = 0
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        logits = model(positions[start:stop])
        batch_targets = targets[start:stop]
        logits_parts.append(logits)
        total_loss += float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits, batch_targets, reduction="sum"
            ).item()
        )
        total_correct += int(
            ((logits >= 0.0) == batch_targets.to(torch.bool)).sum().item()
        )
    count = positions.shape[0]
    return {
        "loss": total_loss / (count * THRESHOLD_COUNT),
        "bit_accuracy": total_correct / (count * THRESHOLD_COUNT),
        "logits": torch.cat(logits_parts, dim=0),
    }


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    positions: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
) -> tuple[float, float]:
    model.train()
    permutation = torch.randperm(positions.shape[0], generator=generator)
    total_loss = 0.0
    total_correct = 0
    for start in range(0, positions.shape[0], BATCH_SIZE):
        indices = permutation[start : start + BATCH_SIZE]
        logits = model(positions[indices])
        batch_targets = targets[indices]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch_targets
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total_loss += float(loss.detach().item()) * indices.shape[0]
        total_correct += int(
            ((logits.detach() >= 0.0) == batch_targets.to(torch.bool))
            .sum()
            .item()
        )
    count = positions.shape[0]
    return (
        total_loss / count,
        total_correct / (count * THRESHOLD_COUNT),
    )


def _candidate_log_scores(logits: torch.Tensor) -> torch.Tensor:
    """Bernoulli log score for every logically valid ordered bit pattern."""
    log_one = torch.nn.functional.logsigmoid(logits)
    log_zero = torch.nn.functional.logsigmoid(-logits)
    scores = []
    for region in range(REGION_COUNT):
        score = log_zero[:, :region].sum(dim=1)
        score = score + log_one[:, region:].sum(dim=1)
        scores.append(score)
    return torch.stack(scores, dim=1)


def _selected_hull(
    selected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = torch.argmax(selected.to(torch.long), dim=1)
    last = REGION_COUNT - 1 - torch.argmax(
        torch.flip(selected, dims=(1,)).to(torch.long), dim=1
    )
    count = selected.sum(dim=1)
    return first, last, count


def _soft_intervals(
    logits: torch.Tensor, calibration_gap: float
) -> dict[str, torch.Tensor]:
    scores = _candidate_log_scores(logits)
    maximum = scores.max(dim=1, keepdim=True).values
    selected = scores >= maximum - calibration_gap - 1e-12
    first, last, count = _selected_hull(selected)
    edges = _edges()
    lower = edges[first]
    upper = edges[last + 1]
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "selected_region_count": count,
        "first_region": first,
        "last_region": last,
        "scores": scores,
    }


def _valid_patterns() -> torch.Tensor:
    indices = torch.arange(THRESHOLD_COUNT).unsqueeze(0)
    regions = torch.arange(REGION_COUNT).unsqueeze(1)
    return indices >= regions


def _hard_intervals(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    bits = logits >= 0.0
    patterns = _valid_patterns()
    distances = (bits[:, None, :] != patterns[None, :, :]).sum(dim=2)
    best = distances.min(dim=1, keepdim=True).values
    selected = distances == best
    first, last, count = _selected_hull(selected)
    edges = _edges()
    lower = edges[first]
    upper = edges[last + 1]
    return {
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "selected_region_count": count,
        "first_region": first,
        "last_region": last,
        "minimum_hamming_distance": best.squeeze(1),
    }


def _consistency(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    probability_drop = torch.relu(
        probabilities[:, :-1] - probabilities[:, 1:]
    )
    probability_violation = probability_drop > 0.0
    bits = probabilities >= 0.5
    hard_violation = bits[:, :-1] & ~bits[:, 1:]
    return {
        "probabilities": probabilities,
        "probability_violation_count": probability_violation.sum(dim=1),
        "probability_max_drop": probability_drop.max(dim=1).values,
        "hard_violation_count": hard_violation.sum(dim=1),
    }


def _interval_metrics(
    reference_c: torch.Tensor,
    intervals: dict[str, torch.Tensor],
) -> dict[str, Any]:
    contains = (reference_c >= intervals["lower"] - 1e-15) & (
        reference_c <= intervals["upper"] + 1e-15
    )
    width = intervals["width"]
    metrics: dict[str, Any] = {
        "coverage": float(contains.to(DTYPE).mean().item()),
        "median_width": float(width.median().item()),
        "mean_width": float(width.mean().item()),
        "worst_width": float(width.max().item()),
        "contains": contains,
    }
    for threshold in WIDTH_THRESHOLDS:
        metrics[f"fraction_width_below_{threshold:g}"] = float(
            (width < threshold).to(DTYPE).mean().item()
        )
    return metrics


def _subset_row(
    name: str,
    mask: torch.Tensor,
    reference_c: torch.Tensor,
    soft: dict[str, torch.Tensor],
    hard: dict[str, torch.Tensor],
    consistency: dict[str, torch.Tensor],
    **extra: Any,
) -> dict[str, Any]:
    count = int(mask.sum().item())
    soft_contains = (reference_c >= soft["lower"] - 1e-15) & (
        reference_c <= soft["upper"] + 1e-15
    )
    hard_contains = (reference_c >= hard["lower"] - 1e-15) & (
        reference_c <= hard["upper"] + 1e-15
    )
    return {
        "row_type": "coverage_subset",
        "subset": name,
        "count": count,
        "soft_coverage": float(soft_contains[mask].to(DTYPE).mean().item())
        if count
        else float("nan"),
        "hard_coverage": float(hard_contains[mask].to(DTYPE).mean().item())
        if count
        else float("nan"),
        "soft_mean_width": float(soft["width"][mask].mean().item())
        if count
        else float("nan"),
        "hard_mean_width": float(hard["width"][mask].mean().item())
        if count
        else float("nan"),
        "fraction_any_probability_monotonicity_violation": float(
            (consistency["probability_violation_count"][mask] > 0)
            .to(DTYPE)
            .mean()
            .item()
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
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    calibration_gap: float,
    test_c: torch.Tensor,
    soft: dict[str, torch.Tensor],
    hard: dict[str, torch.Tensor],
    consistency: dict[str, torch.Tensor],
    runtime_seconds: float,
) -> list[dict[str, Any]]:
    soft_metrics = _interval_metrics(test_c, soft)
    hard_metrics = _interval_metrics(test_c, hard)
    probability_violation_count = consistency["probability_violation_count"]
    hard_violation_count = consistency["hard_violation_count"]
    rows: list[dict[str, Any]] = [
        {
            "row_type": "overall",
            "thresholds": THRESHOLD_COUNT,
            "regions": REGION_COUNT,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "stop_reason": stop_reason,
            "best_validation_bce": best_validation_loss,
            "train_bce": train_metrics["loss"],
            "validation_bce": validation_metrics["loss"],
            "test_bce": test_metrics["loss"],
            "train_bit_accuracy": train_metrics["bit_accuracy"],
            "validation_bit_accuracy": validation_metrics["bit_accuracy"],
            "test_bit_accuracy": test_metrics["bit_accuracy"],
            "validation_calibration_log_score_gap": calibration_gap,
            "soft_test_coverage": soft_metrics["coverage"],
            "soft_median_interval_width": soft_metrics["median_width"],
            "soft_mean_interval_width": soft_metrics["mean_width"],
            "soft_worst_interval_width": soft_metrics["worst_width"],
            "soft_fraction_width_below_0.5": soft_metrics[
                "fraction_width_below_0.5"
            ],
            "soft_fraction_width_below_0.25": soft_metrics[
                "fraction_width_below_0.25"
            ],
            "soft_fraction_width_below_0.125": soft_metrics[
                "fraction_width_below_0.125"
            ],
            "soft_fraction_width_below_0.0625": soft_metrics[
                "fraction_width_below_0.0625"
            ],
            "hard_test_coverage": hard_metrics["coverage"],
            "hard_median_interval_width": hard_metrics["median_width"],
            "hard_mean_interval_width": hard_metrics["mean_width"],
            "hard_worst_interval_width": hard_metrics["worst_width"],
            "fraction_test_with_any_probability_monotonicity_violation": float(
                (probability_violation_count > 0).to(DTYPE).mean().item()
            ),
            "mean_probability_monotonicity_violations_per_test": float(
                probability_violation_count.to(DTYPE).mean().item()
            ),
            "worst_probability_monotonicity_drop": float(
                consistency["probability_max_drop"].max().item()
            ),
            "fraction_test_with_any_hard_monotonicity_violation": float(
                (hard_violation_count > 0).to(DTYPE).mean().item()
            ),
            "mean_hard_monotonicity_violations_per_test": float(
                hard_violation_count.to(DTYPE).mean().item()
            ),
            "runtime_seconds": runtime_seconds,
        }
    ]
    edges = _edges()
    true_regions = _true_regions(test_c)
    for region in range(REGION_COUNT):
        rows.append(
            _subset_row(
                f"c_region_{region + 1}",
                true_regions == region,
                test_c,
                soft,
                hard,
                consistency,
                c_lower=float(edges[region].item()),
                c_upper=float(edges[region + 1].item()),
            )
        )
    difficult = (true_regions >= 6) & (true_regions <= 8)
    rows.append(
        _subset_row(
            "previous_difficult_negative_regions_7_to_9",
            difficult,
            test_c,
            soft,
            hard,
            consistency,
            c_lower=float(edges[6].item()),
            c_upper=float(edges[9].item()),
        )
    )
    rows.append(
        _subset_row(
            "negative_c",
            test_c < 0.0,
            test_c,
            soft,
            hard,
            consistency,
            c_lower=localization.LOWER,
            c_upper=0.0,
        )
    )
    rows.append(
        _subset_row(
            "nonnegative_c",
            test_c >= 0.0,
            test_c,
            soft,
            hard,
            consistency,
            c_lower=0.0,
            c_upper=localization.UPPER,
        )
    )
    nearest_threshold_distance = torch.min(
        torch.abs(test_c[:, None] - _thresholds()[None, :]), dim=1
    ).values
    for distance in (0.01, 0.02, 0.05):
        rows.append(
            _subset_row(
                f"within_{distance:g}_of_any_threshold",
                nearest_threshold_distance < distance,
                test_c,
                soft,
                hard,
                consistency,
                maximum_distance_to_threshold=distance,
            )
        )
    test_targets = _targets(test_c)
    probabilities = consistency["probabilities"]
    for index, threshold in enumerate(_thresholds()):
        predicted = probabilities[:, index] >= 0.5
        target = test_targets[:, index].to(torch.bool)
        rows.append(
            {
                "row_type": "threshold",
                "threshold_index": index + 1,
                "threshold_c": float(threshold.item()),
                "test_binary_accuracy": float(
                    (predicted == target).to(DTYPE).mean().item()
                ),
                "test_mean_probability": float(
                    probabilities[:, index].mean().item()
                ),
                "test_target_positive_fraction": float(
                    target.to(DTYPE).mean().item()
                ),
            }
        )
    return rows


def _per_test_rows(
    reference_c: torch.Tensor,
    soft: dict[str, torch.Tensor],
    hard: dict[str, torch.Tensor],
    consistency: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    soft_contains = (reference_c >= soft["lower"] - 1e-15) & (
        reference_c <= soft["upper"] + 1e-15
    )
    hard_contains = (reference_c >= hard["lower"] - 1e-15) & (
        reference_c <= hard["upper"] + 1e-15
    )
    true_regions = _true_regions(reference_c)
    nearest_threshold_distance = torch.min(
        torch.abs(reference_c[:, None] - _thresholds()[None, :]), dim=1
    ).values
    probabilities = consistency["probabilities"]
    rows: list[dict[str, Any]] = []
    for index in range(reference_c.shape[0]):
        rows.append(
            {
                "trajectory_index": index,
                "reference_c_star": float(reference_c[index].item()),
                "true_region": int(true_regions[index].item()) + 1,
                "nearest_threshold_distance": float(
                    nearest_threshold_distance[index].item()
                ),
                "soft_lower": float(soft["lower"][index].item()),
                "soft_upper": float(soft["upper"][index].item()),
                "soft_width": float(soft["width"][index].item()),
                "soft_contains_c_star": bool(soft_contains[index].item()),
                "soft_selected_region_count": int(
                    soft["selected_region_count"][index].item()
                ),
                "hard_lower": float(hard["lower"][index].item()),
                "hard_upper": float(hard["upper"][index].item()),
                "hard_width": float(hard["width"][index].item()),
                "hard_contains_c_star": bool(hard_contains[index].item()),
                "hard_selected_region_count": int(
                    hard["selected_region_count"][index].item()
                ),
                "hard_minimum_hamming_distance": int(
                    hard["minimum_hamming_distance"][index].item()
                ),
                "probability_monotonicity_violation_count": int(
                    consistency["probability_violation_count"][index].item()
                ),
                "maximum_probability_monotonicity_drop": float(
                    consistency["probability_max_drop"][index].item()
                ),
                "hard_monotonicity_violation_count": int(
                    consistency["hard_violation_count"][index].item()
                ),
                "threshold_probabilities": ";".join(
                    f"{value:.17g}" for value in probabilities[index].tolist()
                ),
            }
        )
    return rows


def _plot_intervals(
    reference_c: torch.Tensor,
    soft: dict[str, torch.Tensor],
    hard: dict[str, torch.Tensor],
) -> None:
    order = torch.argsort(reference_c)
    true_c = reference_c[order].numpy()
    lower = soft["lower"][order].numpy()
    upper = soft["upper"][order].numpy()
    hard_midpoint = (
        0.5 * (hard["lower"][order] + hard["upper"][order])
    ).numpy()
    contains = (
        (reference_c >= soft["lower"] - 1e-15)
        & (reference_c <= soft["upper"] + 1e-15)
    )[order].numpy()
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.fill_between(
        true_c,
        lower,
        upper,
        color="tab:blue",
        alpha=0.25,
        label="probability-based interval",
    )
    axis.plot(true_c, lower, color="tab:blue", linewidth=0.7)
    axis.plot(true_c, upper, color="tab:blue", linewidth=0.7)
    axis.scatter(
        true_c,
        hard_midpoint,
        s=6,
        color="tab:orange",
        alpha=0.6,
        label="hard-threshold interval midpoint",
    )
    axis.plot(
        [localization.LOWER, localization.UPPER],
        [localization.LOWER, localization.UPPER],
        color="black",
        linestyle="--",
        linewidth=1,
        label="true c*",
    )
    if not bool(contains.all()):
        axis.scatter(
            true_c[~contains],
            true_c[~contains],
            marker="x",
            color="red",
            s=35,
            label="soft interval miss",
        )
    edges = _edges()
    axis.axvspan(
        float(edges[6].item()),
        float(edges[9].item()),
        color="tab:purple",
        alpha=0.08,
        label="previous difficult negative region",
    )
    axis.set_xlim(localization.LOWER, localization.UPPER)
    axis.set_ylim(localization.LOWER, localization.UPPER)
    axis.set_xlabel("reference c*")
    axis.set_ylabel("predicted search interval")
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "true_c_vs_predicted_interval.png", dpi=180)
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
        raise RuntimeError("Stored test c* does not match the deterministic split")

    train_targets = _targets(train_c)
    validation_targets = _targets(validation_c)
    test_targets = _targets(test_c_star)
    torch.manual_seed(localization.MODEL_SEED)
    model = localization.TCNHead(THRESHOLD_COUNT).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(localization.MODEL_SEED + 1)
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    stop_reason = "maximum_epochs"
    start_time = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_bit_accuracy = _train_epoch(
            model, optimizer, train_positions, train_targets, generator
        )
        validation = _evaluate(model, validation_positions, validation_targets)
        improved = float(validation["loss"]) < best_validation_loss
        if improved:
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
                    "validation_bit_accuracy": validation["bit_accuracy"],
                    "thresholds": _thresholds(),
                    "model_seed": localization.MODEL_SEED,
                    "shuffle_seed": localization.MODEL_SEED + 1,
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
            f"train bit acc={train_bit_accuracy:.4f} "
            f"val bit acc={validation['bit_accuracy']:.4f} "
            f"stale={stale_epochs}",
            flush=True,
        )
        if stale_epochs >= PATIENCE_EPOCHS:
            stop_reason = "20_epochs_without_validation_bce_improvement"
            break

    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    train_metrics = _evaluate(model, train_positions, train_targets)
    validation_metrics = _evaluate(
        model, validation_positions, validation_targets
    )
    test_metrics = _evaluate(model, test_positions, test_targets)

    validation_scores = _candidate_log_scores(validation_metrics["logits"])
    validation_true_regions = _true_regions(validation_c)
    validation_gap = validation_scores.max(dim=1).values - validation_scores[
        torch.arange(validation_c.shape[0]), validation_true_regions
    ]
    calibration_gap = float(validation_gap.max().item())
    soft = _soft_intervals(test_metrics["logits"], calibration_gap)
    hard = _hard_intervals(test_metrics["logits"])
    consistency = _consistency(test_metrics["logits"])
    runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        best_epoch,
        epoch,
        stop_reason,
        best_validation_loss,
        train_metrics,
        validation_metrics,
        test_metrics,
        calibration_gap,
        test_c_star,
        soft,
        hard,
        consistency,
        runtime_seconds,
    )
    test_rows = _per_test_rows(test_c_star, soft, hard, consistency)
    _write_csv(OUTPUT_DIR / "threshold_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_test_intervals.csv", test_rows)
    _plot_intervals(test_c_star, soft, hard)
    overall = summary_rows[0]
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "best_epoch": best_epoch,
            "epochs_run": epoch,
            "stop_reason": stop_reason,
            "test_bce": test_metrics["loss"],
            "test_bit_accuracy": test_metrics["bit_accuracy"],
            "soft_coverage": overall["soft_test_coverage"],
            "soft_median_width": overall["soft_median_interval_width"],
            "soft_mean_width": overall["soft_mean_interval_width"],
            "soft_worst_width": overall["soft_worst_interval_width"],
            "hard_coverage": overall["hard_test_coverage"],
            "probability_violation_fraction": overall[
                "fraction_test_with_any_probability_monotonicity_violation"
            ],
            "hard_violation_fraction": overall[
                "fraction_test_with_any_hard_monotonicity_violation"
            ],
            "runtime_seconds": runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
