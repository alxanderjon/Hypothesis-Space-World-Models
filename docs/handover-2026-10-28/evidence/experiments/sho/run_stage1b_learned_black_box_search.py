"""Learned black-box c search from evaluated hypothesis/loss pairs."""

from __future__ import annotations

import copy
import csv
import math
import os
import statistics
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

import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P
from experiments.sho.run_stage1b_continuous_c_search_comparison import (
    LOWER,
    UPPER,
    _losses,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    os.environ.get(
        "LEARNED_SEARCH_OUTPUT_DIR",
        "outputs/sho/stage1B/learned_black_box_search",
    )
)
TRAIN_N = int(os.environ.get("LEARNED_SEARCH_TRAIN_N", "4096"))
VALIDATION_N = int(os.environ.get("LEARNED_SEARCH_VALIDATION_N", "512"))
TEST_N = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_STEPS = int(os.environ.get("LEARNED_SEARCH_MAX_STEPS", "2000"))
EVAL_INTERVAL = int(os.environ.get("LEARNED_SEARCH_EVAL_INTERVAL", "100"))
ADAPTIVE_QUERIES = 5
LOG_EPSILON = 1e-30
BASIN_THRESHOLD = 1e-1
ERROR_THRESHOLDS = (1e-1, 1e-2, 1e-3, 1e-4)
FIXED_PROBES = torch.tensor(
    [-1.99, -0.995, 0.0, 0.995, 1.99], dtype=DTYPE
)
MODEL_SEED = SEED + 1_930_000


class DeepSetsSearchController(nn.Module):
    """Permutation-invariant controller over evaluated (c, log loss) pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.element = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, evaluated_c: torch.Tensor, evaluated_loss: torch.Tensor
    ) -> torch.Tensor:
        pairs = torch.stack(
            (evaluated_c, torch.log(evaluated_loss + LOG_EPSILON)), dim=-1
        )
        embedded = self.element(pairs)
        pooled = embedded.mean(dim=1)
        raw = self.head(pooled).squeeze(-1)
        midpoint = 0.5 * (LOWER + UPPER)
        half_width = 0.5 * (UPPER - LOWER)
        return midpoint + half_width * torch.tanh(raw)


def _stored_test_c_star() -> torch.Tensor:
    path = Path(
        "outputs/sho/stage1B/learned_continuous_c_localization/"
        "per_test_results.csv"
    )
    with path.open(encoding="utf-8", newline="") as file:
        rows = sorted(
            csv.DictReader(file), key=lambda row: int(row["trajectory_index"])
        )
    values = torch.tensor([float(row["c_star"]) for row in rows], dtype=DTYPE)
    if values.numel() != TEST_N:
        raise RuntimeError(
            f"Expected {TEST_N} stored test references, found {values.numel()}"
        )
    return values


@torch.no_grad()
def _fixed_probe_losses(positions: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        batch = positions[start : start + BATCH_SIZE]
        probe_losses = []
        for value in FIXED_PROBES:
            c = torch.full(
                (batch.shape[0],), float(value.item()), dtype=DTYPE
            )
            probe_losses.append(_losses(c, batch))
        parts.append(torch.stack(probe_losses, dim=1))
    return torch.cat(parts, dim=0)


def _unroll(
    model: DeepSetsSearchController,
    positions: torch.Tensor,
    fixed_losses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = positions.shape[0]
    history_c = FIXED_PROBES[None, :].expand(count, -1)
    history_loss = fixed_losses
    proposed_c: list[torch.Tensor] = []
    proposed_loss: list[torch.Tensor] = []
    for _ in range(ADAPTIVE_QUERIES):
        query = model(history_c, history_loss)
        query_loss = _losses(query, positions)
        proposed_c.append(query)
        proposed_loss.append(query_loss)
        # Treat the expanded history as black-box observations. The current
        # query is trained through its own real prefix loss above.
        history_c = torch.cat((history_c, query.detach()[:, None]), dim=1)
        history_loss = torch.cat(
            (history_loss, query_loss.detach()[:, None]), dim=1
        )
    return (
        torch.stack(proposed_c, dim=1),
        torch.stack(proposed_loss, dim=1),
        torch.stack(proposed_loss, dim=1).mean(),
    )


@torch.no_grad()
def _evaluate(
    model: DeepSetsSearchController,
    positions: torch.Tensor,
    fixed_losses: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    model.eval()
    all_c: list[torch.Tensor] = []
    all_loss: list[torch.Tensor] = []
    for start in range(0, positions.shape[0], BATCH_SIZE):
        stop = min(start + BATCH_SIZE, positions.shape[0])
        proposed_c, proposed_loss, _ = _unroll(
            model, positions[start:stop], fixed_losses[start:stop]
        )
        all_c.append(proposed_c)
        all_loss.append(proposed_loss)
    proposed_c = torch.cat(all_c, dim=0)
    proposed_loss = torch.cat(all_loss, dim=0)
    history_c = torch.cat(
        (FIXED_PROBES[None, :].expand(positions.shape[0], -1), proposed_c),
        dim=1,
    )
    history_loss = torch.cat((fixed_losses, proposed_loss), dim=1)
    final_best_loss = history_loss.min(dim=1).values
    return {
        "proposed_c": proposed_c,
        "proposed_loss": proposed_loss,
        "history_c": history_c,
        "history_loss": history_loss,
        "mean_query_loss": float(proposed_loss.mean().item()),
        "mean_final_best_loss": float(final_best_loss.mean().item()),
        "median_final_best_loss": float(final_best_loss.median().item()),
        "worst_final_best_loss": float(final_best_loss.max().item()),
    }


def _train(
    train_positions: torch.Tensor,
    train_fixed_losses: torch.Tensor,
    validation_positions: torch.Tensor,
    validation_fixed_losses: torch.Tensor,
) -> tuple[DeepSetsSearchController, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = DeepSetsSearchController().to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    initial_validation = _evaluate(
        model, validation_positions, validation_fixed_losses
    )
    best_score = float(initial_validation["mean_final_best_loss"])
    best_step = 0
    start_time = time.perf_counter()
    print(
        f"step=0000 val mean best loss={best_score:.6e} "
        f"val mean query loss={initial_validation['mean_query_loss']:.6e}",
        flush=True,
    )

    last_training_objective = float("nan")
    for step in range(1, MAX_STEPS + 1):
        model.train()
        indices = torch.randint(
            0,
            train_positions.shape[0],
            (BATCH_SIZE,),
            generator=generator,
        )
        _, _, objective = _unroll(
            model, train_positions[indices], train_fixed_losses[indices]
        )
        optimizer.zero_grad()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        last_training_objective = float(objective.detach().item())

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            validation = _evaluate(
                model, validation_positions, validation_fixed_losses
            )
            score = float(validation["mean_final_best_loss"])
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"step={step:04d} train query loss={last_training_objective:.6e} "
                f"val mean best loss={score:.6e} "
                f"val mean query loss={validation['mean_query_loss']:.6e} "
                f"best_step={best_step}",
                flush=True,
            )

    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_step": best_step,
        "best_validation_mean_final_best_loss": best_score,
        "last_training_query_loss": last_training_objective,
        "training_runtime_seconds": time.perf_counter() - start_time,
    }


def _region_index(c: float) -> int:
    width = (UPPER - LOWER) / 32.0
    return min(31, max(0, int(math.floor((c - LOWER) / width))))


def _bracket(sampled_c: list[float], best_index: int) -> tuple[float, float]:
    center = sampled_c[best_index]
    lower_candidates = [value for value in sampled_c if value < center - 1e-14]
    upper_candidates = [value for value in sampled_c if value > center + 1e-14]
    lower = max(lower_candidates) if lower_candidates else LOWER
    upper = min(upper_candidates) if upper_candidates else UPPER
    return lower, upper


def _result_rows(
    evaluation: dict[str, torch.Tensor | float], c_star: torch.Tensor
) -> list[dict[str, Any]]:
    history_c = evaluation["history_c"]
    history_loss = evaluation["history_loss"]
    assert isinstance(history_c, torch.Tensor)
    assert isinstance(history_loss, torch.Tensor)
    rows: list[dict[str, Any]] = []
    for query_number in range(ADAPTIVE_QUERIES + 1):
        evaluated_count = FIXED_PROBES.numel() + query_number
        for trajectory in range(history_c.shape[0]):
            sampled_c = [
                float(value)
                for value in history_c[trajectory, :evaluated_count].tolist()
            ]
            sampled_loss = [
                float(value)
                for value in history_loss[trajectory, :evaluated_count].tolist()
            ]
            best_index = min(
                range(evaluated_count), key=lambda index: sampled_loss[index]
            )
            best_c = sampled_c[best_index]
            reference = float(c_star[trajectory].item())
            error = abs(best_c - reference)
            bracket_lower, bracket_upper = _bracket(sampled_c, best_index)
            if query_number == 0:
                proposed_c: float | str = ""
                proposed_loss: float | str = ""
                improved: bool | str = ""
            else:
                proposal_index = FIXED_PROBES.numel() + query_number - 1
                proposed_c = sampled_c[proposal_index]
                proposed_loss = sampled_loss[proposal_index]
                improved = proposed_loss < min(sampled_loss[:proposal_index])
            row: dict[str, Any] = {
                "trajectory_index": trajectory,
                "adaptive_query_number": query_number,
                "total_prefix_loss_evaluations": evaluated_count,
                "reference_c_star_posthoc": reference,
                "proposed_c": proposed_c,
                "proposed_prefix_loss": proposed_loss,
                "new_query_improved_best_loss": improved,
                "best_evaluated_c": best_c,
                "absolute_best_c_error": error,
                "best_observed_prefix_loss": sampled_loss[best_index],
                "correct_basin_within_1e_1": error <= BASIN_THRESHOLD,
                "best_c_matches_true_32_bin": _region_index(best_c)
                == _region_index(reference),
                "bracket_lower": bracket_lower,
                "bracket_upper": bracket_upper,
                "bracket_width": bracket_upper - bracket_lower,
                "bracket_contains_c_star": bracket_lower
                <= reference
                <= bracket_upper,
            }
            for threshold in ERROR_THRESHOLDS:
                exponent = int(round(-math.log10(threshold)))
                row[f"best_c_within_1e_{exponent}"] = error <= threshold
            rows.append(row)
    return rows


def _summary_rows(
    result_rows: list[dict[str, Any]],
    training: dict[str, Any],
    total_runtime_seconds: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    parameter_count = sum(
        parameter.numel()
        for parameter in DeepSetsSearchController().parameters()
        if parameter.requires_grad
    )
    for query_number in range(ADAPTIVE_QUERIES + 1):
        subset = [
            row
            for row in result_rows
            if row["adaptive_query_number"] == query_number
        ]
        errors = [float(row["absolute_best_c_error"]) for row in subset]
        losses = [float(row["best_observed_prefix_loss"]) for row in subset]
        widths = [float(row["bracket_width"]) for row in subset]
        summary: dict[str, Any] = {
            "adaptive_query_number": query_number,
            "total_prefix_loss_evaluations": 5 + query_number,
            "test_trajectories": len(subset),
            "median_best_c_error": statistics.median(errors),
            "worst_best_c_error": max(errors),
            "median_best_observed_prefix_loss": statistics.median(losses),
            "mean_best_observed_prefix_loss": statistics.fmean(losses),
            "worst_best_observed_prefix_loss": max(losses),
            "correct_basin_fraction": statistics.fmean(
                float(bool(row["correct_basin_within_1e_1"])) for row in subset
            ),
            "best_c_same_32_bin_fraction": statistics.fmean(
                float(bool(row["best_c_matches_true_32_bin"])) for row in subset
            ),
            "bracket_c_star_coverage": statistics.fmean(
                float(bool(row["bracket_contains_c_star"])) for row in subset
            ),
            "median_bracket_width": statistics.median(widths),
            "worst_bracket_width": max(widths),
            "new_query_improved_best_loss_fraction": ""
            if query_number == 0
            else statistics.fmean(
                float(bool(row["new_query_improved_best_loss"]))
                for row in subset
            ),
            "training_steps": MAX_STEPS,
            "best_validation_step": training["best_step"],
            "best_validation_mean_final_best_loss": training[
                "best_validation_mean_final_best_loss"
            ],
            "training_runtime_seconds": training["training_runtime_seconds"],
            "total_runtime_seconds": total_runtime_seconds,
            "controller_parameter_count": parameter_count,
            "basin_error_threshold": BASIN_THRESHOLD,
            "prior_original_tcn_top1_region_accuracy": 0.646484375,
            "prior_extended_tcn_top1_region_accuracy": 0.82421875,
        }
        for threshold in ERROR_THRESHOLDS:
            exponent = int(round(-math.log10(threshold)))
            summary[f"fraction_best_c_within_1e_{exponent}"] = statistics.fmean(
                float(bool(row[f"best_c_within_1e_{exponent}"]))
                for row in subset
            )
        rows.append(summary)
    return rows


def _plot(summary_rows: list[dict[str, Any]]) -> None:
    query_numbers = [int(row["adaptive_query_number"]) for row in summary_rows]
    median_error = [float(row["median_best_c_error"]) for row in summary_rows]
    worst_error = [float(row["worst_best_c_error"]) for row in summary_rows]
    figure, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(query_numbers, median_error, marker="o", label="median")
    axes[0].plot(query_numbers, worst_error, marker="s", label="worst")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("best evaluated |c - c*|")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for threshold in ERROR_THRESHOLDS:
        exponent = int(round(-math.log10(threshold)))
        fractions = [
            float(row[f"fraction_best_c_within_1e_{exponent}"])
            for row in summary_rows
        ]
        axes[1].plot(
            query_numbers,
            fractions,
            marker="o",
            label=f"within {threshold:g}",
        )
    axes[1].set_xlabel("learned adaptive queries after five fixed probes")
    axes[1].set_ylabel("test fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "c_error_vs_query_number.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = direct32._make_exact_labeled_split(
        TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, _ = direct32._make_exact_labeled_split(
        VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels, test_hidden_c = direct32._make_exact_labeled_split(
        TEST_N, TEST_DATA_SEED + 1_810_300
    )
    print("precomputing five fixed-probe losses", flush=True)
    train_fixed_losses = _fixed_probe_losses(train_positions)
    validation_fixed_losses = _fixed_probe_losses(validation_positions)
    test_fixed_losses = _fixed_probe_losses(test_positions)
    print("training DeepSets black-box controller", flush=True)
    model, training = _train(
        train_positions,
        train_fixed_losses,
        validation_positions,
        validation_fixed_losses,
    )
    test_evaluation = _evaluate(model, test_positions, test_fixed_losses)
    # The search is now frozen. Load c* only for post-hoc evaluation.
    direct32._verify_test_labels(test_labels)
    test_c_star = _stored_test_c_star()
    maximum_reference_difference = float(
        torch.max(torch.abs(test_hidden_c - test_c_star)).item()
    )
    if maximum_reference_difference > 1e-12:
        raise RuntimeError(
            "Stored high-precision test c* does not match the deterministic split"
        )
    result_rows = _result_rows(test_evaluation, test_c_star)
    total_runtime = time.perf_counter() - start_time
    summary_rows = _summary_rows(result_rows, training, total_runtime)
    _write_csv(OUTPUT_DIR / "learned_search_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", result_rows)
    _plot(summary_rows)
    final = summary_rows[-1]
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "training_steps": MAX_STEPS,
            "best_validation_step": training["best_step"],
            "final_median_best_c_error": final["median_best_c_error"],
            "final_worst_best_c_error": final["worst_best_c_error"],
            "final_basin_fraction": final["correct_basin_fraction"],
            "final_bracket_coverage": final["bracket_c_star_coverage"],
            "final_median_bracket_width": final["median_bracket_width"],
            "total_runtime_seconds": total_runtime,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
