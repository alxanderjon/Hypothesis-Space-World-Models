"""Evaluate a five-query-trained black-box controller through 15 queries."""

from __future__ import annotations

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

import experiments.sho.run_stage1b_learned_black_box_search as base
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as direct32
from experiments.sho.run_stage1b_continuous_c_search_comparison import _losses
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    os.environ.get(
        "LEARNED_QUERY_BUDGET_OUTPUT_DIR",
        "outputs/sho/stage1B/learned_black_box_query_budget",
    )
)
MAX_EVALUATION_QUERIES = 15
REQUESTED_BUDGETS = (5, 10, 15)
ERROR_THRESHOLDS = (1e-1, 1e-2, 1e-3)
WIDTH_THRESHOLDS = (0.5, 0.25, 0.1, 0.05, 0.025)
WIDE_BRACKET_THRESHOLD = 1.0
C_REGION_COUNT = 8


@torch.no_grad()
def _evaluate_extended(
    model: base.DeepSetsSearchController,
    positions: torch.Tensor,
    fixed_losses: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.eval()
    all_c: list[torch.Tensor] = []
    all_loss: list[torch.Tensor] = []
    for start in range(0, positions.shape[0], base.BATCH_SIZE):
        stop = min(start + base.BATCH_SIZE, positions.shape[0])
        batch_positions = positions[start:stop]
        history_c = base.FIXED_PROBES[None, :].expand(stop - start, -1)
        history_loss = fixed_losses[start:stop]
        proposed_c: list[torch.Tensor] = []
        proposed_loss: list[torch.Tensor] = []
        for _ in range(MAX_EVALUATION_QUERIES):
            query = model(history_c, history_loss)
            query_loss = _losses(query, batch_positions)
            proposed_c.append(query)
            proposed_loss.append(query_loss)
            history_c = torch.cat((history_c, query[:, None]), dim=1)
            history_loss = torch.cat(
                (history_loss, query_loss[:, None]), dim=1
            )
        all_c.append(torch.stack(proposed_c, dim=1))
        all_loss.append(torch.stack(proposed_loss, dim=1))
    proposed_c = torch.cat(all_c, dim=0)
    proposed_loss = torch.cat(all_loss, dim=0)
    return {
        "proposed_c": proposed_c,
        "proposed_loss": proposed_loss,
        "history_c": torch.cat(
            (
                base.FIXED_PROBES[None, :].expand(positions.shape[0], -1),
                proposed_c,
            ),
            dim=1,
        ),
        "history_loss": torch.cat((fixed_losses, proposed_loss), dim=1),
    }


def _c_region(c_value: float) -> int:
    unit = (c_value - base.LOWER) / (base.UPPER - base.LOWER)
    return min(C_REGION_COUNT - 1, max(0, int(math.floor(unit * C_REGION_COUNT))))


def _per_query_rows(
    evaluation: dict[str, torch.Tensor], c_star: torch.Tensor
) -> list[dict[str, Any]]:
    history_c = evaluation["history_c"]
    history_loss = evaluation["history_loss"]
    rows: list[dict[str, Any]] = []
    fixed_count = base.FIXED_PROBES.numel()
    for query_number in range(1, MAX_EVALUATION_QUERIES + 1):
        evaluated_count = fixed_count + query_number
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
            bracket_lower, bracket_upper = base._bracket(sampled_c, best_index)
            proposal_index = fixed_count + query_number - 1
            proposed_loss = sampled_loss[proposal_index]
            prior_best_loss = min(sampled_loss[:proposal_index])
            row: dict[str, Any] = {
                "trajectory_index": trajectory,
                "adaptive_query_number": query_number,
                "requested_budget": query_number in REQUESTED_BUDGETS,
                "total_prefix_loss_evaluations": evaluated_count,
                "reference_c_star_posthoc": reference,
                "c_region_8": _c_region(reference) + 1,
                "proposed_c": sampled_c[proposal_index],
                "proposed_prefix_loss": proposed_loss,
                "new_query_improved_best_loss": proposed_loss
                < prior_best_loss,
                "best_evaluated_c": best_c,
                "absolute_best_c_error": error,
                "best_observed_prefix_loss": sampled_loss[best_index],
                "bracket_lower": bracket_lower,
                "bracket_upper": bracket_upper,
                "bracket_width": bracket_upper - bracket_lower,
                "bracket_contains_c_star": bracket_lower
                <= reference
                <= bracket_upper,
                "wide_bracket_at_least_1": bracket_upper - bracket_lower
                >= WIDE_BRACKET_THRESHOLD,
            }
            for threshold in ERROR_THRESHOLDS:
                exponent = int(round(-math.log10(threshold)))
                row[f"best_c_within_1e_{exponent}"] = error <= threshold
            for threshold in WIDTH_THRESHOLDS:
                label = str(threshold).replace(".", "p")
                row[f"bracket_width_below_{label}"] = (
                    bracket_upper - bracket_lower < threshold
                )
            rows.append(row)
    return rows


def _summary_rows(
    result_rows: list[dict[str, Any]],
    training: dict[str, Any],
    total_runtime_seconds: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    previous_median_width: float | None = None
    previous_mean_width: float | None = None
    for query_number in range(1, MAX_EVALUATION_QUERIES + 1):
        subset = [
            row
            for row in result_rows
            if row["adaptive_query_number"] == query_number
        ]
        errors = [float(row["absolute_best_c_error"]) for row in subset]
        widths = [float(row["bracket_width"]) for row in subset]
        median_width = statistics.median(widths)
        mean_width = statistics.fmean(widths)
        wide = [
            row for row in subset if bool(row["wide_bracket_at_least_1"])
        ]
        summary: dict[str, Any] = {
            "adaptive_query_number": query_number,
            "requested_budget": query_number in REQUESTED_BUDGETS,
            "total_prefix_loss_evaluations": 5 + query_number,
            "test_trajectories": len(subset),
            "bracket_coverage": statistics.fmean(
                float(bool(row["bracket_contains_c_star"])) for row in subset
            ),
            "median_bracket_width": median_width,
            "mean_bracket_width": mean_width,
            "worst_bracket_width": max(widths),
            "median_width_reduction_from_previous_query": ""
            if previous_median_width is None
            else previous_median_width - median_width,
            "mean_width_reduction_from_previous_query": ""
            if previous_mean_width is None
            else previous_mean_width - mean_width,
            "median_best_c_error": statistics.median(errors),
            "worst_best_c_error": max(errors),
            "median_best_observed_prefix_loss": statistics.median(
                float(row["best_observed_prefix_loss"]) for row in subset
            ),
            "worst_best_observed_prefix_loss": max(
                float(row["best_observed_prefix_loss"]) for row in subset
            ),
            "new_query_improved_best_loss_fraction": statistics.fmean(
                float(bool(row["new_query_improved_best_loss"]))
                for row in subset
            ),
            "wide_bracket_at_least_1_count": len(wide),
            "wide_bracket_at_least_1_fraction": len(wide) / len(subset),
            "wide_bracket_negative_c_count": sum(
                float(row["reference_c_star_posthoc"]) < 0.0 for row in wide
            ),
            "wide_bracket_nonnegative_c_count": sum(
                float(row["reference_c_star_posthoc"]) >= 0.0 for row in wide
            ),
            "training_adaptive_queries": base.ADAPTIVE_QUERIES,
            "training_steps": base.MAX_STEPS,
            "best_validation_step": training["best_step"],
            "best_validation_mean_final_best_loss": training[
                "best_validation_mean_final_best_loss"
            ],
            "training_runtime_seconds": training["training_runtime_seconds"],
            "total_runtime_seconds": total_runtime_seconds,
        }
        for threshold in ERROR_THRESHOLDS:
            exponent = int(round(-math.log10(threshold)))
            summary[f"fraction_best_c_within_1e_{exponent}"] = statistics.fmean(
                float(bool(row[f"best_c_within_1e_{exponent}"]))
                for row in subset
            )
        for threshold in WIDTH_THRESHOLDS:
            label = str(threshold).replace(".", "p")
            summary[f"fraction_bracket_width_below_{label}"] = statistics.fmean(
                float(bool(row[f"bracket_width_below_{label}"]))
                for row in subset
            )
        for region in range(1, C_REGION_COUNT + 1):
            summary[f"wide_bracket_count_c_region_{region}"] = sum(
                int(row["c_region_8"]) == region for row in wide
            )
        summaries.append(summary)
        previous_median_width = median_width
        previous_mean_width = mean_width
    return summaries


def _plot(summary_rows: list[dict[str, Any]]) -> None:
    queries = [int(row["adaptive_query_number"]) for row in summary_rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for key, label, marker in (
        ("median_bracket_width", "median width", "o"),
        ("mean_bracket_width", "mean width", "s"),
        ("worst_bracket_width", "worst width", "^"),
    ):
        axes[0].plot(
            queries,
            [float(row[key]) for row in summary_rows],
            marker=marker,
            label=label,
        )
    axes[0].axhline(0.05, color="black", linestyle=":", linewidth=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("bracket width")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        queries,
        [float(row["bracket_coverage"]) for row in summary_rows],
        color="black",
        marker="o",
        linewidth=2,
        label="coverage",
    )
    for threshold in WIDTH_THRESHOLDS:
        label = str(threshold).replace(".", "p")
        axes[1].plot(
            queries,
            [
                float(row[f"fraction_bracket_width_below_{label}"])
                for row in summary_rows
            ],
            marker=".",
            label=f"width < {threshold:g}",
        )
    for budget in REQUESTED_BUDGETS:
        axes[0].axvline(budget, color="gray", linestyle="--", alpha=0.35)
        axes[1].axvline(budget, color="gray", linestyle="--", alpha=0.35)
    axes[1].set_xlabel("learned adaptive queries after five fixed probes")
    axes[1].set_ylabel("test fraction")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_xticks(queries)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "bracket_width_and_coverage_vs_queries.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = direct32._make_exact_labeled_split(
        base.TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_positions, _, _ = direct32._make_exact_labeled_split(
        base.VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_positions, test_labels, test_hidden_c = direct32._make_exact_labeled_split(
        base.TEST_N, TEST_DATA_SEED + 1_810_300
    )
    print("precomputing unchanged five fixed-probe losses", flush=True)
    train_fixed_losses = base._fixed_probe_losses(train_positions)
    validation_fixed_losses = base._fixed_probe_losses(validation_positions)
    test_fixed_losses = base._fixed_probe_losses(test_positions)
    print("replaying unchanged five-query controller training", flush=True)
    model, training = base._train(
        train_positions,
        train_fixed_losses,
        validation_positions,
        validation_fixed_losses,
    )
    print("evaluating frozen controller through 15 queries", flush=True)
    evaluation = _evaluate_extended(model, test_positions, test_fixed_losses)

    # The controller and all search histories are frozen before c* is loaded.
    direct32._verify_test_labels(test_labels)
    test_c_star = base._stored_test_c_star()
    if float(torch.max(torch.abs(test_hidden_c - test_c_star)).item()) > 1e-12:
        raise RuntimeError("Stored test c* does not match deterministic test data")
    result_rows = _per_query_rows(evaluation, test_c_star)
    total_runtime_seconds = time.perf_counter() - start_time
    summary_rows = _summary_rows(
        result_rows, training, total_runtime_seconds
    )
    _write_csv(OUTPUT_DIR / "query_budget_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "per_query_results.csv", result_rows)
    _plot(summary_rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "training_best_step": training["best_step"],
            "requested_budgets": {
                budget: {
                    "coverage": summary_rows[budget - 1]["bracket_coverage"],
                    "median_width": summary_rows[budget - 1][
                        "median_bracket_width"
                    ],
                    "wide_count": summary_rows[budget - 1][
                        "wide_bracket_at_least_1_count"
                    ],
                }
                for budget in REQUESTED_BUDGETS
            },
            "total_runtime_seconds": total_runtime_seconds,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
