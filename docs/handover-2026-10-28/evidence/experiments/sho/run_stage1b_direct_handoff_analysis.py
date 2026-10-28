#!/usr/bin/env python3
"""Checkpoint-censored N1 handoff analysis of the saved pure-DIRECT run.

The source run persisted six trajectory snapshots, not per-evaluation search
histories.  This analysis therefore reports exact reach-by-5000 outcomes and
interval-censored first-crossing bounds.  It never reconstructs or reruns the
DIRECT search and never substitutes a true-containing rectangle for the
unavailable candidate rectangle without labeling it as a proxy.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_DIR = Path("outputs/sho/stage1B/pure_direct_unique_scaling")
OUTPUT_DIR = Path("outputs/sho/stage1B/direct_handoff_analysis")
THRESHOLDS = (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6)
EXPECTED_BUDGETS = (160, 320, 640, 1280, 2560, 5000)
UNAVAILABLE_REASON = (
    "source artifacts contain checkpoint snapshots only; exact per-evaluation "
    "crossing and the rectangle containing the crossing sample were not persisted"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_membership(distance: float) -> dict[str, bool]:
    return {
        "all": True,
        "interior": distance > 0.2,
        "near_edge": distance <= 0.2,
        "very_near_edge": distance <= 0.05,
    }


def _group_definition(group: str) -> str:
    return {
        "all": "all trajectories",
        "interior": "distance_to_boundary > 0.2",
        "near_edge": "distance_to_boundary <= 0.2",
        "very_near_edge": "distance_to_boundary <= 0.05",
    }[group]


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _median(values: list[float]) -> float | str:
    return float(np.median(values)) if values else ""


def _maximum(values: list[float]) -> float | str:
    return float(np.max(values)) if values else ""


def _minimum(values: list[float]) -> float | str:
    return float(np.min(values)) if values else ""


def _make_per_trajectory_rows(
    by_trajectory: dict[int, list[dict[str, str]]]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for trajectory_id in sorted(by_trajectory):
        snapshots = sorted(
            by_trajectory[trajectory_id],
            key=lambda row: int(row["adaptive_evaluation_budget"]),
        )
        distance = _float(snapshots[0], "distance_to_global_boundary_posthoc")
        membership = _group_membership(distance)
        for threshold in THRESHOLDS:
            hit_index = next(
                (
                    index
                    for index, snapshot in enumerate(snapshots)
                    if _float(snapshot, "best_observed_prefix_loss") <= threshold
                ),
                None,
            )
            hit = snapshots[hit_index] if hit_index is not None else None
            previous = snapshots[hit_index - 1] if hit_index is not None and hit_index > 0 else None
            final = snapshots[-1]
            row: dict[str, object] = {
                "trajectory_id": trajectory_id,
                "loss_threshold": threshold,
                "reference_a_posthoc": snapshots[0]["reference_a_posthoc"],
                "reference_b_posthoc": snapshots[0]["reference_b_posthoc"],
                "distance_to_global_boundary_posthoc": distance,
                "is_interior_gt_0p2": int(membership["interior"]),
                "is_near_edge_le_0p2": int(membership["near_edge"]),
                "is_very_near_edge_le_0p05": int(membership["very_near_edge"]),
                "ever_reached_by_5000": int(hit is not None),
                "crossing_observation_status": (
                    "interval_censored_between_saved_checkpoints"
                    if hit is not None
                    else "right_censored_not_reached_by_5000"
                ),
                "first_crossing_evaluation_exact": "",
                "first_crossing_best_sample_a_exact": "",
                "first_crossing_best_sample_b_exact": "",
                "first_crossing_parameter_error_exact": "",
                "candidate_rectangle_side_a_at_exact_crossing": "",
                "candidate_rectangle_side_b_at_exact_crossing": "",
                "candidate_rectangle_area_at_exact_crossing": "",
                "candidate_rectangle_contains_true_at_exact_crossing": "",
                "true_rectangle_potentially_optimal_at_exact_crossing": "",
                "exact_crossing_fields_unavailable_reason": UNAVAILABLE_REASON,
                "last_saved_best_loss_at_5000": final["best_observed_prefix_loss"],
                "last_saved_best_sample_a_at_5000": final["best_loss_sample_a"],
                "last_saved_best_sample_b_at_5000": final["best_loss_sample_b"],
                "last_saved_best_sample_parameter_error_at_5000": final[
                    "best_loss_sample_parameter_error_posthoc"
                ],
            }
            if hit is not None:
                row.update(
                    {
                        "first_saved_checkpoint_at_or_below_threshold": int(
                            hit["adaptive_evaluation_budget"]
                        ),
                        "last_saved_checkpoint_above_threshold": (
                            int(previous["adaptive_evaluation_budget"])
                            if previous is not None
                            else ""
                        ),
                        "crossing_evaluation_lower_bound_exclusive": (
                            int(previous["adaptive_evaluation_budget"])
                            if previous is not None
                            else ""
                        ),
                        "crossing_evaluation_upper_bound_inclusive": int(
                            hit["adaptive_evaluation_budget"]
                        ),
                        "first_checkpoint_best_observed_loss": hit[
                            "best_observed_prefix_loss"
                        ],
                        "first_checkpoint_best_sample_a": hit["best_loss_sample_a"],
                        "first_checkpoint_best_sample_b": hit["best_loss_sample_b"],
                        "first_checkpoint_best_sample_parameter_error_posthoc": hit[
                            "best_loss_sample_parameter_error_posthoc"
                        ],
                        # These are explicitly proxies: this rectangle contains
                        # the reference, not necessarily the best sampled point.
                        "proxy_true_rectangle_side_a_at_first_checkpoint": hit[
                            "true_rectangle_side_a"
                        ],
                        "proxy_true_rectangle_side_b_at_first_checkpoint": hit[
                            "true_rectangle_side_b"
                        ],
                        "proxy_true_rectangle_area_at_first_checkpoint": hit[
                            "true_rectangle_area"
                        ],
                        "proxy_true_rectangle_half_diagonal_at_first_checkpoint": hit[
                            "true_rectangle_half_diagonal"
                        ],
                        "proxy_true_rectangle_center_distance_at_first_checkpoint": hit[
                            "true_rectangle_center_distance_posthoc"
                        ],
                        "proxy_true_rectangle_contains_reference": 1,
                        "proxy_true_rectangle_is_potentially_optimal_at_first_checkpoint": hit[
                            "true_rectangle_is_potentially_optimal_posthoc"
                        ],
                    }
                )
            else:
                row.update(
                    {
                        "first_saved_checkpoint_at_or_below_threshold": "",
                        "last_saved_checkpoint_above_threshold": 5000,
                        "crossing_evaluation_lower_bound_exclusive": 5000,
                        "crossing_evaluation_upper_bound_inclusive": "",
                        "first_checkpoint_best_observed_loss": "",
                        "first_checkpoint_best_sample_a": "",
                        "first_checkpoint_best_sample_b": "",
                        "first_checkpoint_best_sample_parameter_error_posthoc": "",
                        "proxy_true_rectangle_side_a_at_first_checkpoint": "",
                        "proxy_true_rectangle_side_b_at_first_checkpoint": "",
                        "proxy_true_rectangle_area_at_first_checkpoint": "",
                        "proxy_true_rectangle_half_diagonal_at_first_checkpoint": "",
                        "proxy_true_rectangle_center_distance_at_first_checkpoint": "",
                        "proxy_true_rectangle_contains_reference": "",
                        "proxy_true_rectangle_is_potentially_optimal_at_first_checkpoint": "",
                    }
                )
            output.append(row)
    return output


def _make_summary_rows(
    per_trajectory: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        threshold_rows = [
            row for row in per_trajectory if float(row["loss_threshold"]) == threshold
        ]
        for group in ("all", "interior", "near_edge", "very_near_edge"):
            membership_key = {
                "all": None,
                "interior": "is_interior_gt_0p2",
                "near_edge": "is_near_edge_le_0p2",
                "very_near_edge": "is_very_near_edge_le_0p05",
            }[group]
            group_rows = (
                threshold_rows
                if membership_key is None
                else [row for row in threshold_rows if int(row[membership_key]) == 1]
            )
            reached = [row for row in group_rows if int(row["ever_reached_by_5000"]) == 1]
            never = [row for row in group_rows if int(row["ever_reached_by_5000"]) == 0]
            upper_bounds = [
                float(row["crossing_evaluation_upper_bound_inclusive"]) for row in reached
            ]
            errors = [
                float(row["first_checkpoint_best_sample_parameter_error_posthoc"])
                for row in reached
            ]
            proxy_areas = [
                float(row["proxy_true_rectangle_area_at_first_checkpoint"])
                for row in reached
            ]
            proxy_diagonals = [
                float(row["proxy_true_rectangle_half_diagonal_at_first_checkpoint"])
                for row in reached
            ]
            proxy_max_sides = [
                max(
                    float(row["proxy_true_rectangle_side_a_at_first_checkpoint"]),
                    float(row["proxy_true_rectangle_side_b_at_first_checkpoint"]),
                )
                for row in reached
            ]
            never_distances = [
                float(row["distance_to_global_boundary_posthoc"]) for row in never
            ]
            summaries.append(
                {
                    "loss_threshold": threshold,
                    "subgroup": group,
                    "subgroup_definition": _group_definition(group),
                    "trajectory_count": len(group_rows),
                    "reached_by_5000_count": len(reached),
                    "fraction_ever_reached_by_5000": len(reached) / len(group_rows),
                    "never_reached_by_5000_count": len(never),
                    "never_reached_trajectory_ids": ";".join(
                        str(row["trajectory_id"]) for row in never
                    ),
                    "never_reached_boundary_distance_min": _minimum(never_distances),
                    "never_reached_boundary_distance_median": _median(never_distances),
                    "never_reached_boundary_distance_max": _maximum(never_distances),
                    "first_crossing_evaluations_exact_available": 0,
                    "median_first_crossing_evaluations_exact": "",
                    "worst_first_crossing_evaluations_exact": "",
                    "median_first_saved_checkpoint_upper_bound": _median(upper_bounds),
                    "worst_first_saved_checkpoint_upper_bound": _maximum(upper_bounds),
                    "median_first_checkpoint_best_sample_parameter_error": _median(errors),
                    "worst_first_checkpoint_best_sample_parameter_error": _maximum(errors),
                    "candidate_rectangle_geometry_exact_available": 0,
                    "candidate_region_coverage_exact": "",
                    "median_candidate_rectangle_area_exact": "",
                    "worst_candidate_rectangle_area_exact": "",
                    "proxy_true_rectangle_coverage_at_first_checkpoint": (
                        1.0 if reached else ""
                    ),
                    "proxy_median_true_rectangle_area_at_first_checkpoint": _median(
                        proxy_areas
                    ),
                    "proxy_worst_true_rectangle_area_at_first_checkpoint": _maximum(
                        proxy_areas
                    ),
                    "proxy_median_true_rectangle_half_diagonal_at_first_checkpoint": _median(
                        proxy_diagonals
                    ),
                    "proxy_worst_true_rectangle_half_diagonal_at_first_checkpoint": _maximum(
                        proxy_diagonals
                    ),
                    "proxy_median_true_rectangle_max_side_at_first_checkpoint": _median(
                        proxy_max_sides
                    ),
                    "proxy_worst_true_rectangle_max_side_at_first_checkpoint": _maximum(
                        proxy_max_sides
                    ),
                    "proxy_fraction_true_rectangle_potentially_optimal_at_first_checkpoint": (
                        float(
                            np.mean(
                                [
                                    int(
                                        row[
                                            "proxy_true_rectangle_is_potentially_optimal_at_first_checkpoint"
                                        ]
                                    )
                                    for row in reached
                                ]
                            )
                        )
                        if reached
                        else ""
                    ),
                    "analysis_censoring": "six saved checkpoints only: 160;320;640;1280;2560;5000",
                    "unavailable_exact_fields_reason": UNAVAILABLE_REASON,
                }
            )
    return summaries


def _plot(path: Path, summaries: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    labels = {
        "all": "all",
        "interior": "interior > 0.2",
        "near_edge": "near edge <= 0.2",
        "very_near_edge": "very near <= 0.05",
    }
    for group, label in labels.items():
        rows = [row for row in summaries if row["subgroup"] == group]
        thresholds = np.asarray([float(row["loss_threshold"]) for row in rows])
        success = np.asarray([float(row["fraction_ever_reached_by_5000"]) for row in rows])
        axes[0].plot(thresholds, success, marker="o", linewidth=2.0, label=label)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlabel("prefix-loss handoff threshold")
    axes[0].set_ylabel("fraction reached by 5000 evaluations")
    axes[0].set_title("Handoff success")
    axes[0].legend(frameon=False)

    all_rows = [row for row in summaries if row["subgroup"] == "all"]
    thresholds = np.asarray([float(row["loss_threshold"]) for row in all_rows])
    axes[1].plot(
        thresholds,
        [float(row["median_first_saved_checkpoint_upper_bound"]) for row in all_rows],
        marker="o",
        linewidth=2.0,
        label="median checkpoint upper bound",
    )
    axes[1].plot(
        thresholds,
        [float(row["worst_first_saved_checkpoint_upper_bound"]) for row in all_rows],
        marker="o",
        linewidth=2.0,
        label="worst checkpoint upper bound",
    )
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("prefix-loss handoff threshold")
    axes[1].set_ylabel("saved evaluation checkpoint")
    axes[1].set_title("First crossing is interval-censored")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    source_path = SOURCE_DIR / "per_trajectory_direct.csv"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source_rows = _read_csv(source_path)
    by_trajectory: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_trajectory[int(row["trajectory_id"])].append(row)
    if len(by_trajectory) != 256:
        raise RuntimeError("Expected the existing 256-trajectory DIRECT run")
    for trajectory_id, rows in by_trajectory.items():
        budgets = tuple(
            sorted(int(row["adaptive_evaluation_budget"]) for row in rows)
        )
        if budgets != EXPECTED_BUDGETS:
            raise RuntimeError(
                f"Trajectory {trajectory_id} does not have the expected saved checkpoints"
            )

    per_trajectory = _make_per_trajectory_rows(by_trajectory)
    summaries = _make_summary_rows(per_trajectory)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "direct_handoff_summary.csv", summaries)
    _write_csv(OUTPUT_DIR / "per_trajectory_handoff.csv", per_trajectory)
    _plot(OUTPUT_DIR / "handoff_success_vs_loss_threshold.png", summaries)
    print(
        "Analyzed saved checkpoints only; exact first-crossing histories were not persisted.",
        flush=True,
    )
    for row in summaries:
        if row["subgroup"] in {"all", "near_edge", "very_near_edge"}:
            print(
                f"threshold={float(row['loss_threshold']):.0e} "
                f"group={row['subgroup']} reached={row['reached_by_5000_count']}/"
                f"{row['trajectory_count']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
