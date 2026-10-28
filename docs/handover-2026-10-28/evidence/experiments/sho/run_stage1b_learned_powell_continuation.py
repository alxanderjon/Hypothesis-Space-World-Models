#!/usr/bin/env python3
"""Continue only the evaluation-500 failures of learned Powell to 1500."""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_learned_powell_2d import (
    SUCCESS_LOSS,
    TrajectoryResult,
    _prepare_tasks,
    _run_trajectory,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    _posthoc_reference,
)
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)


SOURCE_RESULTS = Path(
    "outputs/sho/stage1B/learned_powell_2d/per_trajectory_results.csv"
)
OUTPUT_DIR = Path("outputs/sho/stage1B/learned_powell_2d_continuation")
CHECKPOINTS = (500, 750, 1000, 1500)
MAX_EVALUATIONS = 1500
RECENT_WINDOW = 300
PLATEAU_FRACTION = 0.01
TRACKED_THRESHOLDS = (1.0e-16, 1.0e-18, 1.0e-20)
EXPECTED_FAILURE_COUNT = 70
ORIGINAL_SUCCESS_COUNT = 186


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_continuation_task(task: tuple) -> TrajectoryResult:
    return _run_trajectory(
        task,
        max_evaluations=MAX_EVALUATIONS,
        snapshot_budgets=CHECKPOINTS,
    )


def _run_all(tasks: list[tuple], workers: int) -> list[TrajectoryResult]:
    if workers <= 1:
        results = []
        for task in tasks:
            result = _run_continuation_task(task)
            results.append(result)
            print(
                f"continuation {len(results)}/{len(tasks)} success="
                f"{sum(item.success for item in results)}",
                flush=True,
            )
        return results
    results: list[TrajectoryResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_continuation_task, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 5 == 0 or len(results) == len(tasks):
                print(
                    f"continuation {len(results)}/{len(tasks)} success="
                    f"{sum(item.success for item in results)}",
                    flush=True,
                )
    results.sort(key=lambda item: item.trajectory_id)
    return results


def _snapshot(result: TrajectoryResult, budget: int) -> dict[str, object]:
    if budget in result.snapshots:
        snapshot = result.snapshots[budget]
        return {
            "best_loss": float(snapshot["best_loss"]),
            "best_theta": np.asarray(snapshot["best_theta"], dtype=np.float64),
            "cycles": int(snapshot["cycles"]),
            "actual_evaluations": budget,
            "last_best_evaluation": int(snapshot["last_best_evaluation"]),
        }
    if result.success and result.evaluations < budget:
        return {
            "best_loss": result.final_loss,
            "best_theta": result.final_theta.copy(),
            "cycles": result.cycles,
            "actual_evaluations": result.evaluations,
            "last_best_evaluation": result.last_best_evaluation,
        }
    raise RuntimeError(
        f"Trajectory {result.trajectory_id} lacks required snapshot {budget}"
    )


def _verify_replay(
    results: list[TrajectoryResult], source_by_id: dict[int, dict[str, str]]
) -> None:
    disagreements: list[str] = []
    for result in results:
        saved = source_by_id[result.trajectory_id]
        replay = _snapshot(result, 500)
        replay_theta = np.asarray(replay["best_theta"])
        saved_theta = np.asarray((float(saved["final_a"]), float(saved["final_b"])))
        loss_matches = math.isclose(
            float(replay["best_loss"]),
            float(saved["final_loss"]),
            rel_tol=1.0e-13,
            abs_tol=0.0,
        )
        theta_matches = bool(
            np.allclose(replay_theta, saved_theta, rtol=0.0, atol=1.0e-13)
        )
        if not loss_matches or not theta_matches:
            disagreements.append(
                f"ID {result.trajectory_id}: loss {replay['best_loss']} vs "
                f"{saved['final_loss']}; theta {replay_theta} vs {saved_theta}"
            )
    if disagreements:
        raise RuntimeError(
            "Evaluation-500 deterministic replay mismatch:\n" + "\n".join(disagreements)
        )


def _recent_improvement(result: TrajectoryResult) -> float:
    if result.evaluations < MAX_EVALUATIONS:
        return math.nan
    earlier = result.best_loss_history[MAX_EVALUATIONS - RECENT_WINDOW]
    later = result.best_loss_history[MAX_EVALUATIONS]
    return max(0.0, (earlier - later) / max(abs(earlier), 1.0e-300))


def _failure_classification(result: TrajectoryResult) -> tuple[str, int, float]:
    if result.success:
        return "success", 0, math.nan
    improvement = _recent_improvement(result)
    plateau = int(improvement < PLATEAU_FRACTION)
    recent_events = [
        event
        for event in result.line_events
        if int(event["end_evaluation"]) > MAX_EVALUATIONS - RECENT_WINDOW
    ]
    completed_recent = [
        event
        for event in recent_events
        if not (
            int(event["end_evaluation"]) == MAX_EVALUATIONS
            and not bool(event["converged"])
            and not bool(event["hit_line_cap"])
        )
    ]
    convergence_fraction = (
        sum(bool(event["converged"]) for event in completed_recent)
        / len(completed_recent)
        if completed_recent
        else 1.0
    )
    hit_line_cap = any(bool(event["hit_line_cap"]) for event in recent_events)
    recent_direction_update = any(
        evaluation > MAX_EVALUATIONS - RECENT_WINDOW
        for evaluation in result.direction_update_evaluations
    )
    recent_displacement_improvement = any(
        evaluation > MAX_EVALUATIONS - RECENT_WINDOW
        for evaluation in result.displacement_improvement_evaluations
    )
    if improvement >= PLATEAU_FRACTION:
        classification = "still_improving_near_1500"
    elif hit_line_cap or convergence_fraction < 0.8:
        classification = "line_search_convergence_failure"
    elif not recent_direction_update and not recent_displacement_improvement:
        classification = "direction_update_stagnation"
    else:
        classification = "effectively_plateaued"
    return classification, plateau, improvement


def _per_trajectory_rows(
    results: list[TrajectoryResult], references: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reference_by_id = {
        trajectory_id: references[trajectory_id]
        for trajectory_id in range(len(references))
    }
    for result in results:
        reference = reference_by_id[result.trajectory_id]
        classification, plateau, recent_improvement = _failure_classification(result)
        previous_loss: float | None = None
        for budget in CHECKPOINTS:
            snapshot = _snapshot(result, budget)
            theta = np.asarray(snapshot["best_theta"])
            loss = float(snapshot["best_loss"])
            row: dict[str, object] = {
                "trajectory_id": result.trajectory_id,
                "checkpoint_budget": budget,
                "actual_total_evaluations": snapshot["actual_evaluations"],
                "powell_cycles": snapshot["cycles"],
                "best_prefix_loss": loss,
                "best_a": theta[0],
                "best_b": theta[1],
                "parameter_error_posthoc": float(np.linalg.norm(theta - reference)),
                "improved_since_previous_checkpoint": (
                    "" if previous_loss is None else int(loss < previous_loss)
                ),
                "relative_improvement_since_previous_checkpoint": (
                    ""
                    if previous_loss is None
                    else max(0.0, (previous_loss - loss) / max(abs(previous_loss), 1e-300))
                ),
                "last_evaluation_with_new_best": snapshot["last_best_evaluation"],
                "reached_1e-16": int(loss < 1.0e-16),
                "reached_1e-18": int(loss < 1.0e-18),
                "reached_1e-20": int(loss < 1.0e-20),
                "final_failure_classification": (
                    classification if budget == MAX_EVALUATIONS else ""
                ),
                "plateau_last_300_less_than_1pct": (
                    plateau if budget == MAX_EVALUATIONS else ""
                ),
                "relative_improvement_last_300": (
                    recent_improvement if budget == MAX_EVALUATIONS else ""
                ),
                "deterministic_replay_500_verified": 1,
            }
            rows.append(row)
            previous_loss = loss
    return rows


def _summary_rows(per_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for budget in CHECKPOINTS:
        selected = [row for row in per_rows if int(row["checkpoint_budget"]) == budget]
        failures = [row for row in selected if int(row["reached_1e-20"]) == 0]
        errors = [float(row["parameter_error_posthoc"]) for row in failures]
        losses = [float(row["best_prefix_loss"]) for row in failures]
        row: dict[str, object] = {
            "row_type": "checkpoint",
            "checkpoint_budget": budget,
            "continued_trajectories": len(selected),
            "success_count_1e-16": sum(int(item["reached_1e-16"]) for item in selected),
            "success_fraction_1e-16": sum(int(item["reached_1e-16"]) for item in selected) / len(selected),
            "success_count_1e-18": sum(int(item["reached_1e-18"]) for item in selected),
            "success_fraction_1e-18": sum(int(item["reached_1e-18"]) for item in selected) / len(selected),
            "success_count_1e-20": sum(int(item["reached_1e-20"]) for item in selected),
            "success_fraction_1e-20": sum(int(item["reached_1e-20"]) for item in selected) / len(selected),
            "full_256_success_count_1e-20": ORIGINAL_SUCCESS_COUNT + sum(int(item["reached_1e-20"]) for item in selected),
            "remaining_failures_1e-20": len(failures),
            "median_loss_among_failures": float(np.median(losses)) if losses else "",
            "worst_loss_among_failures": max(losses, default=""),
            "median_parameter_error_among_failures_posthoc": float(np.median(errors)) if errors else "",
            "worst_parameter_error_among_failures_posthoc": max(errors, default=""),
            "improved_since_previous_checkpoint_count": sum(
                int(item["improved_since_previous_checkpoint"])
                for item in selected
                if item["improved_since_previous_checkpoint"] != ""
            ),
            "deterministic_replay_500_verified_count": len(selected),
            "plateau_rule": "best loss improvement <1% over evaluations 1200..1500",
        }
        output.append(row)

    final_rows = [
        row for row in per_rows if int(row["checkpoint_budget"]) == MAX_EVALUATIONS
    ]
    for classification in (
        "still_improving_near_1500",
        "effectively_plateaued",
        "line_search_convergence_failure",
        "direction_update_stagnation",
    ):
        output.append(
            {
                "row_type": "final_failure_classification",
                "checkpoint_budget": MAX_EVALUATIONS,
                "continued_trajectories": sum(
                    row["final_failure_classification"] == classification
                    for row in final_rows
                ),
                "plateau_rule": classification,
            }
        )
    return output


def _plot(path: Path, results: list[TrajectoryResult]) -> None:
    grid = np.arange(500, MAX_EVALUATIONS + 1)
    carried = np.empty((len(results), len(grid)), dtype=np.float64)
    for row_index, result in enumerate(results):
        history = np.asarray(result.best_loss_history, dtype=np.float64)
        available = history[500 : min(len(history), MAX_EVALUATIONS + 1)]
        carried[row_index, : len(available)] = available
        carried[row_index, len(available) :] = available[-1]
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    for row in carried:
        axis.semilogy(grid, row, color="0.75", alpha=0.16, linewidth=0.7)
    axis.semilogy(grid, np.median(carried, axis=0), color="tab:blue", linewidth=2.3, label="median")
    axis.semilogy(grid, np.max(carried, axis=0), color="0.25", linewidth=1.7, label="worst")
    id_to_row = {result.trajectory_id: index for index, result in enumerate(results)}
    for trajectory_id, color in ((36, "tab:orange"), (230, "tab:red")):
        axis.semilogy(
            grid,
            carried[id_to_row[trajectory_id]],
            color=color,
            linewidth=1.7,
            label=f"ID {trajectory_id}",
        )
    for threshold in TRACKED_THRESHOLDS:
        axis.axhline(threshold, color="0.72", linestyle="--", linewidth=0.8)
    for checkpoint in CHECKPOINTS[1:]:
        axis.axvline(checkpoint, color="0.84", linestyle=":", linewidth=0.8)
    axis.set_xlabel("total objective evaluations")
    axis.set_ylabel("best prefix loss")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=EXPECTED_FAILURE_COUNT)
    args = parser.parse_args()
    if not SOURCE_RESULTS.exists():
        raise FileNotFoundError(SOURCE_RESULTS)
    source_rows = _read_rows(SOURCE_RESULTS)
    failures = [row for row in source_rows if row["success_1e-20"] == "0"]
    if len(failures) != EXPECTED_FAILURE_COUNT:
        raise RuntimeError(f"Expected 70 saved failures, found {len(failures)}")
    failure_ids = [int(row["trajectory_id"]) for row in failures]
    source_by_id = {int(row["trajectory_id"]): row for row in failures}
    all_tasks, _ = _prepare_tasks()
    selected_tasks = [all_tasks[trajectory_id] for trajectory_id in failure_ids]
    selected_tasks = selected_tasks[: args.limit]
    results = _run_all(selected_tasks, args.workers)
    _verify_replay(results, source_by_id)
    if args.limit != EXPECTED_FAILURE_COUNT:
        print("smoke replay/continuation verified; outputs not written", flush=True)
        return
    _, _, hidden_c = _make_exact_labeled_split(256, FRESH_TEST_SEED)
    references = _posthoc_reference(hidden_c).cpu().numpy()
    per_rows = _per_trajectory_rows(results, references)
    summary_rows = _summary_rows(per_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "continuation_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "continuation_per_trajectory.csv", per_rows)
    _plot(OUTPUT_DIR / "best_loss_vs_evaluations.png", results)
    print(
        f"continuation success={sum(result.success for result in results)}/70; "
        f"wrote {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
