#!/usr/bin/env python3
"""Continue the two unresolved original-eDIRECT adaptive-strip rescues.

The earlier 1,500-evaluation rescue states were not persisted.  This script
loads their saved strips, exactly replays each original eDIRECT search, and
runs the unchanged strip rescue sequence as one uninterrupted search.  The
state at rescue evaluation 1,500 must match the saved endpoint exactly before
the continuation to at most 5,000 rescue evaluations is accepted.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_adaptive_boundary_rescue import (
    RESCUE_RULE,
    SPACING_RULE,
    _replay_original,
)
from experiments.sho.run_stage1b_edirect_vs_direct import (
    EDIRECT_RULE,
    HANDOFF_GATE,
    _cell_proposals,
    _greedy_proposal,
    _initial_design,
    _physical,
    _prefix_losses_numpy,
    _read_csv,
    _write_csv,
)
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    _posthoc_reference,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/adaptive_boundary_rescue_continuation")
SOURCE_CASES = Path(
    "outputs/sho/stage1B/adaptive_boundary_rescue/adaptive_boundary_rescue_cases.csv"
)
TRAJECTORY_IDS = (36, 230)
RECONSTRUCTION_CHECKPOINT = 1500
TOTAL_RESCUE_BUDGET = 5000
SNAPSHOTS = (1500, 2000, 3000, 4000, 5000)
TREND_TOLERANCE = 1.0e-12

CONTINUATION_RULE = (
    "uninterrupted deterministic continuation of the unchanged original "
    "local-RBF eDIRECT rescue sequence in the previously saved d+s strip; "
    "state at evaluation 1500 must exactly reproduce the saved endpoint; "
    "stop immediately at loss<=1e-4; cap=5000 total rescue evaluations"
)
TREND_RULE = (
    "decreasing iff best loss improved by more than 1e-12 relative since the "
    "previous reached snapshot; otherwise plateaued"
)


@dataclass(slots=True)
class ContinuationResult:
    trajectory_id: int
    history: list[dict[str, object]]
    final_evaluations: int
    final_loss: float
    final_best: np.ndarray
    success: bool
    first_gate_evaluation: int | None
    reconstructed_1500_exactly: bool


def _evaluate_point(
    points: np.ndarray,
    losses: np.ndarray,
    proposal: np.ndarray,
    position: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    if len(points) and np.min(np.linalg.norm(points - proposal[None, :], axis=1)) <= 1e-12:
        return points, losses, False, math.nan
    physical = bounds[:, 0] + proposal[None, :] * (bounds[:, 1] - bounds[:, 0])
    query_loss = float(_prefix_losses_numpy(physical, position)[0])
    return (
        np.concatenate((points, proposal[None, :]), axis=0),
        np.concatenate((losses, np.asarray([query_loss]))),
        True,
        query_loss,
    )


def _history_row(
    trajectory_id: int,
    evaluations: int,
    query_loss: float | str,
    points: np.ndarray,
    losses: np.ndarray,
    bounds: np.ndarray,
    event: str,
) -> dict[str, object]:
    best_index = int(np.argmin(losses))
    best = bounds[:, 0] + points[best_index] * (bounds[:, 1] - bounds[:, 0])
    return {
        "trajectory_id": trajectory_id,
        "rescue_evaluations": evaluations,
        "event": event,
        "query_loss": query_loss,
        "best_prefix_loss": float(losses[best_index]),
        "best_a": float(best[0]),
        "best_b": float(best[1]),
        "new_best": int(
            evaluations > 0
            and isinstance(query_loss, float)
            and math.isfinite(query_loss)
            and query_loss == float(losses[best_index])
        ),
    }


def _continue_case(
    arguments: tuple[int, np.ndarray, dict[str, str]]
) -> ContinuationResult:
    trajectory_id, position, saved = arguments
    original = _replay_original(trajectory_id, position)
    original_best = _physical(original.points[original.best_index][None, :])[0]
    if (
        float(original.losses[original.best_index])
        != float(saved["loss_before_rescue"])
        or not np.array_equal(
            original_best,
            np.asarray(
                (float(saved["original_best_a"]), float(saved["original_best_b"]))
            ),
        )
    ):
        raise RuntimeError(f"trajectory {trajectory_id} original replay mismatch")

    bounds = np.asarray(
        (
            (float(saved["strip_a_lower"]), float(saved["strip_a_upper"])),
            (float(saved["strip_b_lower"]), float(saved["strip_b_upper"])),
        ),
        dtype=np.float64,
    )
    original_physical = _physical(original.points)
    inside = np.all(
        (original_physical >= bounds[:, 0] - 1.0e-14)
        & (original_physical <= bounds[:, 1] + 1.0e-14),
        axis=1,
    )
    seed_physical = original_physical[inside]
    points = (seed_physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    losses = original.losses[inside].copy()
    if len(points) != int(saved["reused_original_samples_in_strip"]):
        raise RuntimeError(f"trajectory {trajectory_id} strip seed mismatch")

    evaluations = 0
    history = [
        _history_row(
            trajectory_id, 0, "", points, losses, bounds, "reused_strip_state"
        )
    ]
    reconstructed = False

    def record(proposal: np.ndarray, event: str) -> bool:
        nonlocal points, losses, evaluations, reconstructed
        points, losses, used, query_loss = _evaluate_point(
            points, losses, proposal, position, bounds
        )
        if not used:
            return False
        evaluations += 1
        history.append(
            _history_row(
                trajectory_id,
                evaluations,
                query_loss,
                points,
                losses,
                bounds,
                event,
            )
        )
        if evaluations == RECONSTRUCTION_CHECKPOINT:
            endpoint = history[-1]
            expected_best = np.asarray(
                (float(saved["final_best_a"]), float(saved["final_best_b"]))
            )
            actual_best = np.asarray((endpoint["best_a"], endpoint["best_b"]))
            if (
                float(endpoint["best_prefix_loss"]) != float(saved["final_loss"])
                or not np.array_equal(actual_best, expected_best)
            ):
                raise RuntimeError(
                    f"trajectory {trajectory_id} rescue-1500 replay mismatch"
                )
            reconstructed = True
        return True

    for proposal in _initial_design():
        if evaluations >= TOTAL_RESCUE_BUDGET:
            break
        record(proposal, "initial_design")
        if float(np.min(losses)) <= HANDOFF_GATE:
            break

    iteration = 0
    while (
        evaluations < TOTAL_RESCUE_BUDGET
        and float(np.min(losses)) > HANDOFF_GATE
    ):
        proposals = _cell_proposals(points, losses, trajectory_id, iteration)
        for proposal in proposals:
            if evaluations >= TOTAL_RESCUE_BUDGET:
                break
            record(proposal, "cell_proposal")
            if float(np.min(losses)) <= HANDOFF_GATE:
                break
        if (
            evaluations >= TOTAL_RESCUE_BUDGET
            or float(np.min(losses)) <= HANDOFF_GATE
        ):
            break
        proposal = _greedy_proposal(points, losses, trajectory_id, iteration)
        if proposal is not None:
            record(proposal, "greedy_proposal")
        iteration += 1

    if not reconstructed:
        raise RuntimeError(f"trajectory {trajectory_id} ended before replay checkpoint")
    final = history[-1]
    success = float(final["best_prefix_loss"]) <= HANDOFF_GATE
    first_gate = int(final["rescue_evaluations"]) if success else None
    return ContinuationResult(
        trajectory_id=trajectory_id,
        history=history,
        final_evaluations=int(final["rescue_evaluations"]),
        final_loss=float(final["best_prefix_loss"]),
        final_best=np.asarray((float(final["best_a"]), float(final["best_b"]))),
        success=success,
        first_gate_evaluation=first_gate,
        reconstructed_1500_exactly=reconstructed,
    )


def _load_cases() -> dict[int, dict[str, str]]:
    rows = _read_csv(SOURCE_CASES)
    selected = {
        int(row["trajectory_id"]): row
        for row in rows
        if int(row["trajectory_id"]) in TRAJECTORY_IDS
    }
    if set(selected) != set(TRAJECTORY_IDS):
        raise RuntimeError("Saved adaptive rescue cases are missing")
    if any(row["rescued"] != "0" for row in selected.values()):
        raise RuntimeError("A requested trajectory was not unresolved")
    return selected


def _lookup_history(
    result: ContinuationResult, evaluation: int
) -> dict[str, object] | None:
    if evaluation > result.final_evaluations:
        return None
    return result.history[evaluation]


def _summary_rows(
    results: list[ContinuationResult], references: np.ndarray
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for result in results:
        reference = references[result.trajectory_id]
        reached = [checkpoint for checkpoint in SNAPSHOTS if checkpoint <= result.final_evaluations]
        loss_at_4000 = (
            float(_lookup_history(result, 4000)["best_prefix_loss"])
            if result.final_evaluations >= 4000
            else math.nan
        )
        if result.success:
            final_diagnosis = "succeeded_with_extended_budget"
        elif result.final_evaluations >= 5000:
            relative = (loss_at_4000 - result.final_loss) / max(
                abs(loss_at_4000), 1.0e-300
            )
            final_diagnosis = (
                "insufficient_budget_still_decreasing"
                if relative >= 0.01
                else "genuine_search_stagnation"
            )
        else:
            final_diagnosis = "stopped_without_gate_or_cap"

        previous_loss: float | None = None
        for checkpoint in SNAPSHOTS:
            row = _lookup_history(result, checkpoint)
            if row is None:
                output.append(
                    {
                        "trajectory_id": result.trajectory_id,
                        "snapshot_evaluations": checkpoint,
                        "snapshot_reached": 0,
                        "best_prefix_loss": "",
                        "best_a": "",
                        "best_b": "",
                        "posthoc_parameter_error": "",
                        "absolute_improvement_since_previous_snapshot": "",
                        "relative_improvement_since_previous_snapshot": "",
                        "trend_since_previous_snapshot": "not_reached_after_success",
                        "gate_reached": 1,
                        "first_gate_evaluation": result.first_gate_evaluation,
                        "final_rescue_evaluations": result.final_evaluations,
                        "final_loss": result.final_loss,
                        "final_diagnosis": final_diagnosis,
                        "reconstructed_1500_exactly": int(
                            result.reconstructed_1500_exactly
                        ),
                        "trend_rule": TREND_RULE,
                        "continuation_rule": CONTINUATION_RULE,
                    }
                )
                continue
            loss = float(row["best_prefix_loss"])
            best = np.asarray((float(row["best_a"]), float(row["best_b"])))
            if previous_loss is None:
                absolute_improvement: float | str = ""
                relative_improvement: float | str = ""
                trend = "reconstructed_baseline"
            else:
                absolute_improvement = previous_loss - loss
                relative_improvement = absolute_improvement / max(
                    abs(previous_loss), 1.0e-300
                )
                trend = (
                    "decreasing"
                    if relative_improvement > TREND_TOLERANCE
                    else "plateaued"
                )
            output.append(
                {
                    "trajectory_id": result.trajectory_id,
                    "snapshot_evaluations": checkpoint,
                    "snapshot_reached": 1,
                    "best_prefix_loss": loss,
                    "best_a": best[0],
                    "best_b": best[1],
                    "posthoc_parameter_error": math.hypot(
                        best[0] - reference[0], best[1] - reference[1]
                    ),
                    "absolute_improvement_since_previous_snapshot": absolute_improvement,
                    "relative_improvement_since_previous_snapshot": relative_improvement,
                    "trend_since_previous_snapshot": trend,
                    "gate_reached": int(result.success),
                    "first_gate_evaluation": (
                        result.first_gate_evaluation
                        if result.first_gate_evaluation is not None
                        else ""
                    ),
                    "final_rescue_evaluations": result.final_evaluations,
                    "final_loss": result.final_loss,
                    "final_diagnosis": final_diagnosis,
                    "reconstructed_1500_exactly": int(
                        result.reconstructed_1500_exactly
                    ),
                    "trend_rule": TREND_RULE,
                    "continuation_rule": CONTINUATION_RULE,
                }
            )
            previous_loss = loss
    return output


def _history_rows(
    results: list[ContinuationResult], references: np.ndarray
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for result in results:
        reference = references[result.trajectory_id]
        for row in result.history:
            copy = dict(row)
            copy["posthoc_parameter_error"] = math.hypot(
                float(row["best_a"]) - reference[0],
                float(row["best_b"]) - reference[1],
            )
            copy["is_snapshot"] = int(
                int(row["rescue_evaluations"]) in SNAPSHOTS
            )
            copy["gate_reached"] = int(
                float(row["best_prefix_loss"]) <= HANDOFF_GATE
            )
            output.append(copy)
    return output


def _plot(path: Path, results: list[ContinuationResult]) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.7))
    colors = {36: "tab:blue", 230: "tab:orange"}
    for result in results:
        x = [int(row["rescue_evaluations"]) for row in result.history]
        y = [float(row["best_prefix_loss"]) for row in result.history]
        axis.semilogy(
            x,
            y,
            linewidth=2.0,
            color=colors[result.trajectory_id],
            label=f"trajectory {result.trajectory_id}",
        )
        axis.scatter(
            [result.final_evaluations],
            [result.final_loss],
            s=34,
            color=colors[result.trajectory_id],
            zorder=4,
        )
    axis.axhline(HANDOFF_GATE, color="black", linestyle="--", linewidth=1.2, label="gate 1e-4")
    for checkpoint in SNAPSHOTS:
        axis.axvline(checkpoint, color="0.82", linestyle=":" if checkpoint != 1500 else "--", linewidth=0.9)
    axis.set_xlim(0, TOTAL_RESCUE_BUDGET)
    axis.set_xlabel("rescue objective evaluations")
    axis.set_ylabel("best prefix loss")
    axis.grid(alpha=0.24)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1))
    args = parser.parse_args()
    if not SOURCE_CASES.exists():
        raise FileNotFoundError(SOURCE_CASES)

    saved = _load_cases()
    positions, _, hidden_c = _make_exact_labeled_split(256, FRESH_TEST_SEED)
    positions = positions.to(dtype=DTYPE).cpu().numpy()
    references = _posthoc_reference(hidden_c).cpu().numpy()
    tasks = [(index, positions[index], saved[index]) for index in TRAJECTORY_IDS]

    results: list[ContinuationResult] = []
    if args.workers <= 1:
        for task in tasks:
            result = _continue_case(task)
            results.append(result)
            print(
                f"trajectory {result.trajectory_id}: success={result.success} "
                f"evals={result.final_evaluations} loss={result.final_loss:.6e}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [executor.submit(_continue_case, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"trajectory {result.trajectory_id}: success={result.success} "
                    f"evals={result.final_evaluations} loss={result.final_loss:.6e}",
                    flush=True,
                )
    results.sort(key=lambda result: result.trajectory_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        OUTPUT_DIR / "rescue_continuation_summary.csv",
        _summary_rows(results, references),
    )
    _write_csv(
        OUTPUT_DIR / "rescue_loss_history.csv",
        _history_rows(results, references),
    )
    _plot(OUTPUT_DIR / "best_loss_vs_rescue_evaluations.png", results)
    print(f"wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
