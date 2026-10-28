#!/usr/bin/env python3
"""Adaptive nearest-boundary rescue for the successful original eDIRECT run.

The persisted original eDIRECT result contains final metrics but not the full
search state.  This diagnostic therefore replays only its four failed searches
with the exact original implementation and verifies their final points/losses
against the persisted result before doing any rescue work.

For the nearest global face, local normal-direction spacing is defined without
reference parameters: take the 64 Euclidean-nearest evaluated hypotheses to
the current best point, sort their coordinates along the face-normal axis,
and use the median of the eight adjacent positive coordinate gaps whose
midpoints are closest to the best point.  The strip thickness is exactly
``distance_to_face + local_spacing`` (clipped only to the supplied box), so it
contains the face, the current best point, and one measured local spacing
beyond it.  The free coordinate spans the full supplied domain.

All previous samples inside the strip are retained at zero additional cost.
The rescue then evaluates the original deterministic ten-point design in
strip coordinates and continues with the unchanged original local-RBF
eDIRECT cell and greedy proposal rules.  No GP, reference parameter, loss
gradient, or external optimizer is used by either search phase.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.sho.run_stage1b_edirect_vs_direct import (
    EDIRECT_RULE,
    HANDOFF_GATE,
    MAX_EVALUATIONS,
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
    DOMAIN_WIDTH,
    LOWER,
    UPPER,
    _posthoc_reference,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/adaptive_boundary_rescue")
SOURCE = Path("outputs/sho/stage1B/edirect_vs_direct/per_trajectory_comparison.csv")
TRAJECTORY_COUNT = 256
RESCUE_BUDGET = 1500
FAILED_IDS = (36, 68, 116, 230)
NEIGHBORHOOD_SIZE = 64
SPACING_GAP_COUNT = 8

SPACING_RULE = (
    "median of up to 8 closest positive adjacent face-normal coordinate gaps "
    "among the 64 Euclidean-nearest original-eDIRECT samples to theta_hat"
)
RESCUE_RULE = (
    "nearest global face to original theta_hat; delta=d+s; face-normal strip "
    "contains the boundary, theta_hat, and one local spacing beyond theta_hat; "
    "free coordinate spans [-1.99,1.99]; reuse original in-strip samples; "
    "then unchanged original local-RBF eDIRECT with <=1500 new evaluations; "
    "no second-face rescue"
)


@dataclass(slots=True)
class SearchHistory:
    points: np.ndarray
    losses: np.ndarray
    evaluations: int
    best_index: int


@dataclass(slots=True)
class RescueCase:
    trajectory_id: int
    original_points_physical: np.ndarray
    original_losses: np.ndarray
    original_best: np.ndarray
    original_best_loss: float
    selected_face: str
    normal_axis: int
    distance_to_face: float
    local_spacing: float
    delta: float
    strip_bounds: np.ndarray
    reused_sample_count: int
    rescue_evaluations: int
    final_best: np.ndarray
    final_loss: float
    rescued: bool


def _evaluate(
    normalized: np.ndarray,
    position: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    physical = bounds[:, 0] + normalized * (bounds[:, 1] - bounds[:, 0])
    return _prefix_losses_numpy(physical, position)


def _replay_original(
    trajectory_id: int, position: np.ndarray, cap: int = MAX_EVALUATIONS
) -> SearchHistory:
    """Exact original eDIRECT loop, retaining its evaluated hypotheses."""

    points = _initial_design()
    losses = _prefix_losses_numpy(_physical(points), position)
    evaluations = len(points)
    iteration = 0
    while evaluations < cap and float(np.min(losses)) > HANDOFF_GATE:
        proposals = _cell_proposals(points, losses, trajectory_id, iteration)
        for proposal in proposals:
            if evaluations >= cap:
                break
            loss = float(
                _prefix_losses_numpy(_physical(proposal[None, :]), position)[0]
            )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray([loss])))
            evaluations += 1
            if loss <= HANDOFF_GATE:
                break
        if evaluations >= cap or float(np.min(losses)) <= HANDOFF_GATE:
            break
        proposal = _greedy_proposal(points, losses, trajectory_id, iteration)
        if proposal is not None:
            loss = float(
                _prefix_losses_numpy(_physical(proposal[None, :]), position)[0]
            )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray([loss])))
            evaluations += 1
        iteration += 1
    return SearchHistory(points, losses, evaluations, int(np.argmin(losses)))


def _nearest_face(best: np.ndarray) -> tuple[str, int, bool, float]:
    distances = np.asarray(
        (best[0] - LOWER, UPPER - best[0], best[1] - LOWER, UPPER - best[1])
    )
    face_index = int(np.argmin(distances))
    names = ("a_lower", "a_upper", "b_lower", "b_upper")
    axis = 0 if face_index < 2 else 1
    is_lower = face_index in (0, 2)
    return names[face_index], axis, is_lower, float(distances[face_index])


def _local_normal_spacing(
    points_physical: np.ndarray, best: np.ndarray, axis: int
) -> float:
    distances = np.linalg.norm(points_physical - best[None, :], axis=1)
    neighborhood_count = min(NEIGHBORHOOD_SIZE, len(points_physical))
    neighborhood = points_physical[np.argsort(distances)[:neighborhood_count]]
    coordinates = np.unique(neighborhood[:, axis])
    gaps = np.diff(np.sort(coordinates))
    positive = gaps > 1.0e-14
    gaps = gaps[positive]
    if len(gaps):
        sorted_coordinates = np.sort(coordinates)
        midpoints = 0.5 * (sorted_coordinates[:-1] + sorted_coordinates[1:])
        midpoints = midpoints[positive]
        order = np.argsort(np.abs(midpoints - best[axis]))
        chosen = gaps[order[: min(SPACING_GAP_COUNT, len(order))]]
        spacing = float(np.median(chosen))
    else:
        offsets = np.abs(points_physical[:, axis] - best[axis])
        offsets = offsets[offsets > 1.0e-14]
        if not len(offsets):
            raise RuntimeError("Cannot estimate a positive local normal spacing")
        spacing = float(np.min(offsets))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise RuntimeError("Invalid local normal spacing")
    return spacing


def _strip_bounds(
    best: np.ndarray, axis: int, is_lower: bool, spacing: float
) -> tuple[np.ndarray, float]:
    bounds = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)
    if is_lower:
        bounds[axis, 1] = min(UPPER, best[axis] + spacing)
    else:
        bounds[axis, 0] = max(LOWER, best[axis] - spacing)
    delta = float(bounds[axis, 1] - bounds[axis, 0])
    return bounds, delta


def _append_new_point(
    points: np.ndarray,
    losses: np.ndarray,
    proposal: np.ndarray,
    position: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if len(points) and np.min(np.linalg.norm(points - proposal[None, :], axis=1)) <= 1e-12:
        return points, losses, False
    loss = float(_evaluate(proposal[None, :], position, bounds)[0])
    return (
        np.concatenate((points, proposal[None, :]), axis=0),
        np.concatenate((losses, np.asarray([loss]))),
        True,
    )


def _rescue(
    trajectory_id: int, position: np.ndarray, original: SearchHistory
) -> RescueCase:
    physical = _physical(original.points)
    best = physical[original.best_index]
    best_loss = float(original.losses[original.best_index])
    face, axis, is_lower, distance_to_face = _nearest_face(best)
    spacing = _local_normal_spacing(physical, best, axis)
    bounds, delta = _strip_bounds(best, axis, is_lower, spacing)
    if not math.isclose(
        delta, distance_to_face + spacing, rel_tol=5.0e-14, abs_tol=5.0e-14
    ):
        # This can happen only if clipping expands the strip to the entire box.
        if delta < DOMAIN_WIDTH - 1.0e-14:
            raise RuntimeError("Adaptive strip does not satisfy delta=d+s")

    inside = np.all(
        (physical >= bounds[:, 0] - 1.0e-14)
        & (physical <= bounds[:, 1] + 1.0e-14),
        axis=1,
    )
    seed_physical = physical[inside]
    points = (seed_physical - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    losses = original.losses[inside].copy()
    reused_sample_count = len(points)
    rescue_evaluations = 0

    for proposal in _initial_design():
        if rescue_evaluations >= RESCUE_BUDGET:
            break
        points, losses, used = _append_new_point(
            points, losses, proposal, position, bounds
        )
        rescue_evaluations += int(used)
        if float(np.min(losses)) <= HANDOFF_GATE:
            break

    iteration = 0
    while (
        rescue_evaluations < RESCUE_BUDGET
        and float(np.min(losses)) > HANDOFF_GATE
    ):
        proposals = _cell_proposals(points, losses, trajectory_id, iteration)
        for proposal in proposals:
            if rescue_evaluations >= RESCUE_BUDGET:
                break
            points, losses, used = _append_new_point(
                points, losses, proposal, position, bounds
            )
            rescue_evaluations += int(used)
            if float(np.min(losses)) <= HANDOFF_GATE:
                break
        if (
            rescue_evaluations >= RESCUE_BUDGET
            or float(np.min(losses)) <= HANDOFF_GATE
        ):
            break
        proposal = _greedy_proposal(points, losses, trajectory_id, iteration)
        if proposal is not None:
            points, losses, used = _append_new_point(
                points, losses, proposal, position, bounds
            )
            rescue_evaluations += int(used)
        iteration += 1

    best_index = int(np.argmin(losses))
    final_best = bounds[:, 0] + points[best_index] * (bounds[:, 1] - bounds[:, 0])
    final_loss = float(losses[best_index])
    return RescueCase(
        trajectory_id=trajectory_id,
        original_points_physical=physical,
        original_losses=original.losses,
        original_best=best,
        original_best_loss=best_loss,
        selected_face=face,
        normal_axis=axis,
        distance_to_face=distance_to_face,
        local_spacing=spacing,
        delta=delta,
        strip_bounds=bounds,
        reused_sample_count=reused_sample_count,
        rescue_evaluations=rescue_evaluations,
        final_best=final_best,
        final_loss=final_loss,
        rescued=final_loss <= HANDOFF_GATE,
    )


def _worker(
    arguments: tuple[int, np.ndarray, float, float, float]
) -> RescueCase:
    trajectory_id, position, expected_loss, expected_a, expected_b = arguments
    original = _replay_original(trajectory_id, position)
    best_physical = _physical(original.points[original.best_index][None, :])[0]
    best_loss = float(original.losses[original.best_index])
    expected = np.asarray((expected_a, expected_b))
    if (
        best_loss != expected_loss
        or not np.array_equal(best_physical, expected)
        or original.evaluations != MAX_EVALUATIONS
    ):
        raise RuntimeError(
            f"trajectory {trajectory_id} replay mismatch: "
            f"loss={best_loss} point={best_physical}"
        )
    return _rescue(trajectory_id, position, original)


def _load_source() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = [row for row in _read_csv(SOURCE) if row["method"] == "edirect"]
    rows.sort(key=lambda row: int(row["trajectory_id"]))
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError("Saved eDIRECT cohort is incomplete")
    failures = [row for row in rows if row["success_loss_le_1e-4"] == "0"]
    if [int(row["trajectory_id"]) for row in failures] != list(FAILED_IDS):
        raise RuntimeError("Saved eDIRECT failures are not the expected four cases")
    return rows, failures


def _case_rows(
    cases: list[RescueCase], references: np.ndarray
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for case in cases:
        reference = references[case.trajectory_id]
        true_inside = bool(
            np.all(reference >= case.strip_bounds[:, 0] - 1e-14)
            and np.all(reference <= case.strip_bounds[:, 1] + 1e-14)
        )
        true_face, _, _, _ = _nearest_face(reference)
        if case.rescued:
            diagnosis = "rescued"
        elif not true_inside:
            diagnosis = "adaptive_strip_excluded_true_parameter_posthoc"
        elif true_face != case.selected_face:
            diagnosis = "nearest_face_mismatch_posthoc"
        else:
            diagnosis = "strip_search_insufficient_true_parameter_inside"
        output.append(
            {
                "trajectory_id": case.trajectory_id,
                "original_best_a": case.original_best[0],
                "original_best_b": case.original_best[1],
                "selected_nearest_face": case.selected_face,
                "distance_d": case.distance_to_face,
                "local_spacing_s": case.local_spacing,
                "delta_d_plus_s": case.delta,
                "strip_a_lower": case.strip_bounds[0, 0],
                "strip_a_upper": case.strip_bounds[0, 1],
                "strip_b_lower": case.strip_bounds[1, 0],
                "strip_b_upper": case.strip_bounds[1, 1],
                "reused_original_samples_in_strip": case.reused_sample_count,
                "loss_before_rescue": case.original_best_loss,
                "final_loss": case.final_loss,
                "rescue_evaluations": case.rescue_evaluations,
                "rescued": int(case.rescued),
                "final_best_a": case.final_best[0],
                "final_best_b": case.final_best[1],
                "posthoc_parameter_error": math.hypot(
                    case.final_best[0] - reference[0],
                    case.final_best[1] - reference[1],
                ),
                "reference_a_posthoc": reference[0],
                "reference_b_posthoc": reference[1],
                "reference_inside_adaptive_strip_posthoc": int(true_inside),
                "selected_face_matches_reference_nearest_face_posthoc": int(
                    case.selected_face == true_face
                ),
                "failure_diagnosis_posthoc": diagnosis,
                "normal_replay_exactly_verified": 1,
                "spacing_rule": SPACING_RULE,
                "rescue_rule": RESCUE_RULE,
            }
        )
    return output


def _summary_rows(
    source: list[dict[str, str]], cases: list[RescueCase]
) -> list[dict[str, object]]:
    original_successes = [row for row in source if row["success_loss_le_1e-4"] == "1"]
    rescued = [case for case in cases if case.rescued]
    rescue_evaluations = [case.rescue_evaluations for case in cases]
    rescued_evaluations = [case.rescue_evaluations for case in rescued]
    common = {
        "trajectory_count": TRAJECTORY_COUNT,
        "gate": HANDOFF_GATE,
        "normal_budget": MAX_EVALUATIONS,
        "rescue_budget": RESCUE_BUDGET,
        "original_edirect_rule": EDIRECT_RULE,
        "spacing_rule": SPACING_RULE,
        "rescue_rule": RESCUE_RULE,
    }
    return [
        {
            "stage": "original_edirect",
            **common,
            "success_count": len(original_successes),
            "success_fraction": len(original_successes) / TRAJECTORY_COUNT,
            "rescued_count": 0,
            "remaining_failures": TRAJECTORY_COUNT - len(original_successes),
            "rescue_invocations": 0,
            "total_rescue_evaluations": 0,
            "median_rescue_evaluations_all_invoked": "",
            "mean_rescue_evaluations_all_invoked": "",
            "worst_rescue_evaluations_all_invoked": "",
            "median_rescue_evaluations_among_rescued": "",
            "mean_rescue_evaluations_among_rescued": "",
            "worst_rescue_evaluations_among_rescued": "",
        },
        {
            "stage": "original_edirect_plus_adaptive_strip_rescue",
            **common,
            "success_count": len(original_successes) + len(rescued),
            "success_fraction": (len(original_successes) + len(rescued))
            / TRAJECTORY_COUNT,
            "rescued_count": len(rescued),
            "remaining_failures": len(cases) - len(rescued),
            "rescue_invocations": len(cases),
            "total_rescue_evaluations": int(np.sum(rescue_evaluations)),
            "median_rescue_evaluations_all_invoked": float(
                np.median(rescue_evaluations)
            ),
            "mean_rescue_evaluations_all_invoked": float(
                np.mean(rescue_evaluations)
            ),
            "worst_rescue_evaluations_all_invoked": int(
                np.max(rescue_evaluations)
            ),
            "median_rescue_evaluations_among_rescued": float(
                np.median(rescued_evaluations)
            )
            if rescued_evaluations
            else "",
            "mean_rescue_evaluations_among_rescued": float(
                np.mean(rescued_evaluations)
            )
            if rescued_evaluations
            else "",
            "worst_rescue_evaluations_among_rescued": int(
                np.max(rescued_evaluations)
            )
            if rescued_evaluations
            else "",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    source, failures = _load_source()
    positions, _, hidden_c = _make_exact_labeled_split(
        TRAJECTORY_COUNT, FRESH_TEST_SEED
    )
    positions = positions.to(dtype=DTYPE).cpu().numpy()
    references = _posthoc_reference(hidden_c).cpu().numpy()
    saved_references = np.asarray(
        [
            [float(row["reference_a_posthoc"]), float(row["reference_b_posthoc"])]
            for row in source
        ]
    )
    if not np.array_equal(saved_references, references):
        raise RuntimeError("Saved result does not use the expected test split")

    tasks = [
        (
            int(row["trajectory_id"]),
            positions[int(row["trajectory_id"])],
            float(row["best_prefix_loss"]),
            float(row["best_sample_a"]),
            float(row["best_sample_b"]),
        )
        for row in failures
    ]
    cases: list[RescueCase] = []
    if args.workers <= 1:
        for task in tasks:
            case = _worker(task)
            cases.append(case)
            print(
                f"trajectory {case.trajectory_id}: rescued={case.rescued} "
                f"loss={case.final_loss:.6e} evals={case.rescue_evaluations}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                case = future.result()
                cases.append(case)
                print(
                    f"trajectory {case.trajectory_id}: rescued={case.rescued} "
                    f"loss={case.final_loss:.6e} evals={case.rescue_evaluations}",
                    flush=True,
                )
    cases.sort(key=lambda case: case.trajectory_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(
        OUTPUT_DIR / "adaptive_boundary_rescue_summary.csv",
        _summary_rows(source, cases),
    )
    _write_csv(
        OUTPUT_DIR / "adaptive_boundary_rescue_cases.csv",
        _case_rows(cases, references),
    )
    print(
        f"final success={252 + sum(case.rescued for case in cases)}/256; "
        f"wrote {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
