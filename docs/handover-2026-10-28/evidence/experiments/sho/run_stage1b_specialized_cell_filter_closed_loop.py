#!/usr/bin/env python3
"""Inference-only closed-loop test of the frozen specialized Cell filter."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import numpy as np
import torch

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as base
import experiments.sho.run_stage1b_learned_global_search_imitation as imitation
import experiments.sho.run_stage1b_specialized_cell_filter as specialized
from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/specialized_cell_filter_closed_loop")
SUMMARY_PATH = OUTPUT_DIR / "cell_filter_closed_loop_summary.csv"
PER_TRAJECTORY_PATH = OUTPUT_DIR / "cell_filter_closed_loop_per_trajectory.csv"
DISAGREEMENT_TRACE_PATH = OUTPUT_DIR / "learned_search_traces_disagreements.pt"

FILTER_CHECKPOINT_PATH = specialized.FILTER_CHECKPOINT_PATH
FROZEN_SELECTOR_PATH = specialized.FROZEN_SELECTOR_PATH
SOURCE_SELECTOR_PATH = specialized.SOURCE_SELECTOR
TEACHER_SEARCH_PATH = specialized.SOURCE_TEACHER_SEARCH
TRAIN_TEACHER_PATH = imitation.TEACHER_TRACE_PATH

# This is the 2%-capped operating point that produced the user-authorized
# 99.8721% selected-point retention result.  The failed strict-gate run stored
# None in its checkpoint deliberately, so this inference-only experiment sets
# the operating point without modifying that checkpoint.
SHORTLIST_MULTIPLIER = 64.0
FILTER_CAP_FRACTION = specialized.MAX_SHORTLIST_FRACTION
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = specialized.FULL_BOUNDS
IMITATION_MISS_IDS = (0, 30, 100, 205, 255)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _same_point(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.linalg.norm(left - right) <= 1.0e-12)


def _teacher_cell_order(
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
    raw: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Recover canonical eDIRECT Cell proposal order without a second surrogate."""

    data = edirect._cell_data(points, losses, generator_id, iteration)
    if data.points.shape != raw.shape or not np.array_equal(data.points, raw):
        raise RuntimeError("Cell pool reconstruction changed between filter and teacher")
    ordered: list[np.ndarray] = []
    for owner in data.selected:
        candidates = np.flatnonzero(labels & (data.owners == owner))
        if len(candidates):
            ordered.append(raw[int(candidates[0])].copy())
    if not ordered:
        # This numerical-duplication fallback mirrors eDIRECT exactly and is
        # only expected in a degenerate Cell state.
        return edirect._cell_proposals(points, losses, generator_id, iteration)
    return np.asarray(ordered, dtype=np.float64)


def _run_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    filter_model: specialized.SpecializedCellFilter,
    selector_model: imitation.CandidateRanker,
) -> tuple[imitation.ModeRun, dict[str, Any]]:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    diagnostic_events: list[dict[str, Any]] = []
    evaluations = 0
    iteration = 0
    history_drifted = False
    initial_selector_differences = 0
    cell_states = 0
    teacher_selected_misses = 0
    selector_choice_differences = 0
    extra_candidate_confusions = 0
    drifted_cell_states = 0
    first_drift_evaluation: int | None = None

    pending = edirect._initial_design().copy()
    while len(pending) and evaluations < NORMAL_CAP and (
        not len(losses) or float(np.min(losses)) > GATE
    ):
        if len(points):
            global_features = imitation._global_features(
                points,
                losses,
                len(pending),
                evaluations,
                NORMAL_CAP,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model,
                points,
                losses,
                pending,
                "initial_design",
                global_features,
            )
        else:
            choice = 0
        if int(choice) != 0:
            initial_selector_differences += 1
            if first_drift_evaluation is None:
                first_drift_evaluation = evaluations + 1
            history_drifted = True
        proposal = pending[int(choice)]
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1

    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        raw, raw_features, teacher_mask = base._raw_cell_state(
            points, losses, generator_id, iteration
        )
        teacher_order = _teacher_cell_order(
            points,
            losses,
            generator_id,
            iteration,
            raw,
            teacher_mask,
        )
        teacher_selected = teacher_order[0]
        was_drifted = history_drifted
        if was_drifted:
            drifted_cell_states += 1
        pending, filter_info = specialized._cell_shortlist(
            filter_model,
            points,
            losses,
            raw,
            raw_features,
            iteration,
            SHORTLIST_MULTIPLIER,
        )
        cell_states += 1
        teacher_retained = any(_same_point(point, teacher_selected) for point in pending)
        if not teacher_retained:
            teacher_selected_misses += 1

        first_choice: np.ndarray | None = None
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points,
                losses,
                len(pending),
                evaluations,
                NORMAL_CAP,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, pending, "cell", global_features
            )
            proposal = pending[int(choice)]
            if first_choice is None:
                first_choice = proposal.copy()
                choice_matches_teacher = _same_point(proposal, teacher_selected)
                choice_is_teacher_curated = any(
                    _same_point(proposal, candidate) for candidate in teacher_order
                )
                if not choice_matches_teacher:
                    selector_choice_differences += 1
                    if teacher_retained and not choice_is_teacher_curated:
                        extra_candidate_confusions += 1
                    if first_drift_evaluation is None:
                        first_drift_evaluation = evaluations + 1
                    history_drifted = True
                diagnostic_events.append(
                    {
                        "iteration": iteration,
                        "evaluation_before_selection": evaluations,
                        "history_already_drifted": was_drifted,
                        "raw_candidate_count": int(filter_info["raw_candidate_count"]),
                        "shortlist_count": int(filter_info["shortlist_count"]),
                        "teacher_curated_count": len(teacher_order),
                        "teacher_selected_retained": teacher_retained,
                        "selector_matches_teacher": choice_matches_teacher,
                        "selector_choice_is_teacher_curated": choice_is_teacher_curated,
                        "teacher_selected_candidate": teacher_selected.copy(),
                        "learned_selected_candidate": proposal.copy(),
                    }
                )
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break

        # This is the unchanged deterministic greedy-surrogate proposal.  The
        # frozen selector sees a singleton, exactly as in the prior pipeline.
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            singleton = greedy[None, :]
            global_features = imitation._global_features(
                points,
                losses,
                1,
                evaluations,
                NORMAL_CAP,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, singleton, "greedy", global_features
            )
            proposal = singleton[int(choice)]
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1

    normal = imitation.ModeRun(
        mode="normal_specialized_cell_filter_closed_loop",
        bounds=FULL_BOUNDS.copy(),
        points=points,
        losses=losses,
        new_evaluations=evaluations,
        success=float(np.min(losses)) <= GATE,
        decisions=diagnostic_events,
    )
    diagnostics = {
        "cell_filter_states": cell_states,
        "teacher_selected_point_miss_count": teacher_selected_misses,
        "teacher_selected_point_retention_closed_loop": (
            1.0 - teacher_selected_misses / cell_states if cell_states else 1.0
        ),
        "selector_choice_difference_count": selector_choice_differences,
        "extra_candidate_confusion_count": extra_candidate_confusions,
        "initial_selector_difference_count": initial_selector_differences,
        "drifted_cell_state_count": drifted_cell_states,
        "first_drift_evaluation": first_drift_evaluation,
    }
    return normal, diagnostics


_WORKER_FILTER: specialized.SpecializedCellFilter | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None


def _load_models() -> tuple[
    specialized.SpecializedCellFilter, imitation.CandidateRanker
]:
    global _WORKER_FILTER, _WORKER_SELECTOR
    if _WORKER_FILTER is None:
        checkpoint = torch.load(
            FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        _WORKER_FILTER = specialized.SpecializedCellFilter().to(dtype=torch.float32)
        _WORKER_FILTER.load_state_dict(checkpoint["state_dict"])
        _WORKER_FILTER.eval()
    if _WORKER_SELECTOR is None:
        checkpoint = torch.load(
            FROZEN_SELECTOR_PATH, map_location="cpu", weights_only=False
        )
        _WORKER_SELECTOR = imitation.CandidateRanker().to(dtype=torch.float32)
        _WORKER_SELECTOR.load_state_dict(checkpoint["state_dict"])
        _WORKER_SELECTOR.eval()
    return _WORKER_FILTER, _WORKER_SELECTOR


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    filter_model, selector_model = _load_models()
    normal, diagnostics = _run_normal(
        split, trajectory_id, position, filter_model, selector_model
    )
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    boundary_geometry = None
    if not normal.success:
        boundary_run, boundary_geometry = imitation._run_boundary_rescue(
            split=split,
            trajectory_id=trajectory_id,
            generator_id=generator_id,
            position=position,
            normal=normal,
            collect_trace=False,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    if boundary_run is not None and not boundary_run.success:
        nested_run = imitation._run_nested_rescue(
            split=split,
            trajectory_id=trajectory_id,
            position=position,
            boundary_run=boundary_run,
            collect_trace=False,
        )
        modes.append(nested_run)
        total += nested_run.new_evaluations
    final = next((mode for mode in modes[::-1] if mode.success), modes[-1])
    return {
        "split": split,
        "trajectory_id": trajectory_id,
        "generator_id": generator_id,
        "success": any(mode.success for mode in modes),
        "normal_success": normal.success,
        "normal_evaluations": normal.new_evaluations,
        "normal_best_loss": normal.best_loss,
        "normal_best_theta": normal.best_physical,
        "total_evaluations": total,
        "final_loss": final.best_loss,
        "final_theta": final.best_physical,
        "boundary_invoked": boundary_run is not None,
        "boundary_success": bool(boundary_run and boundary_run.success),
        "nested_invoked": nested_run is not None,
        "nested_success": bool(nested_run and nested_run.success),
        "boundary_geometry": boundary_geometry,
        **diagnostics,
        "modes": [
            {
                "mode": mode.mode,
                "bounds": mode.bounds,
                "evaluated_hypotheses_normalized": mode.points,
                "evaluated_prefix_losses": mode.losses,
                "new_evaluations": mode.new_evaluations,
                "success": mode.success,
                "diagnostic_events": mode.decisions,
            }
            for mode in modes
        ],
    }


def _run_tasks(
    label: str, tasks: list[tuple[str, int, np.ndarray]], workers: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_search_worker(task))
            print(f"{label} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(f"{label} {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda item: int(item["trajectory_id"]))
    return results


def _status(run: dict[str, Any], teacher: bool) -> dict[str, Any]:
    if teacher:
        normal = next(mode for mode in run["modes"] if mode["mode"] == "normal_edirect")
        boundary = next(
            (mode for mode in run["modes"] if mode["mode"] == "boundary_rescue"), None
        )
        nested = next(
            (mode for mode in run["modes"] if mode["mode"] == "nested_1d"), None
        )
        normal_success = bool(normal["success"])
        boundary_invoked = boundary is not None
        boundary_success = bool(boundary and boundary["success"])
        nested_invoked = nested is not None
        nested_success = bool(nested and nested["success"])
    else:
        normal_success = bool(run["normal_success"])
        boundary_invoked = bool(run["boundary_invoked"])
        boundary_success = bool(run["boundary_success"])
        nested_invoked = bool(run["nested_invoked"])
        nested_success = bool(run["nested_success"])
    if normal_success:
        stage = "normal"
    elif boundary_success:
        stage = "boundary_rescue"
    elif nested_success:
        stage = "nested_1d_rescue"
    else:
        stage = "failure"
    return {
        "normal_success": normal_success,
        "boundary_invoked": boundary_invoked,
        "boundary_success": boundary_success,
        "nested_invoked": nested_invoked,
        "nested_success": nested_success,
        "final_success": bool(run["success"]),
        "stage": stage,
    }


def _first_point_drift(learned: dict[str, Any], teacher: dict[str, Any]) -> int | None:
    learned_points = np.asarray(
        learned["modes"][0]["evaluated_hypotheses_normalized"], dtype=np.float64
    )
    teacher_normal = next(
        mode for mode in teacher["modes"] if mode["mode"] == "normal_edirect"
    )
    teacher_points = np.asarray(
        teacher_normal["evaluated_hypotheses_normalized"], dtype=np.float64
    )
    common = min(len(learned_points), len(teacher_points))
    differing = np.flatnonzero(
        np.linalg.norm(learned_points[:common] - teacher_points[:common], axis=1) > 1.0e-12
    )
    if len(differing):
        return int(differing[0]) + 1
    if len(learned_points) != len(teacher_points):
        return common + 1
    return None


def _cause(
    learned: dict[str, Any],
    learned_status: dict[str, Any],
    teacher_status: dict[str, Any],
) -> str:
    if (
        learned_status["normal_success"] == teacher_status["normal_success"]
        and learned_status["stage"] != teacher_status["stage"]
    ):
        return "rescue_difference"
    if int(learned["teacher_selected_point_miss_count"]) > 0:
        return "missing_teacher_selected_candidate"
    if int(learned["extra_candidate_confusion_count"]) > 0:
        return "extra_candidates_confusing_frozen_selector"
    return "accumulated_search_state_drift"


def _trajectory_rows(
    split_label: str,
    learned_runs: list[dict[str, Any]],
    teacher_runs: list[dict[str, Any]],
    selector_baseline_runs: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    teacher_by_id = {int(run["trajectory_id"]): run for run in teacher_runs}
    baseline_by_id = (
        {int(run["trajectory_id"]): run for run in selector_baseline_runs}
        if selector_baseline_runs is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    disagreement_traces: list[dict[str, Any]] = []
    for learned in learned_runs:
        trajectory_id = int(learned["trajectory_id"])
        teacher = teacher_by_id[trajectory_id]
        learned_status = _status(learned, False)
        teacher_status = _status(teacher, True)
        baseline = baseline_by_id.get(trajectory_id)
        baseline_status = _status(baseline, True) if baseline is not None else None
        normal_agreement = (
            learned_status["normal_success"] == teacher_status["normal_success"]
        )
        final_agreement = (
            learned_status["final_success"] == teacher_status["final_success"]
        )
        exact_stage_agreement = learned_status["stage"] == teacher_status["stage"]
        differs = not (normal_agreement and final_agreement and exact_stage_agreement)
        point_drift = _first_point_drift(learned, teacher)
        cause = _cause(learned, learned_status, teacher_status) if differs else ""
        row = {
            "split": split_label,
            "trajectory_id": trajectory_id,
            "specified_imitation_miss_id": trajectory_id in IMITATION_MISS_IDS,
            "learned_normal_success": learned_status["normal_success"],
            "teacher_normal_success": teacher_status["normal_success"],
            "normal_outcome_agreement": normal_agreement,
            "learned_final_success": learned_status["final_success"],
            "teacher_final_success": teacher_status["final_success"],
            "final_success_agreement": final_agreement,
            "learned_final_stage": learned_status["stage"],
            "teacher_final_stage": teacher_status["stage"],
            "exact_stage_outcome_agreement": exact_stage_agreement,
            "learned_total_evaluations": int(learned["total_evaluations"]),
            "teacher_total_evaluations": int(teacher["total_evaluations"]),
            "evaluation_difference": int(learned["total_evaluations"])
            - int(teacher["total_evaluations"]),
            "learned_normal_best_loss": float(learned["normal_best_loss"]),
            "learned_final_loss": float(learned["final_loss"]),
            "teacher_final_loss": float(teacher["final_loss"]),
            "learned_rescue_invoked": learned_status["boundary_invoked"],
            "teacher_rescue_invoked": teacher_status["boundary_invoked"],
            "cell_filter_states": int(learned["cell_filter_states"]),
            "teacher_selected_point_miss_count": int(
                learned["teacher_selected_point_miss_count"]
            ),
            "teacher_selected_point_retention_closed_loop": float(
                learned["teacher_selected_point_retention_closed_loop"]
            ),
            "selector_choice_difference_count": int(
                learned["selector_choice_difference_count"]
            ),
            "extra_candidate_confusion_count": int(
                learned["extra_candidate_confusion_count"]
            ),
            "drifted_cell_state_count": int(learned["drifted_cell_state_count"]),
            "first_drift_evaluation_vs_live_teacher": learned["first_drift_evaluation"],
            "first_point_drift_vs_saved_teacher": point_drift,
            "outcome_disagreement": differs,
            "primary_disagreement_cause": cause,
            "selector_baseline_final_success": (
                baseline_status["final_success"] if baseline_status else ""
            ),
            "selector_baseline_stage": baseline_status["stage"] if baseline_status else "",
        }
        rows.append(row)
        if differs:
            disagreement_traces.append(
                {
                    "split": split_label,
                    "trajectory_id": trajectory_id,
                    "classification": row,
                    "learned": learned,
                    "teacher": teacher,
                    "frozen_selector_baseline": baseline,
                }
            )
    return rows, disagreement_traces


def _summary_row(
    split: str, method: str, statuses: list[dict[str, Any]], evaluations: list[int]
) -> dict[str, object]:
    successes = [index for index, status in enumerate(statuses) if status["final_success"]]
    success_evaluations = [evaluations[index] for index in successes]
    return {
        "row_type": "search_summary",
        "split": split,
        "method": method,
        "trajectories": len(statuses),
        "normal_success_count": sum(status["normal_success"] for status in statuses),
        "normal_success_fraction": float(
            np.mean([status["normal_success"] for status in statuses])
        ),
        "final_success_count": len(successes),
        "final_success_fraction": len(successes) / len(statuses),
        "rescue_invocation_count": sum(
            status["boundary_invoked"] for status in statuses
        ),
        "rescue_invocation_fraction": float(
            np.mean([status["boundary_invoked"] for status in statuses])
        ),
        "median_evaluations_success": float(np.median(success_evaluations))
        if success_evaluations
        else "",
        "mean_evaluations_success": float(np.mean(success_evaluations))
        if success_evaluations
        else "",
        "worst_evaluations_success": max(success_evaluations, default=""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in (
        FILTER_CHECKPOINT_PATH,
        FROZEN_SELECTOR_PATH,
        SOURCE_SELECTOR_PATH,
        TEACHER_SEARCH_PATH,
        TRAIN_TEACHER_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    filter_hash_before = _sha256(FILTER_CHECKPOINT_PATH)
    selector_hash_before = _sha256(FROZEN_SELECTOR_PATH)
    if selector_hash_before != _sha256(SOURCE_SELECTOR_PATH):
        raise RuntimeError("frozen selector is not byte-identical to its source checkpoint")

    checkpoint = torch.load(
        FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    validation_metric = checkpoint["exhaustive_validation_metrics"]
    if not math.isclose(
        float(validation_metric["teacher_selected_point_retention"]),
        0.9987205731832139,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError("specialized filter checkpoint is not the authorized checkpoint")

    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    test_positions, _, _ = _make_exact_labeled_split(
        imitation.TEST_COUNT, imitation.TEST_SEED
    )
    training_positions, _, _ = _make_exact_labeled_split(
        imitation.TRAIN_COUNT, imitation.TRAIN_SEED
    )
    validation_tasks = [
        ("validation", index, validation_positions[index].to(dtype=DTYPE).cpu().numpy())
        for index in range(imitation.VALIDATION_COUNT)
    ]
    test_tasks = [
        ("test", index, test_positions[index].to(dtype=DTYPE).cpu().numpy())
        for index in range(imitation.TEST_COUNT)
    ]
    audit_tasks = [
        ("train", index, training_positions[index].to(dtype=DTYPE).cpu().numpy())
        for index in IMITATION_MISS_IDS
    ]

    validation_runs = _run_tasks("closed-loop validation", validation_tasks, args.workers)
    test_runs = _run_tasks("closed-loop test", test_tasks, args.workers)
    audit_runs = _run_tasks("closed-loop imitation-miss audit", audit_tasks, args.workers)

    teacher_payload = torch.load(
        TEACHER_SEARCH_PATH, map_location="cpu", weights_only=False
    )
    train_teacher_payload = torch.load(
        TRAIN_TEACHER_PATH, map_location="cpu", weights_only=False
    )
    validation_teacher = base._teacher_runs(teacher_payload, "validation")
    test_teacher = base._teacher_runs(teacher_payload, "test")
    validation_baseline = teacher_payload["validation_learned"]["trajectories"]
    test_baseline = teacher_payload["test_learned"]["trajectories"]
    train_teacher_map = {
        int(run["trajectory_id"]): run for run in train_teacher_payload["trajectories"]
    }
    audit_teacher = [train_teacher_map[index] for index in IMITATION_MISS_IDS]

    validation_rows, validation_disagreements = _trajectory_rows(
        "validation", validation_runs, validation_teacher, validation_baseline
    )
    test_rows, test_disagreements = _trajectory_rows(
        "test", test_runs, test_teacher, test_baseline
    )
    audit_rows, audit_disagreements = _trajectory_rows(
        "imitation_miss_audit", audit_runs, audit_teacher, None
    )
    trajectory_rows = validation_rows + test_rows + audit_rows
    _write_csv(PER_TRAJECTORY_PATH, trajectory_rows)

    summary_rows: list[dict[str, object]] = []
    split_data = (
        ("validation", validation_runs, validation_teacher, validation_rows),
        ("test", test_runs, test_teacher, test_rows),
        ("imitation_miss_audit", audit_runs, audit_teacher, audit_rows),
    )
    for split, learned_runs, teacher_runs, rows in split_data:
        summary_rows.append(
            _summary_row(
                split,
                "learned_cell_filter",
                [_status(run, False) for run in learned_runs],
                [int(run["total_evaluations"]) for run in learned_runs],
            )
        )
        summary_rows.append(
            _summary_row(
                split,
                "deterministic_edirect_teacher",
                [_status(run, True) for run in teacher_runs],
                [int(run["total_evaluations"]) for run in teacher_runs],
            )
        )
        summary_rows.append(
            {
                "row_type": "agreement_summary",
                "split": split,
                "method": "learned_vs_teacher",
                "trajectories": len(rows),
                "normal_outcome_agreement_fraction": float(
                    np.mean([bool(row["normal_outcome_agreement"]) for row in rows])
                ),
                "final_success_agreement_fraction": float(
                    np.mean([bool(row["final_success_agreement"]) for row in rows])
                ),
                "exact_stage_outcome_agreement_fraction": float(
                    np.mean(
                        [bool(row["exact_stage_outcome_agreement"]) for row in rows]
                    )
                ),
                "outcome_disagreement_count": sum(
                    bool(row["outcome_disagreement"]) for row in rows
                ),
                "cell_filter_states": sum(
                    int(row["cell_filter_states"]) for row in rows
                ),
                "teacher_selected_point_miss_count": sum(
                    int(row["teacher_selected_point_miss_count"]) for row in rows
                ),
                "teacher_selected_point_retention_closed_loop": 1.0
                - sum(int(row["teacher_selected_point_miss_count"]) for row in rows)
                / max(1, sum(int(row["cell_filter_states"]) for row in rows)),
                "trajectories_with_teacher_selected_point_miss": sum(
                    int(row["teacher_selected_point_miss_count"]) > 0 for row in rows
                ),
                "extra_candidate_confusion_count": sum(
                    int(row["extra_candidate_confusion_count"]) for row in rows
                ),
            }
        )

    for row in trajectory_rows:
        if bool(row["outcome_disagreement"]):
            summary_rows.append(
                {
                    "row_type": "outcome_disagreement",
                    "split": row["split"],
                    "trajectory_id": row["trajectory_id"],
                    "learned_final_stage": row["learned_final_stage"],
                    "teacher_final_stage": row["teacher_final_stage"],
                    "primary_disagreement_cause": row["primary_disagreement_cause"],
                    "teacher_selected_point_miss_count": row[
                        "teacher_selected_point_miss_count"
                    ],
                    "extra_candidate_confusion_count": row[
                        "extra_candidate_confusion_count"
                    ],
                    "first_point_drift_vs_saved_teacher": row[
                        "first_point_drift_vs_saved_teacher"
                    ],
                }
            )
    for row in audit_rows:
        summary_rows.append(
            {
                "row_type": "specified_imitation_miss_audit",
                "split": row["split"],
                "trajectory_id": row["trajectory_id"],
                "learned_normal_success": row["learned_normal_success"],
                "learned_final_success": row["learned_final_success"],
                "teacher_final_success": row["teacher_final_success"],
                "teacher_selected_point_miss_count": row[
                    "teacher_selected_point_miss_count"
                ],
                "outcome_disagreement": row["outcome_disagreement"],
                "primary_disagreement_cause": row["primary_disagreement_cause"],
            }
        )
    summary_rows.append(
        {
            "row_type": "integrity",
            "shortlist_multiplier": SHORTLIST_MULTIPLIER,
            "maximum_shortlist_fraction": FILTER_CAP_FRACTION,
            "authorized_validation_selected_point_retention": float(
                validation_metric["teacher_selected_point_retention"]
            ),
            "authorized_validation_curated_set_macro_recall": float(
                validation_metric["curated_set_macro_recall"]
            ),
            "filter_checkpoint_sha256": filter_hash_before,
            "filter_checkpoint_unchanged": _sha256(FILTER_CHECKPOINT_PATH)
            == filter_hash_before,
            "selector_checkpoint_sha256": selector_hash_before,
            "selector_checkpoint_unchanged": _sha256(FROZEN_SELECTOR_PATH)
            == selector_hash_before,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    _write_csv(SUMMARY_PATH, summary_rows)

    all_disagreements = (
        validation_disagreements + test_disagreements + audit_disagreements
    )
    torch.save(
        {
            "config": {
                "inference_only": True,
                "shortlist_multiplier": SHORTLIST_MULTIPLIER,
                "maximum_shortlist_fraction": FILTER_CAP_FRACTION,
                "gate": GATE,
                "normal_cap": NORMAL_CAP,
                "filter_checkpoint_sha256": filter_hash_before,
                "selector_checkpoint_sha256": selector_hash_before,
            },
            "disagreements": all_disagreements,
        },
        DISAGREEMENT_TRACE_PATH,
    )
    print(
        f"wrote {len(trajectory_rows)} trajectory rows and "
        f"{len(all_disagreements)} disagreement traces",
        flush=True,
    )


if __name__ == "__main__":
    main()
