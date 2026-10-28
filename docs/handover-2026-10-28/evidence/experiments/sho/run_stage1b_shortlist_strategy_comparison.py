#!/usr/bin/env python3
"""Small exact-cardinality training comparison for Cell/Voronoi curation."""

from __future__ import annotations

import argparse
import copy
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
from torch import nn

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
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/shortlist_strategy_comparison")
COMPARISON_PATH = OUTPUT_DIR / "shortlist_strategy_comparison.csv"
CLOSED_LOOP_PATH = OUTPUT_DIR / "small_closed_loop_summary.csv"

RAW_TRACE_PATH = base.RAW_TRACE_PATH
FROZEN_SELECTOR_PATH = specialized.FROZEN_SELECTOR_PATH
SOURCE_SELECTOR_PATH = specialized.SOURCE_SELECTOR
TRAIN_STATE_COUNT = 2_000
VALIDATION_STATE_COUNT = 500
STAGE_GATE_STATE_COUNT = 100
TOTAL_UPDATES = 800
STAGE_UPDATES = 200
CURRICULUM_MULTIPLIERS = (4.0, 2.0, 1.5, 1.0)
LEARNING_RATE = 1.0e-3
MODEL_SEED = SEED + 15_800_000
SAMPLE_SEED = SEED + 15_900_000
RANK_MARGIN = 0.2
RANK_WEIGHT = 2.0
SELECTED_WEIGHT = 0.5
CURRICULUM_RECALL_GATE = 0.98
CURRICULUM_SELECTED_GATE = 0.99
STATIC_OVERLAP_GATE = 0.95
STATIC_SELECTED_GATE = 0.99
STATIC_PERFECT_GATE = 0.75
SCREEN_TRAJECTORIES = 16
SCREEN_BATCH = 8
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = specialized.FULL_BOUNDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _stratified_indices(
    states: list[dict[str, Any]], validation: bool, count: int, seed: int
) -> np.ndarray:
    groups: dict[int, list[int]] = {}
    for index, state in enumerate(states):
        if state["candidate_type"] != "cell":
            continue
        if (int(state["trajectory_id"]) % 5 == 0) != validation:
            continue
        groups.setdefault(len(state["teacher_curated_points"]), []).append(index)
    generator = np.random.default_rng(seed)
    for indices in groups.values():
        generator.shuffle(indices)
    selected: list[int] = []
    offsets = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < count:
        made_progress = False
        for key in keys:
            offset = offsets[key]
            if offset < len(groups[key]):
                selected.append(groups[key][offset])
                offsets[key] += 1
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    if len(selected) != count:
        raise RuntimeError(f"requested {count} states but found {len(selected)}")
    generator.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def _materialize(
    raw_trace: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_index: int,
) -> dict[str, Any]:
    record = raw_trace["states"][int(state_index)]
    trajectory = trajectories[int(record["trajectory_id"])]
    raw, features, labels, selected_index = base._materialize_raw_state(
        record, trajectory
    )
    history_count = int(record["history_count"])
    points = trajectory["evaluated_hypotheses_normalized"][:history_count]
    losses = trajectory["evaluated_prefix_losses"][:history_count]
    history, global_features = base._filter_global_features(
        points, losses, len(raw), int(record["iteration"]), "cell"
    )
    return {
        "state_index": int(state_index),
        "k": int(np.sum(labels)),
        "selected_index": int(selected_index),
        "history": torch.as_tensor(history, dtype=torch.float32),
        "global": torch.as_tensor(global_features, dtype=torch.float32),
        "features": torch.as_tensor(features, dtype=torch.float32),
        "labels": torch.as_tensor(labels, dtype=torch.bool),
    }


def _scores(
    model: specialized.SpecializedCellFilter, example: dict[str, Any]
) -> torch.Tensor:
    context = model.encode_context(
        example["history"][None, :, :],
        torch.ones((1, len(example["history"])), dtype=torch.bool),
        example["global"][None, :],
    )
    return model.score_candidates(context, example["features"][None, :, :])[0]


def _ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    selected_index: int,
    allowed_multiplier: float,
) -> torch.Tensor:
    positive = scores[labels]
    negative = scores[~labels]
    k = len(positive)
    allowed = min(len(scores), max(k, int(math.ceil(allowed_multiplier * k))))
    negative_allowance = allowed - k
    boundary_rank = min(len(negative) - 1, negative_allowance)
    boundary = torch.topk(negative, k=boundary_rank + 1).values[-1]
    positive_bce = nn.functional.softplus(-positive).mean()
    negative_bce = nn.functional.softplus(negative).mean()
    ranking = nn.functional.softplus(boundary - positive + RANK_MARGIN).mean()
    selected_ranking = nn.functional.softplus(
        boundary - scores[selected_index] + RANK_MARGIN
    )
    return (
        0.5 * positive_bce
        + 0.5 * negative_bce
        + RANK_WEIGHT * ranking
        + SELECTED_WEIGHT * selected_ranking
    )


def _state_metrics(scores: np.ndarray, labels: np.ndarray, selected: int, multiplier: float) -> dict[str, Any]:
    k = int(np.sum(labels))
    count = min(len(scores), max(k, int(math.ceil(multiplier * k))))
    order = np.argpartition(scores, -count)[-count:]
    retained = np.zeros(len(scores), dtype=np.bool_)
    retained[order] = True
    overlap = int(np.sum(retained & labels))
    return {
        "k": k,
        "predicted_count": count,
        "overlap": overlap,
        "recall": overlap / k,
        "precision": overlap / count,
        "selected_retained": bool(retained[selected]),
        "perfect": overlap == k and count == k,
    }


def _evaluate(
    model: specialized.SpecializedCellFilter,
    raw_trace: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    indices: np.ndarray,
    multiplier: float,
    label: str,
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    model.eval()
    records: list[dict[str, Any]] = []
    by_k_records: dict[int, list[dict[str, Any]]] = {}
    with torch.no_grad():
        for row, state_index in enumerate(indices):
            example = _materialize(raw_trace, trajectories, int(state_index))
            scores = _scores(model, example).cpu().numpy()
            metrics = _state_metrics(
                scores,
                example["labels"].cpu().numpy(),
                int(example["selected_index"]),
                multiplier,
            )
            records.append(metrics)
            by_k_records.setdefault(int(metrics["k"]), []).append(metrics)
            if (row + 1) % 100 == 0 or row + 1 == len(indices):
                print(f"{label} {row + 1}/{len(indices)}", flush=True)

    def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
        total_overlap = sum(int(item["overlap"]) for item in items)
        total_teacher = sum(int(item["k"]) for item in items)
        total_predicted = sum(int(item["predicted_count"]) for item in items)
        return {
            "states": len(items),
            "macro_overlap": float(np.mean([item["recall"] for item in items])),
            "candidate_precision": total_overlap / total_predicted,
            "candidate_recall": total_overlap / total_teacher,
            "teacher_selected_point_retention": float(
                np.mean([item["selected_retained"] for item in items])
            ),
            "perfect_shortlist_fraction": float(
                np.mean([item["perfect"] for item in items])
            ),
            "perfect_shortlist_count": sum(bool(item["perfect"]) for item in items),
        }

    return aggregate(records), {
        key: aggregate(items) for key, items in sorted(by_k_records.items())
    }


def _train(
    raw_trace: dict[str, Any],
    train_indices: np.ndarray,
    gate_indices: np.ndarray,
) -> tuple[
    specialized.SpecializedCellFilter,
    specialized.SpecializedCellFilter,
    list[dict[str, object]],
]:
    trajectories = base._trajectory_map(raw_trace)
    torch.manual_seed(MODEL_SEED)
    initial = specialized.SpecializedCellFilter().to(dtype=torch.float32)
    initial_state = copy.deepcopy(initial.state_dict())
    direct = specialized.SpecializedCellFilter().to(dtype=torch.float32)
    curriculum = specialized.SpecializedCellFilter().to(dtype=torch.float32)
    direct.load_state_dict(initial_state)
    curriculum.load_state_dict(initial_state)
    direct_optimizer = torch.optim.Adam(direct.parameters(), lr=LEARNING_RATE)
    curriculum_optimizer = torch.optim.Adam(curriculum.parameters(), lr=LEARNING_RATE)
    generator = np.random.default_rng(SAMPLE_SEED + 10)
    schedule = generator.permutation(train_indices)
    update_batches = np.array_split(schedule, TOTAL_UPDATES)
    curriculum_stage = 0
    stage_rows: list[dict[str, object]] = []
    direct_loss_total = curriculum_loss_total = 0.0
    seen_states = 0

    for update, state_batch in enumerate(update_batches, start=1):
        direct.train()
        direct_optimizer.zero_grad(set_to_none=True)
        multiplier = CURRICULUM_MULTIPLIERS[curriculum_stage]
        curriculum.train()
        curriculum_optimizer.zero_grad(set_to_none=True)
        batch_size = len(state_batch)
        for state_index in state_batch:
            example = _materialize(raw_trace, trajectories, int(state_index))
            direct_scores = _scores(direct, example)
            direct_loss = _ranking_loss(
                direct_scores,
                example["labels"],
                int(example["selected_index"]),
                1.0,
            )
            (direct_loss / batch_size).backward()
            direct_loss_total += float(direct_loss.detach())

            curriculum_scores = _scores(curriculum, example)
            curriculum_loss = _ranking_loss(
                curriculum_scores,
                example["labels"],
                int(example["selected_index"]),
                multiplier,
            )
            (curriculum_loss / batch_size).backward()
            curriculum_loss_total += float(curriculum_loss.detach())
        nn.utils.clip_grad_norm_(direct.parameters(), 5.0)
        direct_optimizer.step()
        nn.utils.clip_grad_norm_(curriculum.parameters(), 5.0)
        curriculum_optimizer.step()
        seen_states += batch_size

        if update % STAGE_UPDATES == 0:
            gate_metrics, _ = _evaluate(
                curriculum,
                raw_trace,
                trajectories,
                gate_indices,
                multiplier,
                f"curriculum gate update={update} multiplier={multiplier:g}",
            )
            passed = (
                gate_metrics["macro_overlap"] >= CURRICULUM_RECALL_GATE
                and gate_metrics["teacher_selected_point_retention"]
                >= CURRICULUM_SELECTED_GATE
            )
            stage_rows.append(
                {
                    "row_type": "curriculum_stage",
                    "branch": "trimming_curriculum",
                    "update": update,
                    "allowed_multiplier": multiplier,
                    "stage_macro_retention": gate_metrics["macro_overlap"],
                    "stage_selected_point_retention": gate_metrics[
                        "teacher_selected_point_retention"
                    ],
                    "stage_gate_passed": passed,
                    "advanced_to_next_multiplier": passed
                    and curriculum_stage < len(CURRICULUM_MULTIPLIERS) - 1,
                }
            )
            if passed and curriculum_stage < len(CURRICULUM_MULTIPLIERS) - 1:
                curriculum_stage += 1
            print(
                f"updates={update}/{TOTAL_UPDATES} states={seen_states}/{len(schedule)} "
                f"direct_loss={direct_loss_total/seen_states:.4f} "
                f"curriculum_loss={curriculum_loss_total/seen_states:.4f} "
                f"next_multiplier={CURRICULUM_MULTIPLIERS[curriculum_stage]:g}",
                flush=True,
            )
    stage_rows.append(
        {
            "row_type": "training_budget",
            "branch": "direct_top_k",
            "total_updates": TOTAL_UPDATES,
            "training_states_seen": seen_states,
            "mean_training_loss": direct_loss_total / seen_states,
            "final_training_multiplier": 1.0,
        }
    )
    stage_rows.append(
        {
            "row_type": "training_budget",
            "branch": "trimming_curriculum",
            "total_updates": TOTAL_UPDATES,
            "training_states_seen": seen_states,
            "mean_training_loss": curriculum_loss_total / seen_states,
            "final_training_multiplier": CURRICULUM_MULTIPLIERS[curriculum_stage],
        }
    )
    return direct, curriculum, stage_rows


def _exact_shortlist(
    model: specialized.SpecializedCellFilter,
    points: np.ndarray,
    losses: np.ndarray,
    raw: np.ndarray,
    raw_features: np.ndarray,
    labels: np.ndarray,
    iteration: int,
) -> np.ndarray:
    scores, _ = base._score_full_raw_pool(
        model, points, losses, raw_features, iteration, "cell"
    )
    k = int(np.sum(labels))
    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return raw[indices]


def _run_screen_normal(
    trajectory_id: int,
    position: np.ndarray,
    model: specialized.SpecializedCellFilter,
    selector: imitation.CandidateRanker,
) -> imitation.ModeRun:
    split = "validation"
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    evaluations = 0
    iteration = 0
    pending = edirect._initial_design().copy()
    while len(pending) and evaluations < NORMAL_CAP and (
        not len(losses) or float(np.min(losses)) > GATE
    ):
        if len(points):
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect"
            )
            choice, _ = imitation._model_choice(
                selector, points, losses, pending, "initial_design", global_features
            )
        else:
            choice = 0
        proposal = pending[int(choice)]
        loss = float(edirect._prefix_losses_numpy(
            (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
        )[0])
        points = np.concatenate((points, proposal[None, :]))
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1

    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        raw, raw_features, labels = base._raw_cell_state(
            points, losses, generator_id, iteration
        )
        pending = _exact_shortlist(
            model, points, losses, raw, raw_features, labels, iteration
        )
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect"
            )
            choice, _ = imitation._model_choice(
                selector, points, losses, pending, "cell", global_features
            )
            proposal = pending[int(choice)]
            loss = float(edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0])
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            proposal = greedy
            loss = float(edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0])
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1
    return imitation.ModeRun(
        mode="small_exact_k_screen",
        bounds=FULL_BOUNDS.copy(),
        points=points,
        losses=losses,
        new_evaluations=evaluations,
        success=float(np.min(losses)) <= GATE,
    )


def _screen_worker(
    arguments: tuple[str, dict[str, torch.Tensor], int, np.ndarray]
) -> dict[str, Any]:
    branch, state_dict, trajectory_id, position = arguments
    model = specialized.SpecializedCellFilter().to(dtype=torch.float32)
    model.load_state_dict(state_dict)
    model.eval()
    selector_checkpoint = torch.load(
        FROZEN_SELECTOR_PATH, map_location="cpu", weights_only=False
    )
    selector = imitation.CandidateRanker().to(dtype=torch.float32)
    selector.load_state_dict(selector_checkpoint["state_dict"])
    selector.eval()
    normal = _run_screen_normal(trajectory_id, position, model, selector)
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    if not normal.success:
        generator_id = imitation.SPLIT_ID_OFFSET["validation"] + trajectory_id
        boundary_run, _ = imitation._run_boundary_rescue(
            split="validation", trajectory_id=trajectory_id,
            generator_id=generator_id, position=position, normal=normal,
            collect_trace=False,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    if boundary_run is not None and not boundary_run.success:
        nested_run = imitation._run_nested_rescue(
            split="validation", trajectory_id=trajectory_id, position=position,
            boundary_run=boundary_run, collect_trace=False,
        )
        modes.append(nested_run)
        total += nested_run.new_evaluations
    return {
        "branch": branch,
        "trajectory_id": trajectory_id,
        "normal_success": normal.success,
        "final_success": any(mode.success for mode in modes),
        "rescue_invoked": boundary_run is not None,
        "total_evaluations": total,
    }


def _run_screen_batch(
    branch: str,
    model: specialized.SpecializedCellFilter,
    positions: np.ndarray,
    ids: range,
    workers: int,
) -> list[dict[str, Any]]:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    tasks = [
        (branch, state_dict, index, positions[index]) for index in ids
    ]
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_screen_worker, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["trajectory_id"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (RAW_TRACE_PATH, FROZEN_SELECTOR_PATH, SOURCE_SELECTOR_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    selector_hash = _sha256(FROZEN_SELECTOR_PATH)
    if selector_hash != _sha256(SOURCE_SELECTOR_PATH):
        raise RuntimeError("frozen selector differs from its source checkpoint")

    raw_trace = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = _stratified_indices(
        raw_trace["states"], False, TRAIN_STATE_COUNT, SAMPLE_SEED
    )
    validation_indices = _stratified_indices(
        raw_trace["states"], True, VALIDATION_STATE_COUNT, SAMPLE_SEED + 1
    )
    gate_indices = validation_indices[:STAGE_GATE_STATE_COUNT]
    train_k = [
        len(raw_trace["states"][int(index)]["teacher_curated_points"])
        for index in train_indices
    ]
    validation_k = [
        len(raw_trace["states"][int(index)]["teacher_curated_points"])
        for index in validation_indices
    ]
    print(
        f"stratified states train={len(train_indices)} k={min(train_k)}..{max(train_k)} "
        f"validation={len(validation_indices)} k={min(validation_k)}..{max(validation_k)}",
        flush=True,
    )

    direct, curriculum, comparison_rows = _train(
        raw_trace, train_indices, gate_indices
    )
    trajectories = base._trajectory_map(raw_trace)
    final_metrics: dict[str, dict[str, float]] = {}
    for branch, model in (("direct_top_k", direct), ("trimming_curriculum", curriculum)):
        metrics, by_k = _evaluate(
            model, raw_trace, trajectories, validation_indices, 1.0,
            f"final exact-k {branch}",
        )
        final_metrics[branch] = metrics
        comparison_rows.append(
            {
                "row_type": "final_exact_k",
                "branch": branch,
                "evaluation_multiplier": 1.0,
                **metrics,
                "static_screen_authorized": (
                    metrics["macro_overlap"] >= STATIC_OVERLAP_GATE
                    and metrics["teacher_selected_point_retention"]
                    >= STATIC_SELECTED_GATE
                    and metrics["perfect_shortlist_fraction"] >= STATIC_PERFECT_GATE
                ),
            }
        )
        for k, k_metrics in by_k.items():
            comparison_rows.append(
                {"row_type": "by_teacher_k", "branch": branch, "teacher_k": k, **k_metrics}
            )
    comparison_rows.append(
        {
            "row_type": "configuration",
            "train_states": len(train_indices),
            "validation_states": len(validation_indices),
            "train_k_min": min(train_k),
            "train_k_max": max(train_k),
            "validation_k_min": min(validation_k),
            "validation_k_max": max(validation_k),
            "total_updates_per_branch": TOTAL_UPDATES,
            "same_initialization": True,
            "same_architecture": True,
            "same_optimizer": True,
            "full_raw_pool_scoring": True,
            "selector_checkpoint_sha256": selector_hash,
            "selector_checkpoint_unchanged": _sha256(FROZEN_SELECTOR_PATH) == selector_hash,
            "runtime_seconds_before_screen": time.perf_counter() - started,
        }
    )
    _write_csv(COMPARISON_PATH, comparison_rows)

    validation_positions, _, _ = _make_exact_labeled_split(
        SCREEN_TRAJECTORIES, imitation.VALIDATION_SEED
    )
    positions = validation_positions.to(dtype=DTYPE).cpu().numpy()
    closed_loop_rows: list[dict[str, object]] = []
    for branch, model in (("direct_top_k", direct), ("trimming_curriculum", curriculum)):
        metrics = final_metrics[branch]
        authorized = (
            metrics["macro_overlap"] >= STATIC_OVERLAP_GATE
            and metrics["teacher_selected_point_retention"] >= STATIC_SELECTED_GATE
            and metrics["perfect_shortlist_fraction"] >= STATIC_PERFECT_GATE
        )
        if not authorized:
            closed_loop_rows.append(
                {
                    "branch": branch,
                    "screen_status": "not_run_static_gate_failed",
                    "trajectories_run": 0,
                    "static_macro_overlap": metrics["macro_overlap"],
                    "static_selected_point_retention": metrics[
                        "teacher_selected_point_retention"
                    ],
                    "static_perfect_shortlist_fraction": metrics[
                        "perfect_shortlist_fraction"
                    ],
                }
            )
            continue
        results = _run_screen_batch(
            branch, model, positions, range(0, SCREEN_BATCH), args.workers
        )
        early_stop = (
            np.mean([run["normal_success"] for run in results]) < 0.5
            or np.mean([run["rescue_invoked"] for run in results]) > 0.5
        )
        if not early_stop:
            results.extend(
                _run_screen_batch(
                    branch,
                    model,
                    positions,
                    range(SCREEN_BATCH, SCREEN_TRAJECTORIES),
                    args.workers,
                )
            )
        evaluations = [int(run["total_evaluations"]) for run in results]
        closed_loop_rows.append(
            {
                "branch": branch,
                "screen_status": "stopped_early_poor_normal_search"
                if early_stop
                else "completed_16",
                "trajectories_run": len(results),
                "normal_success_count": sum(run["normal_success"] for run in results),
                "normal_success_fraction": float(
                    np.mean([run["normal_success"] for run in results])
                ),
                "final_success_count": sum(run["final_success"] for run in results),
                "final_success_fraction": float(
                    np.mean([run["final_success"] for run in results])
                ),
                "rescue_invocation_count": sum(run["rescue_invoked"] for run in results),
                "rescue_invocation_fraction": float(
                    np.mean([run["rescue_invoked"] for run in results])
                ),
                "median_evaluations": float(np.median(evaluations)),
                "mean_evaluations": float(np.mean(evaluations)),
                "worst_evaluations": max(evaluations),
                "static_macro_overlap": metrics["macro_overlap"],
                "static_selected_point_retention": metrics[
                    "teacher_selected_point_retention"
                ],
                "static_perfect_shortlist_fraction": metrics[
                    "perfect_shortlist_fraction"
                ],
            }
        )
    _write_csv(CLOSED_LOOP_PATH, closed_loop_rows)
    print("wrote shortlist strategy comparison outputs", flush=True)


if __name__ == "__main__":
    main()
