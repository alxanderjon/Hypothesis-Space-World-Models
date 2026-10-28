#!/usr/bin/env python3
"""Specialize learned filtering to eDIRECT Cell/Voronoi curation only."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
import os
import shutil
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
from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import _make_exact_labeled_split
from experiments.sho.run_stage1b_two_parameter_learned_search import DOMAIN_WIDTH, LOWER, UPPER
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/specialized_cell_filter")
SOURCE_RAW_TRACE = base.RAW_TRACE_PATH
SOURCE_TEACHER_SEARCH = base.SOURCE_SEARCH_TRACE
SOURCE_SELECTOR = base.SOURCE_SELECTOR
CELL_DATASET_PATH = OUTPUT_DIR / "cell_voronoi_teacher_dataset.pt"
FILTER_CHECKPOINT_PATH = OUTPUT_DIR / "best_specialized_filter_checkpoint.pt"
FROZEN_SELECTOR_PATH = OUTPUT_DIR / "frozen_selector_checkpoint.pt"
DIAGNOSTICS_PATH = OUTPUT_DIR / "validation_test_diagnostics.pt"
SEARCH_HISTORY_PATH = OUTPUT_DIR / "learned_search_histories.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

TRAIN_EXAMPLE_CAP = 12_000
VALIDATION_EXAMPLE_CAP = 4_000
MAX_EPOCHS = 30
PATIENCE = 6
BATCH_SIZE = 64
LEARNING_RATE = 1.0e-3
POSITIVE_BCE_WEIGHT = 0.75
NEGATIVE_BCE_WEIGHT = 0.25
RANKING_WEIGHT = 0.5
COUNT_WEIGHT = 0.1
MODEL_SEED = SEED + 15_500_000
SAMPLE_SEED = SEED + 15_600_000
STRICT_RETENTION_GATE = 1.0
MAX_SHORTLIST_FRACTION = 0.02
MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = base.FULL_BOUNDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class SpecializedCellFilter(base.CandidateFilter):
    """Fresh Cell/Voronoi-only filter; architecture stays permutation equivariant."""


def _cell_indices(states: list[dict[str, Any]], validation: bool) -> np.ndarray:
    available = np.asarray(
        [
            index
            for index, state in enumerate(states)
            if state["candidate_type"] == "cell"
            and ((int(state["trajectory_id"]) % 5 == 0) == validation)
        ],
        dtype=np.int64,
    )
    cap = VALIDATION_EXAMPLE_CAP if validation else TRAIN_EXAMPLE_CAP
    if len(available) > cap:
        generator = np.random.default_rng(SAMPLE_SEED + int(validation))
        available = np.sort(generator.choice(available, cap, replace=False))
    return available


def _balanced_cell_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    predicted_log_count: torch.Tensor,
    target_log_count: torch.Tensor,
) -> torch.Tensor:
    positive = labels & mask
    negative = (~labels) & mask
    raw_bce = nn.functional.binary_cross_entropy_with_logits(
        scores, labels.to(scores.dtype), reduction="none"
    )
    positive_loss = (raw_bce * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    negative_loss = (raw_bce * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1)
    bce = (
        POSITIVE_BCE_WEIGHT * positive_loss + NEGATIVE_BCE_WEIGHT * negative_loss
    ).mean()
    positive_floor = scores.masked_fill(~positive, torch.inf).min(dim=1).values
    hard_negative = scores.masked_fill(~negative, -torch.inf).max(dim=1).values
    ranking = nn.functional.softplus(hard_negative - positive_floor).mean()
    count = nn.functional.mse_loss(predicted_log_count, target_log_count)
    return bce + RANKING_WEIGHT * ranking + COUNT_WEIGHT * count


def _sampled_metrics(
    model: SpecializedCellFilter, examples: dict[str, np.ndarray]
) -> dict[str, float]:
    model.eval()
    true_positive = predicted_positive = actual_positive = selected_retained = 0
    count_errors: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples["history_count"]), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(examples["history_count"])))
            batch = base._batch(examples, indices)
            scores, predicted_log_count = model(*batch[:5])
            loss = _balanced_cell_loss(
                scores, batch[5], batch[4], predicted_log_count, batch[6]
            )
            losses.append(float(loss))
            labels = batch[5].cpu().numpy()
            mask = batch[4].cpu().numpy()
            score_values = scores.cpu().numpy()
            predicted_counts = np.clip(
                np.rint(np.expm1(predicted_log_count.cpu().numpy())).astype(int),
                1,
                mask.sum(axis=1),
            )
            for row in range(len(indices)):
                valid = mask[row]
                count = int(predicted_counts[row])
                order = np.argsort(score_values[row, valid])[::-1]
                predicted = np.zeros(int(valid.sum()), dtype=np.bool_)
                predicted[order[:count]] = True
                truth = labels[row, valid]
                true_positive += int(np.sum(predicted & truth))
                predicted_positive += int(np.sum(predicted))
                actual_positive += int(np.sum(truth))
                # Every positive is retained in the training sample. Using the
                # lowest-ranked teacher positive is stricter than only the first.
                selected_retained += int(np.all(predicted[truth]))
                count_errors.append(abs(count - int(np.sum(truth))))
    return {
        "sampled_loss": float(np.mean(losses)),
        "candidate_precision": true_positive / max(1, predicted_positive),
        "candidate_recall": true_positive / max(1, actual_positive),
        "all_teacher_positives_retained_fraction": selected_retained
        / len(examples["history_count"]),
        "mean_absolute_candidate_count_error": float(np.mean(count_errors)),
    }


def _train_filter(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[SpecializedCellFilter, list[dict[str, object]], dict[str, float]]:
    torch.manual_seed(MODEL_SEED)
    model = SpecializedCellFilter().to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_key = (-math.inf, -math.inf, math.inf)
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(
            len(training["history_count"]), generator=generator
        ).numpy()
        total = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch = base._batch(training, indices)
            optimizer.zero_grad(set_to_none=True)
            scores, predicted_log_count = model(*batch[:5])
            loss = _balanced_cell_loss(
                scores, batch[5], batch[4], predicted_log_count, batch[6]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        metrics = _sampled_metrics(model, validation)
        row = {
            "row_type": "training_epoch",
            "epoch": epoch,
            "training_loss": total / len(training["history_count"]),
            **metrics,
        }
        rows.append(row)
        print(
            f"cell-filter epoch={epoch} train={row['training_loss']:.5f} "
            f"val={metrics['sampled_loss']:.5f} precision={metrics['candidate_precision']:.4f} "
            f"recall={metrics['candidate_recall']:.4f}",
            flush=True,
        )
        key = (
            metrics["candidate_recall"],
            metrics["candidate_precision"],
            -metrics["sampled_loss"],
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "epoch": best_epoch,
                    "sampled_validation_metrics": metrics,
                    "config": {
                        "specialization": "Cell/Voronoi curation only",
                        "model_seed": MODEL_SEED,
                        "positive_bce_weight": POSITIVE_BCE_WEIGHT,
                        "true_parameter_supervision": False,
                    },
                },
                FILTER_CHECKPOINT_PATH,
            )
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, rows, {"best_epoch": best_epoch, **_sampled_metrics(model, validation)}


def _shortlist_count(
    raw_count: int, predicted_log_count: float, multiplier: float
) -> int:
    predicted = max(1.0, math.expm1(float(predicted_log_count)))
    fraction_cap = max(1, int(math.ceil(MAX_SHORTLIST_FRACTION * raw_count)))
    return min(raw_count, fraction_cap, max(1, int(math.ceil(multiplier * predicted))))


def _exhaustive_validation(
    model: SpecializedCellFilter,
    raw_trace: dict[str, Any],
    indices: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], float | None]:
    trajectories = base._trajectory_map(raw_trace)
    states = raw_trace["states"]
    aggregate = {
        multiplier: {
            "tp": 0,
            "predicted": 0,
            "actual": 0,
            "macro_recall": [],
            "count_error": [],
            "selected": [],
            "shortlist_fraction": [],
            "count_ratio": [],
        }
        for multiplier in MULTIPLIERS
    }
    per_state: list[dict[str, object]] = []
    for row, state_index in enumerate(indices):
        record = states[int(state_index)]
        trajectory = trajectories[int(record["trajectory_id"])]
        raw, features, labels, selected_index = base._materialize_raw_state(
            record, trajectory
        )
        history_count = int(record["history_count"])
        points = trajectory["evaluated_hypotheses_normalized"][:history_count]
        losses = trajectory["evaluated_prefix_losses"][:history_count]
        scores, predicted_log_count = base._score_full_raw_pool(
            model,
            points,
            losses,
            features,
            int(record["iteration"]),
            "cell",
        )
        order = np.argsort(scores)[::-1]
        inverse_rank = np.empty(len(order), dtype=np.int64)
        inverse_rank[order] = np.arange(len(order))
        positive_indices = np.flatnonzero(labels)
        actual_count = len(positive_indices)
        base_record = {
            "state_index": int(state_index),
            "trajectory_id": int(record["trajectory_id"]),
            "history_count": history_count,
            "iteration": int(record["iteration"]),
            "raw_candidate_count": len(raw),
            "teacher_curated_count": actual_count,
            "predicted_count_before_multiplier": max(
                1.0, math.expm1(float(predicted_log_count))
            ),
            "teacher_selected_rank": int(inverse_rank[selected_index]) + 1,
            "worst_teacher_curated_rank": int(np.max(inverse_rank[positive_indices])) + 1,
        }
        per_state.append(base_record)
        for multiplier in MULTIPLIERS:
            count = _shortlist_count(len(raw), predicted_log_count, multiplier)
            selected = order[:count]
            retained = np.zeros(len(raw), dtype=np.bool_)
            retained[selected] = True
            tp = int(np.sum(retained & labels))
            bucket = aggregate[multiplier]
            bucket["tp"] += tp
            bucket["predicted"] += count
            bucket["actual"] += actual_count
            bucket["macro_recall"].append(tp / actual_count)
            bucket["count_error"].append(abs(count - actual_count))
            bucket["selected"].append(bool(retained[selected_index]))
            bucket["shortlist_fraction"].append(count / len(raw))
            bucket["count_ratio"].append(count / actual_count)
        if (row + 1) % 128 == 0 or row + 1 == len(indices):
            print(f"exhaustive Cell validation {row + 1}/{len(indices)}", flush=True)

    rows: list[dict[str, object]] = []
    chosen: float | None = None
    for multiplier in MULTIPLIERS:
        bucket = aggregate[multiplier]
        metrics = {
            "row_type": "exhaustive_validation",
            "multiplier": multiplier,
            "candidate_precision": bucket["tp"] / max(1, bucket["predicted"]),
            "candidate_recall": bucket["tp"] / max(1, bucket["actual"]),
            "curated_set_macro_recall": float(np.mean(bucket["macro_recall"])),
            "teacher_selected_point_retention": float(np.mean(bucket["selected"])),
            "mean_absolute_candidate_count_error": float(np.mean(bucket["count_error"])),
            "median_shortlist_to_teacher_count_ratio": float(np.median(bucket["count_ratio"])),
            "mean_shortlist_fraction_of_raw_pool": float(np.mean(bucket["shortlist_fraction"])),
            "worst_shortlist_fraction_of_raw_pool": float(np.max(bucket["shortlist_fraction"])),
            "validation_states": len(indices),
        }
        rows.append(metrics)
        if chosen is None and metrics["teacher_selected_point_retention"] == 1.0:
            chosen = multiplier
    return rows, per_state, chosen


def _cell_shortlist(
    filter_model: SpecializedCellFilter,
    points: np.ndarray,
    losses: np.ndarray,
    raw: np.ndarray,
    raw_features: np.ndarray,
    iteration: int,
    multiplier: float,
) -> tuple[np.ndarray, dict[str, float]]:
    scores, predicted_log_count = base._score_full_raw_pool(
        filter_model, points, losses, raw_features, iteration, "cell"
    )
    count = _shortlist_count(len(raw), predicted_log_count, multiplier)
    indices = np.argpartition(scores, -count)[-count:]
    indices = indices[np.argsort(scores[indices])[::-1]]
    return raw[indices], {
        "raw_candidate_count": len(raw),
        "shortlist_count": len(indices),
        "predicted_log_count": float(predicted_log_count),
        "maximum_filter_score": float(np.max(scores)),
    }


def _run_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    filter_model: SpecializedCellFilter,
    selector_model: imitation.CandidateRanker,
    multiplier: float,
) -> imitation.ModeRun:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    decisions: list[dict[str, Any]] = []
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
                selector_model, points, losses, pending, "initial_design", global_features
            )
        else:
            choice = 0
        proposal = pending[int(choice)]
        loss = float(edirect._prefix_losses_numpy(
            (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
        )[0])
        decisions.append({
            "candidate_type": "initial_design", "history_count": len(points),
            "selected_candidate": proposal.copy(), "resulting_prefix_loss": loss,
        })
        points = np.concatenate((points, proposal[None, :]))
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1

    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        raw, raw_features, _ = base._raw_cell_state(
            points, losses, generator_id, iteration
        )
        pending, filter_info = _cell_shortlist(
            filter_model, points, losses, raw, raw_features, iteration, multiplier
        )
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect"
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, pending, "cell", global_features
            )
            proposal = pending[int(choice)]
            loss = float(edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0])
            decisions.append({
                "candidate_type": "cell", "history_count": len(points),
                **filter_info, "pending_count": len(pending),
                "selected_candidate": proposal.copy(), "resulting_prefix_loss": loss,
            })
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break

        # Greedy proposal remains exactly deterministic and unlearned.
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            singleton = greedy[None, :]
            global_features = imitation._global_features(
                points, losses, 1, evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect"
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, singleton, "greedy", global_features
            )
            proposal = singleton[int(choice)]
            loss = float(edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0])
            decisions.append({
                "candidate_type": "greedy_deterministic", "history_count": len(points),
                "selected_candidate": proposal.copy(), "resulting_prefix_loss": loss,
            })
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1
    return imitation.ModeRun(
        mode="normal_specialized_cell_filter", bounds=FULL_BOUNDS.copy(),
        points=points, losses=losses, new_evaluations=evaluations,
        success=float(np.min(losses)) <= GATE, decisions=decisions,
    )


_WORKER_FILTER: SpecializedCellFilter | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None
_WORKER_MULTIPLIER = 1.0


def _load_worker_models() -> tuple[SpecializedCellFilter, imitation.CandidateRanker, float]:
    global _WORKER_FILTER, _WORKER_SELECTOR, _WORKER_MULTIPLIER
    if _WORKER_FILTER is None:
        checkpoint = torch.load(FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _WORKER_FILTER = SpecializedCellFilter().to(dtype=torch.float32)
        _WORKER_FILTER.load_state_dict(checkpoint["state_dict"])
        _WORKER_FILTER.eval()
        _WORKER_MULTIPLIER = float(checkpoint["shortlist_multiplier"])
    if _WORKER_SELECTOR is None:
        checkpoint = torch.load(FROZEN_SELECTOR_PATH, map_location="cpu", weights_only=False)
        _WORKER_SELECTOR = imitation.CandidateRanker().to(dtype=torch.float32)
        _WORKER_SELECTOR.load_state_dict(checkpoint["state_dict"])
        _WORKER_SELECTOR.eval()
    return _WORKER_FILTER, _WORKER_SELECTOR, _WORKER_MULTIPLIER


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    filter_model, selector_model, multiplier = _load_worker_models()
    normal = _run_normal(
        split, trajectory_id, position, filter_model, selector_model, multiplier
    )
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    boundary_geometry = None
    if not normal.success:
        boundary_run, boundary_geometry = imitation._run_boundary_rescue(
            split=split, trajectory_id=trajectory_id, generator_id=generator_id,
            position=position, normal=normal, collect_trace=False,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    if boundary_run is not None and not boundary_run.success:
        nested_run = imitation._run_nested_rescue(
            split=split, trajectory_id=trajectory_id, position=position,
            boundary_run=boundary_run, collect_trace=False,
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
        "modes": [
            {
                "mode": mode.mode,
                "bounds": mode.bounds,
                "evaluated_hypotheses_normalized": mode.points,
                "evaluated_prefix_losses": mode.losses,
                "new_evaluations": mode.new_evaluations,
                "success": mode.success,
                "decisions": mode.decisions,
            }
            for mode in modes
        ],
    }


def _run_split(split: str, positions: np.ndarray, workers: int) -> list[dict[str, Any]]:
    tasks = [(split, index, positions[index]) for index in range(len(positions))]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_search_worker(task))
            print(f"specialized-cell {split} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(f"specialized-cell {split} {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda item: item["trajectory_id"])
    return results


def _outcome_agreement(
    learned: list[dict[str, Any]], teacher: list[dict[str, Any]]
) -> dict[str, float]:
    teacher_by_id = {int(run["trajectory_id"]): run for run in teacher}
    normal = []
    final = []
    for run in learned:
        reference = teacher_by_id[int(run["trajectory_id"])]
        reference_normal = next(
            mode for mode in reference["modes"] if mode["mode"] == "normal_edirect"
        )
        normal.append(bool(run["normal_success"]) == bool(reference_normal["success"]))
        final.append(bool(run["success"]) == bool(reference["success"]))
    return {
        "normal_outcome_agreement": float(np.mean(normal)),
        "final_outcome_agreement": float(np.mean(final)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (SOURCE_RAW_TRACE, SOURCE_TEACHER_SEARCH, SOURCE_SELECTOR):
        if not path.exists():
            raise FileNotFoundError(path)
    selector_hash = _sha256(SOURCE_SELECTOR)
    if not FROZEN_SELECTOR_PATH.exists():
        shutil.copy2(SOURCE_SELECTOR, FROZEN_SELECTOR_PATH)
    if _sha256(FROZEN_SELECTOR_PATH) != selector_hash:
        raise RuntimeError("frozen selector differs from successful selector checkpoint")

    raw_trace = torch.load(SOURCE_RAW_TRACE, map_location="cpu", weights_only=False)
    all_cell_states = [
        state for state in raw_trace["states"] if state["candidate_type"] == "cell"
    ]
    train_indices = _cell_indices(raw_trace["states"], False)
    validation_indices = _cell_indices(raw_trace["states"], True)
    exhaustive_validation_indices = np.asarray(
        [
            index
            for index, state in enumerate(raw_trace["states"])
            if state["candidate_type"] == "cell"
            and int(state["trajectory_id"]) % 5 == 0
        ],
        dtype=np.int64,
    )

    if CELL_DATASET_PATH.exists() and not args.rebuild_dataset:
        dataset = torch.load(CELL_DATASET_PATH, map_location="cpu", weights_only=False)
        print("loaded Cell/Voronoi-only teacher dataset", flush=True)
    else:
        training_examples = base._prepare_examples(raw_trace, train_indices, "Cell train")
        validation_examples = base._prepare_examples(
            raw_trace, validation_indices, "Cell validation"
        )
        dataset = {
            "config": {
                "specialization": "Cell/Voronoi only; zero greedy examples",
                "source_raw_trace": str(SOURCE_RAW_TRACE),
                "source_raw_trace_sha256": _sha256(SOURCE_RAW_TRACE),
                "raw_pool_storage": "lossless deterministic reconstruction",
                "true_parameter_supervision": False,
            },
            "cell_states": all_cell_states,
            "trajectories": raw_trace["trajectories"],
            "train_state_indices": train_indices,
            "sampled_validation_state_indices": validation_indices,
            "exhaustive_validation_state_indices": exhaustive_validation_indices,
            "training_examples": training_examples,
            "validation_examples": validation_examples,
        }
        torch.save(dataset, CELL_DATASET_PATH)
        print(f"persisted Cell dataset with {len(all_cell_states)} states", flush=True)

    if args.skip_training:
        checkpoint = torch.load(FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = SpecializedCellFilter().to(dtype=torch.float32)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        training_rows: list[dict[str, object]] = []
        training_best = {
            "best_epoch": int(checkpoint["epoch"]),
            **checkpoint["sampled_validation_metrics"],
        }
    else:
        model, training_rows, training_best = _train_filter(
            dataset["training_examples"], dataset["validation_examples"]
        )

    diagnostic_rows, per_state, multiplier = _exhaustive_validation(
        model, raw_trace, exhaustive_validation_indices
    )
    gate_passed = multiplier is not None
    selected_row = next(
        (row for row in diagnostic_rows if row["multiplier"] == multiplier),
        diagnostic_rows[-1],
    )
    remaining_failures = []
    if not gate_passed:
        last_multiplier = MULTIPLIERS[-1]
        for record in per_state:
            cap = min(
                math.ceil(MAX_SHORTLIST_FRACTION * int(record["raw_candidate_count"])),
                math.ceil(last_multiplier * float(record["predicted_count_before_multiplier"])),
            )
            if int(record["teacher_selected_rank"]) > cap:
                remaining_failures.append(record)

    checkpoint = torch.load(FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    checkpoint["shortlist_multiplier"] = multiplier
    checkpoint["strict_gate_passed"] = gate_passed
    checkpoint["exhaustive_validation_metrics"] = selected_row
    checkpoint["frozen_selector_sha256"] = selector_hash
    torch.save(checkpoint, FILTER_CHECKPOINT_PATH)

    rows: list[dict[str, object]] = list(training_rows)
    rows.append({
        "row_type": "training_best", **training_best,
        "training_examples": len(dataset["training_examples"]["history_count"]),
        "sampled_validation_examples": len(dataset["validation_examples"]["history_count"]),
        "cell_teacher_states": len(all_cell_states),
        "greedy_training_examples": 0,
    })
    rows.extend(diagnostic_rows)
    rows.append({
        "row_type": "strict_gate", **selected_row,
        "required_teacher_selected_point_retention": STRICT_RETENTION_GATE,
        "closed_loop_authorized": gate_passed,
        "remaining_selected_point_failures": len(remaining_failures),
        "frozen_selector_sha256": selector_hash,
        "selector_copy_unchanged": _sha256(FROZEN_SELECTOR_PATH) == selector_hash,
    })

    if not gate_passed:
        for failure in remaining_failures:
            rows.append({"row_type": "validation_failure", **failure})
        torch.save({
            "config": {
                "closed_loop_authorized": False,
                "reason": "Cell teacher-selected-point retention below 100%",
                "frozen_selector_sha256": selector_hash,
            },
            "exhaustive_validation_rows": diagnostic_rows,
            "per_validation_state": per_state,
            "remaining_failures": remaining_failures,
            "validation_search": [],
            "test_search": [],
        }, DIAGNOSTICS_PATH)
        torch.save({
            "config": {"closed_loop_authorized": False},
            "validation_learned": [], "test_learned": [],
        }, SEARCH_HISTORY_PATH)
        _write_csv(SUMMARY_PATH, rows)
        print("strict 100% Cell retention gate failed; closed loop was not run", flush=True)
        return

    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    test_positions, _, _ = _make_exact_labeled_split(
        imitation.TEST_COUNT, imitation.TEST_SEED
    )
    validation_runs = _run_split(
        "validation", validation_positions.to(dtype=DTYPE).cpu().numpy(), args.workers
    )
    test_runs = _run_split(
        "test", test_positions.to(dtype=DTYPE).cpu().numpy(), args.workers
    )
    teacher_search = torch.load(SOURCE_TEACHER_SEARCH, map_location="cpu", weights_only=False)
    validation_teacher = base._teacher_runs(teacher_search, "validation")
    test_teacher = base._teacher_runs(teacher_search, "test")
    validation_agreement = _outcome_agreement(validation_runs, validation_teacher)
    test_agreement = _outcome_agreement(test_runs, test_teacher)
    torch.save({
        "config": {
            "closed_loop_authorized": True,
            "shortlist_multiplier": multiplier,
            "filter_checkpoint_sha256": _sha256(FILTER_CHECKPOINT_PATH),
            "frozen_selector_sha256": selector_hash,
        },
        "validation_learned": validation_runs,
        "test_learned": test_runs,
        "teacher_reference": {
            "source": str(SOURCE_TEACHER_SEARCH),
            "validation": validation_teacher,
            "test": test_teacher,
        },
    }, SEARCH_HISTORY_PATH)
    torch.save({
        "exhaustive_validation_rows": diagnostic_rows,
        "per_validation_state": per_state,
        "remaining_failures": [],
        "validation_outcome_agreement": validation_agreement,
        "test_outcome_agreement": test_agreement,
    }, DIAGNOSTICS_PATH)
    rows.extend((
        {**base._search_summary("validation", "teacher_edirect", validation_teacher), **validation_agreement},
        {**base._search_summary("validation", "learned_cell_filter_frozen_selector", validation_runs), **validation_agreement},
        {**base._search_summary("test", "teacher_edirect", test_teacher), **test_agreement},
        {**base._search_summary("test", "learned_cell_filter_frozen_selector", test_runs), **test_agreement},
    ))
    for split, runs in (("validation", validation_runs), ("test", test_runs)):
        for run in runs:
            if not run["success"]:
                rows.append({
                    "row_type": "search_failure", "split": split,
                    "trajectory_id": run["trajectory_id"],
                    "normal_best_loss": run["normal_best_loss"],
                    "final_loss": run["final_loss"],
                })
    rows.append({
        "row_type": "runtime",
        "runtime_seconds_this_execution": time.perf_counter() - started,
    })
    _write_csv(SUMMARY_PATH, rows)
    print(f"wrote specialized Cell-filter artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
