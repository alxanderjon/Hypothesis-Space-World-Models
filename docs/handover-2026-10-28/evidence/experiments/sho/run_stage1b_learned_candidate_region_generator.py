#!/usr/bin/env python3
"""Learn eDIRECT candidate-region generation with a frozen candidate selector."""

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

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_global_search_imitation as imitation
from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
    UPPER,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_candidate_region_generator")
SOURCE_DIR = Path("outputs/sho/stage1B/learned_global_search_imitation")
SOURCE_TEACHER_TRACE = SOURCE_DIR / "teacher_traces.pt"
SOURCE_SELECTOR_CHECKPOINT = SOURCE_DIR / "best_selector_checkpoint.pt"
SOURCE_SEARCH_TRACE = SOURCE_DIR / "validation_test_search_traces.pt"
TEACHER_TRACE_PATH = OUTPUT_DIR / "teacher_traces.pt"
GENERATOR_CHECKPOINT_PATH = OUTPUT_DIR / "generator_checkpoint.pt"
SELECTOR_CHECKPOINT_PATH = OUTPUT_DIR / "frozen_selector_checkpoint.pt"
TEST_HISTORY_PATH = OUTPUT_DIR / "learned_test_search_histories.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

HISTORY_FEATURE_DIM = 9
GLOBAL_FEATURE_DIM = 10
REGION_COUNT = 25
GENERATOR_HIDDEN = 96
MAX_HISTORY_POINTS = 256
MIN_ACTIVE_REGIONS = 4
OBJECTNESS_THRESHOLD = 0.5
NORMAL_CAP = imitation.NORMAL_CAP
GATE = imitation.GATE
LOSS_FLOOR = edirect.LOSS_FLOOR
TRAIN_STATE_CAP = 16_000
VALIDATION_STATE_CAP = 4_000
SUPERVISED_EPOCHS = 25
SUPERVISED_PATIENCE = 5
FINETUNE_EPOCHS = 10
FINETUNE_PATIENCE = 3
BATCH_SIZE = 24
LEARNING_RATE = 1.0e-3
FINETUNE_LEARNING_RATE = 3.0e-4
CLOSED_LOOP_TRAJECTORIES = 32
CLOSED_LOOP_MAX_EVALUATIONS = 1_000
MODEL_SEED = SEED + 13_100_000
STATE_SAMPLE_SEED = SEED + 13_200_000
FULL_BOUNDS = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)


class CandidateRegionGenerator(nn.Module):
    """DeepSets history encoder producing a fixed set of candidate regions."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(HISTORY_FEATURE_DIM, GENERATOR_HIDDEN),
            nn.SiLU(),
            nn.Linear(GENERATOR_HIDDEN, GENERATOR_HIDDEN),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * GENERATOR_HIDDEN + GLOBAL_FEATURE_DIM, 2 * GENERATOR_HIDDEN),
            nn.SiLU(),
            nn.Linear(2 * GENERATOR_HIDDEN, REGION_COUNT * 5),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(history_features)
        weight = history_mask[:, :, None].to(encoded.dtype)
        mean = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        negative = torch.finfo(encoded.dtype).min
        maximum = torch.where(history_mask[:, :, None], encoded, negative).max(dim=1).values
        raw = self.head(torch.cat((mean, maximum, global_features), dim=1))
        raw = raw.reshape(-1, REGION_COUNT, 5)
        centers = torch.sigmoid(raw[:, :, :2])
        scales = 1.0e-4 + 0.4999 * torch.sigmoid(raw[:, :, 2:4])
        return torch.cat((centers, scales), dim=2), raw[:, :, 4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _history_indices(losses: np.ndarray, maximum: int = MAX_HISTORY_POINTS) -> np.ndarray:
    count = len(losses)
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    chosen: set[int] = set(np.argsort(losses)[:64].tolist())
    chosen.update(range(max(0, count - 64), count))
    uniform = np.linspace(0, count - 1, maximum, dtype=np.int64)
    for index in uniform:
        chosen.add(int(index))
        if len(chosen) >= maximum:
            break
    if len(chosen) < maximum:
        for index in np.argsort(losses):
            chosen.add(int(index))
            if len(chosen) >= maximum:
                break
    selected = np.asarray(sorted(chosen), dtype=np.int64)
    if len(selected) > maximum:
        priority = np.concatenate(
            (np.argsort(losses)[:64], np.arange(max(0, count - 64), count), uniform)
        )
        ordered: list[int] = []
        seen: set[int] = set()
        for value in priority:
            index = int(value)
            if index not in seen:
                seen.add(index)
                ordered.append(index)
            if len(ordered) == maximum:
                break
        selected = np.asarray(sorted(ordered), dtype=np.int64)
    return selected


def _state_features(
    points: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    selected = _history_indices(losses)
    chosen = points[selected]
    log_losses = np.log10(np.maximum(losses, LOSS_FLOOR))
    best_index = int(np.argmin(losses))
    best = points[best_index]
    tree = cKDTree(points)
    if len(points) > 1:
        nearest = tree.query(chosen, k=2)[0][:, 1]
    else:
        nearest = np.ones(len(chosen), dtype=np.float64)
    boundary_distance = np.min(
        np.concatenate((chosen, 1.0 - chosen), axis=1), axis=1
    )
    recency = selected / float(max(1, len(points) - 1))
    features = np.column_stack(
        (
            chosen[:, 0],
            chosen[:, 1],
            log_losses[selected] / 20.0,
            np.linalg.norm(chosen - best[None, :], axis=1) / math.sqrt(2.0),
            nearest / math.sqrt(2.0),
            boundary_distance,
            np.clip(log_losses[selected] - log_losses[best_index], 0.0, 60.0) / 60.0,
            recency,
            (selected == best_index).astype(np.float64),
        )
    ).astype(np.float32)
    span = np.ptp(points, axis=0)
    global_features = np.asarray(
        (
            math.log1p(len(points)) / math.log1p(NORMAL_CAP),
            best[0],
            best[1],
            log_losses[best_index] / 20.0,
            float(np.median(log_losses)) / 20.0,
            float(np.std(log_losses)) / 20.0,
            span[0],
            span[1],
            float(np.min(np.concatenate((best, 1.0 - best)))),
            float(np.mean(losses <= max(GATE, 10.0 * losses[best_index]))),
        ),
        dtype=np.float32,
    )
    return features, global_features


def _candidate_regions(
    points: np.ndarray, candidates: np.ndarray
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    if not len(candidates):
        return np.empty((0, 4), dtype=np.float32)
    tree = cKDTree(points)
    _, owners = tree.query(candidates, k=1)
    if len(points) > 1:
        unique, inverse = np.unique(owners, return_inverse=True)
        spacing = tree.query(points[unique], k=2)[0][:, 1][inverse]
    else:
        spacing = np.ones(len(candidates), dtype=np.float64)
    owner_points = points[owners]
    base = spacing[:, None] / (2.0 * math.sqrt(2.0))
    scales = np.maximum(np.abs(candidates - owner_points), base)
    scales = np.clip(scales, 1.0e-4, 0.5)
    return np.concatenate((candidates, scales), axis=1).astype(np.float32)


def _extract_teacher_states(source: dict[str, Any]) -> dict[str, Any]:
    decisions_by_trajectory: dict[int, list[dict[str, Any]]] = {}
    for decision in source["decisions"]:
        if decision["mode"] == "normal_edirect":
            decisions_by_trajectory.setdefault(int(decision["trajectory_id"]), []).append(
                decision
            )
    states: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for trajectory in source["trajectories"]:
        trajectory_id = int(trajectory["trajectory_id"])
        normal = next(mode for mode in trajectory["modes"] if mode["mode"] == "normal_edirect")
        points = np.asarray(normal["evaluated_hypotheses_normalized"], dtype=np.float64)
        losses = np.asarray(normal["evaluated_prefix_losses"], dtype=np.float64)
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "generator_id": int(trajectory["generator_id"]),
                "evaluated_hypotheses_normalized": points,
                "evaluated_prefix_losses": losses,
                "normal_success": bool(normal["success"]),
            }
        )
        previous_type = ""
        iteration = 0
        for decision in decisions_by_trajectory.get(trajectory_id, []):
            candidate_type = str(decision["candidate_type"])
            if candidate_type == "cell" and previous_type != "cell":
                history_count = int(decision["history_count"])
                history_points = points[:history_count]
                history_losses = losses[:history_count]
                features, global_features = _state_features(history_points, history_losses)
                targets = _candidate_regions(history_points, decision["candidates"])
                if len(targets) > REGION_COUNT:
                    targets = targets[:REGION_COUNT]
                states.append(
                    {
                        "source": "teacher",
                        "trajectory_id": trajectory_id,
                        "generator_id": int(trajectory["generator_id"]),
                        "iteration": iteration,
                        "history_count": history_count,
                        "history_features": features,
                        "global_features": global_features,
                        "target_regions": targets,
                    }
                )
                iteration += 1
            previous_type = candidate_type
    return {"trajectories": trajectories, "states": states}


def _state_indices(states: list[dict[str, Any]], validation: bool) -> list[int]:
    indices = [
        index
        for index, state in enumerate(states)
        if ((int(state["trajectory_id"]) % 5 == 0) == validation)
    ]
    cap = VALIDATION_STATE_CAP if validation else TRAIN_STATE_CAP
    if len(indices) > cap:
        generator = np.random.default_rng(STATE_SAMPLE_SEED + int(validation))
        indices = sorted(generator.choice(indices, cap, replace=False).tolist())
    return indices


def _batch(
    states: list[dict[str, Any]], indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    maximum = max(len(states[index]["history_features"]) for index in indices)
    history = torch.zeros(
        (len(indices), maximum, HISTORY_FEATURE_DIM), dtype=torch.float32
    )
    mask = torch.zeros((len(indices), maximum), dtype=torch.bool)
    global_features = torch.empty(
        (len(indices), GLOBAL_FEATURE_DIM), dtype=torch.float32
    )
    targets: list[torch.Tensor] = []
    for row, index in enumerate(indices):
        state = states[index]
        count = len(state["history_features"])
        history[row, :count] = torch.as_tensor(
            state["history_features"], dtype=torch.float32
        )
        mask[row, :count] = True
        global_features[row] = torch.as_tensor(
            state["global_features"], dtype=torch.float32
        )
        targets.append(torch.as_tensor(state["target_regions"], dtype=torch.float32))
    return history, mask, global_features, targets


def _set_loss(
    predicted: torch.Tensor,
    object_logits: torch.Tensor,
    targets: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    center_loss = predicted.new_zeros(())
    scale_loss = predicted.new_zeros(())
    object_loss = predicted.new_zeros(())
    duplicate_loss = predicted.new_zeros(())
    for row, target in enumerate(targets):
        cost_center = torch.cdist(predicted[row, :, :2].detach(), target[:, :2])
        cost_scale = torch.cdist(
            torch.log(predicted[row, :, 2:4].detach()),
            torch.log(target[:, 2:4].clamp_min(1.0e-4)),
        )
        matched_rows, matched_columns = linear_sum_assignment(
            (cost_center + 0.2 * cost_scale).cpu().numpy()
        )
        rows = torch.as_tensor(matched_rows, dtype=torch.long)
        columns = torch.as_tensor(matched_columns, dtype=torch.long)
        center_loss = center_loss + nn.functional.smooth_l1_loss(
            predicted[row, rows, :2], target[columns, :2]
        )
        scale_loss = scale_loss + nn.functional.smooth_l1_loss(
            torch.log(predicted[row, rows, 2:4]),
            torch.log(target[columns, 2:4].clamp_min(1.0e-4)),
        )
        object_target = torch.zeros(REGION_COUNT, dtype=predicted.dtype)
        object_target[rows] = 1.0
        positive_weight = torch.tensor(
            max(1.0, (REGION_COUNT - len(rows)) / max(1, len(rows))),
            dtype=predicted.dtype,
        )
        object_loss = object_loss + nn.functional.binary_cross_entropy_with_logits(
            object_logits[row], object_target, pos_weight=positive_weight
        )
        centers = predicted[row, :, :2]
        distances = torch.cdist(centers, centers)
        probabilities = torch.sigmoid(object_logits[row])
        pair_weight = probabilities[:, None] * probabilities[None, :]
        off_diagonal = ~torch.eye(REGION_COUNT, dtype=torch.bool)
        duplicate_loss = duplicate_loss + torch.mean(
            pair_weight[off_diagonal] * torch.exp(-distances[off_diagonal].square() / 0.005)
        )
    divisor = float(len(targets))
    center_loss = center_loss / divisor
    scale_loss = scale_loss / divisor
    object_loss = object_loss / divisor
    duplicate_loss = duplicate_loss / divisor
    total = center_loss + 0.25 * scale_loss + 0.25 * object_loss + 0.05 * duplicate_loss
    return total, {
        "center_loss": float(center_loss.detach()),
        "scale_loss": float(scale_loss.detach()),
        "object_loss": float(object_loss.detach()),
        "duplicate_loss": float(duplicate_loss.detach()),
    }


def _generation_metrics(
    model: CandidateRegionGenerator,
    states: list[dict[str, Any]],
    indices: list[int],
) -> dict[str, float]:
    model.eval()
    squared_centers: list[float] = []
    squared_scales: list[float] = []
    distances: list[float] = []
    count_errors: list[float] = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            selected = indices[start : start + BATCH_SIZE]
            history, mask, global_features, targets = _batch(states, selected)
            predicted, object_logits = model(history, mask, global_features)
            for row, target in enumerate(targets):
                target_count = len(target)
                chosen = torch.topk(object_logits[row], k=target_count).indices
                prediction = predicted[row, chosen]
                cost = torch.cdist(prediction[:, :2], target[:, :2])
                matched_rows, matched_columns = linear_sum_assignment(cost.cpu().numpy())
                center_delta = (
                    prediction[matched_rows, :2] - target[matched_columns, :2]
                ).cpu().numpy()
                scale_delta = (
                    prediction[matched_rows, 2:4] - target[matched_columns, 2:4]
                ).cpu().numpy()
                squared_centers.extend(np.square(center_delta).ravel().tolist())
                squared_scales.extend(np.square(scale_delta).ravel().tolist())
                distances.extend(np.linalg.norm(center_delta, axis=1).tolist())
                predicted_count = int(torch.sum(torch.sigmoid(object_logits[row]) >= 0.5))
                count_errors.append(abs(predicted_count - target_count))
    distance_array = np.asarray(distances)
    return {
        "center_rmse": float(math.sqrt(np.mean(squared_centers))),
        "scale_rmse": float(math.sqrt(np.mean(squared_scales))),
        "candidate_count_mae": float(np.mean(count_errors)),
        "teacher_region_recall_distance_le_0.05": float(np.mean(distance_array <= 0.05)),
        "teacher_region_recall_distance_le_0.1": float(np.mean(distance_array <= 0.1)),
        "median_matched_center_distance": float(np.median(distance_array)),
        "worst_matched_center_distance": float(np.max(distance_array)),
    }


def _train_generator(
    model: CandidateRegionGenerator,
    training_states: list[dict[str, Any]],
    validation_states: list[dict[str, Any]],
    *,
    stage: str,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> tuple[CandidateRegionGenerator, list[dict[str, object]], dict[str, float]]:
    training_indices = list(range(len(training_states)))
    validation_indices = list(range(len(validation_states)))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(MODEL_SEED + (1 if stage == "closed_loop" else 0))
    best_state = copy.deepcopy(model.state_dict())
    baseline_metrics = _generation_metrics(
        model, validation_states, validation_indices
    )
    best_score = baseline_metrics["center_rmse"] + 0.25 * baseline_metrics["scale_rmse"]
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(training_indices), generator=generator).tolist()
        accumulated = 0.0
        examples = 0
        for start in range(0, len(permutation), BATCH_SIZE):
            batch_positions = permutation[start : start + BATCH_SIZE]
            selected = [training_indices[position] for position in batch_positions]
            history, mask, global_features, targets = _batch(training_states, selected)
            optimizer.zero_grad(set_to_none=True)
            predicted, object_logits = model(history, mask, global_features)
            loss, _ = _set_loss(predicted, object_logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            accumulated += float(loss.detach()) * len(selected)
            examples += len(selected)
        metrics = _generation_metrics(model, validation_states, validation_indices)
        score = metrics["center_rmse"] + 0.25 * metrics["scale_rmse"]
        row = {
            "row_type": "generator_training_epoch",
            "stage": stage,
            "epoch": epoch,
            "training_loss": accumulated / max(1, examples),
            **metrics,
        }
        rows.append(row)
        print(
            f"generator {stage} epoch={epoch} loss={row['training_loss']:.6f} "
            f"center_rmse={metrics['center_rmse']:.5f} "
            f"recall@0.1={metrics['teacher_region_recall_distance_le_0.1']:.4f}",
            flush=True,
        )
        if score < best_score - 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    metrics = _generation_metrics(model, validation_states, validation_indices)
    return model, rows, {
        "best_epoch": best_epoch,
        "starting_center_rmse": baseline_metrics["center_rmse"],
        "starting_scale_rmse": baseline_metrics["scale_rmse"],
        **metrics,
    }


def _load_selector() -> imitation.CandidateRanker:
    checkpoint = torch.load(
        SELECTOR_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    model = imitation.CandidateRanker().to(dtype=torch.float32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _predict_regions(
    model: CandidateRegionGenerator,
    points: np.ndarray,
    losses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features, global_features = _state_features(points, losses)
    with torch.no_grad():
        predicted, logits = model(
            torch.as_tensor(features, dtype=torch.float32)[None, :, :],
            torch.ones((1, len(features)), dtype=torch.bool),
            torch.as_tensor(global_features, dtype=torch.float32)[None, :],
        )
    return predicted[0].cpu().numpy(), torch.sigmoid(logits[0]).cpu().numpy()


def _active_candidates(
    regions: np.ndarray,
    probabilities: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.flatnonzero(probabilities >= OBJECTNESS_THRESHOLD)
    if len(active) < MIN_ACTIVE_REGIONS:
        active = np.argsort(probabilities)[-MIN_ACTIVE_REGIONS:]
    active = active[np.argsort(probabilities[active])[::-1]]
    candidates: list[np.ndarray] = []
    kept_regions: list[np.ndarray] = []
    for index in active:
        center = np.clip(regions[index, :2], 0.0, 1.0)
        if np.min(np.linalg.norm(points - center[None, :], axis=1)) <= 1.0e-9:
            continue
        if candidates and np.min(
            np.linalg.norm(np.asarray(candidates) - center[None, :], axis=1)
        ) <= 1.0e-7:
            continue
        candidates.append(center)
        kept_regions.append(regions[index])
    if not candidates:
        distances = cKDTree(points).query(regions[:, :2], k=1)[0]
        index = int(np.argmax(distances))
        candidates.append(np.clip(regions[index, :2], 0.0, 1.0))
        kept_regions.append(regions[index])
    return np.asarray(candidates, dtype=np.float64), np.asarray(kept_regions, dtype=np.float32)


def _run_learned_normal(
    *,
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    generator_model: CandidateRegionGenerator,
    selector_model: imitation.CandidateRanker,
    cap: int,
    collect_teacher_targets: bool,
) -> dict[str, Any]:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = edirect._initial_design().astype(np.float64)
    losses = edirect._prefix_losses_numpy(
        LOWER + DOMAIN_WIDTH * points, position
    ).astype(np.float64)
    evaluations = len(points)
    iteration = 0
    states: list[dict[str, Any]] = []
    while evaluations < cap and float(np.min(losses)) > GATE:
        features, global_features = _state_features(points, losses)
        teacher_regions = np.empty((0, 4), dtype=np.float32)
        if collect_teacher_targets:
            teacher_candidates = edirect._cell_proposals(
                points, losses, generator_id, iteration
            )
            teacher_regions = _candidate_regions(points, teacher_candidates)
            if len(teacher_regions) > REGION_COUNT:
                teacher_regions = teacher_regions[:REGION_COUNT]
        predicted_regions, probabilities = _predict_regions(
            generator_model, points, losses
        )
        pending, active_regions = _active_candidates(
            predicted_regions, probabilities, points
        )
        state_record: dict[str, Any] = {
            "source": "closed_loop" if collect_teacher_targets else "learned_search",
            "split": split,
            "trajectory_id": trajectory_id,
            "generator_id": generator_id,
            "iteration": iteration,
            "history_count": len(points),
            "history_features": features if collect_teacher_targets else None,
            "global_features": global_features,
            "target_regions": teacher_regions if collect_teacher_targets else None,
            "predicted_regions": predicted_regions.astype(np.float32),
            "objectness_probabilities": probabilities.astype(np.float32),
            "active_regions": active_regions,
            "selected_hypotheses": [],
            "resulting_prefix_losses": [],
        }
        while len(pending) and evaluations < cap and float(np.min(losses)) > GATE:
            selector_global = imitation._global_features(
                points,
                losses,
                len(pending),
                evaluations,
                cap,
                iteration,
                FULL_BOUNDS,
                "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model,
                points,
                losses,
                pending,
                "cell",
                selector_global,
            )
            proposal = pending[choice]
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            state_record["selected_hypotheses"].append(proposal.copy())
            state_record["resulting_prefix_losses"].append(loss)
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, choice, axis=0)
            evaluations += 1
        state_record["selected_hypotheses"] = np.asarray(
            state_record["selected_hypotheses"], dtype=np.float64
        ).reshape(-1, 2)
        state_record["resulting_prefix_losses"] = np.asarray(
            state_record["resulting_prefix_losses"], dtype=np.float64
        )
        states.append(state_record)
        iteration += 1
    return {
        "split": split,
        "trajectory_id": trajectory_id,
        "generator_id": generator_id,
        "points": points,
        "losses": losses,
        "evaluations": evaluations,
        "success": float(np.min(losses)) <= GATE,
        "states": states,
    }


_WORKER_GENERATOR: CandidateRegionGenerator | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None


def _load_worker_models() -> tuple[CandidateRegionGenerator, imitation.CandidateRanker]:
    global _WORKER_GENERATOR, _WORKER_SELECTOR
    if _WORKER_GENERATOR is None:
        checkpoint = torch.load(
            GENERATOR_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        _WORKER_GENERATOR = CandidateRegionGenerator().to(dtype=torch.float32)
        _WORKER_GENERATOR.load_state_dict(checkpoint["state_dict"])
        _WORKER_GENERATOR.eval()
    if _WORKER_SELECTOR is None:
        _WORKER_SELECTOR = _load_selector()
    return _WORKER_GENERATOR, _WORKER_SELECTOR


def _closed_loop_worker(arguments: tuple[int, np.ndarray]) -> dict[str, Any]:
    trajectory_id, position = arguments
    generator_model, selector_model = _load_worker_models()
    return _run_learned_normal(
        split="train",
        trajectory_id=trajectory_id,
        position=position,
        generator_model=generator_model,
        selector_model=selector_model,
        cap=CLOSED_LOOP_MAX_EVALUATIONS,
        collect_teacher_targets=True,
    )


def _collect_closed_loop(
    positions: np.ndarray, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [(index, positions[index]) for index in range(CLOSED_LOOP_TRAJECTORIES)]
    runs: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            runs.append(_closed_loop_worker(task))
            print(f"closed-loop {len(runs)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_closed_loop_worker, task) for task in tasks]
            for future in as_completed(futures):
                runs.append(future.result())
                if len(runs) % 4 == 0 or len(runs) == len(tasks):
                    print(f"closed-loop {len(runs)}/{len(tasks)}", flush=True)
    runs.sort(key=lambda item: item["trajectory_id"])
    states: list[dict[str, Any]] = []
    for run in runs:
        for state in run["states"]:
            state = dict(state)
            state.pop("predicted_regions", None)
            state.pop("objectness_probabilities", None)
            state.pop("active_regions", None)
            state.pop("selected_hypotheses", None)
            state.pop("resulting_prefix_losses", None)
            if state["target_regions"] is not None and len(state["target_regions"]):
                states.append(state)
    return runs, states


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    generator_model, selector_model = _load_worker_models()
    normal = _run_learned_normal(
        split=split,
        trajectory_id=trajectory_id,
        position=position,
        generator_model=generator_model,
        selector_model=selector_model,
        cap=NORMAL_CAP,
        collect_teacher_targets=False,
    )
    mode = imitation.ModeRun(
        mode="normal_edirect",
        bounds=FULL_BOUNDS.copy(),
        points=normal["points"],
        losses=normal["losses"],
        new_evaluations=normal["evaluations"],
        success=normal["success"],
    )
    modes = [mode]
    total = normal["evaluations"]
    boundary_run = None
    nested_run = None
    if not mode.success:
        boundary_run, boundary_geometry = imitation._run_boundary_rescue(
            split=split,
            trajectory_id=trajectory_id,
            generator_id=normal["generator_id"],
            position=position,
            normal=mode,
            collect_trace=False,
        )
        modes.append(boundary_run)
        total += boundary_run.new_evaluations
    else:
        boundary_geometry = None
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
    final = next((candidate for candidate in modes[::-1] if candidate.success), modes[-1])
    return {
        "split": split,
        "trajectory_id": trajectory_id,
        "generator_id": normal["generator_id"],
        "success": any(candidate.success for candidate in modes),
        "normal_success": mode.success,
        "final_loss": final.best_loss,
        "final_theta": final.best_physical,
        "total_evaluations": total,
        "boundary_invoked": boundary_run is not None,
        "boundary_success": bool(boundary_run and boundary_run.success),
        "nested_invoked": nested_run is not None,
        "nested_success": bool(nested_run and nested_run.success),
        "boundary_geometry": boundary_geometry,
        "normal_history": {
            "evaluated_hypotheses_normalized": normal["points"],
            "evaluated_prefix_losses": normal["losses"],
            "generation_states": normal["states"],
        },
        "rescue_modes": [
            {
                "mode": candidate.mode,
                "bounds": candidate.bounds,
                "evaluated_hypotheses_normalized": candidate.points,
                "evaluated_prefix_losses": candidate.losses,
                "new_evaluations": candidate.new_evaluations,
                "success": candidate.success,
            }
            for candidate in modes[1:]
        ],
    }


def _run_search_split(
    split: str, positions: np.ndarray, workers: int
) -> list[dict[str, Any]]:
    tasks = [(split, index, positions[index]) for index in range(len(positions))]
    runs: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            runs.append(_search_worker(task))
            print(f"learned-generator {split} {len(runs)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                runs.append(future.result())
                if len(runs) % 8 == 0 or len(runs) == len(tasks):
                    print(
                        f"learned-generator {split} {len(runs)}/{len(tasks)}",
                        flush=True,
                    )
    runs.sort(key=lambda item: item["trajectory_id"])
    return runs


def _teacher_runs(source: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return source[f"{split}_teacher"]["trajectories"]


def _summary_search_row(
    split: str, method: str, runs: list[dict[str, Any]]
) -> dict[str, object]:
    successes = [run for run in runs if bool(run["success"])]
    if method.startswith("teacher"):
        normal_success = [
            bool(next(mode for mode in run["modes"] if mode["mode"] == "normal_edirect")["success"])
            for run in runs
        ]
        boundary_invocations = sum(len(run["modes"]) > 1 for run in runs)
        boundary_successes = sum(
            any(mode["mode"] == "boundary_rescue" and mode["success"] for mode in run["modes"])
            for run in runs
        )
        nested_invocations = sum(
            any(mode["mode"] == "nested_1d" for mode in run["modes"])
            for run in runs
        )
        nested_successes = sum(
            any(mode["mode"] == "nested_1d" and mode["success"] for mode in run["modes"])
            for run in runs
        )
    else:
        normal_success = [bool(run["normal_success"]) for run in runs]
        boundary_invocations = sum(bool(run["boundary_invoked"]) for run in runs)
        boundary_successes = sum(bool(run["boundary_success"]) for run in runs)
        nested_invocations = sum(bool(run["nested_invoked"]) for run in runs)
        nested_successes = sum(bool(run["nested_success"]) for run in runs)
    evaluations = [int(run["total_evaluations"]) for run in successes]
    return {
        "row_type": "search_summary",
        "split": split,
        "method": method,
        "trajectories": len(runs),
        "normal_success_count": sum(normal_success),
        "normal_success_fraction": float(np.mean(normal_success)),
        "boundary_rescue_invocations": boundary_invocations,
        "boundary_rescue_successes": boundary_successes,
        "nested_1d_invocations": nested_invocations,
        "nested_1d_successes": nested_successes,
        "final_success_count": len(successes),
        "final_success_fraction": len(successes) / len(runs),
        "median_total_evaluations_success": float(np.median(evaluations)) if evaluations else "",
        "mean_total_evaluations_success": float(np.mean(evaluations)) if evaluations else "",
        "worst_total_evaluations_success": max(evaluations, default=""),
        "gate": GATE,
        "normal_cap": NORMAL_CAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--rebuild-teacher", action="store_true")
    parser.add_argument("--skip-closed-loop", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_TEACHER_TRACE.exists() or not SOURCE_SELECTOR_CHECKPOINT.exists():
        raise FileNotFoundError("existing learned-selector artifacts are required")
    if not SELECTOR_CHECKPOINT_PATH.exists():
        shutil.copy2(SOURCE_SELECTOR_CHECKPOINT, SELECTOR_CHECKPOINT_PATH)
    if _sha256(SELECTOR_CHECKPOINT_PATH) != _sha256(SOURCE_SELECTOR_CHECKPOINT):
        raise RuntimeError("frozen selector checkpoint changed during copy")

    if TEACHER_TRACE_PATH.exists() and not args.rebuild_teacher:
        teacher_payload = torch.load(
            TEACHER_TRACE_PATH, map_location="cpu", weights_only=False
        )
        print(
            f"loaded generator teacher trace with {len(teacher_payload['states'])} states",
            flush=True,
        )
    else:
        source_teacher = torch.load(
            SOURCE_TEACHER_TRACE, map_location="cpu", weights_only=False
        )
        extracted = _extract_teacher_states(source_teacher)
        teacher_payload = {
            "config": {
                "operator": "K(a,b)=[[0,1],[a,b]]",
                "prefix_length": 128,
                "dtype": "float64 objective / float32 networks",
                "region_count": REGION_COUNT,
                "region_scale_rule": "axiswise max(candidate-owner displacement, owner nearest-spacing/(2*sqrt(2)))",
                "history_compression": "best64 + recent64 + deterministic uniform fill to 256; full H_t reconstructable from trajectory history_count",
                "source_teacher_trace": str(SOURCE_TEACHER_TRACE),
                "source_selector_checkpoint_sha256": _sha256(SOURCE_SELECTOR_CHECKPOINT),
            },
            **extracted,
        }
        torch.save(teacher_payload, TEACHER_TRACE_PATH)
        del source_teacher
        gc.collect()
        print(
            f"persisted generator teacher trace with {len(teacher_payload['states'])} states",
            flush=True,
        )

    states = teacher_payload["states"]
    training_indices = _state_indices(states, False)
    validation_indices = _state_indices(states, True)
    training_states = [states[index] for index in training_indices]
    validation_states = [states[index] for index in validation_indices]
    torch.manual_seed(MODEL_SEED)
    generator_model = CandidateRegionGenerator().to(dtype=torch.float32)
    generator_model, supervised_rows, supervised_best = _train_generator(
        generator_model,
        training_states,
        validation_states,
        stage="supervised",
        epochs=SUPERVISED_EPOCHS,
        patience=SUPERVISED_PATIENCE,
        learning_rate=LEARNING_RATE,
    )
    torch.save(
        {
            "state_dict": generator_model.state_dict(),
            "stage": "supervised",
            "metrics": supervised_best,
            "config": teacher_payload["config"],
            "model_seed": MODEL_SEED,
            "selector_checkpoint_sha256": _sha256(SELECTOR_CHECKPOINT_PATH),
        },
        GENERATOR_CHECKPOINT_PATH,
    )

    train_positions, _, _ = _make_exact_labeled_split(
        imitation.TRAIN_COUNT, imitation.TRAIN_SEED
    )
    closed_loop_runs: list[dict[str, Any]] = []
    closed_loop_states: list[dict[str, Any]] = []
    finetune_rows: list[dict[str, object]] = []
    finetune_best = supervised_best
    if not args.skip_closed_loop:
        closed_loop_runs, closed_loop_states = _collect_closed_loop(
            train_positions.to(dtype=DTYPE).cpu().numpy(), args.workers
        )
        combined_states = training_states + closed_loop_states
        generator_model, finetune_rows, finetune_best = _train_generator(
            generator_model,
            combined_states,
            validation_states,
            stage="closed_loop",
            epochs=FINETUNE_EPOCHS,
            patience=FINETUNE_PATIENCE,
            learning_rate=FINETUNE_LEARNING_RATE,
        )
        torch.save(
            {
                "state_dict": generator_model.state_dict(),
                "stage": "closed_loop",
                "metrics": finetune_best,
                "supervised_metrics": supervised_best,
                "config": teacher_payload["config"],
                "model_seed": MODEL_SEED,
                "closed_loop_state_count": len(closed_loop_states),
                "selector_checkpoint_sha256": _sha256(SELECTOR_CHECKPOINT_PATH),
            },
            GENERATOR_CHECKPOINT_PATH,
        )

    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    test_positions, _, _ = _make_exact_labeled_split(
        imitation.TEST_COUNT, imitation.TEST_SEED
    )
    validation_runs = _run_search_split(
        "validation", validation_positions.to(dtype=DTYPE).cpu().numpy(), args.workers
    )
    test_runs = _run_search_split(
        "test", test_positions.to(dtype=DTYPE).cpu().numpy(), args.workers
    )
    source_search = torch.load(
        SOURCE_SEARCH_TRACE, map_location="cpu", weights_only=False
    )
    torch.save(
        {
            "config": {
                "validation_seed": imitation.VALIDATION_SEED,
                "test_seed": imitation.TEST_SEED,
                "gate": GATE,
                "normal_cap": NORMAL_CAP,
                "generator_checkpoint_sha256": _sha256(GENERATOR_CHECKPOINT_PATH),
                "selector_checkpoint_sha256": _sha256(SELECTOR_CHECKPOINT_PATH),
            },
            "validation_learned_generator": validation_runs,
            "test_learned_generator": test_runs,
            "teacher_reference": {
                "source": str(SOURCE_SEARCH_TRACE),
                "validation_trajectories": _teacher_runs(source_search, "validation"),
                "test_trajectories": _teacher_runs(source_search, "test"),
            },
            "closed_loop_training_runs": closed_loop_runs,
        },
        TEST_HISTORY_PATH,
    )

    summary_rows: list[dict[str, object]] = []
    summary_rows.extend(supervised_rows)
    summary_rows.extend(finetune_rows)
    summary_rows.append(
        {
            "row_type": "generator_best",
            "stage": "supervised",
            **supervised_best,
            "training_states": len(training_states),
            "validation_states": len(validation_states),
            "closed_loop_states": 0,
        }
    )
    summary_rows.append(
        {
            "row_type": "generator_best",
            "stage": "closed_loop" if not args.skip_closed_loop else "skipped",
            **finetune_best,
            "training_states": len(training_states),
            "validation_states": len(validation_states),
            "closed_loop_states": len(closed_loop_states),
            "closed_loop_normal_success_fraction": (
                float(np.mean([run["success"] for run in closed_loop_runs]))
                if closed_loop_runs
                else ""
            ),
            "selector_checkpoint_unchanged": 1,
            "selector_checkpoint_sha256": _sha256(SELECTOR_CHECKPOINT_PATH),
            "runtime_seconds_before_frozen_search": time.perf_counter() - started,
        }
    )
    summary_rows.extend(
        (
            _summary_search_row(
                "validation", "teacher_edirect", _teacher_runs(source_search, "validation")
            ),
            _summary_search_row(
                "validation", "learned_generator_frozen_selector", validation_runs
            ),
            _summary_search_row(
                "test", "teacher_edirect", _teacher_runs(source_search, "test")
            ),
            _summary_search_row(
                "test", "learned_generator_frozen_selector", test_runs
            ),
        )
    )
    _write_csv(SUMMARY_PATH, summary_rows)
    print(f"wrote learned candidate-region artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
