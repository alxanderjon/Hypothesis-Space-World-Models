#!/usr/bin/env python3
"""Learn eDIRECT raw-candidate curation while freezing its learned selector."""

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
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_region_generator as regiongen
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

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_candidate_filter")
SOURCE_DIR = Path("outputs/sho/stage1B/learned_global_search_imitation")
SOURCE_TEACHER_TRACE = SOURCE_DIR / "teacher_traces.pt"
SOURCE_SEARCH_TRACE = SOURCE_DIR / "validation_test_search_traces.pt"
SOURCE_SELECTOR = SOURCE_DIR / "best_selector_checkpoint.pt"
RAW_TRACE_PATH = OUTPUT_DIR / "complete_raw_pool_teacher_traces.pt"
FILTER_DATASET_PATH = OUTPUT_DIR / "filter_training_dataset.pt"
FILTER_CHECKPOINT_PATH = OUTPUT_DIR / "best_filter_checkpoint.pt"
FROZEN_SELECTOR_PATH = OUTPUT_DIR / "frozen_selector_checkpoint.pt"
SEARCH_HISTORY_PATH = OUTPUT_DIR / "learned_search_histories.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

HISTORY_FEATURE_DIM = regiongen.HISTORY_FEATURE_DIM
HISTORY_GLOBAL_DIM = regiongen.GLOBAL_FEATURE_DIM
FILTER_GLOBAL_DIM = HISTORY_GLOBAL_DIM + imitation.GLOBAL_DIM + 3
RAW_FEATURE_DIM = 25
HIDDEN_DIM = 64
MAX_HISTORY_POINTS = regiongen.MAX_HISTORY_POINTS
MAX_EXAMPLE_CANDIDATES = 256
TRAIN_EXAMPLE_CAP = 8_000
VALIDATION_EXAMPLE_CAP = 2_000
FULL_DIAGNOSTIC_CAP = 512
BATCH_SIZE = 64
MAX_EPOCHS = 24
PATIENCE = 5
LEARNING_RATE = 1.0e-3
RANKING_WEIGHT = 0.25
COUNT_WEIGHT = 0.1
SELECTED_RETENTION_GATE = 0.95
MODEL_SEED = SEED + 15_100_000
SAMPLE_SEED = SEED + 15_200_000
FULL_BOUNDS = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class CandidateFilter(nn.Module):
    """DeepSets history encoder with equivariant raw-candidate retention logits."""

    def __init__(self) -> None:
        super().__init__()
        self.history_encoder = nn.Sequential(
            nn.Linear(HISTORY_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(RAW_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        context_dim = 2 * HIDDEN_DIM + FILTER_GLOBAL_DIM
        self.score_head = nn.Sequential(
            nn.Linear(context_dim + HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.count_head = nn.Sequential(
            nn.Linear(context_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def encode_context(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.history_encoder(history_features)
        weights = history_mask[:, :, None].to(encoded.dtype)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        negative = torch.finfo(encoded.dtype).min
        maximum = torch.where(history_mask[:, :, None], encoded, negative).max(dim=1).values
        return torch.cat((mean, maximum, global_features), dim=1)

    def score_candidates(
        self, context: torch.Tensor, candidate_features: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.candidate_encoder(candidate_features)
        expanded = context[:, None, :].expand(-1, encoded.shape[1], -1)
        return self.score_head(torch.cat((encoded, expanded), dim=2)).squeeze(2)

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encode_context(history_features, history_mask, global_features)
        scores = self.score_candidates(context, candidate_features)
        scores = scores.masked_fill(~candidate_mask, 0.0)
        log_count = self.count_head(context).squeeze(1)
        return scores, log_count


def _trajectory_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["trajectory_id"]): item for item in payload["trajectories"]}


def _normal_history(trajectory: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    normal = next(mode for mode in trajectory["modes"] if mode["mode"] == "normal_edirect")
    return (
        np.asarray(normal["evaluated_hypotheses_normalized"], dtype=np.float64),
        np.asarray(normal["evaluated_prefix_losses"], dtype=np.float64),
    )


def _build_raw_trace(source: dict[str, Any]) -> dict[str, Any]:
    """Persist lossless raw-pool recipes and exact teacher curation decisions."""

    decisions_by_trajectory: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(imitation.TRAIN_COUNT)
    }
    for decision in source["decisions"]:
        if decision["mode"] == "normal_edirect":
            decisions_by_trajectory[int(decision["trajectory_id"])].append(decision)
    state_records: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for trajectory in source["trajectories"]:
        trajectory_id = int(trajectory["trajectory_id"])
        points, losses = _normal_history(trajectory)
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "generator_id": int(trajectory["generator_id"]),
                "evaluated_hypotheses_normalized": points,
                "evaluated_prefix_losses": losses,
            }
        )
        decisions = sorted(
            decisions_by_trajectory[trajectory_id], key=lambda item: item["history_count"]
        )
        iteration = 0
        previous_type = ""
        for decision in decisions:
            candidate_type = str(decision["candidate_type"])
            if candidate_type == "cell" and previous_type != "cell":
                candidates = np.asarray(decision["candidates"], dtype=np.float64)
                selected_index = int(decision["selected_index"])
                state_records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "generator_id": int(trajectory["generator_id"]),
                        "history_count": int(decision["history_count"]),
                        "iteration": iteration,
                        "candidate_type": "cell",
                        "teacher_curated_points": candidates,
                        "teacher_selected_point": candidates[selected_index].copy(),
                        "raw_pool_recipe": "edirect._candidate_pool(history_count,generator_id,iteration)",
                    }
                )
            elif candidate_type == "greedy":
                candidates = np.asarray(decision["candidates"], dtype=np.float64)
                selected_index = int(decision["selected_index"])
                state_records.append(
                    {
                        "trajectory_id": trajectory_id,
                        "generator_id": int(trajectory["generator_id"]),
                        "history_count": int(decision["history_count"]),
                        "iteration": iteration,
                        "candidate_type": "greedy",
                        "teacher_curated_points": candidates,
                        "teacher_selected_point": candidates[selected_index].copy(),
                        "raw_pool_recipe": "unique(_candidate_pool(iteration+1000000)+_local_greedy_candidates)",
                    }
                )
                iteration += 1
            previous_type = candidate_type
    return {
        "config": {
            "operator": "K(a,b)=[[0,1],[a,b]]",
            "prefix_length": 128,
            "objective_dtype": "float64",
            "source_teacher_trace": str(SOURCE_TEACHER_TRACE),
            "source_teacher_trace_sha256": _sha256(SOURCE_TEACHER_TRACE),
            "raw_pool_storage": "lossless deterministic reconstruction",
            "raw_pool_note": (
                "Each state stores the full H_t prefix, generator id, iteration, exact "
                "pool recipe, curated coordinates, and selected coordinate. Calling the "
                "unchanged source function exactly reconstructs every raw point."
            ),
            "true_parameter_supervision": False,
        },
        "trajectories": trajectories,
        "states": state_records,
    }


def _common_raw_features(
    points: np.ndarray,
    losses: np.ndarray,
    candidates: np.ndarray,
    candidate_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    distances, owners = tree.query(candidates, k=1)
    best_index = int(np.argmin(losses))
    best = points[best_index]
    log_losses = np.log10(np.maximum(losses, edirect.LOSS_FLOOR))
    if len(points) > 1:
        point_spacing = tree.query(points, k=2)[0][:, 1]
    else:
        point_spacing = np.ones(len(points), dtype=np.float64)
    boundary_distance = np.min(
        np.concatenate((candidates, 1.0 - candidates), axis=1), axis=1
    )
    counts = np.bincount(owners, minlength=len(points)).astype(np.float64)
    features = np.zeros((len(candidates), RAW_FEATURE_DIM), dtype=np.float32)
    features[:, 0:2] = candidates
    features[:, 2] = distances
    features[:, 6] = log_losses[owners] / 20.0
    features[:, 7] = np.clip(
        log_losses[owners] - log_losses[best_index], 0.0, 60.0
    ) / 60.0
    features[:, 8] = point_spacing[owners]
    features[:, 9] = counts[owners] / max(1, len(candidates))
    features[:, 10] = boundary_distance
    features[:, 21] = math.log1p(len(points)) / math.log1p(NORMAL_CAP)
    features[:, 22] = float(candidate_type == "cell")
    features[:, 23] = float(candidate_type == "greedy")
    features[:, 24] = np.linalg.norm(candidates - best[None, :], axis=1) / math.sqrt(2.0)
    return features, distances, owners, log_losses, point_spacing


def _raw_cell_state(
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = edirect._cell_data(points, losses, generator_id, iteration)
    candidates = data.points
    features, distances, owners, log_losses, _ = _common_raw_features(
        points, losses, candidates, "cell"
    )
    features[:, 4] = data.sizes[owners]
    features[:, 5] = edirect._normalize(data.sizes)[owners]
    selected_owner_position = {int(owner): pos for pos, owner in enumerate(data.selected)}
    selected_sizes = data.sizes[data.selected]
    size_weight = edirect._normalize(selected_sizes)
    surface = edirect.GaussianLocalResponse(points, losses)
    loo_weight = edirect._normalize(surface.loo_errors(data.selected))
    weights = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    weights[largest_position] = 1.0
    labels = np.zeros(len(candidates), dtype=np.bool_)
    predicted = np.zeros(len(candidates), dtype=np.float64)
    utility = np.full(len(candidates), -1.0, dtype=np.float64)
    relative_utility = np.zeros(len(candidates), dtype=np.float64)
    for owner, selected_position in selected_owner_position.items():
        indices = np.flatnonzero(owners == owner)
        if not len(indices):
            continue
        local_distances = distances[indices]
        normalized_distance = edirect._normalize(local_distances)
        local_prediction = surface.predict(candidates[indices])
        predicted[indices] = local_prediction
        if selected_position == largest_position:
            local_utility = normalized_distance
        else:
            local_utility = weights[selected_position] * normalized_distance - (
                1.0 - weights[selected_position]
            ) * edirect._normalize(local_prediction)
        utility[indices] = local_utility
        relative_utility[indices] = edirect._normalize(local_utility)
        winner = int(indices[int(np.argmax(local_utility))])
        if distances[winner] > 1.0e-12:
            labels[winner] = True
        features[indices, 11] = local_prediction / 20.0
        features[indices, 12] = np.clip(
            local_prediction - float(np.min(log_losses)), 0.0, 60.0
        ) / 60.0
        features[indices, 13] = 1.0
        features[indices, 14] = 1.0
        features[indices, 15] = float(selected_position == largest_position)
        features[indices, 16] = size_weight[selected_position]
        features[indices, 17] = loo_weight[selected_position]
        features[indices, 18] = weights[selected_position]
    features[:, 19] = np.clip(utility, -1.0, 1.0)
    features[:, 20] = relative_utility
    return candidates, features, labels


def _raw_greedy_state(
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pool = edirect._candidate_pool(len(points), generator_id, iteration + 1_000_000)
    local = edirect._local_greedy_candidates(points, losses)
    candidates = np.unique(np.concatenate((pool, local), axis=0), axis=0)
    features, distances, owners, log_losses, _ = _common_raw_features(
        points, losses, candidates, "greedy"
    )
    surface = edirect.GaussianLocalResponse(points, losses)
    prediction = surface.predict(candidates)
    normalized_prediction = edirect._normalize(prediction)
    utility = 1.0 - normalized_prediction
    features[:, 11] = prediction / 20.0
    features[:, 12] = np.clip(
        prediction - float(np.min(log_losses)), 0.0, 60.0
    ) / 60.0
    features[:, 13] = 1.0
    features[:, 19] = utility
    features[:, 20] = utility
    if len(local):
        local_tree = cKDTree(local)
        features[:, 23] = 1.0 + (
            local_tree.query(candidates, k=1)[0] <= 1.0e-14
        ).astype(np.float32)
    labels = np.zeros(len(candidates), dtype=np.bool_)
    for index in np.argsort(prediction):
        if distances[index] > 1.0e-12:
            labels[int(index)] = True
            break
    return candidates, features, labels


def _materialize_raw_state(
    record: dict[str, Any], trajectory: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    history_count = int(record["history_count"])
    points = trajectory["evaluated_hypotheses_normalized"][:history_count]
    losses = trajectory["evaluated_prefix_losses"][:history_count]
    if record["candidate_type"] == "cell":
        raw, features, labels = _raw_cell_state(
            points, losses, int(record["generator_id"]), int(record["iteration"])
        )
    else:
        raw, features, labels = _raw_greedy_state(
            points, losses, int(record["generator_id"]), int(record["iteration"])
        )
    tree = cKDTree(raw)
    curated = np.asarray(record["teacher_curated_points"], dtype=np.float64)
    distances, retained_indices = tree.query(curated, k=1)
    if float(np.max(distances)) > 1.0e-11:
        raise RuntimeError("teacher curated point is absent from reconstructed raw pool")
    reconstructed = np.zeros(len(raw), dtype=np.bool_)
    reconstructed[np.asarray(retained_indices, dtype=np.int64)] = True
    if not np.array_equal(reconstructed, labels):
        raise RuntimeError("raw curation reconstruction disagrees with saved teacher set")
    selected_distance, selected_index = tree.query(
        np.asarray(record["teacher_selected_point"], dtype=np.float64), k=1
    )
    if float(selected_distance) > 1.0e-11:
        raise RuntimeError("teacher selected point is absent from reconstructed raw pool")
    return raw, features, labels, int(selected_index)


def _filter_global_features(
    points: np.ndarray,
    losses: np.ndarray,
    raw_count: int,
    iteration: int,
    candidate_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    history_features, history_global = regiongen._state_features(points, losses)
    teacher_global = imitation._global_features(
        points,
        losses,
        raw_count,
        len(points),
        NORMAL_CAP,
        iteration,
        FULL_BOUNDS,
        "normal_edirect",
    )
    extra = np.asarray(
        (
            math.log1p(raw_count) / math.log1p(40_000),
            float(candidate_type == "cell"),
            float(candidate_type == "greedy"),
        ),
        dtype=np.float32,
    )
    return history_features, np.concatenate((history_global, teacher_global, extra))


def _sample_state_indices(states: list[dict[str, Any]], validation: bool) -> np.ndarray:
    available = np.asarray(
        [
            index
            for index, state in enumerate(states)
            if (int(state["trajectory_id"]) % 5 == 0) == validation
        ],
        dtype=np.int64,
    )
    cap = VALIDATION_EXAMPLE_CAP if validation else TRAIN_EXAMPLE_CAP
    if len(available) > cap:
        generator = np.random.default_rng(SAMPLE_SEED + int(validation))
        available = np.sort(generator.choice(available, cap, replace=False))
    return available


def _sample_candidates(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    room = MAX_EXAMPLE_CANDIDATES - len(positives)
    hard_count = min(len(negatives), room // 2)
    # Feature 20 is the teacher's available within-locality utility, not a label.
    hard = negatives[np.argsort(features[negatives, 20])[-hard_count:]] if hard_count else np.empty(0, dtype=np.int64)
    remaining = np.setdiff1d(negatives, hard, assume_unique=False)
    random_count = min(len(remaining), room - len(hard))
    generator = np.random.default_rng(seed)
    random = (
        generator.choice(remaining, random_count, replace=False)
        if random_count
        else np.empty(0, dtype=np.int64)
    )
    selected = np.concatenate((positives, hard, random)).astype(np.int64)
    generator.shuffle(selected)
    return features[selected], labels[selected], selected


def _prepare_examples(
    raw_trace: dict[str, Any], indices: np.ndarray, label: str
) -> dict[str, np.ndarray]:
    trajectories = _trajectory_map(raw_trace)
    states = raw_trace["states"]
    count = len(indices)
    history = np.zeros(
        (count, MAX_HISTORY_POINTS, HISTORY_FEATURE_DIM), dtype=np.float16
    )
    history_count = np.zeros(count, dtype=np.uint16)
    global_features = np.empty((count, FILTER_GLOBAL_DIM), dtype=np.float32)
    candidate_features = np.zeros(
        (count, MAX_EXAMPLE_CANDIDATES, RAW_FEATURE_DIM), dtype=np.float16
    )
    candidate_labels = np.zeros((count, MAX_EXAMPLE_CANDIDATES), dtype=np.bool_)
    candidate_count = np.zeros(count, dtype=np.uint16)
    teacher_count = np.zeros(count, dtype=np.uint16)
    raw_count = np.empty(count, dtype=np.int32)
    selected_in_sample = np.zeros(count, dtype=np.bool_)
    for row, state_index in enumerate(indices):
        record = states[int(state_index)]
        trajectory = trajectories[int(record["trajectory_id"])]
        raw, features, labels, selected_index = _materialize_raw_state(record, trajectory)
        sampled_features, sampled_labels, sampled_indices = _sample_candidates(
            features, labels, SAMPLE_SEED + int(state_index)
        )
        prefix_count = int(record["history_count"])
        points = trajectory["evaluated_hypotheses_normalized"][:prefix_count]
        losses = trajectory["evaluated_prefix_losses"][:prefix_count]
        state_history, state_global = _filter_global_features(
            points,
            losses,
            len(raw),
            int(record["iteration"]),
            str(record["candidate_type"]),
        )
        used_history = len(state_history)
        used_candidates = len(sampled_features)
        history[row, :used_history] = state_history.astype(np.float16)
        history_count[row] = used_history
        global_features[row] = state_global
        candidate_features[row, :used_candidates] = sampled_features.astype(np.float16)
        candidate_labels[row, :used_candidates] = sampled_labels
        candidate_count[row] = used_candidates
        teacher_count[row] = int(np.sum(labels))
        raw_count[row] = len(raw)
        selected_in_sample[row] = bool(selected_index in sampled_indices)
        if (row + 1) % 250 == 0 or row + 1 == count:
            print(f"prepared {label} filter examples {row + 1}/{count}", flush=True)
    return {
        "source_state_index": np.asarray(indices, dtype=np.int32),
        "history_features": history,
        "history_count": history_count,
        "global_features": global_features,
        "candidate_features": candidate_features,
        "candidate_labels": candidate_labels,
        "candidate_count": candidate_count,
        "teacher_count": teacher_count,
        "raw_count": raw_count,
        "selected_in_sample": selected_in_sample,
    }


def _batch(
    examples: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[torch.Tensor, ...]:
    history_counts = examples["history_count"][indices].astype(np.int64)
    max_history = int(np.max(history_counts))
    candidate_counts = examples["candidate_count"][indices].astype(np.int64)
    max_candidates = int(np.max(candidate_counts))
    history = torch.as_tensor(
        examples["history_features"][indices, :max_history].astype(np.float32)
    )
    history_mask = torch.arange(max_history)[None, :] < torch.as_tensor(history_counts)[:, None]
    global_features = torch.as_tensor(examples["global_features"][indices])
    candidates = torch.as_tensor(
        examples["candidate_features"][indices, :max_candidates].astype(np.float32)
    )
    candidate_mask = torch.arange(max_candidates)[None, :] < torch.as_tensor(candidate_counts)[:, None]
    labels = torch.as_tensor(
        examples["candidate_labels"][indices, :max_candidates], dtype=torch.bool
    )
    target_log_count = torch.log1p(
        torch.as_tensor(
            examples["teacher_count"][indices].astype(np.float32),
            dtype=torch.float32,
        )
    )
    return (
        history,
        history_mask,
        global_features,
        candidates,
        candidate_mask,
        labels,
        target_log_count,
    )


def _balanced_filter_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    predicted_log_count: torch.Tensor,
    target_log_count: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive = labels & mask
    negative = (~labels) & mask
    raw_bce = nn.functional.binary_cross_entropy_with_logits(
        scores, labels.to(scores.dtype), reduction="none"
    )
    positive_loss = (raw_bce * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    negative_loss = (raw_bce * negative).sum(dim=1) / negative.sum(dim=1).clamp_min(1)
    bce = 0.5 * (positive_loss + negative_loss).mean()
    positive_score = (scores * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    hard_negative = scores.masked_fill(~negative, -torch.inf).max(dim=1).values
    ranking = nn.functional.softplus(hard_negative - positive_score).mean()
    count = nn.functional.mse_loss(predicted_log_count, target_log_count)
    total = bce + RANKING_WEIGHT * ranking + COUNT_WEIGHT * count
    return total, {
        "balanced_bce": float(bce.detach()),
        "ranking_loss": float(ranking.detach()),
        "count_loss": float(count.detach()),
    }


def _sampled_metrics(
    model: CandidateFilter, examples: dict[str, np.ndarray]
) -> dict[str, float]:
    model.eval()
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    count_errors: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples["history_count"]), BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, len(examples["history_count"])))
            batch = _batch(examples, indices)
            scores, predicted_log_count = model(*batch[:5])
            loss, _ = _balanced_filter_loss(
                scores, batch[5], batch[4], predicted_log_count, batch[6]
            )
            losses.append(float(loss))
            labels = batch[5].cpu().numpy()
            mask = batch[4].cpu().numpy()
            predicted_counts = np.clip(
                np.rint(np.expm1(predicted_log_count.cpu().numpy())).astype(int), 1, mask.sum(axis=1)
            )
            score_values = scores.cpu().numpy()
            for row in range(len(indices)):
                count = int(predicted_counts[row])
                order = np.argsort(score_values[row, mask[row]])[::-1]
                predicted = np.zeros(int(mask[row].sum()), dtype=np.bool_)
                predicted[order[:count]] = True
                truth = labels[row, mask[row]]
                true_positive += int(np.sum(predicted & truth))
                predicted_positive += int(np.sum(predicted))
                actual_positive += int(np.sum(truth))
                count_errors.append(abs(count - int(np.sum(truth))))
    return {
        "sampled_loss": float(np.mean(losses)),
        "sampled_precision": true_positive / max(1, predicted_positive),
        "sampled_recall": true_positive / max(1, actual_positive),
        "sampled_mean_absolute_count_error": float(np.mean(count_errors)),
    }


def _train_filter(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[CandidateFilter, list[dict[str, object]], dict[str, float]]:
    torch.manual_seed(MODEL_SEED)
    model = CandidateFilter().to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(
            len(training["history_count"]), generator=generator
        ).numpy()
        train_loss = 0.0
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            batch = _batch(training, indices)
            optimizer.zero_grad(set_to_none=True)
            scores, predicted_log_count = model(*batch[:5])
            loss, _ = _balanced_filter_loss(
                scores, batch[5], batch[4], predicted_log_count, batch[6]
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(indices)
        metrics = _sampled_metrics(model, validation)
        row = {
            "row_type": "training_epoch",
            "epoch": epoch,
            "training_loss": train_loss / len(training["history_count"]),
            **metrics,
        }
        history.append(row)
        print(
            f"filter epoch={epoch} train={row['training_loss']:.5f} "
            f"val={metrics['sampled_loss']:.5f} precision={metrics['sampled_precision']:.4f} "
            f"recall={metrics['sampled_recall']:.4f}",
            flush=True,
        )
        if metrics["sampled_loss"] < best_loss - 1.0e-5:
            best_loss = metrics["sampled_loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "epoch": best_epoch,
                    "sampled_validation_metrics": metrics,
                    "config": {
                        "operator": "K(a,b)=[[0,1],[a,b]]",
                        "raw_feature_dim": RAW_FEATURE_DIM,
                        "filter_global_dim": FILTER_GLOBAL_DIM,
                        "model_seed": MODEL_SEED,
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
    return model, history, {"best_epoch": best_epoch, **_sampled_metrics(model, validation)}


def _score_full_raw_pool(
    model: CandidateFilter,
    points: np.ndarray,
    losses: np.ndarray,
    raw_features: np.ndarray,
    iteration: int,
    candidate_type: str,
) -> tuple[np.ndarray, float]:
    history, global_features = _filter_global_features(
        points, losses, len(raw_features), iteration, candidate_type
    )
    with torch.no_grad():
        context = model.encode_context(
            torch.as_tensor(history, dtype=torch.float32)[None, :, :],
            torch.ones((1, len(history)), dtype=torch.bool),
            torch.as_tensor(global_features, dtype=torch.float32)[None, :],
        )
        log_count = float(model.count_head(context).squeeze())
        scores: list[np.ndarray] = []
        for start in range(0, len(raw_features), 4096):
            candidate_batch = torch.as_tensor(
                raw_features[start : start + 4096], dtype=torch.float32
            )[None, :, :]
            scores.append(
                model.score_candidates(context, candidate_batch)[0].cpu().numpy()
            )
    return np.concatenate(scores), log_count


def _predicted_shortlist_indices(
    scores: np.ndarray,
    predicted_log_count: float,
    multiplier: float,
) -> np.ndarray:
    base_count = max(1.0, math.expm1(float(predicted_log_count)))
    count = min(len(scores), max(1, int(math.ceil(multiplier * base_count))))
    if count == len(scores):
        return np.arange(len(scores), dtype=np.int64)
    return np.argpartition(scores, -count)[-count:].astype(np.int64)


def _full_filter_metrics(
    model: CandidateFilter,
    raw_trace: dict[str, Any],
    indices: np.ndarray,
    multiplier: float,
    label: str,
) -> dict[str, float]:
    trajectories = _trajectory_map(raw_trace)
    states = raw_trace["states"]
    true_positive = 0
    predicted_positive = 0
    actual_positive = 0
    macro_recalls: list[float] = []
    count_errors: list[float] = []
    selected_retained: list[bool] = []
    candidate_ratios: list[float] = []
    for row, state_index in enumerate(indices):
        record = states[int(state_index)]
        trajectory = trajectories[int(record["trajectory_id"])]
        raw, features, labels, selected_index = _materialize_raw_state(record, trajectory)
        history_count = int(record["history_count"])
        points = trajectory["evaluated_hypotheses_normalized"][:history_count]
        losses = trajectory["evaluated_prefix_losses"][:history_count]
        scores, log_count = _score_full_raw_pool(
            model,
            points,
            losses,
            features,
            int(record["iteration"]),
            str(record["candidate_type"]),
        )
        shortlist = _predicted_shortlist_indices(scores, log_count, multiplier)
        predicted = np.zeros(len(raw), dtype=np.bool_)
        predicted[shortlist] = True
        true_positive += int(np.sum(predicted & labels))
        predicted_positive += int(np.sum(predicted))
        actual_positive += int(np.sum(labels))
        recall = float(np.sum(predicted & labels) / max(1, np.sum(labels)))
        macro_recalls.append(recall)
        count_errors.append(abs(int(np.sum(predicted)) - int(np.sum(labels))))
        selected_retained.append(bool(predicted[selected_index]))
        candidate_ratios.append(float(np.sum(predicted) / max(1, np.sum(labels))))
        if (row + 1) % 64 == 0 or row + 1 == len(indices):
            print(
                f"full-filter {label} multiplier={multiplier:g} {row + 1}/{len(indices)}",
                flush=True,
            )
    return {
        "multiplier": multiplier,
        "candidate_precision": true_positive / max(1, predicted_positive),
        "candidate_recall": true_positive / max(1, actual_positive),
        "teacher_curated_set_macro_recall": float(np.mean(macro_recalls)),
        "mean_absolute_candidate_count_error": float(np.mean(count_errors)),
        "median_shortlist_to_teacher_count_ratio": float(np.median(candidate_ratios)),
        "teacher_selected_point_retention": float(np.mean(selected_retained)),
        "diagnostic_states": len(indices),
    }


def _calibrate_filter(
    model: CandidateFilter,
    raw_trace: dict[str, Any],
    validation_indices: np.ndarray,
) -> tuple[float, list[dict[str, float]], dict[str, float]]:
    if len(validation_indices) > FULL_DIAGNOSTIC_CAP:
        generator = np.random.default_rng(SAMPLE_SEED + 91)
        diagnostic = np.sort(
            generator.choice(validation_indices, FULL_DIAGNOSTIC_CAP, replace=False)
        )
    else:
        diagnostic = validation_indices
    rows: list[dict[str, float]] = []
    selected_multiplier = 8.0
    selected_metrics: dict[str, float] | None = None
    for multiplier in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        metrics = _full_filter_metrics(
            model, raw_trace, diagnostic, multiplier, "validation"
        )
        rows.append(metrics)
        if (
            selected_metrics is None
            and metrics["teacher_selected_point_retention"] >= SELECTED_RETENTION_GATE
        ):
            selected_multiplier = multiplier
            selected_metrics = metrics
            break
    if selected_metrics is None:
        selected_metrics = rows[-1]
    return selected_multiplier, rows, selected_metrics


def _load_selector() -> imitation.CandidateRanker:
    checkpoint = torch.load(FROZEN_SELECTOR_PATH, map_location="cpu", weights_only=False)
    model = imitation.CandidateRanker().to(dtype=torch.float32)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _filter_shortlist(
    filter_model: CandidateFilter,
    points: np.ndarray,
    losses: np.ndarray,
    raw: np.ndarray,
    raw_features: np.ndarray,
    iteration: int,
    candidate_type: str,
    multiplier: float,
) -> tuple[np.ndarray, dict[str, float]]:
    scores, log_count = _score_full_raw_pool(
        filter_model,
        points,
        losses,
        raw_features,
        iteration,
        candidate_type,
    )
    indices = _predicted_shortlist_indices(scores, log_count, multiplier)
    order = indices[np.argsort(scores[indices])[::-1]]
    return raw[order], {
        "raw_candidate_count": len(raw),
        "shortlist_count": len(order),
        "predicted_log_count": log_count,
        "maximum_filter_score": float(np.max(scores)),
    }


def _run_filtered_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    filter_model: CandidateFilter,
    selector_model: imitation.CandidateRanker,
    multiplier: float,
) -> imitation.ModeRun:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    decisions: list[dict[str, Any]] = []
    evaluations = 0
    iteration = 0

    # There is no teacher curation in the initial design, so this stage is
    # intentionally identical to the successful frozen-selector pipeline.
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
        proposal = pending[int(choice)]
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        decisions.append(
            {
                "candidate_type": "initial_design",
                "history_count": len(points),
                "raw_candidate_count": len(pending),
                "shortlist_count": len(pending),
                "selected_candidate": proposal.copy(),
                "resulting_prefix_loss": loss,
            }
        )
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1

    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        raw, raw_features, _ = _raw_cell_state(
            points, losses, generator_id, iteration
        )
        pending, filter_info = _filter_shortlist(
            filter_model,
            points,
            losses,
            raw,
            raw_features,
            iteration,
            "cell",
            multiplier,
        )
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
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            decisions.append(
                {
                    "candidate_type": "cell",
                    "history_count": len(points),
                    **filter_info,
                    "pending_count": len(pending),
                    "selected_candidate": proposal.copy(),
                    "resulting_prefix_loss": loss,
                }
            )
            points = np.concatenate((points, proposal[None, :]), axis=0)
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break
        raw, raw_features, _ = _raw_greedy_state(
            points, losses, generator_id, iteration
        )
        pending, filter_info = _filter_shortlist(
            filter_model,
            points,
            losses,
            raw,
            raw_features,
            iteration,
            "greedy",
            multiplier,
        )
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
            selector_model, points, losses, pending, "greedy", global_features
        )
        proposal = pending[int(choice)]
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        decisions.append(
            {
                "candidate_type": "greedy",
                "history_count": len(points),
                **filter_info,
                "pending_count": len(pending),
                "selected_candidate": proposal.copy(),
                "resulting_prefix_loss": loss,
            }
        )
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray((loss,))))
        evaluations += 1
        iteration += 1
    return imitation.ModeRun(
        mode="normal_learned_filter_frozen_selector",
        bounds=FULL_BOUNDS.copy(),
        points=points,
        losses=losses,
        new_evaluations=evaluations,
        success=float(np.min(losses)) <= GATE,
        decisions=decisions,
    )


_WORKER_FILTER: CandidateFilter | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None
_WORKER_MULTIPLIER = 1.0


def _load_worker_models() -> tuple[CandidateFilter, imitation.CandidateRanker, float]:
    global _WORKER_FILTER, _WORKER_SELECTOR, _WORKER_MULTIPLIER
    if _WORKER_FILTER is None:
        checkpoint = torch.load(
            FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        _WORKER_FILTER = CandidateFilter().to(dtype=torch.float32)
        _WORKER_FILTER.load_state_dict(checkpoint["state_dict"])
        _WORKER_FILTER.eval()
        _WORKER_MULTIPLIER = float(checkpoint["shortlist_multiplier"])
    if _WORKER_SELECTOR is None:
        _WORKER_SELECTOR = _load_selector()
    return _WORKER_FILTER, _WORKER_SELECTOR, _WORKER_MULTIPLIER


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    filter_model, selector_model, multiplier = _load_worker_models()
    normal = _run_filtered_normal(
        split,
        trajectory_id,
        position,
        filter_model,
        selector_model,
        multiplier,
    )
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = None
    nested_run = None
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
            print(f"learned-filter {split} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(
                        f"learned-filter {split} {len(results)}/{len(tasks)}",
                        flush=True,
                    )
    results.sort(key=lambda item: item["trajectory_id"])
    return results


def _teacher_runs(source: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return source[f"{split}_teacher"]["trajectories"]


def _search_summary(
    split: str, method: str, runs: list[dict[str, Any]]
) -> dict[str, object]:
    teacher = method == "teacher_edirect"
    successes = [run for run in runs if bool(run["success"])]
    if teacher:
        normal_success = [
            bool(next(mode for mode in run["modes"] if mode["mode"] == "normal_edirect")["success"])
            for run in runs
        ]
        boundary_invoked = [len(run["modes"]) > 1 for run in runs]
        boundary_success = [
            any(mode["mode"] == "boundary_rescue" and mode["success"] for mode in run["modes"])
            for run in runs
        ]
        nested_invoked = [
            any(mode["mode"] == "nested_1d" for mode in run["modes"])
            for run in runs
        ]
        nested_success = [
            any(mode["mode"] == "nested_1d" and mode["success"] for mode in run["modes"])
            for run in runs
        ]
    else:
        normal_success = [bool(run["normal_success"]) for run in runs]
        boundary_invoked = [bool(run["boundary_invoked"]) for run in runs]
        boundary_success = [bool(run["boundary_success"]) for run in runs]
        nested_invoked = [bool(run["nested_invoked"]) for run in runs]
        nested_success = [bool(run["nested_success"]) for run in runs]
    evaluations = [int(run["total_evaluations"]) for run in successes]
    return {
        "row_type": "search_summary",
        "split": split,
        "method": method,
        "trajectories": len(runs),
        "normal_success_count": sum(normal_success),
        "normal_success_fraction": float(np.mean(normal_success)),
        "rescue_invocation_count": sum(boundary_invoked),
        "rescue_invocation_fraction": float(np.mean(boundary_invoked)),
        "boundary_success_count": sum(boundary_success),
        "nested_invocation_count": sum(nested_invoked),
        "nested_success_count": sum(nested_success),
        "final_success_count": len(successes),
        "final_success_fraction": len(successes) / len(runs),
        "median_evaluations_success": float(np.median(evaluations)) if evaluations else "",
        "mean_evaluations_success": float(np.mean(evaluations)) if evaluations else "",
        "worst_evaluations_success": max(evaluations, default=""),
        "gate": GATE,
        "normal_cap": NORMAL_CAP,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--rebuild-trace", action="store_true")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for required in (SOURCE_TEACHER_TRACE, SOURCE_SEARCH_TRACE, SOURCE_SELECTOR):
        if not required.exists():
            raise FileNotFoundError(required)

    selector_hash_before = _sha256(SOURCE_SELECTOR)
    if not FROZEN_SELECTOR_PATH.exists():
        shutil.copy2(SOURCE_SELECTOR, FROZEN_SELECTOR_PATH)
    if _sha256(FROZEN_SELECTOR_PATH) != selector_hash_before:
        raise RuntimeError("frozen selector copy differs from successful checkpoint")

    if RAW_TRACE_PATH.exists() and not args.rebuild_trace:
        raw_trace = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
        print(f"loaded raw trace with {len(raw_trace['states'])} states", flush=True)
    else:
        source = torch.load(SOURCE_TEACHER_TRACE, map_location="cpu", weights_only=False)
        raw_trace = _build_raw_trace(source)
        torch.save(raw_trace, RAW_TRACE_PATH)
        del source
        gc.collect()
        print(f"persisted raw trace with {len(raw_trace['states'])} states", flush=True)

    train_indices = _sample_state_indices(raw_trace["states"], False)
    validation_indices = _sample_state_indices(raw_trace["states"], True)
    if FILTER_DATASET_PATH.exists() and not args.rebuild_dataset:
        dataset = torch.load(FILTER_DATASET_PATH, map_location="cpu", weights_only=False)
        print("loaded filter training dataset", flush=True)
    else:
        training_examples = _prepare_examples(raw_trace, train_indices, "train")
        validation_examples = _prepare_examples(raw_trace, validation_indices, "validation")
        dataset = {
            "config": {
                "raw_trace": str(RAW_TRACE_PATH),
                "raw_trace_sha256": _sha256(RAW_TRACE_PATH),
                "train_example_cap": TRAIN_EXAMPLE_CAP,
                "validation_example_cap": VALIDATION_EXAMPLE_CAP,
                "negative_sampling": "half highest teacher-available utility, half deterministic random",
                "true_parameter_supervision": False,
            },
            "train_state_indices": train_indices,
            "validation_state_indices": validation_indices,
            "training_examples": training_examples,
            "validation_examples": validation_examples,
        }
        torch.save(dataset, FILTER_DATASET_PATH)
        print("persisted filter training dataset", flush=True)

    if args.skip_training:
        checkpoint = torch.load(
            FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        model = CandidateFilter().to(dtype=torch.float32)
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

    multiplier, calibration_rows, selected_metrics = _calibrate_filter(
        model, raw_trace, validation_indices
    )
    checkpoint = torch.load(
        FILTER_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    checkpoint["shortlist_multiplier"] = multiplier
    checkpoint["full_filter_validation_metrics"] = selected_metrics
    checkpoint["frozen_selector_sha256"] = selector_hash_before
    torch.save(checkpoint, FILTER_CHECKPOINT_PATH)

    rows: list[dict[str, object]] = list(training_rows)
    rows.append(
        {
            "row_type": "training_best",
            **training_best,
            "train_examples": len(dataset["training_examples"]["history_count"]),
            "validation_examples": len(dataset["validation_examples"]["history_count"]),
            "raw_teacher_states": len(raw_trace["states"]),
        }
    )
    for metrics in calibration_rows:
        rows.append({"row_type": "filter_calibration", **metrics})
    rows.append(
        {
            "row_type": "filter_gate",
            **selected_metrics,
            "selected_multiplier": multiplier,
            "required_teacher_selected_point_retention": SELECTED_RETENTION_GATE,
            "closed_loop_authorized": bool(
                selected_metrics["teacher_selected_point_retention"]
                >= SELECTED_RETENTION_GATE
            ),
            "frozen_selector_sha256": selector_hash_before,
            "selector_copy_unchanged": _sha256(FROZEN_SELECTOR_PATH)
            == selector_hash_before,
        }
    )

    if selected_metrics["teacher_selected_point_retention"] < SELECTED_RETENTION_GATE:
        torch.save(
            {
                "config": {
                    "closed_loop_authorized": False,
                    "reason": "teacher-selected-point retention below 0.95 gate",
                    "filter_checkpoint_sha256": _sha256(FILTER_CHECKPOINT_PATH),
                    "frozen_selector_sha256": selector_hash_before,
                },
                "validation_learned": [],
                "test_learned": [],
            },
            SEARCH_HISTORY_PATH,
        )
        _write_csv(SUMMARY_PATH, rows)
        print("filter gate failed; closed-loop search was not run", flush=True)
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
    teacher_search = torch.load(
        SOURCE_SEARCH_TRACE, map_location="cpu", weights_only=False
    )
    torch.save(
        {
            "config": {
                "validation_seed": imitation.VALIDATION_SEED,
                "test_seed": imitation.TEST_SEED,
                "gate": GATE,
                "normal_cap": NORMAL_CAP,
                "shortlist_multiplier": multiplier,
                "filter_checkpoint_sha256": _sha256(FILTER_CHECKPOINT_PATH),
                "frozen_selector_sha256": selector_hash_before,
                "selector_copy_unchanged": _sha256(FROZEN_SELECTOR_PATH)
                == selector_hash_before,
            },
            "validation_learned": validation_runs,
            "test_learned": test_runs,
            "teacher_reference": {
                "source": str(SOURCE_SEARCH_TRACE),
                "validation": _teacher_runs(teacher_search, "validation"),
                "test": _teacher_runs(teacher_search, "test"),
            },
        },
        SEARCH_HISTORY_PATH,
    )
    rows.extend(
        (
            _search_summary(
                "validation", "teacher_edirect", _teacher_runs(teacher_search, "validation")
            ),
            _search_summary(
                "validation", "learned_filter_frozen_selector", validation_runs
            ),
            _search_summary(
                "test", "teacher_edirect", _teacher_runs(teacher_search, "test")
            ),
            _search_summary("test", "learned_filter_frozen_selector", test_runs),
        )
    )
    for split, runs in (("validation", validation_runs), ("test", test_runs)):
        for run in runs:
            if not run["success"]:
                geometry = run["boundary_geometry"] or {}
                rows.append(
                    {
                        "row_type": "failure",
                        "split": split,
                        "method": "learned_filter_frozen_selector",
                        "trajectory_id": run["trajectory_id"],
                        "normal_best_loss": run["normal_best_loss"],
                        "final_loss": run["final_loss"],
                        "selected_rescue_face": geometry.get("face", ""),
                        "rescue_strip_delta": geometry.get("delta", ""),
                    }
                )
    rows.append(
        {
            "row_type": "runtime",
            "runtime_seconds_this_execution": time.perf_counter() - started,
        }
    )
    _write_csv(SUMMARY_PATH, rows)
    print(f"wrote learned-candidate-filter artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
