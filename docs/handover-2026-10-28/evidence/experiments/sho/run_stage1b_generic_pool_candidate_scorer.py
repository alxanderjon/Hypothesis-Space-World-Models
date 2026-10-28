#!/usr/bin/env python3
"""Imitate eDIRECT actions by scoring a broad, deterministic generic pool."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc
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

OUTPUT_DIR = Path("outputs/sho/stage1B/generic_pool_candidate_scorer")
SOURCE_DIR = Path("outputs/sho/stage1B/learned_global_search_imitation")
SOURCE_TEACHER_TRACE = SOURCE_DIR / "teacher_traces.pt"
SOURCE_SEARCH_TRACE = SOURCE_DIR / "validation_test_search_traces.pt"
DATASET_PATH = OUTPUT_DIR / "generic_pool_teacher_dataset.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "best_scorer_checkpoint.pt"
SEARCH_TRACE_PATH = OUTPUT_DIR / "learned_search_traces.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"

POOL_SIZE = 256
GLOBAL_COUNT = 128
LOCAL_COUNT = 64
SPARSE_COUNT = 32
EDGE_COUNT = 32
CANDIDATE_FEATURE_DIM = 13
HISTORY_FEATURE_DIM = regiongen.HISTORY_FEATURE_DIM
GLOBAL_FEATURE_DIM = regiongen.GLOBAL_FEATURE_DIM
HIDDEN_DIM = 80
MAX_HISTORY_POINTS = regiongen.MAX_HISTORY_POINTS
TRAIN_EXAMPLE_CAP = 20_000
VALIDATION_EXAMPLE_CAP = 5_000
BATCH_SIZE = 256
MAX_EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 1.0e-3
MODEL_SEED = SEED + 14_100_000
POOL_SEED = SEED + 14_200_000
SAMPLE_SEED = SEED + 14_300_000
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = np.asarray(((LOWER, UPPER), (LOWER, UPPER)), dtype=np.float64)

if GLOBAL_COUNT + LOCAL_COUNT + SPARSE_COUNT + EDGE_COUNT != POOL_SIZE:
    raise RuntimeError("generic candidate-pool counts must sum to POOL_SIZE")


class GenericCandidateScorer(nn.Module):
    """Permutation-invariant history encoder and equivariant pool scorer."""

    def __init__(self) -> None:
        super().__init__()
        self.history_encoder = nn.Sequential(
            nn.Linear(HISTORY_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(CANDIDATE_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * HIDDEN_DIM + GLOBAL_FEATURE_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: torch.Tensor,
        global_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded_history = self.history_encoder(history_features)
        weights = history_mask[:, :, None].to(encoded_history.dtype)
        mean = (encoded_history * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        negative = torch.finfo(encoded_history.dtype).min
        maximum = torch.where(
            history_mask[:, :, None], encoded_history, negative
        ).max(dim=1).values
        encoded_candidates = self.candidate_encoder(candidate_features)
        context = torch.cat((mean, maximum, global_features), dim=1)
        context = context[:, None, :].expand(-1, POOL_SIZE, -1)
        return self.head(torch.cat((encoded_candidates, context), dim=2)).squeeze(2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _sobol(count: int, seed: int) -> np.ndarray:
    exponent = int(math.ceil(math.log2(max(2, count))))
    return qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(exponent)[:count]


def _generic_pool(
    points: np.ndarray,
    losses: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 128 global, 64 local, 32 sparse, and 32 edge proposals."""

    points = np.asarray(points, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    tree = cKDTree(points)
    global_points = _sobol(GLOBAL_COUNT, POOL_SEED + 17 * step)

    if len(points) > 1:
        point_spacing = tree.query(points, k=2)[0][:, 1]
        default_spacing = float(np.median(point_spacing))
    else:
        point_spacing = np.asarray((0.25,), dtype=np.float64)
        default_spacing = 0.25
    best_indices = np.argsort(losses)[: min(8, len(points))]
    angles = (
        np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        + (step % 17) * (math.pi / 68.0)
    )
    radius_multipliers = np.asarray((0.35, 0.6, 1.0, 1.6, 0.45, 0.8, 1.25, 2.0))
    local: list[np.ndarray] = []
    for proposal_index in range(LOCAL_COUNT):
        owner = int(best_indices[proposal_index % len(best_indices)])
        direction_index = (proposal_index // len(best_indices)) % len(angles)
        spacing = float(point_spacing[owner]) if len(points) > 1 else default_spacing
        radius = max(spacing, 1.0e-3) * radius_multipliers[direction_index]
        direction = np.asarray((math.cos(angles[direction_index]), math.sin(angles[direction_index])))
        local.append(np.clip(points[owner] + radius * direction, 0.0, 1.0))
    local_points = np.asarray(local, dtype=np.float64)

    sparse_reservoir = _sobol(1024, POOL_SEED + 1_000_003 + 31 * step)
    sparse_distances = tree.query(sparse_reservoir, k=1)[0]
    sparse_points = sparse_reservoir[
        np.argsort(sparse_distances)[-SPARSE_COUNT:][::-1]
    ]

    edge_coordinates = _sobol(32, POOL_SEED + 2_000_003 + 43 * step)[:, 0]
    inward = min(max(0.5 * default_spacing, 1.0e-4), 0.15)
    edge: list[np.ndarray] = []
    for axis in (0, 1):
        other = 1 - axis
        for side in (0.0, 1.0):
            for local_index in range(8):
                candidate = np.empty(2, dtype=np.float64)
                candidate[other] = edge_coordinates[
                    axis * 16 + int(side) * 8 + local_index
                ]
                if local_index < 4:
                    candidate[axis] = side
                else:
                    candidate[axis] = inward if side == 0.0 else 1.0 - inward
                edge.append(candidate)
    edge_points = np.asarray(edge, dtype=np.float64)

    pool = np.concatenate((global_points, local_points, sparse_points, edge_points), axis=0)
    sources = np.concatenate(
        (
            np.zeros(GLOBAL_COUNT, dtype=np.int64),
            np.ones(LOCAL_COUNT, dtype=np.int64),
            np.full(SPARSE_COUNT, 2, dtype=np.int64),
            np.full(EDGE_COUNT, 3, dtype=np.int64),
        )
    )
    return pool, sources


def _candidate_features(
    points: np.ndarray,
    losses: np.ndarray,
    candidates: np.ndarray,
    sources: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    tree = cKDTree(points)
    neighbor_count = min(8, len(points))
    distances, indices = tree.query(candidates, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    log_losses = np.log10(np.maximum(losses, edirect.LOSS_FLOOR))
    bandwidth = np.maximum(distances[:, -1:], 1.0e-12)
    weights = np.exp(-4.0 * np.square(distances / bandwidth)) + 1.0e-12
    weighted_loss = np.sum(weights * log_losses[indices], axis=1) / np.sum(weights, axis=1)
    nearby_min = np.min(log_losses[indices], axis=1)
    best = points[int(np.argmin(losses))]
    boundary_distance = np.min(
        np.concatenate((candidates, 1.0 - candidates), axis=1), axis=1
    )
    source_one_hot = np.eye(4, dtype=np.float64)[sources]
    features = np.column_stack(
        (
            candidates[:, 0],
            candidates[:, 1],
            np.linalg.norm(candidates - best[None, :], axis=1) / math.sqrt(2.0),
            distances[:, 0] / math.sqrt(2.0),
            1.0 / (1.0 + 10.0 * distances[:, -1]),
            log_losses[indices[:, 0]] / 20.0,
            weighted_loss / 20.0,
            nearby_min / 20.0,
            boundary_distance,
            source_one_hot,
        )
    )
    return features.astype(np.float32)


def _soft_target(candidates: np.ndarray, teacher_action: np.ndarray) -> tuple[np.ndarray, int, float]:
    distances = np.linalg.norm(candidates - teacher_action[None, :], axis=1)
    nearest_index = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_index])
    temperature = max(0.02, nearest_distance)
    logits = -np.square(distances) / (2.0 * temperature * temperature)
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    probabilities /= float(np.sum(probabilities))
    return probabilities.astype(np.float32), nearest_index, nearest_distance


def _teacher_state_table(source: dict[str, Any]) -> dict[str, Any]:
    trajectories = []
    normal_by_trajectory: dict[int, dict[str, Any]] = {}
    for trajectory in source["trajectories"]:
        normal = next(mode for mode in trajectory["modes"] if mode["mode"] == "normal_edirect")
        trajectory_id = int(trajectory["trajectory_id"])
        record = {
            "trajectory_id": trajectory_id,
            "generator_id": int(trajectory["generator_id"]),
            "evaluated_hypotheses_normalized": np.asarray(
                normal["evaluated_hypotheses_normalized"], dtype=np.float64
            ),
            "evaluated_prefix_losses": np.asarray(
                normal["evaluated_prefix_losses"], dtype=np.float64
            ),
            "normal_success": bool(normal["success"]),
        }
        trajectories.append(record)
        normal_by_trajectory[trajectory_id] = record

    trajectory_ids: list[int] = []
    history_counts: list[int] = []
    teacher_actions: list[np.ndarray] = []
    resulting_losses: list[float] = []
    candidate_types: list[int] = []
    type_index = {"initial_design": 0, "cell": 1, "greedy": 2}
    for decision in source["decisions"]:
        if decision["mode"] != "normal_edirect":
            continue
        trajectory_id = int(decision["trajectory_id"])
        selected_index = int(decision["selected_index"])
        trajectory_ids.append(trajectory_id)
        history_counts.append(int(decision["history_count"]))
        teacher_actions.append(
            np.asarray(decision["candidates"][selected_index], dtype=np.float64)
        )
        resulting_losses.append(float(decision["resulting_prefix_loss"]))
        candidate_types.append(type_index[str(decision["candidate_type"])])
    return {
        "trajectories": trajectories,
        "state_table": {
            "trajectory_id": np.asarray(trajectory_ids, dtype=np.int32),
            "history_count": np.asarray(history_counts, dtype=np.int32),
            "teacher_action": np.asarray(teacher_actions, dtype=np.float64),
            "resulting_prefix_loss": np.asarray(resulting_losses, dtype=np.float64),
            "teacher_action_type": np.asarray(candidate_types, dtype=np.int8),
        },
    }


def _sample_indices(table: dict[str, np.ndarray], validation: bool) -> np.ndarray:
    trajectory_ids = table["trajectory_id"]
    indices = np.flatnonzero((trajectory_ids % 5 == 0) == validation)
    cap = VALIDATION_EXAMPLE_CAP if validation else TRAIN_EXAMPLE_CAP
    if len(indices) > cap:
        generator = np.random.default_rng(SAMPLE_SEED + int(validation))
        indices = np.sort(generator.choice(indices, cap, replace=False))
    return indices.astype(np.int64)


def _prepare_examples(
    payload: dict[str, Any], indices: np.ndarray, label: str
) -> dict[str, np.ndarray]:
    trajectories = {item["trajectory_id"]: item for item in payload["trajectories"]}
    table = payload["state_table"]
    count = len(indices)
    history = np.zeros(
        (count, MAX_HISTORY_POINTS, HISTORY_FEATURE_DIM), dtype=np.float16
    )
    history_count = np.zeros(count, dtype=np.uint16)
    global_features = np.empty((count, GLOBAL_FEATURE_DIM), dtype=np.float32)
    candidate_features = np.empty(
        (count, POOL_SIZE, CANDIDATE_FEATURE_DIM), dtype=np.float16
    )
    candidate_points = np.empty((count, POOL_SIZE, 2), dtype=np.float16)
    targets = np.empty((count, POOL_SIZE), dtype=np.float16)
    nearest_indices = np.empty(count, dtype=np.int16)
    nearest_distances = np.empty(count, dtype=np.float32)
    teacher_actions = np.empty((count, 2), dtype=np.float32)
    source_state_indices = np.asarray(indices, dtype=np.int32)
    for row, state_index in enumerate(indices):
        trajectory_id = int(table["trajectory_id"][state_index])
        prefix_count = int(table["history_count"][state_index])
        trajectory = trajectories[trajectory_id]
        points = trajectory["evaluated_hypotheses_normalized"][:prefix_count]
        losses = trajectory["evaluated_prefix_losses"][:prefix_count]
        teacher_action = table["teacher_action"][state_index]
        history_features, state_global = regiongen._state_features(points, losses)
        pool, sources = _generic_pool(points, losses, prefix_count)
        pool_features = _candidate_features(points, losses, pool, sources)
        probabilities, nearest_index, nearest_distance = _soft_target(pool, teacher_action)
        used = len(history_features)
        history[row, :used] = history_features.astype(np.float16)
        history_count[row] = used
        global_features[row] = state_global
        candidate_features[row] = pool_features.astype(np.float16)
        candidate_points[row] = pool.astype(np.float16)
        targets[row] = probabilities.astype(np.float16)
        nearest_indices[row] = nearest_index
        nearest_distances[row] = nearest_distance
        teacher_actions[row] = teacher_action.astype(np.float32)
        if (row + 1) % 1000 == 0 or row + 1 == count:
            print(f"prepared {label} pool examples {row + 1}/{count}", flush=True)
    return {
        "source_state_index": source_state_indices,
        "history_features": history,
        "history_count": history_count,
        "global_features": global_features,
        "candidate_features": candidate_features,
        "candidate_points": candidate_points,
        "soft_targets": targets,
        "nearest_index": nearest_indices,
        "nearest_distance": nearest_distances,
        "teacher_action": teacher_actions,
    }


def _batch(
    examples: dict[str, np.ndarray], indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = examples["history_count"][indices].astype(np.int64)
    maximum = int(np.max(counts))
    history = torch.as_tensor(
        examples["history_features"][indices, :maximum].astype(np.float32)
    )
    mask = torch.arange(maximum)[None, :] < torch.as_tensor(counts)[:, None]
    global_features = torch.as_tensor(examples["global_features"][indices])
    candidate_features = torch.as_tensor(
        examples["candidate_features"][indices].astype(np.float32)
    )
    targets = torch.as_tensor(examples["soft_targets"][indices].astype(np.float32))
    return history, mask, global_features, candidate_features, targets


def _scorer_metrics(
    model: GenericCandidateScorer,
    examples: dict[str, np.ndarray],
) -> dict[str, float]:
    model.eval()
    count = len(examples["history_count"])
    top1_correct = 0
    top5_correct = 0
    selected_distances: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, count, BATCH_SIZE):
            indices = np.arange(start, min(start + BATCH_SIZE, count))
            history, mask, global_features, candidate_features, targets = _batch(
                examples, indices
            )
            scores = model(history, mask, global_features, candidate_features)
            log_probabilities = torch.log_softmax(scores, dim=1)
            losses.extend(
                (-torch.sum(targets * log_probabilities, dim=1)).cpu().numpy().tolist()
            )
            order = torch.argsort(scores, dim=1, descending=True).cpu().numpy()
            nearest = examples["nearest_index"][indices]
            top1_correct += int(np.sum(order[:, 0] == nearest))
            top5_correct += int(
                sum(nearest[row] in order[row, :5] for row in range(len(indices)))
            )
            pools = examples["candidate_points"][indices].astype(np.float64)
            teacher = examples["teacher_action"][indices].astype(np.float64)
            selected = pools[np.arange(len(indices)), order[:, 0]]
            selected_distances.extend(
                np.linalg.norm(selected - teacher, axis=1).tolist()
            )
    # Pool generation and model inputs use [0,1]^2 coordinates, but report
    # action distances in the physical (a,b) domain requested by the experiment.
    selected_array = DOMAIN_WIDTH * np.asarray(selected_distances)
    nearest_array = DOMAIN_WIDTH * examples["nearest_distance"].astype(np.float64)
    return {
        "soft_cross_entropy": float(np.mean(losses)),
        "teacher_nearest_top1_accuracy": top1_correct / count,
        "teacher_nearest_top5_accuracy": top5_correct / count,
        "selected_teacher_distance_median": float(np.median(selected_array)),
        "selected_teacher_distance_worst": float(np.max(selected_array)),
        "selected_within_0.1": float(np.mean(selected_array <= 0.1)),
        "selected_within_0.05": float(np.mean(selected_array <= 0.05)),
        "selected_within_0.01": float(np.mean(selected_array <= 0.01)),
        "pool_nearest_distance_median": float(np.median(nearest_array)),
        "pool_nearest_distance_worst": float(np.max(nearest_array)),
        "pool_within_0.1": float(np.mean(nearest_array <= 0.1)),
        "pool_within_0.05": float(np.mean(nearest_array <= 0.05)),
        "pool_within_0.01": float(np.mean(nearest_array <= 0.01)),
    }


def _train_scorer(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[GenericCandidateScorer, list[dict[str, object]], dict[str, float]]:
    torch.manual_seed(MODEL_SEED)
    model = GenericCandidateScorer().to(dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    rows: list[dict[str, object]] = []
    training_count = len(training["history_count"])
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(training_count, generator=generator).numpy()
        total_loss = 0.0
        for start in range(0, training_count, BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            history, mask, global_features, candidate_features, targets = _batch(
                training, indices
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model(history, mask, global_features, candidate_features)
            loss = -torch.mean(
                torch.sum(targets * torch.log_softmax(scores, dim=1), dim=1)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        metrics = _scorer_metrics(model, validation)
        row = {
            "row_type": "training_epoch",
            "epoch": epoch,
            "training_soft_cross_entropy": total_loss / training_count,
            **metrics,
        }
        rows.append(row)
        print(
            f"scorer epoch={epoch} train_ce={row['training_soft_cross_entropy']:.5f} "
            f"val_ce={metrics['soft_cross_entropy']:.5f} "
            f"top1={metrics['teacher_nearest_top1_accuracy']:.4f} "
            f"selected_med={metrics['selected_teacher_distance_median']:.4f}",
            flush=True,
        )
        if metrics["soft_cross_entropy"] < best_loss - 1.0e-5:
            best_loss = metrics["soft_cross_entropy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "epoch": best_epoch,
                    "validation_metrics": metrics,
                    "config": {
                        "pool_size": POOL_SIZE,
                        "global_count": GLOBAL_COUNT,
                        "local_count": LOCAL_COUNT,
                        "sparse_count": SPARSE_COUNT,
                        "edge_count": EDGE_COUNT,
                        "model_seed": MODEL_SEED,
                        "pool_seed": POOL_SEED,
                        "operator": "K(a,b)=[[0,1],[a,b]]",
                    },
                },
                CHECKPOINT_PATH,
            )
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    metrics = _scorer_metrics(model, validation)
    return model, rows, {"best_epoch": best_epoch, **metrics}


def _score_pool(
    model: GenericCandidateScorer,
    points: np.ndarray,
    losses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_features, global_features = regiongen._state_features(points, losses)
    pool, sources = _generic_pool(points, losses, len(points))
    candidate_features = _candidate_features(points, losses, pool, sources)
    with torch.no_grad():
        scores = model(
            torch.as_tensor(history_features, dtype=torch.float32)[None, :, :],
            torch.ones((1, len(history_features)), dtype=torch.bool),
            torch.as_tensor(global_features, dtype=torch.float32)[None, :],
            torch.as_tensor(candidate_features, dtype=torch.float32)[None, :, :],
        )[0].cpu().numpy()
    return pool, sources, scores


def _run_learned_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    model: GenericCandidateScorer,
) -> dict[str, Any]:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    first = edirect._initial_design()[0:1].astype(np.float64)
    first_loss = edirect._prefix_losses_numpy(
        LOWER + DOMAIN_WIDTH * first, position
    ).astype(np.float64)
    points = first.copy()
    losses = first_loss.copy()
    actions: list[dict[str, Any]] = []
    while len(points) < NORMAL_CAP and float(np.min(losses)) > GATE:
        pool, sources, scores = _score_pool(model, points, losses)
        order = np.argsort(scores)[::-1]
        tree = cKDTree(points)
        choice = int(order[0])
        for candidate_index in order:
            if float(tree.query(pool[candidate_index], k=1)[0]) > 1.0e-10:
                choice = int(candidate_index)
                break
        proposal = pool[choice]
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        actions.append(
            {
                "history_count": len(points),
                "pool_step": len(points),
                "chosen_candidate": proposal.copy(),
                "candidate_source": int(sources[choice]),
                "chosen_score": float(scores[choice]),
                "score_rank": 1,
                "resulting_prefix_loss": loss,
            }
        )
        points = np.concatenate((points, proposal[None, :]), axis=0)
        losses = np.concatenate((losses, np.asarray((loss,))))
    return {
        "split": split,
        "trajectory_id": trajectory_id,
        "generator_id": generator_id,
        "points": points,
        "losses": losses,
        "actions": actions,
        "success": float(np.min(losses)) <= GATE,
    }


_WORKER_MODEL: GenericCandidateScorer | None = None


def _load_worker_model() -> GenericCandidateScorer:
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _WORKER_MODEL = GenericCandidateScorer().to(dtype=torch.float32)
        _WORKER_MODEL.load_state_dict(checkpoint["state_dict"])
        _WORKER_MODEL.eval()
    return _WORKER_MODEL


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    model = _load_worker_model()
    normal = _run_learned_normal(split, trajectory_id, position, model)
    normal_mode = imitation.ModeRun(
        mode="normal_generic_pool",
        bounds=FULL_BOUNDS.copy(),
        points=normal["points"],
        losses=normal["losses"],
        new_evaluations=len(normal["points"]),
        success=normal["success"],
    )
    total = len(normal["points"])
    modes = [normal_mode]
    boundary_run = None
    nested_run = None
    boundary_geometry = None
    if not normal_mode.success:
        boundary_run, boundary_geometry = imitation._run_boundary_rescue(
            split=split,
            trajectory_id=trajectory_id,
            generator_id=normal["generator_id"],
            position=position,
            normal=normal_mode,
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
        "success": any(mode.success for mode in modes),
        "normal_success": normal_mode.success,
        "normal_best_loss": normal_mode.best_loss,
        "normal_best_theta": normal_mode.best_physical,
        "total_evaluations": total,
        "final_loss": final.best_loss,
        "final_theta": final.best_physical,
        "boundary_invoked": boundary_run is not None,
        "boundary_success": bool(boundary_run and boundary_run.success),
        "nested_invoked": nested_run is not None,
        "nested_success": bool(nested_run and nested_run.success),
        "boundary_geometry": boundary_geometry,
        "normal_history": {
            "initial_hypothesis_normalized": normal["points"][0],
            "evaluated_hypotheses_normalized": normal["points"],
            "evaluated_prefix_losses": normal["losses"],
            "actions": normal["actions"],
        },
        "rescue_modes": [
            {
                "mode": mode.mode,
                "bounds": mode.bounds,
                "evaluated_hypotheses_normalized": mode.points,
                "evaluated_prefix_losses": mode.losses,
                "new_evaluations": mode.new_evaluations,
                "success": mode.success,
            }
            for mode in modes[1:]
        ],
    }


def _run_split(split: str, positions: np.ndarray, workers: int) -> list[dict[str, Any]]:
    tasks = [(split, index, positions[index]) for index in range(len(positions))]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_search_worker(task))
            print(f"generic-pool {split} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(
                        f"generic-pool {split} {len(results)}/{len(tasks)}",
                        flush=True,
                    )
    results.sort(key=lambda item: item["trajectory_id"])
    return results


def _teacher_runs(source: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return source[f"{split}_teacher"]["trajectories"]


def _search_summary(
    split: str, method: str, runs: list[dict[str, Any]]
) -> dict[str, object]:
    teacher = method.startswith("teacher")
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
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse the persisted best checkpoint for inference-only replay.",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_TEACHER_TRACE.exists() or not SOURCE_SEARCH_TRACE.exists():
        raise FileNotFoundError("saved eDIRECT teacher artifacts are required")

    if DATASET_PATH.exists() and not args.rebuild_dataset:
        dataset = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
        print(
            f"loaded generic-pool dataset with {len(dataset['state_table']['history_count'])} states",
            flush=True,
        )
    else:
        source = torch.load(SOURCE_TEACHER_TRACE, map_location="cpu", weights_only=False)
        extracted = _teacher_state_table(source)
        train_indices = _sample_indices(extracted["state_table"], False)
        validation_indices = _sample_indices(extracted["state_table"], True)
        training_examples = _prepare_examples(extracted, train_indices, "train")
        validation_examples = _prepare_examples(extracted, validation_indices, "validation")
        dataset = {
            "config": {
                "operator": "K(a,b)=[[0,1],[a,b]]",
                "prefix_length": 128,
                "objective_dtype": "float64",
                "network_dtype": "float32",
                "pool_counts": {
                    "global_sobol": GLOBAL_COUNT,
                    "local_spacing_scaled": LOCAL_COUNT,
                    "sparse_farthest": SPARSE_COUNT,
                    "edge_near_edge": EDGE_COUNT,
                },
                "pool_seed": POOL_SEED,
                "history_reconstruction": "trajectory history prefix [:history_count]",
                "true_parameter_supervision": False,
                "source_teacher_trace_sha256": _sha256(SOURCE_TEACHER_TRACE),
            },
            **extracted,
            "training_examples": training_examples,
            "validation_examples": validation_examples,
        }
        torch.save(dataset, DATASET_PATH)
        del source
        gc.collect()
        print(
            f"persisted generic-pool dataset with {len(dataset['state_table']['history_count'])} states",
            flush=True,
        )

    if args.skip_training:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError("--skip-training requires the saved scorer checkpoint")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = GenericCandidateScorer().to(dtype=torch.float32)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        training_rows: list[dict[str, object]] = []
        best_metrics = {
            "best_epoch": int(checkpoint["epoch"]),
            **checkpoint["validation_metrics"],
        }
    else:
        model, training_rows, best_metrics = _train_scorer(
            dataset["training_examples"], dataset["validation_examples"]
        )
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
                "pool_reconstruction": "deterministic from history and pool_step",
                "scorer_checkpoint_sha256": _sha256(CHECKPOINT_PATH),
            },
            "validation_learned": validation_runs,
            "test_learned": test_runs,
            "teacher_reference": {
                "source": str(SOURCE_SEARCH_TRACE),
                "validation": _teacher_runs(source_search, "validation"),
                "test": _teacher_runs(source_search, "test"),
            },
        },
        SEARCH_TRACE_PATH,
    )

    rows: list[dict[str, object]] = list(training_rows)
    rows.append(
        {
            "row_type": "scorer_best",
            **best_metrics,
            "training_examples": len(dataset["training_examples"]["history_count"]),
            "validation_examples": len(dataset["validation_examples"]["history_count"]),
            "teacher_states": len(dataset["state_table"]["history_count"]),
            "runtime_seconds_this_execution": time.perf_counter() - started,
        }
    )
    rows.extend(
        (
            _search_summary(
                "validation", "teacher_edirect", _teacher_runs(source_search, "validation")
            ),
            _search_summary(
                "validation", "generic_pool_learned_scorer", validation_runs
            ),
            _search_summary(
                "test", "teacher_edirect", _teacher_runs(source_search, "test")
            ),
            _search_summary("test", "generic_pool_learned_scorer", test_runs),
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
                        "method": "generic_pool_learned_scorer",
                        "trajectory_id": run["trajectory_id"],
                        "normal_best_loss": run["normal_best_loss"],
                        "final_loss": run["final_loss"],
                        "normal_best_a": run["normal_best_theta"][0],
                        "normal_best_b": run["normal_best_theta"][1],
                        "selected_rescue_face": geometry.get("face", ""),
                        "rescue_strip_delta": geometry.get("delta", ""),
                    }
                )
    _write_csv(SUMMARY_PATH, rows)
    print(f"wrote generic-pool scorer artifacts to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
