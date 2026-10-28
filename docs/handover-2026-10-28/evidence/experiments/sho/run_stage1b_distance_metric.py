#!/usr/bin/env python3
"""Learn only the Euclidean distance primitive used by eDIRECT geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


OUTPUT_DIR = Path("outputs/sho/stage1B/distance_metric")
SMOKE_PATH = OUTPUT_DIR / "distance_metric_smoke.csv"
FULL_PATH = OUTPUT_DIR / "distance_metric_full_pool.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "geometric_distance_metric_frozen_candidate.pt"

SEED = 94631
SMOKE_TRAIN_STATE_COUNT = 180
SMOKE_VALIDATION_STATE_COUNT = 60
TRAIN_PAIRS_PER_TYPE_PER_STATE = 14
VALIDATION_PAIRS_PER_TYPE_PER_STATE = 9
FULL_RANDOM_PAIRS_PER_TYPE_PER_STATE = 1024
MAX_EPOCHS = 5000
PLATEAU_PATIENCE = 800
SMOKE_RMSE_GATE = 1.0e-15
SMOKE_MAX_ERROR_GATE = 1.0e-14
FULL_RMSE_GATE = 2.0e-15
FULL_MAX_ERROR_GATE = 2.0e-14


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class QuadraticMetric(torch.nn.Module):
    """PSD quadratic metric W = L L^T, evaluated as ||delta L||_2."""

    def __init__(self, generator: torch.Generator) -> None:
        super().__init__()
        initial = 0.4 * torch.randn(
            (2, 2), dtype=torch.float64, generator=generator
        )
        self.L = torch.nn.Parameter(initial)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        metric = self.metric_matrix()
        squared = torch.einsum("ni,ij,nj->n", delta, metric, delta)
        return torch.sqrt(squared)

    def metric_matrix(self) -> torch.Tensor:
        return self.L @ self.L.T


@dataclass
class PairPool:
    delta: np.ndarray
    distance: np.ndarray
    pair_type: np.ndarray
    zero_probe_site_count: int
    zero_site_site_count: int


def _state_geometry(
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    source_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    record = payload["states"][source_index]
    trajectory = trajectories[int(record["trajectory_id"])]
    count = int(record["history_count"])
    sites = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:count], dtype=np.float64
    )
    probes = edirect._candidate_pool(
        count, int(record["generator_id"]), int(record["iteration"])
    )
    return sites, np.asarray(probes, dtype=np.float64), record


def _sample_distinct_pairs(
    count: int, sample_count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    left = rng.integers(0, count, size=sample_count)
    offset = rng.integers(1, count, size=sample_count)
    right = (left + offset) % count
    return left, right


def _build_smoke_pool(
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
    pairs_per_type: int,
    seed: int,
) -> PairPool:
    deltas: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    pair_types: list[np.ndarray] = []
    zero_probe_site = 0
    zero_site_site = 0
    for source_index in state_indices:
        sites, probes, _ = _state_geometry(
            payload, trajectories, int(source_index)
        )
        rng = np.random.default_rng(seed + 1000003 * int(source_index))
        tree = cKDTree(sites)

        # Mix production-nearest and unrestricted real probe/site pairs.
        nearest_count = pairs_per_type // 2
        random_count = pairs_per_type - nearest_count
        probe_indices = rng.integers(0, len(probes), size=nearest_count)
        _, nearest_owners = tree.query(probes[probe_indices], k=1)
        nearest_delta = probes[probe_indices] - sites[nearest_owners]
        random_probe = rng.integers(0, len(probes), size=random_count)
        random_site = rng.integers(0, len(sites), size=random_count)
        probe_delta = np.concatenate(
            [nearest_delta, probes[random_probe] - sites[random_site]], axis=0
        )
        probe_distance = np.linalg.norm(probe_delta, axis=1)
        probe_nonzero = probe_distance > 0.0
        zero_probe_site += int(np.sum(~probe_nonzero))
        deltas.append(probe_delta[probe_nonzero])
        labels.append(probe_distance[probe_nonzero])
        pair_types.append(
            np.full(np.sum(probe_nonzero), "probe_to_site", dtype="U16")
        )

        # Mix nearest-other and unrestricted distinct site/site pairs.
        nearest_site = tree.query(sites, k=2)[1][:, 1]
        site_left = rng.integers(0, len(sites), size=nearest_count)
        nearest_site_delta = sites[site_left] - sites[nearest_site[site_left]]
        left, right = _sample_distinct_pairs(len(sites), random_count, rng)
        site_delta = np.concatenate(
            [nearest_site_delta, sites[left] - sites[right]], axis=0
        )
        site_distance = np.linalg.norm(site_delta, axis=1)
        site_nonzero = site_distance > 0.0
        zero_site_site += int(np.sum(~site_nonzero))
        deltas.append(site_delta[site_nonzero])
        labels.append(site_distance[site_nonzero])
        pair_types.append(
            np.full(np.sum(site_nonzero), "site_to_site", dtype="U16")
        )

    return PairPool(
        delta=np.concatenate(deltas, axis=0),
        distance=np.concatenate(labels, axis=0),
        pair_type=np.concatenate(pair_types, axis=0),
        zero_probe_site_count=zero_probe_site,
        zero_site_site_count=zero_site_site,
    )


def _predict(model: QuadraticMetric, delta: np.ndarray) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(delta), 262144):
            tensor = torch.from_numpy(delta[start : start + 262144])
            predictions.append(model(tensor).cpu().numpy())
    if not predictions:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(predictions)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    if len(target) > 1 and np.std(target) > 0.0 and np.std(prediction) > 0.0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = math.nan
    return {
        "count": len(target),
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
        "mae": float(np.mean(absolute)) if len(error) else math.nan,
        "maximum_absolute_error": float(np.max(absolute)) if len(error) else math.nan,
        "correlation": correlation,
        "p50_absolute_error": float(np.percentile(absolute, 50)) if len(error) else math.nan,
        "p90_absolute_error": float(np.percentile(absolute, 90)) if len(error) else math.nan,
        "p95_absolute_error": float(np.percentile(absolute, 95)) if len(error) else math.nan,
        "p99_absolute_error": float(np.percentile(absolute, 99)) if len(error) else math.nan,
    }


def _metric_rows(
    pool: PairPool, prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_type in ("all", "probe_to_site", "site_to_site"):
        mask = (
            np.ones(len(pool.distance), dtype=bool)
            if pair_type == "all"
            else pool.pair_type == pair_type
        )
        rows.append(
            {
                "record_type": "distance_metrics",
                "split": split,
                "pair_type": pair_type,
                **_metrics(pool.distance[mask], prediction[mask]),
            }
        )

    quartiles = np.quantile(pool.distance, [0.0, 0.25, 0.5, 0.75, 1.0])
    for index in range(4):
        if index < 3:
            mask = (pool.distance >= quartiles[index]) & (
                pool.distance < quartiles[index + 1]
            )
        else:
            mask = (pool.distance >= quartiles[index]) & (
                pool.distance <= quartiles[index + 1]
            )
        rows.append(
            {
                "record_type": "distance_quartile_metrics",
                "split": split,
                "distance_quartile": index + 1,
                "distance_lower": float(quartiles[index]),
                "distance_upper": float(quartiles[index + 1]),
                **_metrics(pool.distance[mask], prediction[mask]),
            }
        )
    return rows


def _ordering_diagnostics(
    model: QuadraticMetric,
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
    probe_limit_per_state: int | None,
    seed: int,
) -> dict[str, Any]:
    probe_unique_nonzero = 0
    probe_exact_ties = 0
    probe_zero_nearest = 0
    probe_owner_changes = 0
    spacing_unique_nonzero = 0
    spacing_exact_ties = 0
    spacing_zero_nearest = 0
    spacing_winner_changes = 0
    for source_index in state_indices:
        sites, probes, _ = _state_geometry(
            payload, trajectories, int(source_index)
        )
        if probe_limit_per_state is not None and len(probes) > probe_limit_per_state:
            rng = np.random.default_rng(seed + 104729 * int(source_index))
            chosen = rng.choice(len(probes), size=probe_limit_per_state, replace=False)
            probes = probes[chosen]

        teacher_tree = cKDTree(sites)
        teacher_probe_distances, teacher_probe_owners = teacher_tree.query(probes, k=2)
        tied = teacher_probe_distances[:, 0] == teacher_probe_distances[:, 1]
        zero = teacher_probe_distances[:, 0] == 0.0
        comparable = ~tied & ~zero
        probe_exact_ties += int(np.sum(tied))
        probe_zero_nearest += int(np.sum(zero))
        probe_unique_nonzero += int(np.sum(comparable))
        first_probe_delta = probes - sites[teacher_probe_owners[:, 0]]
        second_probe_delta = probes - sites[teacher_probe_owners[:, 1]]
        learned_probe_distances = _predict(
            model,
            np.concatenate([first_probe_delta, second_probe_delta], axis=0),
        )
        learned_first_probe = learned_probe_distances[: len(probes)]
        learned_second_probe = learned_probe_distances[len(probes) :]
        probe_owner_changes += int(
            np.sum(
                learned_first_probe[comparable]
                >= learned_second_probe[comparable]
            )
        )

        teacher_spacing_distances, teacher_spacing_owners = teacher_tree.query(
            sites, k=3
        )
        spacing_tied = (
            teacher_spacing_distances[:, 1] == teacher_spacing_distances[:, 2]
        )
        spacing_zero = teacher_spacing_distances[:, 1] == 0.0
        spacing_comparable = ~spacing_tied & ~spacing_zero
        spacing_exact_ties += int(np.sum(spacing_tied))
        spacing_zero_nearest += int(np.sum(spacing_zero))
        spacing_unique_nonzero += int(np.sum(spacing_comparable))
        first_spacing_delta = sites - sites[teacher_spacing_owners[:, 1]]
        second_spacing_delta = sites - sites[teacher_spacing_owners[:, 2]]
        learned_spacing_distances = _predict(
            model,
            np.concatenate([first_spacing_delta, second_spacing_delta], axis=0),
        )
        learned_first_spacing = learned_spacing_distances[: len(sites)]
        learned_second_spacing = learned_spacing_distances[len(sites) :]
        spacing_winner_changes += int(
            np.sum(
                learned_first_spacing[spacing_comparable]
                >= learned_second_spacing[spacing_comparable]
            )
        )
    return {
        "probe_unique_nonzero_ordering_count": probe_unique_nonzero,
        "probe_exact_tie_deterministic_count": probe_exact_ties,
        "probe_zero_distance_excluded_count": probe_zero_nearest,
        "unique_nearest_site_ordering_change_count": probe_owner_changes,
        "unique_nearest_site_ordering_agreement": (
            1.0 - probe_owner_changes / probe_unique_nonzero
            if probe_unique_nonzero
            else math.nan
        ),
        "site_spacing_unique_nonzero_ordering_count": spacing_unique_nonzero,
        "site_spacing_exact_tie_deterministic_count": spacing_exact_ties,
        "site_spacing_zero_distance_excluded_count": spacing_zero_nearest,
        "nearest_site_spacing_winner_change_count": spacing_winner_changes,
        "nearest_site_spacing_winner_agreement": (
            1.0 - spacing_winner_changes / spacing_unique_nonzero
            if spacing_unique_nonzero
            else math.nan
        ),
    }


def _train(
    training: PairPool, validation: PairPool
) -> tuple[QuadraticMetric, dict[str, Any]]:
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    model = QuadraticMetric(generator)
    initial_L = model.L.detach().clone()
    initial_W = model.metric_matrix().detach().clone()
    train_delta = torch.from_numpy(training.delta)
    train_target = torch.from_numpy(training.distance)
    validation_delta = torch.from_numpy(validation.delta)
    validation_target = torch.from_numpy(validation.distance)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-2)

    best_epoch = 0
    best_rmse = math.inf
    best_max_error = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    stop_reason = "maximum_epochs"
    stop_epoch = MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(train_delta)
        loss = torch.mean(torch.square(prediction - train_target))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_prediction = model(validation_delta)
            validation_error = validation_prediction - validation_target
            validation_rmse = float(
                torch.sqrt(torch.mean(torch.square(validation_error))).item()
            )
            validation_max_error = float(torch.max(torch.abs(validation_error)).item())
        if validation_rmse < best_rmse:
            best_epoch = epoch
            best_rmse = validation_rmse
            best_max_error = validation_max_error
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if validation_rmse <= 1.0e-16 and validation_max_error <= 5.0e-16:
            stop_reason = "numerical_precision_gate"
            stop_epoch = epoch
            break
        if epochs_without_improvement >= PLATEAU_PATIENCE:
            stop_reason = "validation_plateau"
            stop_epoch = epoch
            break
        if epoch == 1 or epoch % 250 == 0:
            print(
                f"distance metric epoch={epoch} val_rmse={validation_rmse:.3e} "
                f"val_max={validation_max_error:.3e}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("distance metric training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "initial_L": initial_L.numpy(),
        "initial_W": initial_W.numpy(),
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "best_validation_rmse": best_rmse,
        "best_validation_max_error": best_max_error,
    }


def _full_pool(
    model: QuadraticMetric,
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
) -> PairPool:
    deltas: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    pair_types: list[np.ndarray] = []
    zero_probe_site = 0
    zero_site_site = 0
    for position, source_index in enumerate(state_indices, start=1):
        sites, probes, _ = _state_geometry(
            payload, trajectories, int(source_index)
        )
        rng = np.random.default_rng(SEED + 7000003 * int(source_index))
        tree = cKDTree(sites)

        nearest_distance, nearest_owner = tree.query(probes, k=1)
        nearest_delta = probes - sites[nearest_owner]
        nearest_nonzero = nearest_distance > 0.0
        zero_probe_site += int(np.sum(~nearest_nonzero))
        deltas.append(nearest_delta[nearest_nonzero])
        distances.append(nearest_distance[nearest_nonzero])
        pair_types.append(
            np.full(np.sum(nearest_nonzero), "probe_to_site", dtype="U16")
        )

        random_count = min(FULL_RANDOM_PAIRS_PER_TYPE_PER_STATE, len(probes))
        random_probe = rng.choice(len(probes), size=random_count, replace=False)
        random_site = rng.integers(0, len(sites), size=random_count)
        random_probe_delta = probes[random_probe] - sites[random_site]
        random_probe_distance = np.linalg.norm(random_probe_delta, axis=1)
        random_probe_nonzero = random_probe_distance > 0.0
        zero_probe_site += int(np.sum(~random_probe_nonzero))
        deltas.append(random_probe_delta[random_probe_nonzero])
        distances.append(random_probe_distance[random_probe_nonzero])
        pair_types.append(
            np.full(np.sum(random_probe_nonzero), "probe_to_site", dtype="U16")
        )

        neighbor_distance, neighbor_owner = tree.query(sites, k=2)
        spacing_delta = sites - sites[neighbor_owner[:, 1]]
        spacing = neighbor_distance[:, 1]
        spacing_nonzero = spacing > 0.0
        zero_site_site += int(np.sum(~spacing_nonzero))
        deltas.append(spacing_delta[spacing_nonzero])
        distances.append(spacing[spacing_nonzero])
        pair_types.append(
            np.full(np.sum(spacing_nonzero), "site_to_site", dtype="U16")
        )

        random_count = min(FULL_RANDOM_PAIRS_PER_TYPE_PER_STATE, len(sites))
        left, right = _sample_distinct_pairs(len(sites), random_count, rng)
        random_site_delta = sites[left] - sites[right]
        random_site_distance = np.linalg.norm(random_site_delta, axis=1)
        random_site_nonzero = random_site_distance > 0.0
        zero_site_site += int(np.sum(~random_site_nonzero))
        deltas.append(random_site_delta[random_site_nonzero])
        distances.append(random_site_distance[random_site_nonzero])
        pair_types.append(
            np.full(np.sum(random_site_nonzero), "site_to_site", dtype="U16")
        )
        if position % 20 == 0 or position == len(state_indices):
            print(f"full distance pool states {position}/{len(state_indices)}", flush=True)
    return PairPool(
        delta=np.concatenate(deltas, axis=0),
        distance=np.concatenate(distances, axis=0),
        pair_type=np.concatenate(pair_types, axis=0),
        zero_probe_site_count=zero_probe_site,
        zero_site_site_count=zero_site_site,
    )


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    source_train = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    source_held_out = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    split_rng = np.random.default_rng(SEED)
    shuffled = np.asarray(source_train, dtype=np.int64).copy()
    split_rng.shuffle(shuffled)
    smoke_train_states = shuffled[:SMOKE_TRAIN_STATE_COUNT].tolist()
    smoke_validation_states = shuffled[
        SMOKE_TRAIN_STATE_COUNT : SMOKE_TRAIN_STATE_COUNT
        + SMOKE_VALIDATION_STATE_COUNT
    ].tolist()
    if set(smoke_train_states) & set(smoke_validation_states):
        raise RuntimeError("smoke train/validation states overlap")
    if (set(smoke_train_states) | set(smoke_validation_states)) & set(source_held_out):
        raise RuntimeError("smoke states overlap the full held-out pool")

    training = _build_smoke_pool(
        payload,
        trajectories,
        smoke_train_states,
        TRAIN_PAIRS_PER_TYPE_PER_STATE,
        SEED + 1,
    )
    validation = _build_smoke_pool(
        payload,
        trajectories,
        smoke_validation_states,
        VALIDATION_PAIRS_PER_TYPE_PER_STATE,
        SEED + 2,
    )
    if np.any(training.distance <= 0.0) or np.any(validation.distance <= 0.0):
        raise RuntimeError("zero distance entered the training or validation tensors")
    model, training_info = _train(training, validation)
    train_prediction = _predict(model, training.delta)
    validation_prediction = _predict(model, validation.delta)
    validation_ordering = _ordering_diagnostics(
        model,
        payload,
        trajectories,
        smoke_validation_states,
        probe_limit_per_state=4096,
        seed=SEED + 3,
    )
    L = model.L.detach().cpu().numpy()
    W = model.metric_matrix().detach().cpu().numpy()
    smoke_metrics = _metrics(validation.distance, validation_prediction)
    smoke_passed = bool(
        smoke_metrics["rmse"] <= SMOKE_RMSE_GATE
        and smoke_metrics["maximum_absolute_error"] <= SMOKE_MAX_ERROR_GATE
    )
    smoke_summary = {
        "record_type": "summary",
        "status": "passed" if smoke_passed else "failed",
        "training_state_count": len(smoke_train_states),
        "validation_state_count": len(smoke_validation_states),
        "training_pair_count": len(training.distance),
        "validation_pair_count": len(validation.distance),
        "training_probe_site_zero_distance_exclusion_count": training.zero_probe_site_count,
        "training_site_site_zero_distance_exclusion_count": training.zero_site_site_count,
        "validation_probe_site_zero_distance_exclusion_count": validation.zero_probe_site_count,
        "validation_site_site_zero_distance_exclusion_count": validation.zero_site_site_count,
        "initial_L_00": training_info["initial_L"][0, 0],
        "initial_L_01": training_info["initial_L"][0, 1],
        "initial_L_10": training_info["initial_L"][1, 0],
        "initial_L_11": training_info["initial_L"][1, 1],
        "initial_W_00": training_info["initial_W"][0, 0],
        "initial_W_01": training_info["initial_W"][0, 1],
        "initial_W_10": training_info["initial_W"][1, 0],
        "initial_W_11": training_info["initial_W"][1, 1],
        "learned_L_00": L[0, 0],
        "learned_L_01": L[0, 1],
        "learned_L_10": L[1, 0],
        "learned_L_11": L[1, 1],
        "learned_W_00": W[0, 0],
        "learned_W_01": W[0, 1],
        "learned_W_10": W[1, 0],
        "learned_W_11": W[1, 1],
        "best_epoch": training_info["best_epoch"],
        "stop_epoch": training_info["stop_epoch"],
        "stop_reason": training_info["stop_reason"],
        **{f"validation_{key}": value for key, value in smoke_metrics.items()},
        **validation_ordering,
    }
    smoke_rows: list[dict[str, Any]] = [smoke_summary]
    smoke_rows.extend(_metric_rows(training, train_prediction, "smoke_training"))
    smoke_rows.extend(
        _metric_rows(validation, validation_prediction, "smoke_validation")
    )
    smoke_rows.append(
        {
            "record_type": "ordering_diagnostics",
            "split": "smoke_validation",
            **validation_ordering,
        }
    )
    _write_csv(SMOKE_PATH, smoke_rows)
    print(
        f"smoke status={smoke_summary['status']} rmse={smoke_metrics['rmse']:.3e} "
        f"max={smoke_metrics['maximum_absolute_error']:.3e}",
        flush=True,
    )
    if not smoke_passed:
        return

    full_pool = _full_pool(model, payload, trajectories, source_held_out)
    if np.any(full_pool.distance <= 0.0):
        raise RuntimeError("zero distance entered the full held-out tensor")
    full_prediction = _predict(model, full_pool.delta)
    full_metrics = _metrics(full_pool.distance, full_prediction)
    full_ordering = _ordering_diagnostics(
        model,
        payload,
        trajectories,
        source_held_out,
        probe_limit_per_state=None,
        seed=SEED + 4,
    )
    full_passed = bool(
        full_metrics["rmse"] <= FULL_RMSE_GATE
        and full_metrics["maximum_absolute_error"] <= FULL_MAX_ERROR_GATE
        and full_ordering["unique_nearest_site_ordering_change_count"] == 0
        and full_ordering["nearest_site_spacing_winner_change_count"] == 0
    )
    checkpoint_hash = ""
    if full_passed:
        torch.save(
            {
                "component": "geometric_distance_metric",
                "status": "FROZEN_CANDIDATE",
                "model_definition": "d_hat=sqrt(delta^T W delta), W=L L^T",
                "dtype": "float64",
                "coordinate_domain": "normalized_[0,1]^2",
                "state_dict": model.state_dict(),
                "L": torch.from_numpy(L.copy()),
                "W": torch.from_numpy(W.copy()),
                "seed": SEED,
                "smoke_training_state_indices": smoke_train_states,
                "smoke_validation_state_indices": smoke_validation_states,
                "full_held_out_state_indices": source_held_out,
                "zero_distance_policy": "excluded_without_epsilon",
                "exact_nearest_tie_policy": "deterministic_ckdtree_k1_bookkeeping",
                "smoke_summary": smoke_summary,
                "full_metrics": full_metrics,
                "full_ordering": full_ordering,
            },
            CHECKPOINT_PATH,
        )
        checkpoint_hash = _sha256(CHECKPOINT_PATH)
    full_summary = {
        "record_type": "summary",
        "status": "FROZEN_CANDIDATE" if full_passed else "failed",
        "held_out_state_count": len(source_held_out),
        "full_pair_count": len(full_pool.distance),
        "probe_site_pair_count": int(np.sum(full_pool.pair_type == "probe_to_site")),
        "site_site_pair_count": int(np.sum(full_pool.pair_type == "site_to_site")),
        "probe_site_zero_distance_exclusion_count": full_pool.zero_probe_site_count,
        "site_site_zero_distance_exclusion_count": full_pool.zero_site_site_count,
        "learned_L_00": L[0, 0],
        "learned_L_01": L[0, 1],
        "learned_L_10": L[1, 0],
        "learned_L_11": L[1, 1],
        "learned_W_00": W[0, 0],
        "learned_W_01": W[0, 1],
        "learned_W_10": W[1, 0],
        "learned_W_11": W[1, 1],
        **full_metrics,
        **full_ordering,
        "checkpoint_path": str(CHECKPOINT_PATH) if full_passed else "",
        "checkpoint_sha256": checkpoint_hash,
    }
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(_metric_rows(full_pool, full_prediction, "full_held_out"))
    full_rows.append(
        {
            "record_type": "ordering_diagnostics",
            "split": "full_held_out",
            **full_ordering,
        }
    )
    _write_csv(FULL_PATH, full_rows)
    print(
        f"full status={full_summary['status']} pairs={len(full_pool.distance)} "
        f"rmse={full_metrics['rmse']:.3e} max={full_metrics['maximum_absolute_error']:.3e} "
        f"owner_changes={full_ordering['unique_nearest_site_ordering_change_count']} "
        f"spacing_changes={full_ordering['nearest_site_spacing_winner_change_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
