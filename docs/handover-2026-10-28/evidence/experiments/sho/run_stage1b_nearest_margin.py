#!/usr/bin/env python3
"""Learn only the normalized continuous nearest-distance margin."""

from __future__ import annotations

import copy
import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_distance_metric as distance_metric
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/nearest_distance_margin")
SMOKE_PATH = OUTPUT_DIR / "nearest_margin_smoke.csv"
FULL_PATH = OUTPUT_DIR / "nearest_margin_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "nearest_distance_margin_frozen_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

LOCKED_DISTANCE_PATH = Path(
    "outputs/sho/stage1B/distance_metric_assembled_validation/"
    "geometric_distance_metric_locked.pt"
)
LOCKED_DISTANCE_SHA256 = (
    "29d1e5947a45f6fb79e2d8e5baf7ca4b75406e3e4637d69fe58558e2ebef0c1c"
)

MODEL_SEED = 1_732_051
DATA_SEED = 2_236_067
SMOKE_TRAIN_STATES = 48
SMOKE_VALIDATION_STATES = 16
TRAIN_ORDINARY_PER_TYPE_PER_STATE = 128
VALIDATION_ORDINARY_PER_TYPE_PER_STATE = 256
FULL_ORDINARY_PER_TYPE_PER_STATE = 1024
MAX_ADAM_EPOCHS = 30_000
CHECK_INTERVAL = 100
PATIENCE_CHECKS = 100
LEARNING_RATE = 2.0e-2

CATEGORY_NAMES = {
    0: "ordinary_probe",
    1: "ordinary_spacing",
    2: "nearest_runner_up_probe",
    3: "nearest_runner_up_spacing",
}


@dataclass
class PairPool:
    inputs: np.ndarray
    target: np.ndarray
    exact_margin: np.ndarray
    exact_d1: np.ndarray
    exact_d2: np.ndarray
    locked_d1: np.ndarray
    locked_d2: np.ndarray
    category: np.ndarray
    source_state_index: np.ndarray
    query_index: np.ndarray
    site1: np.ndarray
    site2: np.ndarray
    one_ulp: np.ndarray
    exact_tie_exclusion_count: int
    locked_zero_margin_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _load_locked_distance() -> tuple[distance_metric.QuadraticMetric, dict[str, Any]]:
    actual_hash = _sha256(LOCKED_DISTANCE_PATH)
    if actual_hash != LOCKED_DISTANCE_SHA256:
        raise RuntimeError(
            f"locked distance checkpoint hash mismatch: {actual_hash}"
        )
    checkpoint = torch.load(
        LOCKED_DISTANCE_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("status") != "LOCKED":
        raise RuntimeError("distance checkpoint is not LOCKED")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    model = distance_metric.QuadraticMetric(generator)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _ulp_separation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_bits = np.ascontiguousarray(first, dtype=np.float64).view(np.uint64)
    second_bits = np.ascontiguousarray(second, dtype=np.float64).view(np.uint64)
    return np.where(
        first_bits >= second_bits,
        first_bits - second_bits,
        second_bits - first_bits,
    ).astype(np.uint64)


def _ordered_by_site(
    first_site: np.ndarray,
    second_site: np.ndarray,
    first_distance: np.ndarray,
    second_distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_is_lower = first_site < second_site
    site1 = np.where(first_is_lower, first_site, second_site).astype(np.int64)
    site2 = np.where(first_is_lower, second_site, first_site).astype(np.int64)
    distance1 = np.where(first_is_lower, first_distance, second_distance)
    distance2 = np.where(first_is_lower, second_distance, first_distance)
    return site1, site2, distance1, distance2


def _ordinary_probe_pairs(
    sites: np.ndarray,
    probes: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = rng.integers(0, len(probes), size=count, dtype=np.int64)
    site1 = rng.integers(0, len(sites), size=count, dtype=np.int64)
    offset = rng.integers(1, len(sites), size=count, dtype=np.int64)
    site2 = (site1 + offset) % len(sites)
    return query, site1, site2


def _ordinary_spacing_pairs(
    site_count: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = rng.integers(0, site_count, size=count, dtype=np.int64)
    offset1 = rng.integers(1, site_count, size=count, dtype=np.int64)
    offset2 = rng.integers(1, site_count, size=count, dtype=np.int64)
    same = offset1 == offset2
    while np.any(same):
        offset2[same] = rng.integers(
            1, site_count, size=int(np.sum(same)), dtype=np.int64
        )
        same = offset1 == offset2
    site1 = (query + offset1) % site_count
    site2 = (query + offset2) % site_count
    return query, site1, site2


def _append_pair_block(
    parts: dict[str, list[np.ndarray]],
    model: distance_metric.QuadraticMetric,
    *,
    source_state_index: int,
    query_indices: np.ndarray,
    site1: np.ndarray,
    site2: np.ndarray,
    delta1: np.ndarray,
    delta2: np.ndarray,
    exact_d1: np.ndarray,
    exact_d2: np.ndarray,
    category: int,
) -> tuple[int, int]:
    ties = exact_d1 == exact_d2
    keep = ~ties
    tie_count = int(np.sum(ties))
    if not np.any(keep):
        return tie_count, 0
    query_indices = query_indices[keep]
    site1 = site1[keep]
    site2 = site2[keep]
    delta1 = delta1[keep]
    delta2 = delta2[keep]
    exact_d1 = exact_d1[keep]
    exact_d2 = exact_d2[keep]

    locked_d1 = distance_metric._predict(model, delta1)
    locked_d2 = distance_metric._predict(model, delta2)
    scale = np.maximum(locked_d1, locked_d2)
    if np.any(scale <= 0.0):
        raise RuntimeError("non-tied pair has nonpositive normalization scale")
    x1 = locked_d1 / scale
    x2 = locked_d2 / scale
    target = x1 - x2
    exact_scale = np.maximum(exact_d1, exact_d2)
    if np.any(exact_scale <= 0.0):
        raise RuntimeError("non-tied teacher pair has nonpositive scale")
    exact_margin = exact_d1 / exact_scale - exact_d2 / exact_scale
    locked_zero = target == 0.0

    parts["inputs"].append(np.column_stack((x1, x2)))
    parts["target"].append(target)
    parts["exact_margin"].append(exact_margin)
    parts["exact_d1"].append(exact_d1)
    parts["exact_d2"].append(exact_d2)
    parts["locked_d1"].append(locked_d1)
    parts["locked_d2"].append(locked_d2)
    parts["category"].append(np.full(len(target), category, dtype=np.int8))
    parts["source_state_index"].append(
        np.full(len(target), source_state_index, dtype=np.int64)
    )
    parts["query_index"].append(query_indices.astype(np.int64))
    parts["site1"].append(site1.astype(np.int64))
    parts["site2"].append(site2.astype(np.int64))
    parts["one_ulp"].append(_ulp_separation(exact_d1, exact_d2) == 1)
    return tie_count, int(np.sum(locked_zero))


def _build_pool(
    model: distance_metric.QuadraticMetric,
    payload: dict[str, Any],
    trajectories: dict[int, dict[str, Any]],
    state_indices: list[int],
    ordinary_per_type_per_state: int,
    seed: int,
    label: str,
) -> PairPool:
    names = (
        "inputs",
        "target",
        "exact_margin",
        "exact_d1",
        "exact_d2",
        "locked_d1",
        "locked_d2",
        "category",
        "source_state_index",
        "query_index",
        "site1",
        "site2",
        "one_ulp",
    )
    parts: dict[str, list[np.ndarray]] = {name: [] for name in names}
    tie_count = 0
    locked_zero_count = 0
    for position, source_state_index in enumerate(state_indices, start=1):
        source_state_index = int(source_state_index)
        sites, probes, _ = distance_metric._state_geometry(
            payload, trajectories, source_state_index
        )
        tree = cKDTree(sites)
        rng = np.random.default_rng(seed + 1_000_003 * source_state_index)

        # Exhaustive decisive probe nearest-versus-runner-up pairs.
        distances, candidates = tree.query(probes, k=2)
        site1, site2, exact1, exact2 = _ordered_by_site(
            candidates[:, 0], candidates[:, 1], distances[:, 0], distances[:, 1]
        )
        query = np.arange(len(probes), dtype=np.int64)
        ties, zeros = _append_pair_block(
            parts,
            model,
            source_state_index=source_state_index,
            query_indices=query,
            site1=site1,
            site2=site2,
            delta1=probes - sites[site1],
            delta2=probes - sites[site2],
            exact_d1=exact1,
            exact_d2=exact2,
            category=2,
        )
        tie_count += ties
        locked_zero_count += zeros

        # Exhaustive decisive site nearest-versus-runner-up pairs.
        distances, candidates = tree.query(sites, k=3)
        site1, site2, exact1, exact2 = _ordered_by_site(
            candidates[:, 1], candidates[:, 2], distances[:, 1], distances[:, 2]
        )
        query = np.arange(len(sites), dtype=np.int64)
        ties, zeros = _append_pair_block(
            parts,
            model,
            source_state_index=source_state_index,
            query_indices=query,
            site1=site1,
            site2=site2,
            delta1=sites[query] - sites[site1],
            delta2=sites[query] - sites[site2],
            exact_d1=exact1,
            exact_d2=exact2,
            category=3,
        )
        tie_count += ties
        locked_zero_count += zeros

        # Ordinary real probe/candidate-site pairs from the same state geometry.
        query, site1, site2 = _ordinary_probe_pairs(
            sites, probes, ordinary_per_type_per_state, rng
        )
        delta1 = probes[query] - sites[site1]
        delta2 = probes[query] - sites[site2]
        exact1 = np.linalg.norm(delta1, axis=1)
        exact2 = np.linalg.norm(delta2, axis=1)
        ties, zeros = _append_pair_block(
            parts,
            model,
            source_state_index=source_state_index,
            query_indices=query,
            site1=site1,
            site2=site2,
            delta1=delta1,
            delta2=delta2,
            exact_d1=exact1,
            exact_d2=exact2,
            category=0,
        )
        tie_count += ties
        locked_zero_count += zeros

        # Ordinary real site/candidate-site pairs from the same state geometry.
        query, site1, site2 = _ordinary_spacing_pairs(
            len(sites), ordinary_per_type_per_state, rng
        )
        delta1 = sites[query] - sites[site1]
        delta2 = sites[query] - sites[site2]
        exact1 = np.linalg.norm(delta1, axis=1)
        exact2 = np.linalg.norm(delta2, axis=1)
        ties, zeros = _append_pair_block(
            parts,
            model,
            source_state_index=source_state_index,
            query_indices=query,
            site1=site1,
            site2=site2,
            delta1=delta1,
            delta2=delta2,
            exact_d1=exact1,
            exact_d2=exact2,
            category=1,
        )
        tie_count += ties
        locked_zero_count += zeros

        if position % 10 == 0 or position == len(state_indices):
            print(f"{label} states {position}/{len(state_indices)}", flush=True)

    return PairPool(
        **{name: np.concatenate(parts[name], axis=0) for name in names},
        exact_tie_exclusion_count=tie_count,
        locked_zero_margin_count=locked_zero_count,
    )


def _predict(model: nn.Linear, inputs: np.ndarray) -> np.ndarray:
    weight = model.weight.detach().cpu().numpy()[0]
    return inputs @ weight


def _metrics(pool: PairPool, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    target = pool.target[mask]
    predicted = prediction[mask]
    error = predicted - target
    absolute = np.abs(error)
    teacher_positive = pool.exact_margin[mask] > 0.0
    teacher_negative = pool.exact_margin[mask] < 0.0
    predicted_positive = predicted > 0.0
    predicted_negative = predicted < 0.0
    mistakes = ~(
        (teacher_positive & predicted_positive)
        | (teacher_negative & predicted_negative)
    )
    correlation = (
        float(np.corrcoef(target, predicted)[0, 1])
        if len(target) > 1 and np.std(target) > 0.0 and np.std(predicted) > 0.0
        else math.nan
    )
    return {
        "count": len(target),
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
        "mae": float(np.mean(absolute)) if len(error) else math.nan,
        "maximum_absolute_error": float(np.max(absolute)) if len(error) else math.nan,
        "correlation": correlation,
        "ordering_mistakes": int(np.sum(mistakes)),
        "positive_teacher_count": int(np.sum(teacher_positive)),
        "negative_teacher_count": int(np.sum(teacher_negative)),
        "positive_ordering_mistakes": int(
            np.sum(teacher_positive & ~predicted_positive)
        ),
        "negative_ordering_mistakes": int(
            np.sum(teacher_negative & ~predicted_negative)
        ),
    }


def _metric_rows(pool: PairPool, prediction: np.ndarray, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    masks: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(pool.target), dtype=bool)),
        ("ordinary_pairs", pool.category < 2),
        ("nearest_runner_up_pairs", pool.category >= 2),
    ]
    masks.extend(
        (name, pool.category == code) for code, name in CATEGORY_NAMES.items()
    )
    for name, mask in masks:
        rows.append(
            {
                "record_type": "continuous_and_ordering_metrics",
                "split": split,
                "pair_group": name,
                **_metrics(pool, prediction, mask),
            }
        )
    one_ulp = pool.one_ulp
    rows.append(
        {
            "record_type": "one_ulp_summary",
            "split": split,
            "pair_group": "all_one_ulp",
            **_metrics(pool, prediction, one_ulp),
        }
    )
    return rows


def _one_ulp_rows(pool: PairPool, prediction: np.ndarray, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(pool.one_ulp):
        teacher = "d2_closer" if pool.exact_margin[index] > 0.0 else "d1_closer"
        predicted = (
            "d2_closer"
            if prediction[index] > 0.0
            else "d1_closer" if prediction[index] < 0.0 else "zero_margin"
        )
        rows.append(
            {
                "record_type": "one_ulp_case",
                "split": split,
                "event_index": int(index),
                "source_state_index": int(pool.source_state_index[index]),
                "query_index": int(pool.query_index[index]),
                "pair_group": CATEGORY_NAMES[int(pool.category[index])],
                "site1": int(pool.site1[index]),
                "site2": int(pool.site2[index]),
                "exact_d1": float(pool.exact_d1[index]),
                "exact_d2": float(pool.exact_d2[index]),
                "locked_d1": float(pool.locked_d1[index]),
                "locked_d2": float(pool.locked_d2[index]),
                "x1": float(pool.inputs[index, 0]),
                "x2": float(pool.inputs[index, 1]),
                "m_teacher": float(pool.target[index]),
                "m_hat": float(prediction[index]),
                "absolute_error": float(abs(prediction[index] - pool.target[index])),
                "teacher_ordering": teacher,
                "predicted_ordering": predicted,
                "ordering_exact": teacher == predicted,
            }
        )
    return rows


def _mistake_rows(pool: PairPool, prediction: np.ndarray, split: str) -> list[dict[str, Any]]:
    exact_positive = pool.exact_margin > 0.0
    exact_negative = pool.exact_margin < 0.0
    correct = (exact_positive & (prediction > 0.0)) | (
        exact_negative & (prediction < 0.0)
    )
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(~correct):
        rows.append(
            {
                "record_type": "ordering_mistake",
                "split": split,
                "event_index": int(index),
                "source_state_index": int(pool.source_state_index[index]),
                "query_index": int(pool.query_index[index]),
                "pair_group": CATEGORY_NAMES[int(pool.category[index])],
                "site1": int(pool.site1[index]),
                "site2": int(pool.site2[index]),
                "exact_d1": float(pool.exact_d1[index]),
                "exact_d2": float(pool.exact_d2[index]),
                "exact_normalized_margin": float(pool.exact_margin[index]),
                "locked_normalized_margin": float(pool.target[index]),
                "predicted_margin": float(prediction[index]),
                "one_ulp": bool(pool.one_ulp[index]),
            }
        )
    return rows


def _train(
    training: PairPool, validation: PairPool
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial_weight = model.weight.detach().cpu().numpy()[0].copy()
    if np.array_equal(initial_weight, np.asarray([1.0, -1.0])):
        raise RuntimeError("forbidden exact coefficient initialization")

    train_x = torch.as_tensor(training.inputs, dtype=torch.float64)
    train_y = torch.as_tensor(training.target, dtype=torch.float64)
    gram = (train_x.T @ train_x) / len(train_x)
    cross = (train_x.T @ train_y) / len(train_x)
    yy = (train_y @ train_y) / len(train_y)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1.0e-15)
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    stale_checks = 0
    epoch_rows: list[dict[str, Any]] = []
    stop_epoch = MAX_ADAM_EPOCHS
    stopping_reason = "maximum_epochs"

    def objective() -> torch.Tensor:
        weight = model.weight.squeeze(0)
        return weight @ gram @ weight - 2.0 * (weight @ cross) + yy

    for epoch in range(1, MAX_ADAM_EPOCHS + 1):
        loss = objective()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % CHECK_INTERVAL != 0 and epoch != 1:
            continue
        prediction = _predict(model, validation.inputs)
        all_metrics = _metrics(
            validation, prediction, np.ones(len(validation.target), dtype=bool)
        )
        nearest_metrics = _metrics(validation, prediction, validation.category >= 2)
        ulp_metrics = _metrics(validation, prediction, validation.one_ulp)
        key = (
            nearest_metrics["ordering_mistakes"] + ulp_metrics["ordering_mistakes"],
            all_metrics["maximum_absolute_error"],
        )
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_checks = 0
        else:
            stale_checks += 1
        weight = model.weight.detach().cpu().numpy()[0]
        epoch_rows.append(
            {
                "record_type": "training_checkpoint",
                "epoch": epoch,
                "training_objective": float(loss.item()),
                "learned_w1": float(weight[0]),
                "learned_w2": float(weight[1]),
                "validation_rmse": all_metrics["rmse"],
                "validation_maximum_absolute_error": all_metrics[
                    "maximum_absolute_error"
                ],
                "validation_nearest_ordering_mistakes": nearest_metrics[
                    "ordering_mistakes"
                ],
                "validation_one_ulp_ordering_mistakes": ulp_metrics[
                    "ordering_mistakes"
                ],
            }
        )
        if (
            all_metrics["rmse"] <= 1.0e-15
            and all_metrics["maximum_absolute_error"] <= 1.0e-14
            and nearest_metrics["ordering_mistakes"] == 0
            and ulp_metrics["ordering_mistakes"] == 0
        ):
            stop_epoch = epoch
            stopping_reason = "numerical_precision_and_ordering_gate"
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            break
        if stale_checks >= PATIENCE_CHECKS:
            stop_epoch = epoch
            stopping_reason = "validation_plateau"
            break
        if epoch == 1 or epoch % 1000 == 0:
            print(
                f"nearest margin epoch={epoch} w=({weight[0]:.17g},"
                f"{weight[1]:.17g}) rmse={all_metrics['rmse']:.3e} "
                f"max={all_metrics['maximum_absolute_error']:.3e} "
                f"nearest_errors={nearest_metrics['ordering_mistakes']} "
                f"ulp_errors={ulp_metrics['ordering_mistakes']}",
                flush=True,
            )

    model.load_state_dict(best_state)
    # Polish the same full training regression objective. No ordering labels are
    # used by either optimizer.
    pre_polish = copy.deepcopy(model.state_dict())
    pre_prediction = _predict(model, validation.inputs)
    pre_metrics = _metrics(
        validation, pre_prediction, np.ones(len(validation.target), dtype=bool)
    )
    polish = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        tolerance_grad=1.0e-15,
        tolerance_change=1.0e-18,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        polish.zero_grad(set_to_none=True)
        value = objective()
        value.backward()
        return value

    polish.step(closure)
    post_prediction = _predict(model, validation.inputs)
    post_metrics = _metrics(
        validation, post_prediction, np.ones(len(validation.target), dtype=bool)
    )
    pre_order = pre_metrics["ordering_mistakes"]
    post_order = post_metrics["ordering_mistakes"]
    if (post_order, post_metrics["maximum_absolute_error"]) > (
        pre_order,
        pre_metrics["maximum_absolute_error"],
    ):
        model.load_state_dict(pre_polish)
        post_prediction = pre_prediction
        post_metrics = pre_metrics

    weights = model.weight.detach().cpu().numpy()[0]
    nearest = _metrics(validation, post_prediction, validation.category >= 2)
    one_ulp = _metrics(validation, post_prediction, validation.one_ulp)
    summary = {
        "record_type": "summary",
        "status": "passed" if (
            nearest["ordering_mistakes"] == 0
            and one_ulp["ordering_mistakes"] == 0
            and post_metrics["rmse"] <= 1.0e-15
            and post_metrics["maximum_absolute_error"] <= 1.0e-14
        ) else "failed",
        "training_state_count": len(np.unique(training.source_state_index)),
        "validation_state_count": len(np.unique(validation.source_state_index)),
        "training_pair_count": len(training.target),
        "validation_pair_count": len(validation.target),
        "training_exact_tie_exclusion_count": training.exact_tie_exclusion_count,
        "validation_exact_tie_exclusion_count": validation.exact_tie_exclusion_count,
        "training_locked_zero_margin_count": training.locked_zero_margin_count,
        "validation_locked_zero_margin_count": validation.locked_zero_margin_count,
        "initial_w1": float(initial_weight[0]),
        "initial_w2": float(initial_weight[1]),
        "learned_w1": float(weights[0]),
        "learned_w2": float(weights[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "LBFGS_continuous_polish_used": True,
        "validation_rmse": post_metrics["rmse"],
        "validation_mae": post_metrics["mae"],
        "validation_maximum_absolute_error": post_metrics["maximum_absolute_error"],
        "validation_correlation": post_metrics["correlation"],
        "validation_ordering_mistakes": post_metrics["ordering_mistakes"],
        "validation_nearest_ordering_mistakes": nearest["ordering_mistakes"],
        "validation_one_ulp_count": one_ulp["count"],
        "validation_one_ulp_ordering_mistakes": one_ulp["ordering_mistakes"],
        "locked_distance_sha256_verified": LOCKED_DISTANCE_SHA256,
    }
    return model, epoch_rows, summary


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    locked_hash_before = _sha256(LOCKED_DISTANCE_PATH)
    distance_model, distance_checkpoint = _load_locked_distance()
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)

    source_train = [int(value) for value in distance_checkpoint["smoke_training_state_indices"]]
    source_validation = [
        int(value) for value in distance_checkpoint["smoke_validation_state_indices"]
    ]
    source_held_out = [
        int(value) for value in distance_checkpoint["full_held_out_state_indices"]
    ]
    train_states = source_train[:SMOKE_TRAIN_STATES]
    validation_states = source_validation[:SMOKE_VALIDATION_STATES]
    if set(train_states) & set(validation_states):
        raise RuntimeError("smoke train and validation states overlap")
    if (set(train_states) | set(validation_states)) & set(source_held_out):
        raise RuntimeError("smoke and full held-out states overlap")

    training = _build_pool(
        distance_model,
        payload,
        trajectories,
        train_states,
        TRAIN_ORDINARY_PER_TYPE_PER_STATE,
        DATA_SEED,
        "nearest-margin training",
    )
    validation = _build_pool(
        distance_model,
        payload,
        trajectories,
        validation_states,
        VALIDATION_ORDINARY_PER_TYPE_PER_STATE,
        DATA_SEED + 1,
        "nearest-margin validation",
    )
    model, epoch_rows, smoke_summary = _train(training, validation)
    validation_prediction = _predict(model, validation.inputs)
    smoke_rows: list[dict[str, Any]] = [smoke_summary]
    smoke_rows.extend(_metric_rows(training, _predict(model, training.inputs), "training"))
    smoke_rows.extend(_metric_rows(validation, validation_prediction, "validation"))
    smoke_rows.extend(_one_ulp_rows(training, _predict(model, training.inputs), "training"))
    smoke_rows.extend(_one_ulp_rows(validation, validation_prediction, "validation"))
    smoke_rows.extend(_mistake_rows(validation, validation_prediction, "validation"))
    smoke_rows.extend(epoch_rows)
    _write_csv(SMOKE_PATH, smoke_rows)
    print(
        f"nearest margin smoke status={smoke_summary['status']} "
        f"w=({smoke_summary['learned_w1']:.17g},"
        f"{smoke_summary['learned_w2']:.17g}) "
        f"nearest_errors={smoke_summary['validation_nearest_ordering_mistakes']} "
        f"ulp_errors={smoke_summary['validation_one_ulp_ordering_mistakes']}",
        flush=True,
    )
    if smoke_summary["status"] != "passed":
        _write_csv(
            FULL_PATH,
            [
                {
                    "record_type": "gate_stop",
                    "status": "not_run",
                    "reason": "smoke_gate_failed_near_tied_ordering",
                    "validation_ordering_mistakes": smoke_summary[
                        "validation_ordering_mistakes"
                    ],
                    "validation_nearest_ordering_mistakes": smoke_summary[
                        "validation_nearest_ordering_mistakes"
                    ],
                    "validation_one_ulp_count": smoke_summary[
                        "validation_one_ulp_count"
                    ],
                    "validation_one_ulp_ordering_mistakes": smoke_summary[
                        "validation_one_ulp_ordering_mistakes"
                    ],
                    "candidate_checkpoint_saved": False,
                }
            ],
        )
        if _sha256(LOCKED_DISTANCE_PATH) != locked_hash_before:
            raise RuntimeError("locked distance checkpoint changed")
        return

    full_pool = _build_pool(
        distance_model,
        payload,
        trajectories,
        source_held_out,
        FULL_ORDINARY_PER_TYPE_PER_STATE,
        DATA_SEED + 2,
        "nearest-margin full held-out",
    )
    full_prediction = _predict(model, full_pool.inputs)
    all_metrics = _metrics(
        full_pool, full_prediction, np.ones(len(full_pool.target), dtype=bool)
    )
    nearest_metrics = _metrics(full_pool, full_prediction, full_pool.category >= 2)
    ulp_metrics = _metrics(full_pool, full_prediction, full_pool.one_ulp)
    full_passed = bool(
        all_metrics["ordering_mistakes"] == 0
        and nearest_metrics["ordering_mistakes"] == 0
        and ulp_metrics["ordering_mistakes"] == 0
        and full_pool.locked_zero_margin_count == 0
    )
    weights = model.weight.detach().cpu().numpy()[0]
    full_summary = {
        "record_type": "summary",
        "status": "FROZEN_CANDIDATE" if full_passed else "failed",
        "held_out_state_count": len(source_held_out),
        "held_out_pair_count": len(full_pool.target),
        "held_out_exact_tie_exclusion_count": full_pool.exact_tie_exclusion_count,
        "held_out_locked_zero_margin_count": full_pool.locked_zero_margin_count,
        "learned_w1": float(weights[0]),
        "learned_w2": float(weights[1]),
        "bias": 0.0,
        "rmse": all_metrics["rmse"],
        "mae": all_metrics["mae"],
        "maximum_absolute_error": all_metrics["maximum_absolute_error"],
        "correlation": all_metrics["correlation"],
        "all_ordering_mistakes": all_metrics["ordering_mistakes"],
        "nearest_runner_up_ordering_mistakes": nearest_metrics[
            "ordering_mistakes"
        ],
        "one_ulp_case_count": ulp_metrics["count"],
        "one_ulp_ordering_mistakes": ulp_metrics["ordering_mistakes"],
        "locked_distance_sha256_verified": LOCKED_DISTANCE_SHA256,
    }
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(_metric_rows(full_pool, full_prediction, "full_held_out"))
    full_rows.extend(_one_ulp_rows(full_pool, full_prediction, "full_held_out"))
    full_rows.extend(_mistake_rows(full_pool, full_prediction, "full_held_out"))

    if full_passed:
        checkpoint = {
            "format_version": 1,
            "component": "nearest_distance_margin",
            "status": "FROZEN_CANDIDATE",
            "architecture": "m_hat=w1*x1+w2*x2; no bias",
            "dtype": "float64",
            "state_dict": copy.deepcopy(model.state_dict()),
            "effective_weights": torch.as_tensor(weights, dtype=torch.float64),
            "bias": torch.tensor(0.0, dtype=torch.float64),
            "locked_distance_checkpoint": str(LOCKED_DISTANCE_PATH),
            "locked_distance_sha256": LOCKED_DISTANCE_SHA256,
            "exact_tie_policy": "excluded_from_learning_and_retained_as_deterministic_ckdtree_bookkeeping",
            "training_state_indices": train_states,
            "validation_state_indices": validation_states,
            "full_held_out_state_indices": source_held_out,
            "smoke_summary": smoke_summary,
            "full_summary": full_summary,
        }
        torch.save(checkpoint, CANDIDATE_PATH)
        candidate_hash = _sha256(CANDIDATE_PATH)
        HASH_PATH.write_text(
            f"{candidate_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
        )
        full_summary["checkpoint_path"] = str(CANDIDATE_PATH)
        full_summary["checkpoint_sha256"] = candidate_hash
        full_rows[0] = full_summary

    _write_csv(FULL_PATH, full_rows)
    locked_hash_after = _sha256(LOCKED_DISTANCE_PATH)
    if locked_hash_after != locked_hash_before:
        raise RuntimeError("locked distance checkpoint changed during experiment")
    print(
        f"nearest margin full status={full_summary['status']} "
        f"pairs={len(full_pool.target)} errors={all_metrics['ordering_mistakes']} "
        f"one_ulp={ulp_metrics['count']} one_ulp_errors={ulp_metrics['ordering_mistakes']}",
        flush=True,
    )
    if full_passed:
        print(f"candidate sha256={full_summary['checkpoint_sha256']}", flush=True)


if __name__ == "__main__":
    run()
