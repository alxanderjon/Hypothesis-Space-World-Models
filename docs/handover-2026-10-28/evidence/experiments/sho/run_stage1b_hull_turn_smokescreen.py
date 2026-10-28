#!/usr/bin/env python3
"""Smoke-screen the local three-point turn decision in eDIRECT's hull stack."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_turn_smokescreen")
TURN_PATH = OUTPUT_DIR / "hull_turn_smokescreen.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "hull_reconstruction_smokescreen.csv"
RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH

TRAIN_TRIPLES = 5_000
VALIDATION_TRIPLES = 1_000
RECONSTRUCTION_STATES = 200
TRAIN_SOURCE_STATES = 240
VALIDATION_SOURCE_STATES = 80
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 2.0e-3
HIDDEN_DIM = 32
MODEL_SEED = 1_618_033
DATA_SEED = 2_718_281
SIZE_GROUP_TOLERANCE = 1.0e-14
HULL_TOLERANCE = 5.0e-14
NEAR_TOLERANCE_MULTIPLIER = 10.0
STATE_BINS = (
    ("small_10_64", 0, 64),
    ("medium_65_256", 65, 256),
    ("large_257_1024", 257, 1024),
    ("thousands_1025_plus", 1025, math.inf),
)


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


def _state_bin(count: int) -> str:
    for name, lower, upper in STATE_BINS:
        if lower <= count <= upper:
            return name
    raise ValueError(count)


def _split_name(trajectory_id: int) -> str:
    remainder = trajectory_id % 10
    if remainder <= 6:
        return "train"
    if remainder <= 8:
        return "validation"
    return "reconstruction"


def _stratified_state_indices(
    states: list[dict[str, Any]], split: str, total: int, seed: int
) -> list[int]:
    generator = np.random.default_rng(seed)
    by_bin: dict[str, list[int]] = {name: [] for name, _, _ in STATE_BINS}
    for index, state in enumerate(states):
        if state["candidate_type"] != "cell":
            continue
        if _split_name(int(state["trajectory_id"])) != split:
            continue
        by_bin[_state_bin(int(state["history_count"]))].append(index)
    base, remainder = divmod(total, len(STATE_BINS))
    chosen: list[int] = []
    for position, (name, _, _) in enumerate(STATE_BINS):
        count = base + int(position < remainder)
        candidates = np.asarray(by_bin[name], dtype=np.int64)
        if len(candidates) < count:
            raise RuntimeError(f"not enough {split}/{name} states")
        selected = generator.choice(candidates, size=count, replace=False)
        chosen.extend(int(value) for value in selected)
    generator.shuffle(chosen)
    return chosen


def _same_size_representatives(
    sizes: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(sizes)), losses, sizes))
    groups: list[tuple[float, float, int]] = []
    for index_value in order:
        index = int(index_value)
        size = float(sizes[index])
        loss = float(losses[index])
        if not groups or abs(size - groups[-1][0]) > SIZE_GROUP_TOLERANCE:
            groups.append((size, loss, index))
        elif loss < groups[-1][1]:
            groups[-1] = (groups[-1][0], loss, index)
    return (
        np.asarray([group[0] for group in groups], dtype=np.float64),
        np.asarray([group[1] for group in groups], dtype=np.float64),
        np.asarray([group[2] for group in groups], dtype=np.int64),
    )


def _state_scales(sizes: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    size_scale = float(np.ptp(sizes))
    loss_scale = float(np.ptp(losses))
    if not np.isfinite(size_scale) or size_scale <= 0.0:
        size_scale = 1.0
    if not np.isfinite(loss_scale) or loss_scale <= 0.0:
        loss_scale = 1.0
    return size_scale, loss_scale


def _turn_values(
    sizes: np.ndarray,
    losses: np.ndarray,
    first: int,
    second: int,
    third: int,
    size_scale: float,
    loss_scale: float,
) -> tuple[np.ndarray, float, float, int]:
    delta_d_ab = float(sizes[second] - sizes[first])
    delta_f_ab = float(losses[second] - losses[first])
    delta_d_bc = float(sizes[third] - sizes[second])
    delta_f_bc = float(losses[third] - losses[second])
    cross = delta_d_ab * delta_f_bc - delta_f_ab * delta_d_bc
    scale = max(
        1.0,
        abs(float(losses[first])),
        abs(float(losses[second])),
        abs(float(losses[third])),
    )
    tolerance = HULL_TOLERANCE * scale
    keep = int(not (cross < -tolerance))
    features = np.asarray(
        (
            delta_d_ab / size_scale,
            delta_f_ab / loss_scale,
            delta_d_bc / size_scale,
            delta_f_bc / loss_scale,
        ),
        dtype=np.float64,
    )
    return features, cross, tolerance, keep


def _teacher_hull_and_triples(
    sizes: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    size_scale, loss_scale = _state_scales(sizes, losses)
    hull: list[int] = []
    features: list[np.ndarray] = []
    crosses: list[float] = []
    tolerances: list[float] = []
    labels: list[int] = []
    indices: list[tuple[int, int, int]] = []
    for third in range(len(sizes)):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            feature, cross, tolerance, keep = _turn_values(
                sizes,
                losses,
                first,
                second,
                third,
                size_scale,
                loss_scale,
            )
            features.append(feature)
            crosses.append(cross)
            tolerances.append(tolerance)
            labels.append(keep)
            indices.append((first, second, third))
            if keep:
                break
            hull.pop()
        hull.append(third)
    if features:
        feature_array = np.stack(features)
        index_array = np.asarray(indices, dtype=np.int64)
    else:
        feature_array = np.empty((0, 4), dtype=np.float64)
        index_array = np.empty((0, 3), dtype=np.int64)
    cross_array = np.asarray(crosses, dtype=np.float64)
    tolerance_array = np.asarray(tolerances, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    exact = cross_array == 0.0
    near = np.abs(cross_array) <= NEAR_TOLERANCE_MULTIPLIER * tolerance_array
    return np.asarray(hull, dtype=np.int64), {
        "features": feature_array,
        "cross": cross_array,
        "tolerance": tolerance_array,
        "label": label_array,
        "triple_indices": index_array,
        "exact_collinear": exact,
        "near_collinear": near,
    }


def _reconstruct_state(
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    history_count = int(record["history_count"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    pool = edirect._candidate_pool(
        history_count, int(record["generator_id"]), int(record["iteration"])
    )
    tree = cKDTree(points)
    distances, owners = tree.query(pool, k=1)
    sizes = np.zeros(history_count, dtype=np.float64)
    np.maximum.at(sizes, owners, distances)
    if history_count > 1:
        sizes = np.maximum(sizes, 0.5 * tree.query(points, k=2)[0][:, 1])
    grouped_sizes, grouped_losses, representatives = _same_size_representatives(
        sizes, losses
    )
    hull, triples = _teacher_hull_and_triples(grouped_sizes, grouped_losses)
    return {
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": int(record["iteration"]),
        "history_count": history_count,
        "state_bin": _state_bin(history_count),
        "sizes": grouped_sizes,
        "losses": grouped_losses,
        "representatives": representatives,
        "teacher_hull": hull,
        "triples": triples,
    }


def _build_trajectory_states(
    arguments: tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]
) -> list[dict[str, Any]]:
    records, trajectory = arguments
    return [
        _reconstruct_state(source_index, record, trajectory)
        for source_index, record in records
    ]


def _build_states(
    payload: dict[str, Any], indices: list[int], workers: int, label: str
) -> list[dict[str, Any]]:
    trajectories = ordinary_data._trajectory_map(payload)
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for source_index in indices:
        record = payload["states"][source_index]
        grouped.setdefault(int(record["trajectory_id"]), []).append(
            (source_index, record)
        )
    tasks = [
        (records, trajectories[trajectory_id])
        for trajectory_id, records in sorted(grouped.items())
    ]
    built: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            built.extend(_build_trajectory_states(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_trajectory_states, task) for task in tasks]
            for future in as_completed(futures):
                batch = future.result()
                built.extend(batch)
                if len(built) % 40 < len(batch) or len(built) == len(indices):
                    print(f"{label} states {len(built)}/{len(indices)}", flush=True)
    by_index = {int(state["source_state_index"]): state for state in built}
    return [by_index[index] for index in indices]


def _flatten_triples(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "cross": [],
        "tolerance": [],
        "label": [],
        "triple_indices": [],
        "exact_collinear": [],
        "near_collinear": [],
        "state_position": [],
    }
    for state_position, state in enumerate(states):
        triples = state["triples"]
        count = len(triples["label"])
        if count == 0:
            continue
        for key in arrays:
            if key == "state_position":
                arrays[key].append(np.full(count, state_position, dtype=np.int64))
            else:
                arrays[key].append(triples[key])
    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


def _select_triples(
    pool: dict[str, np.ndarray], target: int, seed: int
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    total = len(pool["label"])
    if total < target:
        raise RuntimeError(f"only {total} triples available for target {target}")
    all_indices = np.arange(total, dtype=np.int64)
    selected: list[int] = []
    used = np.zeros(total, dtype=np.bool_)

    def add(indices: np.ndarray, count: int) -> None:
        available = indices[~used[indices]]
        if len(available) == 0 or count <= 0:
            return
        take = min(count, len(available), target - len(selected))
        chosen = generator.choice(available, size=take, replace=False)
        used[chosen] = True
        selected.extend(int(value) for value in chosen)

    exact = all_indices[pool["exact_collinear"]]
    near_nonexact = all_indices[
        pool["near_collinear"] & ~pool["exact_collinear"]
    ]
    add(exact, target // 20)
    add(near_nonexact, target // 10)

    half = target // 2
    current_keep = sum(int(pool["label"][index]) == 1 for index in selected)
    current_pop = len(selected) - current_keep
    add(all_indices[pool["label"] == 1], max(0, half - current_keep))
    add(all_indices[pool["label"] == 0], max(0, half - current_pop))
    add(all_indices, target - len(selected))
    if len(selected) != target:
        raise RuntimeError(f"selected {len(selected)} triples, expected {target}")
    selected_array = np.asarray(selected, dtype=np.int64)
    generator.shuffle(selected_array)
    return {key: values[selected_array] for key, values in pool.items()}


class TurnMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _confusion(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    predictions_bool = predictions.astype(bool)
    keep_tp = int(np.sum(labels_bool & predictions_bool))
    keep_fp = int(np.sum(~labels_bool & predictions_bool))
    keep_fn = int(np.sum(labels_bool & ~predictions_bool))
    pop_tp = int(np.sum(~labels_bool & ~predictions_bool))
    pop_fp = int(np.sum(labels_bool & ~predictions_bool))
    pop_fn = int(np.sum(~labels_bool & predictions_bool))

    def precision(tp: int, fp: int) -> float:
        return tp / (tp + fp) if tp + fp else 0.0

    def recall(tp: int, fn: int) -> float:
        return tp / (tp + fn) if tp + fn else 0.0

    return {
        "accuracy": float(np.mean(labels == predictions)),
        "keep_count": int(np.sum(labels_bool)),
        "pop_count": int(np.sum(~labels_bool)),
        "keep_precision": precision(keep_tp, keep_fp),
        "keep_recall": recall(keep_tp, keep_fn),
        "pop_precision": precision(pop_tp, pop_fp),
        "pop_recall": recall(pop_tp, pop_fn),
        "keep_tp": keep_tp,
        "keep_fp": keep_fp,
        "keep_fn": keep_fn,
        "pop_tp": pop_tp,
        "pop_fp": pop_fp,
        "pop_fn": pop_fn,
    }


def _predict(model: TurnMLP, dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    features = torch.as_tensor(dataset["features"], dtype=torch.float64)
    labels = torch.as_tensor(dataset["label"], dtype=torch.float64)
    logits_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            logits_parts.append(model(features[start : start + 2048]))
    logits = torch.cat(logits_parts)
    loss = float(nn.functional.binary_cross_entropy_with_logits(logits, labels).item())
    probabilities = torch.sigmoid(logits).cpu().numpy()
    return (probabilities >= 0.5).astype(np.int64), probabilities, loss


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[TurnMLP, list[dict[str, Any]], int, np.ndarray, np.ndarray, float]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED)
    model = TurnMLP().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_features = torch.as_tensor(training["features"], dtype=torch.float64)
    train_labels = torch.as_tensor(training["label"], dtype=torch.float64)
    keep_count = int(np.sum(training["label"] == 1))
    pop_count = len(training["label"]) - keep_count
    positive_weight = torch.tensor(pop_count / max(keep_count, 1), dtype=torch.float64)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_accuracy = -math.inf
    best_loss = math.inf
    stale = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = generator.permutation(len(train_features))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            logits = model(train_features[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_labels[batch], pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        predictions, _, validation_loss = _predict(model, validation)
        metrics = _confusion(validation["label"], predictions)
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "train_weighted_bce": total_loss / len(order),
                "validation_bce": validation_loss,
                "validation_accuracy": metrics["accuracy"],
                "validation_keep_precision": metrics["keep_precision"],
                "validation_keep_recall": metrics["keep_recall"],
                "validation_pop_precision": metrics["pop_precision"],
                "validation_pop_recall": metrics["pop_recall"],
            }
        )
        improved = (
            float(metrics["accuracy"]) > best_accuracy + 1.0e-12
            or (
                abs(float(metrics["accuracy"]) - best_accuracy) <= 1.0e-12
                and validation_loss < best_loss - 1.0e-12
            )
        )
        if improved:
            best_accuracy = float(metrics["accuracy"])
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch} validation_accuracy={float(metrics['accuracy']):.6f} "
            f"validation_bce={validation_loss:.6e}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    predictions, probabilities, validation_loss = _predict(model, validation)
    return model, rows, best_epoch, predictions, probabilities, validation_loss


def _subset_row(
    name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    record_type: str,
) -> dict[str, Any]:
    count = int(np.sum(mask))
    if count == 0:
        return {
            "record_type": record_type,
            "subset": name,
            "count": 0,
            "accuracy": "",
        }
    return {
        "record_type": record_type,
        "subset": name,
        "count": count,
        **_confusion(labels[mask], predictions[mask]),
    }


def _learned_hull(
    model: TurnMLP, sizes: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, int]:
    size_scale, loss_scale = _state_scales(sizes, losses)
    stack: list[int] = []
    decision_count = 0
    model.eval()
    with torch.no_grad():
        for third in range(len(sizes)):
            while len(stack) >= 2:
                first, second = stack[-2], stack[-1]
                features, _, _, _ = _turn_values(
                    sizes,
                    losses,
                    first,
                    second,
                    third,
                    size_scale,
                    loss_scale,
                )
                logit = model(torch.as_tensor(features).unsqueeze(0))
                keep = bool(torch.sigmoid(logit)[0].item() >= 0.5)
                decision_count += 1
                if keep:
                    break
                stack.pop()
            stack.append(third)
    return np.asarray(stack, dtype=np.int64), decision_count


def _deterministic_hull(sizes: np.ndarray, losses: np.ndarray) -> np.ndarray:
    hull, _ = _teacher_hull_and_triples(sizes, losses)
    return hull


def _reconstruction_rows(
    model: TurnMLP, states: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    per_state: list[dict[str, Any]] = []
    for state in states:
        teacher = np.asarray(state["teacher_hull"], dtype=np.int64)
        deterministic = _deterministic_hull(state["sizes"], state["losses"])
        predicted, decision_count = _learned_hull(
            model, state["sizes"], state["losses"]
        )
        teacher_set = set(int(value) for value in teacher)
        predicted_set = set(int(value) for value in predicted)
        false_additions = sorted(predicted_set - teacher_set)
        false_removals = sorted(teacher_set - predicted_set)
        per_state.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "iteration": state["iteration"],
                "history_count": state["history_count"],
                "state_bin": state["state_bin"],
                "representative_cell_count": len(state["sizes"]),
                "teacher_hull_size": len(teacher),
                "predicted_hull_size": len(predicted),
                "learned_turn_decisions": decision_count,
                "false_hull_additions": len(false_additions),
                "false_hull_removals": len(false_removals),
                "false_addition_indices": ";".join(map(str, false_additions)),
                "false_removal_indices": ";".join(map(str, false_removals)),
                "exact_hull": len(false_additions) == 0 and len(false_removals) == 0,
                "deterministic_exact": bool(np.array_equal(teacher, deterministic)),
            }
        )

    def summary_row(
        record_type: str, subset: str, subset_rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "record_type": record_type,
            "subset": subset,
            "state_count": len(subset_rows),
            "exact_hull_states": sum(bool(row["exact_hull"]) for row in subset_rows),
            "exact_hull_fraction": float(
                np.mean([row["exact_hull"] for row in subset_rows])
            ),
            "average_false_hull_additions": float(
                np.mean([row["false_hull_additions"] for row in subset_rows])
            ),
            "average_false_hull_removals": float(
                np.mean([row["false_hull_removals"] for row in subset_rows])
            ),
            "maximum_false_hull_additions": max(
                int(row["false_hull_additions"]) for row in subset_rows
            ),
            "maximum_false_hull_removals": max(
                int(row["false_hull_removals"]) for row in subset_rows
            ),
            "deterministic_exact_states": sum(
                bool(row["deterministic_exact"]) for row in subset_rows
            ),
        }

    rows = [summary_row("final_summary", "all", per_state)]
    for name, lower, upper in STATE_BINS:
        subset = [
            row
            for row in per_state
            if lower <= int(row["representative_cell_count"]) <= upper
        ]
        if subset:
            rows.append(summary_row("by_representative_cell_count", name, subset))
    rows.extend(per_state)
    return rows


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = _stratified_state_indices(
        payload["states"], "train", TRAIN_SOURCE_STATES, DATA_SEED
    )
    validation_indices = _stratified_state_indices(
        payload["states"], "validation", VALIDATION_SOURCE_STATES, DATA_SEED + 1
    )
    reconstruction_indices = _stratified_state_indices(
        payload["states"], "reconstruction", RECONSTRUCTION_STATES, DATA_SEED + 2
    )
    train_states = _build_states(payload, train_indices, workers, "train")
    validation_states = _build_states(
        payload, validation_indices, workers, "validation"
    )
    reconstruction_states = _build_states(
        payload, reconstruction_indices, workers, "reconstruction"
    )

    train_pool = _flatten_triples(train_states)
    validation_pool = _flatten_triples(validation_states)
    training = _select_triples(train_pool, TRAIN_TRIPLES, DATA_SEED + 3)
    validation = _select_triples(
        validation_pool, VALIDATION_TRIPLES, DATA_SEED + 4
    )
    print(
        f"triple pools train={len(train_pool['label'])} validation={len(validation_pool['label'])}; "
        f"selected={len(training['label'])}/{len(validation['label'])}",
        flush=True,
    )

    model, rows, best_epoch, predictions, probabilities, validation_loss = _train(
        training, validation
    )
    final_metrics = _confusion(validation["label"], predictions)
    teacher_recomputed = (
        validation["cross"] < -validation["tolerance"]
    ).astype(np.int64)
    teacher_recomputed = 1 - teacher_recomputed
    deterministic_exact = int(
        np.sum(teacher_recomputed == validation["label"])
    )
    rows.append(
        {
            "record_type": "final_summary",
            "best_epoch": best_epoch,
            "epochs_trained": sum(row["record_type"] == "epoch" for row in rows),
            "train_source_states": len(train_states),
            "validation_source_states": len(validation_states),
            "train_triple_pool": len(train_pool["label"]),
            "validation_triple_pool": len(validation_pool["label"]),
            "training_triples": len(training["label"]),
            "validation_triples": len(validation["label"]),
            "validation_bce": validation_loss,
            **final_metrics,
            "training_near_collinear_count": int(
                np.sum(training["near_collinear"])
            ),
            "validation_near_collinear_count": int(
                np.sum(validation["near_collinear"])
            ),
            "training_exact_collinear_count": int(
                np.sum(training["exact_collinear"])
            ),
            "validation_exact_collinear_count": int(
                np.sum(validation["exact_collinear"])
            ),
            "train_pool_exact_collinear_count": int(
                np.sum(train_pool["exact_collinear"])
            ),
            "validation_pool_exact_collinear_count": int(
                np.sum(validation_pool["exact_collinear"])
            ),
            "near_collinear_count": int(np.sum(validation["near_collinear"])),
            "near_collinear_accuracy": float(
                np.mean(
                    predictions[validation["near_collinear"]]
                    == validation["label"][validation["near_collinear"]]
                )
            )
            if np.any(validation["near_collinear"])
            else "",
            "exact_collinear_count": int(np.sum(validation["exact_collinear"])),
            "exact_collinear_accuracy": float(
                np.mean(
                    predictions[validation["exact_collinear"]]
                    == validation["label"][validation["exact_collinear"]]
                )
            )
            if np.any(validation["exact_collinear"])
            else "",
            "deterministic_teacher_sanity_correct": deterministic_exact,
            "deterministic_teacher_sanity_total": len(validation["label"]),
            "full_hull_mask_trained": False,
            "closed_loop_search_run": False,
            "frozen_networks_loaded_or_modified": False,
        }
    )
    labels = validation["label"]
    near = validation["near_collinear"]
    exact = validation["exact_collinear"]
    rows.append(
        _subset_row("near_collinear", labels, predictions, near, "geometry_subset")
    )
    rows.append(
        _subset_row("exact_collinear", labels, predictions, exact, "geometry_subset")
    )
    rows.append(
        _subset_row(
            "clear_keep",
            labels,
            predictions,
            (labels == 1) & ~near,
            "geometry_subset",
        )
    )
    rows.append(
        _subset_row(
            "clear_pop",
            labels,
            predictions,
            (labels == 0) & ~near,
            "geometry_subset",
        )
    )
    magnitude_order = np.argsort(np.abs(validation["cross"]), kind="stable")
    for quartile, indices in enumerate(np.array_split(magnitude_order, 4), start=1):
        mask = np.zeros(len(labels), dtype=np.bool_)
        mask[indices] = True
        row = _subset_row(
            f"Q{quartile}", labels, predictions, mask, "cross_magnitude_quartile"
        )
        row["minimum_abs_cross"] = float(np.min(np.abs(validation["cross"][indices])))
        row["maximum_abs_cross"] = float(np.max(np.abs(validation["cross"][indices])))
        rows.append(row)

    error_indices = np.flatnonzero(predictions != labels)
    for index in error_indices:
        state = validation_states[int(validation["state_position"][index])]
        triple_indices = validation["triple_indices"][index]
        rows.append(
            {
                "record_type": "validation_error",
                "validation_triple_index": int(index),
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "history_count": state["history_count"],
                "group_a": int(triple_indices[0]),
                "group_b": int(triple_indices[1]),
                "group_c": int(triple_indices[2]),
                "teacher_label_keep": int(labels[index]),
                "predicted_label_keep": int(predictions[index]),
                "predicted_keep_probability": float(probabilities[index]),
                "teacher_cross": float(validation["cross"][index]),
                "teacher_tolerance": float(validation["tolerance"][index]),
                "near_collinear": bool(near[index]),
                "exact_collinear": bool(exact[index]),
            }
        )
    _write_csv(TURN_PATH, rows)

    reconstruction_rows = _reconstruction_rows(model, reconstruction_states)
    _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
