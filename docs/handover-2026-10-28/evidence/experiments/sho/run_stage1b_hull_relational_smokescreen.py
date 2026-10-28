#!/usr/bin/env python3
"""Relational smoke screen for DIRECT lower-right-hull membership only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_relational_smokescreen")
SUMMARY_PATH = OUTPUT_DIR / "hull_smokescreen_summary.csv"
ERRORS_PATH = OUTPUT_DIR / "hull_smokescreen_errors.csv"
RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH

LOCKED_PATHS = {
    "largest_ordinary_router": Path(
        "outputs/sho/stage1B/largest_cell_router/best_router_checkpoint.pt"
    ),
    "ordinary_cell_winner": Path(
        "outputs/sho/stage1B/ordinary_cell_score_full/best_score_network_checkpoint.pt"
    ),
    "largest_cell_winner": Path(
        "outputs/sho/stage1B/largest_cell_score/best_largest_cell_checkpoint.pt"
    ),
    "final_selector": Path(
        "outputs/sho/stage1B/specialized_cell_filter/frozen_selector_checkpoint.pt"
    ),
}

TRAIN_STATES = 500
VALIDATION_STATES = 150
MAX_EPOCHS = 30
PATIENCE = 8
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 48
HEAD_COUNT = 4
INDUCING_COUNT = 24
TOKEN_BUDGET = 8192
MAX_BATCH_STATES = 16
MODEL_SEED = 31_415_927
DATA_SEED = 27_182_818
SIZE_GROUP_TOLERANCE = 1.0e-14
HULL_TOLERANCE = 5.0e-14
STATE_BINS = (
    ("small_10_64", 0, 64),
    ("medium_65_256", 65, 256),
    ("large_257_1024", 257, 1024),
    ("thousands_1025_plus", 1025, math.inf),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(
    path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...] = ()
) -> None:
    fields: list[str] = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _state_bin(history_count: int) -> str:
    for name, lower, upper in STATE_BINS:
        if lower <= history_count <= upper:
            return name
    raise ValueError(history_count)


def _stratified_indices(
    states: list[dict[str, Any]], validation: bool, total: int, seed: int
) -> list[int]:
    generator = np.random.default_rng(seed)
    by_bin: dict[str, list[int]] = {name: [] for name, _, _ in STATE_BINS}
    for index, state in enumerate(states):
        if state["candidate_type"] != "cell":
            continue
        trajectory_id = int(state["trajectory_id"])
        if ((trajectory_id % 5) == 0) != validation:
            continue
        by_bin[_state_bin(int(state["history_count"]))].append(index)
    base, remainder = divmod(total, len(STATE_BINS))
    chosen: list[int] = []
    for bin_position, (name, _, _) in enumerate(STATE_BINS):
        count = base + int(bin_position < remainder)
        candidates = np.asarray(by_bin[name], dtype=np.int64)
        selected = generator.choice(candidates, size=count, replace=False)
        chosen.extend(int(index) for index in selected)
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


def _teacher_hull_labels(
    grouped_sizes: np.ndarray, grouped_losses: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    hull: list[int] = []
    collinear_case = False
    for index in range(len(grouped_sizes)):
        size = float(grouped_sizes[index])
        loss = float(grouped_losses[index])
        while len(hull) >= 2:
            first = hull[-2]
            second = hull[-1]
            cross = (grouped_sizes[second] - grouped_sizes[first]) * (
                loss - grouped_losses[first]
            ) - (grouped_losses[second] - grouped_losses[first]) * (
                size - grouped_sizes[first]
            )
            scale = max(
                1.0,
                abs(float(grouped_losses[first])),
                abs(float(grouped_losses[second])),
                abs(loss),
            )
            tolerance = HULL_TOLERANCE * scale
            if cross < -tolerance:
                hull.pop()
            else:
                if abs(float(cross)) <= tolerance:
                    collinear_case = True
                break
        hull.append(index)
    labels = np.zeros(len(grouped_sizes), dtype=np.float64)
    labels[hull] = 1.0
    return labels, np.asarray(hull, dtype=np.int64), collinear_case


def _independent_hull_labels(
    grouped_sizes: np.ndarray, grouped_losses: np.ndarray
) -> np.ndarray:
    stack: list[int] = []
    for candidate in range(len(grouped_sizes)):
        while len(stack) > 1:
            left, middle = stack[-2], stack[-1]
            orientation = (grouped_sizes[middle] - grouped_sizes[left]) * (
                grouped_losses[candidate] - grouped_losses[left]
            ) - (grouped_losses[middle] - grouped_losses[left]) * (
                grouped_sizes[candidate] - grouped_sizes[left]
            )
            scale = max(
                1.0,
                abs(float(grouped_losses[left])),
                abs(float(grouped_losses[middle])),
                abs(float(grouped_losses[candidate])),
            )
            if orientation < -HULL_TOLERANCE * scale:
                stack.pop()
            else:
                break
        stack.append(candidate)
    labels = np.zeros(len(grouped_sizes), dtype=np.float64)
    labels[stack] = 1.0
    return labels


def _affine_normalize(values: np.ndarray) -> np.ndarray:
    scale = max(float(np.max(np.abs(values))), 1.0)
    scaled = values / scale
    lower = float(np.min(scaled))
    upper = float(np.max(scaled))
    if upper <= lower:
        return np.full_like(scaled, 0.5)
    return (scaled - lower) / (upper - lower)


def _build_one(
    source_state_index: int,
    record: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    history_count = int(record["history_count"])
    generator_id = int(record["generator_id"])
    iteration = int(record["iteration"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    pool = edirect._candidate_pool(history_count, generator_id, iteration)
    tree = cKDTree(points)
    distances, owners = tree.query(pool, k=1)
    sizes = np.zeros(history_count, dtype=np.float64)
    np.maximum.at(sizes, owners, distances)
    if history_count > 1:
        sizes = np.maximum(sizes, 0.5 * tree.query(points, k=2)[0][:, 1])
    grouped_sizes, grouped_losses, representatives = _same_size_representatives(
        sizes, losses
    )
    labels, hull, collinear_case = _teacher_hull_labels(
        grouped_sizes, grouped_losses
    )
    deterministic_labels = _independent_hull_labels(grouped_sizes, grouped_losses)
    deterministic_exact = bool(np.array_equal(labels, deterministic_labels))
    inputs = np.column_stack(
        (_affine_normalize(grouped_sizes), _affine_normalize(grouped_losses))
    )
    return {
        "source_state_index": source_state_index,
        "trajectory_id": int(record["trajectory_id"]),
        "iteration": iteration,
        "history_count": history_count,
        "state_bin": _state_bin(history_count),
        "representative_count": len(representatives),
        "hull_size": len(hull),
        "positive_fraction": len(hull) / len(representatives),
        "collinear_case": collinear_case,
        "deterministic_exact": deterministic_exact,
        "inputs": inputs,
        "labels": labels,
        "representatives": representatives,
    }


def _build_trajectory(
    arguments: tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]
) -> list[dict[str, Any]]:
    records, trajectory = arguments
    return [
        _build_one(source_state_index, record, trajectory)
        for source_state_index, record in records
    ]


def _build_dataset(
    payload: dict[str, Any], indices: list[int], workers: int, label: str
) -> list[dict[str, Any]]:
    trajectories = ordinary_data._trajectory_map(payload)
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for source_state_index in indices:
        record = payload["states"][source_state_index]
        grouped.setdefault(int(record["trajectory_id"]), []).append(
            (source_state_index, record)
        )
    tasks = [
        (records, trajectories[trajectory_id])
        for trajectory_id, records in sorted(grouped.items())
    ]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.extend(_build_trajectory(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_build_trajectory, task) for task in tasks]
            for future in as_completed(futures):
                batch = future.result()
                results.extend(batch)
                if len(results) % 50 < len(batch) or len(results) == len(indices):
                    print(f"{label} features {len(results)}/{len(indices)}", flush=True)
    by_index = {int(record["source_state_index"]): record for record in results}
    return [by_index[index] for index in indices]


class InducedSetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(INDUCING_COUNT, HIDDEN_DIM) * 0.02)
        self.induce_attention = nn.MultiheadAttention(
            HIDDEN_DIM, HEAD_COUNT, batch_first=True
        )
        self.project_attention = nn.MultiheadAttention(
            HIDDEN_DIM, HEAD_COUNT, batch_first=True
        )
        self.induce_norm_1 = nn.LayerNorm(HIDDEN_DIM)
        self.induce_norm_2 = nn.LayerNorm(HIDDEN_DIM)
        self.project_norm_1 = nn.LayerNorm(HIDDEN_DIM)
        self.project_norm_2 = nn.LayerNorm(HIDDEN_DIM)
        self.induce_feedforward = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 2 * HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
        )
        self.project_feedforward = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 2 * HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(2 * HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        inducing = self.inducing.unsqueeze(0).expand(len(values), -1, -1)
        induced, _ = self.induce_attention(
            inducing,
            values,
            values,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        induced = self.induce_norm_1(inducing + induced)
        induced = self.induce_norm_2(induced + self.induce_feedforward(induced))
        projected, _ = self.project_attention(
            values, induced, induced, need_weights=False
        )
        projected = self.project_norm_1(values + projected)
        return self.project_norm_2(
            projected + self.project_feedforward(projected)
        )


class HullSetTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        )
        self.relational = InducedSetBlock()
        self.output = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, 1)
        )

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.input(inputs)
        relational = self.relational(encoded, padding_mask)
        return self.output(relational).squeeze(-1)


def _batches(
    records: list[dict[str, Any]], shuffle: bool, seed: int
) -> list[list[int]]:
    generator = np.random.default_rng(seed)
    indices = np.arange(len(records), dtype=np.int64)
    if shuffle:
        noise = generator.random(len(records))
        indices = np.asarray(
            sorted(indices, key=lambda index: (records[int(index)]["representative_count"], noise[int(index)])),
            dtype=np.int64,
        )
    else:
        indices = np.asarray(
            sorted(indices, key=lambda index: records[int(index)]["representative_count"]),
            dtype=np.int64,
        )
    batches: list[list[int]] = []
    current: list[int] = []
    maximum = 0
    for index_value in indices:
        index = int(index_value)
        candidate_maximum = max(maximum, int(records[index]["representative_count"]))
        if current and (
            len(current) >= MAX_BATCH_STATES
            or candidate_maximum * (len(current) + 1) > TOKEN_BUDGET
        ):
            batches.append(current)
            current = []
            maximum = 0
        current.append(index)
        maximum = max(maximum, int(records[index]["representative_count"]))
    if current:
        batches.append(current)
    if shuffle:
        generator.shuffle(batches)
    return batches


def _collate(
    records: list[dict[str, Any]], indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(int(records[index]["representative_count"]) for index in indices)
    inputs = np.zeros((len(indices), maximum, 2), dtype=np.float64)
    labels = np.zeros((len(indices), maximum), dtype=np.float64)
    padding = np.ones((len(indices), maximum), dtype=np.bool_)
    for row, index in enumerate(indices):
        count = int(records[index]["representative_count"])
        inputs[row, :count] = records[index]["inputs"]
        labels[row, :count] = records[index]["labels"]
        padding[row, :count] = False
    return (
        torch.as_tensor(inputs, dtype=torch.float64),
        torch.as_tensor(labels, dtype=torch.float64),
        torch.as_tensor(padding),
    )


def _state_counts(labels: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int]:
    true = labels.astype(bool)
    predicted = predictions.astype(bool)
    return (
        int(np.sum(true & predicted)),
        int(np.sum(~true & predicted)),
        int(np.sum(true & ~predicted)),
    )


def _metrics(
    records: list[dict[str, Any]], predictions: list[np.ndarray]
) -> dict[str, Any]:
    state_rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    for record, prediction in zip(records, predictions):
        tp, fp, fn = _state_counts(record["labels"], prediction)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        state_rows.append(
            {
                "source_state_index": record["source_state_index"],
                "trajectory_id": record["trajectory_id"],
                "iteration": record["iteration"],
                "history_count": record["history_count"],
                "state_bin": record["state_bin"],
                "representative_count": record["representative_count"],
                "teacher_hull_size": record["hull_size"],
                "predicted_hull_size": int(np.sum(prediction)),
                "collinear_case": record["collinear_case"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "exact": fp == 0 and fn == 0,
                "deterministic_exact": record["deterministic_exact"],
            }
        )
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "per_cell_precision": precision,
        "per_cell_recall": recall,
        "per_cell_f1": f1,
        "macro_state_precision": float(np.mean([row["precision"] for row in state_rows])),
        "macro_state_recall": float(np.mean([row["recall"] for row in state_rows])),
        "exact_states": sum(bool(row["exact"]) for row in state_rows),
        "exact_state_fraction": float(np.mean([row["exact"] for row in state_rows])),
        "average_false_positives_per_state": float(np.mean([row["fp"] for row in state_rows])),
        "average_false_negatives_per_state": float(np.mean([row["fn"] for row in state_rows])),
        "state_rows": state_rows,
    }


def _predict(
    model: HullSetTransformer, records: list[dict[str, Any]]
) -> tuple[list[np.ndarray], float]:
    model.eval()
    predictions: list[np.ndarray | None] = [None] * len(records)
    loss_sum = 0.0
    cell_count = 0
    with torch.no_grad():
        for indices in _batches(records, shuffle=False, seed=0):
            inputs, labels, padding = _collate(records, indices)
            logits = model(inputs, padding)
            valid = ~padding
            loss_sum += float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits[valid], labels[valid], reduction="sum"
                ).item()
            )
            cell_count += int(torch.sum(valid).item())
            masks = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
            for row, index in enumerate(indices):
                count = int(records[index]["representative_count"])
                predictions[index] = masks[row, :count]
    return [np.asarray(item) for item in predictions], loss_sum / cell_count


def _train(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> tuple[HullSetTransformer, list[dict[str, Any]], int, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = HullSetTransformer().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    positives = sum(int(record["hull_size"]) for record in train)
    total = sum(int(record["representative_count"]) for record in train)
    positive_weight = torch.tensor((total - positives) / positives, dtype=torch.float64)
    initial_predictions, initial_loss = _predict(model, validation)
    best_metrics = _metrics(validation, initial_predictions)
    best_metrics["validation_bce"] = initial_loss
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_cells = 0
        for indices in _batches(train, shuffle=True, seed=MODEL_SEED + epoch):
            inputs, labels, padding = _collate(train, indices)
            logits = model(inputs, padding)
            valid = ~padding
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[valid], labels[valid], pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            valid_count = int(torch.sum(valid).item())
            train_loss_sum += float(loss.item()) * valid_count
            train_cells += valid_count
        predictions, validation_loss = _predict(model, validation)
        metrics = _metrics(validation, predictions)
        metrics["validation_bce"] = validation_loss
        history.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "train_weighted_bce": train_loss_sum / train_cells,
                "validation_bce": validation_loss,
                "validation_per_cell_precision": metrics["per_cell_precision"],
                "validation_per_cell_recall": metrics["per_cell_recall"],
                "validation_per_cell_f1": metrics["per_cell_f1"],
                "validation_exact_states": metrics["exact_states"],
                "validation_exact_state_fraction": metrics["exact_state_fraction"],
            }
        )
        improved = (
            metrics["exact_states"] > best_metrics["exact_states"]
            or (
                metrics["exact_states"] == best_metrics["exact_states"]
                and metrics["per_cell_f1"] > best_metrics["per_cell_f1"] + 1.0e-10
            )
        )
        if improved:
            best_metrics = metrics
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch} precision={metrics['per_cell_precision']:.4f} "
            f"recall={metrics['per_cell_recall']:.4f} f1={metrics['per_cell_f1']:.4f} "
            f"exact={metrics['exact_states']}/{len(validation)}",
            flush=True,
        )
        if epoch >= 8 and stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    predictions, validation_loss = _predict(model, validation)
    final_metrics = _metrics(validation, predictions)
    final_metrics["validation_bce"] = validation_loss
    final_metrics["predictions"] = predictions
    return model, history, best_epoch, final_metrics


def _subset_metric_row(
    record_type: str,
    name: str,
    records: list[dict[str, Any]],
    predictions: list[np.ndarray],
) -> dict[str, Any]:
    metrics = _metrics(records, predictions)
    return {
        "record_type": record_type,
        "subset": name,
        "state_count": len(records),
        **{key: value for key, value in metrics.items() if key != "state_rows"},
    }


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {name: _sha256(path) for name, path in LOCKED_PATHS.items()}
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = _stratified_indices(
        payload["states"], validation=False, total=TRAIN_STATES, seed=DATA_SEED
    )
    validation_indices = _stratified_indices(
        payload["states"], validation=True, total=VALIDATION_STATES, seed=DATA_SEED + 1
    )
    train = _build_dataset(payload, train_indices, workers, "train")
    validation = _build_dataset(payload, validation_indices, workers, "validation")
    if not all(record["deterministic_exact"] for record in validation):
        raise RuntimeError("independent deterministic hull reconstruction failed")
    model, rows, best_epoch, final = _train(train, validation)
    del model
    predictions = final.pop("predictions")
    rows.append(
        {
            "record_type": "final_summary",
            "best_epoch": best_epoch,
            "train_states": len(train),
            "validation_states": len(validation),
            "train_representative_cells": sum(
                int(record["representative_count"]) for record in train
            ),
            "validation_representative_cells": sum(
                int(record["representative_count"]) for record in validation
            ),
            "validation_teacher_hull_cells": sum(
                int(record["hull_size"]) for record in validation
            ),
            "validation_bce": final["validation_bce"],
            "per_cell_precision": final["per_cell_precision"],
            "per_cell_recall": final["per_cell_recall"],
            "per_cell_f1": final["per_cell_f1"],
            "macro_state_precision": final["macro_state_precision"],
            "macro_state_recall": final["macro_state_recall"],
            "exact_hull_states": final["exact_states"],
            "exact_hull_state_fraction": final["exact_state_fraction"],
            "average_false_positives_per_state": final[
                "average_false_positives_per_state"
            ],
            "average_false_negatives_per_state": final[
                "average_false_negatives_per_state"
            ],
            "deterministic_sanity_exact_states": sum(
                bool(record["deterministic_exact"]) for record in validation
            ),
            "deterministic_sanity_fraction": float(
                np.mean([record["deterministic_exact"] for record in validation])
            ),
            "collinear_train_states": sum(
                bool(record["collinear_case"]) for record in train
            ),
            "collinear_validation_states": sum(
                bool(record["collinear_case"]) for record in validation
            ),
            "full_training_run": False,
            "closed_loop_search_run": False,
            "improvement_condition_labels_used": False,
        }
    )
    for state_bin, _, _ in STATE_BINS:
        indices = [
            index
            for index, record in enumerate(validation)
            if record["state_bin"] == state_bin
        ]
        rows.append(
            _subset_metric_row(
                "metrics_by_state_size",
                state_bin,
                [validation[index] for index in indices],
                [predictions[index] for index in indices],
            )
        )
    hull_bins = (
        ("hull_1_4", 1, 4),
        ("hull_5_8", 5, 8),
        ("hull_9_16", 9, 16),
        ("hull_17_plus", 17, math.inf),
    )
    for name, lower, upper in hull_bins:
        indices = [
            index
            for index, record in enumerate(validation)
            if lower <= int(record["hull_size"]) <= upper
        ]
        if indices:
            rows.append(
                _subset_metric_row(
                    "metrics_by_hull_size",
                    name,
                    [validation[index] for index in indices],
                    [predictions[index] for index in indices],
                )
            )
    collinear_indices = [
        index for index, record in enumerate(validation) if record["collinear_case"]
    ]
    if collinear_indices:
        rows.append(
            _subset_metric_row(
                "metrics_collinear_cases",
                "collinear_teacher_hull",
                [validation[index] for index in collinear_indices],
                [predictions[index] for index in collinear_indices],
            )
        )
    hashes_after = {name: _sha256(path) for name, path in LOCKED_PATHS.items()}
    rows.append(
        {
            "record_type": "integrity",
            **{
                f"{name}_before": hashes_before[name] for name in LOCKED_PATHS
            },
            **{f"{name}_after": hashes_after[name] for name in LOCKED_PATHS},
            **{
                f"{name}_unchanged": hashes_before[name] == hashes_after[name]
                for name in LOCKED_PATHS
            },
        }
    )
    errors = [
        row for row in final["state_rows"] if not bool(row["exact"])
    ]
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(
        ERRORS_PATH,
        errors,
        default_fields=(
            "source_state_index",
            "trajectory_id",
            "iteration",
            "history_count",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
