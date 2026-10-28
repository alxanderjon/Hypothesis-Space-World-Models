#!/usr/bin/env python3
"""Learn only the largest-selected-cell farthest-candidate operation."""

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

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_learned_global_search_imitation as imitation
import experiments.sho.run_stage1b_ordinary_cell_score_full as ordinary_full
import experiments.sho.run_stage1b_ordinary_cell_score_network as score_base
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary_data
import experiments.sho.run_stage1b_specialized_cell_filter as specialized
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    DOMAIN_WIDTH,
    LOWER,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/largest_cell_score")
DATASET_PATH = OUTPUT_DIR / "largest_cell_dataset.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "best_largest_cell_checkpoint.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
STATIC_ERRORS_PATH = OUTPUT_DIR / "largest_cell_errors.csv"
PER_TRAJECTORY_PATH = OUTPUT_DIR / "search_per_trajectory.csv"
SEARCH_TRACE_PATH = OUTPUT_DIR / "search_traces.pt"

ORDINARY_CHECKPOINT = ordinary_full.CHECKPOINT_PATH
ORDINARY_REFERENCE_TRACE = ordinary_full.SEARCH_TRACE_PATH
SELECTOR_PATH = specialized.FROZEN_SELECTOR_PATH
SOURCE_SELECTOR_PATH = specialized.SOURCE_SELECTOR

MODEL_SEED = SEED + 16_800_000
DATA_SEED = SEED + 16_900_000
HIDDEN_DIM = 32
LEARNING_RATE = 2.0e-3
RANKING_WEIGHT = 0.05
CANDIDATE_BATCH_SIZE = 65_536
FULL_CANDIDATE_BATCH_SIZE = 262_144
PAIR_BATCH_SIZE = 16_384
QUICK_TRAIN_CELLS = 3_000
QUICK_VALIDATION_CELLS = 750
QUICK_MAX_EPOCHS = 100
QUICK_PATIENCE = 12
FULL_MAX_EPOCHS = 600
FULL_PATIENCE = 30
MIN_DELTA = 1.0e-9
QUICK_TOP1_GATE = 0.99
QUICK_CORRELATION_GATE = 0.999
FULL_TOP1_GATE = 0.999
SCREEN_COUNT = 16
GATE = imitation.GATE
NORMAL_CAP = imitation.NORMAL_CAP
FULL_BOUNDS = specialized.FULL_BOUNDS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()

def _write_csv(
    path: Path, rows: list[dict[str, Any]], default_fields: tuple[str, ...] = ()
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = list(default_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class LargestCellScorer(nn.Module):
    """Small shared candidate scorer; no cell pooling or ordinary examples."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def _set_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second):
        return math.inf
    if len(first) == 0:
        return 0.0
    pairwise = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        max(np.max(np.min(pairwise, axis=0)), np.max(np.min(pairwise, axis=1)))
    )


def _largest_example_for_state(
    record: dict[str, Any], trajectory: dict[str, Any]
) -> tuple[dict[str, Any] | None, float]:
    history_count = int(record["history_count"])
    points = np.asarray(
        trajectory["evaluated_hypotheses_normalized"][:history_count],
        dtype=np.float64,
    )
    losses = np.asarray(
        trajectory["evaluated_prefix_losses"][:history_count], dtype=np.float64
    )
    generator_id = int(record["generator_id"])
    iteration = int(record["iteration"])
    data = edirect._cell_data(points, losses, generator_id, iteration)
    surface = edirect.GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    size_weight = edirect._normalize(selected_sizes)
    loo_weight = edirect._normalize(surface.loo_errors(data.selected))
    u_values = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    reconstructed: list[np.ndarray] = []
    largest_example: dict[str, Any] | None = None
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        indices = np.flatnonzero(data.owners == site_index)
        if len(indices) == 0:
            continue
        candidates = data.points[indices]
        distances = data.distances[indices]
        normalized_distance = edirect._normalize(distances)
        if selected_position == largest_position:
            winner = int(np.argmax(distances))
            scale = max(float(data.sizes[site_index]), 1.0e-12)
            normalized_offset = (candidates - points[site_index][None, :]) / scale
            features = np.column_stack(
                (normalized_offset, normalized_distance)
            ).astype(np.float32)
            ordered = np.sort(normalized_distance)
            largest_example = {
                "inputs": features,
                "count": len(candidates),
                "winner": winner,
                "cell_size": float(data.sizes[site_index]),
                "margin": (
                    math.inf
                    if len(ordered) == 1
                    else float(ordered[-1] - ordered[-2])
                ),
                "iteration": iteration,
                "site_index": site_index,
            }
        else:
            prediction = surface.predict(candidates)
            normalized_prediction = edirect._normalize(prediction)
            u_value = float(u_values[selected_position])
            utility = (
                u_value * normalized_distance
                - (1.0 - u_value) * normalized_prediction
            )
            winner = int(np.argmax(utility))
        proposal = candidates[winner]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            reconstructed.append(proposal)
    expected = np.asarray(record["teacher_curated_points"], dtype=np.float64)
    actual = np.asarray(reconstructed, dtype=np.float64).reshape(-1, 2)
    return largest_example, _set_distance(actual, expected)


def _extract_trajectory(
    arguments: tuple[int, list[dict[str, Any]], dict[str, Any]]
) -> dict[str, Any]:
    trajectory_id, records, trajectory = arguments
    inputs: list[np.ndarray] = []
    counts: list[int] = []
    winners: list[int] = []
    sizes: list[float] = []
    margins: list[float] = []
    iterations: list[int] = []
    site_indices: list[int] = []
    worst_mismatch = 0.0
    for record in records:
        example, mismatch = _largest_example_for_state(record, trajectory)
        if not np.isfinite(mismatch) or mismatch > 1.0e-12:
            raise RuntimeError(
                f"trajectory {trajectory_id} iteration {record['iteration']} mismatch={mismatch}"
            )
        worst_mismatch = max(worst_mismatch, mismatch)
        if example is None:
            continue
        inputs.append(example["inputs"])
        counts.append(int(example["count"]))
        winners.append(int(example["winner"]))
        sizes.append(float(example["cell_size"]))
        margins.append(float(example["margin"]))
        iterations.append(int(example["iteration"]))
        site_indices.append(int(example["site_index"]))
    return {
        "trajectory_id": trajectory_id,
        "inputs": np.concatenate(inputs) if inputs else np.empty((0, 3), np.float32),
        "counts": np.asarray(counts, dtype=np.int32),
        "winners": np.asarray(winners, dtype=np.int32),
        "cell_size": np.asarray(sizes, dtype=np.float32),
        "margin": np.asarray(margins, dtype=np.float32),
        "iteration": np.asarray(iterations, dtype=np.int32),
        "site_index": np.asarray(site_indices, dtype=np.int32),
        "worst_mismatch": worst_mismatch,
    }


def _pack(results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    results = sorted(results, key=lambda row: int(row["trajectory_id"]))
    counts = np.concatenate([row["counts"] for row in results])
    return {
        "inputs": np.concatenate([row["inputs"] for row in results]),
        "counts": counts,
        "offsets": np.concatenate(
            (np.asarray((0,), dtype=np.int64), np.cumsum(counts, dtype=np.int64))
        ),
        "winners": np.concatenate([row["winners"] for row in results]),
        "cell_size": np.concatenate([row["cell_size"] for row in results]),
        "margin": np.concatenate([row["margin"] for row in results]),
        "iteration": np.concatenate([row["iteration"] for row in results]),
        "site_index": np.concatenate([row["site_index"] for row in results]),
        "trajectory_id": np.concatenate(
            [
                np.full(len(row["counts"]), int(row["trajectory_id"]), np.int32)
                for row in results
            ]
        ),
        "worst_reconstruction_error": np.asarray(
            max(row["worst_mismatch"] for row in results)
        ),
    }


def _build_dataset(workers: int) -> dict[str, Any]:
    payload = torch.load(filter_data.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary_data._trajectory_map(payload)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in payload["states"]:
        if record["candidate_type"] == "cell":
            grouped.setdefault(int(record["trajectory_id"]), []).append(record)
    tasks = [
        (
            trajectory_id,
            sorted(grouped[trajectory_id], key=lambda row: int(row["history_count"])),
            trajectories[trajectory_id],
        )
        for trajectory_id in sorted(grouped)
    ]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_extract_trajectory(task))
            print(f"largest dataset {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_trajectory, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(f"largest dataset {len(results)}/{len(tasks)}", flush=True)
    dataset = {
        "config": {
            "largest_cells_only": True,
            "ordinary_examples": 0,
            "features": "normalized_offset_a|normalized_offset_b|normalized_distance",
            "target": "normalized_distance",
            "source_sha256": _sha256(filter_data.RAW_TRACE_PATH),
        },
        "train": _pack([row for row in results if row["trajectory_id"] % 5 != 0]),
        "validation": _pack([row for row in results if row["trajectory_id"] % 5 == 0]),
    }
    torch.save(dataset, DATASET_PATH)
    return dataset


def _cell_subset(split: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    blocks = [
        split["inputs"][
            int(split["offsets"][index]) : int(split["offsets"][index + 1])
        ]
        for index in indices
    ]
    counts = split["counts"][indices]
    return {
        "inputs": np.concatenate(blocks),
        "counts": counts,
        "offsets": np.concatenate(
            (np.asarray((0,), np.int64), np.cumsum(counts, dtype=np.int64))
        ),
        "winners": split["winners"][indices],
        "cell_size": split["cell_size"][indices],
        "margin": split["margin"][indices],
        "iteration": split["iteration"][indices],
        "site_index": split["site_index"][indices],
        "trajectory_id": split["trajectory_id"][indices],
    }


def _pair_indices(split: dict[str, np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []
    for cell_index in range(len(split["counts"])):
        start = int(split["offsets"][cell_index])
        stop = int(split["offsets"][cell_index + 1])
        if stop - start < 2:
            continue
        winner = start + int(split["winners"][cell_index])
        alternatives = generator.choice(
            stop - start - 1, size=min(4, stop - start - 1), replace=False
        )
        alternatives = alternatives + (alternatives >= int(split["winners"][cell_index]))
        first.extend([winner] * len(alternatives))
        second.extend((start + alternatives).tolist())
    return np.asarray(first, np.int64), np.asarray(second, np.int64)


def _regression_metrics(model: LargestCellScorer, inputs: np.ndarray) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), CANDIDATE_BATCH_SIZE):
            batch = torch.as_tensor(
                inputs[start : start + CANDIDATE_BATCH_SIZE], dtype=torch.float64
            )
            predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions)
    target = inputs[:, 2].astype(np.float64)
    error = prediction - target
    return {
        "score_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "score_mae": float(np.mean(np.abs(error))),
        "score_correlation": float(np.corrcoef(prediction, target)[0, 1]),
        "score_max_abs_error": float(np.max(np.abs(error))),
    }


def _validation_metrics(
    model: LargestCellScorer, split: dict[str, np.ndarray]
) -> dict[str, float]:
    prediction = _predict(model, split["inputs"])
    target = split["inputs"][:, 2].astype(np.float64)
    error = prediction - target
    winners, _, _ = _winner_metrics(split, prediction)
    return {
        "score_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "score_mae": float(np.mean(np.abs(error))),
        "score_correlation": float(np.corrcoef(prediction, target)[0, 1]),
        "score_max_abs_error": float(np.max(np.abs(error))),
        **winners,
    }


def _train_phase(
    model: LargestCellScorer,
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    phase: str,
    max_epochs: int,
    patience: int,
    batch_size: int = CANDIDATE_BATCH_SIZE,
) -> tuple[LargestCellScorer, list[dict[str, Any]], int, dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pair_first, pair_second = _pair_indices(train, MODEL_SEED + len(phase))
    generator = np.random.default_rng(MODEL_SEED + 100 + len(phase))
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = _validation_metrics(model, validation)
    best_rmse = best_metrics["score_rmse"]
    best_top1 = best_metrics["winner_top1"]
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        order = generator.permutation(len(train["inputs"]))
        model.train()
        mse_sum = rank_sum = 0.0
        updates = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_np = train["inputs"][indices]
            batch = torch.as_tensor(batch_np, dtype=torch.float64)
            target = batch[:, 2]
            pair_indices = generator.integers(
                0, len(pair_first), size=min(PAIR_BATCH_SIZE, len(pair_first))
            )
            first = torch.as_tensor(train["inputs"][pair_first[pair_indices]], dtype=torch.float64)
            second = torch.as_tensor(train["inputs"][pair_second[pair_indices]], dtype=torch.float64)
            true_margin = torch.clamp(first[:, 2] - second[:, 2], max=0.05)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            mse = nn.functional.mse_loss(prediction, target)
            ranking = torch.relu(true_margin - (model(first) - model(second))).mean()
            loss = mse + RANKING_WEIGHT * ranking
            loss.backward()
            optimizer.step()
            mse_sum += float(mse.item())
            rank_sum += float(ranking.item())
            updates += 1
        metrics = _validation_metrics(model, validation)
        history.append(
            {
                "record_type": "epoch",
                "phase": phase,
                "epoch": epoch,
                "train_mse_batch_mean": mse_sum / updates,
                "train_ranking_loss_batch_mean": rank_sum / updates,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print(f"{phase} epoch={epoch} rmse={metrics['score_rmse']:.3e}", flush=True)
        top1_improved = metrics["winner_top1"] > best_top1 + 1.0e-15
        rmse_tie_break = (
            metrics["winner_top1"] >= best_top1 - 1.0e-15
            and metrics["score_rmse"] < best_rmse - MIN_DELTA
        )
        if top1_improved or rmse_tie_break:
            best_rmse = metrics["score_rmse"]
            best_top1 = metrics["winner_top1"]
            best_metrics = metrics
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0 if top1_improved else stale + 1
        else:
            stale += 1
        if epoch >= 15 and stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, best_metrics


def _predict(model: LargestCellScorer, inputs: np.ndarray) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), CANDIDATE_BATCH_SIZE):
            batch = torch.as_tensor(
                inputs[start : start + CANDIDATE_BATCH_SIZE], dtype=torch.float64
            )
            result.append(model(batch).cpu().numpy())
    return np.concatenate(result)


def _winner_metrics(
    split: dict[str, np.ndarray], prediction: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    hits = np.zeros((len(split["counts"]), 3), dtype=np.bool_)
    predicted_winner = np.zeros(len(split["counts"]), dtype=np.int32)
    for cell_index in range(len(split["counts"])):
        start = int(split["offsets"][cell_index])
        stop = int(split["offsets"][cell_index + 1])
        order = np.argsort(prediction[start:stop])[::-1]
        teacher = int(split["winners"][cell_index])
        predicted_winner[cell_index] = int(order[0])
        for k in range(3):
            hits[cell_index, k] = teacher in order[: min(k + 1, len(order))]
    return (
        {
            "winner_top1": float(hits[:, 0].mean()),
            "winner_top2": float(hits[:, 1].mean()),
            "winner_top3": float(hits[:, 2].mean()),
            "deterministic_winner_accuracy": float(
                np.mean(
                    [
                        int(
                            np.argmax(
                                split["inputs"][
                                    int(split["offsets"][index]) : int(split["offsets"][index + 1]),
                                    2,
                                ]
                            )
                        )
                        == int(split["winners"][index])
                        for index in range(len(split["counts"]))
                    ]
                )
            ),
        },
        hits,
        predicted_winner,
    )


def _diagnostic_rows(
    train: dict[str, np.ndarray], validation: dict[str, np.ndarray], hits: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, train_values, validation_values in (
        ("cell_size_quartile", train["cell_size"], validation["cell_size"]),
        ("candidate_count_quartile", train["counts"], validation["counts"]),
    ):
        edges = np.quantile(train_values, (0.25, 0.5, 0.75))
        bins = np.searchsorted(edges, validation_values, side="right")
        for index in range(4):
            selected = np.flatnonzero(bins == index)
            rows.append(
                {
                    "record_type": kind,
                    "subset": f"Q{index + 1}",
                    "count": len(selected),
                    "winner_top1": float(hits[selected, 0].mean()),
                    "winner_top2": float(hits[selected, 1].mean()),
                    "winner_top3": float(hits[selected, 2].mean()),
                    "lower": -math.inf if index == 0 else edges[index - 1],
                    "upper": math.inf if index == 3 else edges[index],
                }
            )
    return rows


def _static_error_rows(
    split: dict[str, np.ndarray], prediction: np.ndarray, winners: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, predicted in enumerate(winners):
        teacher = int(split["winners"][index])
        if int(predicted) == teacher:
            continue
        start = int(split["offsets"][index])
        rows.append(
            {
                "cell_index": index,
                "trajectory_id": int(split["trajectory_id"][index]),
                "iteration": int(split["iteration"][index]),
                "site_index": int(split["site_index"][index]),
                "candidate_count": int(split["counts"][index]),
                "cell_size": float(split["cell_size"][index]),
                "teacher_margin": float(split["margin"][index]),
                "teacher_winner": teacher,
                "learned_winner": int(predicted),
                "teacher_true_distance": float(split["inputs"][start + teacher, 2]),
                "teacher_predicted_score": float(prediction[start + teacher]),
                "learned_true_distance": float(split["inputs"][start + int(predicted), 2]),
                "learned_predicted_score": float(prediction[start + int(predicted)]),
            }
        )
    return rows


def _save_checkpoint(
    model: LargestCellScorer,
    epoch: int,
    metrics: dict[str, float],
    ordinary_hash: str,
    selector_hash: str,
    assembled_preserved: bool | None,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture": "LargestCellScorer(3,32,32,1; SiLU)",
            "epoch": epoch,
            "validation_metrics": metrics,
            "ordinary_checkpoint_sha256": ordinary_hash,
            "selector_checkpoint_sha256": selector_hash,
            "assembled_reliability_preserved": assembled_preserved,
            "largest_cells_only": True,
        },
        CHECKPOINT_PATH,
    )


def _joint_cell_proposals(
    ordinary_model: score_base.ScoreNetwork,
    largest_model: LargestCellScorer,
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
) -> tuple[np.ndarray, dict[str, int]]:
    data = edirect._cell_data(points, losses, generator_id, iteration)
    surface = edirect.GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    size_weight = edirect._normalize(selected_sizes)
    loo_weight = edirect._normalize(surface.loo_errors(data.selected))
    u_values = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    proposals: list[np.ndarray] = []
    ordinary_cells = ordinary_disagreements = 0
    largest_cells = largest_disagreements = 0
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        mask = data.owners == site_index
        candidates = data.points[mask]
        distances = data.distances[mask]
        if len(candidates) == 0:
            continue
        normalized_distance = edirect._normalize(distances)
        if selected_position == largest_position:
            largest_cells += 1
            scale = max(float(data.sizes[site_index]), 1.0e-12)
            inputs = np.column_stack(
                ((candidates - points[site_index][None, :]) / scale, normalized_distance)
            )
            with torch.no_grad():
                scores = largest_model(torch.as_tensor(inputs, dtype=torch.float64)).cpu().numpy()
            winner = int(np.argmax(scores))
            largest_disagreements += int(winner != int(np.argmax(distances)))
        else:
            ordinary_cells += 1
            prediction = surface.predict(candidates)
            normalized_prediction = edirect._normalize(prediction)
            u_value = float(u_values[selected_position])
            inputs = np.column_stack(
                (normalized_distance, normalized_prediction, np.full(len(candidates), u_value))
            )
            with torch.no_grad():
                scores = ordinary_model(torch.as_tensor(inputs, dtype=torch.float64)).cpu().numpy()
            winner = int(np.argmax(scores))
            deterministic = u_value * normalized_distance - (1.0 - u_value) * normalized_prediction
            ordinary_disagreements += int(winner != int(np.argmax(deterministic)))
        proposal = candidates[winner]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            proposals.append(proposal)
    if not proposals:
        nearest = edirect.cKDTree(points).query(data.points, k=1)[0]
        for index in np.argsort(nearest)[::-1]:
            if nearest[index] > 1.0e-12:
                proposals.append(data.points[index])
                break
    return np.asarray(proposals, np.float64), {
        "ordinary_cells": ordinary_cells,
        "ordinary_winner_disagreements": ordinary_disagreements,
        "largest_cells": largest_cells,
        "largest_winner_disagreements": largest_disagreements,
    }


def _run_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    ordinary_model: score_base.ScoreNetwork,
    largest_model: LargestCellScorer,
    selector_model: imitation.CandidateRanker,
) -> tuple[imitation.ModeRun, dict[str, Any]]:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), np.float64)
    losses = np.empty(0, np.float64)
    evaluations = iteration = 0
    totals = {
        "ordinary_cells": 0,
        "ordinary_winner_disagreements": 0,
        "largest_cells": 0,
        "largest_winner_disagreements": 0,
    }
    first_largest_disagreement: int | None = None
    events: list[dict[str, Any]] = []
    pending = edirect._initial_design().copy()
    while len(pending) and evaluations < NORMAL_CAP and (
        not len(losses) or float(np.min(losses)) > GATE
    ):
        if len(points):
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, pending, "initial_design", global_features
            )
        else:
            choice = 0
        proposal = pending[int(choice)]
        loss = float(edirect._prefix_losses_numpy((LOWER + DOMAIN_WIDTH * proposal)[None, :], position)[0])
        points = np.concatenate((points, proposal[None, :]))
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1
    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        pending, info = _joint_cell_proposals(
            ordinary_model, largest_model, points, losses, generator_id, iteration
        )
        for key in totals:
            totals[key] += int(info[key])
        if info["largest_winner_disagreements"]:
            if first_largest_disagreement is None:
                first_largest_disagreement = evaluations
            events.append({"iteration": iteration, "evaluation": evaluations, **info})
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, pending, "cell", global_features
            )
            proposal = pending[int(choice)]
            loss = float(edirect._prefix_losses_numpy((LOWER + DOMAIN_WIDTH * proposal)[None, :], position)[0])
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            loss = float(edirect._prefix_losses_numpy((LOWER + DOMAIN_WIDTH * greedy)[None, :], position)[0])
            points = np.concatenate((points, greedy[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1
    return (
        imitation.ModeRun(
            mode="normal_learned_ordinary_and_largest",
            bounds=FULL_BOUNDS.copy(),
            points=points,
            losses=losses,
            new_evaluations=evaluations,
            success=float(np.min(losses)) <= GATE,
            decisions=events,
        ),
        {**totals, "first_largest_disagreement_evaluation": first_largest_disagreement},
    )


_WORKER_ORDINARY: score_base.ScoreNetwork | None = None
_WORKER_LARGEST: LargestCellScorer | None = None
_WORKER_SELECTOR: imitation.CandidateRanker | None = None


def _load_models() -> tuple[score_base.ScoreNetwork, LargestCellScorer, imitation.CandidateRanker]:
    global _WORKER_ORDINARY, _WORKER_LARGEST, _WORKER_SELECTOR
    if _WORKER_ORDINARY is None:
        checkpoint = torch.load(ORDINARY_CHECKPOINT, map_location="cpu", weights_only=False)
        _WORKER_ORDINARY = score_base.ScoreNetwork().to(torch.float64)
        _WORKER_ORDINARY.load_state_dict(checkpoint["state_dict"])
        _WORKER_ORDINARY.eval()
    if _WORKER_LARGEST is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _WORKER_LARGEST = LargestCellScorer().to(torch.float64)
        _WORKER_LARGEST.load_state_dict(checkpoint["state_dict"])
        _WORKER_LARGEST.eval()
    if _WORKER_SELECTOR is None:
        checkpoint = torch.load(SELECTOR_PATH, map_location="cpu", weights_only=False)
        _WORKER_SELECTOR = imitation.CandidateRanker().to(torch.float32)
        _WORKER_SELECTOR.load_state_dict(checkpoint["state_dict"])
        _WORKER_SELECTOR.eval()
    return _WORKER_ORDINARY, _WORKER_LARGEST, _WORKER_SELECTOR


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    ordinary_model, largest_model, selector_model = _load_models()
    normal, diagnostics = _run_normal(
        split, trajectory_id, position, ordinary_model, largest_model, selector_model
    )
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    if not normal.success:
        boundary_run, _ = imitation._run_boundary_rescue(
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
        "normal_success": normal.success,
        "success": any(mode.success for mode in modes),
        "normal_evaluations": normal.new_evaluations,
        "total_evaluations": total,
        "final_loss": final.best_loss,
        "boundary_invoked": boundary_run is not None,
        "nested_invoked": nested_run is not None,
        **diagnostics,
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
            print(f"largest search {split} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 4 == 0 or len(results) == len(tasks):
                    print(f"largest search {split} {len(results)}/{len(tasks)}", flush=True)
    return sorted(results, key=lambda row: int(row["trajectory_id"]))


def _reference_fields(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "normal_success": bool(run["normal_success"]),
        "success": bool(run["success"]),
        "normal_evaluations": int(run["normal_evaluations"]),
        "total_evaluations": int(run["total_evaluations"]),
        "boundary_invoked": bool(run["boundary_invoked"]),
        "nested_invoked": bool(run["nested_invoked"]),
    }


def _comparison(
    learned: list[dict[str, Any]], reference: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    by_id = {int(row["trajectory_id"]): row for row in reference}
    rows: list[dict[str, Any]] = []
    for run in learned:
        teacher = _reference_fields(by_id[int(run["trajectory_id"])])
        normal_agreement = bool(run["normal_success"]) == teacher["normal_success"]
        final_agreement = bool(run["success"]) == teacher["success"]
        rows.append(
            {
                "phase": phase,
                "split": run["split"],
                "trajectory_id": run["trajectory_id"],
                "learned_normal_success": run["normal_success"],
                "reference_normal_success": teacher["normal_success"],
                "normal_outcome_agreement": normal_agreement,
                "learned_final_success": run["success"],
                "reference_final_success": teacher["success"],
                "final_outcome_agreement": final_agreement,
                "learned_total_evaluations": run["total_evaluations"],
                "reference_total_evaluations": teacher["total_evaluations"],
                "ordinary_winner_disagreements": run["ordinary_winner_disagreements"],
                "largest_winner_disagreements": run["largest_winner_disagreements"],
                "first_largest_disagreement_evaluation": run[
                    "first_largest_disagreement_evaluation"
                ],
                "outcome_difference_has_largest_disagreement": (
                    (not normal_agreement or not final_agreement)
                    and int(run["largest_winner_disagreements"]) > 0
                ),
            }
        )
    return rows


def _search_summary(
    split: str,
    method: str,
    runs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    successes = [run for run in runs if bool(run["success"])]
    row: dict[str, Any] = {
        "record_type": "search_summary",
        "split": split,
        "method": method,
        "trajectories": len(runs),
        "normal_success_count": sum(bool(run["normal_success"]) for run in runs),
        "final_success_count": len(successes),
        "median_total_evaluations_success": float(
            np.median([run["total_evaluations"] for run in successes])
        ),
        "mean_total_evaluations_success": float(
            np.mean([run["total_evaluations"] for run in successes])
        ),
        "worst_total_evaluations_success": max(run["total_evaluations"] for run in successes),
        "rescue_invocations": sum(bool(run["boundary_invoked"]) for run in runs),
    }
    if method.startswith("learned"):
        row["ordinary_winner_disagreements"] = sum(
            int(run["ordinary_winner_disagreements"]) for run in runs
        )
        row["largest_winner_disagreements"] = sum(
            int(run["largest_winner_disagreements"]) for run in runs
        )
    if comparisons is not None:
        row["normal_outcome_agreement"] = float(
            np.mean([item["normal_outcome_agreement"] for item in comparisons])
        )
        row["final_outcome_agreement"] = float(
            np.mean([item["final_outcome_agreement"] for item in comparisons])
        )
    return row


def run(workers: int, rebuild_dataset: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordinary_hash_before = _sha256(ORDINARY_CHECKPOINT)
    selector_hash_before = _sha256(SELECTOR_PATH)
    if selector_hash_before != _sha256(SOURCE_SELECTOR_PATH):
        raise RuntimeError("frozen selector differs from source")
    dataset = (
        _build_dataset(workers)
        if rebuild_dataset or not DATASET_PATH.exists()
        else torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    )
    generator = np.random.default_rng(DATA_SEED)
    train_indices = generator.choice(
        len(dataset["train"]["counts"]), QUICK_TRAIN_CELLS, replace=False
    )
    validation_indices = generator.choice(
        len(dataset["validation"]["counts"]), QUICK_VALIDATION_CELLS, replace=False
    )
    quick_train = _cell_subset(dataset["train"], train_indices)
    quick_validation = _cell_subset(dataset["validation"], validation_indices)
    torch.manual_seed(MODEL_SEED)
    model = LargestCellScorer().to(torch.float64)
    model, quick_history, quick_epoch, quick_regression = _train_phase(
        model, quick_train, quick_validation, "quick", QUICK_MAX_EPOCHS, QUICK_PATIENCE
    )
    quick_prediction = _predict(model, quick_validation["inputs"])
    quick_winner, quick_hits, _ = _winner_metrics(quick_validation, quick_prediction)
    quick_metrics = {**quick_regression, **quick_winner}
    quick_gate = bool(
        quick_metrics["winner_top1"] >= QUICK_TOP1_GATE
        and quick_metrics["score_correlation"] >= QUICK_CORRELATION_GATE
        and quick_metrics["deterministic_winner_accuracy"] == 1.0
    )
    rows: list[dict[str, Any]] = list(quick_history)
    rows.append(
        {
            "record_type": "quick_screen",
            "best_epoch": quick_epoch,
            **quick_metrics,
            "train_cells": QUICK_TRAIN_CELLS,
            "validation_cells": QUICK_VALIDATION_CELLS,
            "passed": quick_gate,
        }
    )
    rows.extend(_diagnostic_rows(quick_train, quick_validation, quick_hits))
    if not quick_gate:
        _write_csv(SUMMARY_PATH, rows)
        _write_csv(STATIC_ERRORS_PATH, [], default_fields=("cell_index",))
        _write_csv(PER_TRAJECTORY_PATH, [], default_fields=("phase", "split", "trajectory_id"))
        print("largest-cell quick screen failed; stopped", flush=True)
        return

    model, full_history, full_epoch, full_regression = _train_phase(
        model,
        dataset["train"],
        dataset["validation"],
        "full",
        FULL_MAX_EPOCHS,
        FULL_PATIENCE,
        FULL_CANDIDATE_BATCH_SIZE,
    )
    rows.extend(full_history)
    prediction = _predict(model, dataset["validation"]["inputs"])
    full_winner, full_hits, winners = _winner_metrics(dataset["validation"], prediction)
    full_metrics = {**full_regression, **full_winner}
    full_gate = bool(
        full_metrics["winner_top1"] >= FULL_TOP1_GATE
        and full_metrics["deterministic_winner_accuracy"] == 1.0
    )
    rows.append(
        {
            "record_type": "full_static",
            "best_epoch": full_epoch,
            **full_metrics,
            "train_cells": len(dataset["train"]["counts"]),
            "validation_cells": len(dataset["validation"]["counts"]),
            "passed": full_gate,
        }
    )
    rows.extend(_diagnostic_rows(dataset["train"], dataset["validation"], full_hits))
    _write_csv(
        STATIC_ERRORS_PATH,
        _static_error_rows(dataset["validation"], prediction, winners),
        default_fields=("cell_index", "trajectory_id", "iteration", "site_index"),
    )
    _save_checkpoint(
        model, full_epoch, full_metrics, ordinary_hash_before, selector_hash_before, None
    )
    if not full_gate:
        _write_csv(SUMMARY_PATH, rows)
        _write_csv(PER_TRAJECTORY_PATH, [], default_fields=("phase", "split", "trajectory_id"))
        print("largest-cell full static gate failed; stopped", flush=True)
        return

    reference_payload = torch.load(
        ORDINARY_REFERENCE_TRACE, map_location="cpu", weights_only=False
    )
    validation_reference = reference_payload["full_validation"]
    test_reference = reference_payload["full_test"]
    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    test_positions, _, _ = _make_exact_labeled_split(
        imitation.TEST_COUNT, imitation.TEST_SEED
    )
    validation_np = validation_positions.to(dtype=DTYPE).cpu().numpy()
    test_np = test_positions.to(dtype=DTYPE).cpu().numpy()
    screen = _run_split("validation", validation_np[:SCREEN_COUNT], workers)
    screen_reference = validation_reference[:SCREEN_COUNT]
    screen_comparison = _comparison(screen, screen_reference, "screen")
    reference_normal = sum(bool(run["normal_success"]) for run in screen_reference)
    reference_final = sum(bool(run["success"]) for run in screen_reference)
    screen_normal = sum(bool(run["normal_success"]) for run in screen)
    screen_final = sum(bool(run["success"]) for run in screen)
    screen_promising = bool(
        screen_final >= reference_final
        and screen_normal >= reference_normal - 1
        and float(np.mean([row["final_outcome_agreement"] for row in screen_comparison])) >= 0.875
    )
    rows.extend(
        (
            _search_summary("validation_screen", "reference_learned_ordinary", screen_reference, None),
            _search_summary("validation_screen", "learned_largest", screen, screen_comparison),
            {
                "record_type": "screen_gate",
                "passed": screen_promising,
                "screen_normal_success": screen_normal,
                "reference_normal_success": reference_normal,
                "screen_final_success": screen_final,
                "reference_final_success": reference_final,
            },
        )
    )
    comparisons = list(screen_comparison)
    trace: dict[str, Any] = {
        "screen": screen,
        "screen_reference": screen_reference,
        "full_validation": [],
        "full_test": [],
    }
    preserved = False
    if screen_promising:
        validation_runs = _run_split("validation", validation_np, workers)
        test_runs = _run_split("test", test_np, workers)
        validation_comparison = _comparison(validation_runs, validation_reference, "full")
        test_comparison = _comparison(test_runs, test_reference, "full")
        comparisons.extend(validation_comparison)
        comparisons.extend(test_comparison)
        rows.extend(
            (
                _search_summary("validation", "reference_learned_ordinary", validation_reference, None),
                _search_summary("validation", "learned_largest", validation_runs, validation_comparison),
                _search_summary("test", "reference_learned_ordinary", test_reference, None),
                _search_summary("test", "learned_largest", test_runs, test_comparison),
            )
        )
        preserved = bool(
            sum(bool(run["success"]) for run in validation_runs)
            >= sum(bool(run["success"]) for run in validation_reference)
            and sum(bool(run["success"]) for run in test_runs)
            >= sum(bool(run["success"]) for run in test_reference)
        )
        trace.update(
            {
                "full_validation": validation_runs,
                "full_validation_reference": validation_reference,
                "full_test": test_runs,
                "full_test_reference": test_reference,
            }
        )
    ordinary_hash_after = _sha256(ORDINARY_CHECKPOINT)
    selector_hash_after = _sha256(SELECTOR_PATH)
    rows.append(
        {
            "record_type": "integrity_and_freeze",
            "assembled_reliability_preserved": preserved,
            "ordinary_checkpoint_before": ordinary_hash_before,
            "ordinary_checkpoint_after": ordinary_hash_after,
            "ordinary_checkpoint_unchanged": ordinary_hash_before == ordinary_hash_after,
            "selector_checkpoint_before": selector_hash_before,
            "selector_checkpoint_after": selector_hash_after,
            "selector_checkpoint_unchanged": selector_hash_before == selector_hash_after,
        }
    )
    _save_checkpoint(
        model, full_epoch, full_metrics, ordinary_hash_before, selector_hash_before, preserved
    )
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(
        PER_TRAJECTORY_PATH,
        comparisons,
        default_fields=("phase", "split", "trajectory_id"),
    )
    torch.save(trace, SEARCH_TRACE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--rebuild-dataset", action="store_true")
    arguments = parser.parse_args()
    run(arguments.workers, arguments.rebuild_dataset)


if __name__ == "__main__":
    main()
