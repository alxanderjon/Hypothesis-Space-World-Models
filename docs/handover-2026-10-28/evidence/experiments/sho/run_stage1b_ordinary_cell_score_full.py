#!/usr/bin/env python3
"""Full ordinary-cell score regression and gated assembled-search test.

Only the ordinary selected-cell winner is learned.  Candidate generation,
Voronoi ownership, potentially-optimal cell selection, largest-cell behavior,
greedy proposals, the existing final selector, and both rescue stages remain
unchanged.
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_learned_global_search_imitation as imitation
import experiments.sho.run_stage1b_ordinary_cell_score_network as quickscore
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary
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

OUTPUT_DIR = Path("outputs/sho/stage1B/ordinary_cell_score_full")
DATASET_PATH = OUTPUT_DIR / "ordinary_cell_full_dataset.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "best_score_network_checkpoint.pt"
SUMMARY_PATH = OUTPUT_DIR / "summary.csv"
STATIC_ERRORS_PATH = OUTPUT_DIR / "static_winner_errors.csv"
PER_TRAJECTORY_PATH = OUTPUT_DIR / "search_per_trajectory.csv"
SEARCH_TRACE_PATH = OUTPUT_DIR / "search_traces.pt"

RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
SELECTOR_PATH = specialized.FROZEN_SELECTOR_PATH
SOURCE_SELECTOR_PATH = specialized.SOURCE_SELECTOR
TEACHER_SEARCH_PATH = imitation.SEARCH_TRACE_PATH

MODEL_SEED = quickscore.MODEL_SEED
HIDDEN_DIM = quickscore.HIDDEN_DIM
LEARNING_RATE = quickscore.LEARNING_RATE
RANKING_WEIGHT = quickscore.RANKING_WEIGHT
MAX_EPOCHS = 600
PATIENCE = 30
MIN_EPOCHS = 30
MIN_DELTA = 1.0e-8
TRAIN_CANDIDATES_PER_EPOCH = 1_000_000
CANDIDATE_BATCH_SIZE = quickscore.CANDIDATE_BATCH_SIZE
PAIR_BATCH_SIZE = quickscore.PAIR_BATCH_SIZE
PAIR_COUNT_PER_CELL = 4
VALIDATION_BATCH_SIZE = 262_144
STATIC_TOP1_GATE = 0.95
STATIC_TOP3_GATE = 0.99
SCREEN_COUNT = 16
SCREEN_FINAL_SUCCESS_MIN = 15
SCREEN_OUTCOME_AGREEMENT_MIN = 0.875
NORMAL_CAP = imitation.NORMAL_CAP
GATE = imitation.GATE
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


def _true_scores(inputs: np.ndarray) -> np.ndarray:
    inputs64 = inputs.astype(np.float64, copy=False)
    return inputs64[:, 2] * inputs64[:, 0] - (
        1.0 - inputs64[:, 2]
    ) * inputs64[:, 1]


def _extract_trajectory(
    arguments: tuple[int, list[dict[str, Any]], dict[str, Any]]
) -> dict[str, Any]:
    trajectory_id, records, trajectory = arguments
    input_blocks: list[np.ndarray] = []
    counts: list[int] = []
    winners: list[int] = []
    u_values: list[float] = []
    cell_sizes: list[float] = []
    margins: list[float] = []
    iterations: list[int] = []
    site_indices: list[int] = []
    worst_reconstruction_error = 0.0
    for record in records:
        examples, mismatch = ordinary._ordinary_examples_for_state(record, trajectory)
        if not np.isfinite(mismatch) or mismatch > 1.0e-12:
            raise RuntimeError(
                f"trajectory {trajectory_id} state {record['iteration']} mismatch={mismatch}"
            )
        worst_reconstruction_error = max(worst_reconstruction_error, mismatch)
        for example in examples:
            input_blocks.append(example["candidate_inputs"].astype(np.float32))
            counts.append(int(example["candidate_count"]))
            winners.append(int(example["winner_index"]))
            u_values.append(float(example["u"]))
            cell_sizes.append(float(example["cell_size"]))
            margins.append(float(example["teacher_margin"]))
            iterations.append(int(example["iteration"]))
            site_indices.append(int(example["site_index"]))
    return {
        "trajectory_id": trajectory_id,
        "inputs": (
            np.concatenate(input_blocks, axis=0)
            if input_blocks
            else np.empty((0, 3), dtype=np.float32)
        ),
        "counts": np.asarray(counts, dtype=np.int32),
        "winners": np.asarray(winners, dtype=np.int32),
        "u": np.asarray(u_values, dtype=np.float32),
        "cell_size": np.asarray(cell_sizes, dtype=np.float32),
        "margin": np.asarray(margins, dtype=np.float32),
        "iteration": np.asarray(iterations, dtype=np.int32),
        "site_index": np.asarray(site_indices, dtype=np.int32),
        "worst_reconstruction_error": worst_reconstruction_error,
    }


def _pack_split(results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    results = sorted(results, key=lambda row: int(row["trajectory_id"]))
    inputs = np.concatenate([row["inputs"] for row in results], axis=0)
    counts = np.concatenate([row["counts"] for row in results])
    cell_count = len(counts)
    offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    trajectory_id = np.concatenate(
        [
            np.full(len(row["counts"]), int(row["trajectory_id"]), dtype=np.int32)
            for row in results
        ]
    )
    return {
        "inputs": inputs,
        "counts": counts,
        "offsets": offsets,
        "winners": np.concatenate([row["winners"] for row in results]),
        "u": np.concatenate([row["u"] for row in results]),
        "cell_size": np.concatenate([row["cell_size"] for row in results]),
        "margin": np.concatenate([row["margin"] for row in results]),
        "iteration": np.concatenate([row["iteration"] for row in results]),
        "site_index": np.concatenate([row["site_index"] for row in results]),
        "trajectory_id": trajectory_id,
        "cell_count": np.asarray(cell_count, dtype=np.int64),
        "candidate_count": np.asarray(len(inputs), dtype=np.int64),
        "worst_reconstruction_error": np.asarray(
            max(row["worst_reconstruction_error"] for row in results),
            dtype=np.float64,
        ),
    }


def _build_dataset(workers: int) -> dict[str, Any]:
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    trajectories = ordinary._trajectory_map(payload)
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
            print(f"dataset trajectories {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_extract_trajectory, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 8 == 0 or len(results) == len(tasks):
                    print(f"dataset trajectories {len(results)}/{len(tasks)}", flush=True)
    train_results = [row for row in results if int(row["trajectory_id"]) % 5 != 0]
    validation_results = [row for row in results if int(row["trajectory_id"]) % 5 == 0]
    dataset = {
        "config": {
            "source": str(RAW_TRACE_PATH),
            "source_sha256": _sha256(RAW_TRACE_PATH),
            "split": "trajectory_id modulo 5; validation iff remainder is zero",
            "ordinary_cells_only": True,
            "largest_cell_examples": 0,
            "inputs": "normalized_distance|normalized_surrogate|u",
            "target": "u*d_norm-(1-u)*s_norm",
        },
        "train": _pack_split(train_results),
        "validation": _pack_split(validation_results),
    }
    torch.save(dataset, DATASET_PATH)
    return dataset


class CyclicCandidateSampler:
    def __init__(self, count: int, seed: int) -> None:
        self.count = count
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(count)
        self.position = 0

    def take(self, count: int) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = count
        while remaining:
            available = self.count - self.position
            used = min(available, remaining)
            pieces.append(self.order[self.position : self.position + used])
            self.position += used
            remaining -= used
            if self.position == self.count:
                self.order = self.generator.permutation(self.count)
                self.position = 0
        return np.concatenate(pieces)


def _pair_indices(
    split: dict[str, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    first: list[int] = []
    second: list[int] = []
    signs: list[float] = []
    margins: list[float] = []
    inputs = split["inputs"]
    offsets = split["offsets"]
    for cell_index in range(len(split["counts"])):
        start = int(offsets[cell_index])
        stop = int(offsets[cell_index + 1])
        if stop - start < 2:
            continue
        local_scores = _true_scores(inputs[start:stop])
        for _ in range(PAIR_COUNT_PER_CELL):
            pair = generator.choice(stop - start, size=2, replace=False)
            difference = float(local_scores[int(pair[0])] - local_scores[int(pair[1])])
            if abs(difference) <= 1.0e-14:
                continue
            first.append(start + int(pair[0]))
            second.append(start + int(pair[1]))
            signs.append(math.copysign(1.0, difference))
            margins.append(min(abs(difference), 0.05))
    return (
        np.asarray(first, dtype=np.int64),
        np.asarray(second, dtype=np.int64),
        np.asarray(signs, dtype=np.float32),
        np.asarray(margins, dtype=np.float32),
    )


def _validation_regression_metrics(
    model: quickscore.ScoreNetwork, inputs: np.ndarray
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    maximum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(inputs), VALIDATION_BATCH_SIZE):
            batch_np = inputs[start : start + VALIDATION_BATCH_SIZE]
            batch = torch.as_tensor(batch_np, dtype=torch.float64)
            target = torch.as_tensor(_true_scores(batch_np), dtype=torch.float64)
            error = model(batch) - target
            squared_error += float(torch.sum(torch.square(error)).item())
            absolute_error += float(torch.sum(torch.abs(error)).item())
            maximum = max(maximum, float(torch.max(torch.abs(error)).item()))
            count += len(batch_np)
    return {
        "score_rmse": math.sqrt(squared_error / count),
        "score_mae": absolute_error / count,
        "score_max_abs_error": maximum,
    }


def _save_checkpoint(
    model: quickscore.ScoreNetwork,
    epoch: int,
    metrics: dict[str, float],
    dataset: dict[str, Any],
) -> None:
    torch.save(
        {
            "state_dict": copy.deepcopy(model.state_dict()),
            "architecture": "ScoreNetwork(3,32,32,1; SiLU)",
            "epoch": epoch,
            "validation_metrics": metrics,
            "model_seed": MODEL_SEED,
            "learning_rate": LEARNING_RATE,
            "ranking_weight": RANKING_WEIGHT,
            "dataset_source_sha256": dataset["config"]["source_sha256"],
            "train_cells": int(dataset["train"]["cell_count"]),
            "validation_cells": int(dataset["validation"]["cell_count"]),
        },
        CHECKPOINT_PATH,
    )


def _train(
    dataset: dict[str, Any], resume: bool
) -> tuple[quickscore.ScoreNetwork, list[dict[str, Any]], int]:
    train = dataset["train"]
    validation = dataset["validation"]
    torch.manual_seed(MODEL_SEED)
    model = quickscore.ScoreNetwork().to(torch.float64)
    start_epoch = 0
    if resume and CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = int(checkpoint["epoch"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sampler = CyclicCandidateSampler(len(train["inputs"]), MODEL_SEED + 1)
    pair_first, pair_second, pair_sign, pair_margin = _pair_indices(
        train, MODEL_SEED + 2
    )
    pair_generator = np.random.default_rng(MODEL_SEED + 3)
    history: list[dict[str, Any]] = []
    initial_metrics = _validation_regression_metrics(model, validation["inputs"])
    best_rmse = initial_metrics["score_rmse"]
    best_epoch = start_epoch
    _save_checkpoint(model, best_epoch, initial_metrics, dataset)
    stale = 0
    for local_epoch in range(1, MAX_EPOCHS + 1):
        epoch = start_epoch + local_epoch
        selected = sampler.take(
            min(TRAIN_CANDIDATES_PER_EPOCH, len(train["inputs"]))
        )
        model.train()
        regression_sum = 0.0
        ranking_sum = 0.0
        updates = 0
        for start in range(0, len(selected), CANDIDATE_BATCH_SIZE):
            indices = selected[start : start + CANDIDATE_BATCH_SIZE]
            batch_np = train["inputs"][indices]
            batch = torch.as_tensor(batch_np, dtype=torch.float64)
            target = torch.as_tensor(_true_scores(batch_np), dtype=torch.float64)
            pair_indices = pair_generator.integers(
                0, len(pair_first), size=min(PAIR_BATCH_SIZE, len(pair_first))
            )
            first = torch.as_tensor(
                train["inputs"][pair_first[pair_indices]], dtype=torch.float64
            )
            second = torch.as_tensor(
                train["inputs"][pair_second[pair_indices]], dtype=torch.float64
            )
            signs = torch.as_tensor(pair_sign[pair_indices], dtype=torch.float64)
            margins = torch.as_tensor(pair_margin[pair_indices], dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            regression = nn.functional.mse_loss(prediction, target)
            predicted_difference = model(first) - model(second)
            ranking = torch.relu(margins - signs * predicted_difference).mean()
            loss = regression + RANKING_WEIGHT * ranking
            loss.backward()
            optimizer.step()
            regression_sum += float(regression.item())
            ranking_sum += float(ranking.item())
            updates += 1
        validation_metrics = _validation_regression_metrics(
            model, validation["inputs"]
        )
        row = {
            "record_type": "epoch",
            "epoch": epoch,
            "train_mse_batch_mean": regression_sum / updates,
            "train_ranking_loss_batch_mean": ranking_sum / updates,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        if local_epoch == 1 or local_epoch % 10 == 0:
            print(
                f"epoch={epoch} validation_rmse={validation_metrics['score_rmse']:.3e}",
                flush=True,
            )
        if validation_metrics["score_rmse"] < best_rmse - MIN_DELTA:
            best_rmse = validation_metrics["score_rmse"]
            best_epoch = epoch
            _save_checkpoint(model, epoch, validation_metrics, dataset)
            stale = 0
        else:
            stale += 1
        if local_epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, history, best_epoch


def _predict_all(model: quickscore.ScoreNetwork, inputs: np.ndarray) -> np.ndarray:
    result: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), VALIDATION_BATCH_SIZE):
            batch = torch.as_tensor(
                inputs[start : start + VALIDATION_BATCH_SIZE], dtype=torch.float64
            )
            result.append(model(batch).cpu().numpy())
    return np.concatenate(result)


def _static_metrics(
    split: dict[str, np.ndarray], predicted: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    target = _true_scores(split["inputs"])
    errors = predicted - target
    correlation = float(np.corrcoef(predicted, target)[0, 1])
    hits = np.zeros((len(split["counts"]), 3), dtype=np.bool_)
    learned_winner = np.zeros(len(split["counts"]), dtype=np.int32)
    deterministic_hits = np.zeros(len(split["counts"]), dtype=np.bool_)
    for cell_index in range(len(split["counts"])):
        start = int(split["offsets"][cell_index])
        stop = int(split["offsets"][cell_index + 1])
        order = np.argsort(predicted[start:stop])[::-1]
        teacher = int(split["winners"][cell_index])
        learned_winner[cell_index] = int(order[0])
        deterministic_hits[cell_index] = int(np.argmax(target[start:stop])) == teacher
        for k in range(3):
            hits[cell_index, k] = teacher in order[: min(k + 1, len(order))]
    return (
        {
            "score_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "score_mae": float(np.mean(np.abs(errors))),
            "score_max_abs_error": float(np.max(np.abs(errors))),
            "score_correlation": correlation,
            "winner_top1": float(hits[:, 0].mean()),
            "winner_top2": float(hits[:, 1].mean()),
            "winner_top3": float(hits[:, 2].mean()),
            "deterministic_winner_accuracy": float(deterministic_hits.mean()),
        },
        hits,
        learned_winner,
        deterministic_hits,
    )


def _subset_row(
    record_type: str,
    subset: str,
    indices: np.ndarray,
    hits: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "subset": subset,
        "count": int(len(indices)),
        "winner_top1": float(hits[indices, 0].mean()) if len(indices) else math.nan,
        "winner_top2": float(hits[indices, 1].mean()) if len(indices) else math.nan,
        "winner_top3": float(hits[indices, 2].mean()) if len(indices) else math.nan,
        **extra,
    }


def _static_rows(
    dataset: dict[str, Any], metrics: dict[str, float], hits: np.ndarray
) -> list[dict[str, Any]]:
    validation = dataset["validation"]
    train = dataset["train"]
    rows: list[dict[str, Any]] = [
        {
            "record_type": "static_validation",
            "train_cells": int(train["cell_count"]),
            "train_candidates": int(train["candidate_count"]),
            "validation_cells": int(validation["cell_count"]),
            "validation_candidates": int(validation["candidate_count"]),
            **metrics,
        }
    ]
    u_edges = np.linspace(0.0, 1.0, 6)
    for index in range(5):
        mask = (validation["u"] >= u_edges[index]) & (
            (validation["u"] <= u_edges[index + 1])
            if index == 4
            else (validation["u"] < u_edges[index + 1])
        )
        rows.append(
            _subset_row(
                "u_range",
                f"[{u_edges[index]:.1f},{u_edges[index + 1]:.1f}]",
                np.flatnonzero(mask),
                hits,
                lower=u_edges[index],
                upper=u_edges[index + 1],
            )
        )
    count_edges = np.quantile(train["counts"], (0.25, 0.5, 0.75))
    count_bins = np.searchsorted(count_edges, validation["counts"], side="right")
    for index in range(4):
        rows.append(
            _subset_row(
                "candidate_count_quartile",
                f"Q{index + 1}",
                np.flatnonzero(count_bins == index),
                hits,
                lower=(-math.inf if index == 0 else count_edges[index - 1]),
                upper=(math.inf if index == 3 else count_edges[index]),
            )
        )
    return rows


def _static_error_rows(
    split: dict[str, np.ndarray], predicted: np.ndarray, winners: np.ndarray
) -> list[dict[str, Any]]:
    target = _true_scores(split["inputs"])
    rows: list[dict[str, Any]] = []
    for cell_index, learned_value in enumerate(winners):
        teacher = int(split["winners"][cell_index])
        learned = int(learned_value)
        if learned == teacher:
            continue
        start = int(split["offsets"][cell_index])
        rows.append(
            {
                "cell_index": cell_index,
                "trajectory_id": int(split["trajectory_id"][cell_index]),
                "iteration": int(split["iteration"][cell_index]),
                "site_index": int(split["site_index"][cell_index]),
                "candidate_count": int(split["counts"][cell_index]),
                "u": float(split["u"][cell_index]),
                "teacher_margin": float(split["margin"][cell_index]),
                "teacher_winner": teacher,
                "learned_winner": learned,
                "teacher_true_score": target[start + teacher],
                "teacher_predicted_score": predicted[start + teacher],
                "learned_true_score": target[start + learned],
                "learned_predicted_score": predicted[start + learned],
            }
        )
    return rows


def _learned_cell_proposals(
    model: quickscore.ScoreNetwork,
    points: np.ndarray,
    losses: np.ndarray,
    generator_id: int,
    iteration: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = edirect._cell_data(points, losses, generator_id, iteration)
    surface = edirect.GaussianLocalResponse(points, losses)
    selected_sizes = data.sizes[data.selected]
    size_weight = edirect._normalize(selected_sizes)
    loo_weight = edirect._normalize(surface.loo_errors(data.selected))
    u_values = 0.5 * size_weight + 0.5 * loo_weight
    largest_position = int(np.argmax(selected_sizes))
    proposals: list[np.ndarray] = []
    ordinary_cells = 0
    ordinary_disagreements = 0
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        mask = data.owners == site_index
        candidates = data.points[mask]
        distances = data.distances[mask]
        if len(candidates) == 0:
            continue
        if selected_position == largest_position:
            winner = int(np.argmax(distances))
        else:
            ordinary_cells += 1
            normalized_distance = edirect._normalize(distances)
            prediction = surface.predict(candidates)
            normalized_surrogate = edirect._normalize(prediction)
            u_value = float(u_values[selected_position])
            inputs = np.column_stack(
                (
                    normalized_distance,
                    normalized_surrogate,
                    np.full(len(candidates), u_value),
                )
            )
            with torch.no_grad():
                learned_scores = model(
                    torch.as_tensor(inputs, dtype=torch.float64)
                ).cpu().numpy()
            winner = int(np.argmax(learned_scores))
            deterministic_scores = (
                u_value * normalized_distance
                - (1.0 - u_value) * normalized_surrogate
            )
            ordinary_disagreements += int(winner != int(np.argmax(deterministic_scores)))
        proposal = candidates[winner]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            proposals.append(proposal)
    if not proposals:
        nearest = edirect.cKDTree(points).query(data.points, k=1)[0]
        for proposal_index in np.argsort(nearest)[::-1]:
            if nearest[proposal_index] > 1.0e-12:
                proposals.append(data.points[proposal_index])
                break
    return np.asarray(proposals, dtype=np.float64), {
        "ordinary_cells": ordinary_cells,
        "ordinary_winner_disagreements": ordinary_disagreements,
    }


def _run_normal(
    split: str,
    trajectory_id: int,
    position: np.ndarray,
    score_model: quickscore.ScoreNetwork,
    selector_model: imitation.CandidateRanker,
) -> tuple[imitation.ModeRun, dict[str, Any]]:
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    points = np.empty((0, 2), dtype=np.float64)
    losses = np.empty(0, dtype=np.float64)
    evaluations = 0
    iteration = 0
    ordinary_cells = 0
    ordinary_disagreements = 0
    first_ordinary_disagreement_evaluation: int | None = None
    disagreement_events: list[dict[str, Any]] = []

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
        loss = float(
            edirect._prefix_losses_numpy(
                (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
            )[0]
        )
        points = np.concatenate((points, proposal[None, :]))
        losses = np.concatenate((losses, np.asarray((loss,))))
        pending = np.delete(pending, int(choice), axis=0)
        evaluations += 1

    while evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
        proposals, cell_info = _learned_cell_proposals(
            score_model, points, losses, generator_id, iteration
        )
        ordinary_cells += int(cell_info["ordinary_cells"])
        if int(cell_info["ordinary_winner_disagreements"]):
            ordinary_disagreements += int(cell_info["ordinary_winner_disagreements"])
            if first_ordinary_disagreement_evaluation is None:
                first_ordinary_disagreement_evaluation = evaluations
            disagreement_events.append(
                {
                    "iteration": iteration,
                    "evaluation": evaluations,
                    **cell_info,
                }
            )
        pending = proposals.copy()
        while len(pending) and evaluations < NORMAL_CAP and float(np.min(losses)) > GATE:
            global_features = imitation._global_features(
                points, losses, len(pending), evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect",
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
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            pending = np.delete(pending, int(choice), axis=0)
            evaluations += 1
        if evaluations >= NORMAL_CAP or float(np.min(losses)) <= GATE:
            break
        greedy = edirect._greedy_proposal(points, losses, generator_id, iteration)
        if greedy is not None:
            singleton = greedy[None, :]
            global_features = imitation._global_features(
                points, losses, 1, evaluations, NORMAL_CAP, iteration,
                FULL_BOUNDS, "normal_edirect",
            )
            choice, _ = imitation._model_choice(
                selector_model, points, losses, singleton, "greedy", global_features
            )
            proposal = singleton[int(choice)]
            loss = float(
                edirect._prefix_losses_numpy(
                    (LOWER + DOMAIN_WIDTH * proposal)[None, :], position
                )[0]
            )
            points = np.concatenate((points, proposal[None, :]))
            losses = np.concatenate((losses, np.asarray((loss,))))
            evaluations += 1
        iteration += 1
    return (
        imitation.ModeRun(
            mode="normal_learned_ordinary_score",
            bounds=FULL_BOUNDS.copy(),
            points=points,
            losses=losses,
            new_evaluations=evaluations,
            success=float(np.min(losses)) <= GATE,
            decisions=disagreement_events,
        ),
        {
            "ordinary_cells": ordinary_cells,
            "ordinary_winner_disagreements": ordinary_disagreements,
            "first_ordinary_disagreement_evaluation": first_ordinary_disagreement_evaluation,
        },
    )


_WORKER_SCORE_MODEL: quickscore.ScoreNetwork | None = None
_WORKER_SELECTOR_MODEL: imitation.CandidateRanker | None = None


def _load_search_models() -> tuple[quickscore.ScoreNetwork, imitation.CandidateRanker]:
    global _WORKER_SCORE_MODEL, _WORKER_SELECTOR_MODEL
    if _WORKER_SCORE_MODEL is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        _WORKER_SCORE_MODEL = quickscore.ScoreNetwork().to(torch.float64)
        _WORKER_SCORE_MODEL.load_state_dict(checkpoint["state_dict"])
        _WORKER_SCORE_MODEL.eval()
    if _WORKER_SELECTOR_MODEL is None:
        checkpoint = torch.load(SELECTOR_PATH, map_location="cpu", weights_only=False)
        _WORKER_SELECTOR_MODEL = imitation.CandidateRanker().to(torch.float32)
        _WORKER_SELECTOR_MODEL.load_state_dict(checkpoint["state_dict"])
        _WORKER_SELECTOR_MODEL.eval()
    return _WORKER_SCORE_MODEL, _WORKER_SELECTOR_MODEL


def _search_worker(arguments: tuple[str, int, np.ndarray]) -> dict[str, Any]:
    split, trajectory_id, position = arguments
    score_model, selector_model = _load_search_models()
    normal, diagnostics = _run_normal(
        split, trajectory_id, position, score_model, selector_model
    )
    generator_id = imitation.SPLIT_ID_OFFSET[split] + trajectory_id
    modes = [normal]
    total = normal.new_evaluations
    boundary_run = nested_run = None
    if not normal.success:
        boundary_run, _ = imitation._run_boundary_rescue(
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
        "success": any(mode.success for mode in modes),
        "normal_success": normal.success,
        "normal_evaluations": normal.new_evaluations,
        "normal_best_loss": normal.best_loss,
        "total_evaluations": total,
        "final_loss": final.best_loss,
        "boundary_invoked": boundary_run is not None,
        "boundary_success": bool(boundary_run and boundary_run.success),
        "nested_invoked": nested_run is not None,
        "nested_success": bool(nested_run and nested_run.success),
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


def _run_search_split(
    split: str, positions: np.ndarray, workers: int
) -> list[dict[str, Any]]:
    tasks = [(split, index, positions[index]) for index in range(len(positions))]
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            results.append(_search_worker(task))
            print(f"search {split} {len(results)}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_search_worker, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())
                if len(results) % 4 == 0 or len(results) == len(tasks):
                    print(f"search {split} {len(results)}/{len(tasks)}", flush=True)
    results.sort(key=lambda row: int(row["trajectory_id"]))
    return results


def _reference_fields(run: dict[str, Any]) -> dict[str, Any]:
    normal = next(mode for mode in run["modes"] if mode["mode"] == "normal_edirect")
    return {
        "success": bool(run["success"]),
        "normal_success": bool(normal["success"]),
        "normal_evaluations": int(normal["new_evaluations"]),
        "total_evaluations": int(run["total_evaluations"]),
        "boundary_invoked": len(run["modes"]) > 1,
        "nested_invoked": any(mode["mode"] == "nested_1d" for mode in run["modes"]),
    }


def _comparison_rows(
    learned: list[dict[str, Any]], reference: list[dict[str, Any]], phase: str
) -> list[dict[str, Any]]:
    reference_by_id = {int(row["trajectory_id"]): row for row in reference}
    rows: list[dict[str, Any]] = []
    for row in learned:
        teacher = _reference_fields(reference_by_id[int(row["trajectory_id"])])
        normal_agree = bool(row["normal_success"]) == teacher["normal_success"]
        final_agree = bool(row["success"]) == teacher["success"]
        rows.append(
            {
                "phase": phase,
                "split": row["split"],
                "trajectory_id": row["trajectory_id"],
                "learned_normal_success": row["normal_success"],
                "teacher_normal_success": teacher["normal_success"],
                "normal_outcome_agreement": normal_agree,
                "learned_final_success": row["success"],
                "teacher_final_success": teacher["success"],
                "final_outcome_agreement": final_agree,
                "learned_normal_evaluations": row["normal_evaluations"],
                "teacher_normal_evaluations": teacher["normal_evaluations"],
                "learned_total_evaluations": row["total_evaluations"],
                "teacher_total_evaluations": teacher["total_evaluations"],
                "learned_boundary_invoked": row["boundary_invoked"],
                "teacher_boundary_invoked": teacher["boundary_invoked"],
                "learned_nested_invoked": row["nested_invoked"],
                "teacher_nested_invoked": teacher["nested_invoked"],
                "ordinary_cells": row["ordinary_cells"],
                "ordinary_winner_disagreements": row[
                    "ordinary_winner_disagreements"
                ],
                "first_ordinary_disagreement_evaluation": row[
                    "first_ordinary_disagreement_evaluation"
                ],
                "outcome_difference_has_ordinary_winner_disagreement": (
                    (not normal_agree or not final_agree)
                    and int(row["ordinary_winner_disagreements"]) > 0
                ),
            }
        )
    return rows


def _search_summary_row(
    split: str,
    method: str,
    runs: list[dict[str, Any]],
    comparison: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if method == "teacher_deterministic_score_frozen_selector":
        fields = [_reference_fields(run) for run in runs]
        ordinary_cells = ordinary_disagreements = ""
    else:
        fields = runs
        ordinary_cells = sum(int(run["ordinary_cells"]) for run in runs)
        ordinary_disagreements = sum(
            int(run["ordinary_winner_disagreements"]) for run in runs
        )
    successes = [row for row in fields if bool(row["success"])]
    result: dict[str, Any] = {
        "record_type": "search_summary",
        "split": split,
        "method": method,
        "trajectories": len(fields),
        "normal_success_count": sum(bool(row["normal_success"]) for row in fields),
        "normal_success_fraction": float(
            np.mean([bool(row["normal_success"]) for row in fields])
        ),
        "final_success_count": len(successes),
        "final_success_fraction": len(successes) / len(fields),
        "median_total_evaluations_success": (
            float(np.median([row["total_evaluations"] for row in successes]))
            if successes
            else math.nan
        ),
        "mean_total_evaluations_success": (
            float(np.mean([row["total_evaluations"] for row in successes]))
            if successes
            else math.nan
        ),
        "worst_total_evaluations_success": max(
            [row["total_evaluations"] for row in successes], default=math.nan
        ),
        "rescue_invocation_count": sum(bool(row["boundary_invoked"]) for row in fields),
        "nested_invocation_count": sum(bool(row["nested_invoked"]) for row in fields),
        "ordinary_cells": ordinary_cells,
        "ordinary_winner_disagreements": ordinary_disagreements,
    }
    if comparison is not None:
        result["normal_outcome_agreement"] = float(
            np.mean([row["normal_outcome_agreement"] for row in comparison])
        )
        result["final_outcome_agreement"] = float(
            np.mean([row["final_outcome_agreement"] for row in comparison])
        )
        result["outcome_differences"] = sum(
            not row["normal_outcome_agreement"] or not row["final_outcome_agreement"]
            for row in comparison
        )
        result["outcome_differences_with_ordinary_disagreement"] = sum(
            row["outcome_difference_has_ordinary_winner_disagreement"]
            for row in comparison
        )
    return result


def _existing_training_rows() -> list[dict[str, Any]]:
    if not SUMMARY_PATH.exists():
        return []
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("record_type") == "epoch"
        ]


def run(
    workers: int,
    rebuild_dataset: bool,
    resume: bool,
    evaluate_checkpoint: bool,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if _sha256(SELECTOR_PATH) != _sha256(SOURCE_SELECTOR_PATH):
        raise RuntimeError("the frozen final selector is not byte-identical to its source")
    dataset = (
        _build_dataset(workers)
        if rebuild_dataset or not DATASET_PATH.exists()
        else torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    )
    if evaluate_checkpoint:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location="cpu", weights_only=False
        )
        model = quickscore.ScoreNetwork().to(torch.float64)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        training_history = _existing_training_rows()
        best_epoch = int(checkpoint["epoch"])
    else:
        model, training_history, best_epoch = _train(dataset, resume)
    validation_predictions = _predict_all(model, dataset["validation"]["inputs"])
    static_metrics, hits, static_winners, deterministic_hits = _static_metrics(
        dataset["validation"], validation_predictions
    )
    static_gate = bool(
        static_metrics["winner_top1"] >= STATIC_TOP1_GATE
        and static_metrics["winner_top3"] >= STATIC_TOP3_GATE
        and static_metrics["deterministic_winner_accuracy"] == 1.0
    )
    rows = list(training_history)
    rows.extend(_static_rows(dataset, static_metrics, hits))
    rows.append(
        {
            "record_type": "static_gate",
            "best_epoch": best_epoch,
            "required_top1": STATIC_TOP1_GATE,
            "required_top3": STATIC_TOP3_GATE,
            "passed": static_gate,
            "assembled_screen_run": static_gate,
            "deterministic_winners_correct": int(deterministic_hits.sum()),
            "deterministic_winners_total": len(deterministic_hits),
            "frozen_selector_sha256": _sha256(SELECTOR_PATH),
        }
    )
    _write_csv(
        STATIC_ERRORS_PATH,
        _static_error_rows(
            dataset["validation"], validation_predictions, static_winners
        ),
        default_fields=("cell_index", "trajectory_id", "iteration", "site_index"),
    )
    if not static_gate:
        _write_csv(SUMMARY_PATH, rows)
        _write_csv(PER_TRAJECTORY_PATH, [], default_fields=("phase", "split", "trajectory_id"))
        torch.save({"static_gate_passed": False, "screen": [], "full": []}, SEARCH_TRACE_PATH)
        print("static gate failed; assembled search not run", flush=True)
        return

    validation_positions, _, _ = _make_exact_labeled_split(
        imitation.VALIDATION_COUNT, imitation.VALIDATION_SEED
    )
    test_positions, _, _ = _make_exact_labeled_split(
        imitation.TEST_COUNT, imitation.TEST_SEED
    )
    teacher_payload = torch.load(
        TEACHER_SEARCH_PATH, map_location="cpu", weights_only=False
    )
    validation_reference = teacher_payload["validation_learned"]["trajectories"]
    test_reference = teacher_payload["test_learned"]["trajectories"]
    validation_np = validation_positions.to(dtype=DTYPE).cpu().numpy()
    test_np = test_positions.to(dtype=DTYPE).cpu().numpy()

    screen_runs = _run_search_split(
        "validation", validation_np[:SCREEN_COUNT], workers
    )
    screen_reference = validation_reference[:SCREEN_COUNT]
    screen_comparison = _comparison_rows(
        screen_runs, screen_reference, "assembled_16_screen"
    )
    teacher_normal_success = sum(
        _reference_fields(run)["normal_success"] for run in screen_reference
    )
    screen_final_success = sum(bool(run["success"]) for run in screen_runs)
    screen_normal_success = sum(bool(run["normal_success"]) for run in screen_runs)
    screen_agreement = float(
        np.mean([row["final_outcome_agreement"] for row in screen_comparison])
    )
    screen_promising = bool(
        screen_final_success >= SCREEN_FINAL_SUCCESS_MIN
        and screen_normal_success >= teacher_normal_success - 2
        and screen_agreement >= SCREEN_OUTCOME_AGREEMENT_MIN
    )
    rows.extend(
        (
            _search_summary_row(
                "validation_screen",
                "teacher_deterministic_score_frozen_selector",
                screen_reference,
                None,
            ),
            _search_summary_row(
                "validation_screen",
                "learned_ordinary_score_frozen_selector",
                screen_runs,
                screen_comparison,
            ),
            {
                "record_type": "screen_gate",
                "screen_promising": screen_promising,
                "screen_final_success": screen_final_success,
                "screen_normal_success": screen_normal_success,
                "teacher_normal_success": teacher_normal_success,
                "screen_final_outcome_agreement": screen_agreement,
                "full_search_run": screen_promising,
            },
        )
    )
    all_comparisons = list(screen_comparison)
    trace: dict[str, Any] = {
        "static_gate_passed": True,
        "screen_promising": screen_promising,
        "screen": screen_runs,
        "screen_reference": screen_reference,
        "full_validation": [],
        "full_test": [],
    }
    if screen_promising:
        validation_runs = _run_search_split("validation", validation_np, workers)
        test_runs = _run_search_split("test", test_np, workers)
        validation_comparison = _comparison_rows(
            validation_runs, validation_reference, "full"
        )
        test_comparison = _comparison_rows(test_runs, test_reference, "full")
        all_comparisons.extend(validation_comparison)
        all_comparisons.extend(test_comparison)
        rows.extend(
            (
                _search_summary_row(
                    "validation",
                    "teacher_deterministic_score_frozen_selector",
                    validation_reference,
                    None,
                ),
                _search_summary_row(
                    "validation",
                    "learned_ordinary_score_frozen_selector",
                    validation_runs,
                    validation_comparison,
                ),
                _search_summary_row(
                    "test",
                    "teacher_deterministic_score_frozen_selector",
                    test_reference,
                    None,
                ),
                _search_summary_row(
                    "test",
                    "learned_ordinary_score_frozen_selector",
                    test_runs,
                    test_comparison,
                ),
            )
        )
        trace["full_validation"] = validation_runs
        trace["full_validation_reference"] = validation_reference
        trace["full_test"] = test_runs
        trace["full_test_reference"] = test_reference
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(
        PER_TRAJECTORY_PATH,
        all_comparisons,
        default_fields=("phase", "split", "trajectory_id"),
    )
    torch.save(trace, SEARCH_TRACE_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-checkpoint", action="store_true")
    arguments = parser.parse_args()
    run(
        arguments.workers,
        arguments.rebuild_dataset,
        arguments.resume,
        arguments.evaluate_checkpoint,
    )


if __name__ == "__main__":
    main()
