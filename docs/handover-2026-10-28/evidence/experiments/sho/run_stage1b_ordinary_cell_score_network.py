#!/usr/bin/env python3
"""Regress the ordinary-cell eDIRECT candidate score, then audit argmaxes."""

from __future__ import annotations

import copy
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_ordinary_cell_winner as ordinary
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import SEED


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/ordinary_cell_score_network")
SUMMARY_PATH = OUTPUT_DIR / "score_network_summary.csv"
ERRORS_PATH = OUTPUT_DIR / "score_network_errors.csv"

TRAIN_CELLS = 3_000
VALIDATION_CELLS = 750
SAMPLE_SEED = ordinary.SAMPLE_SEED
MODEL_SEED = SEED + 16_700_000
HIDDEN_DIM = 32
CANDIDATE_BATCH_SIZE = 65_536
PAIR_BATCH_SIZE = 16_384
PAIRS_PER_CELL = 8
LEARNING_RATE = 2.0e-3
RANKING_WEIGHT = 0.05
MAX_EPOCHS = 100
MIN_EPOCHS = 15
PATIENCE = 12


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


def _score(inputs: np.ndarray) -> np.ndarray:
    return inputs[:, 2] * inputs[:, 0] - (1.0 - inputs[:, 2]) * inputs[:, 1]


def _flatten(examples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.concatenate([row["candidate_inputs"] for row in examples], axis=0)
    targets = np.concatenate([_score(row["candidate_inputs"]) for row in examples])
    return inputs.astype(np.float64, copy=False), targets.astype(np.float64, copy=False)


def _pair_dataset(
    examples: list[dict[str, Any]], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    first_inputs: list[np.ndarray] = []
    second_inputs: list[np.ndarray] = []
    signs: list[float] = []
    margins: list[float] = []
    for row in examples:
        inputs = row["candidate_inputs"]
        if len(inputs) < 2:
            continue
        true_scores = _score(inputs)
        # Always include winner-versus-near-contender comparisons, then add
        # random pairs so the ordering loss covers the rest of the cell.
        order = np.argsort(true_scores)[::-1]
        pairs: list[tuple[int, int]] = [
            (int(order[0]), int(order[index]))
            for index in range(1, min(4, len(order)))
        ]
        while len(pairs) < PAIRS_PER_CELL:
            pair = generator.choice(len(inputs), size=2, replace=False)
            pairs.append((int(pair[0]), int(pair[1])))
        for first, second in pairs[:PAIRS_PER_CELL]:
            difference = float(true_scores[first] - true_scores[second])
            if abs(difference) <= 1.0e-14:
                continue
            first_inputs.append(inputs[first])
            second_inputs.append(inputs[second])
            signs.append(math.copysign(1.0, difference))
            # Do not demand an artificial separation larger than the teacher's
            # actual score difference.
            margins.append(min(abs(difference), 0.05))
    return (
        np.asarray(first_inputs, dtype=np.float64),
        np.asarray(second_inputs, dtype=np.float64),
        np.asarray(signs, dtype=np.float64),
        np.asarray(margins, dtype=np.float64),
    )


class ScoreNetwork(nn.Module):
    """Very small shared candidate scorer with no pooled cell context."""

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


def _permutation(count: int, seed: int) -> np.ndarray:
    result = np.arange(count, dtype=np.int64)
    np.random.default_rng(seed).shuffle(result)
    return result


def _predict_flat(model: ScoreNetwork, inputs: np.ndarray) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), CANDIDATE_BATCH_SIZE):
            batch = torch.as_tensor(
                inputs[start : start + CANDIDATE_BATCH_SIZE], dtype=torch.float64
            )
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


def _score_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = predicted - target
    centered_predicted = predicted - float(np.mean(predicted))
    centered_target = target - float(np.mean(target))
    denominator = float(
        np.sqrt(np.sum(np.square(centered_predicted)))
        * np.sqrt(np.sum(np.square(centered_target)))
    )
    correlation = (
        float(np.sum(centered_predicted * centered_target) / denominator)
        if denominator > 0.0
        else math.nan
    )
    return {
        "score_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "score_mae": float(np.mean(np.abs(errors))),
        "score_correlation": correlation,
        "score_max_abs_error": float(np.max(np.abs(errors))),
    }


def _winner_metrics(
    examples: list[dict[str, Any]], predicted_flat: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    hits = np.zeros((len(examples), 3), dtype=np.bool_)
    predicted_winners = np.zeros(len(examples), dtype=np.int64)
    deterministic_hits = np.zeros(len(examples), dtype=np.bool_)
    offset = 0
    for row_index, row in enumerate(examples):
        count = row["candidate_count"]
        predicted_scores = predicted_flat[offset : offset + count]
        true_scores = _score(row["candidate_inputs"])
        teacher_winner = int(row["winner_index"])
        order = np.argsort(predicted_scores)[::-1]
        predicted_winners[row_index] = int(order[0])
        deterministic_hits[row_index] = int(np.argmax(true_scores)) == teacher_winner
        for k in range(3):
            hits[row_index, k] = teacher_winner in order[: min(k + 1, len(order))]
        offset += count
    return (
        {
            "winner_top1": float(hits[:, 0].mean()),
            "winner_top2": float(hits[:, 1].mean()),
            "winner_top3": float(hits[:, 2].mean()),
            "deterministic_winner_accuracy": float(deterministic_hits.mean()),
        },
        hits,
        predicted_winners,
        deterministic_hits,
    )


def _evaluate(
    model: ScoreNetwork,
    examples: list[dict[str, Any]],
    flat_inputs: np.ndarray,
    targets: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predicted = _predict_flat(model, flat_inputs)
    winner, hits, predicted_winners, deterministic_hits = _winner_metrics(
        examples, predicted
    )
    return (
        {**_score_metrics(predicted, targets), **winner},
        predicted,
        hits,
        predicted_winners,
        deterministic_hits,
    )


def _train(
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
) -> tuple[
    ScoreNetwork,
    list[dict[str, Any]],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    train_inputs, train_targets = _flatten(train_examples)
    validation_inputs, validation_targets = _flatten(validation_examples)
    pair_first, pair_second, pair_sign, pair_margin = _pair_dataset(
        train_examples, MODEL_SEED + 1
    )
    torch.manual_seed(MODEL_SEED)
    model = ScoreNetwork().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_rmse = math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        candidate_order = _permutation(len(train_inputs), MODEL_SEED + 10 * epoch)
        pair_order = _permutation(len(pair_first), MODEL_SEED + 10 * epoch + 1)
        regression_sum = 0.0
        ranking_sum = 0.0
        update_count = 0
        candidate_batches = math.ceil(len(candidate_order) / CANDIDATE_BATCH_SIZE)
        for batch_index in range(candidate_batches):
            candidate_indices = candidate_order[
                batch_index
                * CANDIDATE_BATCH_SIZE : (batch_index + 1)
                * CANDIDATE_BATCH_SIZE
            ]
            pair_start = (batch_index * PAIR_BATCH_SIZE) % len(pair_order)
            pair_indices = pair_order[pair_start : pair_start + PAIR_BATCH_SIZE]
            if len(pair_indices) < PAIR_BATCH_SIZE:
                pair_indices = np.concatenate(
                    (pair_indices, pair_order[: PAIR_BATCH_SIZE - len(pair_indices)])
                )
            inputs = torch.as_tensor(train_inputs[candidate_indices], dtype=torch.float64)
            targets = torch.as_tensor(train_targets[candidate_indices], dtype=torch.float64)
            first = torch.as_tensor(pair_first[pair_indices], dtype=torch.float64)
            second = torch.as_tensor(pair_second[pair_indices], dtype=torch.float64)
            signs = torch.as_tensor(pair_sign[pair_indices], dtype=torch.float64)
            margins = torch.as_tensor(pair_margin[pair_indices], dtype=torch.float64)

            optimizer.zero_grad(set_to_none=True)
            predicted = model(inputs)
            regression = nn.functional.mse_loss(predicted, targets)
            predicted_difference = model(first) - model(second)
            ranking = torch.relu(margins - signs * predicted_difference).mean()
            loss = regression + RANKING_WEIGHT * ranking
            loss.backward()
            optimizer.step()
            regression_sum += float(regression.item())
            ranking_sum += float(ranking.item())
            update_count += 1

        validation_metrics, _, _, _, _ = _evaluate(
            model, validation_examples, validation_inputs, validation_targets
        )
        history.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "train_mse_batch_mean": regression_sum / update_count,
                "train_ranking_loss_batch_mean": ranking_sum / update_count,
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
            }
        )
        print(
            f"epoch={epoch} rmse={validation_metrics['score_rmse']:.3e} "
            f"winner={validation_metrics['winner_top1']:.6f}",
            flush=True,
        )
        if validation_metrics["score_rmse"] < best_rmse - 1.0e-10:
            best_rmse = validation_metrics["score_rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("training produced no best state")
    model.load_state_dict(best_state)
    metrics, predicted, hits, winners, deterministic_hits = _evaluate(
        model, validation_examples, validation_inputs, validation_targets
    )
    return (
        model,
        history,
        metrics,
        predicted,
        hits,
        winners,
        deterministic_hits,
        best_epoch,
    )


def _subset_row(
    record_type: str,
    subset: str,
    indices: np.ndarray,
    hits: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    if len(indices) == 0:
        values = {
            "count": 0,
            "winner_top1": math.nan,
            "winner_top2": math.nan,
            "winner_top3": math.nan,
        }
    else:
        values = {
            "count": int(len(indices)),
            "winner_top1": float(hits[indices, 0].mean()),
            "winner_top2": float(hits[indices, 1].mean()),
            "winner_top3": float(hits[indices, 2].mean()),
        }
    return {"record_type": record_type, "subset": subset, **values, **extra}


def _diagnostics(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], hits: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_u = np.asarray([row["u"] for row in validation])
    u_edges = np.linspace(0.0, 1.0, 6)
    for index in range(5):
        included = (validation_u >= u_edges[index]) & (
            (validation_u <= u_edges[index + 1])
            if index == 4
            else (validation_u < u_edges[index + 1])
        )
        rows.append(
            _subset_row(
                "u_range",
                f"[{u_edges[index]:.1f},{u_edges[index + 1]:.1f}]",
                np.flatnonzero(included),
                hits,
                lower=u_edges[index],
                upper=u_edges[index + 1],
            )
        )

    train_counts = np.asarray([row["candidate_count"] for row in train])
    validation_counts = np.asarray([row["candidate_count"] for row in validation])
    count_edges = np.quantile(train_counts, (0.25, 0.5, 0.75))
    bins = np.searchsorted(count_edges, validation_counts, side="right")
    for index in range(4):
        rows.append(
            _subset_row(
                "candidate_count_quartile",
                f"Q{index + 1}",
                np.flatnonzero(bins == index),
                hits,
                lower=(-math.inf if index == 0 else count_edges[index - 1]),
                upper=(math.inf if index == 3 else count_edges[index]),
            )
        )

    margins = np.asarray(
        [row["teacher_margin"] for row in validation if np.isfinite(row["teacher_margin"])]
    )
    rows.append(
        {
            "record_type": "teacher_winner_margin",
            "count": len(margins),
            "minimum": float(np.min(margins)),
            "p01": float(np.quantile(margins, 0.01)),
            "p10": float(np.quantile(margins, 0.10)),
            "median": float(np.median(margins)),
            "p90": float(np.quantile(margins, 0.90)),
            "fraction_at_most_1e-6": float(np.mean(margins <= 1.0e-6)),
        }
    )
    return rows


def _error_rows(
    validation: list[dict[str, Any]],
    predicted_flat: np.ndarray,
    predicted_winners: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    for cell_index, row in enumerate(validation):
        count = row["candidate_count"]
        true_scores = _score(row["candidate_inputs"])
        predicted_scores = predicted_flat[offset : offset + count]
        teacher_index = int(row["winner_index"])
        predicted_index = int(predicted_winners[cell_index])
        if teacher_index != predicted_index:
            rows.append(
                {
                    "validation_cell_index": cell_index,
                    "trajectory_id": row["trajectory_id"],
                    "iteration": row["iteration"],
                    "site_index": row["site_index"],
                    "candidate_count": count,
                    "u": row["u"],
                    "teacher_margin": row["teacher_margin"],
                    "teacher_winner_index": teacher_index,
                    "predicted_winner_index": predicted_index,
                    "teacher_true_score": true_scores[teacher_index],
                    "teacher_predicted_score": predicted_scores[teacher_index],
                    "predicted_winner_true_score": true_scores[predicted_index],
                    "predicted_winner_predicted_score": predicted_scores[predicted_index],
                    "teacher_score_prediction_error": predicted_scores[teacher_index]
                    - true_scores[teacher_index],
                    "predicted_score_prediction_error": predicted_scores[predicted_index]
                    - true_scores[predicted_index],
                }
            )
        offset += count
    return rows


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(filter_data.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_state_indices = ordinary._split_state_indices(
        payload["states"], False, SAMPLE_SEED
    )
    validation_state_indices = ordinary._split_state_indices(
        payload["states"], True, SAMPLE_SEED + 1
    )
    train, train_info = ordinary._collect_examples(
        payload, train_state_indices, TRAIN_CELLS
    )
    validation, validation_info = ordinary._collect_examples(
        payload, validation_state_indices, VALIDATION_CELLS
    )

    started = time.perf_counter()
    (
        _,
        history,
        metrics,
        predicted_flat,
        hits,
        predicted_winners,
        deterministic_hits,
        best_epoch,
    ) = _train(train, validation)
    runtime_seconds = time.perf_counter() - started

    summary = list(history)
    summary.append(
        {
            "record_type": "quick_screen",
            "best_epoch": best_epoch,
            "train_cells": len(train),
            "validation_cells": len(validation),
            **metrics,
            "deterministic_winners_correct": int(deterministic_hits.sum()),
            "deterministic_winners_total": len(deterministic_hits),
            "runtime_seconds": runtime_seconds,
            "closed_loop_run": False,
            "pooled_context_used": False,
        }
    )
    summary.extend(_diagnostics(train, validation, hits))
    summary.append(
        {
            "record_type": "dataset",
            **{f"train_{key}": value for key, value in train_info.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_info.items()
            },
            "largest_cell_examples": 0,
            "candidate_input": "normalized_distance|normalized_surrogate|u",
            "target": "u*d_norm-(1-u)*s_norm",
            "winner_classification_loss_used": False,
            "regression_loss": "MSE",
            "pairwise_ranking_weight": RANKING_WEIGHT,
        }
    )
    _write_csv(SUMMARY_PATH, summary)
    _write_csv(
        ERRORS_PATH,
        _error_rows(validation, predicted_flat, predicted_winners),
        default_fields=(
            "validation_cell_index",
            "trajectory_id",
            "iteration",
            "site_index",
            "candidate_count",
            "u",
            "teacher_margin",
            "teacher_winner_index",
            "predicted_winner_index",
        ),
    )
    print(
        f"rmse={metrics['score_rmse']:.6e} mae={metrics['score_mae']:.6e} "
        f"corr={metrics['score_correlation']:.8f} "
        f"top1={metrics['winner_top1']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    run()
