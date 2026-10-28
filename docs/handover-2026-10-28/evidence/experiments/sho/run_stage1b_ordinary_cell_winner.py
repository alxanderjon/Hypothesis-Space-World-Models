#!/usr/bin/env python3
"""Quick imitation screen for ordinary-cell eDIRECT winner selection only."""

from __future__ import annotations

import argparse
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

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import SEED


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/ordinary_cell_winner")
SUMMARY_PATH = OUTPUT_DIR / "ordinary_cell_summary.csv"
ERRORS_PATH = OUTPUT_DIR / "ordinary_cell_errors.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "ordinary_cell_checkpoint.pt"

RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
TRAIN_CELLS = 3_000
VALIDATION_CELLS = 750
SAMPLE_SEED = SEED + 16_500_000
MODEL_SEED = SEED + 16_600_000
HIDDEN_DIM = 48
BATCH_SIZE = 512
LEARNING_RATE = 1.0e-3
MAX_EPOCHS = 60
MIN_EPOCHS = 15
PATIENCE = 12
STRONG_TOP1_GATE = 0.99


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


def _trajectory_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["trajectory_id"]): row for row in payload["trajectories"]}


def _split_state_indices(
    states: list[dict[str, Any]], validation: bool, seed: int
) -> np.ndarray:
    indices = [
        index
        for index, state in enumerate(states)
        if state["candidate_type"] == "cell"
        and ((int(state["trajectory_id"]) % 5 == 0) == validation)
    ]
    generator = np.random.default_rng(seed)
    generator.shuffle(indices)
    return np.asarray(indices, dtype=np.int64)


def _set_distance(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) != len(second):
        return math.inf
    if len(first) == 0:
        return 0.0
    pairwise = np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2)
    return float(
        max(np.max(np.min(pairwise, axis=0)), np.max(np.min(pairwise, axis=1)))
    )


def _ordinary_examples_for_state(
    record: dict[str, Any], trajectory: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
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
    normalized_sizes = edirect._normalize(selected_sizes)
    loo_errors = surface.loo_errors(data.selected)
    normalized_loo = edirect._normalize(loo_errors)
    ordinary_u = 0.5 * normalized_sizes + 0.5 * normalized_loo
    largest_position = int(np.argmax(selected_sizes))

    reconstructed_proposals: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        pool_indices = np.flatnonzero(data.owners == site_index)
        if len(pool_indices) == 0:
            continue
        candidates = data.points[pool_indices]
        distances = data.distances[pool_indices]
        normalized_distance = edirect._normalize(distances)
        surrogate = surface.predict(candidates)
        normalized_surrogate = edirect._normalize(surrogate)
        is_largest = selected_position == largest_position
        if is_largest:
            winner_index = int(np.argmax(distances))
        else:
            u_value = float(ordinary_u[selected_position])
            teacher_score = (
                u_value * normalized_distance
                - (1.0 - u_value) * normalized_surrogate
            )
            winner_index = int(np.argmax(teacher_score))
            ordered = np.sort(teacher_score)
            margin = (
                math.inf
                if len(ordered) == 1
                else float(ordered[-1] - ordered[-2])
            )
            inputs = np.column_stack(
                (
                    normalized_distance,
                    normalized_surrogate,
                    np.full(len(candidates), u_value, dtype=np.float64),
                )
            )
            examples.append(
                {
                    "trajectory_id": int(record["trajectory_id"]),
                    "generator_id": generator_id,
                    "iteration": iteration,
                    "history_count": history_count,
                    "site_index": site_index,
                    "candidate_inputs": inputs,
                    "candidate_points": candidates.astype(np.float64, copy=True),
                    "candidate_count": int(len(candidates)),
                    "u": u_value,
                    "cell_size": float(data.sizes[site_index]),
                    "winner_index": winner_index,
                    "teacher_margin": margin,
                }
            )
        proposal = candidates[winner_index]
        if np.min(np.linalg.norm(points - proposal[None, :], axis=1)) > 1.0e-12:
            reconstructed_proposals.append(proposal)

    expected = np.asarray(record["teacher_curated_points"], dtype=np.float64)
    reconstructed = np.asarray(reconstructed_proposals, dtype=np.float64).reshape(-1, 2)
    return examples, _set_distance(reconstructed, expected)


def _collect_examples(
    payload: dict[str, Any], state_indices: np.ndarray, target: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories = _trajectory_map(payload)
    examples: list[dict[str, Any]] = []
    states_used = 0
    worst_reconstruction_error = 0.0
    for state_index_value in state_indices:
        record = payload["states"][int(state_index_value)]
        state_examples, mismatch = _ordinary_examples_for_state(
            record, trajectories[int(record["trajectory_id"])]
        )
        if not np.isfinite(mismatch) or mismatch > 1.0e-12:
            raise RuntimeError(
                f"teacher reconstruction failed for state {state_index_value}: {mismatch}"
            )
        states_used += 1
        worst_reconstruction_error = max(worst_reconstruction_error, mismatch)
        examples.extend(state_examples)
        if len(examples) >= target:
            examples = examples[:target]
            break
    if len(examples) != target:
        raise RuntimeError(f"requested {target} ordinary cells; found {len(examples)}")
    return examples, {
        "states_used": states_used,
        "worst_teacher_set_reconstruction_error": worst_reconstruction_error,
        "candidate_count_min": min(row["candidate_count"] for row in examples),
        "candidate_count_median": float(
            np.median([row["candidate_count"] for row in examples])
        ),
        "candidate_count_max": max(row["candidate_count"] for row in examples),
        "u_min": min(row["u"] for row in examples),
        "u_median": float(np.median([row["u"] for row in examples])),
        "u_max": max(row["u"] for row in examples),
    }


class OrdinaryCellScorer(nn.Module):
    """One shared candidate MLP; independent scoring is permutation equivariant."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, candidate_inputs: torch.Tensor) -> torch.Tensor:
        return self.network(candidate_inputs).squeeze(1)


def _batch_indices(count: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(count, dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [indices[start : start + BATCH_SIZE] for start in range(0, count, BATCH_SIZE)]


def _packed_batch(
    examples: list[dict[str, Any]], indices: np.ndarray
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    counts = np.asarray(
        [examples[int(index)]["candidate_count"] for index in indices], dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(counts)))
    packed = np.concatenate(
        [examples[int(index)]["candidate_inputs"] for index in indices], axis=0
    )
    winners = np.asarray(
        [examples[int(index)]["winner_index"] for index in indices], dtype=np.int64
    )
    return torch.as_tensor(packed, dtype=torch.float64), offsets, winners


def _packed_cross_entropy(
    scores: torch.Tensor, offsets: np.ndarray, winners: np.ndarray
) -> torch.Tensor:
    losses = [
        torch.logsumexp(scores[offsets[row] : offsets[row + 1]], dim=0)
        - scores[offsets[row] + winners[row]]
        for row in range(len(winners))
    ]
    return torch.stack(losses).mean()


def _evaluate(
    model: OrdinaryCellScorer, examples: list[dict[str, Any]]
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    hits = np.zeros((len(examples), 3), dtype=np.bool_)
    predicted = np.zeros(len(examples), dtype=np.int64)
    total_loss = 0.0
    with torch.no_grad():
        for indices in _batch_indices(len(examples), False, 0):
            inputs, offsets, winners = _packed_batch(examples, indices)
            scores = model(inputs)
            total_loss += float(
                _packed_cross_entropy(scores, offsets, winners).item()
            ) * len(indices)
            for local_row, example_index in enumerate(indices):
                local_scores = scores[
                    offsets[local_row] : offsets[local_row + 1]
                ]
                order = torch.argsort(local_scores, descending=True)
                predicted[int(example_index)] = int(order[0])
                for k in range(3):
                    hits[int(example_index), k] = bool(
                        (order[: min(k + 1, len(order))] == winners[local_row]).any()
                    )
    return (
        {
            "cross_entropy": total_loss / len(examples),
            "top1": float(hits[:, 0].mean()),
            "top2": float(hits[:, 1].mean()),
            "top3": float(hits[:, 2].mean()),
        },
        hits,
        predicted,
    )


def _train(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> tuple[
    OrdinaryCellScorer,
    list[dict[str, Any]],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    int,
]:
    torch.manual_seed(MODEL_SEED)
    model = OrdinaryCellScorer().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_top1 = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for indices in _batch_indices(len(train), True, MODEL_SEED + epoch):
            inputs, offsets, winners = _packed_batch(train, indices)
            optimizer.zero_grad(set_to_none=True)
            scores = model(inputs)
            loss = _packed_cross_entropy(scores, offsets, winners)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(indices)
        validation_metrics, _, _ = _evaluate(model, validation)
        history.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "train_cross_entropy": train_loss / len(train),
                "validation_cross_entropy": validation_metrics["cross_entropy"],
                "validation_top1": validation_metrics["top1"],
                "validation_top2": validation_metrics["top2"],
                "validation_top3": validation_metrics["top3"],
            }
        )
        print(
            f"epoch={epoch} val_top1={validation_metrics['top1']:.6f} "
            f"val_top3={validation_metrics['top3']:.6f}",
            flush=True,
        )
        improved = validation_metrics["top1"] > best_top1 + 1.0e-12 or (
            abs(validation_metrics["top1"] - best_top1) <= 1.0e-12
            and validation_metrics["cross_entropy"] < best_loss
        )
        if improved:
            best_top1 = validation_metrics["top1"]
            best_loss = validation_metrics["cross_entropy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= MIN_EPOCHS and stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("no best checkpoint was produced")
    model.load_state_dict(best_state)
    metrics, hits, predicted = _evaluate(model, validation)
    return model, history, metrics, hits, predicted, best_epoch


def _subset_row(
    record_type: str,
    subset: str,
    indices: np.ndarray,
    hits: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    if len(indices) == 0:
        values = {"count": 0, "top1": math.nan, "top2": math.nan, "top3": math.nan}
    else:
        values = {
            "count": int(len(indices)),
            "top1": float(hits[indices, 0].mean()),
            "top2": float(hits[indices, 1].mean()),
            "top3": float(hits[indices, 2].mean()),
        }
    return {"record_type": record_type, "subset": subset, **values, **extra}


def _diagnostic_rows(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], hits: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = np.asarray([row["candidate_count"] for row in validation])
    train_counts = np.asarray([row["candidate_count"] for row in train])
    count_edges = np.quantile(train_counts, (0.25, 0.5, 0.75))
    count_bins = np.searchsorted(count_edges, counts, side="right")
    for bin_index in range(4):
        rows.append(
            _subset_row(
                "candidate_count_quartile",
                f"Q{bin_index + 1}",
                np.flatnonzero(count_bins == bin_index),
                hits,
                lower=(-math.inf if bin_index == 0 else count_edges[bin_index - 1]),
                upper=(math.inf if bin_index == 3 else count_edges[bin_index]),
            )
        )

    u_values = np.asarray([row["u"] for row in validation])
    u_edges = np.linspace(0.0, 1.0, 6)
    for bin_index in range(5):
        include = (u_values >= u_edges[bin_index]) & (
            (u_values <= u_edges[bin_index + 1])
            if bin_index == 4
            else (u_values < u_edges[bin_index + 1])
        )
        rows.append(
            _subset_row(
                "u_range",
                f"[{u_edges[bin_index]:.1f},{u_edges[bin_index + 1]:.1f}]",
                np.flatnonzero(include),
                hits,
                lower=u_edges[bin_index],
                upper=u_edges[bin_index + 1],
            )
        )

    sizes = np.asarray([row["cell_size"] for row in validation])
    train_sizes = np.asarray([row["cell_size"] for row in train])
    size_edges = np.quantile(train_sizes, (0.25, 0.5, 0.75))
    size_bins = np.searchsorted(size_edges, sizes, side="right")
    for bin_index in range(4):
        rows.append(
            _subset_row(
                "cell_size_quartile",
                f"Q{bin_index + 1}",
                np.flatnonzero(size_bins == bin_index),
                hits,
                lower=(-math.inf if bin_index == 0 else size_edges[bin_index - 1]),
                upper=(math.inf if bin_index == 3 else size_edges[bin_index]),
            )
        )

    margins = np.asarray(
        [row["teacher_margin"] for row in validation if np.isfinite(row["teacher_margin"])]
    )
    rows.append(
        {
            "record_type": "teacher_margin",
            "count": len(margins),
            "minimum": float(np.min(margins)),
            "p01": float(np.quantile(margins, 0.01)),
            "p10": float(np.quantile(margins, 0.10)),
            "median": float(np.median(margins)),
            "p90": float(np.quantile(margins, 0.90)),
            "fraction_at_most_1e-6": float(np.mean(margins <= 1.0e-6)),
        }
    )
    exact = []
    for row in validation:
        inputs = row["candidate_inputs"]
        deterministic = inputs[:, 2] * inputs[:, 0] - (
            1.0 - inputs[:, 2]
        ) * inputs[:, 1]
        exact.append(int(np.argmax(deterministic)) == int(row["winner_index"]))
    rows.append(
        {
            "record_type": "deterministic_sanity",
            "count": len(validation),
            "top1": float(np.mean(exact)),
            "expected_top1": 1.0,
        }
    )
    return rows


def _error_rows(
    validation: list[dict[str, Any]], predicted: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (example, predicted_index_value) in enumerate(
        zip(validation, predicted, strict=True)
    ):
        teacher_index = int(example["winner_index"])
        predicted_index = int(predicted_index_value)
        if teacher_index == predicted_index:
            continue
        teacher_input = example["candidate_inputs"][teacher_index]
        predicted_input = example["candidate_inputs"][predicted_index]
        teacher_point = example["candidate_points"][teacher_index]
        predicted_point = example["candidate_points"][predicted_index]
        rows.append(
            {
                "validation_cell_index": index,
                "trajectory_id": example["trajectory_id"],
                "iteration": example["iteration"],
                "site_index": example["site_index"],
                "candidate_count": example["candidate_count"],
                "u": example["u"],
                "cell_size": example["cell_size"],
                "teacher_margin": example["teacher_margin"],
                "teacher_winner_index": teacher_index,
                "predicted_winner_index": predicted_index,
                "teacher_normalized_distance": teacher_input[0],
                "teacher_normalized_surrogate": teacher_input[1],
                "predicted_normalized_distance": predicted_input[0],
                "predicted_normalized_surrogate": predicted_input[1],
                "teacher_a_normalized": teacher_point[0],
                "teacher_b_normalized": teacher_point[1],
                "predicted_a_normalized": predicted_point[0],
                "predicted_b_normalized": predicted_point[1],
            }
        )
    return rows


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = _split_state_indices(payload["states"], False, SAMPLE_SEED)
    validation_indices = _split_state_indices(
        payload["states"], True, SAMPLE_SEED + 1
    )
    train, train_info = _collect_examples(payload, train_indices, TRAIN_CELLS)
    validation, validation_info = _collect_examples(
        payload, validation_indices, VALIDATION_CELLS
    )
    started = time.perf_counter()
    model, history, metrics, hits, predicted, best_epoch = _train(train, validation)
    runtime_seconds = time.perf_counter() - started
    strong = bool(metrics["top1"] >= STRONG_TOP1_GATE)

    summary = list(history)
    summary.append(
        {
            "record_type": "quick_screen",
            "best_epoch": best_epoch,
            "count": len(validation),
            **metrics,
            "strong_gate": STRONG_TOP1_GATE,
            "strong": strong,
            "checkpoint_saved": strong,
            "closed_loop_run": False,
            "runtime_seconds": runtime_seconds,
        }
    )
    summary.extend(_diagnostic_rows(train, validation, hits))
    summary.append(
        {
            "record_type": "dataset",
            "train_cells": len(train),
            "validation_cells": len(validation),
            **{f"train_{key}": value for key, value in train_info.items()},
            **{
                f"validation_{key}": value
                for key, value in validation_info.items()
            },
            "largest_cell_training_examples": 0,
            "candidate_input": "normalized_distance|normalized_surrogate|u",
            "final_deterministic_score_provided": False,
        }
    )
    _write_csv(SUMMARY_PATH, summary)
    errors = _error_rows(validation, predicted)
    _write_csv(
        ERRORS_PATH,
        errors,
        default_fields=(
            "validation_cell_index",
            "trajectory_id",
            "iteration",
            "site_index",
            "candidate_count",
            "u",
            "cell_size",
            "teacher_margin",
            "teacher_winner_index",
            "predicted_winner_index",
        ),
    )
    if strong:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": "OrdinaryCellScorer",
                "input_features": (
                    "normalized_distance",
                    "normalized_surrogate_prediction",
                    "u",
                ),
                "best_epoch": best_epoch,
                "validation_metrics": metrics,
                "model_seed": MODEL_SEED,
                "sample_seed": SAMPLE_SEED,
                "largest_cell_behavior": "excluded; remains deterministic",
            },
            CHECKPOINT_PATH,
        )
    print(
        f"top1={metrics['top1']:.6f} top2={metrics['top2']:.6f} "
        f"top3={metrics['top3']:.6f} strong={strong}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
