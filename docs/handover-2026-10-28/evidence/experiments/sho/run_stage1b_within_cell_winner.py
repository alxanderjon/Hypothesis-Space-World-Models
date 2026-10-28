#!/usr/bin/env python3
"""Imitate only eDIRECT's within-Voronoi-cell winner operation.

The quick screen is deliberately self-contained.  It reconstructs selected
Voronoi cells from the persisted successful eDIRECT traces, preserves every
eligible candidate in each cell, and trains one permutation-equivariant
scorer to identify the exact candidate chosen by eDIRECT.  The already
combined deterministic eDIRECT utility is used only to form the label and is
never supplied as a feature.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX2")

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_learned_candidate_filter as filter_data
import experiments.sho.run_stage1b_specialized_cell_filter as specialized
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import SEED


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/within_cell_winner")
DATASET_PATH = OUTPUT_DIR / "within_cell_quick_teacher_dataset.pt"
CHECKPOINT_PATH = OUTPUT_DIR / "best_quick_checkpoint.pt"
SUMMARY_PATH = OUTPUT_DIR / "within_cell_winner_summary.csv"
MANIFEST_PATH = OUTPUT_DIR / "within_cell_dataset_manifest.csv"
DIAGNOSTICS_PATH = OUTPUT_DIR / "quick_screen_diagnostics.csv"

RAW_TRACE_PATH = filter_data.RAW_TRACE_PATH
FROZEN_SELECTOR_PATH = specialized.FROZEN_SELECTOR_PATH
SOURCE_SELECTOR_PATH = specialized.SOURCE_SELECTOR

TRAIN_CELLS = 3_000
VALIDATION_CELLS = 750
MODEL_SEED = SEED + 16_300_000
SAMPLE_SEED = SEED + 16_400_000
HIDDEN_DIM = 32
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
MAX_EPOCHS = 20
PATIENCE = 4

# A strong quick imitation screen must be good for both qualitatively distinct
# branches of the teacher, not merely for the majority ordinary-cell branch.
TOP1_GATE = 0.95
ORDINARY_TOP1_GATE = 0.95
LARGEST_TOP1_GATE = 0.98

CANDIDATE_FEATURE_NAMES = (
    "candidate_a_normalized",
    "candidate_b_normalized",
    "distance_to_site",
    "surrogate_log10_loss",
    "within_cell_normalized_distance",
    "within_cell_normalized_surrogate",
)
CELL_FEATURE_NAMES = (
    "site_a_normalized",
    "site_b_normalized",
    "site_log10_loss",
    "cell_size",
    "cell_size_among_selected_normalized",
    "loo_error",
    "loo_error_among_selected_normalized",
    "largest_selected_cell",
    "log1p_candidate_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
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


def _cell_examples(
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
    teacher_weights = 0.5 * normalized_sizes + 0.5 * normalized_loo
    largest_position = int(np.argmax(selected_sizes))
    teacher_weights[largest_position] = 1.0

    reconstructed: list[np.ndarray] = []
    examples: list[dict[str, Any]] = []
    for selected_position, site_index_value in enumerate(data.selected):
        site_index = int(site_index_value)
        mask = data.owners == site_index
        candidates = data.points[mask]
        distances = data.distances[mask]
        if len(candidates) == 0:
            continue
        predictions = surface.predict(candidates)
        normalized_distance = edirect._normalize(distances)
        normalized_prediction = edirect._normalize(predictions)
        is_largest = selected_position == largest_position
        if is_largest:
            winner = int(np.argmax(distances))
        else:
            weight = float(teacher_weights[selected_position])
            # Label construction only.  This combined score is intentionally
            # absent from candidate_features and from persisted model inputs.
            teacher_score = (
                weight * normalized_distance
                - (1.0 - weight) * normalized_prediction
            )
            winner = int(np.argmax(teacher_score))
        winner_point = candidates[winner]
        duplicate = bool(
            np.min(np.linalg.norm(points - winner_point[None, :], axis=1))
            <= 1.0e-12
        )
        if not duplicate:
            reconstructed.append(winner_point)

        candidate_features = np.column_stack(
            (
                candidates,
                distances,
                predictions,
                normalized_distance,
                normalized_prediction,
            )
        ).astype(np.float32)
        cell_features = np.asarray(
            (
                points[site_index, 0],
                points[site_index, 1],
                math.log10(max(float(losses[site_index]), edirect.LOSS_FLOOR)),
                data.sizes[site_index],
                normalized_sizes[selected_position],
                loo_errors[selected_position],
                normalized_loo[selected_position],
                float(is_largest),
                math.log1p(len(candidates)),
            ),
            dtype=np.float32,
        )
        examples.append(
            {
                "trajectory_id": int(record["trajectory_id"]),
                "generator_id": generator_id,
                "iteration": iteration,
                "history_count": history_count,
                "site_index": site_index,
                "candidate_features": candidate_features,
                "cell_features": cell_features,
                "winner_index": winner,
                "winner_point": winner_point.astype(np.float64),
                "candidate_count": int(len(candidates)),
                "cell_size": float(data.sizes[site_index]),
                "largest_selected_cell": bool(is_largest),
                "teacher_winner_duplicate": duplicate,
            }
        )

    expected = np.asarray(record["teacher_curated_points"], dtype=np.float64)
    actual = np.asarray(reconstructed, dtype=np.float64).reshape(-1, 2)
    if len(actual) != len(expected):
        return examples, math.inf
    # Candidate ordering in the saved selector trace is not the selected-cell
    # ordering.  Curation is a set operation, so validate the set in both
    # directions rather than imposing an order that the teacher did not use.
    if len(actual) == 0:
        mismatch = 0.0
    else:
        pairwise = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2)
        mismatch = float(
            max(np.max(np.min(pairwise, axis=0)), np.max(np.min(pairwise, axis=1)))
        )
    return examples, mismatch


def _collect_examples(
    payload: dict[str, Any], state_indices: np.ndarray, target: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trajectories = _trajectory_map(payload)
    result: list[dict[str, Any]] = []
    states_used = 0
    worst_reconstruction_error = 0.0
    for state_index_value in state_indices:
        record = payload["states"][int(state_index_value)]
        examples, mismatch = _cell_examples(
            record, trajectories[int(record["trajectory_id"])]
        )
        if not np.isfinite(mismatch) or mismatch > 1.0e-12:
            raise RuntimeError(
                f"teacher reconstruction failed at state {state_index_value}: {mismatch}"
            )
        worst_reconstruction_error = max(worst_reconstruction_error, mismatch)
        states_used += 1
        result.extend(examples)
        if len(result) >= target:
            result = result[:target]
            break
    if len(result) != target:
        raise RuntimeError(f"requested {target} cells but reconstructed {len(result)}")
    return result, {
        "states_used": states_used,
        "worst_teacher_reconstruction_error": worst_reconstruction_error,
        "largest_cells": sum(row["largest_selected_cell"] for row in result),
        "ordinary_cells": sum(not row["largest_selected_cell"] for row in result),
        "candidate_count_min": min(row["candidate_count"] for row in result),
        "candidate_count_median": float(
            np.median([row["candidate_count"] for row in result])
        ),
        "candidate_count_max": max(row["candidate_count"] for row in result),
    }


def _feature_statistics(
    examples: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    candidate_sum = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float64)
    candidate_square_sum = np.zeros_like(candidate_sum)
    candidate_count = 0
    cell = np.stack([row["cell_features"] for row in examples]).astype(np.float64)
    for row in examples:
        values = row["candidate_features"].astype(np.float64)
        candidate_sum += values.sum(axis=0)
        candidate_square_sum += np.square(values).sum(axis=0)
        candidate_count += len(values)
    candidate_mean = candidate_sum / candidate_count
    candidate_variance = np.maximum(
        candidate_square_sum / candidate_count - np.square(candidate_mean), 1.0e-12
    )
    return {
        "candidate_mean": candidate_mean.astype(np.float32),
        "candidate_std": np.sqrt(candidate_variance).astype(np.float32),
        "cell_mean": cell.mean(axis=0).astype(np.float32),
        "cell_std": np.maximum(cell.std(axis=0), 1.0e-6).astype(np.float32),
    }


class WithinCellScorer(nn.Module):
    """Shared candidate encoder plus pooled, permutation-invariant context."""

    def __init__(self) -> None:
        super().__init__()
        candidate_dim = len(CANDIDATE_FEATURE_NAMES)
        cell_dim = len(CELL_FEATURE_NAMES)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(
            nn.Linear(4 * HIDDEN_DIM + cell_dim + candidate_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        cell_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.candidate_encoder(candidate_features)
        weights = candidate_mask[:, :, None].to(encoded.dtype)
        mean = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        negative = torch.finfo(encoded.dtype).min
        positive = torch.finfo(encoded.dtype).max
        maximum = torch.where(candidate_mask[:, :, None], encoded, negative).max(
            dim=1
        ).values
        minimum = torch.where(candidate_mask[:, :, None], encoded, positive).min(
            dim=1
        ).values
        context = torch.cat((mean, maximum, minimum, cell_features), dim=1)
        expanded = context[:, None, :].expand(-1, encoded.shape[1], -1)
        scores = self.score_head(
            torch.cat((candidate_features, encoded, expanded), dim=2)
        ).squeeze(2)
        return scores.masked_fill(~candidate_mask, -torch.inf)


def _batches(count: int, shuffle: bool, seed: int) -> list[np.ndarray]:
    indices = np.arange(count)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    return [indices[start : start + BATCH_SIZE] for start in range(0, count, BATCH_SIZE)]


def _collate(
    examples: list[dict[str, Any]],
    indices: np.ndarray,
    statistics: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [examples[int(index)] for index in indices]
    # Fixed padding makes all regular batches use identical GEMM shapes.  It
    # is also comfortably above the observed quick-screen maximum (440).
    maximum = max(512, max(row["candidate_count"] for row in rows))
    candidates = np.zeros(
        (len(rows), maximum, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32
    )
    mask = np.zeros((len(rows), maximum), dtype=np.bool_)
    cells = np.stack([row["cell_features"] for row in rows]).astype(np.float32)
    winners = np.asarray([row["winner_index"] for row in rows], dtype=np.int64)
    for row_index, row in enumerate(rows):
        count = row["candidate_count"]
        candidates[row_index, :count] = row["candidate_features"]
        mask[row_index, :count] = True
    candidates = (candidates - statistics["candidate_mean"]) / statistics[
        "candidate_std"
    ]
    cells = (cells - statistics["cell_mean"]) / statistics["cell_std"]
    return (
        torch.from_numpy(candidates),
        torch.from_numpy(mask),
        torch.from_numpy(cells),
        torch.from_numpy(winners),
    )


def _evaluate_predictions(
    model: WithinCellScorer,
    examples: list[dict[str, Any]],
    statistics: dict[str, np.ndarray],
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    hits = np.zeros((len(examples), 3), dtype=np.bool_)
    losses: list[float] = []
    with torch.no_grad():
        for indices in _batches(len(examples), False, 0):
            candidates, mask, cells, winners = _collate(examples, indices, statistics)
            scores = model(candidates, mask, cells)
            losses.append(
                float(nn.functional.cross_entropy(scores, winners).item()) * len(indices)
            )
            order = torch.topk(scores, k=min(3, scores.shape[1]), dim=1).indices
            for local_row, global_index in enumerate(indices):
                match = order[local_row] == winners[local_row]
                for k in range(3):
                    hits[int(global_index), k] = bool(match[: min(k + 1, len(match))].any())
    metrics = {
        "cross_entropy": float(sum(losses) / len(examples)),
        "top1": float(hits[:, 0].mean()),
        "top2": float(hits[:, 1].mean()),
        "top3": float(hits[:, 2].mean()),
    }
    return metrics, hits


def _subset_metrics(hits: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if len(indices) == 0:
        return {"count": 0, "top1": math.nan, "top2": math.nan, "top3": math.nan}
    return {
        "count": int(len(indices)),
        "top1": float(hits[indices, 0].mean()),
        "top2": float(hits[indices, 1].mean()),
        "top3": float(hits[indices, 2].mean()),
    }


def _teacher_diagnostic_rows(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exact_hits: list[bool] = []
    farthest_hits: list[bool] = []
    surrogate_hits: list[bool] = []
    margins: list[float] = []
    largest: list[bool] = []
    for row in examples:
        features = row["candidate_features"]
        winner = int(row["winner_index"])
        is_largest = bool(row["largest_selected_cell"])
        weight = 0.5 * (
            float(row["cell_features"][4]) + float(row["cell_features"][6])
        )
        utility = (
            features[:, 4]
            if is_largest
            else weight * features[:, 4] - (1.0 - weight) * features[:, 5]
        )
        exact_hits.append(int(np.argmax(utility)) == winner)
        farthest_hits.append(int(np.argmax(features[:, 2])) == winner)
        surrogate_hits.append(int(np.argmin(features[:, 3])) == winner)
        ordered = np.sort(utility)
        margins.append(
            math.inf if len(ordered) == 1 else float(ordered[-1] - ordered[-2])
        )
        largest.append(is_largest)
    finite_margins = np.asarray([value for value in margins if np.isfinite(value)])
    largest_mask = np.asarray(largest, dtype=np.bool_)
    farthest = np.asarray(farthest_hits, dtype=np.bool_)
    surrogate = np.asarray(surrogate_hits, dtype=np.bool_)
    return [
        {
            "diagnostic": "teacher_rule_reconstruction",
            "count": len(examples),
            "accuracy": float(np.mean(exact_hits)),
            "meaning": "input-derived teacher utility reproduces persisted winner",
        },
        {
            "diagnostic": "farthest_candidate_baseline",
            "count": len(examples),
            "accuracy": float(farthest.mean()),
            "ordinary_accuracy": float(farthest[~largest_mask].mean()),
            "largest_accuracy": float(farthest[largest_mask].mean()),
        },
        {
            "diagnostic": "minimum_surrogate_baseline",
            "count": len(examples),
            "accuracy": float(surrogate.mean()),
            "ordinary_accuracy": float(surrogate[~largest_mask].mean()),
            "largest_accuracy": float(surrogate[largest_mask].mean()),
        },
        {
            "diagnostic": "teacher_winner_utility_margin",
            "count": len(finite_margins),
            "minimum": float(np.min(finite_margins)),
            "p01": float(np.quantile(finite_margins, 0.01)),
            "p10": float(np.quantile(finite_margins, 0.10)),
            "median": float(np.median(finite_margins)),
            "p90": float(np.quantile(finite_margins, 0.90)),
            "fraction_at_most_1e-6": float(np.mean(finite_margins <= 1.0e-6)),
            "meaning": "no exact/near tie at 1e-6; low top-1 is not label ambiguity",
        },
    ]


def analyze_saved() -> None:
    saved = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = WithinCellScorer().to(torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_metrics, hits = _evaluate_predictions(
        model, saved["validation"], saved["statistics"]
    )
    largest = np.asarray(
        [row["largest_selected_cell"] for row in saved["validation"]], dtype=np.bool_
    )
    rows = _teacher_diagnostic_rows(saved["validation"])
    rows.extend(
        (
            {
                "diagnostic": "trained_model_overall",
                "count": len(saved["validation"]),
                "accuracy": validation_metrics["top1"],
                "top2": validation_metrics["top2"],
                "top3": validation_metrics["top3"],
            },
            {
                "diagnostic": "trained_model_ordinary",
                **_subset_metrics(hits, np.flatnonzero(~largest)),
            },
            {
                "diagnostic": "trained_model_largest",
                **_subset_metrics(hits, np.flatnonzero(largest)),
            },
        )
    )
    _write_csv(DIAGNOSTICS_PATH, rows)
    print(f"wrote {DIAGNOSTICS_PATH}")


def _train_quick(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    statistics: dict[str, np.ndarray],
) -> tuple[WithinCellScorer, list[dict[str, Any]], dict[str, float], np.ndarray, int]:
    torch.manual_seed(MODEL_SEED)
    model = WithinCellScorer().to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -math.inf
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for indices in _batches(len(train), True, MODEL_SEED + epoch):
            candidates, mask, cells, winners = _collate(train, indices, statistics)
            optimizer.zero_grad(set_to_none=True)
            scores = model(candidates, mask, cells)
            loss = nn.functional.cross_entropy(scores, winners)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(indices)
        validation_metrics, _ = _evaluate_predictions(model, validation, statistics)
        row = {
            "record_type": "epoch",
            "epoch": epoch,
            "train_cross_entropy": train_loss / len(train),
            "validation_cross_entropy": validation_metrics["cross_entropy"],
            "validation_top1": validation_metrics["top1"],
            "validation_top2": validation_metrics["top2"],
            "validation_top3": validation_metrics["top3"],
        }
        history.append(row)
        print(
            f"epoch={epoch} validation_top1={validation_metrics['top1']:.6f} "
            f"top3={validation_metrics['top3']:.6f}",
            flush=True,
        )
        is_better = validation_metrics["top1"] > best_accuracy + 1.0e-12 or (
            abs(validation_metrics["top1"] - best_accuracy) <= 1.0e-12
            and validation_metrics["cross_entropy"] < best_loss
        )
        if is_better:
            best_accuracy = validation_metrics["top1"]
            best_loss = validation_metrics["cross_entropy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics, hits = _evaluate_predictions(model, validation, statistics)
    return model, history, metrics, hits, best_epoch


def _summary_rows(
    history: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    validation_metrics: dict[str, float],
    hits: np.ndarray,
    best_epoch: int,
    train_info: dict[str, Any],
    validation_info: dict[str, Any],
    selector_hash_before: str,
    selector_hash_after: str,
) -> tuple[list[dict[str, Any]], bool]:
    rows = list(history)
    largest = np.asarray(
        [row["largest_selected_cell"] for row in validation], dtype=np.bool_
    )
    ordinary_metrics = _subset_metrics(hits, np.flatnonzero(~largest))
    largest_metrics = _subset_metrics(hits, np.flatnonzero(largest))
    gate_passed = bool(
        validation_metrics["top1"] >= TOP1_GATE
        and ordinary_metrics["top1"] >= ORDINARY_TOP1_GATE
        and largest_metrics["top1"] >= LARGEST_TOP1_GATE
    )
    rows.append(
        {
            "record_type": "quick_screen_overall",
            "best_epoch": best_epoch,
            "count": len(validation),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "gate_passed": gate_passed,
            "gate_top1": TOP1_GATE,
            "gate_ordinary_top1": ORDINARY_TOP1_GATE,
            "gate_largest_top1": LARGEST_TOP1_GATE,
            "next_action": "full_training" if gate_passed else "stop_after_quick_screen",
        }
    )
    for name, metrics in (("ordinary", ordinary_metrics), ("largest", largest_metrics)):
        rows.append({"record_type": "cell_kind", "subset": name, **metrics})

    sizes = np.asarray([row["cell_size"] for row in validation], dtype=np.float64)
    train_sizes = np.asarray(
        [row["cell_size"] for row in train_examples_global], dtype=np.float64
    )
    boundaries = np.quantile(train_sizes, (0.25, 0.5, 0.75))
    bins = np.searchsorted(boundaries, sizes, side="right")
    for bin_index in range(4):
        indices = np.flatnonzero(bins == bin_index)
        lower = -math.inf if bin_index == 0 else float(boundaries[bin_index - 1])
        upper = math.inf if bin_index == 3 else float(boundaries[bin_index])
        rows.append(
            {
                "record_type": "cell_size_quartile",
                "subset": f"Q{bin_index + 1}",
                "cell_size_lower": lower,
                "cell_size_upper": upper,
                **_subset_metrics(hits, indices),
            }
        )
    rows.append(
        {
            "record_type": "dataset",
            "train_cells": TRAIN_CELLS,
            "validation_cells": VALIDATION_CELLS,
            **{f"train_{key}": value for key, value in train_info.items()},
            **{f"validation_{key}": value for key, value in validation_info.items()},
            "candidate_features": "|".join(CANDIDATE_FEATURE_NAMES),
            "cell_features": "|".join(CELL_FEATURE_NAMES),
            "deterministic_combined_score_provided": False,
        }
    )
    rows.append(
        {
            "record_type": "frozen_selector_integrity",
            "selector_sha256_before": selector_hash_before,
            "selector_sha256_after": selector_hash_after,
            "selector_byte_identical": selector_hash_before == selector_hash_after,
        }
    )
    return rows, gate_passed


# Used only by _summary_rows so the public function signature stays compact.
train_examples_global: list[dict[str, Any]] = []


def run(force_dataset: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selector_hash_before = _sha256(FROZEN_SELECTOR_PATH)
    source_selector_hash = _sha256(SOURCE_SELECTOR_PATH)
    if selector_hash_before != source_selector_hash:
        raise RuntimeError("frozen selector is not byte-identical to its source checkpoint")
    if DATASET_PATH.exists() and not force_dataset:
        saved = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
        train = saved["train"]
        validation = saved["validation"]
        statistics = saved["statistics"]
        train_info = saved["train_info"]
        validation_info = saved["validation_info"]
    else:
        raw = torch.load(RAW_TRACE_PATH, map_location="cpu", weights_only=False)
        train_indices = _split_state_indices(raw["states"], False, SAMPLE_SEED)
        validation_indices = _split_state_indices(raw["states"], True, SAMPLE_SEED + 1)
        train, train_info = _collect_examples(raw, train_indices, TRAIN_CELLS)
        validation, validation_info = _collect_examples(
            raw, validation_indices, VALIDATION_CELLS
        )
        statistics = _feature_statistics(train)
        torch.save(
            {
                "config": {
                    "operation": "within_selected_voronoi_cell_winner_only",
                    "train_cells": TRAIN_CELLS,
                    "validation_cells": VALIDATION_CELLS,
                    "sample_seed": SAMPLE_SEED,
                    "candidate_features": CANDIDATE_FEATURE_NAMES,
                    "cell_features": CELL_FEATURE_NAMES,
                    "combined_teacher_score_is_input": False,
                },
                "train": train,
                "validation": validation,
                "statistics": statistics,
                "train_info": train_info,
                "validation_info": validation_info,
            },
            DATASET_PATH,
        )
    global train_examples_global
    train_examples_global = train
    started = time.perf_counter()
    model, history, metrics, hits, best_epoch = _train_quick(
        train, validation, statistics
    )
    elapsed = time.perf_counter() - started
    selector_hash_after = _sha256(FROZEN_SELECTOR_PATH)
    rows, gate_passed = _summary_rows(
        history,
        validation,
        metrics,
        hits,
        best_epoch,
        train_info,
        validation_info,
        selector_hash_before,
        selector_hash_after,
    )
    rows.append(
        {
            "record_type": "runtime",
            "quick_training_seconds": elapsed,
            "full_training_run": False,
            "assembled_search_run": False,
            "reason": (
                "quick gate passed; full phase requires conditional continuation"
                if gate_passed
                else "quick top-1 gate not met; stopped as requested"
            ),
        }
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "statistics": statistics,
            "architecture": "WithinCellScorer",
            "best_epoch": best_epoch,
            "validation_metrics": metrics,
            "quick_gate_passed": gate_passed,
            "candidate_feature_names": CANDIDATE_FEATURE_NAMES,
            "cell_feature_names": CELL_FEATURE_NAMES,
            "model_seed": MODEL_SEED,
        },
        CHECKPOINT_PATH,
    )
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(
        MANIFEST_PATH,
        [
            {
                "split": split,
                "cell_index": index,
                "trajectory_id": row["trajectory_id"],
                "iteration": row["iteration"],
                "site_index": row["site_index"],
                "candidate_count": row["candidate_count"],
                "cell_size": row["cell_size"],
                "largest_selected_cell": row["largest_selected_cell"],
                "winner_index": row["winner_index"],
                "winner_a_normalized": row["winner_point"][0],
                "winner_b_normalized": row["winner_point"][1],
            }
            for split, examples in (("train", train), ("validation", validation))
            for index, row in enumerate(examples)
        ],
    )
    print(
        f"quick screen: top1={metrics['top1']:.6f}, top2={metrics['top2']:.6f}, "
        f"top3={metrics['top3']:.6f}, gate_passed={gate_passed}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--diagnostics-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.diagnostics_only:
        analyze_saved()
    else:
        run(force_dataset=arguments.force_dataset)


if __name__ == "__main__":
    main()
