#!/usr/bin/env python3
"""Compare cross-product and slope-difference hull-turn score regression."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_continuous_score_comparison")
COMPARISON_PATH = OUTPUT_DIR / "hull_continuous_score_comparison.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "hull_reconstruction_comparison.csv"

MAX_EPOCHS = 75
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 32
OFFLINE_GATE = 0.99
MODEL_SEED = 1_414_213
DATA_SEED = turn.DATA_SEED


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


def _continuous_dataset(
    selected: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(selected["label"])
    cross_inputs = np.empty((count, 4), dtype=np.float64)
    slope_inputs = np.empty((count, 2), dtype=np.float64)
    cross_target = np.empty(count, dtype=np.float64)
    slope_target = np.empty(count, dtype=np.float64)
    cross_tolerance = np.empty(count, dtype=np.float64)
    slope_tolerance = np.empty(count, dtype=np.float64)
    for row in range(count):
        state = states[int(selected["state_position"][row])]
        sizes = state["sizes"]
        losses = state["losses"]
        first, second, third = (
            int(value) for value in selected["triple_indices"][row]
        )
        size_scale, loss_scale = turn._state_scales(sizes, losses)
        delta_d_ab = float(sizes[second] - sizes[first]) / size_scale
        delta_f_ab = float(losses[second] - losses[first]) / loss_scale
        delta_d_ac = float(sizes[third] - sizes[first]) / size_scale
        delta_f_ac = float(losses[third] - losses[first]) / loss_scale
        if delta_d_ab <= 0.0 or delta_d_ac <= 0.0:
            raise RuntimeError("same-size grouping did not produce increasing d")
        cross = delta_d_ab * delta_f_ac - delta_f_ab * delta_d_ac
        slope_ab = delta_f_ab / delta_d_ab
        slope_ac = delta_f_ac / delta_d_ac
        slope = slope_ac - slope_ab
        tolerance_cross = float(selected["tolerance"][row]) / (
            size_scale * loss_scale
        )
        tolerance_slope = tolerance_cross / (delta_d_ab * delta_d_ac)
        cross_inputs[row] = (delta_d_ab, delta_f_ab, delta_d_ac, delta_f_ac)
        slope_inputs[row] = (slope_ab, slope_ac)
        cross_target[row] = cross
        slope_target[row] = slope
        cross_tolerance[row] = tolerance_cross
        slope_tolerance[row] = tolerance_slope
    cross_labels = (cross_target >= -cross_tolerance).astype(np.int64)
    slope_labels = (slope_target >= -slope_tolerance).astype(np.int64)
    if not np.array_equal(cross_labels, selected["label"]):
        raise RuntimeError("normalized cross score changed deterministic labels")
    if not np.array_equal(slope_labels, selected["label"]):
        raise RuntimeError("slope score changed deterministic labels")
    return {
        "cross_inputs": cross_inputs,
        "slope_inputs": slope_inputs,
        "cross_target": cross_target,
        "slope_target": slope_target,
        "cross_tolerance": cross_tolerance,
        "slope_tolerance": slope_tolerance,
        "teacher_keep": selected["label"].copy(),
        "raw_cross": selected["cross"].copy(),
        "raw_tolerance": selected["tolerance"].copy(),
        "near_collinear": selected["near_collinear"].copy(),
        "exact_collinear": selected["exact_collinear"].copy(),
        "state_position": selected["state_position"].copy(),
        "triple_indices": selected["triple_indices"].copy(),
    }


@dataclass
class Standardization:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: float
    target_scale: float

    @classmethod
    def fit(cls, inputs: np.ndarray, target: np.ndarray) -> "Standardization":
        input_mean = np.mean(inputs, axis=0)
        input_scale = np.std(inputs, axis=0)
        input_scale = np.where(input_scale > 1.0e-15, input_scale, 1.0)
        target_mean = float(np.mean(target))
        target_scale = float(np.std(target))
        if not np.isfinite(target_scale) or target_scale <= 1.0e-15:
            target_scale = 1.0
        return cls(input_mean, input_scale, target_mean, target_scale)

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_mean) / self.input_scale

    def target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale + self.target_mean


class ScoreMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(target) == 0.0 or np.std(prediction) == 0.0:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    tolerance: np.ndarray,
    teacher_keep: np.ndarray,
) -> dict[str, float | int]:
    error = prediction - target
    predicted_keep = (prediction >= -tolerance).astype(np.int64)
    teacher_sign = target >= 0.0
    predicted_sign = prediction >= 0.0
    return {
        "count": len(target),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "target_standard_deviation": float(np.std(target)),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / max(np.std(target), 1e-300)),
        "correlation": _correlation(target, prediction),
        "sign_accuracy": float(np.mean(teacher_sign == predicted_sign)),
        "tolerance_accuracy": float(np.mean(teacher_keep == predicted_keep)),
        "false_keep": int(np.sum((teacher_keep == 0) & (predicted_keep == 1))),
        "false_pop": int(np.sum((teacher_keep == 1) & (predicted_keep == 0))),
    }


def _predict(
    model: ScoreMLP,
    standardization: Standardization,
    inputs: np.ndarray,
) -> np.ndarray:
    standardized = torch.as_tensor(
        standardization.inputs(inputs), dtype=torch.float64
    )
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(standardized), 2048):
            outputs.append(model(standardized[start : start + 2048]))
    normalized_prediction = torch.cat(outputs).cpu().numpy()
    return standardization.inverse_target(normalized_prediction)


def _paired_train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[
    dict[str, ScoreMLP],
    dict[str, Standardization],
    list[dict[str, Any]],
    dict[str, int],
    int,
]:
    branches = {"cross": ("cross_inputs", "cross_target", 4), "slope": ("slope_inputs", "slope_target", 2)}
    models: dict[str, ScoreMLP] = {}
    standardizations: dict[str, Standardization] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    best_losses = {branch: math.inf for branch in branches}
    best_epochs = {branch: 0 for branch in branches}
    stale = {branch: 0 for branch in branches}
    for branch, (input_key, target_key, input_dim) in branches.items():
        torch.manual_seed(MODEL_SEED)
        model = ScoreMLP(input_dim).to(torch.float64)
        standardization = Standardization.fit(
            training[input_key], training[target_key]
        )
        models[branch] = model
        standardizations[branch] = standardization
        optimizers[branch] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        tensors[branch] = (
            torch.as_tensor(
                standardization.inputs(training[input_key]), dtype=torch.float64
            ),
            torch.as_tensor(
                standardization.target(training[target_key]), dtype=torch.float64
            ),
        )
        best_states[branch] = copy.deepcopy(model.state_dict())

    generator = np.random.default_rng(MODEL_SEED)
    rows: list[dict[str, Any]] = []
    updates = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(training["teacher_keep"]))
        epoch_losses = {branch: 0.0 for branch in branches}
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            for branch in branches:
                inputs, target = tensors[branch]
                model = models[branch]
                model.train()
                predicted = model(inputs[batch])
                loss = nn.functional.mse_loss(predicted, target[batch])
                optimizers[branch].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizers[branch].step()
                epoch_losses[branch] += float(loss.item()) * len(batch)
            updates += 1
        for branch, (input_key, target_key, _) in branches.items():
            prediction = _predict(
                models[branch], standardizations[branch], validation[input_key]
            )
            metrics = _regression_metrics(
                validation[target_key],
                prediction,
                validation[f"{branch}_tolerance"],
                validation["teacher_keep"],
            )
            normalized_error = (
                prediction - validation[target_key]
            ) / standardizations[branch].target_scale
            validation_mse = float(np.mean(normalized_error**2))
            rows.append(
                {
                    "record_type": "epoch",
                    "branch": branch,
                    "epoch": epoch,
                    "shared_update_count": updates,
                    "training_standardized_mse": epoch_losses[branch] / len(order),
                    "validation_standardized_mse": validation_mse,
                    "validation_rmse": metrics["rmse"],
                    "validation_correlation": metrics["correlation"],
                    "validation_sign_accuracy": metrics["sign_accuracy"],
                    "validation_tolerance_accuracy": metrics["tolerance_accuracy"],
                }
            )
            if validation_mse < best_losses[branch] - 1.0e-12:
                best_losses[branch] = validation_mse
                best_epochs[branch] = epoch
                best_states[branch] = copy.deepcopy(models[branch].state_dict())
                stale[branch] = 0
            else:
                stale[branch] += 1
        print(
            f"epoch={epoch} updates={updates} "
            + " ".join(
                f"{branch}_mse={best_losses[branch]:.3e} stale={stale[branch]}"
                for branch in branches
            ),
            flush=True,
        )
        if all(stale[branch] >= PATIENCE for branch in branches):
            break
    for branch in branches:
        models[branch].load_state_dict(best_states[branch])
        models[branch].eval()
    return models, standardizations, rows, best_epochs, updates


def _metric_rows(
    branch: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    best_epoch: int,
    epochs_trained: int,
    updates: int,
) -> list[dict[str, Any]]:
    target = dataset[f"{branch}_target"]
    tolerance = dataset[f"{branch}_tolerance"]
    labels = dataset["teacher_keep"]
    rows = [
        {
            "record_type": "final_summary",
            "branch": branch,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "shared_total_update_count": updates,
            **_regression_metrics(target, prediction, tolerance, labels),
            "near_collinear_count": int(np.sum(dataset["near_collinear"])),
            "exact_collinear_count": int(np.sum(dataset["exact_collinear"])),
        }
    ]
    subset_masks = {
        "near_collinear": dataset["near_collinear"],
        "exact_collinear": dataset["exact_collinear"],
        "non_near": ~dataset["near_collinear"],
    }
    for subset, mask in subset_masks.items():
        if not np.any(mask):
            rows.append(
                {
                    "record_type": "geometry_subset",
                    "branch": branch,
                    "subset": subset,
                    "count": 0,
                }
            )
            continue
        rows.append(
            {
                "record_type": "geometry_subset",
                "branch": branch,
                "subset": subset,
                **_regression_metrics(
                    target[mask], prediction[mask], tolerance[mask], labels[mask]
                ),
            }
        )
    magnitude_order = np.argsort(np.abs(target), kind="stable")
    for quartile, indices in enumerate(np.array_split(magnitude_order, 4), start=1):
        metrics = _regression_metrics(
            target[indices], prediction[indices], tolerance[indices], labels[indices]
        )
        rows.append(
            {
                "record_type": "score_magnitude_quartile",
                "branch": branch,
                "subset": f"Q{quartile}",
                **metrics,
                "minimum_abs_teacher_score": float(np.min(np.abs(target[indices]))),
                "maximum_abs_teacher_score": float(np.max(np.abs(target[indices]))),
            }
        )
    return rows


def _predict_turn(
    branch: str,
    model: ScoreMLP,
    standardization: Standardization,
    sizes: np.ndarray,
    losses: np.ndarray,
    first: int,
    second: int,
    third: int,
    size_scale: float,
    loss_scale: float,
) -> tuple[float, float]:
    delta_d_ab = float(sizes[second] - sizes[first]) / size_scale
    delta_f_ab = float(losses[second] - losses[first]) / loss_scale
    delta_d_ac = float(sizes[third] - sizes[first]) / size_scale
    delta_f_ac = float(losses[third] - losses[first]) / loss_scale
    raw_scale = max(
        1.0,
        abs(float(losses[first])),
        abs(float(losses[second])),
        abs(float(losses[third])),
    )
    cross_tolerance = turn.HULL_TOLERANCE * raw_scale / (
        size_scale * loss_scale
    )
    if branch == "cross":
        inputs = np.asarray(
            [[delta_d_ab, delta_f_ab, delta_d_ac, delta_f_ac]], dtype=np.float64
        )
        tolerance = cross_tolerance
    else:
        slope_ab = delta_f_ab / delta_d_ab
        slope_ac = delta_f_ac / delta_d_ac
        inputs = np.asarray([[slope_ab, slope_ac]], dtype=np.float64)
        tolerance = cross_tolerance / (delta_d_ab * delta_d_ac)
    prediction = float(_predict(model, standardization, inputs)[0])
    return prediction, tolerance


def _network_hull(
    branch: str,
    model: ScoreMLP,
    standardization: Standardization,
    sizes: np.ndarray,
    losses: np.ndarray,
) -> tuple[np.ndarray, int]:
    size_scale, loss_scale = turn._state_scales(sizes, losses)
    stack: list[int] = []
    decisions = 0
    for third in range(len(sizes)):
        while len(stack) >= 2:
            prediction, tolerance = _predict_turn(
                branch,
                model,
                standardization,
                sizes,
                losses,
                stack[-2],
                stack[-1],
                third,
                size_scale,
                loss_scale,
            )
            decisions += 1
            if prediction >= -tolerance:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64), decisions


def _offline_reconstruction(
    branch: str,
    model: ScoreMLP,
    standardization: Standardization,
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_rows: list[dict[str, Any]] = []
    for state in states:
        prediction, decisions = _network_hull(
            branch, model, standardization, state["sizes"], state["losses"]
        )
        teacher = np.asarray(state["teacher_hull"], dtype=np.int64)
        teacher_set = set(int(value) for value in teacher)
        prediction_set = set(int(value) for value in prediction)
        additions = prediction_set - teacher_set
        removals = teacher_set - prediction_set
        state_rows.append(
            {
                "record_type": "state",
                "branch": branch,
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "history_count": state["history_count"],
                "representative_cell_count": len(state["sizes"]),
                "teacher_hull_size": len(teacher),
                "predicted_hull_size": len(prediction),
                "network_turn_decisions": decisions,
                "false_hull_additions": len(additions),
                "false_hull_removals": len(removals),
                "exact_hull": not additions and not removals,
            }
        )

    def aggregate(record_type: str, subset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "record_type": record_type,
            "branch": branch,
            "subset": subset,
            "offline_run": True,
            "state_count": len(rows),
            "exact_hull_states": sum(bool(row["exact_hull"]) for row in rows),
            "exact_hull_fraction": float(np.mean([row["exact_hull"] for row in rows])),
            "average_false_hull_additions": float(
                np.mean([row["false_hull_additions"] for row in rows])
            ),
            "average_false_hull_removals": float(
                np.mean([row["false_hull_removals"] for row in rows])
            ),
            "maximum_false_hull_additions": max(
                int(row["false_hull_additions"]) for row in rows
            ),
            "maximum_false_hull_removals": max(
                int(row["false_hull_removals"]) for row in rows
            ),
        }

    output = [aggregate("final_summary", "all", state_rows)]
    for name, lower, upper in turn.STATE_BINS:
        subset = [
            row
            for row in state_rows
            if lower <= int(row["representative_cell_count"]) <= upper
        ]
        if subset:
            output.append(aggregate("by_representative_cell_count", name, subset))
    output.extend(state_rows)
    return output


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, DATA_SEED
    )
    validation_indices = turn._stratified_state_indices(
        payload["states"], "validation", turn.VALIDATION_SOURCE_STATES, DATA_SEED + 1
    )
    reconstruction_indices = turn._stratified_state_indices(
        payload["states"], "reconstruction", turn.RECONSTRUCTION_STATES, DATA_SEED + 2
    )
    train_states = turn._build_states(payload, train_indices, workers, "train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "validation"
    )
    reconstruction_states = turn._build_states(
        payload, reconstruction_indices, workers, "reconstruction"
    )
    train_pool = turn._flatten_triples(train_states)
    validation_pool = turn._flatten_triples(validation_states)
    train_selected = turn._select_triples(
        train_pool, turn.TRAIN_TRIPLES, DATA_SEED + 3
    )
    validation_selected = turn._select_triples(
        validation_pool, turn.VALIDATION_TRIPLES, DATA_SEED + 4
    )
    training = _continuous_dataset(train_selected, train_states)
    validation = _continuous_dataset(validation_selected, validation_states)
    for branch in ("cross", "slope"):
        values = training[f"{branch}_target"]
        print(
            f"{branch} target min={np.min(values):.3e} median_abs={np.median(np.abs(values)):.3e} "
            f"max={np.max(values):.3e} std={np.std(values):.3e}",
            flush=True,
        )

    models, standardizations, rows, best_epochs, updates = _paired_train(
        training, validation
    )
    epochs_trained = max(int(row["epoch"]) for row in rows)
    predictions: dict[str, np.ndarray] = {}
    final_metrics: dict[str, dict[str, float | int]] = {}
    for branch in ("cross", "slope"):
        predictions[branch] = _predict(
            models[branch],
            standardizations[branch],
            validation[f"{branch}_inputs"],
        )
        metrics = _regression_metrics(
            validation[f"{branch}_target"],
            predictions[branch],
            validation[f"{branch}_tolerance"],
            validation["teacher_keep"],
        )
        final_metrics[branch] = metrics
        rows.extend(
            _metric_rows(
                branch,
                validation,
                predictions[branch],
                best_epochs[branch],
                epochs_trained,
                updates,
            )
        )
        exact_score = validation[f"{branch}_target"]
        exact_labels = (
            exact_score >= -validation[f"{branch}_tolerance"]
        ).astype(np.int64)
        rows.append(
            {
                "record_type": "deterministic_sanity",
                "branch": branch,
                "correct": int(
                    np.sum(exact_labels == validation["teacher_keep"])
                ),
                "total": len(exact_labels),
                "accuracy": float(
                    np.mean(exact_labels == validation["teacher_keep"])
                ),
            }
        )
    _write_csv(COMPARISON_PATH, rows)

    reconstruction_rows: list[dict[str, Any]] = []
    for branch in ("cross", "slope"):
        metrics = final_metrics[branch]
        eligible = bool(
            float(metrics["correlation"]) >= OFFLINE_GATE
            and float(metrics["sign_accuracy"]) >= OFFLINE_GATE
            and float(metrics["tolerance_accuracy"]) >= OFFLINE_GATE
        )
        if not eligible:
            reconstruction_rows.append(
                {
                    "record_type": "final_summary",
                    "branch": branch,
                    "subset": "all",
                    "offline_run": False,
                    "offline_gate": OFFLINE_GATE,
                    "validation_correlation": metrics["correlation"],
                    "validation_sign_accuracy": metrics["sign_accuracy"],
                    "validation_tolerance_accuracy": metrics["tolerance_accuracy"],
                    "skip_reason": "continuous-score accuracy did not pass the predeclared gate",
                }
            )
            continue
        reconstruction_rows.extend(
            _offline_reconstruction(
                branch,
                models[branch],
                standardizations[branch],
                reconstruction_states,
            )
        )
    _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
