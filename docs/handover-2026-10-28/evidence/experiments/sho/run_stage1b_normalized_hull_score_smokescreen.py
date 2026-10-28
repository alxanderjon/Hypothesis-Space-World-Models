#!/usr/bin/env python3
"""Smoke-test normalized cross and angle hull-turn score regression."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/normalized_hull_score_smokescreen")
SUMMARY_PATH = OUTPUT_DIR / "normalized_hull_score_summary.csv"
ERRORS_PATH = OUTPUT_DIR / "normalized_hull_score_errors.csv"

MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 32
MODEL_SEED = 1_732_050
DATA_SEED = turn.DATA_SEED
NORMALIZATION_EPSILON = np.finfo(np.float64).tiny
NEAR_ZERO_THRESHOLD = 1.0e-3
OFFLINE_GATE = 0.99
OFFLINE_STATES = 100


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


def _normalized_dataset(
    selected: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(selected["label"])
    cross_inputs = np.empty((count, 4), dtype=np.float64)
    angle_inputs = np.empty((count, 2), dtype=np.float64)
    cross_target = np.empty(count, dtype=np.float64)
    angle_target = np.empty(count, dtype=np.float64)
    norm_product = np.empty(count, dtype=np.float64)
    normalized_cross_tolerance = np.empty(count, dtype=np.float64)
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
        delta_d_bc = float(sizes[third] - sizes[second]) / size_scale
        delta_f_bc = float(losses[third] - losses[second]) / loss_scale
        if delta_d_ab <= 0.0 or delta_d_bc <= 0.0:
            raise RuntimeError("same-size grouping did not produce increasing d")
        length_ab = math.hypot(delta_d_ab, delta_f_ab)
        length_bc = math.hypot(delta_d_bc, delta_f_bc)
        product = length_ab * length_bc
        cross = delta_d_ab * delta_f_bc - delta_f_ab * delta_d_bc
        slope_ab = delta_f_ab / delta_d_ab
        slope_bc = delta_f_bc / delta_d_bc
        angle = math.atan(slope_bc) - math.atan(slope_ab)
        cross_inputs[row] = (
            delta_d_ab,
            delta_f_ab,
            delta_d_bc,
            delta_f_bc,
        )
        angle_inputs[row] = (slope_ab, slope_bc)
        cross_target[row] = cross / (product + NORMALIZATION_EPSILON)
        angle_target[row] = angle
        norm_product[row] = product
        normalized_cross_tolerance[row] = float(selected["tolerance"][row]) / (
            size_scale * loss_scale
        )
    labels_from_cross = (
        cross_target * (norm_product + NORMALIZATION_EPSILON)
        >= -normalized_cross_tolerance
    ).astype(np.int64)
    labels_from_angle = (
        norm_product * np.sin(angle_target) >= -normalized_cross_tolerance
    ).astype(np.int64)
    if not np.array_equal(labels_from_cross, selected["label"]):
        raise RuntimeError("normalized cross changed deterministic labels")
    if not np.array_equal(labels_from_angle, selected["label"]):
        raise RuntimeError("angle score changed deterministic labels")
    return {
        "cross_inputs": cross_inputs,
        "angle_inputs": angle_inputs,
        "cross_target": cross_target,
        "angle_target": angle_target,
        "norm_product": norm_product,
        "normalized_cross_tolerance": normalized_cross_tolerance,
        "teacher_keep": selected["label"].copy(),
        "near_collinear": selected["near_collinear"].copy(),
        "exact_collinear": selected["exact_collinear"].copy(),
        "state_position": selected["state_position"].copy(),
        "triple_indices": selected["triple_indices"].copy(),
    }


@dataclass
class InputStandardization:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "InputStandardization":
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1.0e-15, scale, 1.0)
        return cls(mean, scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


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


def _predict(
    model: ScoreMLP,
    standardization: InputStandardization,
    inputs: np.ndarray,
) -> np.ndarray:
    values = torch.as_tensor(standardization.apply(inputs), dtype=torch.float64)
    output: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            output.append(model(values[start : start + 2048]))
    return torch.cat(output).cpu().numpy()


def _predicted_orientation(
    branch: str, prediction: np.ndarray, dataset: dict[str, np.ndarray]
) -> np.ndarray:
    if branch == "cross":
        return prediction * (
            dataset["norm_product"] + NORMALIZATION_EPSILON
        )
    return dataset["norm_product"] * np.sin(prediction)


def _correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(target) == 0.0 or np.std(prediction) == 0.0:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _metrics(
    branch: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    indices: np.ndarray | None = None,
) -> dict[str, float | int]:
    if indices is None:
        indices = np.arange(len(prediction), dtype=np.int64)
    target = dataset[f"{branch}_target"][indices]
    predicted = prediction[indices]
    error = predicted - target
    orientation = _predicted_orientation(branch, prediction, dataset)[indices]
    teacher_keep = dataset["teacher_keep"][indices]
    predicted_keep = (
        orientation >= -dataset["normalized_cross_tolerance"][indices]
    ).astype(np.int64)
    return {
        "count": len(indices),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "correlation": _correlation(target, predicted),
        "sign_accuracy": float(np.mean((target >= 0.0) == (predicted >= 0.0))),
        "tolerance_accuracy": float(np.mean(teacher_keep == predicted_keep)),
        "false_keep": int(np.sum((teacher_keep == 0) & (predicted_keep == 1))),
        "false_pop": int(np.sum((teacher_keep == 1) & (predicted_keep == 0))),
    }


def _paired_train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[
    dict[str, ScoreMLP],
    dict[str, InputStandardization],
    list[dict[str, Any]],
    dict[str, int],
    int,
]:
    specifications = {
        "cross": ("cross_inputs", "cross_target", 4),
        "angle": ("angle_inputs", "angle_target", 2),
    }
    models: dict[str, ScoreMLP] = {}
    scalers: dict[str, InputStandardization] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    best_losses = {branch: math.inf for branch in specifications}
    best_epochs = {branch: 0 for branch in specifications}
    stale = {branch: 0 for branch in specifications}
    for branch, (input_key, target_key, input_dim) in specifications.items():
        torch.manual_seed(MODEL_SEED)
        model = ScoreMLP(input_dim).to(torch.float64)
        scaler = InputStandardization.fit(training[input_key])
        models[branch] = model
        scalers[branch] = scaler
        optimizers[branch] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        tensors[branch] = (
            torch.as_tensor(scaler.apply(training[input_key]), dtype=torch.float64),
            torch.as_tensor(training[target_key], dtype=torch.float64),
        )
        best_states[branch] = copy.deepcopy(model.state_dict())

    generator = np.random.default_rng(MODEL_SEED)
    rows: list[dict[str, Any]] = []
    shared_updates = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(training["teacher_keep"]))
        train_loss = {branch: 0.0 for branch in specifications}
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            for branch in specifications:
                inputs, target = tensors[branch]
                model = models[branch]
                model.train()
                loss = nn.functional.mse_loss(model(inputs[batch]), target[batch])
                optimizers[branch].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizers[branch].step()
                train_loss[branch] += float(loss.item()) * len(batch)
            shared_updates += 1
        for branch, (input_key, target_key, _) in specifications.items():
            prediction = _predict(models[branch], scalers[branch], validation[input_key])
            validation_mse = float(
                np.mean((prediction - validation[target_key]) ** 2)
            )
            metrics = _metrics(branch, validation, prediction)
            rows.append(
                {
                    "record_type": "epoch",
                    "branch": branch,
                    "epoch": epoch,
                    "shared_update_count": shared_updates,
                    "training_mse": train_loss[branch] / len(order),
                    "validation_mse": validation_mse,
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
            f"epoch={epoch} updates={shared_updates} "
            + " ".join(
                f"{branch}_mse={best_losses[branch]:.3e} stale={stale[branch]}"
                for branch in specifications
            ),
            flush=True,
        )
        if all(stale[branch] >= PATIENCE for branch in specifications):
            break
    for branch in specifications:
        models[branch].load_state_dict(best_states[branch])
        models[branch].eval()
    return models, scalers, rows, best_epochs, shared_updates


def _summary_rows(
    branch: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    best_epoch: int,
    epochs_trained: int,
    updates: int,
) -> list[dict[str, Any]]:
    target = dataset[f"{branch}_target"]
    full_indices = np.arange(len(target), dtype=np.int64)
    rows = [
        {
            "record_type": "final_summary",
            "branch": branch,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "shared_total_update_count": updates,
            **_metrics(branch, dataset, prediction),
            "target_minimum": float(np.min(target)),
            "target_maximum": float(np.max(target)),
            "target_mean": float(np.mean(target)),
            "target_standard_deviation": float(np.std(target)),
            "target_median_absolute": float(np.median(np.abs(target))),
            "target_q01": float(np.quantile(target, 0.01)),
            "target_q25": float(np.quantile(target, 0.25)),
            "target_q50": float(np.quantile(target, 0.50)),
            "target_q75": float(np.quantile(target, 0.75)),
            "target_q99": float(np.quantile(target, 0.99)),
            "normalization_epsilon": NORMALIZATION_EPSILON,
            "near_zero_threshold": NEAR_ZERO_THRESHOLD,
            "near_zero_count": int(np.sum(np.abs(target) <= NEAR_ZERO_THRESHOLD)),
            "near_collinear_count": int(np.sum(dataset["near_collinear"])),
            "exact_collinear_count": int(np.sum(dataset["exact_collinear"])),
        }
    ]
    subsets = {
        "near_zero": np.flatnonzero(np.abs(target) <= NEAR_ZERO_THRESHOLD),
        "near_collinear": np.flatnonzero(dataset["near_collinear"]),
        "non_near_zero": np.flatnonzero(np.abs(target) > NEAR_ZERO_THRESHOLD),
    }
    for name, indices in subsets.items():
        if len(indices) == 0:
            rows.append(
                {
                    "record_type": "target_subset",
                    "branch": branch,
                    "subset": name,
                    "count": 0,
                }
            )
        else:
            rows.append(
                {
                    "record_type": "target_subset",
                    "branch": branch,
                    "subset": name,
                    **_metrics(branch, dataset, prediction, indices),
                }
            )
    magnitude_order = np.argsort(np.abs(target), kind="stable")
    for quartile, indices in enumerate(np.array_split(magnitude_order, 4), start=1):
        rows.append(
            {
                "record_type": "target_magnitude_quartile",
                "branch": branch,
                "subset": f"Q{quartile}",
                **_metrics(branch, dataset, prediction, indices),
                "minimum_abs_target": float(np.min(np.abs(target[indices]))),
                "maximum_abs_target": float(np.max(np.abs(target[indices]))),
            }
        )
    exact_orientation = _predicted_orientation(branch, target, dataset)
    exact_keep = (
        exact_orientation >= -dataset["normalized_cross_tolerance"]
    ).astype(np.int64)
    rows.append(
        {
            "record_type": "deterministic_sanity",
            "branch": branch,
            "correct": int(np.sum(exact_keep == dataset["teacher_keep"])),
            "total": len(full_indices),
            "accuracy": float(np.mean(exact_keep == dataset["teacher_keep"])),
        }
    )
    return rows


def _network_orientation(
    branch: str,
    model: ScoreMLP,
    scaler: InputStandardization,
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
    delta_d_bc = float(sizes[third] - sizes[second]) / size_scale
    delta_f_bc = float(losses[third] - losses[second]) / loss_scale
    length_product = math.hypot(delta_d_ab, delta_f_ab) * math.hypot(
        delta_d_bc, delta_f_bc
    )
    raw_scale = max(
        1.0,
        abs(float(losses[first])),
        abs(float(losses[second])),
        abs(float(losses[third])),
    )
    tolerance = turn.HULL_TOLERANCE * raw_scale / (size_scale * loss_scale)
    if branch == "cross":
        inputs = np.asarray(
            [[delta_d_ab, delta_f_ab, delta_d_bc, delta_f_bc]], dtype=np.float64
        )
        score = float(_predict(model, scaler, inputs)[0])
        orientation = score * (length_product + NORMALIZATION_EPSILON)
    else:
        inputs = np.asarray(
            [[delta_f_ab / delta_d_ab, delta_f_bc / delta_d_bc]], dtype=np.float64
        )
        score = float(_predict(model, scaler, inputs)[0])
        orientation = length_product * math.sin(score)
    return orientation, tolerance


def _network_hull(
    branch: str,
    model: ScoreMLP,
    scaler: InputStandardization,
    sizes: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    size_scale, loss_scale = turn._state_scales(sizes, losses)
    stack: list[int] = []
    for third in range(len(sizes)):
        while len(stack) >= 2:
            orientation, tolerance = _network_orientation(
                branch,
                model,
                scaler,
                sizes,
                losses,
                stack[-2],
                stack[-1],
                third,
                size_scale,
                loss_scale,
            )
            if orientation >= -tolerance:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64)


def _offline_rows(
    branch: str,
    model: ScoreMLP,
    scaler: InputStandardization,
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        predicted = _network_hull(
            branch, model, scaler, state["sizes"], state["losses"]
        )
        teacher = np.asarray(state["teacher_hull"], dtype=np.int64)
        predicted_set = set(int(value) for value in predicted)
        teacher_set = set(int(value) for value in teacher)
        additions = predicted_set - teacher_set
        removals = teacher_set - predicted_set
        rows.append(
            {
                "record_type": "offline_state",
                "branch": branch,
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "history_count": state["history_count"],
                "representative_cell_count": len(state["sizes"]),
                "teacher_hull_size": len(teacher),
                "predicted_hull_size": len(predicted),
                "exact_hull": not additions and not removals,
                "false_hull_additions": len(additions),
                "false_hull_removals": len(removals),
            }
        )
    summary = {
        "record_type": "offline_summary",
        "branch": branch,
        "state_count": len(rows),
        "exact_hull_states": sum(bool(row["exact_hull"]) for row in rows),
        "exact_hull_fraction": float(np.mean([row["exact_hull"] for row in rows])),
        "average_false_hull_additions": float(
            np.mean([row["false_hull_additions"] for row in rows])
        ),
        "average_false_hull_removals": float(
            np.mean([row["false_hull_removals"] for row in rows])
        ),
    }
    output = [summary]
    for name, lower, upper in turn.STATE_BINS:
        subset = [
            row
            for row in rows
            if lower <= int(row["representative_cell_count"]) <= upper
        ]
        if subset:
            output.append(
                {
                    "record_type": "offline_by_cell_count",
                    "branch": branch,
                    "subset": name,
                    "state_count": len(subset),
                    "exact_hull_states": sum(
                        bool(row["exact_hull"]) for row in subset
                    ),
                    "exact_hull_fraction": float(
                        np.mean([row["exact_hull"] for row in subset])
                    ),
                    "average_false_hull_additions": float(
                        np.mean([row["false_hull_additions"] for row in subset])
                    ),
                    "average_false_hull_removals": float(
                        np.mean([row["false_hull_removals"] for row in subset])
                    ),
                }
            )
    output.extend(rows)
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
    train_states = turn._build_states(payload, train_indices, workers, "train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "validation"
    )
    train_pool = turn._flatten_triples(train_states)
    validation_pool = turn._flatten_triples(validation_states)
    train_selected = turn._select_triples(
        train_pool, turn.TRAIN_TRIPLES, DATA_SEED + 3
    )
    validation_selected = turn._select_triples(
        validation_pool, turn.VALIDATION_TRIPLES, DATA_SEED + 4
    )
    training = _normalized_dataset(train_selected, train_states)
    validation = _normalized_dataset(validation_selected, validation_states)
    for branch in ("cross", "angle"):
        target = training[f"{branch}_target"]
        print(
            f"{branch} range=[{np.min(target):.6e},{np.max(target):.6e}] "
            f"median_abs={np.median(np.abs(target)):.6e}",
            flush=True,
        )

    models, scalers, rows, best_epochs, updates = _paired_train(
        training, validation
    )
    epochs_trained = max(int(row["epoch"]) for row in rows)
    predictions: dict[str, np.ndarray] = {}
    final_metrics: dict[str, dict[str, float | int]] = {}
    error_rows: list[dict[str, Any]] = []
    eligible: list[str] = []
    for branch in ("cross", "angle"):
        predictions[branch] = _predict(
            models[branch], scalers[branch], validation[f"{branch}_inputs"]
        )
        final_metrics[branch] = _metrics(branch, validation, predictions[branch])
        rows.extend(
            _summary_rows(
                branch,
                validation,
                predictions[branch],
                best_epochs[branch],
                epochs_trained,
                updates,
            )
        )
        metrics = final_metrics[branch]
        if (
            float(metrics["correlation"]) >= OFFLINE_GATE
            and float(metrics["sign_accuracy"]) >= OFFLINE_GATE
            and float(metrics["tolerance_accuracy"]) >= OFFLINE_GATE
        ):
            eligible.append(branch)
        orientation = _predicted_orientation(branch, predictions[branch], validation)
        predicted_keep = (
            orientation >= -validation["normalized_cross_tolerance"]
        ).astype(np.int64)
        mismatch = (
            (predictions[branch] >= 0.0)
            != (validation[f"{branch}_target"] >= 0.0)
        ) | (predicted_keep != validation["teacher_keep"])
        for index in np.flatnonzero(mismatch):
            state = validation_states[int(validation["state_position"][index])]
            error_rows.append(
                {
                    "record_type": "validation_error",
                    "branch": branch,
                    "validation_index": int(index),
                    "source_state_index": state["source_state_index"],
                    "trajectory_id": state["trajectory_id"],
                    "history_count": state["history_count"],
                    "teacher_target": float(validation[f"{branch}_target"][index]),
                    "predicted_target": float(predictions[branch][index]),
                    "absolute_error": float(
                        abs(
                            predictions[branch][index]
                            - validation[f"{branch}_target"][index]
                        )
                    ),
                    "teacher_keep": int(validation["teacher_keep"][index]),
                    "predicted_keep": int(predicted_keep[index]),
                    "near_zero": bool(
                        abs(validation[f"{branch}_target"][index])
                        <= NEAR_ZERO_THRESHOLD
                    ),
                    "near_collinear": bool(validation["near_collinear"][index]),
                }
            )

    rows.append(
        {
            "record_type": "offline_gate",
            "offline_gate": OFFLINE_GATE,
            "eligible_branches": ";".join(eligible),
            "eligible_branch_count": len(eligible),
            "offline_state_count_if_run": OFFLINE_STATES,
        }
    )
    if eligible:
        reconstruction_indices = turn._stratified_state_indices(
            payload["states"], "reconstruction", OFFLINE_STATES, DATA_SEED + 2
        )
        reconstruction_states = turn._build_states(
            payload, reconstruction_indices, workers, "reconstruction"
        )
        for branch in eligible:
            error_rows.extend(
                _offline_rows(
                    branch, models[branch], scalers[branch], reconstruction_states
                )
            )
    else:
        error_rows.append(
            {
                "record_type": "offline_summary",
                "offline_run": False,
                "skip_reason": "neither branch passed the predeclared continuous-score gate",
                "offline_gate": OFFLINE_GATE,
            }
        )
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(ERRORS_PATH, error_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
