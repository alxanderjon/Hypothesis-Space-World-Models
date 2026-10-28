#!/usr/bin/env python3
"""No-epsilon geometric and tolerance-relative hull score smoke test."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_noepsilon_tolerance_smokescreen")
DISTRIBUTION_PATH = OUTPUT_DIR / "hull_noepsilon_distribution.csv"
COMPARISON_PATH = OUTPUT_DIR / "hull_tolerance_score_comparison.csv"

MAX_EPOCHS = 75
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 32
MODEL_SEED = 2_236_067
DATA_SEED = turn.DATA_SEED
EXTREME_GATE = 0.999
BOUNDARY_BANDS = (0.1, 0.5, 1.0)


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


def _dataset(
    selected: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(selected["label"])
    geom_inputs = np.empty((count, 5), dtype=np.float64)
    tolerance_inputs = np.empty((count, 5), dtype=np.float64)
    raw_cross = np.empty(count, dtype=np.float64)
    denominator = np.empty(count, dtype=np.float64)
    geom_target = np.empty(count, dtype=np.float64)
    raw_tolerance = np.empty(count, dtype=np.float64)
    geom_tolerance = np.empty(count, dtype=np.float64)
    tolerance_target = np.empty(count, dtype=np.float64)
    for row in range(count):
        state = states[int(selected["state_position"][row])]
        sizes = state["sizes"]
        losses = state["losses"]
        first, second, third = (
            int(value) for value in selected["triple_indices"][row]
        )
        delta_d_ab = float(sizes[second] - sizes[first])
        delta_f_ab = float(losses[second] - losses[first])
        delta_d_bc = float(sizes[third] - sizes[second])
        delta_f_bc = float(losses[third] - losses[second])
        norm_ab = math.hypot(delta_d_ab, delta_f_ab)
        norm_bc = math.hypot(delta_d_bc, delta_f_bc)
        if not norm_ab > 0.0:
            raise RuntimeError(
                f"zero ||v1|| at source state {state['source_state_index']} triple {row}"
            )
        if not norm_bc > 0.0:
            raise RuntimeError(
                f"zero ||v2|| at source state {state['source_state_index']} triple {row}"
            )
        cross = delta_d_ab * delta_f_bc - delta_f_ab * delta_d_bc
        exact_cross = float(selected["cross"][row])
        if cross != exact_cross:
            raise RuntimeError("local raw cross differs from audited stack cross")
        product = norm_ab * norm_bc
        if not product > 0.0 or not np.isfinite(product):
            raise RuntimeError(
                f"invalid D={product} at source state {state['source_state_index']} triple {row}"
            )
        tolerance = float(selected["tolerance"][row])
        tau_geom = tolerance / product
        y_geom = cross / product
        y_tolerance = cross / tolerance
        unit_ab = (delta_d_ab / norm_ab, delta_f_ab / norm_ab)
        unit_bc = (delta_d_bc / norm_bc, delta_f_bc / norm_bc)
        geom_inputs[row] = (*unit_ab, *unit_bc, 0.0)
        tolerance_inputs[row] = (*unit_ab, *unit_bc, tau_geom)
        raw_cross[row] = cross
        denominator[row] = product
        geom_target[row] = y_geom
        raw_tolerance[row] = tolerance
        geom_tolerance[row] = tau_geom
        tolerance_target[row] = y_tolerance
    geom_labels = (geom_target >= -geom_tolerance).astype(np.int64)
    tolerance_labels = (tolerance_target >= -1.0).astype(np.int64)
    if not np.array_equal(geom_labels, selected["label"]):
        raise RuntimeError("geometric target changed the exact teacher decision")
    if not np.array_equal(tolerance_labels, selected["label"]):
        raise RuntimeError("tolerance-relative target changed the exact teacher decision")
    linked = geom_target / geom_tolerance
    if not np.array_equal(linked, tolerance_target):
        relative = np.abs(linked - tolerance_target) / np.maximum(
            np.abs(tolerance_target), 1.0
        )
        if float(np.max(relative)) > 5.0e-16:
            raise RuntimeError("y_tol != y_geom/tau_geom within float64 rounding")
    return {
        "geom_inputs": geom_inputs,
        "tolerance_inputs": tolerance_inputs,
        "raw_cross": raw_cross,
        "denominator": denominator,
        "geom_target": geom_target,
        "raw_tolerance": raw_tolerance,
        "geom_tolerance": geom_tolerance,
        "tolerance_target": tolerance_target,
        "teacher_keep": selected["label"].copy(),
        "near_collinear": selected["near_collinear"].copy(),
        "exact_collinear": selected["exact_collinear"].copy(),
        "state_position": selected["state_position"].copy(),
        "triple_indices": selected["triple_indices"].copy(),
        "norm_failure_count": np.asarray([0], dtype=np.int64),
    }


def _distribution_row(
    split: str, quantity: str, values: np.ndarray
) -> dict[str, Any]:
    absolute = np.abs(values)
    return {
        "record_type": "distribution",
        "split": split,
        "quantity": quantity,
        "count": len(values),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "median_absolute": float(np.median(absolute)),
        "abs_p01": float(np.quantile(absolute, 0.01)),
        "abs_p10": float(np.quantile(absolute, 0.10)),
        "abs_p25": float(np.quantile(absolute, 0.25)),
        "abs_p50": float(np.quantile(absolute, 0.50)),
        "abs_p75": float(np.quantile(absolute, 0.75)),
        "abs_p90": float(np.quantile(absolute, 0.90)),
        "abs_p99": float(np.quantile(absolute, 0.99)),
        "abs_p100": float(np.max(absolute)),
    }


def _distribution_rows(
    split: str, dataset: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = [
        _distribution_row(split, "C_raw", dataset["raw_cross"]),
        _distribution_row(split, "D", dataset["denominator"]),
        _distribution_row(split, "y_geom", dataset["geom_target"]),
        _distribution_row(split, "tau_raw", dataset["raw_tolerance"]),
        _distribution_row(split, "tau_geom", dataset["geom_tolerance"]),
        _distribution_row(split, "y_tol", dataset["tolerance_target"]),
    ]
    y_geom = np.abs(dataset["geom_target"])
    for threshold in (1.0e-3, 1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40):
        count = int(np.sum(y_geom < threshold))
        rows.append(
            {
                "record_type": "small_y_geom_count",
                "split": split,
                "quantity": "y_geom",
                "threshold": threshold,
                "count": count,
                "fraction": count / len(y_geom),
            }
        )
    rows.append(
        {
            "record_type": "norm_assertion",
            "split": split,
            "quantity": "segment_norms",
            "count": len(y_geom),
            "zero_or_invalid_norm_count": int(dataset["norm_failure_count"][0]),
            "assertion_passed": int(dataset["norm_failure_count"][0]) == 0,
            "additive_epsilon_used": False,
        }
    )
    return rows


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
        input_scale = np.where(input_scale > 0.0, input_scale, 1.0)
        target_mean = float(np.mean(target))
        target_scale = float(np.std(target))
        if not target_scale > 0.0 or not np.isfinite(target_scale):
            target_scale = 1.0
        return cls(input_mean, input_scale, target_mean, target_scale)

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_mean) / self.input_scale

    def targets(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_scale

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
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


def _predict(
    model: ScoreMLP,
    standardization: Standardization,
    inputs: np.ndarray,
) -> np.ndarray:
    values = torch.as_tensor(standardization.inputs(inputs), dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            parts.append(model(values[start : start + 2048]))
    return standardization.inverse_targets(torch.cat(parts).cpu().numpy())


def _correlation(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(target) == 0.0 or np.std(prediction) == 0.0:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _metrics(
    branch: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    indices: np.ndarray | None = None,
) -> dict[str, Any]:
    if indices is None:
        indices = np.arange(len(prediction), dtype=np.int64)
    target = dataset[f"{branch}_target"][indices]
    predicted = prediction[indices]
    teacher_keep = dataset["teacher_keep"][indices]
    if branch == "geom":
        predicted_keep = (
            predicted >= -dataset["geom_tolerance"][indices]
        ).astype(np.int64)
    else:
        predicted_keep = (predicted >= -1.0).astype(np.int64)
    error = predicted - target
    return {
        "count": len(indices),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
        "normalized_rmse": float(
            np.sqrt(np.mean(error**2)) / max(float(np.std(target)), 1.0e-300)
        ),
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
    dict[str, Standardization],
    list[dict[str, Any]],
    dict[str, int],
    int,
]:
    specifications = {
        "geom": ("geom_inputs", "geom_target", 5),
        "tolerance": ("tolerance_inputs", "tolerance_target", 5),
    }
    models: dict[str, ScoreMLP] = {}
    scalers: dict[str, Standardization] = {}
    optimizers: dict[str, torch.optim.Optimizer] = {}
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    best_states: dict[str, dict[str, torch.Tensor]] = {}
    best_losses = {branch: math.inf for branch in specifications}
    best_epochs = {branch: 0 for branch in specifications}
    stale = {branch: 0 for branch in specifications}
    for branch, (input_key, target_key, input_dim) in specifications.items():
        torch.manual_seed(MODEL_SEED)
        model = ScoreMLP(input_dim).to(torch.float64)
        scaler = Standardization.fit(training[input_key], training[target_key])
        models[branch] = model
        scalers[branch] = scaler
        optimizers[branch] = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        tensors[branch] = (
            torch.as_tensor(scaler.inputs(training[input_key]), dtype=torch.float64),
            torch.as_tensor(scaler.targets(training[target_key]), dtype=torch.float64),
        )
        best_states[branch] = copy.deepcopy(model.state_dict())

    generator = np.random.default_rng(MODEL_SEED)
    rows: list[dict[str, Any]] = []
    shared_updates = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(training["teacher_keep"]))
        train_losses = {branch: 0.0 for branch in specifications}
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
                train_losses[branch] += float(loss.item()) * len(batch)
            shared_updates += 1
        for branch, (input_key, target_key, _) in specifications.items():
            prediction = _predict(models[branch], scalers[branch], validation[input_key])
            standardized_error = (
                prediction - validation[target_key]
            ) / scalers[branch].target_scale
            validation_mse = float(np.mean(standardized_error**2))
            metrics = _metrics(branch, validation, prediction)
            rows.append(
                {
                    "record_type": "epoch",
                    "branch": branch,
                    "epoch": epoch,
                    "shared_update_count": shared_updates,
                    "training_standardized_mse": train_losses[branch] / len(order),
                    "validation_standardized_mse": validation_mse,
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


def _comparison_rows(
    branch: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    best_epoch: int,
    epochs_trained: int,
    updates: int,
) -> list[dict[str, Any]]:
    target = dataset[f"{branch}_target"]
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
            "exact_tolerance_rule": (
                "keep iff predicted y_geom >= -tau_geom"
                if branch == "geom"
                else "keep iff predicted y_tol >= -1"
            ),
            "training_target_was_continuous_only": True,
        }
    ]
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
    y_tolerance = dataset["tolerance_target"]
    for band in BOUNDARY_BANDS:
        indices = np.flatnonzero(np.abs(y_tolerance + 1.0) < band)
        row: dict[str, Any] = {
            "record_type": "teacher_boundary_band",
            "branch": branch,
            "subset": f"abs_y_tol_plus_1_lt_{band:g}",
            "boundary_half_width": band,
            "count": len(indices),
        }
        if len(indices):
            row.update(_metrics(branch, dataset, prediction, indices))
        rows.append(row)
    if branch == "tolerance":
        bins = (
            ("abs_y_tol_lt_1", 0.0, 1.0),
            ("abs_y_tol_1_to_2", 1.0, 2.0),
            ("abs_y_tol_2_to_10", 2.0, 10.0),
            ("abs_y_tol_ge_10", 10.0, math.inf),
        )
        absolute = np.abs(y_tolerance)
        for name, lower, upper in bins:
            indices = np.flatnonzero((absolute >= lower) & (absolute < upper))
            row = {
                "record_type": "tolerance_magnitude_bin",
                "branch": branch,
                "subset": name,
                "count": len(indices),
            }
            if len(indices):
                row.update(_metrics(branch, dataset, prediction, indices))
            rows.append(row)
    exact_prediction = target
    exact_metrics = _metrics(branch, dataset, exact_prediction)
    rows.append(
        {
            "record_type": "deterministic_sanity",
            "branch": branch,
            "correct": int(
                round(float(exact_metrics["tolerance_accuracy"]) * len(target))
            ),
            "total": len(target),
            "accuracy": exact_metrics["tolerance_accuracy"],
        }
    )
    return rows


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
    training = _dataset(train_selected, train_states)
    validation = _dataset(validation_selected, validation_states)
    distribution_rows = _distribution_rows("train", training)
    distribution_rows.extend(_distribution_rows("validation", validation))
    _write_csv(DISTRIBUTION_PATH, distribution_rows)
    print(
        "no-epsilon validation: "
        f"median|C_raw|={np.median(np.abs(validation['raw_cross'])):.3e} "
        f"median D={np.median(validation['denominator']):.3e} "
        f"median|y_geom|={np.median(np.abs(validation['geom_target'])):.3e}",
        flush=True,
    )

    models, scalers, rows, best_epochs, updates = _paired_train(
        training, validation
    )
    epochs_trained = max(int(row["epoch"]) for row in rows)
    eligible: list[str] = []
    for branch in ("geom", "tolerance"):
        prediction = _predict(
            models[branch], scalers[branch], validation[f"{branch}_inputs"]
        )
        rows.extend(
            _comparison_rows(
                branch,
                validation,
                prediction,
                best_epochs[branch],
                epochs_trained,
                updates,
            )
        )
        metrics = _metrics(branch, validation, prediction)
        boundary_indices = np.flatnonzero(
            np.abs(validation["tolerance_target"] + 1.0) < 0.5
        )
        boundary_accuracy = (
            float(_metrics(branch, validation, prediction, boundary_indices)["tolerance_accuracy"])
            if len(boundary_indices)
            else 0.0
        )
        if (
            float(metrics["correlation"]) >= EXTREME_GATE
            and float(metrics["sign_accuracy"]) >= EXTREME_GATE
            and float(metrics["tolerance_accuracy"]) >= EXTREME_GATE
            and boundary_accuracy >= EXTREME_GATE
        ):
            eligible.append(branch)
    rows.append(
        {
            "record_type": "stack_eligibility",
            "extreme_accuracy_gate": EXTREME_GATE,
            "eligible_branches": ";".join(eligible),
            "eligible_branch_count": len(eligible),
            "full_hull_reconstruction_run": False,
            "trajectory_search_run": False,
            "reason": (
                "eligible branch requires a separately reported stack test"
                if eligible
                else "neither continuous predictor was extremely accurate near the teacher boundary"
            ),
        }
    )
    _write_csv(COMPARISON_PATH, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
