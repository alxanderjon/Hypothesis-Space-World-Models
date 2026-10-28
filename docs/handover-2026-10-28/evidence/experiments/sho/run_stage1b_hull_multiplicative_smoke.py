#!/usr/bin/env python3
"""Balanced bilinear-vs-generic multiplicative hull-turn smoke test."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_balanced_hull_turn_smokescreen as balanced
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_multiplicative_smoke")
BILINEAR_PATH = OUTPUT_DIR / "hull_bilinear_smoke.csv"
MULTIPLICATIVE_PATH = OUTPUT_DIR / "hull_multiplicative_smoke.csv"

TRAIN_EVENTS = 5_000
VALIDATION_EVENTS = 1_000
MAX_EPOCHS = 500
PATIENCE = 20
PLATEAU_RELATIVE_IMPROVEMENT = 1.0e-4
SUCCESS_RMSE = 1.0e-12
SUCCESS_MAXIMUM_ABSOLUTE_ERROR = 1.0e-10
BATCH_SIZE = 256
LEARNING_RATE = 3.0e-3
LATENT_DIM = 8
HIDDEN_DIM = 16
MODEL_SEED = 1_618_033
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


def _dataset(
    selected: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(selected["label"])
    inputs = np.empty((count, 4), dtype=np.float64)
    target = np.empty(count, dtype=np.float64)
    tau_geom = np.empty(count, dtype=np.float64)
    decision_margin = np.empty(count, dtype=np.float64)
    raw_cross = np.empty(count, dtype=np.float64)
    denominator = np.empty(count, dtype=np.float64)
    equivalence_error = np.empty(count, dtype=np.float64)
    for row in range(count):
        state = states[int(selected["state_position"][row])]
        sizes = state["sizes"]
        losses = state["losses"]
        first, second, third = (
            int(value) for value in selected["triple_indices"][row]
        )
        delta_d_1 = float(sizes[second] - sizes[first])
        delta_f_1 = float(losses[second] - losses[first])
        delta_d_2 = float(sizes[third] - sizes[second])
        delta_f_2 = float(losses[third] - losses[second])
        norm_1 = math.hypot(delta_d_1, delta_f_1)
        norm_2 = math.hypot(delta_d_2, delta_f_2)
        if not norm_1 > 0.0:
            raise RuntimeError(
                f"zero ||v1|| at state {state['source_state_index']} event {row}"
            )
        if not norm_2 > 0.0:
            raise RuntimeError(
                f"zero ||v2|| at state {state['source_state_index']} event {row}"
            )
        product = norm_1 * norm_2
        if not product > 0.0 or not np.isfinite(product):
            raise RuntimeError("invalid ||v1||*||v2||")
        cross = delta_d_1 * delta_f_2 - delta_f_1 * delta_d_2
        if cross != float(selected["cross"][row]):
            raise RuntimeError("reconstructed cross differs from audited teacher")
        tolerance = float(selected["tolerance"][row])
        if not tolerance > 0.0 or not np.isfinite(tolerance):
            raise RuntimeError("invalid audited teacher tolerance")
        unit_1 = (delta_d_1 / norm_1, delta_f_1 / norm_1)
        unit_2 = (delta_d_2 / norm_2, delta_f_2 / norm_2)
        y_geom = cross / product
        unit_cross = unit_1[0] * unit_2[1] - unit_1[1] * unit_2[0]
        error = abs(unit_cross - y_geom)
        if error > 8.0 * np.finfo(np.float64).eps:
            raise RuntimeError("cross(v1,v2)/norms != cross(u1,u2)")
        geometric_tolerance = tolerance / product
        margin = cross / tolerance + 1.0
        inputs[row] = (*unit_1, *unit_2)
        target[row] = y_geom
        tau_geom[row] = geometric_tolerance
        decision_margin[row] = margin
        raw_cross[row] = cross
        denominator[row] = product
        equivalence_error[row] = error
    exact_keep = (target >= -tau_geom).astype(np.int64)
    if not np.array_equal(exact_keep, selected["label"]):
        raise RuntimeError("geometric target/tolerance changed teacher decisions")
    return {
        "inputs": inputs,
        "target": target,
        "tau_geom": tau_geom,
        "decision_margin": decision_margin,
        "teacher_keep": selected["label"].copy(),
        "state_position": selected["state_position"].copy(),
        "triple_indices": selected["triple_indices"].copy(),
        "raw_cross": raw_cross,
        "denominator": denominator,
        "equivalence_error": equivalence_error,
    }


class BilinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2, 2, dtype=torch.float64))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = values[:, :2]
        second = values[:, 2:]
        return torch.einsum("bi,ij,bj->b", first, self.weight, second)


class MultiplicativeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_projection = nn.Linear(2, LATENT_DIM)
        self.second_projection = nn.Linear(2, LATENT_DIM)
        self.head = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = self.first_projection(values[:, :2])
        second = self.second_projection(values[:, 2:])
        return self.head(first * second).squeeze(-1)


def _predict(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(inputs, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            parts.append(model(values[start : start + 2048]))
    return torch.cat(parts).cpu().numpy()


def _continuous_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    if np.std(target) > 0.0 and np.std(prediction) > 0.0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = 0.0
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "correlation": correlation,
        "absolute_error_p50": float(np.quantile(absolute, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
        "absolute_error_p99": float(np.quantile(absolute, 0.99)),
        "absolute_error_p100": float(np.max(absolute)),
    }


def _decision_metrics(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    teacher_keep = dataset["teacher_keep"].astype(bool)
    predicted_keep = prediction >= -dataset["tau_geom"]
    true_keep_pred_keep = int(np.sum(teacher_keep & predicted_keep))
    true_keep_pred_pop = int(np.sum(teacher_keep & ~predicted_keep))
    true_pop_pred_pop = int(np.sum(~teacher_keep & ~predicted_keep))
    true_pop_pred_keep = int(np.sum(~teacher_keep & predicted_keep))
    keep_count = int(np.sum(teacher_keep))
    pop_count = int(np.sum(~teacher_keep))
    return {
        "decision_accuracy": float(np.mean(teacher_keep == predicted_keep)),
        "keep_agreement": true_keep_pred_keep / keep_count if keep_count else "",
        "pop_agreement": true_pop_pred_pop / pop_count if pop_count else "",
        "mistake_count": int(np.sum(teacher_keep != predicted_keep)),
        "true_keep_pred_keep": true_keep_pred_keep,
        "true_keep_pred_pop": true_keep_pred_pop,
        "true_pop_pred_pop": true_pop_pred_pop,
        "true_pop_pred_keep": true_pop_pred_keep,
        "keep_count": keep_count,
        "pop_count": pop_count,
    }


def _train(
    branch: str,
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
) -> tuple[nn.Module, list[dict[str, Any]], int, int, str, bool]:
    torch.manual_seed(MODEL_SEED)
    model: nn.Module
    if branch == "bilinear":
        model = BilinearModel()
    elif branch == "multiplicative":
        model = MultiplicativeModel()
    else:
        raise ValueError(branch)
    model = model.to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_inputs = torch.as_tensor(training["inputs"], dtype=torch.float64)
    train_target = torch.as_tensor(training["target"], dtype=torch.float64)
    generator = np.random.default_rng(MODEL_SEED + 97)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mse = math.inf
    plateau_validation_mse = math.inf
    best_epoch = 0
    stale = 0
    stopping_reason = "maximum_epochs"
    success = False
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(train_target))
        total_mse = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(
                order[start : start + BATCH_SIZE], dtype=torch.long
            )
            prediction = model(train_inputs[indices])
            loss = nn.functional.mse_loss(prediction, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.item()) * len(indices)
        validation_prediction = _predict(model, validation["inputs"])
        validation_mse = float(
            np.mean((validation_prediction - validation["target"]) ** 2)
        )
        row = {
            "record_type": "epoch",
            "branch": branch,
            "epoch": epoch,
            "training_mse": total_mse / len(order),
            "validation_mse": validation_mse,
            **_continuous_metrics(validation["target"], validation_prediction),
            **_decision_metrics(validation, validation_prediction),
        }
        rows.append(row)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if validation_mse < plateau_validation_mse * (
            1.0 - PLATEAU_RELATIVE_IMPROVEMENT
        ):
            plateau_validation_mse = validation_mse
            stale = 0
        else:
            stale += 1
        success = bool(
            row["rmse"] <= SUCCESS_RMSE
            and row["maximum_absolute_error"]
            <= SUCCESS_MAXIMUM_ABSOLUTE_ERROR
            and row["decision_accuracy"] == 1.0
        )
        if epoch == 1 or epoch % 10 == 0 or epoch == MAX_EPOCHS:
            print(
                f"branch={branch} epoch={epoch} rmse={row['rmse']:.6e} "
                f"decision={row['decision_accuracy']:.6f}",
                flush=True,
            )
        if success:
            stopping_reason = "numerical_success"
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, rows, best_epoch, len(rows), stopping_reason, success


def _subset_row(
    name: str,
    mask: np.ndarray,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    row: dict[str, Any] = {
        "record_type": "subset",
        "subset": name,
        "count": len(indices),
        "fraction": len(indices) / len(prediction),
    }
    if len(indices):
        subset = {
            key: value[indices]
            for key, value in dataset.items()
            if isinstance(value, np.ndarray) and len(value) == len(prediction)
        }
        row.update(_continuous_metrics(subset["target"], prediction[indices]))
        row.update(_decision_metrics(subset, prediction[indices]))
        row["minimum_abs_teacher_margin"] = float(
            np.min(np.abs(subset["decision_margin"]))
        )
        row["maximum_abs_teacher_margin"] = float(
            np.max(np.abs(subset["decision_margin"]))
        )
    return row


def _diagnostic_rows(
    branch: str,
    model: nn.Module,
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    prediction: np.ndarray,
    history: list[dict[str, Any]],
    best_epoch: int,
    epochs_trained: int,
    stopping_reason: str,
    success: bool,
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
) -> list[dict[str, Any]]:
    metrics = {
        **_continuous_metrics(validation["target"], prediction),
        **_decision_metrics(validation, prediction),
    }
    rows = history + [
        {
            "record_type": "final_summary",
            "branch": branch,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "maximum_epochs": MAX_EPOCHS,
            "stopping_reason": stopping_reason,
            "success": success,
            "plateau_patience": PATIENCE,
            "plateau_relative_improvement": PLATEAU_RELATIVE_IMPROVEMENT,
            "success_rmse_threshold": SUCCESS_RMSE,
            "success_maximum_absolute_error_threshold": SUCCESS_MAXIMUM_ABSOLUTE_ERROR,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": epochs_trained
            * math.ceil(TRAIN_EVENTS / BATCH_SIZE),
            "train_events": len(training["target"]),
            "validation_events": len(validation["target"]),
            "target_minimum": float(np.min(validation["target"])),
            "target_maximum": float(np.max(validation["target"])),
            "target_mean": float(np.mean(validation["target"])),
            "target_standard_deviation": float(np.std(validation["target"])),
            **metrics,
            "classification_target_used": False,
            "additive_epsilon_used": False,
            "hull_reconstruction_run": False,
            "trajectory_search_run": False,
        },
        {
            "record_type": "dataset_balance",
            "branch": branch,
            "split": "train",
            **train_counts,
        },
        {
            "record_type": "dataset_balance",
            "branch": branch,
            "split": "validation",
            **validation_counts,
        },
        {
            "record_type": "assertions",
            "branch": branch,
            "positive_norm_assertion": True,
            "maximum_cross_equivalence_error": float(
                max(
                    np.max(training["equivalence_error"]),
                    np.max(validation["equivalence_error"]),
                )
            ),
            "deterministic_teacher_accuracy": float(
                np.mean(
                    (validation["target"] >= -validation["tau_geom"])
                    == validation["teacher_keep"]
                )
            ),
        },
    ]
    if branch == "bilinear":
        weight = model.weight.detach().cpu().numpy()  # type: ignore[attr-defined]
        reference = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        rows.append(
            {
                "record_type": "learned_matrix",
                "branch": branch,
                "W_00": float(weight[0, 0]),
                "W_01": float(weight[0, 1]),
                "W_10": float(weight[1, 0]),
                "W_11": float(weight[1, 1]),
                "posthoc_reference_frobenius_error": float(
                    np.linalg.norm(weight - reference)
                ),
                "posthoc_reference_maximum_coefficient_error": float(
                    np.max(np.abs(weight - reference))
                ),
            }
        )
    margin = validation["decision_margin"]
    labels = validation["teacher_keep"].astype(bool)
    subset_definitions = (
        ("all_keep", labels),
        ("all_pop", ~labels),
        ("clearly_keep_abs_margin_ge_10", labels & (np.abs(margin) >= 10.0)),
        ("clearly_pop_abs_margin_ge_10", ~labels & (np.abs(margin) >= 10.0)),
        ("near_boundary_abs_margin_lt_0.1", np.abs(margin) < 0.1),
        ("near_boundary_abs_margin_lt_0.5", np.abs(margin) < 0.5),
        ("near_boundary_abs_margin_lt_1", np.abs(margin) < 1.0),
        (
            "boundary_margin_1_to_10",
            (np.abs(margin) >= 1.0) & (np.abs(margin) < 10.0),
        ),
        ("far_from_boundary_abs_margin_ge_10", np.abs(margin) >= 10.0),
    )
    rows.extend(
        _subset_row(name, mask, validation, prediction)
        for name, mask in subset_definitions
    )
    predicted_keep = prediction >= -validation["tau_geom"]
    mistake_indices = np.flatnonzero(predicted_keep != labels)
    for index in mistake_indices:
        state_position = int(validation["state_position"][index])
        triple = validation["triple_indices"][index]
        rows.append(
            {
                "record_type": "mistake",
                "branch": branch,
                "validation_index": int(index),
                "state_position": state_position,
                "triple_first": int(triple[0]),
                "triple_second": int(triple[1]),
                "triple_third": int(triple[2]),
                "teacher_decision": "KEEP" if labels[index] else "POP",
                "predicted_decision": "KEEP" if predicted_keep[index] else "POP",
                "y_geom": float(validation["target"][index]),
                "predicted_y_geom": float(prediction[index]),
                "tau_geom": float(validation["tau_geom"][index]),
                "teacher_boundary": float(-validation["tau_geom"][index]),
                "signed_tolerance_relative_margin": float(margin[index]),
                "absolute_tolerance_relative_margin": float(abs(margin[index])),
                "signed_geometric_margin": float(
                    validation["target"][index] + validation["tau_geom"][index]
                ),
                "absolute_geometric_margin": float(
                    abs(
                        validation["target"][index]
                        + validation["tau_geom"][index]
                    )
                ),
                "continuous_absolute_error": float(
                    abs(prediction[index] - validation["target"][index])
                ),
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
    train_selected, train_counts = balanced._balanced_sample(
        train_pool, TRAIN_EVENTS, DATA_SEED + 11
    )
    validation_selected, validation_counts = balanced._balanced_sample(
        validation_pool, VALIDATION_EVENTS, DATA_SEED + 12
    )
    training = _dataset(train_selected, train_states)
    validation = _dataset(validation_selected, validation_states)
    for branch, path in (
        ("bilinear", BILINEAR_PATH),
        ("multiplicative", MULTIPLICATIVE_PATH),
    ):
        (
            model,
            history,
            best_epoch,
            epochs_trained,
            stopping_reason,
            success,
        ) = _train(branch, training, validation)
        prediction = _predict(model, validation["inputs"])
        rows = _diagnostic_rows(
            branch,
            model,
            training,
            validation,
            prediction,
            history,
            best_epoch,
            epochs_trained,
            stopping_reason,
            success,
            train_counts,
            validation_counts,
        )
        _write_csv(path, rows)
        summary = next(row for row in rows if row["record_type"] == "final_summary")
        print(
            f"{branch}: rmse={summary['rmse']:.6e} "
            f"decision={summary['decision_accuracy']:.6f} "
            f"mistakes={summary['mistake_count']}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
