#!/usr/bin/env python3
"""Train and diagnose a skew-symmetric bilinear eDIRECT hull-turn score."""

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
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.validate_stage1b_bilinear_hull_turn as validation


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/skew_bilinear_hull_turn")
TRAINING_PATH = OUTPUT_DIR / "skew_bilinear_training.csv"
FULL_POOL_PATH = OUTPUT_DIR / "skew_bilinear_full_pool.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "skew_bilinear_hull_reconstruction.csv"

MAX_EPOCHS = 2_000
PATIENCE = 50
PLATEAU_RELATIVE_IMPROVEMENT = 1.0e-5
NUMERICAL_RMSE = 1.0e-15
NUMERICAL_MAX_ERROR = 1.0e-14
MODEL_SEED = smoke.MODEL_SEED + 271_828


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


class SkewBilinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw_matrix = nn.Parameter(torch.empty(2, 2, dtype=torch.float64))
        nn.init.xavier_uniform_(self.raw_matrix)

    def effective_weight(self) -> torch.Tensor:
        return self.raw_matrix - self.raw_matrix.transpose(0, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        first = values[:, :2]
        second = values[:, 2:]
        return torch.einsum("bi,ij,bj->b", first, self.effective_weight(), second)


def _predict(model: SkewBilinearModel, inputs: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(inputs, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(model(values[start : start + 4096]))
    return torch.cat(parts).cpu().numpy()


def _predict_fixed_weight(weight: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(inputs, dtype=torch.float64)
    matrix = torch.as_tensor(weight, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            batch = values[start : start + 4096]
            parts.append(
                torch.einsum("bi,ij,bj->b", batch[:, :2], matrix, batch[:, 2:])
            )
    return torch.cat(parts).cpu().numpy()


def _train(
    training: dict[str, np.ndarray], validation_set: dict[str, np.ndarray]
) -> tuple[SkewBilinearModel, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED + 97)
    model = SkewBilinearModel().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=smoke.LEARNING_RATE)
    train_inputs = torch.as_tensor(training["inputs"], dtype=torch.float64)
    train_target = torch.as_tensor(training["target"], dtype=torch.float64)
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mse = math.inf
    plateau_validation_mse = math.inf
    best_epoch = 0
    stale = 0
    numerical_epoch = 0
    stopping_reason = "maximum_epochs"
    rows: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(train_target))
        total_mse = 0.0
        model.train()
        for start in range(0, len(order), smoke.BATCH_SIZE):
            indices = torch.as_tensor(
                order[start : start + smoke.BATCH_SIZE], dtype=torch.long
            )
            prediction = model(train_inputs[indices])
            loss = nn.functional.mse_loss(prediction, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.item()) * len(indices)

        validation_prediction = _predict(model, validation_set["inputs"])
        validation_mse = float(
            np.mean((validation_prediction - validation_set["target"]) ** 2)
        )
        metrics = {
            **smoke._continuous_metrics(
                validation_set["target"], validation_prediction
            ),
            **smoke._decision_metrics(validation_set, validation_prediction),
        }
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
        numerical_now = bool(
            metrics["rmse"] <= NUMERICAL_RMSE
            and metrics["maximum_absolute_error"] <= NUMERICAL_MAX_ERROR
        )
        if numerical_now and numerical_epoch == 0:
            numerical_epoch = epoch
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_mse": total_mse / len(order),
                "validation_mse": validation_mse,
                "plateau_stale_epochs": stale,
                "numerical_precision_reached": numerical_now,
                **metrics,
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"epoch={epoch} rmse={metrics['rmse']:.6e} "
                f"max={metrics['maximum_absolute_error']:.6e} "
                f"decision={metrics['decision_accuracy']:.6f} stale={stale}",
                flush=True,
            )
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break

    stop_epoch = len(rows)
    model.load_state_dict(best_state)
    model.eval()
    best_prediction = _predict(model, validation_set["inputs"])
    best_metrics = {
        **smoke._continuous_metrics(validation_set["target"], best_prediction),
        **smoke._decision_metrics(validation_set, best_prediction),
    }
    raw_matrix = model.raw_matrix.detach().cpu().numpy()
    weight = model.effective_weight().detach().cpu().numpy()
    reference = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "maximum_epochs": MAX_EPOCHS,
        "stopping_reason": stopping_reason,
        "plateau_patience": PATIENCE,
        "plateau_relative_improvement": PLATEAU_RELATIVE_IMPROVEMENT,
        "first_numerical_precision_epoch": numerical_epoch,
        "optimizer": "Adam",
        "learning_rate": smoke.LEARNING_RATE,
        "optimizer_updates": stop_epoch
        * math.ceil(smoke.TRAIN_EVENTS / smoke.BATCH_SIZE),
        "training_events": len(training["target"]),
        "validation_events": len(validation_set["target"]),
        **best_metrics,
        "A_00": float(raw_matrix[0, 0]),
        "A_01": float(raw_matrix[0, 1]),
        "A_10": float(raw_matrix[1, 0]),
        "A_11": float(raw_matrix[1, 1]),
        "W_00": float(weight[0, 0]),
        "W_01": float(weight[0, 1]),
        "W_10": float(weight[1, 0]),
        "W_11": float(weight[1, 1]),
        "effective_off_diagonal_coefficient": float(weight[0, 1]),
        "posthoc_reference_frobenius_error": float(
            np.linalg.norm(weight - reference)
        ),
        "posthoc_reference_maximum_coefficient_error": float(
            np.max(np.abs(weight - reference))
        ),
        "skew_symmetry_residual": float(np.max(np.abs(weight + weight.T))),
        "classification_target_used": False,
        "comparator_trained": False,
        "hull_reconstruction_run": False,
        "trajectory_search_run": False,
    }
    rows.append(summary)
    return model, rows, summary


def _full_pool_rows(
    dataset: dict[str, np.ndarray],
    learned_prediction: np.ndarray,
    ideal_prediction: np.ndarray,
    unrestricted_baseline_mistakes: int,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    all_mask = np.ones(len(learned_prediction), dtype=np.bool_)
    summary = validation._metric_row(
        "final_summary", "all", all_mask, dataset, learned_prediction
    )
    teacher = dataset["label"].astype(bool)
    learned_decision = learned_prediction >= -dataset["tau_geom"]
    ideal_decision = ideal_prediction >= -dataset["tau_geom"]
    unit_cross = (
        dataset["inputs"][:, 0] * dataset["inputs"][:, 3]
        - dataset["inputs"][:, 1] * dataset["inputs"][:, 2]
    )
    unit_decision = unit_cross >= -dataset["tau_geom"]
    summary.update(
        {
            "unrestricted_epoch341_mistake_count": unrestricted_baseline_mistakes,
            "mistake_reduction_vs_unrestricted": unrestricted_baseline_mistakes
            - int(summary["mistake_count"]),
            "ideal_float64_bilinear_mistake_count": int(
                np.sum(ideal_decision != teacher)
            ),
            "direct_unit_cross_mistake_count": int(
                np.sum(unit_decision != teacher)
            ),
            "full_pool_gate_passed": summary["decision_accuracy"] == 1.0,
        }
    )
    rows: list[dict[str, Any]] = [summary]
    absolute_target = np.abs(dataset["target"])
    for threshold in (1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40):
        rows.append(
            validation._metric_row(
                "near_collinear_subset",
                f"abs_y_geom_lt_{threshold:.0e}",
                absolute_target < threshold,
                dataset,
                learned_prediction,
            )
        )
    mistake_indices = np.flatnonzero(learned_decision != teacher)
    evaluation_order_count = 0
    coefficient_count = 0
    for index in mistake_indices:
        evaluation_order = bool(ideal_decision[index] != teacher[index])
        if evaluation_order:
            evaluation_order_count += 1
            attribution = "floating_point_evaluation_order"
        else:
            coefficient_count += 1
            attribution = "learned_coefficient_error"
        geometric_margin = dataset["target"][index] + dataset["tau_geom"][index]
        rows.append(
            {
                "record_type": "mistake",
                "event_index": int(index),
                "teacher_decision": "KEEP" if teacher[index] else "POP",
                "learned_decision": "KEEP" if learned_decision[index] else "POP",
                "exact_y_geom": float(dataset["target"][index]),
                "predicted_y_geom": float(learned_prediction[index]),
                "ideal_float64_bilinear_y_geom": float(ideal_prediction[index]),
                "direct_unit_cross_y_geom": float(unit_cross[index]),
                "raw_teacher_tolerance": float(dataset["tau_raw"][index]),
                "normalized_teacher_tolerance": float(
                    dataset["tau_geom"][index]
                ),
                "teacher_boundary": float(-dataset["tau_geom"][index]),
                "signed_distance_from_teacher_boundary": float(geometric_margin),
                "absolute_distance_from_teacher_boundary": float(
                    abs(geometric_margin)
                ),
                "u1_0": float(dataset["inputs"][index, 0]),
                "u1_1": float(dataset["inputs"][index, 1]),
                "u2_0": float(dataset["inputs"][index, 2]),
                "u2_1": float(dataset["inputs"][index, 3]),
                "attribution": attribution,
                "floating_point_evaluation_order_attributed": evaluation_order,
                "ideal_bilinear_matches_teacher_decision": bool(
                    ideal_decision[index] == teacher[index]
                ),
            }
        )
    summary["evaluation_order_attributed_mistakes"] = evaluation_order_count
    summary["coefficient_error_attributed_mistakes"] = coefficient_count
    passed = bool(summary["decision_accuracy"] == 1.0)
    return rows, passed, summary


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECONSTRUCTION_PATH.exists():
        RECONSTRUCTION_PATH.unlink()
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    train_indices = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    validation_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    train_states = turn._build_states(payload, train_indices, workers, "train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "validation"
    )
    train_pool = turn._flatten_triples(train_states)
    validation_pool = turn._flatten_triples(validation_states)
    train_selected, train_counts = balanced._balanced_sample(
        train_pool, smoke.TRAIN_EVENTS, smoke.DATA_SEED + 11
    )
    validation_selected, validation_counts = balanced._balanced_sample(
        validation_pool, smoke.VALIDATION_EVENTS, smoke.DATA_SEED + 12
    )
    training = smoke._dataset(train_selected, train_states)
    validation_set = smoke._dataset(validation_selected, validation_states)
    model, training_rows, training_summary = _train(training, validation_set)
    training_rows.extend(
        [
            {"record_type": "dataset_balance", "split": "train", **train_counts},
            {
                "record_type": "dataset_balance",
                "split": "validation",
                **validation_counts,
            },
        ]
    )
    _write_csv(TRAINING_PATH, training_rows)

    full_dataset = validation._event_dataset(validation_pool, validation_states)
    learned_prediction = _predict(model, full_dataset["inputs"])
    ideal_weight = np.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
    ideal_prediction = _predict_fixed_weight(ideal_weight, full_dataset["inputs"])
    full_rows, full_passed, full_summary = _full_pool_rows(
        full_dataset, learned_prediction, ideal_prediction, 222
    )
    full_summary.update(
        {
            "training_best_epoch": training_summary["best_epoch"],
            "training_stop_epoch": training_summary["stop_epoch"],
            "effective_off_diagonal_coefficient": training_summary[
                "effective_off_diagonal_coefficient"
            ],
            "posthoc_reference_frobenius_error": training_summary[
                "posthoc_reference_frobenius_error"
            ],
        }
    )
    _write_csv(FULL_POOL_PATH, full_rows)

    reconstruction_run = False
    reconstruction_passed = False
    if full_passed:
        reconstruction_indices = turn._stratified_state_indices(
            payload["states"],
            "reconstruction",
            turn.RECONSTRUCTION_STATES,
            smoke.DATA_SEED + 2,
        )
        reconstruction_states = turn._build_states(
            payload, reconstruction_indices, workers, "hull reconstruction"
        )
        weight = model.effective_weight().detach().cpu().numpy()
        reconstruction_rows, reconstruction_passed, _ = (
            validation._reconstruction_rows(weight, reconstruction_states)
        )
        _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)
        reconstruction_run = True
    print(
        f"best_epoch={training_summary['best_epoch']} "
        f"stop_epoch={training_summary['stop_epoch']} "
        f"rmse={training_summary['rmse']:.6e} "
        f"k={training_summary['effective_off_diagonal_coefficient']:.17g}",
        flush=True,
    )
    print(
        f"full_pool_mistakes={full_summary['mistake_count']} "
        f"evaluation_order={full_summary['evaluation_order_attributed_mistakes']} "
        f"coefficient={full_summary['coefficient_error_attributed_mistakes']} "
        f"reconstruction_run={reconstruction_run} "
        f"reconstruction_passed={reconstruction_passed}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
