#!/usr/bin/env python3
"""Balanced continuous asinh-margin regression for eDIRECT hull turns."""

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

import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/balanced_hull_turn_smokescreen")
SUMMARY_PATH = OUTPUT_DIR / "balanced_hull_turn_summary.csv"
BOUNDARY_PATH = OUTPUT_DIR / "balanced_hull_turn_boundary.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "hull_reconstruction_smoke.csv"

TRAIN_EVENTS = 5_000
VALIDATION_EVENTS = 1_000
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 32
MODEL_SEED = 2_645_751
DATA_SEED = turn.DATA_SEED
PASS_ACCURACY = 0.999
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


def _balanced_sample(
    pool: dict[str, np.ndarray], total: int, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    generator = np.random.default_rng(seed)
    keep_indices = np.flatnonzero(pool["label"] == 1)
    pop_indices = np.flatnonzero(pool["label"] == 0)
    available_per_class = min(len(keep_indices), len(pop_indices))
    requested_per_class = total // 2
    if available_per_class < requested_per_class:
        requested_per_class = available_per_class
    selected = np.concatenate(
        (
            generator.choice(keep_indices, requested_per_class, replace=False),
            generator.choice(pop_indices, requested_per_class, replace=False),
        )
    )
    generator.shuffle(selected)
    return (
        {key: values[selected] for key, values in pool.items()},
        {
            "raw_keep": len(keep_indices),
            "raw_pop": len(pop_indices),
            "raw_total": len(pool["label"]),
            "raw_keep_fraction": len(keep_indices) / len(pool["label"]),
            "raw_pop_fraction": len(pop_indices) / len(pool["label"]),
            "balanced_per_class": requested_per_class,
            "balanced_total": 2 * requested_per_class,
        },
    )


def _continuous_dataset(
    selected: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(selected["label"])
    inputs = np.empty((count, 5), dtype=np.float64)
    raw_cross = np.empty(count, dtype=np.float64)
    raw_tolerance = np.empty(count, dtype=np.float64)
    y_tolerance = np.empty(count, dtype=np.float64)
    margin = np.empty(count, dtype=np.float64)
    target = np.empty(count, dtype=np.float64)
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
        if not norm_1 > 0.0 or not norm_2 > 0.0:
            raise RuntimeError(
                f"zero local-vector norm at state {state['source_state_index']}"
            )
        geometry_scale = max(norm_1, norm_2)
        if not geometry_scale > 0.0 or not np.isfinite(geometry_scale):
            raise RuntimeError("invalid local geometry scale")
        cross = delta_d_1 * delta_f_2 - delta_f_1 * delta_d_2
        if cross != float(selected["cross"][row]):
            raise RuntimeError("reconstructed cross differs from audited teacher")
        tolerance = float(selected["tolerance"][row])
        if not tolerance > 0.0 or not np.isfinite(tolerance):
            raise RuntimeError("invalid audited teacher tolerance")
        relative = cross / tolerance
        centered_margin = relative + 1.0
        z = np.arcsinh(centered_margin)
        log_scale_ratio = 2.0 * math.log(geometry_scale) - math.log(tolerance)
        inputs[row] = (
            delta_d_1 / geometry_scale,
            delta_f_1 / geometry_scale,
            delta_d_2 / geometry_scale,
            delta_f_2 / geometry_scale,
            log_scale_ratio,
        )
        raw_cross[row] = cross
        raw_tolerance[row] = tolerance
        y_tolerance[row] = relative
        margin[row] = centered_margin
        target[row] = z
    exact_keep = (target >= 0.0).astype(np.int64)
    if not np.array_equal(exact_keep, selected["label"]):
        raise RuntimeError("continuous z target changed the teacher decision")
    return {
        "inputs": inputs,
        "raw_cross": raw_cross,
        "raw_tolerance": raw_tolerance,
        "y_tolerance": y_tolerance,
        "margin": margin,
        "target": target,
        "teacher_keep": selected["label"].copy(),
        "state_position": selected["state_position"].copy(),
        "triple_indices": selected["triple_indices"].copy(),
    }


class HullTurnMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


class InputStandardization:
    def __init__(self, values: np.ndarray) -> None:
        self.mean = np.mean(values, axis=0)
        self.scale = np.std(values, axis=0)
        self.scale = np.where(self.scale > 0.0, self.scale, 1.0)

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale


def _predict(
    model: HullTurnMLP,
    standardization: InputStandardization,
    inputs: np.ndarray,
) -> np.ndarray:
    values = torch.as_tensor(standardization.apply(inputs), dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            parts.append(model(values[start : start + 2048]))
    return torch.cat(parts).cpu().numpy()


def _decision_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    predicted_keep = prediction >= 0.0
    teacher_keep = labels.astype(bool)
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
        "true_keep_pred_keep": true_keep_pred_keep,
        "true_keep_pred_pop": true_keep_pred_pop,
        "true_pop_pred_pop": true_pop_pred_pop,
        "true_pop_pred_keep": true_pop_pred_keep,
        "keep_count": keep_count,
        "pop_count": pop_count,
    }


def _continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 0.0 and np.std(prediction) > 0.0
        else 0.0
    )
    return {
        "rmse_z": float(np.sqrt(np.mean(error**2))),
        "mae_z": float(np.mean(np.abs(error))),
        "maximum_absolute_error_z": float(np.max(np.abs(error))),
        "correlation_z": correlation,
    }


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[HullTurnMLP, InputStandardization, list[dict[str, Any]], int]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED)
    standardization = InputStandardization(training["inputs"])
    train_inputs = torch.as_tensor(
        standardization.apply(training["inputs"]), dtype=torch.float64
    )
    train_target = torch.as_tensor(training["target"], dtype=torch.float64)
    model = HullTurnMLP().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(train_target))
        total_loss = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            predicted = model(train_inputs[batch])
            loss = nn.functional.smooth_l1_loss(
                predicted, train_target[batch], beta=1.0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
        validation_prediction = _predict(
            model, standardization, validation["inputs"]
        )
        validation_loss = float(
            nn.functional.smooth_l1_loss(
                torch.as_tensor(validation_prediction),
                torch.as_tensor(validation["target"]),
                beta=1.0,
            ).item()
        )
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_huber": total_loss / len(order),
                "validation_huber": validation_loss,
                **_continuous_metrics(validation["target"], validation_prediction),
                **_decision_metrics(
                    validation["teacher_keep"], validation_prediction
                ),
            }
        )
        if validation_loss < best_loss - 1.0e-12:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(
            f"epoch={epoch} validation_huber={validation_loss:.6e} "
            f"decision_accuracy={rows[-1]['decision_accuracy']:.6f} stale={stale}",
            flush=True,
        )
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, standardization, rows, best_epoch


def _boundary_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> list[dict[str, Any]]:
    absolute_margin = np.abs(dataset["margin"])
    definitions = (
        ("abs_margin_lt_0.1", absolute_margin < 0.1),
        ("abs_margin_lt_0.5", absolute_margin < 0.5),
        ("abs_margin_lt_1", absolute_margin < 1.0),
        (
            "abs_margin_1_to_10",
            (absolute_margin >= 1.0) & (absolute_margin < 10.0),
        ),
        ("abs_margin_ge_10", absolute_margin >= 10.0),
    )
    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        indices = np.flatnonzero(mask)
        row: dict[str, Any] = {
            "record_type": "boundary_bucket",
            "bucket": name,
            "count": len(indices),
            "fraction": len(indices) / len(prediction),
        }
        if len(indices):
            row.update(
                _continuous_metrics(
                    dataset["target"][indices], prediction[indices]
                )
            )
            row.update(
                _decision_metrics(
                    dataset["teacher_keep"][indices], prediction[indices]
                )
            )
            row["minimum_abs_margin"] = float(np.min(absolute_margin[indices]))
            row["maximum_abs_margin"] = float(np.max(absolute_margin[indices]))
        rows.append(row)
    return rows


def _source_margin_rows(
    split: str, pool: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    margin = pool["cross"] / pool["tolerance"] + 1.0
    absolute_margin = np.abs(margin)
    labels = pool["label"]
    definitions = (
        ("abs_margin_lt_0.1", absolute_margin < 0.1),
        ("abs_margin_lt_0.5", absolute_margin < 0.5),
        ("abs_margin_lt_1", absolute_margin < 1.0),
        (
            "abs_margin_1_to_10",
            (absolute_margin >= 1.0) & (absolute_margin < 10.0),
        ),
        ("abs_margin_ge_10", absolute_margin >= 10.0),
    )
    rows: list[dict[str, Any]] = []
    for name, mask in definitions:
        indices = np.flatnonzero(mask)
        rows.append(
            {
                "record_type": "raw_source_boundary_bucket",
                "split": split,
                "bucket": name,
                "count": len(indices),
                "fraction": len(indices) / len(margin),
                "keep_count": int(np.sum(labels[indices] == 1)),
                "pop_count": int(np.sum(labels[indices] == 0)),
                "minimum_abs_margin": float(np.min(absolute_margin[indices]))
                if len(indices)
                else "",
                "maximum_abs_margin": float(np.max(absolute_margin[indices]))
                if len(indices)
                else "",
            }
        )
    return rows


def _stack_passes(
    final_metrics: dict[str, Any], boundary_rows: list[dict[str, Any]]
) -> bool:
    if (
        float(final_metrics["decision_accuracy"]) < PASS_ACCURACY
        or float(final_metrics["keep_agreement"]) < PASS_ACCURACY
        or float(final_metrics["pop_agreement"]) < PASS_ACCURACY
        or float(final_metrics["correlation_z"]) < PASS_ACCURACY
    ):
        return False
    near_rows = [
        row
        for row in boundary_rows
        if row["bucket"] in ("abs_margin_lt_0.1", "abs_margin_lt_0.5", "abs_margin_lt_1")
        and int(row["count"]) > 0
    ]
    if not near_rows:
        return False
    return all(float(row["decision_accuracy"]) >= PASS_ACCURACY for row in near_rows)


def _network_input(
    sizes: np.ndarray,
    losses: np.ndarray,
    first: int,
    second: int,
    third: int,
) -> np.ndarray:
    delta_d_1 = float(sizes[second] - sizes[first])
    delta_f_1 = float(losses[second] - losses[first])
    delta_d_2 = float(sizes[third] - sizes[second])
    delta_f_2 = float(losses[third] - losses[second])
    norm_1 = math.hypot(delta_d_1, delta_f_1)
    norm_2 = math.hypot(delta_d_2, delta_f_2)
    if not norm_1 > 0.0 or not norm_2 > 0.0:
        raise RuntimeError("zero local-vector norm in offline hull")
    geometry_scale = max(norm_1, norm_2)
    teacher_scale = max(
        1.0,
        abs(float(losses[first])),
        abs(float(losses[second])),
        abs(float(losses[third])),
    )
    tolerance = turn.HULL_TOLERANCE * teacher_scale
    return np.asarray(
        [
            [
                delta_d_1 / geometry_scale,
                delta_f_1 / geometry_scale,
                delta_d_2 / geometry_scale,
                delta_f_2 / geometry_scale,
                2.0 * math.log(geometry_scale) - math.log(tolerance),
            ]
        ],
        dtype=np.float64,
    )


def _network_hull(
    model: HullTurnMLP,
    standardization: InputStandardization,
    sizes: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    stack: list[int] = []
    for third in range(len(sizes)):
        while len(stack) >= 2:
            predicted_z = float(
                _predict(
                    model,
                    standardization,
                    _network_input(
                        sizes, losses, stack[-2], stack[-1], third
                    ),
                )[0]
            )
            if predicted_z >= 0.0:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64)


def _offline_rows(
    model: HullTurnMLP,
    standardization: InputStandardization,
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    state_rows: list[dict[str, Any]] = []
    for state in states:
        predicted = _network_hull(
            model, standardization, state["sizes"], state["losses"]
        )
        teacher = set(int(value) for value in state["teacher_hull"])
        learned = set(int(value) for value in predicted)
        state_rows.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "representative_cell_count": len(state["sizes"]),
                "exact_hull": teacher == learned,
                "false_hull_additions": len(learned - teacher),
                "false_hull_removals": len(teacher - learned),
            }
        )
    return [
        {
            "record_type": "final_summary",
            "state_count": len(state_rows),
            "exact_hull_states": sum(bool(row["exact_hull"]) for row in state_rows),
            "exact_hull_fraction": float(
                np.mean([row["exact_hull"] for row in state_rows])
            ),
            "average_false_hull_additions": float(
                np.mean([row["false_hull_additions"] for row in state_rows])
            ),
            "average_false_hull_removals": float(
                np.mean([row["false_hull_removals"] for row in state_rows])
            ),
        }
    ] + state_rows


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECONSTRUCTION_PATH.exists():
        RECONSTRUCTION_PATH.unlink()
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
    train_selected, train_counts = _balanced_sample(
        train_pool, TRAIN_EVENTS, DATA_SEED + 11
    )
    validation_selected, validation_counts = _balanced_sample(
        validation_pool, VALIDATION_EVENTS, DATA_SEED + 12
    )
    training = _continuous_dataset(train_selected, train_states)
    validation = _continuous_dataset(validation_selected, validation_states)
    model, standardization, rows, best_epoch = _train(training, validation)
    prediction = _predict(model, standardization, validation["inputs"])
    final_metrics = {
        **_continuous_metrics(validation["target"], prediction),
        **_decision_metrics(validation["teacher_keep"], prediction),
    }
    rows.append(
        {
            "record_type": "final_summary",
            "best_epoch": best_epoch,
            "epochs_trained": sum(row["record_type"] == "epoch" for row in rows),
            "train_events": len(training["target"]),
            "validation_events": len(validation["target"]),
            **final_metrics,
            "target_minimum": float(np.min(validation["target"])),
            "target_maximum": float(np.max(validation["target"])),
            "target_mean": float(np.mean(validation["target"])),
            "target_standard_deviation": float(np.std(validation["target"])),
            "target_p01": float(np.quantile(validation["target"], 0.01)),
            "target_p10": float(np.quantile(validation["target"], 0.10)),
            "target_p25": float(np.quantile(validation["target"], 0.25)),
            "target_p50": float(np.quantile(validation["target"], 0.50)),
            "target_p75": float(np.quantile(validation["target"], 0.75)),
            "target_p90": float(np.quantile(validation["target"], 0.90)),
            "target_p99": float(np.quantile(validation["target"], 0.99)),
            "regression_target": "asinh(C_raw/tau_raw + 1)",
            "classification_target_used": False,
        }
    )
    for split, counts in (("train", train_counts), ("validation", validation_counts)):
        rows.append(
            {
                "record_type": "dataset_balance",
                "split": split,
                **counts,
            }
        )
    exact_accuracy = float(
        np.mean((validation["target"] >= 0.0) == validation["teacher_keep"])
    )
    rows.append(
        {
            "record_type": "deterministic_sanity",
            "correct": int(round(exact_accuracy * len(validation["target"]))),
            "total": len(validation["target"]),
            "accuracy": exact_accuracy,
        }
    )
    balanced_boundary_rows = _boundary_rows(validation, prediction)
    boundary_rows = _source_margin_rows("train", train_pool)
    boundary_rows.extend(_source_margin_rows("validation", validation_pool))
    boundary_rows.extend(balanced_boundary_rows)
    passed = _stack_passes(final_metrics, balanced_boundary_rows)
    rows.append(
        {
            "record_type": "pass_condition",
            "required_accuracy": PASS_ACCURACY,
            "passed": passed,
            "offline_hull_run": passed,
            "trajectory_search_run": False,
        }
    )
    _write_csv(SUMMARY_PATH, rows)
    _write_csv(BOUNDARY_PATH, boundary_rows)
    if passed:
        reconstruction_indices = turn._stratified_state_indices(
            payload["states"], "reconstruction", OFFLINE_STATES, DATA_SEED + 2
        )
        reconstruction_states = turn._build_states(
            payload, reconstruction_indices, workers, "reconstruction"
        )
        _write_csv(
            RECONSTRUCTION_PATH,
            _offline_rows(model, standardization, reconstruction_states),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
