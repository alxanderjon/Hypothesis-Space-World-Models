#!/usr/bin/env python3
"""Learn only log-magnitude of the finite neighboring-hull slope."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_slope_logmag")
SMOKE_PATH = OUTPUT_DIR / "hull_slope_logmag_smoke.csv"
FULL_PATH = OUTPUT_DIR / "hull_slope_logmag_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "hull_slope_logmag_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

TRAIN_TARGET = 5_000
VALIDATION_TARGET = 1_000
MAX_EPOCHS = 10_000
PATIENCE = 800
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
MODEL_SEED = smoke.MODEL_SEED + 1_732_050
LOG_RMSE_GATE = 1.0e-12
LOG_MAX_ERROR_GATE = 1.0e-10
RAW_MAX_RELATIVE_ERROR_GATE = 1.0e-10


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_edges(state: dict[str, Any]) -> dict[str, np.ndarray]:
    hull = np.asarray(state["teacher_hull"], dtype=np.int64)
    sizes = state["sizes"]
    losses = state["losses"]
    count = max(0, len(hull) - 1)
    delta_d = np.empty(count, dtype=np.float64)
    delta_f = np.empty(count, dtype=np.float64)
    slope = np.empty(count, dtype=np.float64)
    for position in range(count):
        left = int(hull[position])
        right = int(hull[position + 1])
        dd = float(sizes[right] - sizes[left])
        df = float(losses[right] - losses[left])
        if not dd > 0.0:
            raise RuntimeError(
                f"delta_d <= 0 at source state {state['source_state_index']}"
            )
        if df == 0.0:
            raise RuntimeError(
                f"delta_f == 0 at source state {state['source_state_index']}"
            )
        if not math.isfinite(dd) or not math.isfinite(df):
            raise RuntimeError("non-finite hull-edge delta")
        value = df / dd
        if not math.isfinite(value) or value == 0.0:
            raise RuntimeError("invalid finite neighboring slope")
        delta_d[position] = dd
        delta_f[position] = df
        slope[position] = value
    return {"delta_d": delta_d, "delta_f": delta_f, "K": slope}


def _build_role_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    inputs: list[tuple[float, float]] = []
    target: list[float] = []
    exact_k: list[float] = []
    state_position: list[int] = []
    edge_position: list[int] = []
    role: list[str] = []
    identity_error: list[float] = []
    unique_edges = 0
    for state_index, state in enumerate(states):
        edge = _state_edges(state)
        for position, (dd, df, value) in enumerate(
            zip(edge["delta_d"], edge["delta_f"], edge["K"])
        ):
            x_f = math.log(abs(float(df)))
            x_d = math.log(float(dd))
            y = math.log(abs(float(value)))
            relation_error = abs(y - (x_f - x_d))
            if relation_error != 0.0:
                # Require exactly the same elementary relation within a small
                # float64 operation-order allowance; never alter it with epsilon.
                if relation_error > 64.0 * np.finfo(np.float64).eps * max(
                    1.0, abs(y), abs(x_f), abs(x_d)
                ):
                    raise RuntimeError("log quotient identity failed")
            for neighbor_role in ("K_next_of_left", "K_prev_of_right"):
                inputs.append((x_f, x_d))
                target.append(y)
                exact_k.append(float(value))
                state_position.append(state_index)
                edge_position.append(position)
                role.append(neighbor_role)
                identity_error.append(relation_error)
            unique_edges += 1
    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "exact_K": np.asarray(exact_k, dtype=np.float64),
        "state_position": np.asarray(state_position, dtype=np.int64),
        "edge_position": np.asarray(edge_position, dtype=np.int64),
        "role": np.asarray(role, dtype=object),
        "log_identity_absolute_error": np.asarray(identity_error, dtype=np.float64),
        "unique_edge_count": np.asarray([unique_edges], dtype=np.int64),
        "split": np.asarray([split], dtype=object),
    }


def _random_select(
    pool: dict[str, np.ndarray], count: int, seed: int
) -> dict[str, np.ndarray]:
    total = len(pool["target"])
    generator = np.random.default_rng(seed)
    indices = (
        np.arange(total, dtype=np.int64)
        if total <= count
        else generator.choice(total, size=count, replace=False)
    )
    return {
        key: (
            values[indices]
            if key not in ("unique_edge_count", "split")
            else values.copy()
        )
        for key, values in pool.items()
    }


def _state_preserving_select(
    pool: dict[str, np.ndarray], target: int, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    generator = np.random.default_rng(seed)
    states = np.unique(pool["state_position"])
    generator.shuffle(states)
    selected_states: list[int] = []
    count = 0
    for state in states:
        selected_states.append(int(state))
        count += int(np.sum(pool["state_position"] == state))
        if count >= target:
            break
    state_array = np.asarray(selected_states, dtype=np.int64)
    mask = np.isin(pool["state_position"], state_array)
    return (
        {
            key: (
                values[mask]
                if key not in ("unique_edge_count", "split")
                else values.copy()
            )
            for key, values in pool.items()
        },
        state_array,
    )


class HomogeneousScaling:
    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        self.input_scale = np.sqrt(np.mean(inputs**2, axis=0))
        self.target_scale = float(np.sqrt(np.mean(target**2)))
        if np.any(self.input_scale <= 0.0) or not np.all(
            np.isfinite(self.input_scale)
        ):
            raise RuntimeError("invalid input RMS scale")
        if not self.target_scale > 0.0 or not math.isfinite(self.target_scale):
            raise RuntimeError("invalid target RMS scale")

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return values / self.input_scale

    def target(self, values: np.ndarray) -> np.ndarray:
        return values / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale

    def coefficients(self, model: nn.Linear) -> np.ndarray:
        weight = model.weight.detach().cpu().numpy()[0]
        return self.target_scale * weight / self.input_scale


def _predict(
    model: nn.Linear, scaling: HomogeneousScaling, inputs: np.ndarray
) -> np.ndarray:
    values = torch.as_tensor(scaling.inputs(inputs), dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(model(values[start : start + 4096]).squeeze(-1))
    return scaling.inverse_target(torch.cat(parts).cpu().numpy())


def _metrics(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    target = dataset["target"]
    log_error = prediction - target
    absolute_log = np.abs(log_error)
    reconstructed = np.sign(dataset["exact_K"]) * np.exp(prediction)
    raw_relative = np.abs(reconstructed - dataset["exact_K"]) / np.abs(
        dataset["exact_K"]
    )
    return {
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "log_mae": float(np.mean(absolute_log)),
        "maximum_absolute_log_error": float(np.max(absolute_log)),
        "log_correlation": float(np.corrcoef(target, prediction)[0, 1]),
        "raw_K_relative_error_p50": float(np.percentile(raw_relative, 50.0)),
        "raw_K_relative_error_p90": float(np.percentile(raw_relative, 90.0)),
        "raw_K_relative_error_p99": float(np.percentile(raw_relative, 99.0)),
        "raw_K_maximum_relative_error": float(np.max(raw_relative)),
        "teacher_sign_used_for_reconstruction": True,
        "learned_sign_used": False,
    }


def _bucket(value: float) -> str:
    absolute = abs(value)
    if absolute < 1.0:
        return "abs_K_lt_1"
    if absolute < 1.0e3:
        return "abs_K_1_to_1e3"
    if absolute < 1.0e6:
        return "abs_K_1e3_to_1e6"
    if absolute < 1.0e12:
        return "abs_K_1e6_to_1e12"
    if absolute < 1.0e24:
        return "abs_K_1e12_to_1e24"
    if absolute < 1.0e40:
        return "abs_K_1e24_to_1e40"
    return "abs_K_ge_1e40"


def _bucket_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    labels = np.asarray([_bucket(value) for value in dataset["exact_K"]], dtype=object)
    order = (
        "abs_K_lt_1",
        "abs_K_1_to_1e3",
        "abs_K_1e3_to_1e6",
        "abs_K_1e6_to_1e12",
        "abs_K_1e12_to_1e24",
        "abs_K_1e24_to_1e40",
        "abs_K_ge_1e40",
    )
    rows: list[dict[str, Any]] = []
    for bucket in order:
        mask = labels == bucket
        row: dict[str, Any] = {
            "record_type": "magnitude_bucket",
            "split": split,
            "bucket": bucket,
            "event_count": int(np.sum(mask)),
        }
        if np.any(mask):
            subset = {
                key: values[mask]
                for key, values in dataset.items()
                if key not in ("unique_edge_count", "split")
            }
            row.update(_metrics(subset, prediction[mask]))
        rows.append(row)
    return rows


def _reconstruct_edges(
    state: dict[str, Any], coefficients: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edge = _state_edges(state)
    inputs = np.column_stack(
        (np.log(np.abs(edge["delta_f"])), np.log(edge["delta_d"]))
    )
    predicted_log = inputs @ coefficients
    reconstructed = np.sign(edge["K"]) * np.exp(predicted_log)
    return edge["K"], reconstructed


def _active_source(k_prev: float, required: float) -> str:
    lower = max(k_prev, required, 0.0)
    if lower == k_prev:
        return "K_prev"
    if lower == required:
        return "K_required"
    return "zero"


def _downstream_metrics(
    states: list[dict[str, Any]], coefficients: np.ndarray
) -> dict[str, Any]:
    ordering_total = 0
    ordering_changes = 0
    lower_max_total = 0
    lower_winner_changes = 0
    feasibility_total = 0
    feasibility_changes = 0
    pass_total = 0
    pass_changes = 0
    k_prev_roles = 0
    k_next_roles = 0
    winning_k_prev_events = 0
    for state in states:
        exact_edges, reconstructed_edges = _reconstruct_edges(state, coefficients)
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = state["sizes"]
        losses = state["losses"]
        f_min = float(np.min(losses))
        target = f_min - 1.0e-4 * abs(f_min)
        for position, index_value in enumerate(hull):
            index = int(index_value)
            size = float(sizes[index])
            loss = float(losses[index])
            exact_prev = -math.inf if position == 0 else float(exact_edges[position - 1])
            predicted_prev = (
                -math.inf if position == 0 else float(reconstructed_edges[position - 1])
            )
            exact_next = math.inf if position == len(hull) - 1 else float(exact_edges[position])
            predicted_next = (
                math.inf
                if position == len(hull) - 1
                else float(reconstructed_edges[position])
            )
            k_prev_roles += int(position > 0)
            k_next_roles += int(position < len(hull) - 1)
            required = (loss - target) / size if size > 0.0 else math.inf
            exact_lower = max(exact_prev, required, 0.0)
            predicted_lower = max(predicted_prev, required, 0.0)
            if position > 0:
                lower_max_total += 1
                exact_source = _active_source(exact_prev, required)
                predicted_source = _active_source(predicted_prev, required)
                lower_winner_changes += int(exact_source != predicted_source)
                winning_k_prev_events += int(exact_source == "K_prev")
            if 0 < position < len(hull) - 1:
                ordering_total += 1
                ordering_changes += int(
                    (exact_prev <= exact_next)
                    != (predicted_prev <= predicted_next)
                )
            exact_feasible = exact_lower <= exact_next + turn.HULL_TOLERANCE
            predicted_feasible = (
                predicted_lower <= predicted_next + turn.HULL_TOLERANCE
            )
            feasibility_total += 1
            feasibility_changes += int(exact_feasible != predicted_feasible)
            exact_pass = exact_next > 0.0 and exact_feasible
            predicted_pass = predicted_next > 0.0 and predicted_feasible
            pass_total += 1
            pass_changes += int(exact_pass != predicted_pass)
    return {
        "comparison_state_count": len(states),
        "K_prev_role_count": k_prev_roles,
        "K_next_role_count": k_next_roles,
        "finite_neighbor_ordering_comparisons": ordering_total,
        "finite_neighbor_ordering_changes": ordering_changes,
        "finite_neighbor_ordering_agreement": (
            1.0 - ordering_changes / ordering_total if ordering_total else 1.0
        ),
        "lower_bound_max_comparisons": lower_max_total,
        "winning_K_prev_events": winning_k_prev_events,
        "lower_bound_winner_changes": lower_winner_changes,
        "lower_bound_winner_agreement": (
            1.0 - lower_winner_changes / lower_max_total if lower_max_total else 1.0
        ),
        "feasibility_comparisons": feasibility_total,
        "feasibility_comparison_changes": feasibility_changes,
        "feasibility_comparison_agreement": 1.0 - feasibility_changes / feasibility_total,
        "improvement_pass_comparisons": pass_total,
        "improvement_pass_changes": pass_changes,
        "improvement_pass_agreement": 1.0 - pass_changes / pass_total,
    }


def _gate(metrics: dict[str, Any], downstream: dict[str, Any]) -> bool:
    return bool(
        metrics["log_rmse"] <= LOG_RMSE_GATE
        and metrics["maximum_absolute_log_error"] <= LOG_MAX_ERROR_GATE
        and metrics["raw_K_maximum_relative_error"] <= RAW_MAX_RELATIVE_ERROR_GATE
        and downstream["finite_neighbor_ordering_changes"] == 0
        and downstream["lower_bound_winner_changes"] == 0
        and downstream["feasibility_comparison_changes"] == 0
        and downstream["improvement_pass_changes"] == 0
    )


def _train(
    training: dict[str, np.ndarray],
    validation_set: dict[str, np.ndarray],
    validation_states: list[dict[str, Any]],
) -> tuple[nn.Linear, HomogeneousScaling, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    scaling = HomogeneousScaling(training["inputs"], training["target"])
    train_x = torch.as_tensor(scaling.inputs(training["inputs"]), dtype=torch.float64)
    train_y = torch.as_tensor(scaling.target(training["target"]), dtype=torch.float64)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial_coefficients = scaling.coefficients(model).copy()
    if np.array_equal(initial_coefficients, np.asarray([1.0, -1.0])):
        raise RuntimeError("forbidden exact coefficient initialization occurred")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, eps=1.0e-15
    )
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    success_streak = 0
    stopping_reason = "maximum_epochs"
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        output = model(train_x).squeeze(-1)
        loss = nn.functional.mse_loss(output, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prediction = _predict(model, scaling, validation_set["inputs"])
        metrics = _metrics(validation_set, prediction)
        mse = metrics["log_rmse"] ** 2
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        coefficients = scaling.coefficients(model)
        local_success = bool(
            metrics["log_rmse"] <= LOG_RMSE_GATE
            and metrics["maximum_absolute_log_error"] <= LOG_MAX_ERROR_GATE
            and metrics["raw_K_maximum_relative_error"]
            <= RAW_MAX_RELATIVE_ERROR_GATE
        )
        success_streak = success_streak + 1 if local_success else 0
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_scaled_mse": float(loss.item()),
                "stale_epochs": stale,
                "success_streak": success_streak,
                "learned_w_f": float(coefficients[0]),
                "learned_w_d": float(coefficients[1]),
                "bias": 0.0,
                **metrics,
            }
        )
        if epoch == 1 or epoch % 250 == 0:
            print(
                f"logmag epoch={epoch} wf={coefficients[0]:.15g} "
                f"wd={coefficients[1]:.15g} rmse={metrics['log_rmse']:.3e} "
                f"max={metrics['maximum_absolute_log_error']:.3e} stale={stale}",
                flush=True,
            )
        if success_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_numerical_precision"
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    coefficients = scaling.coefficients(model)
    prediction = _predict(model, scaling, validation_set["inputs"])
    metrics = _metrics(validation_set, prediction)
    downstream = _downstream_metrics(validation_states, coefficients)
    passed = _gate(metrics, downstream)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "training_events": len(training["target"]),
        "validation_events": len(validation_set["target"]),
        "learned_w_f": float(coefficients[0]),
        "learned_w_d": float(coefficients[1]),
        "bias": 0.0,
        "initial_w_f": float(initial_coefficients[0]),
        "initial_w_d": float(initial_coefficients[1]),
        "log_rmse_gate": LOG_RMSE_GATE,
        "log_max_error_gate": LOG_MAX_ERROR_GATE,
        "raw_max_relative_error_gate": RAW_MAX_RELATIVE_ERROR_GATE,
        "continuous_regression_only": True,
        "sign_supervision_used": False,
        **metrics,
        **downstream,
        "smoke_gate_passed": passed,
    }
    rows.append(summary)
    rows.extend(_bucket_rows(validation_set, prediction, "validation_smoke"))
    rows.append(
        {
            "record_type": "data_assertions",
            "delta_d_nonpositive_count": 0,
            "delta_f_exact_zero_count": 0,
            "epsilon_used": False,
            "maximum_log_identity_operation_order_error": float(
                np.max(validation_set["log_identity_absolute_error"])
            ),
            "training_unique_edges_available": int(training["unique_edge_count"][0]),
            "validation_unique_edges_available": int(validation_set["unique_edge_count"][0]),
        }
    )
    return model, scaling, rows, summary


def _lock_candidate(
    model: nn.Linear,
    scaling: HomogeneousScaling,
    summary: dict[str, Any],
) -> str:
    coefficients = scaling.coefficients(model)
    checkpoint = {
        "format_version": 1,
        "component": "hull_slope_log_magnitude",
        "status": "CANDIDATE_NOT_COMPLETE_SLOPE_LOCK",
        "architecture": "log|K| = w_f*log|delta_f| + w_d*log(delta_d); no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_scale": torch.as_tensor(scaling.input_scale, dtype=torch.float64),
        "target_scale": torch.tensor(scaling.target_scale, dtype=torch.float64),
        "effective_coefficients": torch.as_tensor(coefficients, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "teacher_sign_required_for_reconstruction": True,
        "full_pool_mistake_counts": {
            "ordering": int(summary["finite_neighbor_ordering_changes"]),
            "lower_winner": int(summary["lower_bound_winner_changes"]),
            "feasibility": int(summary["feasibility_comparison_changes"]),
            "improvement_pass": int(summary["improvement_pass_changes"]),
        },
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "CANDIDATE_NOT_COMPLETE_SLOPE_LOCK":
        raise RuntimeError("candidate checkpoint readback failed")
    checkpoint_hash = _sha256(CANDIDATE_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
    )
    return checkpoint_hash


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for conditional in (FULL_PATH, CANDIDATE_PATH, HASH_PATH):
        if conditional.exists():
            conditional.unlink()
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
    train_states = turn._build_states(payload, train_indices, workers, "logmag train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "logmag validation"
    )
    train_pool = _build_role_pool(train_states, "train")
    validation_pool = _build_role_pool(validation_states, "validation")
    training = _random_select(train_pool, TRAIN_TARGET, MODEL_SEED + 11)
    validation_set, selected_state_indices = _state_preserving_select(
        validation_pool, VALIDATION_TARGET, MODEL_SEED + 12
    )
    smoke_states = [validation_states[int(index)] for index in selected_state_indices]
    model, scaling, rows, summary = _train(
        training, validation_set, smoke_states
    )
    _write_csv(SMOKE_PATH, rows)
    if not bool(summary["smoke_gate_passed"]):
        print("log-magnitude smoke gate failed; stopped", flush=True)
        return

    full_prediction = _predict(model, scaling, validation_pool["inputs"])
    full_metrics = _metrics(validation_pool, full_prediction)
    coefficients = scaling.coefficients(model)
    downstream = _downstream_metrics(validation_states, coefficients)
    full_summary = {
        "record_type": "final_summary",
        "event_count": len(validation_pool["target"]),
        "unique_edge_count": int(validation_pool["unique_edge_count"][0]),
        "learned_w_f": float(coefficients[0]),
        "learned_w_d": float(coefficients[1]),
        "bias": 0.0,
        **full_metrics,
        **downstream,
    }
    full_passed = _gate(full_metrics, downstream)
    full_summary["full_pool_gate_passed"] = full_passed
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(_bucket_rows(validation_pool, full_prediction, "full_held_out_pool"))
    if not full_passed:
        _write_csv(FULL_PATH, full_rows)
        print("log-magnitude full-pool gate failed; no candidate saved", flush=True)
        return
    checkpoint_hash = _lock_candidate(model, scaling, full_summary)
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["candidate_status"] = "CANDIDATE_NOT_COMPLETE_SLOPE_LOCK"
    _write_csv(FULL_PATH, full_rows)
    print(
        f"log-magnitude candidate saved wf={coefficients[0]:.17g} "
        f"wd={coefficients[1]:.17g} sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
