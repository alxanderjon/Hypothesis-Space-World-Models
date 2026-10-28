#!/usr/bin/env python3
"""Learn only log(A/d_i) for K_required with deterministic A."""

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

import experiments.sho.audit_stage1b_improvement_condition as improvement
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_slope_logmag as slope_logmag
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/k_required_logquotient")
SMOKE_PATH = OUTPUT_DIR / "k_required_logquotient_smoke.csv"
FULL_PATH = OUTPUT_DIR / "k_required_logquotient_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "k_required_logquotient_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

LOGMAG_PATH = Path(
    "outputs/sho/stage1B/hull_slope_logmag/hull_slope_logmag_candidate.pt"
)
LOGMAG_SHA256 = "a4e1f1dbedf5f42121c441b6ced7b0f243fd78aed28b7e8c9c838178f675b249"
SIGN_PATH = Path(
    "outputs/sho/stage1B/hull_slope_sign/hull_slope_sign_locked.pt"
)
SIGN_SHA256 = "fb43410c48f89189757d1a397e95945e8717f7b069ebda88eeb019684a98a848"

MODEL_SEED = smoke.MODEL_SEED + 3_162_277
VALIDATION_TARGET = 1_000
MAX_EPOCHS = 10_000
PATIENCE = 800
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
LOG_RMSE_GATE = 1.0e-12
LOG_MAX_ERROR_GATE = 1.0e-10
RAW_MAX_RELATIVE_ERROR_GATE = 1.0e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_locked_hashes() -> None:
    for path, expected, name in (
        (LOGMAG_PATH, LOGMAG_SHA256, "locked neighboring-slope log magnitude"),
        (SIGN_PATH, SIGN_SHA256, "locked neighboring-slope sign"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {name}: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} hash mismatch: {actual} != {expected}")


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


def _build_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    inputs: list[tuple[float, float]] = []
    target: list[float] = []
    exact_k: list[float] = []
    improvement_amount: list[float] = []
    d_i_values: list[float] = []
    k_prev_values: list[float] = []
    k_next_values: list[float] = []
    exact_lower_values: list[float] = []
    teacher_winner: list[str] = []
    exact_raw_order: list[bool] = []
    exact_feasibility: list[bool] = []
    exact_pass: list[bool] = []
    state_position: list[int] = []
    source_state_index: list[int] = []
    hull_position_values: list[int] = []
    identity_error: list[float] = []
    for local_state, state in enumerate(states):
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = np.asarray(state["sizes"], dtype=np.float64)
        losses = np.asarray(state["losses"], dtype=np.float64)
        f_min = float(np.min(losses))
        improvement_target = f_min - edirect.EDIRECT_EPSILON * abs(f_min)
        for hull_position, index_value in enumerate(hull):
            index = int(index_value)
            event = improvement._event(
                local_state,
                state,
                hull_position,
                index,
                hull,
                improvement_target,
                f_min,
            )
            f_i = float(losses[index])
            d_i = float(sizes[index])
            # Preserve the teacher's actual float64 operation order.  This is
            # algebraically f_i - f_min + epsilon*abs(f_min), but expanding it
            # before evaluation changes rounding for some events.
            amount = f_i - improvement_target
            if not amount > 0.0:
                raise RuntimeError(
                    f"A <= 0 split={split} state={local_state} hull={hull_position}"
                )
            if not d_i > 0.0:
                raise RuntimeError(
                    f"d_i <= 0 split={split} state={local_state} hull={hull_position}"
                )
            teacher_k = float(event["improvement_required_slope"])
            direct_k = amount / d_i
            if direct_k != teacher_k:
                raise RuntimeError("deterministic numerator quotient differs from teacher")
            x_a = math.log(amount)
            x_d = math.log(d_i)
            y = math.log(teacher_k)
            relation_error = abs(y - (x_a - x_d))
            allowance = 64.0 * np.finfo(np.float64).eps * max(
                1.0, abs(y), abs(x_a), abs(x_d)
            )
            if relation_error > allowance:
                raise RuntimeError("log quotient identity exceeded float64 allowance")
            k_prev = float(event["lower_hull_slope"])
            k_next = float(event["upper_hull_slope"])
            exact_lower = max(k_prev, teacher_k, 0.0)
            raw_order = exact_lower <= k_next
            feasible = exact_lower <= k_next + turn.HULL_TOLERANCE
            passed = k_next > 0.0 and feasible
            if passed != bool(event["improvement_pass"]):
                raise RuntimeError("teacher improvement PASS reconstruction mismatch")
            inputs.append((x_a, x_d))
            target.append(y)
            exact_k.append(teacher_k)
            improvement_amount.append(amount)
            d_i_values.append(d_i)
            k_prev_values.append(k_prev)
            k_next_values.append(k_next)
            exact_lower_values.append(exact_lower)
            teacher_winner.append(slope_logmag._active_source(k_prev, teacher_k))
            exact_raw_order.append(raw_order)
            exact_feasibility.append(feasible)
            exact_pass.append(passed)
            state_position.append(local_state)
            source_state_index.append(int(state["source_state_index"]))
            hull_position_values.append(hull_position)
            identity_error.append(relation_error)
    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "target": np.asarray(target, dtype=np.float64),
        "exact_K_required": np.asarray(exact_k, dtype=np.float64),
        "improvement_amount": np.asarray(improvement_amount, dtype=np.float64),
        "d_i": np.asarray(d_i_values, dtype=np.float64),
        "K_prev": np.asarray(k_prev_values, dtype=np.float64),
        "K_next": np.asarray(k_next_values, dtype=np.float64),
        "exact_K_lower": np.asarray(exact_lower_values, dtype=np.float64),
        "teacher_winner": np.asarray(teacher_winner, dtype=object),
        "exact_raw_order": np.asarray(exact_raw_order, dtype=bool),
        "exact_feasibility": np.asarray(exact_feasibility, dtype=bool),
        "exact_pass": np.asarray(exact_pass, dtype=bool),
        "state_position": np.asarray(state_position, dtype=np.int64),
        "source_state_index": np.asarray(source_state_index, dtype=np.int64),
        "hull_position": np.asarray(hull_position_values, dtype=np.int64),
        "log_identity_absolute_error": np.asarray(identity_error, dtype=np.float64),
        "split": np.asarray([split], dtype=object),
    }


def _subset(pool: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        key: values[mask] if key != "split" else values.copy()
        for key, values in pool.items()
    }


def _state_preserving_subset(
    pool: dict[str, np.ndarray], target: int, seed: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    generator = np.random.default_rng(seed)
    states = np.unique(pool["state_position"])
    generator.shuffle(states)
    chosen: list[int] = []
    count = 0
    for state in states:
        chosen.append(int(state))
        count += int(np.sum(pool["state_position"] == state))
        if count >= target:
            break
    chosen_array = np.asarray(chosen, dtype=np.int64)
    return _subset(pool, np.isin(pool["state_position"], chosen_array)), chosen_array


def _bucket(value: float) -> str:
    if value < 1.0:
        return "K_required_lt_1"
    if value < 1.0e3:
        return "K_required_1_to_1e3"
    if value < 1.0e6:
        return "K_required_1e3_to_1e6"
    if value < 1.0e12:
        return "K_required_1e6_to_1e12"
    if value < 1.0e24:
        return "K_required_1e12_to_1e24"
    if value < 1.0e40:
        return "K_required_1e24_to_1e40"
    return "K_required_ge_1e40"


def _continuous_metrics(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    error = prediction - dataset["target"]
    absolute = np.abs(error)
    reconstructed = np.exp(prediction)
    relative = np.abs(reconstructed - dataset["exact_K_required"]) / dataset[
        "exact_K_required"
    ]
    return {
        "log_rmse": float(np.sqrt(np.mean(error**2))),
        "log_mae": float(np.mean(absolute)),
        "maximum_absolute_log_error": float(np.max(absolute)),
        "log_correlation": float(np.corrcoef(dataset["target"], prediction)[0, 1]),
        "raw_K_required_median_relative_error": float(np.median(relative)),
        "raw_K_required_relative_error_p90": float(np.percentile(relative, 90.0)),
        "raw_K_required_relative_error_p95": float(np.percentile(relative, 95.0)),
        "raw_K_required_relative_error_p99": float(np.percentile(relative, 99.0)),
        "raw_K_required_maximum_relative_error": float(np.max(relative)),
    }


def _downstream_metrics(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    learned_k = np.exp(prediction)
    learned_lower = np.maximum(
        np.maximum(dataset["K_prev"], learned_k), 0.0
    )
    learned_winner = np.asarray(
        [
            slope_logmag._active_source(float(previous), float(required))
            for previous, required in zip(dataset["K_prev"], learned_k)
        ],
        dtype=object,
    )
    learned_raw_order = learned_lower <= dataset["K_next"]
    learned_feasibility = (
        learned_lower <= dataset["K_next"] + turn.HULL_TOLERANCE
    )
    learned_pass = (dataset["K_next"] > 0.0) & learned_feasibility
    winner_changes = learned_winner != dataset["teacher_winner"]
    raw_order_changes = learned_raw_order != dataset["exact_raw_order"]
    feasibility_changes = learned_feasibility != dataset["exact_feasibility"]
    pass_changes = learned_pass != dataset["exact_pass"]
    teacher_required_wins = dataset["teacher_winner"] == "K_required"
    subset_relative = np.abs(
        learned_k[teacher_required_wins]
        - dataset["exact_K_required"][teacher_required_wins]
    ) / dataset["exact_K_required"][teacher_required_wins]
    return {
        "downstream_event_count": len(prediction),
        "teacher_K_required_winner_count": int(np.sum(teacher_required_wins)),
        "lower_bound_winner_changes": int(np.sum(winner_changes)),
        "K_lower_vs_K_upper_ordering_changes": int(np.sum(raw_order_changes)),
        "downstream_feasibility_comparison_changes": int(
            np.sum(feasibility_changes)
        ),
        "final_improvement_PASS_FAIL_changes": int(np.sum(pass_changes)),
        "teacher_winner_subset_lower_bound_winner_changes": int(
            np.sum(winner_changes[teacher_required_wins])
        ),
        "teacher_winner_subset_ordering_changes": int(
            np.sum(raw_order_changes[teacher_required_wins])
        ),
        "teacher_winner_subset_feasibility_changes": int(
            np.sum(feasibility_changes[teacher_required_wins])
        ),
        "teacher_winner_subset_PASS_FAIL_changes": int(
            np.sum(pass_changes[teacher_required_wins])
        ),
        "teacher_winner_subset_maximum_relative_error": float(
            np.max(subset_relative)
        ),
    }


def _bucket_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    labels = np.asarray(
        [_bucket(value) for value in dataset["exact_K_required"]], dtype=object
    )
    order = (
        "K_required_lt_1",
        "K_required_1_to_1e3",
        "K_required_1e3_to_1e6",
        "K_required_1e6_to_1e12",
        "K_required_1e12_to_1e24",
        "K_required_1e24_to_1e40",
        "K_required_ge_1e40",
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
            row.update(_continuous_metrics(_subset(dataset, mask), prediction[mask]))
        rows.append(row)
    return rows


def _predict(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    inputs: np.ndarray,
) -> np.ndarray:
    return slope_logmag._predict(model, scaling, inputs)


def _gate(
    continuous: dict[str, Any], downstream: dict[str, Any]
) -> bool:
    return bool(
        continuous["log_rmse"] <= LOG_RMSE_GATE
        and continuous["maximum_absolute_log_error"] <= LOG_MAX_ERROR_GATE
        and continuous["raw_K_required_maximum_relative_error"]
        <= RAW_MAX_RELATIVE_ERROR_GATE
        and downstream["lower_bound_winner_changes"] == 0
        and downstream["K_lower_vs_K_upper_ordering_changes"] == 0
        and downstream["downstream_feasibility_comparison_changes"] == 0
        and downstream["final_improvement_PASS_FAIL_changes"] == 0
    )


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[nn.Linear, slope_logmag.HomogeneousScaling, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    scaling = slope_logmag.HomogeneousScaling(
        training["inputs"], training["target"]
    )
    train_x = torch.as_tensor(scaling.inputs(training["inputs"]), dtype=torch.float64)
    train_y = torch.as_tensor(scaling.target(training["target"]), dtype=torch.float64)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial_coefficients = scaling.coefficients(model).copy()
    if np.array_equal(initial_coefficients, np.asarray([1.0, -1.0])):
        raise RuntimeError("forbidden exact coefficient initialization")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1.0e-15)
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    success_streak = 0
    stopping_reason = "maximum_epochs"
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        output = model(train_x).squeeze(-1)
        loss = nn.functional.mse_loss(output, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prediction = _predict(model, scaling, validation["inputs"])
        continuous = _continuous_metrics(validation, prediction)
        downstream = _downstream_metrics(validation, prediction)
        mse = continuous["log_rmse"] ** 2
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        local_success = _gate(continuous, downstream)
        success_streak = success_streak + 1 if local_success else 0
        coefficients = scaling.coefficients(model)
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_scaled_mse": float(loss.item()),
                "learned_w_A": float(coefficients[0]),
                "learned_w_d": float(coefficients[1]),
                "bias": 0.0,
                "stale_epochs": stale,
                "success_streak": success_streak,
                **continuous,
                **downstream,
            }
        )
        if epoch == 1 or epoch % 250 == 0:
            print(
                f"Kreq logquot epoch={epoch} wA={coefficients[0]:.15g} "
                f"wd={coefficients[1]:.15g} rmse={continuous['log_rmse']:.3e} "
                f"max={continuous['maximum_absolute_log_error']:.3e}",
                flush=True,
            )
        if success_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_numerical_precision"
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break

    model.load_state_dict(best_state)
    prediction = _predict(model, scaling, validation["inputs"])
    continuous = _continuous_metrics(validation, prediction)
    downstream = _downstream_metrics(validation, prediction)
    coefficients = scaling.coefficients(model)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "training_event_count": len(training["target"]),
        "validation_event_count": len(validation["target"]),
        "training_state_count": len(np.unique(training["state_position"])),
        "validation_state_count": len(np.unique(validation["state_position"])),
        "learned_w_A": float(coefficients[0]),
        "learned_w_d": float(coefficients[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "initial_w_A": float(initial_coefficients[0]),
        "initial_w_d": float(initial_coefficients[1]),
        "deterministic_improvement_numerator_used": True,
        "numerator_learning_performed": False,
        "no_epsilon_used": True,
        "log_rmse_gate": LOG_RMSE_GATE,
        "log_max_error_gate": LOG_MAX_ERROR_GATE,
        "raw_max_relative_error_gate": RAW_MAX_RELATIVE_ERROR_GATE,
        **continuous,
        **downstream,
    }
    summary["smoke_gate_passed"] = _gate(continuous, downstream)
    rows.append(summary)
    rows.extend(_bucket_rows(validation, prediction, "validation_smoke"))
    rows.append(
        {
            "record_type": "data_assertions",
            "A_nonpositive_count": int(np.sum(training["improvement_amount"] <= 0.0))
            + int(np.sum(validation["improvement_amount"] <= 0.0)),
            "d_i_nonpositive_count": int(np.sum(training["d_i"] <= 0.0))
            + int(np.sum(validation["d_i"] <= 0.0)),
            "maximum_log_identity_operation_order_error": float(
                max(
                    np.max(training["log_identity_absolute_error"]),
                    np.max(validation["log_identity_absolute_error"]),
                )
            ),
            "locked_log_magnitude_sha256_verified": LOGMAG_SHA256,
            "locked_sign_sha256_verified": SIGN_SHA256,
        }
    )
    return model, scaling, rows, summary


def _save_candidate(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    full_summary: dict[str, Any],
) -> str:
    coefficients = scaling.coefficients(model)
    checkpoint = {
        "format_version": 1,
        "component": "K_required_log_quotient",
        "status": "FROZEN_CANDIDATE",
        "architecture": "log(K_required) = w_A*log(A) + w_d*log(d_i); no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_scale": torch.as_tensor(scaling.input_scale, dtype=torch.float64),
        "target_scale": torch.tensor(scaling.target_scale, dtype=torch.float64),
        "effective_coefficients": torch.as_tensor(coefficients, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "deterministic_numerator_definition": "target=f_min-1e-4*abs(f_min); A=f_i-target (algebraically f_i-f_min+1e-4*abs(f_min))",
        "deterministic_numerator_required": True,
        "complete_K_required_operator_locked": False,
        "neighboring_hull_slope_logmag_sha256": LOGMAG_SHA256,
        "neighboring_hull_slope_sign_sha256": SIGN_SHA256,
        "full_pool_summary": full_summary,
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored["status"] != "FROZEN_CANDIDATE":
        raise RuntimeError("candidate checkpoint readback failed")
    if float(restored["bias"].item()) != 0.0:
        raise RuntimeError("candidate bias readback is not exactly zero")
    checkpoint_hash = _sha256(CANDIDATE_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
    )
    return checkpoint_hash


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_locked_hashes()
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
    train_states = turn._build_states(
        payload, train_indices, workers, "Kreq logquot train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "Kreq logquot validation"
    )
    training = _build_pool(train_states, "train")
    full_validation = _build_pool(validation_states, "validation")
    smoke_validation, smoke_state_indices = _state_preserving_subset(
        full_validation, VALIDATION_TARGET, MODEL_SEED + 1
    )
    model, scaling, smoke_rows, smoke_summary = _train(
        training, smoke_validation
    )
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        print("K_required log-quotient smoke gate failed; stopped", flush=True)
        return

    full_prediction = _predict(model, scaling, full_validation["inputs"])
    continuous = _continuous_metrics(full_validation, full_prediction)
    downstream = _downstream_metrics(full_validation, full_prediction)
    coefficients = scaling.coefficients(model)
    full_summary = {
        "record_type": "final_summary",
        "event_count": len(full_validation["target"]),
        "state_count": len(validation_states),
        "smoke_validation_state_count": len(smoke_state_indices),
        "learned_w_A": float(coefficients[0]),
        "learned_w_d": float(coefficients[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "deterministic_improvement_numerator_used": True,
        **continuous,
        **downstream,
    }
    full_summary["full_pool_gate_passed"] = _gate(continuous, downstream)
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(
        _bucket_rows(full_validation, full_prediction, "full_held_out_pool")
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("K_required log-quotient full-pool gate failed; stopped", flush=True)
        return

    checkpoint_hash = _save_candidate(model, scaling, full_summary)
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["candidate_status"] = "FROZEN_CANDIDATE"
    full_summary["complete_K_required_operator_locked"] = False
    _write_csv(FULL_PATH, full_rows)
    print(
        f"K_required log quotient FROZEN_CANDIDATE wA={coefficients[0]:.17g} "
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
