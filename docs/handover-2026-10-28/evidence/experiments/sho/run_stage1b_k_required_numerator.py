#!/usr/bin/env python3
"""Learn only the normalized improvement numerator A = f_i - target."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/k_required_numerator")
AUDIT_PATH = OUTPUT_DIR / "k_required_numerator_audit.csv"
SMOKE_PATH = OUTPUT_DIR / "k_required_numerator_smoke.csv"
FULL_PATH = OUTPUT_DIR / "k_required_numerator_full_pipeline.csv"
NUMERATOR_PATH = OUTPUT_DIR / "k_required_numerator_locked.pt"
NUMERATOR_HASH_PATH = NUMERATOR_PATH.with_suffix(".sha256")
LOCK_PATH = OUTPUT_DIR / "k_required_operator_lock_summary.csv"

LOGQUOTIENT_PATH = Path(
    "outputs/sho/stage1B/k_required_logquotient/k_required_logquotient_candidate.pt"
)
LOGQUOTIENT_SHA256 = (
    "0f3d4c2b59a41d23af99d2074e477944307d32bf14273b1239e6ffa5748bdb02"
)
SLOPE_LOGMAG_PATH = Path(
    "outputs/sho/stage1B/hull_slope_logmag/hull_slope_logmag_candidate.pt"
)
SLOPE_LOGMAG_SHA256 = (
    "a4e1f1dbedf5f42121c441b6ced7b0f243fd78aed28b7e8c9c838178f675b249"
)
SLOPE_SIGN_PATH = Path(
    "outputs/sho/stage1B/hull_slope_sign/hull_slope_sign_locked.pt"
)
SLOPE_SIGN_SHA256 = (
    "fb43410c48f89189757d1a397e95945e8717f7b069ebda88eeb019684a98a848"
)

MODEL_SEED = smoke.MODEL_SEED + 4_472_135
MAX_EPOCHS = 20_000
PATIENCE = 1_200
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
NORMALIZED_RMSE_GATE = 1.0e-12
NORMALIZED_MAX_ERROR_GATE = 1.0e-11
A_MAX_RELATIVE_ERROR_GATE = 1.0e-9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_hashes() -> None:
    for path, expected, name in (
        (LOGQUOTIENT_PATH, LOGQUOTIENT_SHA256, "frozen K_required log quotient"),
        (SLOPE_LOGMAG_PATH, SLOPE_LOGMAG_SHA256, "locked slope log magnitude"),
        (SLOPE_SIGN_PATH, SLOPE_SIGN_SHA256, "locked slope sign"),
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


def _stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "median": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
        "exact_zero_count": int(np.sum(values == 0.0)),
    }


def _build_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    inputs: list[tuple[float, float]] = []
    target_y: list[float] = []
    f_i_values: list[float] = []
    f_min_values: list[float] = []
    target_values: list[float] = []
    a_values: list[float] = []
    scale_values: list[float] = []
    d_i_values: list[float] = []
    exact_k_required: list[float] = []
    exact_k_prev: list[float] = []
    exact_k_next: list[float] = []
    teacher_winner: list[str] = []
    exact_raw_order: list[bool] = []
    exact_feasibility: list[bool] = []
    exact_pass: list[bool] = []
    state_position: list[int] = []
    source_state_index: list[int] = []
    hull_position_values: list[int] = []
    violations: list[tuple[int, int, str]] = []
    for local_state, state in enumerate(states):
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = np.asarray(state["sizes"], dtype=np.float64)
        losses = np.asarray(state["losses"], dtype=np.float64)
        f_min = float(np.min(losses))
        teacher_target = f_min - edirect.EDIRECT_EPSILON * abs(f_min)
        for hull_position, index_value in enumerate(hull):
            index = int(index_value)
            f_i = float(losses[index])
            d_i = float(sizes[index])
            # Preserve exact teacher float64 order; never algebraically expand.
            amount = f_i - teacher_target
            scale = max(abs(f_i), abs(f_min))
            checks = {
                "f_i_lt_f_min": f_i < f_min,
                "f_min_negative": f_min < 0.0,
                "A_nonpositive": amount <= 0.0,
                "S_nonpositive": scale <= 0.0,
            }
            for reason, failed in checks.items():
                if failed:
                    violations.append((local_state, hull_position, reason))
            if any(checks.values()):
                continue
            event = improvement._event(
                local_state,
                state,
                hull_position,
                index,
                hull,
                teacher_target,
                f_min,
            )
            teacher_k = float(event["improvement_required_slope"])
            if d_i <= 0.0 or amount / d_i != teacher_k:
                violations.append((local_state, hull_position, "quotient_mismatch_or_d_i"))
                continue
            x_i = f_i / scale
            x_min = f_min / scale
            y = amount / scale
            k_prev = float(event["lower_hull_slope"])
            k_next = float(event["upper_hull_slope"])
            exact_lower = max(k_prev, teacher_k, 0.0)
            feasible = exact_lower <= k_next + turn.HULL_TOLERANCE
            passed = k_next > 0.0 and feasible
            if passed != bool(event["improvement_pass"]):
                violations.append((local_state, hull_position, "teacher_PASS_mismatch"))
                continue
            inputs.append((x_i, x_min))
            target_y.append(y)
            f_i_values.append(f_i)
            f_min_values.append(f_min)
            target_values.append(teacher_target)
            a_values.append(amount)
            scale_values.append(scale)
            d_i_values.append(d_i)
            exact_k_required.append(teacher_k)
            exact_k_prev.append(k_prev)
            exact_k_next.append(k_next)
            teacher_winner.append(slope_logmag._active_source(k_prev, teacher_k))
            exact_raw_order.append(exact_lower <= k_next)
            exact_feasibility.append(feasible)
            exact_pass.append(passed)
            state_position.append(local_state)
            source_state_index.append(int(state["source_state_index"]))
            hull_position_values.append(hull_position)
    return {
        "inputs": np.asarray(inputs, dtype=np.float64),
        "target_y": np.asarray(target_y, dtype=np.float64),
        "f_i": np.asarray(f_i_values, dtype=np.float64),
        "f_min": np.asarray(f_min_values, dtype=np.float64),
        "teacher_target": np.asarray(target_values, dtype=np.float64),
        "A": np.asarray(a_values, dtype=np.float64),
        "S": np.asarray(scale_values, dtype=np.float64),
        "d_i": np.asarray(d_i_values, dtype=np.float64),
        "exact_K_required": np.asarray(exact_k_required, dtype=np.float64),
        "exact_K_prev": np.asarray(exact_k_prev, dtype=np.float64),
        "exact_K_next": np.asarray(exact_k_next, dtype=np.float64),
        "teacher_winner": np.asarray(teacher_winner, dtype=object),
        "exact_raw_order": np.asarray(exact_raw_order, dtype=bool),
        "exact_feasibility": np.asarray(exact_feasibility, dtype=bool),
        "exact_pass": np.asarray(exact_pass, dtype=bool),
        "state_position": np.asarray(state_position, dtype=np.int64),
        "source_state_index": np.asarray(source_state_index, dtype=np.int64),
        "hull_position": np.asarray(hull_position_values, dtype=np.int64),
        "violations": np.asarray(violations, dtype=object),
        "split": np.asarray([split], dtype=object),
    }


def _audit_rows(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], bool]:
    violation_rows: list[dict[str, Any]] = []
    for pool in (training, validation):
        for violation in pool["violations"]:
            local_state, hull_position, reason = violation
            violation_rows.append(
                {
                    "record_type": "violation",
                    "split": str(pool["split"][0]),
                    "state_position": int(local_state),
                    "hull_position": int(hull_position),
                    "reason": str(reason),
                }
            )
    if violation_rows:
        return violation_rows, False
    combined = {
        name: np.concatenate((training[name], validation[name]))
        for name in ("f_i", "f_min", "A", "S", "target_y")
    }
    rows: list[dict[str, Any]] = [
        {
            "record_type": "assumption_summary",
            "total_event_count": len(combined["A"]),
            "training_event_count": len(training["A"]),
            "validation_event_count": len(validation["A"]),
            "f_i_lt_f_min_count": 0,
            "f_min_negative_count": 0,
            "A_nonpositive_count": 0,
            "S_nonpositive_count": 0,
            "all_assumptions_passed": True,
            "teacher_arithmetic_order": "target=f_min-1e-4*abs(f_min); A=f_i-target",
            "epsilon_added": False,
            "frozen_logquotient_sha256_verified": LOGQUOTIENT_SHA256,
            "slope_logmag_sha256_verified": SLOPE_LOGMAG_SHA256,
            "slope_sign_sha256_verified": SLOPE_SIGN_SHA256,
        }
    ]
    for name, values in (
        ("f_i", combined["f_i"]),
        ("f_min", combined["f_min"]),
        ("A", combined["A"]),
        ("A_over_S", combined["target_y"]),
    ):
        rows.append(
            {
                "record_type": "variable_distribution",
                "variable": name,
                **_stats(values),
            }
        )
    return rows, True


def _predict(model: nn.Linear, inputs: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return (
            model(torch.as_tensor(inputs, dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )


def _continuous_metrics(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    error = prediction - dataset["target_y"]
    absolute = np.abs(error)
    a_hat = dataset["S"] * prediction
    relative = np.abs(a_hat - dataset["A"]) / dataset["A"]
    return {
        "normalized_target_rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_target_mae": float(np.mean(absolute)),
        "normalized_target_maximum_absolute_error": float(np.max(absolute)),
        "normalized_target_correlation": float(
            np.corrcoef(dataset["target_y"], prediction)[0, 1]
        ),
        "A_hat_nonpositive_count": int(np.sum(a_hat <= 0.0)),
        "A_median_relative_error": float(np.median(relative)),
        "A_relative_error_p90": float(np.percentile(relative, 90.0)),
        "A_relative_error_p95": float(np.percentile(relative, 95.0)),
        "A_relative_error_p99": float(np.percentile(relative, 99.0)),
        "A_maximum_relative_error": float(np.max(relative)),
    }


def _numerator_gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["normalized_target_rmse"] <= NORMALIZED_RMSE_GATE
        and metrics["normalized_target_maximum_absolute_error"]
        <= NORMALIZED_MAX_ERROR_GATE
        and metrics["A_hat_nonpositive_count"] == 0
        and metrics["A_maximum_relative_error"] <= A_MAX_RELATIVE_ERROR_GATE
    )


def _magnitude_bucket(value: float) -> str:
    if value < 1.0e-4:
        return "A_over_S_lt_1e-4"
    if value < 1.0e-3:
        return "A_over_S_1e-4_to_1e-3"
    if value < 1.0e-2:
        return "A_over_S_1e-3_to_1e-2"
    if value < 1.0e-1:
        return "A_over_S_1e-2_to_1e-1"
    if value < 0.5:
        return "A_over_S_1e-1_to_0p5"
    return "A_over_S_0p5_to_1"


def _bucket_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    labels = np.asarray(
        [_magnitude_bucket(value) for value in dataset["target_y"]], dtype=object
    )
    order = (
        "A_over_S_lt_1e-4",
        "A_over_S_1e-4_to_1e-3",
        "A_over_S_1e-3_to_1e-2",
        "A_over_S_1e-2_to_1e-1",
        "A_over_S_1e-1_to_0p5",
        "A_over_S_0p5_to_1",
    )
    rows: list[dict[str, Any]] = []
    for bucket in order:
        mask = labels == bucket
        row: dict[str, Any] = {
            "record_type": "A_over_S_bucket",
            "split": split,
            "bucket": bucket,
            "event_count": int(np.sum(mask)),
        }
        if np.any(mask):
            subset = {
                key: values[mask]
                for key, values in dataset.items()
                if key not in ("split", "violations")
            }
            row.update(_continuous_metrics(subset, prediction[mask]))
        rows.append(row)
    return rows


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial = model.weight.detach().cpu().numpy()[0].copy()
    known = np.asarray([1.0, -0.9999], dtype=np.float64)
    if np.array_equal(initial, known):
        raise RuntimeError("forbidden known coefficient initialization")
    train_x = torch.as_tensor(training["inputs"], dtype=torch.float64)
    train_y = torch.as_tensor(training["target_y"], dtype=torch.float64)
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
        prediction = _predict(model, validation["inputs"])
        metrics = _continuous_metrics(validation, prediction)
        mse = metrics["normalized_target_rmse"] ** 2
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        success = _numerator_gate(metrics)
        success_streak = success_streak + 1 if success else 0
        weight = model.weight.detach().cpu().numpy()[0]
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_mse": float(loss.item()),
                "learned_w_i": float(weight[0]),
                "learned_w_min": float(weight[1]),
                "bias": 0.0,
                "stale_epochs": stale,
                "success_streak": success_streak,
                **metrics,
            }
        )
        if epoch == 1 or epoch % 250 == 0:
            print(
                f"numerator epoch={epoch} wi={weight[0]:.15g} "
                f"wmin={weight[1]:.15g} rmse={metrics['normalized_target_rmse']:.3e} "
                f"maxrel={metrics['A_maximum_relative_error']:.3e}",
                flush=True,
            )
        if success_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_numerical_precision"
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    prediction = _predict(model, validation["inputs"])
    metrics = _continuous_metrics(validation, prediction)
    weight = model.weight.detach().cpu().numpy()[0]
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "training_event_count": len(training["target_y"]),
        "validation_event_count": len(validation["target_y"]),
        "training_state_count": len(np.unique(training["state_position"])),
        "validation_state_count": len(np.unique(validation["state_position"])),
        "learned_w_i": float(weight[0]),
        "learned_w_min": float(weight[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "initial_w_i": float(initial[0]),
        "initial_w_min": float(initial[1]),
        "continuous_regression_only": True,
        "teacher_arithmetic_order": "target=f_min-1e-4*abs(f_min); A=f_i-target",
        "normalized_rmse_gate": NORMALIZED_RMSE_GATE,
        "normalized_max_error_gate": NORMALIZED_MAX_ERROR_GATE,
        "A_max_relative_error_gate": A_MAX_RELATIVE_ERROR_GATE,
        **metrics,
    }
    summary["numerator_smoke_gate_passed"] = _numerator_gate(metrics)
    rows.append(summary)
    rows.extend(_bucket_rows(validation, prediction, "validation_smoke"))
    return model, rows, summary


def _frozen_linear_predict(checkpoint: dict[str, Any], inputs: np.ndarray) -> np.ndarray:
    input_scale = checkpoint["input_scale"].detach().cpu().numpy()
    target_scale = float(checkpoint["target_scale"].item())
    weight = checkpoint["state_dict"]["weight"].detach().cpu().numpy()[0]
    return ((inputs / input_scale) @ weight) * target_scale


def _locked_slope_edges(
    state: dict[str, Any],
    logmag_checkpoint: dict[str, Any],
    sign_checkpoint: dict[str, Any],
) -> np.ndarray:
    hull = np.asarray(state["teacher_hull"], dtype=np.int64)
    sizes = np.asarray(state["sizes"], dtype=np.float64)
    losses = np.asarray(state["losses"], dtype=np.float64)
    predicted: list[float] = []
    sign_scale = float(sign_checkpoint["training_scale_S"].item())
    theta = float(sign_checkpoint["learned_theta"].item())
    alpha = float(sign_checkpoint["learned_alpha"].item())
    for position in range(len(hull) - 1):
        left = int(hull[position])
        right = int(hull[position + 1])
        delta_d = float(sizes[right] - sizes[left])
        delta_f = float(losses[right] - losses[left])
        if not delta_d > 0.0 or delta_f == 0.0:
            raise RuntimeError("invalid locked neighboring-slope edge")
        log_input = np.asarray(
            [[math.log(abs(delta_f)), math.log(delta_d)]], dtype=np.float64
        )
        log_magnitude = float(
            _frozen_linear_predict(logmag_checkpoint, log_input)[0]
        )
        compressed = math.asinh(delta_f / sign_scale)
        sign = 1.0 if alpha * (compressed - theta) >= 0.0 else -1.0
        predicted.append(sign * math.exp(log_magnitude))
    return np.asarray(predicted, dtype=np.float64)


def _assemble_full_pipeline(
    model: nn.Linear,
    validation: dict[str, np.ndarray],
    validation_states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numerator_prediction = _predict(model, validation["inputs"])
    a_hat = validation["S"] * numerator_prediction
    if np.any(a_hat <= 0.0):
        raise RuntimeError("A_hat <= 0 reached assembly despite numerator gate")

    logquotient = torch.load(
        LOGQUOTIENT_PATH, map_location="cpu", weights_only=False
    )
    slope_logmag_checkpoint = torch.load(
        SLOPE_LOGMAG_PATH, map_location="cpu", weights_only=False
    )
    slope_sign_checkpoint = torch.load(
        SLOPE_SIGN_PATH, map_location="cpu", weights_only=False
    )
    log_inputs = np.column_stack((np.log(a_hat), np.log(validation["d_i"])))
    log_k_hat = _frozen_linear_predict(logquotient, log_inputs)
    k_required_hat = np.exp(log_k_hat)
    k_relative = np.abs(
        k_required_hat - validation["exact_K_required"]
    ) / validation["exact_K_required"]

    locked_prev = np.empty(len(k_required_hat), dtype=np.float64)
    locked_next = np.empty(len(k_required_hat), dtype=np.float64)
    for state_position, state in enumerate(validation_states):
        state_mask = validation["state_position"] == state_position
        positions = validation["hull_position"][state_mask]
        edges = _locked_slope_edges(
            state, slope_logmag_checkpoint, slope_sign_checkpoint
        )
        locked_prev[state_mask] = np.asarray(
            [-math.inf if position == 0 else edges[position - 1] for position in positions],
            dtype=np.float64,
        )
        locked_next[state_mask] = np.asarray(
            [math.inf if position == len(edges) else edges[position] for position in positions],
            dtype=np.float64,
        )

    learned_lower = np.maximum(np.maximum(locked_prev, k_required_hat), 0.0)
    learned_winner = np.asarray(
        [
            slope_logmag._active_source(float(previous), float(required))
            for previous, required in zip(locked_prev, k_required_hat)
        ],
        dtype=object,
    )
    learned_raw_order = learned_lower <= locked_next
    learned_feasible = learned_lower <= locked_next + turn.HULL_TOLERANCE
    learned_pass = (locked_next > 0.0) & learned_feasible
    winner_changes = learned_winner != validation["teacher_winner"]
    ordering_changes = learned_raw_order != validation["exact_raw_order"]
    feasibility_changes = learned_feasible != validation["exact_feasibility"]
    pass_changes = learned_pass != validation["exact_pass"]
    teacher_wins = validation["teacher_winner"] == "K_required"
    numerator_metrics = _continuous_metrics(validation, numerator_prediction)
    summary = {
        "record_type": "final_summary",
        "event_count": len(k_required_hat),
        "state_count": len(validation_states),
        **numerator_metrics,
        "K_required_median_relative_error": float(np.median(k_relative)),
        "K_required_relative_error_p90": float(np.percentile(k_relative, 90.0)),
        "K_required_relative_error_p95": float(np.percentile(k_relative, 95.0)),
        "K_required_relative_error_p99": float(np.percentile(k_relative, 99.0)),
        "K_required_maximum_relative_error": float(np.max(k_relative)),
        "teacher_K_required_winner_count": int(np.sum(teacher_wins)),
        "lower_bound_winner_changes": int(np.sum(winner_changes)),
        "K_lower_vs_K_upper_ordering_changes": int(np.sum(ordering_changes)),
        "feasibility_comparison_changes": int(np.sum(feasibility_changes)),
        "final_improvement_PASS_FAIL_changes": int(np.sum(pass_changes)),
        "teacher_winner_subset_lower_bound_winner_changes": int(
            np.sum(winner_changes[teacher_wins])
        ),
        "teacher_winner_subset_ordering_changes": int(
            np.sum(ordering_changes[teacher_wins])
        ),
        "teacher_winner_subset_feasibility_changes": int(
            np.sum(feasibility_changes[teacher_wins])
        ),
        "teacher_winner_subset_PASS_FAIL_changes": int(
            np.sum(pass_changes[teacher_wins])
        ),
        "teacher_winner_subset_K_required_maximum_relative_error": float(
            np.max(k_relative[teacher_wins])
        ),
        "frozen_logquotient_sha256": LOGQUOTIENT_SHA256,
        "locked_slope_logmag_sha256": SLOPE_LOGMAG_SHA256,
        "locked_slope_sign_sha256": SLOPE_SIGN_SHA256,
    }
    summary["full_pipeline_gate_passed"] = bool(
        _numerator_gate(numerator_metrics)
        and summary["lower_bound_winner_changes"] == 0
        and summary["K_lower_vs_K_upper_ordering_changes"] == 0
        and summary["feasibility_comparison_changes"] == 0
        and summary["final_improvement_PASS_FAIL_changes"] == 0
    )
    rows: list[dict[str, Any]] = [summary]
    rows.extend(_bucket_rows(validation, numerator_prediction, "full_held_out_pool"))
    rows.append(
        {
            "record_type": "teacher_K_required_winner_subset",
            "event_count": int(np.sum(teacher_wins)),
            "lower_bound_winner_changes": int(np.sum(winner_changes[teacher_wins])),
            "K_lower_vs_K_upper_ordering_changes": int(
                np.sum(ordering_changes[teacher_wins])
            ),
            "feasibility_comparison_changes": int(
                np.sum(feasibility_changes[teacher_wins])
            ),
            "final_improvement_PASS_FAIL_changes": int(
                np.sum(pass_changes[teacher_wins])
            ),
            "K_required_maximum_relative_error": float(
                np.max(k_relative[teacher_wins])
            ),
        }
    )
    return rows, summary


def _save_and_lock(
    model: nn.Linear,
    smoke_summary: dict[str, Any],
    pipeline_summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    weight = model.weight.detach().cpu().numpy()[0]
    checkpoint = {
        "format_version": 1,
        "component": "K_required_improvement_numerator",
        "status": "LOCKED_COMPONENT",
        "architecture": "A/S = w_i*(f_i/S) + w_min*(f_min/S); no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "effective_coefficients": torch.as_tensor(weight, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "teacher_arithmetic_order": "target=f_min-1e-4*abs(f_min); A=f_i-target",
        "scale_definition": "S=max(abs(f_i),abs(f_min))",
        "smoke_summary": smoke_summary,
        "full_pipeline_summary": pipeline_summary,
        "frozen_logquotient_sha256": LOGQUOTIENT_SHA256,
        "complete_K_required_operator_locked": True,
    }
    torch.save(checkpoint, NUMERATOR_PATH)
    restored = torch.load(NUMERATOR_PATH, map_location="cpu", weights_only=False)
    if restored["status"] != "LOCKED_COMPONENT" or float(restored["bias"].item()) != 0.0:
        raise RuntimeError("numerator checkpoint readback failed")
    numerator_hash = _sha256(NUMERATOR_PATH)
    NUMERATOR_HASH_PATH.write_text(
        f"{numerator_hash}  {NUMERATOR_PATH.name}\n", encoding="utf-8"
    )
    lock_rows = [
        {
            "component": "K_required_improvement_numerator",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(NUMERATOR_PATH),
            "checkpoint_sha256": numerator_hash,
            "learned_w_i": float(weight[0]),
            "learned_w_min": float(weight[1]),
            "bias": 0.0,
        },
        {
            "component": "K_required_log_quotient",
            "status": "FROZEN_CANDIDATE_COMPONENT_OF_LOCKED_OPERATOR",
            "checkpoint": str(LOGQUOTIENT_PATH),
            "checkpoint_sha256": LOGQUOTIENT_SHA256,
            "modified_or_retrained": False,
        },
        {
            "component": "K_required_operator",
            "status": "LOCKED",
            "assembled_formula": "learned_A -> frozen_logquotient(log(A),log(d_i)) -> exp",
            "A_hat_nonpositive_count": pipeline_summary["A_hat_nonpositive_count"],
            "lower_bound_winner_changes": pipeline_summary["lower_bound_winner_changes"],
            "K_lower_vs_K_upper_ordering_changes": pipeline_summary[
                "K_lower_vs_K_upper_ordering_changes"
            ],
            "feasibility_comparison_changes": pipeline_summary[
                "feasibility_comparison_changes"
            ],
            "final_improvement_PASS_FAIL_changes": pipeline_summary[
                "final_improvement_PASS_FAIL_changes"
            ],
        },
    ]
    return numerator_hash, lock_rows


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_frozen_hashes()
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
    train_states = turn._build_states(payload, train_indices, workers, "numerator train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "numerator validation"
    )
    training = _build_pool(train_states, "train")
    validation = _build_pool(validation_states, "validation")
    audit_rows, assumptions_passed = _audit_rows(training, validation)
    _write_csv(AUDIT_PATH, audit_rows)
    if not assumptions_passed:
        print("numerator pre-training assumptions failed; stopped", flush=True)
        return

    model, smoke_rows, smoke_summary = _train(training, validation)
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["numerator_smoke_gate_passed"]):
        print("numerator smoke gate failed; stopped", flush=True)
        return

    pipeline_rows, pipeline_summary = _assemble_full_pipeline(
        model, validation, validation_states
    )
    if not bool(pipeline_summary["full_pipeline_gate_passed"]):
        _write_csv(FULL_PATH, pipeline_rows)
        print("assembled K_required pipeline gate failed; stopped", flush=True)
        return

    numerator_hash, lock_rows = _save_and_lock(
        model, smoke_summary, pipeline_summary
    )
    pipeline_summary["numerator_checkpoint"] = str(NUMERATOR_PATH)
    pipeline_summary["numerator_checkpoint_sha256"] = numerator_hash
    pipeline_summary["assembled_K_required_operator_status"] = "LOCKED"
    _write_csv(FULL_PATH, pipeline_rows)
    _write_csv(LOCK_PATH, lock_rows)
    weight = model.weight.detach().cpu().numpy()[0]
    print(
        f"K_required operator LOCKED wi={weight[0]:.17g} "
        f"wmin={weight[1]:.17g} numerator_sha256={numerator_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
