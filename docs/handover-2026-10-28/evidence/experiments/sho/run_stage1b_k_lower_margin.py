#!/usr/bin/env python3
"""Learn only the continuous log comparison margin for K_lower."""

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
import experiments.sho.run_stage1b_hull_slope_logmag as slope_logmag
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_k_required_numerator as numerator


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/k_lower_margin")
AUDIT_PATH = OUTPUT_DIR / "k_lower_margin_audit.csv"
SMOKE_PATH = OUTPUT_DIR / "k_lower_margin_smoke.csv"
FULL_PATH = OUTPUT_DIR / "k_lower_margin_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "k_lower_margin_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

NUMERATOR_PATH = Path(
    "outputs/sho/stage1B/k_required_numerator/k_required_numerator_locked.pt"
)
NUMERATOR_SHA256 = (
    "590d4d3e7f547e4113d022ba94b9a8f5a871818046a55b313892fc7b9485684f"
)
LOGQUOTIENT_PATH = numerator.LOGQUOTIENT_PATH
LOGQUOTIENT_SHA256 = numerator.LOGQUOTIENT_SHA256
SLOPE_LOGMAG_PATH = numerator.SLOPE_LOGMAG_PATH
SLOPE_LOGMAG_SHA256 = numerator.SLOPE_LOGMAG_SHA256
SLOPE_SIGN_PATH = numerator.SLOPE_SIGN_PATH
SLOPE_SIGN_SHA256 = numerator.SLOPE_SIGN_SHA256

MODEL_SEED = smoke.MODEL_SEED + 5_656_854
MAX_EPOCHS = 30_000
PATIENCE = 2_500
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
RMSE_GATE = 1.0e-11
MAX_ERROR_GATE = 1.0e-9
NUMERICAL_RMSE_STOP = 1.0e-14
NUMERICAL_MAX_ERROR_STOP = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_hashes() -> None:
    for path, expected, label in (
        (NUMERATOR_PATH, NUMERATOR_SHA256, "locked K_required numerator"),
        (LOGQUOTIENT_PATH, LOGQUOTIENT_SHA256, "frozen K_required log quotient"),
        (SLOPE_LOGMAG_PATH, SLOPE_LOGMAG_SHA256, "locked slope log magnitude"),
        (SLOPE_SIGN_PATH, SLOPE_SIGN_SHA256, "locked slope sign"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing {label}: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"{label} hash mismatch: {actual} != {expected}")


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


def _checkpoint_linear_predict(
    checkpoint: dict[str, Any], inputs: np.ndarray
) -> np.ndarray:
    input_scale = checkpoint["input_scale"].detach().cpu().numpy()
    target_scale = float(checkpoint["target_scale"].item())
    weight = checkpoint["state_dict"]["weight"].detach().cpu().numpy()[0]
    return ((inputs / input_scale) @ weight) * target_scale


def _locked_edge_outputs(
    state: dict[str, Any],
    logmag_checkpoint: dict[str, Any],
    sign_checkpoint: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    hull = np.asarray(state["teacher_hull"], dtype=np.int64)
    sizes = np.asarray(state["sizes"], dtype=np.float64)
    losses = np.asarray(state["losses"], dtype=np.float64)
    inputs: list[tuple[float, float]] = []
    delta_f_values: list[float] = []
    for position in range(len(hull) - 1):
        left = int(hull[position])
        right = int(hull[position + 1])
        delta_d = float(sizes[right] - sizes[left])
        delta_f = float(losses[right] - losses[left])
        if not delta_d > 0.0 or delta_f == 0.0:
            raise RuntimeError("invalid finite edge for locked slope operator")
        inputs.append((math.log(abs(delta_f)), math.log(delta_d)))
        delta_f_values.append(delta_f)
    log_magnitudes = _checkpoint_linear_predict(
        logmag_checkpoint, np.asarray(inputs, dtype=np.float64)
    )
    scale = float(sign_checkpoint["training_scale_S"].item())
    theta = float(sign_checkpoint["learned_theta"].item())
    alpha = float(sign_checkpoint["learned_alpha"].item())
    compressed = np.arcsinh(np.asarray(delta_f_values, dtype=np.float64) / scale)
    signs = np.where(alpha * (compressed - theta) >= 0.0, 1.0, -1.0)
    return log_magnitudes, signs


def _locked_upstream_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    base = numerator._build_pool(states, split)
    if len(base["violations"]):
        raise RuntimeError(f"upstream base-pool violations in {split}")

    numerator_checkpoint = torch.load(
        NUMERATOR_PATH, map_location="cpu", weights_only=False
    )
    quotient_checkpoint = torch.load(
        LOGQUOTIENT_PATH, map_location="cpu", weights_only=False
    )
    slope_logmag_checkpoint = torch.load(
        SLOPE_LOGMAG_PATH, map_location="cpu", weights_only=False
    )
    slope_sign_checkpoint = torch.load(
        SLOPE_SIGN_PATH, map_location="cpu", weights_only=False
    )

    numerator_weight = (
        numerator_checkpoint["state_dict"]["weight"].detach().cpu().numpy()[0]
    )
    learned_normalized_a = base["inputs"] @ numerator_weight
    learned_a = base["S"] * learned_normalized_a
    if np.any(learned_a <= 0.0):
        raise RuntimeError("locked K_required numerator produced A <= 0")
    required_inputs = np.column_stack((np.log(learned_a), np.log(base["d_i"])))
    locked_log_required = _checkpoint_linear_predict(
        quotient_checkpoint, required_inputs
    )
    if not np.all(np.isfinite(locked_log_required)):
        raise RuntimeError("non-finite frozen K_required log output")

    locked_log_prev = np.full(len(base["A"]), np.nan, dtype=np.float64)
    locked_sign_prev = np.zeros(len(base["A"]), dtype=np.float64)
    locked_log_next = np.full(len(base["A"]), np.nan, dtype=np.float64)
    locked_sign_next = np.zeros(len(base["A"]), dtype=np.float64)
    for state_position, state in enumerate(states):
        mask = base["state_position"] == state_position
        positions = base["hull_position"][mask]
        edge_log, edge_sign = _locked_edge_outputs(
            state, slope_logmag_checkpoint, slope_sign_checkpoint
        )
        for output_index, position_value in zip(np.flatnonzero(mask), positions):
            position = int(position_value)
            if position > 0:
                locked_log_prev[output_index] = edge_log[position - 1]
                locked_sign_prev[output_index] = edge_sign[position - 1]
            if position < len(edge_log):
                locked_log_next[output_index] = edge_log[position]
                locked_sign_next[output_index] = edge_sign[position]

    exact_prev = base["exact_K_prev"]
    finite_prev = np.isfinite(exact_prev)
    positive_mask = finite_prev & (exact_prev > 0.0)
    automatic_mask = ~positive_mask
    if np.any(locked_sign_prev[positive_mask] != 1.0):
        raise RuntimeError("locked slope sign disagrees on positive K_prev event")
    finite_nonpositive = finite_prev & (exact_prev <= 0.0)
    if np.any(locked_sign_prev[finite_nonpositive] != -1.0):
        raise RuntimeError("locked slope sign disagrees on nonpositive K_prev event")
    if np.any(base["exact_K_required"] <= 0.0):
        raise RuntimeError("K_required is not strictly positive")

    exact_margin = np.full(len(base["A"]), np.nan, dtype=np.float64)
    exact_margin[positive_mask] = (
        np.log(exact_prev[positive_mask])
        - np.log(base["exact_K_required"][positive_mask])
    )
    margin_inputs = np.column_stack(
        (locked_log_prev[positive_mask], locked_log_required[positive_mask])
    )
    teacher_positive_winner = exact_margin[positive_mask] >= 0.0
    automatic_teacher_required = (
        base["teacher_winner"][automatic_mask] == "K_required"
    )
    return {
        **base,
        "locked_log_required": locked_log_required,
        "locked_log_prev": locked_log_prev,
        "locked_sign_prev": locked_sign_prev,
        "locked_log_next": locked_log_next,
        "locked_sign_next": locked_sign_next,
        "positive_mask": positive_mask,
        "automatic_mask": automatic_mask,
        "finite_nonpositive_mask": finite_nonpositive,
        "endpoint_mask": ~finite_prev,
        "exact_margin": exact_margin,
        "margin_inputs": margin_inputs,
        "margin_target": exact_margin[positive_mask],
        "teacher_positive_winner": teacher_positive_winner,
        "automatic_teacher_required": automatic_teacher_required,
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
    }


def _audit_rows(pool: dict[str, np.ndarray], split: str) -> list[dict[str, Any]]:
    margin = pool["margin_target"]
    positive = margin > 0.0
    negative = margin < 0.0
    zeros = margin == 0.0
    rows: list[dict[str, Any]] = [
        {
            "record_type": "margin_audit_summary",
            "split": split,
            "total_improvement_events": len(pool["A"]),
            "positive_K_prev_events": len(margin),
            "K_prev_winner_events": int(np.sum(margin >= 0.0)),
            "K_required_winner_events_with_positive_K_prev": int(np.sum(negative)),
            "exact_zero_margin_events": int(np.sum(zeros)),
            "closest_positive_margin_to_zero": (
                float(np.min(margin[positive])) if np.any(positive) else ""
            ),
            "closest_negative_margin_to_zero": (
                float(np.max(margin[negative])) if np.any(negative) else ""
            ),
            "automatic_K_prev_nonpositive_events": int(
                np.sum(pool["automatic_mask"])
            ),
            "finite_K_prev_nonpositive_events": int(
                np.sum(pool["finite_nonpositive_mask"])
            ),
            "negative_infinity_endpoint_events": int(np.sum(pool["endpoint_mask"])),
            "automatic_route_teacher_K_required_correct": bool(
                np.all(pool["automatic_teacher_required"])
            ),
            **_stats(margin),
        }
    ]
    for percentile in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
        rows.append(
            {
                "record_type": "margin_percentile",
                "split": split,
                "percentile": percentile,
                "m_teacher": float(np.percentile(margin, percentile)),
                "absolute_m_teacher": float(
                    np.percentile(np.abs(margin), percentile)
                ),
            }
        )
    return rows


def _predict(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    inputs: np.ndarray,
) -> np.ndarray:
    return slope_logmag._predict(model, scaling, inputs)


def _continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "maximum_absolute_margin_error": float(np.max(absolute)),
        "correlation": float(np.corrcoef(target, prediction)[0, 1]),
        "error_p50": float(np.percentile(absolute, 50.0)),
        "error_p90": float(np.percentile(absolute, 90.0)),
        "error_p95": float(np.percentile(absolute, 95.0)),
        "error_p99": float(np.percentile(absolute, 99.0)),
    }


def _winner_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    teacher_prev = target >= 0.0
    learned_prev = prediction >= 0.0
    changes = teacher_prev != learned_prev
    previous_mask = teacher_prev
    required_mask = ~teacher_prev
    return {
        "winner_agreement": float(np.mean(~changes)),
        "K_prev_winner_agreement": float(
            np.mean(~changes[previous_mask]) if np.any(previous_mask) else 1.0
        ),
        "K_required_winner_agreement": float(
            np.mean(~changes[required_mask]) if np.any(required_mask) else 1.0
        ),
        "winner_mistakes": int(np.sum(changes)),
    }


def _margin_bucket(value: float) -> str:
    value = abs(value)
    if value < 1.0e-12:
        return "abs_margin_lt_1e-12"
    if value < 1.0e-9:
        return "abs_margin_1e-12_to_1e-9"
    if value < 1.0e-6:
        return "abs_margin_1e-9_to_1e-6"
    if value < 1.0e-3:
        return "abs_margin_1e-6_to_1e-3"
    if value < 1.0:
        return "abs_margin_1e-3_to_1"
    if value < 10.0:
        return "abs_margin_1_to_10"
    return "abs_margin_ge_10"


def _bucket_rows(
    target: np.ndarray, prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    labels = np.asarray([_margin_bucket(value) for value in target], dtype=object)
    order = (
        "abs_margin_lt_1e-12",
        "abs_margin_1e-12_to_1e-9",
        "abs_margin_1e-9_to_1e-6",
        "abs_margin_1e-6_to_1e-3",
        "abs_margin_1e-3_to_1",
        "abs_margin_1_to_10",
        "abs_margin_ge_10",
    )
    rows: list[dict[str, Any]] = []
    for bucket in order:
        mask = labels == bucket
        row: dict[str, Any] = {
            "record_type": "absolute_margin_bucket",
            "split": split,
            "bucket": bucket,
            "event_count": int(np.sum(mask)),
        }
        if np.any(mask):
            row.update(_continuous_metrics(target[mask], prediction[mask]))
            row.update(_winner_metrics(target[mask], prediction[mask]))
        rows.append(row)
    return rows


def _smoke_gate(continuous: dict[str, Any], winners: dict[str, Any]) -> bool:
    return bool(
        continuous["rmse"] <= RMSE_GATE
        and continuous["maximum_absolute_margin_error"] <= MAX_ERROR_GATE
        and winners["winner_mistakes"] == 0
    )


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[nn.Linear, slope_logmag.HomogeneousScaling, list[dict[str, Any]], dict[str, Any]]:
    train_x = training["margin_inputs"]
    train_y = training["margin_target"]
    valid_x = validation["margin_inputs"]
    valid_y = validation["margin_target"]
    torch.manual_seed(MODEL_SEED)
    scaling = slope_logmag.HomogeneousScaling(train_x, train_y)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial = scaling.coefficients(model).copy()
    if np.array_equal(initial, np.asarray([1.0, -1.0], dtype=np.float64)):
        raise RuntimeError("forbidden exact coefficient initialization")
    x_tensor = torch.as_tensor(scaling.inputs(train_x), dtype=torch.float64)
    y_tensor = torch.as_tensor(scaling.target(train_y), dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1.0e-15)
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    success_streak = 0
    stopping_reason = "maximum_epochs"
    stop_epoch = MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        output = model(x_tensor).squeeze(-1)
        loss = nn.functional.mse_loss(output, y_tensor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prediction = _predict(model, scaling, valid_x)
        continuous = _continuous_metrics(valid_y, prediction)
        winners = _winner_metrics(valid_y, prediction)
        mse = continuous["rmse"] ** 2
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        # Passing the downstream gate is not a convergence condition.  Keep
        # optimizing until either the continuous fit itself reaches a much
        # stricter numerical floor or validation genuinely plateaus.
        success = bool(
            continuous["rmse"] <= NUMERICAL_RMSE_STOP
            and continuous["maximum_absolute_margin_error"]
            <= NUMERICAL_MAX_ERROR_STOP
            and winners["winner_mistakes"] == 0
        )
        success_streak = success_streak + 1 if success else 0
        if epoch == 1 or epoch % 250 == 0:
            coefficients = scaling.coefficients(model)
            print(
                f"Klower margin epoch={epoch} wp={coefficients[0]:.15g} "
                f"wr={coefficients[1]:.15g} rmse={continuous['rmse']:.3e} "
                f"max={continuous['maximum_absolute_margin_error']:.3e} "
                f"mistakes={winners['winner_mistakes']}",
                flush=True,
            )
        if success_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_numerical_precision"
            stop_epoch = epoch
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            stop_epoch = epoch
            break

    model.load_state_dict(best_state)
    prediction = _predict(model, scaling, valid_x)
    continuous = _continuous_metrics(valid_y, prediction)
    winners = _winner_metrics(valid_y, prediction)
    coefficients = scaling.coefficients(model)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "training_positive_K_prev_events": len(train_y),
        "validation_positive_K_prev_events": len(valid_y),
        "training_state_count": len(np.unique(training["state_position"])),
        "validation_state_count": len(np.unique(validation["state_position"])),
        "learned_w_prev": float(coefficients[0]),
        "learned_w_req": float(coefficients[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "initial_w_prev": float(initial[0]),
        "initial_w_req": float(initial[1]),
        "continuous_regression_only": True,
        "winner_labels_used_for_training": False,
        "rmse_gate": RMSE_GATE,
        "maximum_error_gate": MAX_ERROR_GATE,
        "numerical_rmse_stop": NUMERICAL_RMSE_STOP,
        "numerical_maximum_error_stop": NUMERICAL_MAX_ERROR_STOP,
        **continuous,
        **winners,
    }
    summary["smoke_gate_passed"] = _smoke_gate(continuous, winners)
    rows = [summary]
    rows.extend(_bucket_rows(valid_y, prediction, "validation_smoke"))
    mistakes = (valid_y >= 0.0) != (prediction >= 0.0)
    for event_index in np.flatnonzero(mistakes):
        rows.append(
            {
                "record_type": "winner_mistake",
                "event_index": int(event_index),
                "exact_teacher_margin": float(valid_y[event_index]),
                "learned_margin": float(prediction[event_index]),
                "absolute_margin_error": float(
                    abs(prediction[event_index] - valid_y[event_index])
                ),
            }
        )
    return model, scaling, rows, summary


def _full_downstream(
    pool: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    positive_mask = pool["positive_mask"]
    automatic_mask = pool["automatic_mask"]
    learned_prev_for_positive = prediction >= 0.0
    learned_winner = np.full(len(pool["A"]), "K_required", dtype=object)
    learned_winner[positive_mask] = np.where(
        learned_prev_for_positive, "K_prev", "K_required"
    )

    learned_lower = np.empty(len(pool["A"]), dtype=np.float64)
    learned_lower[automatic_mask] = np.exp(
        pool["locked_log_required"][automatic_mask]
    )
    positive_indices = np.flatnonzero(positive_mask)
    learned_lower[positive_indices] = np.where(
        learned_prev_for_positive,
        np.exp(pool["locked_log_prev"][positive_mask]),
        np.exp(pool["locked_log_required"][positive_mask]),
    )

    learned_next = np.full(len(pool["A"]), math.inf, dtype=np.float64)
    finite_next = np.isfinite(pool["exact_K_next"])
    learned_next[finite_next] = (
        pool["locked_sign_next"][finite_next]
        * np.exp(pool["locked_log_next"][finite_next])
    )
    learned_raw_order = learned_lower <= learned_next
    learned_feasible = learned_lower <= learned_next + turn.HULL_TOLERANCE
    learned_pass = (learned_next > 0.0) & learned_feasible
    winner_changes = learned_winner != pool["teacher_winner"]
    ordering_changes = learned_raw_order != pool["exact_raw_order"]
    feasibility_changes = learned_feasible != pool["exact_feasibility"]
    pass_changes = learned_pass != pool["exact_pass"]
    return {
        "automatic_routing_event_count": int(np.sum(automatic_mask)),
        "automatic_routing_errors": int(
            np.sum(learned_winner[automatic_mask] != "K_required")
        ),
        "K_lower_winner_changes": int(np.sum(winner_changes)),
        "K_lower_K_upper_ordering_changes": int(np.sum(ordering_changes)),
        "feasibility_changes": int(np.sum(feasibility_changes)),
        "final_improvement_PASS_FAIL_changes": int(np.sum(pass_changes)),
    }


def _full_rows(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    pool: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction = _predict(model, scaling, pool["margin_inputs"])
    continuous = _continuous_metrics(pool["margin_target"], prediction)
    winners = _winner_metrics(pool["margin_target"], prediction)
    downstream = _full_downstream(pool, prediction)
    coefficients = scaling.coefficients(model)
    summary = {
        "record_type": "final_summary",
        "held_out_total_event_count": len(pool["A"]),
        "held_out_state_count": len(np.unique(pool["state_position"])),
        "positive_K_prev_event_count": len(pool["margin_target"]),
        "automatically_routed_K_prev_nonpositive_event_count": int(
            np.sum(pool["automatic_mask"])
        ),
        "learned_w_prev": float(coefficients[0]),
        "learned_w_req": float(coefficients[1]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        **continuous,
        **winners,
        **downstream,
        "locked_numerator_sha256": NUMERATOR_SHA256,
        "frozen_logquotient_sha256": LOGQUOTIENT_SHA256,
        "locked_slope_logmag_sha256": SLOPE_LOGMAG_SHA256,
        "locked_slope_sign_sha256": SLOPE_SIGN_SHA256,
    }
    summary["full_pool_gate_passed"] = bool(
        _smoke_gate(continuous, winners)
        and downstream["automatic_routing_errors"] == 0
        and downstream["K_lower_winner_changes"] == 0
        and downstream["K_lower_K_upper_ordering_changes"] == 0
        and downstream["feasibility_changes"] == 0
        and downstream["final_improvement_PASS_FAIL_changes"] == 0
    )
    rows = [summary]
    rows.extend(
        _bucket_rows(pool["margin_target"], prediction, "full_held_out_pool")
    )
    mistakes = (pool["margin_target"] >= 0.0) != (prediction >= 0.0)
    positive_indices = np.flatnonzero(pool["positive_mask"])
    for local_index in np.flatnonzero(mistakes):
        event_index = int(positive_indices[local_index])
        rows.append(
            {
                "record_type": "winner_mistake",
                "event_index": event_index,
                "source_state_index": int(pool["source_state_index"][event_index]),
                "hull_position": int(pool["hull_position"][event_index]),
                "exact_teacher_margin": float(pool["margin_target"][local_index]),
                "learned_margin": float(prediction[local_index]),
            }
        )
    return rows, summary


def _save_candidate(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
) -> str:
    coefficients = scaling.coefficients(model)
    checkpoint = {
        "format_version": 1,
        "component": "K_lower_comparison_margin",
        "status": "FROZEN_CANDIDATE",
        "architecture": "m_hat=w_prev*L_prev+w_req*L_req; no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_scale": torch.as_tensor(scaling.input_scale, dtype=torch.float64),
        "target_scale": torch.tensor(scaling.target_scale, dtype=torch.float64),
        "effective_coefficients": torch.as_tensor(coefficients, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "continuous_regression_only": True,
        "final_winner_threshold_learned": False,
        "locked_numerator_sha256": NUMERATOR_SHA256,
        "frozen_logquotient_sha256": LOGQUOTIENT_SHA256,
        "locked_slope_logmag_sha256": SLOPE_LOGMAG_SHA256,
        "locked_slope_sign_sha256": SLOPE_SIGN_SHA256,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored["status"] != "FROZEN_CANDIDATE":
        raise RuntimeError("candidate checkpoint readback failed")
    if float(restored["bias"].item()) != 0.0:
        raise RuntimeError("candidate checkpoint bias is not exactly zero")
    checkpoint_hash = _sha256(CANDIDATE_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
    )
    return checkpoint_hash


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
    train_states = turn._build_states(
        payload, train_indices, workers, "Klower margin train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "Klower margin validation"
    )
    training = _locked_upstream_pool(train_states, "train")
    validation = _locked_upstream_pool(validation_states, "validation")
    audit_rows = _audit_rows(training, "training")
    audit_rows.extend(_audit_rows(validation, "full_held_out"))
    _write_csv(AUDIT_PATH, audit_rows)

    model, scaling, smoke_rows, smoke_summary = _train(training, validation)
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        print("K_lower comparison-margin smoke gate failed; stopped", flush=True)
        return

    full_rows, full_summary = _full_rows(model, scaling, validation)
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("K_lower comparison-margin full-pool gate failed; stopped", flush=True)
        return

    checkpoint_hash = _save_candidate(
        model, scaling, smoke_summary, full_summary
    )
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["candidate_status"] = "FROZEN_CANDIDATE"
    full_summary["complete_max_operator_locked"] = False
    _write_csv(FULL_PATH, full_rows)
    coefficients = scaling.coefficients(model)
    print(
        f"K_lower comparison margin FROZEN_CANDIDATE "
        f"w_prev={coefficients[0]:.17g} w_req={coefficients[1]:.17g} "
        f"sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
