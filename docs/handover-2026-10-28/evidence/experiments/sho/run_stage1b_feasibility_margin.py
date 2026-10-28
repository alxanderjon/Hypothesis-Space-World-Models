#!/usr/bin/env python3
"""Learn only the continuous normalized feasibility margin."""

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
import experiments.sho.run_stage1b_k_lower_margin as lower_margin
import experiments.sho.run_stage1b_k_lower_threshold as lower_threshold


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/feasibility_margin")
SMOKE_PATH = OUTPUT_DIR / "feasibility_margin_smoke.csv"
FULL_PATH = OUTPUT_DIR / "feasibility_margin_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "feasibility_margin_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

LOWER_MARGIN_PATH = lower_margin.CANDIDATE_PATH
LOWER_MARGIN_SHA256 = lower_threshold.MARGIN_SHA256
LOWER_THRESHOLD_PATH = lower_threshold.THRESHOLD_PATH
LOWER_THRESHOLD_SHA256 = (
    "61568109cee304f7ce3e8e96b587f68e978d2302275fada2894007a72844fd10"
)

TAU = turn.HULL_TOLERANCE
MODEL_SEED = smoke.MODEL_SEED + 7_071_067
MAX_EPOCHS = 30_000
PATIENCE = 2_500
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
RMSE_GATE = 1.0e-12
MAX_ERROR_GATE = 1.0e-10
NUMERICAL_RMSE_STOP = 1.0e-14
NUMERICAL_MAX_ERROR_STOP = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_hashes() -> None:
    lower_margin._verify_frozen_hashes()
    for path, expected, label in (
        (LOWER_MARGIN_PATH, LOWER_MARGIN_SHA256, "frozen K_lower margin"),
        (LOWER_THRESHOLD_PATH, LOWER_THRESHOLD_SHA256, "locked K_lower threshold"),
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


def _frozen_margin(pool: dict[str, np.ndarray]) -> np.ndarray:
    checkpoint = torch.load(
        LOWER_MARGIN_PATH, map_location="cpu", weights_only=False
    )
    return lower_margin._checkpoint_linear_predict(
        checkpoint, pool["margin_inputs"]
    )


def _threshold_parameters() -> tuple[float, float]:
    checkpoint = torch.load(
        LOWER_THRESHOLD_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("K_lower threshold is not locked")
    return (
        float(checkpoint["learned_theta"].item()),
        float(checkpoint["learned_alpha"].item()),
    )


def _build_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    upstream = lower_margin._locked_upstream_pool(states, split)
    frozen_margin = _frozen_margin(upstream)
    theta, alpha = _threshold_parameters()
    positive_mask = upstream["positive_mask"]
    automatic_mask = upstream["automatic_mask"]
    positive_indices = np.flatnonzero(positive_mask)
    learned_previous = alpha * (frozen_margin - theta) >= 0.0

    locked_required = np.exp(upstream["locked_log_required"])
    locked_lower = np.empty(len(upstream["A"]), dtype=np.float64)
    locked_lower[automatic_mask] = locked_required[automatic_mask]
    locked_lower[positive_indices] = np.where(
        learned_previous,
        np.exp(upstream["locked_log_prev"][positive_mask]),
        locked_required[positive_mask],
    )
    locked_upper = np.full(len(upstream["A"]), math.inf, dtype=np.float64)
    finite_upper = np.isfinite(upstream["exact_K_next"])
    locked_upper[finite_upper] = (
        upstream["locked_sign_next"][finite_upper]
        * np.exp(upstream["locked_log_next"][finite_upper])
    )
    endpoint_mask = ~finite_upper
    if np.any(locked_lower <= 0.0):
        raise RuntimeError("locked K_lower is not strictly positive")
    if np.any(~np.isfinite(locked_upper[finite_upper])):
        raise RuntimeError("finite K_upper event has non-finite locked value")

    scale = np.maximum.reduce(
        (
            np.abs(locked_upper[finite_upper]),
            np.abs(locked_lower[finite_upper]),
            np.full(int(np.sum(finite_upper)), TAU, dtype=np.float64),
        )
    )
    if np.any(scale <= 0.0) or np.any(~np.isfinite(scale)):
        raise RuntimeError("invalid per-event feasibility scale")
    upper = locked_upper[finite_upper]
    lower = locked_lower[finite_upper]
    # Preserve the exact audited float64 arithmetic order.
    raw_margin = (upper + TAU) - lower
    inputs = np.column_stack((upper / scale, lower / scale, TAU / scale))
    target = raw_margin / scale
    teacher_feasible = raw_margin >= 0.0
    endpoint_feasible = np.ones(int(np.sum(endpoint_mask)), dtype=bool)
    return {
        "inputs": inputs,
        "target": target,
        "scale": scale,
        "raw_margin": raw_margin,
        "locked_K_upper": upper,
        "locked_K_lower": lower,
        "teacher_feasible": teacher_feasible,
        "finite_event_indices": np.flatnonzero(finite_upper),
        "finite_state_position": upstream["state_position"][finite_upper],
        "finite_source_state_index": upstream["source_state_index"][finite_upper],
        "finite_hull_position": upstream["hull_position"][finite_upper],
        "endpoint_event_indices": np.flatnonzero(endpoint_mask),
        "endpoint_feasible": endpoint_feasible,
        "upstream": upstream,
        "all_locked_K_upper": locked_upper,
        "all_locked_K_lower": locked_lower,
        "split": np.asarray([split], dtype=object),
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
        "exact_zero_count": int(np.sum(values == 0.0)),
    }


def _audit_rows(pool: dict[str, np.ndarray], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "record_type": "special_case_summary",
            "split": split,
            "total_event_count": len(pool["upstream"]["A"]),
            "finite_regression_event_count": len(pool["target"]),
            "positive_infinity_K_upper_endpoint_count": len(
                pool["endpoint_event_indices"]
            ),
            "endpoint_automatically_feasible_count": int(
                np.sum(pool["endpoint_feasible"])
            ),
            "endpoint_included_in_regression": False,
            "tau": TAU,
            "scale_nonpositive_count": int(np.sum(pool["scale"] <= 0.0)),
            "epsilon_added": False,
        }
    ]
    for name, values in (
        ("locked_K_upper", pool["locked_K_upper"]),
        ("locked_K_lower", pool["locked_K_lower"]),
        ("raw_m_feas", pool["raw_margin"]),
        ("scale_S", pool["scale"]),
        ("normalized_target_y", pool["target"]),
    ):
        rows.append(
            {
                "record_type": "variable_distribution",
                "split": split,
                "variable": name,
                **_stats(values),
            }
        )
    return rows


def _predict(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    inputs: np.ndarray,
) -> np.ndarray:
    return slope_logmag._predict(model, scaling, inputs)


def _metrics(
    pool: dict[str, np.ndarray], prediction: np.ndarray
) -> dict[str, Any]:
    error = prediction - pool["target"]
    absolute = np.abs(error)
    raw_prediction = pool["scale"] * prediction
    raw_error = raw_prediction - pool["raw_margin"]
    raw_absolute = np.abs(raw_error)
    teacher = pool["teacher_feasible"]
    learned = prediction >= 0.0
    mistakes = teacher != learned
    nonzero_margin = pool["raw_margin"] != 0.0
    relative = np.abs(raw_error[nonzero_margin]) / np.abs(
        pool["raw_margin"][nonzero_margin]
    )
    return {
        "normalized_rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_mae": float(np.mean(absolute)),
        "normalized_maximum_absolute_error": float(np.max(absolute)),
        "normalized_correlation": float(
            np.corrcoef(pool["target"], prediction)[0, 1]
        ),
        "raw_margin_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "raw_margin_mae": float(np.mean(raw_absolute)),
        "raw_margin_maximum_absolute_error": float(np.max(raw_absolute)),
        "raw_margin_scaled_error_p50": float(np.percentile(absolute, 50.0)),
        "raw_margin_scaled_error_p90": float(np.percentile(absolute, 90.0)),
        "raw_margin_scaled_error_p99": float(np.percentile(absolute, 99.0)),
        "raw_margin_relative_error_p50_nonzero": float(
            np.percentile(relative, 50.0)
        ),
        "raw_margin_relative_error_p99_nonzero": float(
            np.percentile(relative, 99.0)
        ),
        "raw_margin_maximum_relative_error_nonzero": float(np.max(relative)),
        "feasibility_decision_agreement": float(np.mean(~mistakes)),
        "feasible_accuracy": float(
            np.mean(~mistakes[teacher]) if np.any(teacher) else 1.0
        ),
        "infeasible_accuracy": float(
            np.mean(~mistakes[~teacher]) if np.any(~teacher) else 1.0
        ),
        "feasibility_decision_mistakes": int(np.sum(mistakes)),
        "teacher_feasible_count": int(np.sum(teacher)),
        "teacher_infeasible_count": int(np.sum(~teacher)),
    }


def _magnitude_bucket(value: float) -> str:
    value = abs(value)
    if value < 1.0e-12:
        return "abs_y_lt_1e-12"
    if value < 1.0e-9:
        return "abs_y_1e-12_to_1e-9"
    if value < 1.0e-6:
        return "abs_y_1e-9_to_1e-6"
    if value < 1.0e-3:
        return "abs_y_1e-6_to_1e-3"
    if value < 1.0e-1:
        return "abs_y_1e-3_to_1e-1"
    if value < 1.0:
        return "abs_y_1e-1_to_1"
    return "abs_y_ge_1"


def _subset(pool: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    keys = (
        "inputs",
        "target",
        "scale",
        "raw_margin",
        "locked_K_upper",
        "locked_K_lower",
        "teacher_feasible",
        "finite_event_indices",
        "finite_state_position",
        "finite_source_state_index",
        "finite_hull_position",
    )
    return {key: pool[key][mask] for key in keys}


def _bucket_rows(
    pool: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    labels = np.asarray(
        [_magnitude_bucket(value) for value in pool["target"]], dtype=object
    )
    order = (
        "abs_y_lt_1e-12",
        "abs_y_1e-12_to_1e-9",
        "abs_y_1e-9_to_1e-6",
        "abs_y_1e-6_to_1e-3",
        "abs_y_1e-3_to_1e-1",
        "abs_y_1e-1_to_1",
        "abs_y_ge_1",
    )
    rows: list[dict[str, Any]] = []
    for bucket in order:
        mask = labels == bucket
        row: dict[str, Any] = {
            "record_type": "normalized_margin_bucket",
            "split": split,
            "bucket": bucket,
            "event_count": int(np.sum(mask)),
        }
        if np.any(mask):
            row.update(_metrics(_subset(pool, mask), prediction[mask]))
        rows.append(row)
    return rows


def _gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["normalized_rmse"] <= RMSE_GATE
        and metrics["normalized_maximum_absolute_error"] <= MAX_ERROR_GATE
        and metrics["feasibility_decision_mistakes"] == 0
    )


def _mistake_rows(
    pool: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    teacher = pool["teacher_feasible"]
    learned = prediction >= 0.0
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(teacher != learned):
        rows.append(
            {
                "record_type": "feasibility_mistake",
                "split": split,
                "event_index": int(pool["finite_event_indices"][local_index]),
                "source_state_index": int(
                    pool["finite_source_state_index"][local_index]
                ),
                "hull_position": int(pool["finite_hull_position"][local_index]),
                "locked_K_upper": float(pool["locked_K_upper"][local_index]),
                "locked_K_lower": float(pool["locked_K_lower"][local_index]),
                "scale_S": float(pool["scale"][local_index]),
                "exact_raw_margin": float(pool["raw_margin"][local_index]),
                "exact_normalized_margin": float(pool["target"][local_index]),
                "predicted_normalized_margin": float(prediction[local_index]),
            }
        )
    return rows


def _train(
    training: dict[str, np.ndarray], validation: dict[str, np.ndarray]
) -> tuple[nn.Linear, slope_logmag.HomogeneousScaling, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    scaling = slope_logmag.HomogeneousScaling(
        training["inputs"], training["target"]
    )
    model = nn.Linear(3, 1, bias=False).to(torch.float64)
    initial = scaling.coefficients(model).copy()
    if np.array_equal(initial, np.asarray([1.0, -1.0, 1.0])):
        raise RuntimeError("forbidden known coefficient initialization")
    train_x = torch.as_tensor(
        scaling.inputs(training["inputs"]), dtype=torch.float64
    )
    train_y = torch.as_tensor(
        scaling.target(training["target"]), dtype=torch.float64
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1.0e-15)
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    numerical_streak = 0
    stopping_reason = "maximum_epochs"
    stop_epoch = MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        output = model(train_x).squeeze(-1)
        loss = nn.functional.mse_loss(output, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        prediction = _predict(model, scaling, validation["inputs"])
        metrics = _metrics(validation, prediction)
        mse = metrics["normalized_rmse"] ** 2
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        numerical = bool(
            metrics["normalized_rmse"] <= NUMERICAL_RMSE_STOP
            and metrics["normalized_maximum_absolute_error"]
            <= NUMERICAL_MAX_ERROR_STOP
            and metrics["feasibility_decision_mistakes"] == 0
        )
        numerical_streak = numerical_streak + 1 if numerical else 0
        if epoch == 1 or epoch % 250 == 0:
            weights = scaling.coefficients(model)
            print(
                f"feas margin epoch={epoch} wu={weights[0]:.15g} "
                f"wl={weights[1]:.15g} wt={weights[2]:.15g} "
                f"rmse={metrics['normalized_rmse']:.3e} "
                f"mistakes={metrics['feasibility_decision_mistakes']}",
                flush=True,
            )
        if numerical_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_numerical_precision"
            stop_epoch = epoch
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            stop_epoch = epoch
            break
    model.load_state_dict(best_state)
    prediction = _predict(model, scaling, validation["inputs"])
    metrics = _metrics(validation, prediction)
    weights = scaling.coefficients(model)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "training_finite_event_count": len(training["target"]),
        "validation_finite_event_count": len(validation["target"]),
        "training_state_count": len(np.unique(training["finite_state_position"])),
        "validation_state_count": len(np.unique(validation["finite_state_position"])),
        "training_infinite_endpoint_count": len(training["endpoint_event_indices"]),
        "validation_infinite_endpoint_count": len(validation["endpoint_event_indices"]),
        "learned_w_upper": float(weights[0]),
        "learned_w_lower": float(weights[1]),
        "learned_w_tau": float(weights[2]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        "initial_w_upper": float(initial[0]),
        "initial_w_lower": float(initial[1]),
        "initial_w_tau": float(initial[2]),
        "continuous_regression_only": True,
        "PASS_FAIL_labels_used_for_training": False,
        "tau": TAU,
        **metrics,
    }
    summary["smoke_gate_passed"] = _gate(metrics)
    rows = [summary]
    rows.extend(_audit_rows(training, "training"))
    rows.extend(_audit_rows(validation, "validation"))
    rows.extend(_bucket_rows(validation, prediction, "validation_smoke"))
    rows.extend(_mistake_rows(validation, prediction, "validation_smoke"))
    return model, scaling, rows, summary


def _full_check(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    pool: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prediction = _predict(model, scaling, pool["inputs"])
    continuous = _metrics(pool, prediction)
    learned_finite_feasible = prediction >= 0.0
    learned_all_feasible = np.ones(len(pool["upstream"]["A"]), dtype=bool)
    learned_all_feasible[pool["finite_event_indices"]] = learned_finite_feasible
    locked_upper = pool["all_locked_K_upper"]
    learned_pass = (locked_upper > 0.0) & learned_all_feasible
    feasibility_changes = learned_all_feasible != pool["upstream"]["exact_feasibility"]
    pass_changes = learned_pass != pool["upstream"]["exact_pass"]
    weights = scaling.coefficients(model)
    summary = {
        "record_type": "final_summary",
        "held_out_total_event_count": len(pool["upstream"]["A"]),
        "held_out_finite_event_count": len(pool["target"]),
        "held_out_infinite_endpoint_count": len(pool["endpoint_event_indices"]),
        "infinite_endpoint_automatic_feasibility_errors": int(
            np.sum(~learned_all_feasible[pool["endpoint_event_indices"]])
        ),
        "learned_w_upper": float(weights[0]),
        "learned_w_lower": float(weights[1]),
        "learned_w_tau": float(weights[2]),
        "bias": 0.0,
        "bias_exactly_zero": True,
        **continuous,
        "feasibility_decision_changes_all_events": int(
            np.sum(feasibility_changes)
        ),
        "final_improvement_PASS_FAIL_changes": int(np.sum(pass_changes)),
        "locked_K_lower_margin_sha256": LOWER_MARGIN_SHA256,
        "locked_K_lower_threshold_sha256": LOWER_THRESHOLD_SHA256,
        "locked_slope_logmag_sha256": lower_margin.SLOPE_LOGMAG_SHA256,
        "locked_slope_sign_sha256": lower_margin.SLOPE_SIGN_SHA256,
        "locked_K_required_numerator_sha256": lower_margin.NUMERATOR_SHA256,
        "locked_K_required_logquotient_sha256": lower_margin.LOGQUOTIENT_SHA256,
    }
    summary["full_pool_gate_passed"] = bool(
        _gate(continuous)
        and summary["infinite_endpoint_automatic_feasibility_errors"] == 0
        and summary["feasibility_decision_changes_all_events"] == 0
        and summary["final_improvement_PASS_FAIL_changes"] == 0
    )
    rows = [summary]
    rows.extend(_bucket_rows(pool, prediction, "full_held_out_pool"))
    rows.extend(_mistake_rows(pool, prediction, "full_held_out_pool"))
    return rows, summary


def _save_candidate(
    model: nn.Linear,
    scaling: slope_logmag.HomogeneousScaling,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
) -> str:
    weights = scaling.coefficients(model)
    checkpoint = {
        "format_version": 1,
        "component": "feasibility_margin",
        "status": "FROZEN_CANDIDATE",
        "architecture": "y_hat=w_upper*x_upper+w_lower*x_lower+w_tau*x_tau; no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_scale": torch.as_tensor(scaling.input_scale, dtype=torch.float64),
        "target_scale": torch.tensor(scaling.target_scale, dtype=torch.float64),
        "effective_coefficients": torch.as_tensor(weights, dtype=torch.float64),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "tau": torch.tensor(TAU, dtype=torch.float64),
        "scale_definition": "S=max(abs(K_upper),abs(K_lower),tau)",
        "infinite_K_upper_endpoint_handling": "excluded_from_regression_and_automatically_feasible",
        "continuous_regression_only": True,
        "final_feasibility_threshold_learned": False,
        "locked_K_lower_margin_sha256": LOWER_MARGIN_SHA256,
        "locked_K_lower_threshold_sha256": LOWER_THRESHOLD_SHA256,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("feasibility-margin candidate readback failed")
    if float(restored["bias"].item()) != 0.0:
        raise RuntimeError("feasibility-margin candidate bias is not zero")
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
        payload, train_indices, workers, "feasibility margin train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "feasibility margin validation"
    )
    training = _build_pool(train_states, "train")
    validation = _build_pool(validation_states, "validation")
    model, scaling, smoke_rows, smoke_summary = _train(training, validation)
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        print("feasibility-margin smoke gate failed; stopped", flush=True)
        return

    full_rows, full_summary = _full_check(model, scaling, validation)
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("feasibility-margin full-pool gate failed; stopped", flush=True)
        return

    checkpoint_hash = _save_candidate(model, scaling, smoke_summary, full_summary)
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["candidate_status"] = "FROZEN_CANDIDATE"
    full_summary["complete_feasibility_operator_locked"] = False
    _write_csv(FULL_PATH, full_rows)
    weights = scaling.coefficients(model)
    print(
        f"feasibility margin FROZEN_CANDIDATE "
        f"wu={weights[0]:.17g} wl={weights[1]:.17g} wt={weights[2]:.17g} "
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
