#!/usr/bin/env python3
"""Learn only the finite neighboring-hull slope sign and validate assembly."""

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
import experiments.sho.run_stage1b_hull_slope_logmag as logmag
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_slope_sign")
SMOKE_PATH = OUTPUT_DIR / "hull_slope_sign_smoke.csv"
FULL_PATH = OUTPUT_DIR / "hull_slope_sign_full_pool.csv"
ASSEMBLED_PATH = OUTPUT_DIR / "hull_slope_assembled_validation.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "slope_operator_lock_summary.csv"

SOURCE_LOGMAG_PATH = Path(
    "outputs/sho/stage1B/hull_slope_logmag/hull_slope_logmag_candidate.pt"
)
SOURCE_LOGMAG_SHA256 = (
    "a4e1f1dbedf5f42121c441b6ced7b0f243fd78aed28b7e8c9c838178f675b249"
)
FROZEN_LOGMAG_PATH = OUTPUT_DIR / "slope_log_magnitude_frozen_candidate.pt"
FROZEN_LOGMAG_HASH_PATH = FROZEN_LOGMAG_PATH.with_suffix(".sha256")
SIGN_CHECKPOINT_PATH = OUTPUT_DIR / "hull_slope_sign_locked.pt"
SIGN_HASH_PATH = SIGN_CHECKPOINT_PATH.with_suffix(".sha256")

MODEL_SEED = smoke.MODEL_SEED + 2_236_067
MAX_EPOCHS = 20_000
PATIENCE = 2_000
EXACT_STABILITY_EPOCHS = 300
LEARNING_RATE = 1.0e-2


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


def _verify_and_freeze_logmag() -> tuple[dict[str, Any], str]:
    if not SOURCE_LOGMAG_PATH.exists():
        raise FileNotFoundError(f"missing source log-magnitude checkpoint: {SOURCE_LOGMAG_PATH}")
    source_hash = _sha256(SOURCE_LOGMAG_PATH)
    if source_hash != SOURCE_LOGMAG_SHA256:
        raise RuntimeError(
            f"source log-magnitude hash mismatch: {source_hash} != {SOURCE_LOGMAG_SHA256}"
        )
    checkpoint = torch.load(
        SOURCE_LOGMAG_PATH, map_location="cpu", weights_only=False
    )
    coefficients = checkpoint["effective_coefficients"].detach().cpu().numpy()
    expected = np.asarray(
        [0.9999999999999998, -1.000000000000002], dtype=np.float64
    )
    if not np.array_equal(coefficients, expected):
        raise RuntimeError(f"unexpected log-magnitude coefficients: {coefficients}")
    if float(checkpoint["bias"].item()) != 0.0:
        raise RuntimeError("log-magnitude bias is not exactly zero")

    frozen = copy.deepcopy(checkpoint)
    frozen["status"] = "FROZEN_CANDIDATE"
    frozen["source_checkpoint"] = str(SOURCE_LOGMAG_PATH)
    frozen["source_checkpoint_sha256"] = source_hash
    frozen["training_performed_during_freeze"] = False
    frozen["never_retrain_without_explicit_request"] = True
    torch.save(frozen, FROZEN_LOGMAG_PATH)
    frozen_hash = _sha256(FROZEN_LOGMAG_PATH)
    FROZEN_LOGMAG_HASH_PATH.write_text(
        f"{frozen_hash}  {FROZEN_LOGMAG_PATH.name}\n", encoding="utf-8"
    )
    restored = torch.load(
        FROZEN_LOGMAG_PATH, map_location="cpu", weights_only=False
    )
    if restored["status"] != "FROZEN_CANDIDATE":
        raise RuntimeError("frozen-candidate readback failed")
    if restored["source_checkpoint_sha256"] != SOURCE_LOGMAG_SHA256:
        raise RuntimeError("frozen-candidate source provenance failed")
    return restored, frozen_hash


def _build_edge_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    delta_d: list[float] = []
    delta_f: list[float] = []
    exact_k: list[float] = []
    state_position: list[int] = []
    source_state_index: list[int] = []
    edge_position: list[int] = []
    for local_state, state in enumerate(states):
        edges = logmag._state_edges(state)
        for position, (dd, df, value) in enumerate(
            zip(edges["delta_d"], edges["delta_f"], edges["K"])
        ):
            if not float(dd) > 0.0:
                raise RuntimeError("delta_d <= 0 in finite edge pool")
            if float(df) == 0.0:
                raise RuntimeError("delta_f == 0 in finite edge pool")
            delta_d.append(float(dd))
            delta_f.append(float(df))
            exact_k.append(float(value))
            state_position.append(local_state)
            source_state_index.append(int(state["source_state_index"]))
            edge_position.append(position)
    return {
        "delta_d": np.asarray(delta_d, dtype=np.float64),
        "delta_f": np.asarray(delta_f, dtype=np.float64),
        "exact_K": np.asarray(exact_k, dtype=np.float64),
        "state_position": np.asarray(state_position, dtype=np.int64),
        "source_state_index": np.asarray(source_state_index, dtype=np.int64),
        "edge_position": np.asarray(edge_position, dtype=np.int64),
        "split": np.asarray([split], dtype=object),
    }


def _balanced_indices(pool: dict[str, np.ndarray], seed: int) -> np.ndarray:
    positive = np.flatnonzero(pool["delta_f"] > 0.0)
    negative = np.flatnonzero(pool["delta_f"] < 0.0)
    count = min(len(positive), len(negative))
    generator = np.random.default_rng(seed)
    # Keep the balanced screen boundary-complete: the real positive edge
    # closest to zero is always included, while the remainder is sampled.
    closest_positive = positive[np.argmin(np.abs(pool["delta_f"][positive]))]
    positive_remainder = positive[positive != closest_positive]
    if count == 1:
        selected_positive = np.asarray([closest_positive], dtype=np.int64)
    else:
        selected_positive = np.concatenate(
            (
                np.asarray([closest_positive], dtype=np.int64),
                generator.choice(positive_remainder, size=count - 1, replace=False),
            )
        )
    selected_negative = generator.choice(negative, size=count, replace=False)
    selected = np.concatenate((selected_positive, selected_negative))
    generator.shuffle(selected)
    return selected.astype(np.int64)


class MonotoneThreshold(nn.Module):
    def __init__(self, theta_scale: float, alpha_scale: float) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(MODEL_SEED)
        # Nonzero random threshold initialization, as explicitly required.
        initial_theta_scaled = torch.empty((), dtype=torch.float64).uniform_(
            -0.25, 0.25, generator=generator
        )
        if float(initial_theta_scaled.item()) == 0.0:
            raise RuntimeError("forbidden zero theta initialization")
        initial_raw_alpha = torch.empty((), dtype=torch.float64).uniform_(
            -0.5, 0.5, generator=generator
        )
        self.theta_scaled = nn.Parameter(initial_theta_scaled)
        self.raw_alpha = nn.Parameter(initial_raw_alpha)
        self.register_buffer(
            "theta_scale", torch.tensor(theta_scale, dtype=torch.float64)
        )
        self.register_buffer(
            "alpha_scale", torch.tensor(alpha_scale, dtype=torch.float64)
        )

    def theta(self) -> torch.Tensor:
        return self.theta_scaled * self.theta_scale

    def alpha(self) -> torch.Tensor:
        return torch.exp(self.raw_alpha) / self.alpha_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha() * (x - self.theta())


def _sign_metrics(
    model: MonotoneThreshold, x: np.ndarray, delta_f: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(x, dtype=torch.float64)).cpu().numpy()
    predicted_positive = logits >= 0.0
    teacher_positive = delta_f > 0.0
    correct = predicted_positive == teacher_positive
    positive = teacher_positive
    negative = ~teacher_positive
    metrics = {
        "event_count": len(x),
        "positive_count": int(np.sum(positive)),
        "negative_count": int(np.sum(negative)),
        "overall_sign_accuracy": float(np.mean(correct)),
        "positive_accuracy": float(np.mean(correct[positive])) if np.any(positive) else math.nan,
        "negative_accuracy": float(np.mean(correct[negative])) if np.any(negative) else math.nan,
        "sign_mistakes": int(np.sum(~correct)),
        "smallest_absolute_delta_f_positive": float(np.min(np.abs(delta_f[positive]))),
        "smallest_absolute_delta_f_negative": float(np.min(np.abs(delta_f[negative]))),
    }
    return metrics, logits, correct


def _train_sign(
    train_pool: dict[str, np.ndarray],
    validation_pool: dict[str, np.ndarray],
    scale_s: float,
) -> tuple[MonotoneThreshold, list[dict[str, Any]], dict[str, Any]]:
    train_indices = _balanced_indices(train_pool, MODEL_SEED + 1)
    validation_indices = _balanced_indices(validation_pool, MODEL_SEED + 2)
    train_x_all = np.arcsinh(train_pool["delta_f"] / scale_s)
    validation_x_all = np.arcsinh(validation_pool["delta_f"] / scale_s)
    train_x = train_x_all[train_indices]
    train_y = (train_pool["delta_f"][train_indices] > 0.0).astype(np.float64)
    validation_x = validation_x_all[validation_indices]
    validation_df = validation_pool["delta_f"][validation_indices]
    # Scale the two learned parameters by the RMS compressed input for ordinary
    # conditioning.  This does not use labels or the zero/sign boundary.
    theta_scale = float(np.sqrt(np.mean(train_x**2)))
    if not theta_scale > 0.0 or not math.isfinite(theta_scale):
        raise RuntimeError("invalid nonzero theta scale")
    alpha_scale = float(np.min(np.abs(train_x)))
    if not alpha_scale > 0.0 or not math.isfinite(alpha_scale):
        raise RuntimeError("invalid nonzero alpha scale")

    torch.manual_seed(MODEL_SEED)
    model = MonotoneThreshold(theta_scale, alpha_scale).to(torch.float64)
    initial_theta = float(model.theta().detach().item())
    initial_alpha = float(model.alpha().detach().item())
    if initial_theta == 0.0:
        raise RuntimeError("theta was initialized at the teacher boundary")
    x_tensor = torch.as_tensor(train_x, dtype=torch.float64)
    y_tensor = torch.as_tensor(train_y, dtype=torch.float64)
    rows: list[dict[str, Any]] = []
    signed_target = 2.0 * y_tensor - 1.0
    boundary_temperature = 0.05 * alpha_scale
    theta_optimizer = torch.optim.LBFGS(
        [model.theta_scaled],
        lr=0.5,
        max_iter=1_000,
        tolerance_grad=1.0e-15,
        tolerance_change=1.0e-18,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def theta_closure() -> torch.Tensor:
        theta_optimizer.zero_grad(set_to_none=True)
        # A smooth maximum of all signed classification violations learns the
        # maximum-margin threshold from labels, without supplying zero or an
        # analytic threshold as a target.
        violations = -signed_target * (x_tensor - model.theta())
        boundary_loss = boundary_temperature * torch.logsumexp(
            violations / boundary_temperature, dim=0
        )
        boundary_loss.backward()
        return boundary_loss

    theta_loss = float(theta_optimizer.step(theta_closure).item())
    after_theta_metrics, after_theta_logits, _ = _sign_metrics(
        model, validation_x, validation_df
    )
    rows.append(
        {
            "record_type": "training_phase",
            "phase": "supervised_max_margin_theta",
            "boundary_temperature": boundary_temperature,
            "training_objective": theta_loss,
            "learned_theta": float(model.theta().detach().item()),
            "learned_alpha": float(model.alpha().detach().item()),
            "validation_accuracy": after_theta_metrics["overall_sign_accuracy"],
            "validation_mistakes": after_theta_metrics["sign_mistakes"],
        }
    )

    # Once theta is learned, fit only positive alpha as a confidence scale.
    model.theta_scaled.requires_grad_(False)
    alpha_optimizer = torch.optim.Adam(
        [model.raw_alpha], lr=LEARNING_RATE, eps=1.0e-15
    )
    best_state = copy.deepcopy(model.state_dict())
    best_bce = math.inf
    best_epoch = 0
    stale = 0
    stopping_reason = "maximum_alpha_epochs"
    alpha_epochs = min(MAX_EPOCHS, 2_000)
    validation_targets = (validation_df > 0.0).astype(np.float64)
    for epoch in range(1, alpha_epochs + 1):
        logits = model(x_tensor)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_tensor)
        alpha_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        alpha_optimizer.step()
        validation_metrics, validation_logits, _ = _sign_metrics(
            model, validation_x, validation_df
        )
        validation_bce = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.as_tensor(validation_logits, dtype=torch.float64),
                torch.as_tensor(validation_targets, dtype=torch.float64),
            ).item()
        )
        if validation_bce < best_bce:
            best_bce = validation_bce
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 100 == 0:
            rows.append(
                {
                    "record_type": "training_phase",
                    "phase": "positive_alpha_confidence",
                    "epoch": epoch,
                    "training_bce": float(loss.item()),
                    "validation_bce": validation_bce,
                    "learned_theta": float(model.theta().detach().item()),
                    "learned_alpha": float(model.alpha().detach().item()),
                    "validation_accuracy": validation_metrics["overall_sign_accuracy"],
                    "validation_mistakes": validation_metrics["sign_mistakes"],
                }
            )
        if stale >= 300:
            stopping_reason = "alpha_validation_bce_plateau"
            break

    model.load_state_dict(best_state)
    model.theta_scaled.requires_grad_(True)
    metrics, logits, correct = _sign_metrics(model, validation_x, validation_df)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": epoch,
        "stopping_reason": stopping_reason,
        "theta_training_loss": "smooth_max_signed_violation",
        "theta_training_boundary_supplied": False,
        "alpha_training_loss": "balanced_binary_cross_entropy",
        "training_scale_S": scale_s,
        "scale_source": "median_absolute_delta_f_of_training_unique_edges_only",
        "theta_scale_for_optimization_only": theta_scale,
        "alpha_scale_for_optimization_only": alpha_scale,
        "initial_theta_nonzero": initial_theta,
        "initial_alpha_positive": initial_alpha,
        "learned_theta": float(model.theta().detach().item()),
        "learned_alpha": float(model.alpha().detach().item()),
        "alpha_positive": bool(model.alpha().detach().item() > 0.0),
        "raw_training_positive_count": int(np.sum(train_pool["delta_f"] > 0.0)),
        "raw_training_negative_count": int(np.sum(train_pool["delta_f"] < 0.0)),
        "raw_validation_positive_count": int(np.sum(validation_pool["delta_f"] > 0.0)),
        "raw_validation_negative_count": int(np.sum(validation_pool["delta_f"] < 0.0)),
        "balanced_training_events": len(train_indices),
        "balanced_validation_events": len(validation_indices),
        **metrics,
        "smoke_gate_passed": bool(metrics["sign_mistakes"] == 0),
    }
    rows.append(summary)
    for class_name, mask in (
        ("positive", validation_df > 0.0),
        ("negative", validation_df < 0.0),
    ):
        rows.append(
            {
                "record_type": "class_summary",
                "class": class_name,
                "event_count": int(np.sum(mask)),
                "accuracy": float(np.mean(correct[mask])),
                "minimum_absolute_delta_f": float(np.min(np.abs(validation_df[mask]))),
                "minimum_absolute_x": float(np.min(np.abs(validation_x[mask]))),
            }
        )
    for local_index in np.flatnonzero(~correct):
        source_index = int(validation_indices[local_index])
        rows.append(
            {
                "record_type": "mistake",
                "source_state_index": int(validation_pool["source_state_index"][source_index]),
                "edge_position": int(validation_pool["edge_position"][source_index]),
                "delta_f": float(validation_df[local_index]),
                "x": float(validation_x[local_index]),
                "logit": float(logits[local_index]),
                "theta": float(model.theta().detach().item()),
                "distance_from_delta_f_zero": float(abs(validation_df[local_index])),
                "distance_from_x_zero": float(abs(validation_x[local_index])),
            }
        )
    return model, rows, summary


def _logmag_predict(
    checkpoint: dict[str, Any], delta_d: np.ndarray, delta_f: np.ndarray
) -> np.ndarray:
    values = np.column_stack((np.log(np.abs(delta_f)), np.log(delta_d)))
    input_scale = checkpoint["input_scale"].detach().cpu().numpy()
    target_scale = float(checkpoint["target_scale"].item())
    weight = checkpoint["state_dict"]["weight"].detach().cpu().numpy()[0]
    return ((values / input_scale) @ weight) * target_scale


def _full_sign_rows(
    model: MonotoneThreshold, pool: dict[str, np.ndarray], scale_s: float
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    x = np.arcsinh(pool["delta_f"] / scale_s)
    metrics, logits, correct = _sign_metrics(model, x, pool["delta_f"])
    summary = {
        "record_type": "final_summary",
        "split": "entire_held_out_finite_edge_pool",
        "training_scale_S": scale_s,
        "learned_theta": float(model.theta().detach().item()),
        "learned_alpha": float(model.alpha().detach().item()),
        **metrics,
        "full_pool_gate_passed": bool(metrics["sign_mistakes"] == 0),
    }
    rows: list[dict[str, Any]] = [summary]
    for class_name, mask in (
        ("positive", pool["delta_f"] > 0.0),
        ("negative", pool["delta_f"] < 0.0),
    ):
        rows.append(
            {
                "record_type": "class_summary",
                "class": class_name,
                "event_count": int(np.sum(mask)),
                "accuracy": float(np.mean(correct[mask])),
                "minimum_absolute_delta_f": float(np.min(np.abs(pool["delta_f"][mask]))),
                "minimum_absolute_x": float(np.min(np.abs(x[mask]))),
            }
        )
    for index in np.flatnonzero(~correct):
        rows.append(
            {
                "record_type": "mistake",
                "source_state_index": int(pool["source_state_index"][index]),
                "edge_position": int(pool["edge_position"][index]),
                "delta_f": float(pool["delta_f"][index]),
                "x": float(x[index]),
                "logit": float(logits[index]),
                "theta": float(model.theta().detach().item()),
                "distance_from_delta_f_zero": float(abs(pool["delta_f"][index])),
                "distance_from_x_zero": float(abs(x[index])),
            }
        )
    sign_hat = np.where(logits >= 0.0, 1.0, -1.0)
    return rows, summary, sign_hat


def _assemble_and_validate(
    states: list[dict[str, Any]],
    pool: dict[str, np.ndarray],
    sign_hat_pool: np.ndarray,
    logmag_checkpoint: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predicted_log = _logmag_predict(
        logmag_checkpoint, pool["delta_d"], pool["delta_f"]
    )
    predicted_k_pool = sign_hat_pool * np.exp(predicted_log)
    exact_k_pool = pool["exact_K"]
    relative = np.abs(predicted_k_pool - exact_k_pool) / np.abs(exact_k_pool)

    state_edge_predictions: dict[int, np.ndarray] = {}
    state_edge_exact: dict[int, np.ndarray] = {}
    for state_position in range(len(states)):
        mask = pool["state_position"] == state_position
        positions = pool["edge_position"][mask]
        order = np.argsort(positions)
        state_edge_predictions[state_position] = predicted_k_pool[mask][order]
        state_edge_exact[state_position] = exact_k_pool[mask][order]

    ordering_total = 0
    ordering_changes = 0
    kupper_total = 0
    kupper_positive_changes = 0
    lower_total = 0
    lower_winner_changes = 0
    feasibility_total = 0
    feasibility_changes = 0
    pass_total = 0
    pass_changes = 0
    for state_position, state in enumerate(states):
        exact_edges = state_edge_exact[state_position]
        predicted_edges = state_edge_predictions[state_position]
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
            learned_prev = -math.inf if position == 0 else float(predicted_edges[position - 1])
            exact_next = math.inf if position == len(hull) - 1 else float(exact_edges[position])
            learned_next = math.inf if position == len(hull) - 1 else float(predicted_edges[position])
            required = (loss - target) / size if size > 0.0 else math.inf
            exact_lower = max(exact_prev, required, 0.0)
            learned_lower = max(learned_prev, required, 0.0)
            if 0 < position < len(hull) - 1:
                ordering_total += 1
                ordering_changes += int(
                    (exact_prev <= exact_next) != (learned_prev <= learned_next)
                )
            if position < len(hull) - 1:
                kupper_total += 1
                kupper_positive_changes += int(
                    (exact_next > 0.0) != (learned_next > 0.0)
                )
            if position > 0:
                lower_total += 1
                lower_winner_changes += int(
                    logmag._active_source(exact_prev, required)
                    != logmag._active_source(learned_prev, required)
                )
            exact_feasible = exact_lower <= exact_next + turn.HULL_TOLERANCE
            learned_feasible = learned_lower <= learned_next + turn.HULL_TOLERANCE
            feasibility_total += 1
            feasibility_changes += int(exact_feasible != learned_feasible)
            exact_pass = exact_next > 0.0 and exact_feasible
            learned_pass = learned_next > 0.0 and learned_feasible
            pass_total += 1
            pass_changes += int(exact_pass != learned_pass)

    summary = {
        "record_type": "final_summary",
        "held_out_states": len(states),
        "finite_edges": len(exact_k_pool),
        "endpoint_K_prev_first": "-inf_deterministic",
        "endpoint_K_next_last": "+inf_deterministic",
        "maximum_relative_K_error": float(np.max(relative)),
        "median_relative_K_error": float(np.median(relative)),
        "neighbor_ordering_comparisons": ordering_total,
        "neighbor_ordering_changes": ordering_changes,
        "K_upper_positive_comparisons": kupper_total,
        "K_upper_positive_changes": kupper_positive_changes,
        "lower_bound_winner_comparisons": lower_total,
        "lower_bound_winner_changes": lower_winner_changes,
        "feasibility_comparisons": feasibility_total,
        "feasibility_comparison_changes": feasibility_changes,
        "final_improvement_pass_comparisons": pass_total,
        "final_improvement_pass_changes": pass_changes,
    }
    summary["assembled_gate_passed"] = bool(
        ordering_changes == 0
        and kupper_positive_changes == 0
        and lower_winner_changes == 0
        and feasibility_changes == 0
        and pass_changes == 0
    )
    rows: list[dict[str, Any]] = [summary]
    for index in range(len(exact_k_pool)):
        rows.append(
            {
                "record_type": "finite_edge",
                "source_state_index": int(pool["source_state_index"][index]),
                "edge_position": int(pool["edge_position"][index]),
                "delta_d": float(pool["delta_d"][index]),
                "delta_f": float(pool["delta_f"][index]),
                "teacher_K": float(exact_k_pool[index]),
                "learned_log_magnitude": float(predicted_log[index]),
                "learned_sign": int(sign_hat_pool[index]),
                "learned_K": float(predicted_k_pool[index]),
                "relative_K_error": float(relative[index]),
            }
        )
    return rows, summary


def _save_sign_checkpoint(
    model: MonotoneThreshold,
    scale_s: float,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
    assembled_summary: dict[str, Any],
    frozen_logmag_hash: str,
) -> str:
    checkpoint = {
        "format_version": 1,
        "component": "hull_slope_sign",
        "status": "LOCKED_AS_NEIGHBORING_HULL_SLOPE_COMPONENT",
        "architecture": "logit = exp(raw_alpha)/alpha_scale * (asinh(delta_f/S) - theta); no hidden layers",
        "state_dict": copy.deepcopy(model.state_dict()),
        "training_scale_S": torch.tensor(scale_s, dtype=torch.float64),
        "learned_theta": model.theta().detach().cpu(),
        "learned_alpha": model.alpha().detach().cpu(),
        "prediction_rule": "POSITIVE iff logit >= 0; NEGATIVE otherwise",
        "source_log_magnitude_checkpoint_sha256": SOURCE_LOGMAG_SHA256,
        "frozen_log_magnitude_artifact_sha256": frozen_logmag_hash,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
        "assembled_summary": assembled_summary,
        "true_parameter_or_improvement_labels_used": False,
    }
    torch.save(checkpoint, SIGN_CHECKPOINT_PATH)
    sign_hash = _sha256(SIGN_CHECKPOINT_PATH)
    SIGN_HASH_PATH.write_text(
        f"{sign_hash}  {SIGN_CHECKPOINT_PATH.name}\n", encoding="utf-8"
    )
    restored = torch.load(
        SIGN_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if restored["status"] != "LOCKED_AS_NEIGHBORING_HULL_SLOPE_COMPONENT":
        raise RuntimeError("sign checkpoint readback failed")
    return sign_hash


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logmag_checkpoint, frozen_logmag_hash = _verify_and_freeze_logmag()

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
    train_states = turn._build_states(payload, train_indices, workers, "slope sign train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "slope sign validation"
    )
    train_pool = _build_edge_pool(train_states, "train")
    validation_pool = _build_edge_pool(validation_states, "validation")
    scale_s = float(np.median(np.abs(train_pool["delta_f"])))
    if not scale_s > 0.0 or not math.isfinite(scale_s):
        raise RuntimeError("invalid training-only asinh scale")

    model, smoke_rows, smoke_summary = _train_sign(
        train_pool, validation_pool, scale_s
    )
    smoke_summary["source_log_magnitude_checkpoint_sha256"] = SOURCE_LOGMAG_SHA256
    smoke_summary["frozen_log_magnitude_artifact_sha256"] = frozen_logmag_hash
    smoke_summary["slope_log_magnitude_status"] = "FROZEN_CANDIDATE"
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "slope_log_magnitude",
                    "status": "FROZEN_CANDIDATE",
                    "source_sha256": SOURCE_LOGMAG_SHA256,
                    "frozen_artifact_sha256": frozen_logmag_hash,
                    "retrained": False,
                },
                {
                    "component": "neighboring_hull_slope_operator",
                    "status": "NOT_LOCKED_SIGN_SMOKE_FAILED",
                },
            ],
        )
        print("slope-sign smoke gate failed; stopped", flush=True)
        return

    full_rows, full_summary, sign_hat = _full_sign_rows(
        model, validation_pool, scale_s
    )
    _write_csv(FULL_PATH, full_rows)
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "slope_log_magnitude",
                    "status": "FROZEN_CANDIDATE",
                    "source_sha256": SOURCE_LOGMAG_SHA256,
                    "frozen_artifact_sha256": frozen_logmag_hash,
                    "retrained": False,
                },
                {
                    "component": "neighboring_hull_slope_operator",
                    "status": "NOT_LOCKED_SIGN_FULL_POOL_FAILED",
                },
            ],
        )
        print("slope-sign full-pool gate failed; stopped", flush=True)
        return

    assembled_rows, assembled_summary = _assemble_and_validate(
        validation_states,
        validation_pool,
        sign_hat,
        logmag_checkpoint,
    )
    _write_csv(ASSEMBLED_PATH, assembled_rows)
    if not bool(assembled_summary["assembled_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "slope_log_magnitude",
                    "status": "FROZEN_CANDIDATE",
                    "source_sha256": SOURCE_LOGMAG_SHA256,
                    "frozen_artifact_sha256": frozen_logmag_hash,
                    "retrained": False,
                },
                {
                    "component": "hull_slope_sign",
                    "status": "VALIDATED_BUT_NOT_LOCKED_ASSEMBLY_FAILED",
                },
                {
                    "component": "neighboring_hull_slope_operator",
                    "status": "NOT_LOCKED_ASSEMBLY_FAILED",
                },
            ],
        )
        print("assembled learned-slope gate failed; stopped", flush=True)
        return

    sign_hash = _save_sign_checkpoint(
        model,
        scale_s,
        smoke_summary,
        full_summary,
        assembled_summary,
        frozen_logmag_hash,
    )
    lock_rows = [
        {
            "component": "slope_log_magnitude",
            "status": "FROZEN_CANDIDATE_COMPONENT_OF_LOCKED_OPERATOR",
            "source_checkpoint": str(SOURCE_LOGMAG_PATH),
            "source_sha256": SOURCE_LOGMAG_SHA256,
            "frozen_artifact": str(FROZEN_LOGMAG_PATH),
            "frozen_artifact_sha256": frozen_logmag_hash,
            "retrained": False,
        },
        {
            "component": "hull_slope_sign",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(SIGN_CHECKPOINT_PATH),
            "checkpoint_sha256": sign_hash,
            "learned_theta": float(model.theta().detach().item()),
            "learned_alpha": float(model.alpha().detach().item()),
            "training_scale_S": scale_s,
        },
        {
            "component": "neighboring_hull_slope_operator",
            "status": "LOCKED",
            "finite_slope_formula": "learned_sign * exp(frozen_learned_log_magnitude)",
            "endpoint_sentinels": "K_prev(first)=-inf; K_next(last)=+inf",
            "full_pool_sign_mistakes": full_summary["sign_mistakes"],
            "neighbor_ordering_changes": assembled_summary["neighbor_ordering_changes"],
            "K_upper_positive_changes": assembled_summary["K_upper_positive_changes"],
            "lower_bound_winner_changes": assembled_summary["lower_bound_winner_changes"],
            "feasibility_comparison_changes": assembled_summary["feasibility_comparison_changes"],
            "final_improvement_pass_changes": assembled_summary["final_improvement_pass_changes"],
        },
    ]
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        f"neighboring hull slope operator LOCKED theta={model.theta().item():.17g} "
        f"alpha={model.alpha().item():.17g} sign_sha256={sign_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
