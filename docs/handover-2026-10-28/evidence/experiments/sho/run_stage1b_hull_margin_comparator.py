#!/usr/bin/env python3
"""Lock the skew turn score and learn only the continuous tolerance margin."""

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

import experiments.sho.run_stage1b_balanced_hull_turn_smokescreen as balanced
import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.validate_stage1b_bilinear_hull_turn as validation


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_margin_comparator")
AUDIT_PATH = OUTPUT_DIR / "comparator_audit.txt"
SMOKE_PATH = OUTPUT_DIR / "hull_margin_smoke.csv"
FULL_POOL_PATH = OUTPUT_DIR / "hull_margin_full_pool.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "hull_margin_reconstruction.csv"

SKEW_SOURCE_PATH = Path(
    "outputs/sho/stage1B/skew_bilinear_hull_turn/skew_bilinear_training.csv"
)
LOCKED_CHECKPOINT_PATH = Path(
    "outputs/sho/stage1B/skew_bilinear_hull_turn/"
    "skew_bilinear_epoch35_locked.pt"
)
LOCKED_HASH_PATH = Path(
    "outputs/sho/stage1B/skew_bilinear_hull_turn/"
    "skew_bilinear_epoch35_locked.sha256"
)

MAX_EPOCHS = 1_000
PATIENCE = 50
PLATEAU_RELATIVE_IMPROVEMENT = 1.0e-6
LEARNING_RATE = 1.0e-2
MODEL_SEED = smoke.MODEL_SEED + 577_215
RMSE_GATE = 1.0e-12
MAX_ERROR_GATE = 1.0e-10


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


def _load_skew_source() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    with SKEW_SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = next(row for row in rows if row["record_type"] == "final_summary")
    if int(summary["best_epoch"]) != 35:
        raise RuntimeError("skew source is not the successful epoch-35 result")
    if summary["mistake_count"] != "0":
        raise RuntimeError("skew source was not locally exact")
    raw = np.asarray(
        [
            [float(summary["A_00"]), float(summary["A_01"])],
            [float(summary["A_10"]), float(summary["A_11"])],
        ],
        dtype=np.float64,
    )
    weight = raw - raw.T
    recorded = np.asarray(
        [
            [float(summary["W_00"]), float(summary["W_01"])],
            [float(summary["W_10"]), float(summary["W_11"])],
        ],
        dtype=np.float64,
    )
    if not np.array_equal(weight, recorded):
        raise RuntimeError("reconstructed skew state differs from recorded W")
    return raw, weight, summary


def _lock_skew_checkpoint() -> tuple[np.ndarray, str]:
    raw, weight, summary = _load_skew_source()
    source_hash = _sha256(SKEW_SOURCE_PATH)
    checkpoint = {
        "format_version": 1,
        "component": "hull_turn_continuous_score",
        "lock_status": "LOCKED",
        "architecture": "W=A-A^T; y_hat=u1^T W u2",
        "source_epoch": 35,
        "state_dict": {
            "raw_matrix": torch.as_tensor(raw.copy(), dtype=torch.float64)
        },
        "effective_weight": torch.as_tensor(weight.copy(), dtype=torch.float64),
        "source_training_path": str(SKEW_SOURCE_PATH),
        "source_training_sha256": source_hash,
        "full_pool_event_count": 98_509,
        "full_pool_mistake_count": 0,
        "exact_hull_states": 200,
        "exact_po_states": 200,
        "training_performed_during_locking": False,
        "never_retrain_without_explicit_request": True,
    }
    torch.save(checkpoint, LOCKED_CHECKPOINT_PATH)
    checkpoint_hash = _sha256(LOCKED_CHECKPOINT_PATH)
    restored = torch.load(
        LOCKED_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    restored_raw = restored["state_dict"]["raw_matrix"].cpu().numpy()
    restored_weight = restored["effective_weight"].cpu().numpy()
    if not np.array_equal(restored_raw, raw) or not np.array_equal(
        restored_weight, weight
    ):
        raise RuntimeError("locked skew checkpoint failed exact readback")
    LOCKED_HASH_PATH.write_text(
        f"{checkpoint_hash}  {LOCKED_CHECKPOINT_PATH.name}\n",
        encoding="utf-8",
    )
    return weight, checkpoint_hash


def _write_audit(checkpoint_hash: str) -> None:
    text = f"""Exact deterministic hull-turn comparator audit
================================================

Audited implementation:
  experiments/sho/run_stage1b_hull_turn_smokescreen.py::_turn_values

Exact implementation logic:

    cross = delta_d_ab * delta_f_bc - delta_f_ab * delta_d_bc
    scale = max(1.0, abs(loss_A), abs(loss_B), abs(loss_C))
    tolerance = 5.0e-14 * scale
    keep = int(not (cross < -tolerance))

Raw score:
  C_raw = cross(v1, v2)

Exact tolerance:
  tau_raw = 5.0e-14 * max(1, |f_A|, |f_B|, |f_C|)

Exact inequality:
  POP  iff C_raw < -tau_raw
  KEEP iff C_raw >= -tau_raw

Equality and collinearity:
  C_raw == -tau_raw is KEEP because the implementation uses strict '<'.
  C_raw == 0 is KEEP because tau_raw is strictly positive.

Exact continuous raw margin:
  margin_raw = C_raw + tau_raw
  KEEP iff margin_raw >= 0; POP iff margin_raw < 0.

Normalized equivalent (D = ||v1||*||v2|| > 0, with no epsilon):
  y_geom = C_raw / D
  tau_geom = tau_raw / D
  margin_geom_exact = (C_raw + tau_raw) / D
  KEEP iff margin_geom_exact >= 0; POP iff margin_geom_exact < 0.

The learned next operation receives:
  [frozen_skew_y_hat, tau_geom]
and regresses margin_geom_exact continuously. KEEP/POP is never a training target.

Locked component:
  hull_turn_continuous_score = LOCKED
  checkpoint = {LOCKED_CHECKPOINT_PATH}
  SHA-256 = {checkpoint_hash}
  skew network retrained in this run = False
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


def _skew_predict(weight: np.ndarray, unit_inputs: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(unit_inputs, dtype=torch.float64)
    matrix = torch.as_tensor(weight, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            batch = values[start : start + 4096]
            parts.append(
                torch.einsum("bi,ij,bj->b", batch[:, :2], matrix, batch[:, 2:])
            )
    return torch.cat(parts).cpu().numpy()


def _margin_dataset(
    selected: dict[str, np.ndarray],
    states: list[dict[str, Any]],
    locked_weight: np.ndarray,
) -> dict[str, np.ndarray]:
    geometric = smoke._dataset(selected, states)
    skew_score = _skew_predict(locked_weight, geometric["inputs"])
    tolerance_raw = selected["tolerance"].astype(np.float64, copy=True)
    target_margin = (
        geometric["raw_cross"] + tolerance_raw
    ) / geometric["denominator"]
    labels = selected["label"].astype(np.int64, copy=True)
    if not np.array_equal((target_margin >= 0.0).astype(np.int64), labels):
        raise RuntimeError("exact continuous margin changed teacher decisions")
    return {
        "inputs": np.column_stack((skew_score, geometric["tau_geom"])),
        "target": target_margin,
        "label": labels,
        "skew_score": skew_score,
        "tau_geom": geometric["tau_geom"],
        "tau_raw": tolerance_raw,
        "raw_cross": geometric["raw_cross"],
        "denominator": geometric["denominator"],
        "unit_inputs": geometric["inputs"],
        "state_position": geometric["state_position"],
        "triple_indices": geometric["triple_indices"],
    }


class AffineStandardization:
    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        self.input_mean = np.mean(inputs, axis=0)
        self.input_scale = np.std(inputs, axis=0)
        if np.any(self.input_scale <= 0.0) or not np.all(
            np.isfinite(self.input_scale)
        ):
            raise RuntimeError("non-positive margin input scale")
        self.target_mean = float(np.mean(target))
        self.target_scale = float(np.std(target))
        if not self.target_scale > 0.0 or not np.isfinite(self.target_scale):
            raise RuntimeError("non-positive margin target scale")

    def transform_inputs(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_mean) / self.input_scale

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale + self.target_mean

    def original_coefficients(self, model: nn.Linear) -> tuple[np.ndarray, float]:
        standardized_weight = model.weight.detach().cpu().numpy()[0]
        standardized_bias = float(model.bias.detach().cpu().numpy()[0])
        coefficient = self.target_scale * standardized_weight / self.input_scale
        bias = (
            self.target_mean
            + self.target_scale * standardized_bias
            - float(np.dot(coefficient, self.input_mean))
        )
        return coefficient, bias


def _predict_margin(
    model: nn.Linear, standardization: AffineStandardization, inputs: np.ndarray
) -> np.ndarray:
    values = torch.as_tensor(
        standardization.transform_inputs(inputs), dtype=torch.float64
    )
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(model(values[start : start + 4096]).squeeze(-1))
    standardized = torch.cat(parts).cpu().numpy()
    return standardization.inverse_target(standardized)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "correlation": float(np.corrcoef(target, prediction)[0, 1]),
        "absolute_error_p50": float(np.quantile(absolute, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p99": float(np.quantile(absolute, 0.99)),
        "absolute_error_p100": float(np.max(absolute)),
    }


def _decision_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    teacher = labels.astype(bool)
    predicted = prediction >= 0.0
    keep_count = int(np.sum(teacher))
    pop_count = int(np.sum(~teacher))
    keep_correct = int(np.sum(teacher & predicted))
    pop_correct = int(np.sum(~teacher & ~predicted))
    return {
        "decision_accuracy": float(np.mean(predicted == teacher)),
        "keep_agreement": keep_correct / keep_count if keep_count else "",
        "pop_agreement": pop_correct / pop_count if pop_count else "",
        "mistake_count": int(np.sum(predicted != teacher)),
        "keep_count": keep_count,
        "pop_count": pop_count,
        "keep_correct": keep_correct,
        "pop_correct": pop_correct,
    }


def _train_margin(
    training: dict[str, np.ndarray], validation_set: dict[str, np.ndarray]
) -> tuple[nn.Linear, AffineStandardization, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED + 97)
    standardization = AffineStandardization(training["inputs"], training["target"])
    train_inputs = torch.as_tensor(
        standardization.transform_inputs(training["inputs"]), dtype=torch.float64
    )
    train_target = torch.as_tensor(
        standardization.transform_target(training["target"]), dtype=torch.float64
    )
    model = nn.Linear(2, 1).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    plateau_mse = math.inf
    best_epoch = 0
    stale = 0
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
            prediction = model(train_inputs[indices]).squeeze(-1)
            loss = nn.functional.mse_loss(prediction, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.item()) * len(indices)
        validation_prediction = _predict_margin(
            model, standardization, validation_set["inputs"]
        )
        validation_mse = float(
            np.mean((validation_prediction - validation_set["target"]) ** 2)
        )
        epoch_metrics = {
            **_metrics(validation_set["target"], validation_prediction),
            **_decision_metrics(validation_set["label"], validation_prediction),
        }
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if validation_mse < plateau_mse * (1.0 - PLATEAU_RELATIVE_IMPROVEMENT):
            plateau_mse = validation_mse
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_standardized_mse": total_mse / len(order),
                "validation_margin_mse": validation_mse,
                "plateau_stale_epochs": stale,
                **epoch_metrics,
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"margin epoch={epoch} rmse={epoch_metrics['rmse']:.6e} "
                f"max={epoch_metrics['maximum_absolute_error']:.6e} "
                f"decision={epoch_metrics['decision_accuracy']:.6f} stale={stale}",
                flush=True,
            )
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    model.eval()
    prediction = _predict_margin(model, standardization, validation_set["inputs"])
    final_metrics = {
        **_metrics(validation_set["target"], prediction),
        **_decision_metrics(validation_set["label"], prediction),
    }
    coefficients, bias = standardization.original_coefficients(model)
    passed = bool(
        final_metrics["rmse"] <= RMSE_GATE
        and final_metrics["maximum_absolute_error"] <= MAX_ERROR_GATE
        and final_metrics["decision_accuracy"] == 1.0
        and final_metrics["keep_agreement"] == 1.0
        and final_metrics["pop_agreement"] == 1.0
    )
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "maximum_epochs": MAX_EPOCHS,
        "plateau_patience": PATIENCE,
        "training_events": len(training["target"]),
        "validation_events": len(validation_set["target"]),
        **final_metrics,
        "learned_score_coefficient": float(coefficients[0]),
        "learned_tolerance_coefficient": float(coefficients[1]),
        "learned_bias": bias,
        "posthoc_score_coefficient_error": float(abs(coefficients[0] - 1.0)),
        "posthoc_tolerance_coefficient_error": float(abs(coefficients[1] - 1.0)),
        "posthoc_bias_error": float(abs(bias)),
        "minimum_absolute_teacher_margin": float(
            np.min(np.abs(validation_set["target"]))
        ),
        "maximum_error_to_minimum_boundary_distance_ratio": float(
            final_metrics["maximum_absolute_error"]
            / np.min(np.abs(validation_set["target"]))
        ),
        "rmse_gate": RMSE_GATE,
        "maximum_error_gate": MAX_ERROR_GATE,
        "smoke_gate_passed": passed,
        "classification_target_used": False,
        "final_sign_learned": False,
    }
    rows.append(summary)
    return model, standardization, rows, summary


def _evaluation_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, record: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = {
        "record_type": "final_summary",
        "evaluation": record,
        "event_count": len(prediction),
        **_metrics(dataset["target"], prediction),
        **_decision_metrics(dataset["label"], prediction),
    }
    rows: list[dict[str, Any]] = [summary]
    teacher = dataset["label"].astype(bool)
    predicted = prediction >= 0.0
    for index in np.flatnonzero(predicted != teacher):
        triple = dataset["triple_indices"][index]
        rows.append(
            {
                "record_type": "mistake",
                "event_index": int(index),
                "state_position": int(dataset["state_position"][index]),
                "triple_first": int(triple[0]),
                "triple_second": int(triple[1]),
                "triple_third": int(triple[2]),
                "teacher_decision": "KEEP" if teacher[index] else "POP",
                "predicted_decision": "KEEP" if predicted[index] else "POP",
                "frozen_skew_score": float(dataset["skew_score"][index]),
                "tau_geom": float(dataset["tau_geom"][index]),
                "exact_teacher_margin": float(dataset["target"][index]),
                "predicted_margin": float(prediction[index]),
                "absolute_error": float(
                    abs(prediction[index] - dataset["target"][index])
                ),
                "absolute_distance_from_boundary": float(
                    abs(dataset["target"][index])
                ),
            }
        )
    return rows, summary


def _learned_margin_hull(
    locked_weight: np.ndarray,
    coefficient: np.ndarray,
    bias: float,
    sizes: np.ndarray,
    losses: np.ndarray,
) -> tuple[np.ndarray, int]:
    stack: list[int] = []
    decisions = 0
    for third in range(len(sizes)):
        while len(stack) >= 2:
            score, tolerance, _ = validation._frozen_turn(
                locked_weight, sizes, losses, stack[-2], stack[-1], third
            )
            margin = float(coefficient[0] * score + coefficient[1] * tolerance + bias)
            decisions += 1
            if margin >= 0.0:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64), decisions


def _reconstruction_rows(
    locked_weight: np.ndarray,
    coefficient: np.ndarray,
    bias: float,
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_state: list[dict[str, Any]] = []
    for state in states:
        sizes = state["sizes"]
        losses = state["losses"]
        teacher_hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        learned_hull, decisions = _learned_margin_hull(
            locked_weight, coefficient, bias, sizes, losses
        )
        teacher_po = edirect._potentially_optimal_cells(sizes, losses)
        learned_po = validation._improvement_filter(learned_hull, sizes, losses)
        teacher_hull_set = set(int(value) for value in teacher_hull)
        learned_hull_set = set(int(value) for value in learned_hull)
        teacher_po_set = set(int(value) for value in teacher_po)
        learned_po_set = set(int(value) for value in learned_po)
        deterministic_hull, _ = turn._teacher_hull_and_triples(sizes, losses)
        per_state.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "state_bin": state["state_bin"],
                "representative_cell_count": len(sizes),
                "teacher_hull_size": len(teacher_hull),
                "learned_hull_size": len(learned_hull),
                "learned_margin_decisions": decisions,
                "false_hull_additions": len(learned_hull_set - teacher_hull_set),
                "false_hull_removals": len(teacher_hull_set - learned_hull_set),
                "exact_hull": learned_hull_set == teacher_hull_set,
                "false_po_additions": len(learned_po_set - teacher_po_set),
                "false_po_removals": len(teacher_po_set - learned_po_set),
                "exact_po_set": learned_po_set == teacher_po_set,
                "deterministic_teacher_exact": bool(
                    np.array_equal(teacher_hull, deterministic_hull)
                ),
            }
        )
    summary = {
        "record_type": "final_summary",
        "state_count": len(per_state),
        "exact_hull_states": sum(bool(row["exact_hull"]) for row in per_state),
        "exact_hull_fraction": float(np.mean([row["exact_hull"] for row in per_state])),
        "false_hull_additions": sum(int(row["false_hull_additions"]) for row in per_state),
        "false_hull_removals": sum(int(row["false_hull_removals"]) for row in per_state),
        "exact_po_states": sum(bool(row["exact_po_set"]) for row in per_state),
        "exact_po_fraction": float(np.mean([row["exact_po_set"] for row in per_state])),
        "false_po_additions": sum(int(row["false_po_additions"]) for row in per_state),
        "false_po_removals": sum(int(row["false_po_removals"]) for row in per_state),
        "deterministic_teacher_exact_states": sum(
            bool(row["deterministic_teacher_exact"]) for row in per_state
        ),
    }
    summary["reconstruction_passed"] = bool(
        summary["exact_hull_fraction"] == 1.0
        and summary["exact_po_fraction"] == 1.0
        and summary["deterministic_teacher_exact_states"] == len(per_state)
    )
    return [summary] + per_state, summary


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECONSTRUCTION_PATH.exists():
        RECONSTRUCTION_PATH.unlink()
    locked_weight, checkpoint_hash = _lock_skew_checkpoint()
    _write_audit(checkpoint_hash)

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
    training = _margin_dataset(train_selected, train_states, locked_weight)
    validation_set = _margin_dataset(
        validation_selected, validation_states, locked_weight
    )
    model, standardization, smoke_rows, smoke_summary = _train_margin(
        training, validation_set
    )
    smoke_summary["locked_skew_checkpoint_sha256"] = checkpoint_hash
    smoke_rows.extend(
        [
            {"record_type": "dataset_balance", "split": "train", **train_counts},
            {
                "record_type": "dataset_balance",
                "split": "validation",
                **validation_counts,
            },
        ]
    )
    _write_csv(SMOKE_PATH, smoke_rows)

    if not bool(smoke_summary["smoke_gate_passed"]):
        _write_csv(
            FULL_POOL_PATH,
            [{"record_type": "not_run", "reason": "smoke gate failed"}],
        )
        print("margin smoke gate failed; stopped before full-pool evaluation", flush=True)
        return

    full_dataset = _margin_dataset(validation_pool, validation_states, locked_weight)
    full_prediction = _predict_margin(
        model, standardization, full_dataset["inputs"]
    )
    full_rows, full_summary = _evaluation_rows(
        full_dataset, full_prediction, "full_held_out_pool"
    )
    full_passed = bool(
        full_summary["rmse"] <= RMSE_GATE
        and full_summary["maximum_absolute_error"] <= MAX_ERROR_GATE
        and full_summary["decision_accuracy"] == 1.0
        and full_summary["keep_agreement"] == 1.0
        and full_summary["pop_agreement"] == 1.0
    )
    full_summary["full_pool_gate_passed"] = full_passed
    full_summary["locked_skew_checkpoint_sha256"] = checkpoint_hash
    full_summary["learned_score_coefficient"] = smoke_summary[
        "learned_score_coefficient"
    ]
    full_summary["learned_tolerance_coefficient"] = smoke_summary[
        "learned_tolerance_coefficient"
    ]
    full_summary["learned_bias"] = smoke_summary["learned_bias"]
    _write_csv(FULL_POOL_PATH, full_rows)
    if not full_passed:
        print(
            f"full pool failed: mistakes={full_summary['mistake_count']}",
            flush=True,
        )
        return

    coefficients, bias = standardization.original_coefficients(model)
    reconstruction_indices = turn._stratified_state_indices(
        payload["states"],
        "reconstruction",
        turn.RECONSTRUCTION_STATES,
        smoke.DATA_SEED + 2,
    )
    reconstruction_states = turn._build_states(
        payload, reconstruction_indices, workers, "hull reconstruction"
    )
    reconstruction_rows, reconstruction_summary = _reconstruction_rows(
        locked_weight, coefficients, bias, reconstruction_states
    )
    _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)
    print(
        f"margin best_epoch={smoke_summary['best_epoch']} "
        f"stop_epoch={smoke_summary['stop_epoch']} "
        f"coefficients=({coefficients[0]:.17g},{coefficients[1]:.17g}) "
        f"bias={bias:.6e}",
        flush=True,
    )
    print(
        f"full_pool_mistakes={full_summary['mistake_count']} "
        f"exact_hulls={reconstruction_summary['exact_hull_states']}/"
        f"{reconstruction_summary['state_count']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
