#!/usr/bin/env python3
"""Learn only the final Boolean AND in the improvement condition."""

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

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_feasibility_margin as feasibility_margin
import experiments.sho.run_stage1b_feasibility_threshold as feasibility_threshold
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/improvement_and")
AUDIT_PATH = OUTPUT_DIR / "improvement_and_audit.csv"
SMOKE_PATH = OUTPUT_DIR / "improvement_and_smoke.csv"
FULL_PATH = OUTPUT_DIR / "improvement_and_full_pool.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "improvement_condition_operator_lock_summary.csv"
AND_PATH = OUTPUT_DIR / "improvement_and_locked.pt"
HASH_PATH = AND_PATH.with_suffix(".sha256")

FEASIBILITY_MARGIN_PATH = feasibility_margin.CANDIDATE_PATH
FEASIBILITY_MARGIN_SHA256 = feasibility_threshold.MARGIN_SHA256
FEASIBILITY_THRESHOLD_PATH = feasibility_threshold.THRESHOLD_PATH
FEASIBILITY_THRESHOLD_SHA256 = (
    "de943f82988a6cf5bfbba22578a1f1dbd7c56c534fe435755a37a18ca4e24fc5"
)
SLOPE_SIGN_PATH = feasibility_margin.lower_margin.SLOPE_SIGN_PATH
SLOPE_SIGN_SHA256 = feasibility_margin.lower_margin.SLOPE_SIGN_SHA256

MODEL_SEED = smoke.MODEL_SEED + 9_424_777
MAX_EPOCHS = 5_000
PATIENCE = 500
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_hashes() -> None:
    feasibility_threshold._verify_frozen_hashes()
    for path, expected, label in (
        (
            FEASIBILITY_THRESHOLD_PATH,
            FEASIBILITY_THRESHOLD_SHA256,
            "locked feasibility threshold",
        ),
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


def _feasibility_threshold_parameters() -> tuple[float, float]:
    checkpoint = torch.load(
        FEASIBILITY_THRESHOLD_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("feasibility threshold is not locked")
    return (
        float(checkpoint["learned_theta"].item()),
        float(checkpoint["learned_alpha"].item()),
    )


def _build_pool(
    states: list[dict[str, Any]], split: str
) -> dict[str, np.ndarray]:
    feasibility = feasibility_margin._build_pool(states, split)
    learned_y = feasibility_threshold._frozen_margin(feasibility)
    theta, alpha = _feasibility_threshold_parameters()
    learned_finite_feasible = alpha * (learned_y - theta) >= 0.0

    upstream = feasibility["upstream"]
    finite_indices = feasibility["finite_event_indices"]
    endpoint_indices = feasibility["endpoint_event_indices"]
    learned_feasible = np.ones(len(upstream["A"]), dtype=bool)
    learned_feasible[finite_indices] = learned_finite_feasible

    # Reuse the locked learned slope-sign output for finite K_upper.  The
    # endpoint +inf sentinel is deterministically positive.
    learned_positive_upper = np.ones(len(upstream["A"]), dtype=bool)
    learned_positive_upper[finite_indices] = (
        upstream["locked_sign_next"][finite_indices] > 0.0
    )
    inputs = np.column_stack(
        (
            learned_positive_upper.astype(np.float64),
            learned_feasible.astype(np.float64),
        )
    )
    teacher_pass = upstream["exact_pass"].astype(bool)
    exact_positive = upstream["exact_K_next"] > 0.0
    exact_feasible = upstream["exact_feasibility"].astype(bool)
    if np.any(learned_positive_upper != exact_positive):
        raise RuntimeError("locked slope-sign positivity differs from teacher")
    if np.any(learned_feasible != exact_feasible):
        raise RuntimeError("locked feasibility operator differs from teacher")
    if np.any(teacher_pass != (exact_positive & exact_feasible)):
        raise RuntimeError("teacher PASS is not positivity AND feasibility")
    return {
        "inputs": inputs,
        "positive_upper": learned_positive_upper,
        "feasible": learned_feasible,
        "teacher_pass": teacher_pass,
        "state_position": upstream["state_position"],
        "source_state_index": upstream["source_state_index"],
        "hull_position": upstream["hull_position"],
        "endpoint_indices": endpoint_indices,
        "finite_indices": finite_indices,
        "split": np.asarray([split], dtype=object),
    }


def _combination_rows(
    pool: dict[str, np.ndarray], split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for positive in (False, True):
        for feasible in (False, True):
            mask = (pool["positive_upper"] == positive) & (
                pool["feasible"] == feasible
            )
            passes = pool["teacher_pass"][mask]
            rows.append(
                {
                    "record_type": "input_combination",
                    "split": split,
                    "positive_upper": int(positive),
                    "feasible": int(feasible),
                    "event_count": int(np.sum(mask)),
                    "teacher_PASS_count": int(np.sum(passes)),
                    "teacher_FAIL_count": int(np.sum(~passes)),
                    "combination_absent_from_real_data": not np.any(mask),
                    "expected_AND_output": int(positive and feasible),
                }
            )
    rows.append(
        {
            "record_type": "audit_summary",
            "split": split,
            "event_count": len(pool["teacher_pass"]),
            "teacher_PASS_count": int(np.sum(pool["teacher_pass"])),
            "teacher_FAIL_count": int(np.sum(~pool["teacher_pass"])),
            "endpoint_positive_count": len(pool["endpoint_indices"]),
            "finite_positivity_from_locked_slope_sign": True,
            "new_K_upper_positivity_network_trained": False,
        }
    )
    return rows


def _balanced_indices(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    passing = np.flatnonzero(labels)
    failing = np.flatnonzero(~labels)
    count = min(len(passing), len(failing))
    if count == 0:
        raise RuntimeError("cannot form balanced PASS/FAIL dataset")
    generator = np.random.default_rng(seed)
    chosen_pass = generator.choice(passing, size=count, replace=False)
    chosen_fail = generator.choice(failing, size=count, replace=False)
    selected = np.concatenate((chosen_pass, chosen_fail))
    generator.shuffle(selected)
    return selected, {
        "raw_PASS_count": len(passing),
        "raw_FAIL_count": len(failing),
        "selected_PASS_count": count,
        "selected_FAIL_count": count,
        "selected_event_count": 2 * count,
        "all_real_minority_examples_used": True,
        "synthetic_examples_used": False,
    }


def _metrics(
    model: nn.Linear, inputs: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    with torch.no_grad():
        logits = (
            model(torch.as_tensor(inputs, dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    prediction = logits >= 0.0
    mistakes = prediction != labels
    return {
        "overall_accuracy": float(np.mean(~mistakes)),
        "PASS_accuracy": float(
            np.mean(~mistakes[labels]) if np.any(labels) else 1.0
        ),
        "FAIL_accuracy": float(
            np.mean(~mistakes[~labels]) if np.any(~labels) else 1.0
        ),
        "mistake_count": int(np.sum(mistakes)),
        "prediction": prediction,
        "logits": logits,
    }


def _mistake_rows(
    model: nn.Linear,
    pool: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    split: str,
    original_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    metrics = _metrics(model, inputs, labels)
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(metrics["prediction"] != labels):
        event_index = (
            int(original_indices[local_index])
            if original_indices is not None
            else int(local_index)
        )
        rows.append(
            {
                "record_type": "mistake",
                "split": split,
                "event_index": event_index,
                "positive_upper": int(inputs[local_index, 0]),
                "feasible": int(inputs[local_index, 1]),
                "teacher_PASS": bool(labels[local_index]),
                "predicted_PASS": bool(metrics["prediction"][local_index]),
                "logit": float(metrics["logits"][local_index]),
                "source_state_index": (
                    int(pool["source_state_index"][event_index])
                    if original_indices is not None
                    else ""
                ),
                "hull_position": (
                    int(pool["hull_position"][event_index])
                    if original_indices is not None
                    else ""
                ),
            }
        )
    return rows


def _train(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[nn.Linear, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(2, 1, bias=True).to(torch.float64)
    initial_weight = model.weight.detach().cpu().numpy()[0].copy()
    initial_bias = float(model.bias.detach().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    x_train = torch.as_tensor(train_inputs, dtype=torch.float64)
    y_train = torch.as_tensor(train_labels.astype(np.float64), dtype=torch.float64)
    x_validation = torch.as_tensor(validation_inputs, dtype=torch.float64)
    y_validation = torch.as_tensor(
        validation_labels.astype(np.float64), dtype=torch.float64
    )
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    stale = 0
    exact_streak = 0
    stopping_reason = "maximum_epochs"
    stop_epoch = MAX_EPOCHS
    for epoch in range(1, MAX_EPOCHS + 1):
        logits = model(x_train).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_logits = model(x_validation).squeeze(-1)
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, y_validation
                ).item()
            )
        validation_metrics = _metrics(
            model, validation_inputs, validation_labels
        )
        key = (validation_metrics["mistake_count"], validation_loss)
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        exact_streak = (
            exact_streak + 1
            if validation_metrics["mistake_count"] == 0
            else 0
        )
        if epoch == 1 or epoch % 100 == 0:
            weight = model.weight.detach().cpu().numpy()[0]
            bias = float(model.bias.detach().item())
            print(
                f"AND epoch={epoch} w1={weight[0]:.6g} w2={weight[1]:.6g} "
                f"b={bias:.6g} mistakes={validation_metrics['mistake_count']}",
                flush=True,
            )
        if exact_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_exact_validation"
            stop_epoch = epoch
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            stop_epoch = epoch
            break
    model.load_state_dict(best_state)
    metrics = _metrics(model, validation_inputs, validation_labels)
    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "training_event_count": len(train_inputs),
        "validation_event_count": len(validation_inputs),
        "learned_w1": float(weight[0]),
        "learned_w2": float(weight[1]),
        "learned_bias": bias,
        "initial_w1": float(initial_weight[0]),
        "initial_w2": float(initial_weight[1]),
        "initial_bias": initial_bias,
        "hidden_layer_count": 0,
        "direct_classification_supervision": True,
        "teacher_coefficients_hardcoded": False,
        "overall_accuracy": metrics["overall_accuracy"],
        "PASS_accuracy": metrics["PASS_accuracy"],
        "FAIL_accuracy": metrics["FAIL_accuracy"],
        "mistake_count": metrics["mistake_count"],
    }
    summary["smoke_gate_passed"] = metrics["mistake_count"] == 0
    return model, [summary], summary


def _learned_po_rows(
    model: nn.Linear,
    pool: dict[str, np.ndarray],
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = _metrics(model, pool["inputs"], pool["teacher_pass"])
    prediction = metrics["prediction"]
    rows: list[dict[str, Any]] = []
    for state_position, state in enumerate(states):
        mask = pool["state_position"] == state_position
        positions = pool["hull_position"][mask]
        state_prediction = prediction[mask]
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = np.asarray(state["sizes"], dtype=np.float64)
        losses = np.asarray(state["losses"], dtype=np.float64)
        selected = [
            int(hull[int(position)])
            for position, passed in zip(positions, state_prediction)
            if bool(passed)
        ]
        fallback_used = not selected
        if fallback_used:
            selected.append(int(np.argmax(sizes)))
        learned_po = np.asarray(
            sorted(selected, key=lambda index: (sizes[index], losses[index], index)),
            dtype=np.int64,
        )
        teacher_po = edirect._potentially_optimal_cells(sizes, losses)
        learned_set = set(int(value) for value in learned_po)
        teacher_set = set(int(value) for value in teacher_po)
        rows.append(
            {
                "record_type": "state",
                "state_position": state_position,
                "source_state_index": int(state["source_state_index"]),
                "representative_cell_count": len(sizes),
                "hull_size": len(hull),
                "teacher_PO_count": len(teacher_po),
                "learned_PO_count": len(learned_po),
                "exact_ordered_PO_set": bool(np.array_equal(learned_po, teacher_po)),
                "exact_unordered_PO_set": learned_set == teacher_set,
                "false_PO_additions": len(learned_set - teacher_set),
                "false_PO_removals": len(teacher_set - learned_set),
                "fallback_used": fallback_used,
            }
        )
    summary = {
        "state_count": len(rows),
        "exact_ordered_PO_states": int(
            sum(bool(row["exact_ordered_PO_set"]) for row in rows)
        ),
        "exact_unordered_PO_states": int(
            sum(bool(row["exact_unordered_PO_set"]) for row in rows)
        ),
        "false_PO_additions": int(sum(int(row["false_PO_additions"]) for row in rows)),
        "false_PO_removals": int(sum(int(row["false_PO_removals"]) for row in rows)),
        "fallback_state_count": int(sum(bool(row["fallback_used"]) for row in rows)),
    }
    summary["exact_PO_gate_passed"] = bool(
        summary["exact_ordered_PO_states"] == len(rows)
        and summary["exact_unordered_PO_states"] == len(rows)
        and summary["false_PO_additions"] == 0
        and summary["false_PO_removals"] == 0
    )
    return rows, summary


def _save_and_lock(
    model: nn.Linear,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    checkpoint = {
        "format_version": 1,
        "component": "improvement_condition_AND",
        "status": "LOCKED_COMPONENT",
        "architecture": "logit=w1*positive_upper+w2*feasible+b; no hidden layers",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_w1": torch.tensor(float(weight[0]), dtype=torch.float64),
        "learned_w2": torch.tensor(float(weight[1]), dtype=torch.float64),
        "learned_bias": torch.tensor(bias, dtype=torch.float64),
        "frozen_slope_sign_sha256": SLOPE_SIGN_SHA256,
        "frozen_feasibility_margin_sha256": FEASIBILITY_MARGIN_SHA256,
        "locked_feasibility_threshold_sha256": FEASIBILITY_THRESHOLD_SHA256,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
        "complete_improvement_condition_operator_locked": True,
        "new_K_upper_positivity_network_trained": False,
    }
    torch.save(checkpoint, AND_PATH)
    restored = torch.load(AND_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("AND checkpoint readback failed")
    checkpoint_hash = _sha256(AND_PATH)
    HASH_PATH.write_text(f"{checkpoint_hash}  {AND_PATH.name}\n", encoding="utf-8")
    lock_rows = [
        {
            "component": "neighboring_hull_slope_sign",
            "status": "LOCKED_UNCHANGED",
            "checkpoint": str(SLOPE_SIGN_PATH),
            "checkpoint_sha256": SLOPE_SIGN_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "feasibility_operator",
            "status": "LOCKED_UNCHANGED",
            "margin_checkpoint_sha256": FEASIBILITY_MARGIN_SHA256,
            "threshold_checkpoint_sha256": FEASIBILITY_THRESHOLD_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "improvement_condition_AND",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(AND_PATH),
            "checkpoint_sha256": checkpoint_hash,
            "learned_w1": float(weight[0]),
            "learned_w2": float(weight[1]),
            "learned_bias": bias,
        },
        {
            "component": "improvement_condition_operator",
            "status": "LOCKED",
            "assembled_formula": "locked slope-sign K_upper positivity AND locked feasibility output -> learned affine Boolean AND",
            "PASS_FAIL_mistakes": full_summary["mistake_count"],
            "exact_ordered_PO_states": full_summary["exact_ordered_PO_states"],
            "state_count": full_summary["state_count"],
            "new_K_upper_positivity_network_trained": False,
        },
    ]
    return checkpoint_hash, lock_rows


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
        payload, train_indices, workers, "improvement AND train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "improvement AND validation"
    )
    training = _build_pool(train_states, "train")
    validation = _build_pool(validation_states, "validation")
    audit_rows = _combination_rows(training, "training")
    audit_rows.extend(_combination_rows(validation, "held_out"))
    _write_csv(AUDIT_PATH, audit_rows)

    train_selected, train_balance = _balanced_indices(
        training["teacher_pass"], MODEL_SEED + 1
    )
    validation_selected, validation_balance = _balanced_indices(
        validation["teacher_pass"], MODEL_SEED + 2
    )
    model, smoke_rows, smoke_summary = _train(
        training["inputs"][train_selected],
        training["teacher_pass"][train_selected],
        validation["inputs"][validation_selected],
        validation["teacher_pass"][validation_selected],
    )
    smoke_rows.extend(
        _mistake_rows(
            model,
            validation,
            validation["inputs"][validation_selected],
            validation["teacher_pass"][validation_selected],
            "balanced_validation",
        )
    )
    smoke_rows.extend(
        (
            {"record_type": "balanced_dataset", "split": "train", **train_balance},
            {
                "record_type": "balanced_dataset",
                "split": "validation",
                **validation_balance,
            },
        )
    )
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [{"component": "improvement_condition_operator", "status": "NOT_LOCKED_SMOKE_GATE_FAILED"}],
        )
        print("improvement AND smoke gate failed; stopped", flush=True)
        return

    full_metrics = _metrics(model, validation["inputs"], validation["teacher_pass"])
    po_rows, po_summary = _learned_po_rows(model, validation, validation_states)
    full_summary = {
        "record_type": "final_summary",
        "held_out_event_count": len(validation["teacher_pass"]),
        "overall_accuracy": full_metrics["overall_accuracy"],
        "PASS_accuracy": full_metrics["PASS_accuracy"],
        "FAIL_accuracy": full_metrics["FAIL_accuracy"],
        "mistake_count": full_metrics["mistake_count"],
        **po_summary,
        "locked_slope_sign_sha256": SLOPE_SIGN_SHA256,
        "locked_feasibility_margin_sha256": FEASIBILITY_MARGIN_SHA256,
        "locked_feasibility_threshold_sha256": FEASIBILITY_THRESHOLD_SHA256,
    }
    full_summary["full_pool_gate_passed"] = bool(
        full_summary["mistake_count"] == 0
        and full_summary["exact_PO_gate_passed"]
    )
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(po_rows)
    full_rows.extend(
        _mistake_rows(
            model,
            validation,
            validation["inputs"],
            validation["teacher_pass"],
            "full_held_out_pool",
            np.arange(len(validation["teacher_pass"]), dtype=np.int64),
        )
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        _write_csv(
            LOCK_SUMMARY_PATH,
            [{"component": "improvement_condition_operator", "status": "NOT_LOCKED_FULL_POOL_GATE_FAILED"}],
        )
        print("improvement AND full-pool gate failed; stopped", flush=True)
        return

    checkpoint_hash, lock_rows = _save_and_lock(model, smoke_summary, full_summary)
    full_summary["AND_checkpoint"] = str(AND_PATH)
    full_summary["AND_checkpoint_sha256"] = checkpoint_hash
    full_summary["improvement_condition_operator_status"] = "LOCKED"
    _write_csv(FULL_PATH, full_rows)
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    weight = model.weight.detach().cpu().numpy()[0]
    bias = float(model.bias.detach().item())
    print(
        f"improvement condition LOCKED w1={weight[0]:.17g} "
        f"w2={weight[1]:.17g} b={bias:.17g} sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
