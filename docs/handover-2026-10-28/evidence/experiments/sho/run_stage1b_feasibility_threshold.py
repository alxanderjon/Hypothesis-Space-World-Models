#!/usr/bin/env python3
"""Learn only the final monotone threshold for normalized feasibility margin."""

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

import experiments.sho.run_stage1b_feasibility_margin as feasibility_margin
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/feasibility_threshold")
SMOKE_PATH = OUTPUT_DIR / "feasibility_threshold_smoke.csv"
FULL_PATH = OUTPUT_DIR / "feasibility_threshold_full_pool.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "feasibility_operator_lock_summary.csv"
THRESHOLD_PATH = OUTPUT_DIR / "feasibility_threshold_locked.pt"
HASH_PATH = THRESHOLD_PATH.with_suffix(".sha256")

MARGIN_PATH = feasibility_margin.CANDIDATE_PATH
MARGIN_SHA256 = (
    "b777e69b38923da34b7e513ff47fdbb0324d04d3982bce3b78ff5d2c9c14655a"
)

MODEL_SEED = smoke.MODEL_SEED + 8_164_965
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
    feasibility_margin._verify_frozen_hashes()
    if not MARGIN_PATH.exists():
        raise FileNotFoundError(f"missing feasibility-margin candidate: {MARGIN_PATH}")
    actual = _sha256(MARGIN_PATH)
    if actual != MARGIN_SHA256:
        raise RuntimeError(f"feasibility-margin hash mismatch: {actual}")


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
    checkpoint = torch.load(MARGIN_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("feasibility-margin checkpoint is not frozen")
    if float(checkpoint["bias"].item()) != 0.0:
        raise RuntimeError("feasibility-margin checkpoint has nonzero bias")
    return feasibility_margin.lower_margin._checkpoint_linear_predict(
        checkpoint, pool["inputs"]
    )


def _audit_rows(
    pool: dict[str, np.ndarray], learned_margin: np.ndarray, split: str
) -> list[dict[str, Any]]:
    feasible = pool["teacher_feasible"]
    infeasible = ~feasible
    positive = learned_margin[feasible & (learned_margin > 0.0)]
    negative = learned_margin[infeasible & (learned_margin < 0.0)]
    rows: list[dict[str, Any]] = [
        {
            "record_type": "audit_summary",
            "split": split,
            "total_event_count": len(pool["upstream"]["A"]),
            "finite_threshold_event_count": len(learned_margin),
            "feasible_count": int(np.sum(feasible)),
            "infeasible_count": int(np.sum(infeasible)),
            "closest_positive_frozen_y_feas_to_zero": (
                float(np.min(positive)) if len(positive) else ""
            ),
            "closest_negative_frozen_y_feas_to_zero": (
                float(np.max(negative)) if len(negative) else ""
            ),
            "exact_zero_frozen_y_feas_count": int(np.sum(learned_margin == 0.0)),
            "exact_zero_teacher_y_feas_count": int(np.sum(pool["target"] == 0.0)),
            "frozen_margin_sign_mistakes": int(
                np.sum((learned_margin >= 0.0) != feasible)
            ),
            "positive_infinity_endpoint_count": len(pool["endpoint_event_indices"]),
            "endpoints_automatically_feasible": True,
            "endpoints_in_threshold_training": False,
        }
    ]
    for label, values in (
        ("feasible_frozen_y_feas", learned_margin[feasible]),
        ("infeasible_frozen_y_feas", learned_margin[infeasible]),
    ):
        for percentile in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            rows.append(
                {
                    "record_type": "margin_percentile",
                    "split": split,
                    "quantity": label,
                    "percentile": percentile,
                    "value": float(np.percentile(values, percentile)),
                    "event_count": len(values),
                }
            )
    return rows


def _balanced_indices(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    feasible = np.flatnonzero(labels)
    infeasible = np.flatnonzero(~labels)
    count = min(len(feasible), len(infeasible))
    if count == 0:
        raise RuntimeError("cannot form real balanced feasibility dataset")
    generator = np.random.default_rng(seed)
    selected_feasible = generator.choice(feasible, size=count, replace=False)
    selected_infeasible = generator.choice(infeasible, size=count, replace=False)
    selected = np.concatenate((selected_feasible, selected_infeasible))
    generator.shuffle(selected)
    return selected, {
        "raw_feasible_count": len(feasible),
        "raw_infeasible_count": len(infeasible),
        "selected_feasible_count": count,
        "selected_infeasible_count": count,
        "selected_event_count": 2 * count,
        "all_real_minority_examples_used": count == min(len(feasible), len(infeasible)),
        "synthetic_examples_used": False,
    }


class MonotoneThreshold(nn.Module):
    """logit = alpha * (y_feas - theta), alpha > 0."""

    def __init__(self, threshold_scale: float) -> None:
        super().__init__()
        if not threshold_scale > 0.0 or not math.isfinite(threshold_scale):
            raise RuntimeError("invalid threshold scale")
        self.register_buffer(
            "threshold_scale", torch.tensor(threshold_scale, dtype=torch.float64)
        )
        self.raw_alpha = nn.Parameter(torch.empty((), dtype=torch.float64))
        self.scaled_theta = nn.Parameter(torch.empty((), dtype=torch.float64))
        nn.init.normal_(self.raw_alpha, mean=0.0, std=0.35)
        nn.init.normal_(self.scaled_theta, mean=0.0, std=0.35)
        if float(self.theta().detach()) == 0.0:
            raise RuntimeError("forbidden exact-zero theta initialization")

    def alpha(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_alpha) / self.threshold_scale

    def theta(self) -> torch.Tensor:
        return self.scaled_theta * self.threshold_scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.alpha() * (value - self.theta())


def _metrics(
    values: np.ndarray, labels: np.ndarray, theta: float, alpha: float
) -> dict[str, Any]:
    prediction = alpha * (values - theta) >= 0.0
    mistakes = prediction != labels
    return {
        "overall_accuracy": float(np.mean(~mistakes)),
        "feasible_accuracy": float(
            np.mean(~mistakes[labels]) if np.any(labels) else 1.0
        ),
        "infeasible_accuracy": float(
            np.mean(~mistakes[~labels]) if np.any(~labels) else 1.0
        ),
        "mistake_count": int(np.sum(mistakes)),
        "learned_theta": theta,
        "learned_alpha": alpha,
        "absolute_theta_distance_from_teacher_zero": abs(theta),
    }


def _mistake_rows(
    values: np.ndarray,
    exact_values: np.ndarray,
    labels: np.ndarray,
    theta: float,
    alpha: float,
    split: str,
    event_indices: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    prediction = alpha * (values - theta) >= 0.0
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(prediction != labels):
        rows.append(
            {
                "record_type": "threshold_mistake",
                "split": split,
                "event_index": (
                    int(event_indices[index]) if event_indices is not None else int(index)
                ),
                "teacher_class": "FEASIBLE" if labels[index] else "INFEASIBLE",
                "predicted_class": (
                    "FEASIBLE" if prediction[index] else "INFEASIBLE"
                ),
                "frozen_y_feas": float(values[index]),
                "exact_teacher_y_feas": float(exact_values[index]),
                "learned_theta": theta,
                "distance_from_teacher_zero": float(abs(values[index])),
                "distance_from_learned_threshold": float(
                    abs(values[index] - theta)
                ),
            }
        )
    return rows


def _train(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_exact: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[MonotoneThreshold, list[dict[str, Any]], dict[str, Any]]:
    nonzero = np.abs(train_values[train_values != 0.0])
    if not len(nonzero):
        raise RuntimeError("all feasibility training margins are zero")
    threshold_scale = float(np.min(nonzero))
    torch.manual_seed(MODEL_SEED)
    model = MonotoneThreshold(threshold_scale).to(torch.float64)
    initial_theta = float(model.theta().detach().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x = torch.as_tensor(train_values, dtype=torch.float64)
    train_y = torch.as_tensor(train_labels.astype(np.float64), dtype=torch.float64)
    valid_x = torch.as_tensor(validation_values, dtype=torch.float64)
    valid_y = torch.as_tensor(
        validation_labels.astype(np.float64), dtype=torch.float64
    )
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    stale = 0
    exact_streak = 0
    stop_epoch = MAX_EPOCHS
    stopping_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_logits = model(valid_x)
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, valid_y
                ).item()
            )
            theta = float(model.theta().item())
            alpha = float(model.alpha().item())
        metrics = _metrics(validation_values, validation_labels, theta, alpha)
        key = (metrics["mistake_count"], validation_loss)
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        exact_streak = exact_streak + 1 if metrics["mistake_count"] == 0 else 0
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"feas threshold epoch={epoch} theta={theta:.6e} "
                f"alpha={alpha:.6e} mistakes={metrics['mistake_count']}",
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
    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    metrics = _metrics(validation_values, validation_labels, theta, alpha)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "training_event_count": len(train_values),
        "validation_event_count": len(validation_values),
        "threshold_parameter_scale": threshold_scale,
        "random_initial_theta": initial_theta,
        "initial_theta_was_exactly_zero": initial_theta == 0.0,
        "positive_alpha_constraint": "softplus(raw_alpha)",
        "hidden_layer_count": 0,
        "direct_classification_supervision": True,
        **metrics,
    }
    summary["smoke_gate_passed"] = metrics["mistake_count"] == 0
    rows = [summary]
    rows.extend(
        _mistake_rows(
            validation_values,
            validation_exact,
            validation_labels,
            theta,
            alpha,
            "balanced_validation",
        )
    )
    return model, rows, summary


def _full_check(
    pool: dict[str, np.ndarray],
    learned_margin: np.ndarray,
    theta: float,
    alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finite_feasible = alpha * (learned_margin - theta) >= 0.0
    finite_metrics = _metrics(
        learned_margin, pool["teacher_feasible"], theta, alpha
    )
    all_feasible = np.ones(len(pool["upstream"]["A"]), dtype=bool)
    all_feasible[pool["finite_event_indices"]] = finite_feasible
    locked_upper = pool["all_locked_K_upper"]
    final_pass = (locked_upper > 0.0) & all_feasible
    final_changes = final_pass != pool["upstream"]["exact_pass"]
    summary = {
        "record_type": "final_summary",
        "held_out_total_event_count": len(pool["upstream"]["A"]),
        "held_out_finite_event_count": len(learned_margin),
        "held_out_infinite_endpoint_count": len(pool["endpoint_event_indices"]),
        **finite_metrics,
        "endpoint_automatic_feasibility_mistakes": int(
            np.sum(~all_feasible[pool["endpoint_event_indices"]])
        ),
        "final_improvement_PASS_FAIL_changes": int(np.sum(final_changes)),
        "deterministic_K_upper_positive_condition_used": True,
        "K_upper_positive_condition_learned": False,
        "final_AND_learned": False,
        "frozen_feasibility_margin_sha256": MARGIN_SHA256,
    }
    summary["full_pool_gate_passed"] = bool(
        summary["mistake_count"] == 0
        and summary["endpoint_automatic_feasibility_mistakes"] == 0
        and summary["final_improvement_PASS_FAIL_changes"] == 0
    )
    rows = [summary]
    rows.extend(
        _mistake_rows(
            learned_margin,
            pool["target"],
            pool["teacher_feasible"],
            theta,
            alpha,
            "full_held_out_pool",
            pool["finite_event_indices"],
        )
    )
    return rows, summary


def _save_and_lock(
    model: MonotoneThreshold,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    checkpoint = {
        "format_version": 1,
        "component": "feasibility_threshold",
        "status": "LOCKED_COMPONENT",
        "architecture": "logit=softplus(raw_alpha)/scale*(y_feas-theta)",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_theta": torch.tensor(theta, dtype=torch.float64),
        "learned_alpha": torch.tensor(alpha, dtype=torch.float64),
        "hidden_layer_count": 0,
        "frozen_feasibility_margin_sha256": MARGIN_SHA256,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
        "complete_feasibility_operator_locked": True,
        "K_upper_positive_condition_learned": False,
        "final_AND_learned": False,
    }
    torch.save(checkpoint, THRESHOLD_PATH)
    restored = torch.load(THRESHOLD_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("feasibility-threshold checkpoint readback failed")
    checkpoint_hash = _sha256(THRESHOLD_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {THRESHOLD_PATH.name}\n", encoding="utf-8"
    )
    lock_rows = [
        {
            "component": "feasibility_margin",
            "status": "FROZEN_CANDIDATE_UNCHANGED",
            "checkpoint": str(MARGIN_PATH),
            "checkpoint_sha256": MARGIN_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "feasibility_threshold",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(THRESHOLD_PATH),
            "checkpoint_sha256": checkpoint_hash,
            "learned_theta": theta,
            "learned_alpha": alpha,
        },
        {
            "component": "feasibility_operator",
            "status": "LOCKED",
            "assembled_formula": "frozen normalized feasibility margin -> learned threshold; +inf endpoints automatically FEASIBLE",
            "threshold_mistakes": full_summary["mistake_count"],
            "endpoint_mistakes": full_summary[
                "endpoint_automatic_feasibility_mistakes"
            ],
            "final_PASS_FAIL_changes_with_deterministic_K_upper_positive": full_summary[
                "final_improvement_PASS_FAIL_changes"
            ],
            "K_upper_positive_condition_learned": False,
            "final_AND_learned": False,
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
        payload, train_indices, workers, "feasibility threshold train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "feasibility threshold validation"
    )
    training = feasibility_margin._build_pool(train_states, "train")
    validation = feasibility_margin._build_pool(validation_states, "validation")
    train_margin = _frozen_margin(training)
    validation_margin = _frozen_margin(validation)

    train_selected, train_balance = _balanced_indices(
        training["teacher_feasible"], MODEL_SEED + 1
    )
    validation_selected, validation_balance = _balanced_indices(
        validation["teacher_feasible"], MODEL_SEED + 2
    )
    model, smoke_rows, smoke_summary = _train(
        train_margin[train_selected],
        training["teacher_feasible"][train_selected],
        validation_margin[validation_selected],
        validation["target"][validation_selected],
        validation["teacher_feasible"][validation_selected],
    )
    smoke_rows.extend(_audit_rows(training, train_margin, "training_raw"))
    smoke_rows.extend(_audit_rows(validation, validation_margin, "held_out_raw"))
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
    smoke_summary["frozen_feasibility_margin_sha256"] = MARGIN_SHA256
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [{"component": "feasibility_operator", "status": "NOT_LOCKED_SMOKE_GATE_FAILED"}],
        )
        print("feasibility-threshold smoke gate failed; stopped", flush=True)
        return

    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    full_rows, full_summary = _full_check(
        validation, validation_margin, theta, alpha
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        _write_csv(
            LOCK_SUMMARY_PATH,
            [{"component": "feasibility_operator", "status": "NOT_LOCKED_FULL_POOL_GATE_FAILED"}],
        )
        print("feasibility-threshold full-pool gate failed; stopped", flush=True)
        return

    checkpoint_hash, lock_rows = _save_and_lock(model, smoke_summary, full_summary)
    full_summary["threshold_checkpoint"] = str(THRESHOLD_PATH)
    full_summary["threshold_checkpoint_sha256"] = checkpoint_hash
    full_summary["feasibility_operator_status"] = "LOCKED"
    _write_csv(FULL_PATH, full_rows)
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        f"feasibility operator LOCKED theta={theta:.17g} alpha={alpha:.17g} "
        f"threshold_sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
