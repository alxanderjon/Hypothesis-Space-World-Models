#!/usr/bin/env python3
"""Learn only the final monotone winner threshold for K_lower."""

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
import experiments.sho.run_stage1b_k_lower_margin as margin


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/k_lower_threshold")
SMOKE_PATH = OUTPUT_DIR / "k_lower_threshold_smoke.csv"
FULL_PATH = OUTPUT_DIR / "k_lower_threshold_full_pool.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "k_lower_operator_lock_summary.csv"
THRESHOLD_PATH = OUTPUT_DIR / "k_lower_threshold_locked.pt"
HASH_PATH = THRESHOLD_PATH.with_suffix(".sha256")

MARGIN_PATH = margin.CANDIDATE_PATH
MARGIN_SHA256 = (
    "b7bf8791d6f62dd7672b4a6a0dd3066a576b3422248f244fae9f4fbbc4dc31ab"
)

MODEL_SEED = smoke.MODEL_SEED + 6_283_185
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
    margin._verify_frozen_hashes()
    if not MARGIN_PATH.exists():
        raise FileNotFoundError(f"missing frozen comparison margin: {MARGIN_PATH}")
    actual = _sha256(MARGIN_PATH)
    if actual != MARGIN_SHA256:
        raise RuntimeError(f"frozen comparison-margin hash mismatch: {actual}")


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
        raise RuntimeError("comparison-margin checkpoint is not frozen candidate")
    if float(checkpoint["bias"].item()) != 0.0:
        raise RuntimeError("comparison-margin checkpoint bias is not zero")
    return margin._checkpoint_linear_predict(
        checkpoint, pool["margin_inputs"]
    )


def _audit_rows(
    pool: dict[str, np.ndarray], frozen_margin: np.ndarray, split: str
) -> list[dict[str, Any]]:
    teacher_prev = pool["margin_target"] > 0.0
    teacher_required = pool["margin_target"] < 0.0
    zero = pool["margin_target"] == 0.0
    positive_values = frozen_margin[teacher_prev]
    negative_values = frozen_margin[teacher_required]
    rows: list[dict[str, Any]] = [
        {
            "record_type": "audit_summary",
            "split": split,
            "total_improvement_events": len(pool["A"]),
            "positive_branch_events": len(frozen_margin),
            "K_prev_winner_events": int(np.sum(teacher_prev)),
            "K_required_winner_events": int(np.sum(teacher_required)),
            "exact_zero_teacher_margin_events": int(np.sum(zero)),
            "closest_positive_frozen_margin_to_zero": (
                float(np.min(positive_values)) if len(positive_values) else ""
            ),
            "closest_negative_frozen_margin_to_zero": (
                float(np.max(negative_values)) if len(negative_values) else ""
            ),
            "closest_positive_exact_margin_to_zero": (
                float(np.min(pool["margin_target"][teacher_prev]))
                if np.any(teacher_prev)
                else ""
            ),
            "closest_negative_exact_margin_to_zero": (
                float(np.max(pool["margin_target"][teacher_required]))
                if np.any(teacher_required)
                else ""
            ),
            "automatic_K_prev_nonpositive_or_endpoint_events": int(
                np.sum(pool["automatic_mask"])
            ),
            "automatic_teacher_K_required_correct": bool(
                np.all(pool["automatic_teacher_required"])
            ),
            "minority_examples_available": int(np.sum(teacher_required)),
            "minority_data_limitation": bool(np.sum(teacher_required) < 100),
            "frozen_margin_upstream_winner_mistakes": int(
                np.sum((frozen_margin >= 0.0) != (pool["margin_target"] >= 0.0))
            ),
        }
    ]
    for label, values in (
        ("K_prev_winner_frozen_margin", positive_values),
        ("K_required_winner_frozen_margin", negative_values),
    ):
        for percentile in (0, 1, 5, 25, 50, 75, 95, 99, 100):
            if len(values):
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


def _balanced_real_indices(
    target: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    previous = np.flatnonzero(target > 0.0)
    required = np.flatnonzero(target < 0.0)
    zeros = np.flatnonzero(target == 0.0)
    if len(zeros):
        raise RuntimeError("unexpected exact-zero comparison margin")
    minority = min(len(previous), len(required))
    if minority == 0:
        raise RuntimeError("cannot form real balanced threshold dataset")
    generator = np.random.default_rng(seed)
    previous_selected = generator.choice(previous, size=minority, replace=False)
    required_selected = generator.choice(required, size=minority, replace=False)
    selected = np.concatenate((previous_selected, required_selected))
    generator.shuffle(selected)
    return selected, {
        "raw_K_prev_winner_count": len(previous),
        "raw_K_required_winner_count": len(required),
        "selected_K_prev_winner_count": minority,
        "selected_K_required_winner_count": minority,
        "selected_event_count": 2 * minority,
        "all_real_minority_examples_used": minority == len(required),
        "synthetic_examples_used": False,
    }


class MonotoneThreshold(nn.Module):
    """logit = alpha * (m - theta), alpha > 0."""

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
    labels = labels.astype(bool)
    previous = labels
    required = ~labels
    mistakes = prediction != labels
    return {
        "overall_accuracy": float(np.mean(~mistakes)),
        "K_prev_win_accuracy": float(
            np.mean(~mistakes[previous]) if np.any(previous) else 1.0
        ),
        "K_required_win_accuracy": float(
            np.mean(~mistakes[required]) if np.any(required) else 1.0
        ),
        "mistake_count": int(np.sum(mistakes)),
        "learned_theta": theta,
        "learned_alpha": alpha,
        "absolute_theta_distance_from_teacher_zero": abs(theta),
    }


def _mistake_rows(
    values: np.ndarray,
    exact_margin: np.ndarray,
    labels: np.ndarray,
    theta: float,
    alpha: float,
    split: str,
) -> list[dict[str, Any]]:
    prediction = alpha * (values - theta) >= 0.0
    truth = labels.astype(bool)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(prediction != truth):
        rows.append(
            {
                "record_type": "mistake",
                "split": split,
                "event_index": int(index),
                "teacher_winner": "K_prev" if truth[index] else "K_required",
                "predicted_winner": (
                    "K_prev" if prediction[index] else "K_required"
                ),
                "frozen_comparison_margin": float(values[index]),
                "exact_teacher_margin": float(exact_margin[index]),
                "learned_theta": theta,
                "distance_from_learned_threshold": float(
                    abs(values[index] - theta)
                ),
                "distance_from_teacher_zero": float(abs(values[index])),
            }
        )
    return rows


def _train_threshold(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_exact_margin: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[MonotoneThreshold, list[dict[str, Any]], dict[str, Any]]:
    nonzero = np.abs(train_values[train_values != 0.0])
    if not len(nonzero):
        raise RuntimeError("all training margins are zero")
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
    stopping_reason = "maximum_epochs"
    stop_epoch = MAX_EPOCHS
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
        metrics = _metrics(
            validation_values, validation_labels, theta, alpha
        )
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
                f"Klower threshold epoch={epoch} theta={theta:.6e} "
                f"alpha={alpha:.6e} val_mistakes={metrics['mistake_count']}",
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
            validation_exact_margin,
            validation_labels,
            theta,
            alpha,
            "balanced_validation",
        )
    )
    return model, rows, summary


def _full_assembled_metrics(
    pool: dict[str, np.ndarray],
    frozen_values: np.ndarray,
    theta: float,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positive_mask = pool["positive_mask"]
    automatic_mask = pool["automatic_mask"]
    learned_prev_positive = alpha * (frozen_values - theta) >= 0.0
    truth_prev_positive = pool["margin_target"] >= 0.0

    learned_winner = np.full(len(pool["A"]), "K_required", dtype=object)
    teacher_winner = pool["teacher_winner"]
    learned_winner[positive_mask] = np.where(
        learned_prev_positive, "K_prev", "K_required"
    )

    positive_indices = np.flatnonzero(positive_mask)
    candidate_lower = np.empty(len(pool["A"]), dtype=np.float64)
    expected_frozen_lower = np.empty(len(pool["A"]), dtype=np.float64)
    locked_required = np.exp(pool["locked_log_required"])
    locked_previous = np.zeros(len(pool["A"]), dtype=np.float64)
    locked_previous[positive_mask] = np.exp(pool["locked_log_prev"][positive_mask])
    candidate_lower[automatic_mask] = locked_required[automatic_mask]
    expected_frozen_lower[automatic_mask] = locked_required[automatic_mask]
    candidate_lower[positive_indices] = np.where(
        learned_prev_positive,
        locked_previous[positive_mask],
        locked_required[positive_mask],
    )
    expected_frozen_lower[positive_indices] = np.where(
        truth_prev_positive,
        locked_previous[positive_mask],
        locked_required[positive_mask],
    )

    learned_next = np.full(len(pool["A"]), math.inf, dtype=np.float64)
    finite_next = np.isfinite(pool["exact_K_next"])
    learned_next[finite_next] = (
        pool["locked_sign_next"][finite_next]
        * np.exp(pool["locked_log_next"][finite_next])
    )
    raw_order = candidate_lower <= learned_next
    feasibility = candidate_lower <= learned_next + turn.HULL_TOLERANCE
    passed = (learned_next > 0.0) & feasibility

    positive_prediction = learned_prev_positive
    positive_truth = truth_prev_positive
    metrics = {
        "held_out_total_event_count": len(pool["A"]),
        "positive_branch_event_count": len(frozen_values),
        "automatic_routing_event_count": int(np.sum(automatic_mask)),
        "positive_branch_winner_mistakes": int(
            np.sum(positive_prediction != positive_truth)
        ),
        "automatic_routing_mistakes": int(
            np.sum(learned_winner[automatic_mask] != "K_required")
        ),
        "winner_mistakes": int(np.sum(learned_winner != teacher_winner)),
        "K_lower_value_mismatches": int(
            np.sum(candidate_lower != expected_frozen_lower)
        ),
        "K_lower_K_upper_ordering_changes": int(
            np.sum(raw_order != pool["exact_raw_order"])
        ),
        "feasibility_changes": int(
            np.sum(feasibility != pool["exact_feasibility"])
        ),
        "final_improvement_PASS_FAIL_changes": int(
            np.sum(passed != pool["exact_pass"])
        ),
    }
    metrics["full_assembled_gate_passed"] = all(
        metrics[key] == 0
        for key in (
            "positive_branch_winner_mistakes",
            "automatic_routing_mistakes",
            "winner_mistakes",
            "K_lower_value_mismatches",
            "K_lower_K_upper_ordering_changes",
            "feasibility_changes",
            "final_improvement_PASS_FAIL_changes",
        )
    )
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(positive_prediction != positive_truth):
        event_index = int(positive_indices[local_index])
        rows.append(
            {
                "record_type": "winner_mistake",
                "event_index": event_index,
                "source_state_index": int(pool["source_state_index"][event_index]),
                "hull_position": int(pool["hull_position"][event_index]),
                "frozen_comparison_margin": float(frozen_values[local_index]),
                "exact_teacher_margin": float(pool["margin_target"][local_index]),
                "learned_theta": theta,
            }
        )
    return metrics, rows


def _save_and_lock(
    model: MonotoneThreshold,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    checkpoint = {
        "format_version": 1,
        "component": "K_lower_winner_threshold",
        "status": "LOCKED_COMPONENT",
        "architecture": "logit=softplus(raw_alpha)/scale*(m-theta)",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_theta": torch.tensor(theta, dtype=torch.float64),
        "learned_alpha": torch.tensor(alpha, dtype=torch.float64),
        "hidden_layer_count": 0,
        "frozen_comparison_margin_sha256": MARGIN_SHA256,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
        "complete_K_lower_operator_locked": True,
    }
    torch.save(checkpoint, THRESHOLD_PATH)
    restored = torch.load(THRESHOLD_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("threshold checkpoint readback failed")
    checkpoint_hash = _sha256(THRESHOLD_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {THRESHOLD_PATH.name}\n", encoding="utf-8"
    )
    rows = [
        {
            "component": "neighboring_hull_slope_operator",
            "status": "LOCKED_UNCHANGED",
            "log_magnitude_sha256": margin.SLOPE_LOGMAG_SHA256,
            "sign_sha256": margin.SLOPE_SIGN_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "K_required_operator",
            "status": "LOCKED_UNCHANGED",
            "numerator_sha256": margin.NUMERATOR_SHA256,
            "log_quotient_sha256": margin.LOGQUOTIENT_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "K_lower_comparison_margin",
            "status": "FROZEN_CANDIDATE_UNCHANGED",
            "checkpoint": str(MARGIN_PATH),
            "checkpoint_sha256": MARGIN_SHA256,
            "retrained_or_modified": False,
        },
        {
            "component": "K_lower_winner_threshold",
            "status": "LOCKED_COMPONENT",
            "checkpoint": str(THRESHOLD_PATH),
            "checkpoint_sha256": checkpoint_hash,
            "learned_theta": theta,
            "learned_alpha": alpha,
        },
        {
            "component": "K_lower_operator",
            "status": "LOCKED",
            "assembled_formula": "automatic K_required for K_prev<=0/sentinel; otherwise learned threshold(frozen margin) selects frozen K_prev or K_required",
            "winner_mistakes": full_summary["winner_mistakes"],
            "K_lower_value_mismatches": full_summary["K_lower_value_mismatches"],
            "ordering_changes": full_summary["K_lower_K_upper_ordering_changes"],
            "feasibility_changes": full_summary["feasibility_changes"],
            "PASS_FAIL_changes": full_summary[
                "final_improvement_PASS_FAIL_changes"
            ],
        },
    ]
    return checkpoint_hash, rows


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
        payload, train_indices, workers, "Klower threshold train"
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "Klower threshold validation"
    )
    training = margin._locked_upstream_pool(train_states, "train")
    validation = margin._locked_upstream_pool(validation_states, "validation")
    train_margin = _frozen_margin(training)
    validation_margin = _frozen_margin(validation)

    train_selected, train_balance = _balanced_real_indices(
        training["margin_target"], MODEL_SEED + 1
    )
    validation_selected, validation_balance = _balanced_real_indices(
        validation["margin_target"], MODEL_SEED + 2
    )
    train_labels = training["margin_target"][train_selected] > 0.0
    validation_labels = validation["margin_target"][validation_selected] > 0.0

    model, smoke_rows, smoke_summary = _train_threshold(
        train_margin[train_selected],
        train_labels,
        validation_margin[validation_selected],
        validation["margin_target"][validation_selected],
        validation_labels,
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
    smoke_summary["frozen_comparison_margin_sha256"] = MARGIN_SHA256
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "K_lower_operator",
                    "status": "NOT_LOCKED_SMOKE_GATE_FAILED",
                }
            ],
        )
        print("K_lower threshold smoke gate failed; stopped", flush=True)
        return

    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    metrics, detail_rows = _full_assembled_metrics(
        validation, validation_margin, theta, alpha
    )
    full_summary = {
        "record_type": "final_summary",
        "learned_theta": theta,
        "learned_alpha": alpha,
        **metrics,
        "frozen_comparison_margin_sha256": MARGIN_SHA256,
    }
    full_rows = [full_summary] + detail_rows
    if not bool(full_summary["full_assembled_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        _write_csv(
            LOCK_SUMMARY_PATH,
            [
                {
                    "component": "K_lower_operator",
                    "status": "NOT_LOCKED_FULL_POOL_GATE_FAILED",
                }
            ],
        )
        print("K_lower threshold full assembled gate failed; stopped", flush=True)
        return

    checkpoint_hash, lock_rows = _save_and_lock(
        model, smoke_summary, full_summary
    )
    full_summary["threshold_checkpoint"] = str(THRESHOLD_PATH)
    full_summary["threshold_checkpoint_sha256"] = checkpoint_hash
    full_summary["K_lower_operator_status"] = "LOCKED"
    _write_csv(FULL_PATH, full_rows)
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        f"K_lower operator LOCKED theta={theta:.17g} alpha={alpha:.17g} "
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
