#!/usr/bin/env python3
"""Train only the SAME/NEW threshold over the frozen same-size margin."""

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

import experiments.sho.audit_stage1b_same_size_grouping as grouping_audit
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_same_size_margin as margin_data


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/same_size_grouping_threshold")
BOUNDARY_PATH = OUTPUT_DIR / "same_size_threshold_boundary.csv"
FULL_PATH = OUTPUT_DIR / "same_size_threshold_full_pool.csv"
THRESHOLD_PATH = OUTPUT_DIR / "same_size_threshold_locked.pt"
HASH_PATH = THRESHOLD_PATH.with_suffix(".sha256")
LOCK_SUMMARY_PATH = OUTPUT_DIR / "same_size_grouping_operator_lock_summary.csv"

FROZEN_MARGIN_PATH = margin_data.CANDIDATE_PATH
FROZEN_MARGIN_SHA256 = (
    "4521a50534443f740f354903764de68fcbb1d6df209f93eb171d92f5b7ba3ac3"
)
MODEL_SEED = margin_data.MODEL_SEED + 1_000_003
BOUNDARY_SCALE = 1.0e-24
SYNTHETIC_MAGNITUDES = np.asarray(
    [1.0e-24, 1.0e-20, 1.0e-16, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3, 1.0e-1, 1.0],
    dtype=np.float64,
)
REAL_PER_CLASS = 5_000
VALIDATION_PER_CLASS = 1_000
LEARNING_RATE = 5.0e-2
MAX_EPOCHS = 10_000
SUCCESS_STABILITY_EPOCHS = 100
CLEAR_LOGIT_MARGIN = 1.0


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


class MonotoneThreshold(nn.Module):
    """Physical form: logit = alpha * (margin - theta), alpha > 0."""

    def __init__(self) -> None:
        super().__init__()
        self.theta_scaled = nn.Parameter(torch.randn((), dtype=torch.float64))
        self.raw_alpha_scaled = nn.Parameter(torch.randn((), dtype=torch.float64))

    def alpha_scaled(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_alpha_scaled)

    def theta(self) -> torch.Tensor:
        return BOUNDARY_SCALE * self.theta_scaled

    def alpha(self) -> torch.Tensor:
        return self.alpha_scaled() / BOUNDARY_SCALE

    def forward(self, margin: torch.Tensor) -> torch.Tensor:
        return self.alpha_scaled() * (
            margin / BOUNDARY_SCALE - self.theta_scaled
        )


def _load_frozen_margin() -> tuple[nn.Linear, dict[str, Any]]:
    if _sha256(FROZEN_MARGIN_PATH) != FROZEN_MARGIN_SHA256:
        raise RuntimeError("frozen same-size margin hash mismatch")
    checkpoint = torch.load(FROZEN_MARGIN_PATH, map_location="cpu", weights_only=False)
    if checkpoint.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("same-size margin checkpoint is not frozen")
    model = nn.Linear(1, 1, bias=True).to(torch.float64)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _frozen_margins(model: nn.Linear, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            model(torch.as_tensor(x[:, None], dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )


def _boundary_values() -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        (-SYNTHETIC_MAGNITUDES[::-1], np.asarray([0.0]), SYNTHETIC_MAGNITUDES)
    )
    labels = values >= 0.0
    return values, labels


def _dense_grid() -> tuple[np.ndarray, np.ndarray]:
    logarithmic = np.logspace(-24, 0, 481, dtype=np.float64)
    local_linear = np.linspace(-1.0e-12, 1.0e-12, 2001, dtype=np.float64)
    values = np.unique(
        np.concatenate((-logarithmic[::-1], local_linear, logarithmic, np.asarray([0.0])))
    )
    return values, values >= 0.0


def _threshold_predictions(
    model: MonotoneThreshold, margins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = (
            model(torch.as_tensor(margins, dtype=torch.float64)).cpu().numpy()
        )
    return logits, logits >= 0.0


def _balanced_real_indices(
    pool: dict[str, np.ndarray], per_class: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    same = np.flatnonzero(pool["same"])
    new = np.flatnonzero(~pool["same"])
    count = min(per_class, len(same), len(new))
    selected = np.concatenate(
        (
            generator.choice(same, size=count, replace=False),
            generator.choice(new, size=count, replace=False),
        )
    )
    generator.shuffle(selected)
    return selected


def _classification_metrics(
    threshold: MonotoneThreshold, margins: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    logits, predictions = _threshold_predictions(threshold, margins)
    mistakes = predictions != labels
    return {
        "event_count": len(labels),
        "SAME_count": int(np.sum(labels)),
        "NEW_count": int(np.sum(~labels)),
        "overall_accuracy": float(np.mean(~mistakes)),
        "SAME_accuracy": float(np.mean(~mistakes[labels])) if np.any(labels) else 1.0,
        "NEW_accuracy": float(np.mean(~mistakes[~labels])) if np.any(~labels) else 1.0,
        "mistake_count": int(np.sum(mistakes)),
        "minimum_absolute_logit_margin": float(np.min(np.abs(logits))),
        "logits": logits,
        "predictions": predictions,
        "mistakes": mistakes,
    }


def _train_threshold(
    real_train_margins: np.ndarray,
    real_train_labels: np.ndarray,
    real_validation_margins: np.ndarray,
    real_validation_labels: np.ndarray,
) -> tuple[MonotoneThreshold, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = MonotoneThreshold().to(torch.float64)
    initial_theta = float(model.theta().detach().item())
    initial_alpha = float(model.alpha().detach().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_margins = torch.as_tensor(real_train_margins, dtype=torch.float64)
    train_labels = torch.as_tensor(
        real_train_labels.astype(np.float64), dtype=torch.float64
    )
    synthetic_margins_np, synthetic_labels_np = _boundary_values()
    synthetic_margins = torch.as_tensor(synthetic_margins_np, dtype=torch.float64)
    synthetic_labels = torch.as_tensor(
        synthetic_labels_np.astype(np.float64), dtype=torch.float64
    )
    dense_values, dense_labels = _dense_grid()
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf, math.inf, math.inf)
    best_epoch = 0
    exact_streak = 0
    stop_epoch = MAX_EPOCHS
    stopping_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            model(train_margins), train_labels
        )
        synthetic_loss = nn.functional.binary_cross_entropy_with_logits(
            model(synthetic_margins), synthetic_labels
        )
        loss = 0.5 * real_loss + 0.5 * synthetic_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        real_metrics = _classification_metrics(
            model, real_validation_margins, real_validation_labels
        )
        synthetic_metrics = _classification_metrics(
            model, synthetic_margins_np, synthetic_labels_np
        )
        dense_metrics = _classification_metrics(model, dense_values, dense_labels)
        theta = float(model.theta().detach().item())
        margin_deficit = max(
            0.0,
            CLEAR_LOGIT_MARGIN
            - synthetic_metrics["minimum_absolute_logit_margin"],
        )
        key = (
            real_metrics["mistake_count"]
            + synthetic_metrics["mistake_count"]
            + dense_metrics["mistake_count"],
            margin_deficit,
            abs(theta),
            -synthetic_metrics["minimum_absolute_logit_margin"],
        )
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        exact_now = bool(
            real_metrics["mistake_count"] == 0
            and synthetic_metrics["mistake_count"] == 0
            and dense_metrics["mistake_count"] == 0
            and synthetic_metrics["minimum_absolute_logit_margin"]
            >= CLEAR_LOGIT_MARGIN
        )
        exact_streak = exact_streak + 1 if exact_now else 0
        if exact_streak >= SUCCESS_STABILITY_EPOCHS:
            stop_epoch = epoch
            stopping_reason = "stable_boundary_and_real_exact"
            break
        if epoch == 1 or epoch % 500 == 0:
            print(
                f"same-size threshold epoch={epoch} "
                f"theta={theta:.6e} alpha={float(model.alpha().detach().item()):.6e} "
                f"synthetic_mistakes={synthetic_metrics['mistake_count']} "
                f"dense_mistakes={dense_metrics['mistake_count']}",
                flush=True,
            )
    model.load_state_dict(best_state)
    real_metrics = _classification_metrics(
        model, real_validation_margins, real_validation_labels
    )
    synthetic_metrics = _classification_metrics(
        model, synthetic_margins_np, synthetic_labels_np
    )
    dense_metrics = _classification_metrics(model, dense_values, dense_labels)
    summary = {
        "record_type": "summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "learned_theta": float(model.theta().detach().item()),
        "learned_alpha": float(model.alpha().detach().item()),
        "initial_theta": initial_theta,
        "initial_alpha": initial_alpha,
        "positive_alpha_parameterization": "softplus",
        "hidden_layer_count": 0,
        "boundary_conditioning_scale": BOUNDARY_SCALE,
        "real_training_event_count": len(real_train_labels),
        "real_validation_event_count": len(real_validation_labels),
        "synthetic_boundary_event_count": len(synthetic_labels_np),
        "real_balanced_accuracy": real_metrics["overall_accuracy"],
        "real_SAME_accuracy": real_metrics["SAME_accuracy"],
        "real_NEW_accuracy": real_metrics["NEW_accuracy"],
        "real_mistake_count": real_metrics["mistake_count"],
        "synthetic_accuracy": synthetic_metrics["overall_accuracy"],
        "synthetic_mistake_count": synthetic_metrics["mistake_count"],
        "minimum_absolute_synthetic_logit_margin": synthetic_metrics[
            "minimum_absolute_logit_margin"
        ],
        "m_zero_logit": float(
            synthetic_metrics["logits"][np.flatnonzero(synthetic_margins_np == 0.0)[0]]
        ),
        "m_zero_predicted_SAME": bool(
            synthetic_metrics["predictions"][
                np.flatnonzero(synthetic_margins_np == 0.0)[0]
            ]
        ),
        "dense_grid_count": len(dense_values),
        "dense_grid_mistake_count": dense_metrics["mistake_count"],
        "dense_grid_minimum_negative_magnitude": float(
            np.min(np.abs(dense_values[dense_values < 0.0]))
        ),
        "dense_transition_resolved_to_teacher_boundary": bool(
            dense_metrics["mistake_count"] == 0
        ),
    }
    summary["smoke_gate_passed"] = bool(
        real_metrics["mistake_count"] == 0
        and synthetic_metrics["mistake_count"] == 0
        and dense_metrics["mistake_count"] == 0
        and synthetic_metrics["minimum_absolute_logit_margin"]
        >= CLEAR_LOGIT_MARGIN
    )
    return model, summary


def _bookkeeping_reconstruction(
    threshold: MonotoneThreshold,
    frozen_margin: nn.Linear,
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    margin_w = float(frozen_margin.weight.detach().item())
    margin_b = float(frozen_margin.bias.detach().item())
    theta = float(threshold.theta().detach().item())
    rows: list[dict[str, Any]] = []
    exact_states = 0
    total_changes = 0
    for state_position, state in enumerate(states):
        teacher = grouping_audit._teacher_trace(state["sizes"], state["losses"])
        order = teacher["order"]
        membership = np.full(len(order), -1, dtype=np.int64)
        group_index = -1
        anchor = math.nan
        for order_position, index_value in enumerate(order):
            index = int(index_value)
            d_i = float(state["sizes"][index])
            if order_position == 0:
                group_index = 0
                anchor = d_i
            else:
                x = abs(d_i - anchor) / margin_data.TEACHER_SCALE
                frozen_value = margin_w * x + margin_b
                if frozen_value < theta:
                    group_index += 1
                    anchor = d_i
            membership[index] = group_index
        changes = int(np.sum(membership != teacher["membership"]))
        exact = changes == 0
        exact_states += int(exact)
        total_changes += changes
        rows.append(
            {
                "record_type": "state_reconstruction",
                "state_position": state_position,
                "source_state_index": state["source_state_index"],
                "cell_count": len(order),
                "teacher_group_count": len(teacher["groups"]),
                "learned_group_count": group_index + 1,
                "membership_change_count": changes,
                "exact_group_membership": exact,
            }
        )
    return rows, {
        "state_count": len(states),
        "exact_group_membership_states": exact_states,
        "group_membership_change_count": total_changes,
    }


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frozen_hash_before = _sha256(FROZEN_MARGIN_PATH)
    frozen_margin, _ = _load_frozen_margin()
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    source_train = turn._stratified_state_indices(
        payload["states"], "train", turn.TRAIN_SOURCE_STATES, smoke.DATA_SEED
    )
    source_validation = turn._stratified_state_indices(
        payload["states"],
        "validation",
        turn.VALIDATION_SOURCE_STATES,
        smoke.DATA_SEED + 1,
    )
    indexed = [(index, "training") for index in source_train]
    indexed.extend((index, "held_out") for index in source_validation)
    all_states = grouping_audit._build_states(payload, indexed, workers)
    train_states = all_states[: len(source_train)]
    validation_states = all_states[len(source_train) :]
    train_pool = margin_data._flatten_events(train_states)
    validation_pool = margin_data._flatten_events(validation_states)
    train_real_indices = _balanced_real_indices(
        train_pool, REAL_PER_CLASS, MODEL_SEED + 1
    )
    validation_real_indices = _balanced_real_indices(
        validation_pool, VALIDATION_PER_CLASS, MODEL_SEED + 2
    )
    train_margins = _frozen_margins(
        frozen_margin, train_pool["x"][train_real_indices]
    )
    validation_margins = _frozen_margins(
        frozen_margin, validation_pool["x"][validation_real_indices]
    )
    threshold, boundary_summary = _train_threshold(
        train_margins,
        train_pool["same"][train_real_indices],
        validation_margins,
        validation_pool["same"][validation_real_indices],
    )
    synthetic_values, synthetic_labels = _boundary_values()
    synthetic_metrics = _classification_metrics(
        threshold, synthetic_values, synthetic_labels
    )
    boundary_rows: list[dict[str, Any]] = [boundary_summary]
    for index, value in enumerate(synthetic_values):
        boundary_rows.append(
            {
                "record_type": "synthetic_boundary_value",
                "margin": float(value),
                "teacher_SAME": bool(synthetic_labels[index]),
                "logit": float(synthetic_metrics["logits"][index]),
                "predicted_SAME": bool(synthetic_metrics["predictions"][index]),
                "correct": bool(
                    synthetic_metrics["predictions"][index] == synthetic_labels[index]
                ),
            }
        )
    _write_csv(BOUNDARY_PATH, boundary_rows)
    if not bool(boundary_summary["smoke_gate_passed"]):
        print("same-size threshold boundary gate failed; stopped", flush=True)
        return

    full_indices = np.arange(len(validation_pool["same"]), dtype=np.int64)
    full_margins = _frozen_margins(frozen_margin, validation_pool["x"])
    full_metrics = _classification_metrics(
        threshold, full_margins, validation_pool["same"]
    )
    state_rows, bookkeeping = _bookkeeping_reconstruction(
        threshold, frozen_margin, validation_states
    )
    frozen_hash_after = _sha256(FROZEN_MARGIN_PATH)
    full_summary = {
        "record_type": "summary",
        "held_out_event_count": len(full_indices),
        "held_out_SAME_count": full_metrics["SAME_count"],
        "held_out_NEW_count": full_metrics["NEW_count"],
        "learned_theta": float(threshold.theta().detach().item()),
        "learned_alpha": float(threshold.alpha().detach().item()),
        "overall_accuracy": full_metrics["overall_accuracy"],
        "SAME_accuracy": full_metrics["SAME_accuracy"],
        "NEW_accuracy": full_metrics["NEW_accuracy"],
        "mistake_count": full_metrics["mistake_count"],
        **bookkeeping,
        "frozen_margin_sha256_before": frozen_hash_before,
        "frozen_margin_sha256_after": frozen_hash_after,
        "frozen_margin_unchanged": frozen_hash_before == frozen_hash_after,
        "threshold_network_only_trained": True,
    }
    full_summary["full_pool_gate_passed"] = bool(
        full_metrics["mistake_count"] == 0
        and bookkeeping["exact_group_membership_states"] == len(validation_states)
        and bookkeeping["group_membership_change_count"] == 0
        and frozen_hash_before == frozen_hash_after == FROZEN_MARGIN_SHA256
    )
    full_rows: list[dict[str, Any]] = [full_summary, *state_rows]
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("same-size threshold full-pool gate failed; stopped", flush=True)
        return

    checkpoint = {
        "format_version": 1,
        "component": "same_size_threshold",
        "status": "LOCKED_COMPONENT",
        "architecture": "logit=alpha*(frozen_margin-theta), alpha=softplus(raw)>0",
        "state_dict": copy.deepcopy(threshold.state_dict()),
        "learned_theta": torch.tensor(
            float(threshold.theta().detach().item()), dtype=torch.float64
        ),
        "learned_alpha": torch.tensor(
            float(threshold.alpha().detach().item()), dtype=torch.float64
        ),
        "boundary_conditioning_scale": BOUNDARY_SCALE,
        "frozen_margin_checkpoint": str(FROZEN_MARGIN_PATH),
        "frozen_margin_sha256": FROZEN_MARGIN_SHA256,
        "boundary_summary": boundary_summary,
        "full_pool_summary": full_summary,
        "complete_same_size_grouping_operator_locked": True,
    }
    torch.save(checkpoint, THRESHOLD_PATH)
    restored = torch.load(THRESHOLD_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "LOCKED_COMPONENT":
        raise RuntimeError("same-size threshold checkpoint readback failed")
    checkpoint_hash = _sha256(THRESHOLD_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {THRESHOLD_PATH.name}\n", encoding="utf-8"
    )
    full_summary["threshold_checkpoint"] = str(THRESHOLD_PATH)
    full_summary["threshold_checkpoint_sha256"] = checkpoint_hash
    full_summary["same_size_grouping_operator_status"] = "LOCKED"
    _write_csv(FULL_PATH, full_rows)
    _write_csv(
        LOCK_SUMMARY_PATH,
        [
            {
                "component": "same_size_margin",
                "status": "FROZEN_CANDIDATE_UNCHANGED",
                "checkpoint": str(FROZEN_MARGIN_PATH),
                "checkpoint_sha256": FROZEN_MARGIN_SHA256,
                "retrained_or_modified": False,
            },
            {
                "component": "same_size_threshold",
                "status": "LOCKED_COMPONENT",
                "checkpoint": str(THRESHOLD_PATH),
                "checkpoint_sha256": checkpoint_hash,
                "learned_theta": float(threshold.theta().detach().item()),
                "learned_alpha": float(threshold.alpha().detach().item()),
            },
            {
                "component": "same_size_grouping_operator",
                "status": "LOCKED",
                "held_out_mistakes": full_metrics["mistake_count"],
                "exact_group_membership_states": bookkeeping[
                    "exact_group_membership_states"
                ],
                "state_count": bookkeeping["state_count"],
            },
        ],
    )
    print(
        "same-size grouping operator LOCKED "
        f"theta={float(threshold.theta().detach().item()):.17g} "
        f"alpha={float(threshold.alpha().detach().item()):.17g} "
        f"sha256={checkpoint_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args()
    run(args.workers)


if __name__ == "__main__":
    main()
