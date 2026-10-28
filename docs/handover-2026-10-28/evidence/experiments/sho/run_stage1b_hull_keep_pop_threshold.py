#!/usr/bin/env python3
"""Lock the homogeneous margin and learn only its final monotone threshold."""

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
import experiments.sho.run_stage1b_hull_margin_comparator as margin
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.validate_stage1b_bilinear_hull_turn as validation
import experiments.sho.run_stage1b_edirect_vs_direct as edirect


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_keep_pop_threshold")
AUDIT_PATH = OUTPUT_DIR / "hull_threshold_audit.csv"
SMOKE_PATH = OUTPUT_DIR / "hull_threshold_smoke.csv"
FULL_PATH = OUTPUT_DIR / "hull_threshold_full_pool.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "hull_threshold_reconstruction.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "checkpoint_lock_summary.csv"

SKEW_CHECKPOINT_PATH = Path(
    "outputs/sho/stage1B/skew_bilinear_hull_turn/"
    "skew_bilinear_epoch35_locked.pt"
)
EXPECTED_SKEW_HASH = (
    "b3bd16fe2b56b0577fb024195313cdd08856eaa607139817360ca5169c10e92d"
)
HOMOGENEOUS_SOURCE_PATH = Path(
    "outputs/sho/stage1B/homogeneous_hull_margin/homogeneous_margin_smoke.csv"
)
HOMOGENEOUS_CHECKPOINT_PATH = Path(
    "outputs/sho/stage1B/homogeneous_hull_margin/"
    "homogeneous_margin_locked.pt"
)
HOMOGENEOUS_HASH_PATH = HOMOGENEOUS_CHECKPOINT_PATH.with_suffix(".sha256")
THRESHOLD_CHECKPOINT_PATH = OUTPUT_DIR / "hull_keep_pop_threshold_locked.pt"
THRESHOLD_HASH_PATH = THRESHOLD_CHECKPOINT_PATH.with_suffix(".sha256")

EXPECTED_MARGIN_WEIGHTS = np.asarray(
    [0.9999999999999998, 0.9999999999999957], dtype=np.float64
)
MAX_EPOCHS = 5_000
PATIENCE = 400
SUCCESS_STABILITY_EPOCHS = 100
LEARNING_RATE = 2.0e-2
MODEL_SEED = smoke.MODEL_SEED + 2_718_281


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


def _load_skew() -> tuple[np.ndarray, str]:
    actual_hash = _sha256(SKEW_CHECKPOINT_PATH)
    if actual_hash != EXPECTED_SKEW_HASH:
        raise RuntimeError("locked skew checkpoint hash mismatch")
    checkpoint = torch.load(
        SKEW_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("lock_status") != "LOCKED":
        raise RuntimeError("skew checkpoint is not locked")
    return checkpoint["effective_weight"].cpu().numpy().copy(), actual_hash


def _source_margin_summary() -> dict[str, str]:
    with HOMOGENEOUS_SOURCE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = next(row for row in rows if row["record_type"] == "final_summary")
    checks = {
        "learned_a": EXPECTED_MARGIN_WEIGHTS[0],
        "learned_b": EXPECTED_MARGIN_WEIGHTS[1],
        "learned_bias": 0.0,
    }
    for key, expected in checks.items():
        if float(result[key]) != float(expected):
            raise RuntimeError(f"homogeneous source {key} differs from locked value")
    if result["smoke_gate_passed"] != "True" or result["mistake_count"] != "0":
        raise RuntimeError("homogeneous source did not pass its smoke gate")
    return result


def _lock_homogeneous_margin() -> tuple[np.ndarray, str, bool]:
    source = _source_margin_summary()
    created = False
    if not HOMOGENEOUS_CHECKPOINT_PATH.exists():
        checkpoint = {
            "format_version": 1,
            "component": "hull_margin_network",
            "lock_status": "LOCKED",
            "architecture": "margin_hat = w_s*s + w_t*t; bias = 0",
            "state_dict": {
                "weight": torch.as_tensor(
                    EXPECTED_MARGIN_WEIGHTS.reshape(1, 2), dtype=torch.float64
                )
            },
            "effective_weight": torch.as_tensor(
                EXPECTED_MARGIN_WEIGHTS.copy(), dtype=torch.float64
            ),
            "bias": torch.tensor(0.0, dtype=torch.float64),
            "source_best_epoch": int(source["best_epoch"]),
            "source_smoke_path": str(HOMOGENEOUS_SOURCE_PATH),
            "source_smoke_sha256": _sha256(HOMOGENEOUS_SOURCE_PATH),
            "full_pool_event_count": 98_509,
            "full_pool_mistake_count": 0,
            "exact_hull_states": 200,
            "exact_po_states": 200,
            "training_performed_during_locking": False,
            "never_retrain_without_explicit_request": True,
        }
        torch.save(checkpoint, HOMOGENEOUS_CHECKPOINT_PATH)
        created = True
    restored = torch.load(
        HOMOGENEOUS_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if restored.get("component") != "hull_margin_network":
        raise RuntimeError("wrong homogeneous checkpoint component")
    if restored.get("lock_status") != "LOCKED":
        raise RuntimeError("homogeneous checkpoint is not locked")
    weights = restored["effective_weight"].cpu().numpy().copy()
    bias = float(restored["bias"].item())
    if not np.array_equal(weights, EXPECTED_MARGIN_WEIGHTS) or bias != 0.0:
        raise RuntimeError("locked homogeneous parameters differ from source")
    checkpoint_hash = _sha256(HOMOGENEOUS_CHECKPOINT_PATH)
    HOMOGENEOUS_HASH_PATH.write_text(
        f"{checkpoint_hash}  {HOMOGENEOUS_CHECKPOINT_PATH.name}\n",
        encoding="utf-8",
    )
    return weights, checkpoint_hash, created


def _frozen_margin(dataset: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(dataset["inputs"], dtype=torch.float64)
    matrix = torch.as_tensor(weights.reshape(1, 2), dtype=torch.float64)
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(nn.functional.linear(values[start : start + 4096], matrix).squeeze(-1))
    return torch.cat(parts).cpu().numpy()


def _percentile_rows(values: np.ndarray, side: str, quantity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for percentile in (0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0):
        rows.append(
            {
                "record_type": "percentile",
                "quantity": quantity,
                "side": side,
                "percentile": percentile,
                "value": float(np.percentile(values, percentile)),
                "event_count": len(values),
            }
        )
    return rows


def _audit_rows(
    dataset: dict[str, np.ndarray], learned_margin: np.ndarray
) -> list[dict[str, Any]]:
    labels = dataset["label"].astype(bool)
    keep = learned_margin[labels]
    pop = learned_margin[~labels]
    exact = dataset["target"]
    positive_keep = keep[keep > 0.0]
    negative_pop = pop[pop < 0.0]
    smallest_positive = float(np.min(positive_keep)) if len(positive_keep) else math.nan
    largest_negative = float(np.max(negative_pop)) if len(negative_pop) else math.nan
    rows: list[dict[str, Any]] = [
        {
            "record_type": "audit_summary",
            "event_count": len(labels),
            "keep_count": int(np.sum(labels)),
            "pop_count": int(np.sum(~labels)),
            "smallest_positive_keep_learned_margin": smallest_positive,
            "largest_negative_pop_learned_margin": largest_negative,
            "nearest_class_gap": smallest_positive - largest_negative,
            "learned_margin_exact_zero_count": int(np.sum(learned_margin == 0.0)),
            "exact_teacher_margin_exact_zero_count": int(np.sum(exact == 0.0)),
            "upstream_margin_mistake_count": int(
                np.sum((learned_margin >= 0.0) != labels)
            ),
            "minimum_absolute_learned_margin": float(np.min(np.abs(learned_margin))),
            "minimum_absolute_exact_teacher_margin": float(np.min(np.abs(exact))),
            "maximum_upstream_absolute_error": float(
                np.max(np.abs(learned_margin - exact))
            ),
        }
    ]
    rows.extend(_percentile_rows(keep, "KEEP", "learned_margin"))
    rows.extend(_percentile_rows(pop, "POP", "learned_margin"))
    rows.extend(_percentile_rows(np.abs(keep), "KEEP", "absolute_learned_margin"))
    rows.extend(_percentile_rows(np.abs(pop), "POP", "absolute_learned_margin"))
    return rows


class MonotoneThreshold(nn.Module):
    """logit = alpha * (margin - theta), alpha > 0."""

    def __init__(self, threshold_scale: float) -> None:
        super().__init__()
        if not threshold_scale > 0.0 or not math.isfinite(threshold_scale):
            raise RuntimeError("invalid threshold parameter scale")
        self.register_buffer(
            "threshold_scale", torch.tensor(threshold_scale, dtype=torch.float64)
        )
        self.raw_alpha = nn.Parameter(torch.empty((), dtype=torch.float64))
        self.scaled_theta = nn.Parameter(torch.empty((), dtype=torch.float64))
        nn.init.normal_(self.raw_alpha, mean=0.0, std=0.35)
        nn.init.normal_(self.scaled_theta, mean=0.0, std=0.35)
        if float(self.scaled_theta.detach()) == 0.0:
            raise RuntimeError("theta was initialized to forbidden exact zero")

    def alpha(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_alpha) / self.threshold_scale

    def theta(self) -> torch.Tensor:
        return self.scaled_theta * self.threshold_scale

    def forward(self, learned_margin: torch.Tensor) -> torch.Tensor:
        return self.alpha() * (learned_margin - self.theta())


def _threshold_metrics(
    learned_margin: np.ndarray,
    exact_margin: np.ndarray,
    labels: np.ndarray,
    theta: float,
    alpha: float,
) -> dict[str, Any]:
    logits = alpha * (learned_margin - theta)
    predicted = logits >= 0.0
    truth = labels.astype(bool)
    keep_count = int(np.sum(truth))
    pop_count = int(np.sum(~truth))
    keep_correct = int(np.sum(predicted & truth))
    pop_correct = int(np.sum(~predicted & ~truth))
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "keep_accuracy": keep_correct / keep_count if keep_count else "",
        "pop_accuracy": pop_correct / pop_count if pop_count else "",
        "mistake_count": int(np.sum(predicted != truth)),
        "keep_count": keep_count,
        "pop_count": pop_count,
        "learned_theta": theta,
        "learned_alpha": alpha,
        "absolute_theta_error_from_teacher_zero": abs(theta),
        "minimum_absolute_learned_margin": float(np.min(np.abs(learned_margin))),
        "minimum_absolute_exact_margin": float(np.min(np.abs(exact_margin))),
        "maximum_upstream_margin_error": float(
            np.max(np.abs(learned_margin - exact_margin))
        ),
    }


def _slice_rows(
    record_type: str,
    learned_margin: np.ndarray,
    exact_margin: np.ndarray,
    labels: np.ndarray,
    theta: float,
    alpha: float,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bound in (1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40, 1.0e-50):
        mask = np.abs(learned_margin) < bound
        row: dict[str, Any] = {
            "record_type": record_type,
            "split": split,
            "slice": f"abs_learned_margin_lt_{bound:.0e}",
            "event_count": int(np.sum(mask)),
        }
        if np.any(mask):
            row.update(
                _threshold_metrics(
                    learned_margin[mask],
                    exact_margin[mask],
                    labels[mask],
                    theta,
                    alpha,
                )
            )
        rows.append(row)
    return rows


def _mistake_rows(
    learned_margin: np.ndarray,
    exact_margin: np.ndarray,
    labels: np.ndarray,
    theta: float,
    alpha: float,
    dataset: dict[str, np.ndarray],
    split: str,
) -> list[dict[str, Any]]:
    predicted = alpha * (learned_margin - theta) >= 0.0
    truth = labels.astype(bool)
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(predicted != truth):
        triple = dataset["triple_indices"][index]
        rows.append(
            {
                "record_type": "mistake",
                "split": split,
                "event_index": int(index),
                "state_position": int(dataset["state_position"][index]),
                "triple_first": int(triple[0]),
                "triple_second": int(triple[1]),
                "triple_third": int(triple[2]),
                "teacher_class": "KEEP" if truth[index] else "POP",
                "predicted_class": "KEEP" if predicted[index] else "POP",
                "learned_margin": float(learned_margin[index]),
                "exact_teacher_margin": float(exact_margin[index]),
                "learned_theta": theta,
                "distance_of_learned_margin_from_zero": float(
                    abs(learned_margin[index])
                ),
                "distance_of_exact_margin_from_zero": float(
                    abs(exact_margin[index])
                ),
                "distance_from_learned_threshold": float(
                    abs(learned_margin[index] - theta)
                ),
                "upstream_numerical_error": float(
                    learned_margin[index] - exact_margin[index]
                ),
            }
        )
    return rows


def _train_threshold(
    training_margin: np.ndarray,
    training_labels: np.ndarray,
    validation_margin: np.ndarray,
    validation_exact: np.ndarray,
    validation_labels: np.ndarray,
) -> tuple[MonotoneThreshold, list[dict[str, Any]], dict[str, Any]]:
    nonzero = np.abs(training_margin[training_margin != 0.0])
    if len(nonzero) == 0:
        raise RuntimeError("training margins are all exactly zero")
    threshold_scale = float(np.min(nonzero))
    torch.manual_seed(MODEL_SEED)
    model = MonotoneThreshold(threshold_scale).to(torch.float64)
    if float(model.theta().detach()) == 0.0:
        raise RuntimeError("theta initialization was exactly zero")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x = torch.as_tensor(training_margin, dtype=torch.float64)
    train_y = torch.as_tensor(training_labels, dtype=torch.float64)
    validation_x = torch.as_tensor(validation_margin, dtype=torch.float64)
    validation_y = torch.as_tensor(validation_labels, dtype=torch.float64)
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    stale = 0
    exact_streak = 0
    stopping_reason = "maximum_epochs"
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        logits = model(train_x)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x)
            validation_loss = float(
                nn.functional.binary_cross_entropy_with_logits(
                    validation_logits, validation_y
                ).item()
            )
            theta = float(model.theta().item())
            alpha = float(model.alpha().item())
        metrics = _threshold_metrics(
            validation_margin,
            validation_exact,
            validation_labels,
            theta,
            alpha,
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
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_bce": float(loss.item()),
                "validation_bce": validation_loss,
                "stale_epochs": stale,
                "exact_validation_streak": exact_streak,
                **metrics,
            }
        )
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"threshold epoch={epoch} theta={theta:.6e} alpha={alpha:.6e} "
                f"mistakes={metrics['mistake_count']} bce={validation_loss:.6e}",
                flush=True,
            )
        if exact_streak >= SUCCESS_STABILITY_EPOCHS:
            stopping_reason = "stable_exact_validation"
            break
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    model.eval()
    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    metrics = _threshold_metrics(
        validation_margin,
        validation_exact,
        validation_labels,
        theta,
        alpha,
    )
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "maximum_epochs": MAX_EPOCHS,
        "plateau_patience": PATIENCE,
        "success_stability_epochs": SUCCESS_STABILITY_EPOCHS,
        "training_events": len(training_margin),
        "validation_events": len(validation_margin),
        "threshold_parameter_scale": threshold_scale,
        "initial_theta_was_zero": False,
        "classification_supervision_used": True,
        "extra_hidden_layers": 0,
        "additional_bias": False,
        **metrics,
        "smoke_gate_passed": metrics["mistake_count"] == 0,
    }
    rows.append(summary)
    rows.extend(
        _slice_rows(
            "margin_slice",
            validation_margin,
            validation_exact,
            validation_labels,
            theta,
            alpha,
            "validation",
        )
    )
    return model, rows, summary


def _threshold_hull(
    skew_weight: np.ndarray,
    margin_weight: np.ndarray,
    theta: float,
    alpha: float,
    sizes: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    stack: list[int] = []
    for third in range(len(sizes)):
        while len(stack) >= 2:
            score, tolerance, _ = validation._frozen_turn(
                skew_weight, sizes, losses, stack[-2], stack[-1], third
            )
            learned_margin = float(
                margin_weight[0] * score + margin_weight[1] * tolerance
            )
            logit = alpha * (learned_margin - theta)
            if logit >= 0.0:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64)


def _reconstruction_rows(
    skew_weight: np.ndarray,
    margin_weight: np.ndarray,
    theta: float,
    alpha: float,
    states: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        sizes = state["sizes"]
        losses = state["losses"]
        teacher_hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        learned_hull = _threshold_hull(
            skew_weight, margin_weight, theta, alpha, sizes, losses
        )
        teacher_po = edirect._potentially_optimal_cells(sizes, losses)
        learned_po = validation._improvement_filter(learned_hull, sizes, losses)
        teacher_hull_set = set(int(value) for value in teacher_hull)
        learned_hull_set = set(int(value) for value in learned_hull)
        teacher_po_set = set(int(value) for value in teacher_po)
        learned_po_set = set(int(value) for value in learned_po)
        deterministic_hull, _ = turn._teacher_hull_and_triples(sizes, losses)
        rows.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "state_bin": state["state_bin"],
                "representative_cell_count": len(sizes),
                "teacher_hull_size": len(teacher_hull),
                "learned_hull_size": len(learned_hull),
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
        "state_count": len(rows),
        "exact_hull_states": sum(bool(row["exact_hull"]) for row in rows),
        "exact_hull_fraction": float(np.mean([row["exact_hull"] for row in rows])),
        "false_hull_additions": sum(int(row["false_hull_additions"]) for row in rows),
        "false_hull_removals": sum(int(row["false_hull_removals"]) for row in rows),
        "exact_po_states": sum(bool(row["exact_po_set"]) for row in rows),
        "exact_po_fraction": float(
            np.mean([row["exact_po_set"] for row in rows])
        ),
        "false_po_additions": sum(int(row["false_po_additions"]) for row in rows),
        "false_po_removals": sum(int(row["false_po_removals"]) for row in rows),
        "deterministic_teacher_exact_states": sum(
            bool(row["deterministic_teacher_exact"]) for row in rows
        ),
    }
    summary["reconstruction_passed"] = bool(
        summary["exact_hull_states"] == len(rows)
        and summary["exact_po_states"] == len(rows)
        and summary["deterministic_teacher_exact_states"] == len(rows)
    )
    return [summary] + rows, summary


def _lock_threshold(
    model: MonotoneThreshold,
    smoke_summary: dict[str, Any],
    full_summary: dict[str, Any],
    reconstruction_summary: dict[str, Any],
    skew_hash: str,
    margin_hash: str,
) -> str:
    checkpoint = {
        "format_version": 1,
        "component": "hull_keep_pop_threshold",
        "lock_status": "LOCKED",
        "architecture": "logit = softplus(raw_alpha)/q * (m_hat - q*scaled_theta)",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_theta": torch.tensor(
            float(model.theta().item()), dtype=torch.float64
        ),
        "learned_alpha": torch.tensor(
            float(model.alpha().item()), dtype=torch.float64
        ),
        "teacher_threshold": 0.0,
        "smoke_mistake_count": int(smoke_summary["mistake_count"]),
        "full_pool_event_count": int(full_summary["event_count"]),
        "full_pool_mistake_count": int(full_summary["mistake_count"]),
        "exact_hull_states": int(reconstruction_summary["exact_hull_states"]),
        "exact_po_states": int(reconstruction_summary["exact_po_states"]),
        "locked_skew_checkpoint_sha256": skew_hash,
        "locked_margin_checkpoint_sha256": margin_hash,
        "never_retrain_without_explicit_request": True,
    }
    torch.save(checkpoint, THRESHOLD_CHECKPOINT_PATH)
    restored = torch.load(
        THRESHOLD_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if restored.get("lock_status") != "LOCKED":
        raise RuntimeError("threshold checkpoint readback was not locked")
    checkpoint_hash = _sha256(THRESHOLD_CHECKPOINT_PATH)
    THRESHOLD_HASH_PATH.write_text(
        f"{checkpoint_hash}  {THRESHOLD_CHECKPOINT_PATH.name}\n",
        encoding="utf-8",
    )
    return checkpoint_hash


def _lock_summary_rows(
    skew_hash: str,
    margin_hash: str,
    margin_created: bool,
    threshold_hash: str | None,
) -> list[dict[str, Any]]:
    rows = [
        {
            "component": "hull_turn_continuous_score",
            "status": "LOCKED",
            "checkpoint": str(SKEW_CHECKPOINT_PATH),
            "sha256": skew_hash,
            "retrained_in_this_run": False,
        },
        {
            "component": "hull_margin_network",
            "status": "LOCKED",
            "checkpoint": str(HOMOGENEOUS_CHECKPOINT_PATH),
            "sha256": margin_hash,
            "retrained_in_this_run": False,
            "checkpoint_created_from_successful_weights": margin_created,
        },
    ]
    if threshold_hash is not None:
        rows.append(
            {
                "component": "hull_keep_pop_threshold",
                "status": "LOCKED",
                "checkpoint": str(THRESHOLD_CHECKPOINT_PATH),
                "sha256": threshold_hash,
                "retrained_in_this_run": True,
            }
        )
    else:
        rows.append(
            {
                "component": "hull_keep_pop_threshold",
                "status": "NOT_LOCKED_GATE_FAILED",
                "checkpoint": "",
                "sha256": "",
                "retrained_in_this_run": True,
            }
        )
    return rows


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for conditional in (RECONSTRUCTION_PATH, THRESHOLD_CHECKPOINT_PATH, THRESHOLD_HASH_PATH):
        if conditional.exists():
            conditional.unlink()
    skew_weight, skew_hash = _load_skew()
    margin_weight, margin_hash, margin_created = _lock_homogeneous_margin()
    print(f"verified skew lock {skew_hash}", flush=True)
    print(f"locked homogeneous margin {margin_hash}", flush=True)

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
    train_dataset = margin._margin_dataset(train_selected, train_states, skew_weight)
    validation_dataset = margin._margin_dataset(
        validation_selected, validation_states, skew_weight
    )
    full_dataset = margin._margin_dataset(
        validation_pool, validation_states, skew_weight
    )
    training_margin = _frozen_margin(train_dataset, margin_weight)
    validation_margin = _frozen_margin(validation_dataset, margin_weight)
    full_margin = _frozen_margin(full_dataset, margin_weight)

    audit_rows = _audit_rows(full_dataset, full_margin)
    audit_rows[0]["locked_skew_checkpoint_sha256"] = skew_hash
    audit_rows[0]["locked_margin_checkpoint_sha256"] = margin_hash
    _write_csv(AUDIT_PATH, audit_rows)

    model, smoke_rows, smoke_summary = _train_threshold(
        training_margin,
        train_dataset["label"],
        validation_margin,
        validation_dataset["target"],
        validation_dataset["label"],
    )
    theta = float(model.theta().item())
    alpha = float(model.alpha().item())
    smoke_summary["locked_skew_checkpoint_sha256"] = skew_hash
    smoke_summary["locked_margin_checkpoint_sha256"] = margin_hash
    smoke_rows.extend(
        _mistake_rows(
            validation_margin,
            validation_dataset["target"],
            validation_dataset["label"],
            theta,
            alpha,
            validation_dataset,
            "validation",
        )
    )
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
        _write_csv(FULL_PATH, [{"record_type": "not_run", "reason": "smoke gate failed"}])
        _write_csv(
            LOCK_SUMMARY_PATH,
            _lock_summary_rows(skew_hash, margin_hash, margin_created, None),
        )
        print("threshold smoke gate failed; stopped", flush=True)
        return

    full_metrics = _threshold_metrics(
        full_margin,
        full_dataset["target"],
        full_dataset["label"],
        theta,
        alpha,
    )
    full_summary = {
        "record_type": "final_summary",
        "event_count": len(full_margin),
        **full_metrics,
        "full_pool_gate_passed": full_metrics["mistake_count"] == 0,
        "locked_skew_checkpoint_sha256": skew_hash,
        "locked_margin_checkpoint_sha256": margin_hash,
    }
    full_rows: list[dict[str, Any]] = [full_summary]
    full_rows.extend(
        _slice_rows(
            "margin_slice",
            full_margin,
            full_dataset["target"],
            full_dataset["label"],
            theta,
            alpha,
            "full_pool",
        )
    )
    full_rows.extend(
        _mistake_rows(
            full_margin,
            full_dataset["target"],
            full_dataset["label"],
            theta,
            alpha,
            full_dataset,
            "full_pool",
        )
    )
    _write_csv(FULL_PATH, full_rows)
    if full_metrics["mistake_count"] != 0:
        _write_csv(
            LOCK_SUMMARY_PATH,
            _lock_summary_rows(skew_hash, margin_hash, margin_created, None),
        )
        print(
            f"threshold full-pool gate failed: mistakes={full_metrics['mistake_count']}",
            flush=True,
        )
        return

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
        skew_weight,
        margin_weight,
        theta,
        alpha,
        reconstruction_states,
    )
    _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)
    if not bool(reconstruction_summary["reconstruction_passed"]):
        _write_csv(
            LOCK_SUMMARY_PATH,
            _lock_summary_rows(skew_hash, margin_hash, margin_created, None),
        )
        print("threshold hull reconstruction gate failed", flush=True)
        return

    threshold_hash = _lock_threshold(
        model,
        smoke_summary,
        full_summary,
        reconstruction_summary,
        skew_hash,
        margin_hash,
    )
    _write_csv(
        LOCK_SUMMARY_PATH,
        _lock_summary_rows(skew_hash, margin_hash, margin_created, threshold_hash),
    )
    print(
        f"threshold locked theta={theta:.17e} alpha={alpha:.17e} "
        f"full_mistakes=0 hulls={reconstruction_summary['exact_hull_states']}/"
        f"{reconstruction_summary['state_count']} sha256={threshold_hash}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
