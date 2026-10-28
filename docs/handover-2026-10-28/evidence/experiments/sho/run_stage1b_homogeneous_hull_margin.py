#!/usr/bin/env python3
"""Learn only the homogeneous continuous hull-comparator margin."""

from __future__ import annotations

import argparse
import copy
import csv
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


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/homogeneous_hull_margin")
SMOKE_PATH = OUTPUT_DIR / "homogeneous_margin_smoke.csv"
FULL_POOL_PATH = OUTPUT_DIR / "homogeneous_margin_full_pool.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "homogeneous_margin_hull_reconstruction.csv"
LOCKED_CHECKPOINT_PATH = Path(
    "outputs/sho/stage1B/skew_bilinear_hull_turn/"
    "skew_bilinear_epoch35_locked.pt"
)
EXPECTED_LOCKED_HASH = (
    "b3bd16fe2b56b0577fb024195313cdd08856eaa607139817360ca5169c10e92d"
)
AFFINE_BASELINE_PATH = Path(
    "outputs/sho/stage1B/hull_margin_comparator/hull_margin_smoke.csv"
)

MAX_EPOCHS = 2_000
PATIENCE = 80
PLATEAU_RELATIVE_IMPROVEMENT = 1.0e-7
LEARNING_RATE = 1.0e-2
MODEL_SEED = smoke.MODEL_SEED + 1_414_213
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


def _load_locked_weight() -> tuple[np.ndarray, str]:
    actual_hash = margin._sha256(LOCKED_CHECKPOINT_PATH)
    if actual_hash != EXPECTED_LOCKED_HASH:
        raise RuntimeError(
            "locked skew checkpoint hash mismatch: "
            f"expected {EXPECTED_LOCKED_HASH}, got {actual_hash}"
        )
    checkpoint = torch.load(
        LOCKED_CHECKPOINT_PATH, map_location="cpu", weights_only=False
    )
    if checkpoint.get("lock_status") != "LOCKED":
        raise RuntimeError("skew checkpoint is not marked LOCKED")
    weight = checkpoint["effective_weight"].detach().cpu().numpy().copy()
    expected = np.asarray(
        [[0.0, 1.0000000000000002], [-1.0000000000000002, 0.0]],
        dtype=np.float64,
    )
    if not np.array_equal(weight, expected):
        raise RuntimeError("locked skew checkpoint has an unexpected effective W")
    return weight, actual_hash


class HomogeneousScaling:
    """Positive scale-only conditioning; never introduces an intercept."""

    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        self.input_scale = np.sqrt(np.mean(inputs**2, axis=0))
        self.target_scale = float(np.sqrt(np.mean(target**2)))
        if np.any(self.input_scale <= 0.0) or not np.all(
            np.isfinite(self.input_scale)
        ):
            raise RuntimeError("invalid homogeneous input scale")
        if not self.target_scale > 0.0 or not np.isfinite(self.target_scale):
            raise RuntimeError("invalid homogeneous target scale")

    def transform_inputs(self, values: np.ndarray) -> np.ndarray:
        return values / self.input_scale

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return values / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale

    def original_coefficients(self, model: nn.Linear) -> np.ndarray:
        standardized = model.weight.detach().cpu().numpy()[0]
        return self.target_scale * standardized / self.input_scale


def _predict(
    model: nn.Linear, scaling: HomogeneousScaling, inputs: np.ndarray
) -> np.ndarray:
    values = torch.as_tensor(
        scaling.transform_inputs(inputs), dtype=torch.float64
    )
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(model(values[start : start + 4096]).squeeze(-1))
    return scaling.inverse_target(torch.cat(parts).cpu().numpy())


def _gate(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["rmse"] <= RMSE_GATE
        and metrics["maximum_absolute_error"] <= MAX_ERROR_GATE
        and metrics["decision_accuracy"] == 1.0
        and metrics["keep_agreement"] == 1.0
        and metrics["pop_agreement"] == 1.0
    )


def _metric_row(
    record_type: str,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
    **extra: Any,
) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(len(prediction), dtype=bool)
    count = int(np.sum(mask))
    row: dict[str, Any] = {"record_type": record_type, "event_count": count, **extra}
    if count == 0:
        return row
    row.update(margin._metrics(dataset["target"][mask], prediction[mask]))
    row.update(margin._decision_metrics(dataset["label"][mask], prediction[mask]))
    row["minimum_absolute_true_margin"] = float(
        np.min(np.abs(dataset["target"][mask]))
    )
    row["maximum_absolute_true_margin"] = float(
        np.max(np.abs(dataset["target"][mask]))
    )
    return row


def _magnitude_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    absolute = np.abs(dataset["target"])
    buckets = [
        ("abs_margin_lt_1e-40", absolute < 1.0e-40),
        ("abs_margin_1e-40_to_1e-24", (absolute >= 1.0e-40) & (absolute < 1.0e-24)),
        ("abs_margin_1e-24_to_1e-12", (absolute >= 1.0e-24) & (absolute < 1.0e-12)),
        ("abs_margin_1e-12_to_1e-6", (absolute >= 1.0e-12) & (absolute < 1.0e-6)),
        ("abs_margin_ge_1e-6", absolute >= 1.0e-6),
    ]
    return [
        _metric_row(
            "margin_magnitude", dataset, prediction, mask, split=split, bucket=name
        )
        for name, mask in buckets
    ]


def _affine_baseline_row() -> dict[str, Any]:
    with AFFINE_BASELINE_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = next(row for row in rows if row["record_type"] == "final_summary")
    return {
        "record_type": "affine_bias_baseline",
        "source": str(AFFINE_BASELINE_PATH),
        "rmse": summary["rmse"],
        "mae": summary["mae"],
        "maximum_absolute_error": summary["maximum_absolute_error"],
        "decision_accuracy": summary["decision_accuracy"],
        "keep_agreement": summary["keep_agreement"],
        "pop_agreement": summary["pop_agreement"],
        "mistake_count": summary["mistake_count"],
        "learned_score_coefficient": summary["learned_score_coefficient"],
        "learned_tolerance_coefficient": summary[
            "learned_tolerance_coefficient"
        ],
        "learned_bias": summary["learned_bias"],
    }


def _train(
    training: dict[str, np.ndarray], validation_set: dict[str, np.ndarray]
) -> tuple[nn.Linear, HomogeneousScaling, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED + 97)
    scaling = HomogeneousScaling(training["inputs"], training["target"])
    train_inputs = torch.as_tensor(
        scaling.transform_inputs(training["inputs"]), dtype=torch.float64
    )
    train_target = torch.as_tensor(
        scaling.transform_target(training["target"]), dtype=torch.float64
    )
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial_coefficients = scaling.original_coefficients(model).copy()
    if np.array_equal(initial_coefficients, np.ones(2, dtype=np.float64)):
        raise RuntimeError("forbidden a=b=1 initialization occurred")
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
            output = model(train_inputs[indices]).squeeze(-1)
            loss = nn.functional.mse_loss(output, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_mse += float(loss.item()) * len(indices)
        validation_prediction = _predict(
            model, scaling, validation_set["inputs"]
        )
        validation_mse = float(
            np.mean((validation_prediction - validation_set["target"]) ** 2)
        )
        metrics = {
            **margin._metrics(validation_set["target"], validation_prediction),
            **margin._decision_metrics(
                validation_set["label"], validation_prediction
            ),
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
                "training_scaled_mse": total_mse / len(order),
                "validation_margin_mse": validation_mse,
                "plateau_stale_epochs": stale,
                **metrics,
            }
        )
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"homogeneous epoch={epoch} rmse={metrics['rmse']:.6e} "
                f"max={metrics['maximum_absolute_error']:.6e} "
                f"decision={metrics['decision_accuracy']:.6f} stale={stale}",
                flush=True,
            )
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    model.eval()
    prediction = _predict(model, scaling, validation_set["inputs"])
    metrics = {
        **margin._metrics(validation_set["target"], prediction),
        **margin._decision_metrics(validation_set["label"], prediction),
    }
    coefficients = scaling.original_coefficients(model)
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "maximum_epochs": MAX_EPOCHS,
        "plateau_patience": PATIENCE,
        "training_events": len(training["target"]),
        "validation_events": len(validation_set["target"]),
        **metrics,
        "learned_a": float(coefficients[0]),
        "learned_b": float(coefficients[1]),
        "learned_bias": 0.0,
        "initial_a": float(initial_coefficients[0]),
        "initial_b": float(initial_coefficients[1]),
        "posthoc_a_error_from_one": float(abs(coefficients[0] - 1.0)),
        "posthoc_b_error_from_one": float(abs(coefficients[1] - 1.0)),
        "minimum_absolute_teacher_margin": float(
            np.min(np.abs(validation_set["target"]))
        ),
        "rmse_gate": RMSE_GATE,
        "maximum_error_gate": MAX_ERROR_GATE,
        "smoke_gate_passed": _gate(metrics),
        "classification_target_used": False,
        "homogeneous_no_bias": True,
        "input_mean_centering_used": False,
        "target_mean_centering_used": False,
    }
    rows.append(summary)
    rows.extend(_magnitude_rows(validation_set, prediction, "validation"))
    rows.append(_affine_baseline_row())
    return model, scaling, rows, summary


def _full_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _metric_row("final_summary", dataset, prediction, evaluation="full_pool")
    rows: list[dict[str, Any]] = [summary]
    rows.extend(_magnitude_rows(dataset, prediction, "full_pool"))
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
                "absolute_error": float(abs(prediction[index] - dataset["target"][index])),
                "absolute_distance_from_boundary": float(abs(dataset["target"][index])),
            }
        )
    rows.append(_affine_baseline_row())
    return rows, summary


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RECONSTRUCTION_PATH.exists():
        RECONSTRUCTION_PATH.unlink()
    locked_weight, checkpoint_hash = _load_locked_weight()
    print(f"verified locked skew checkpoint {checkpoint_hash}", flush=True)

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
    training = margin._margin_dataset(train_selected, train_states, locked_weight)
    validation_set = margin._margin_dataset(
        validation_selected, validation_states, locked_weight
    )
    model, scaling, smoke_rows, smoke_summary = _train(training, validation_set)
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
        print("homogeneous smoke gate failed; stopping", flush=True)
        return

    full_dataset = margin._margin_dataset(
        validation_pool, validation_states, locked_weight
    )
    full_prediction = _predict(model, scaling, full_dataset["inputs"])
    full_rows, full_summary = _full_rows(full_dataset, full_prediction)
    full_passed = _gate(full_summary)
    coefficients = scaling.original_coefficients(model)
    full_summary["full_pool_gate_passed"] = full_passed
    full_summary["locked_skew_checkpoint_sha256"] = checkpoint_hash
    full_summary["learned_a"] = float(coefficients[0])
    full_summary["learned_b"] = float(coefficients[1])
    full_summary["learned_bias"] = 0.0
    _write_csv(FULL_POOL_PATH, full_rows)
    if not full_passed:
        print(
            f"full-pool gate failed: mistakes={full_summary['mistake_count']}",
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
    reconstruction_rows, reconstruction_summary = margin._reconstruction_rows(
        locked_weight, coefficients, 0.0, reconstruction_states
    )
    _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)
    print(
        f"best_epoch={smoke_summary['best_epoch']} stop_epoch={smoke_summary['stop_epoch']} "
        f"a={coefficients[0]:.17g} b={coefficients[1]:.17g} "
        f"full_mistakes={full_summary['mistake_count']} "
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
