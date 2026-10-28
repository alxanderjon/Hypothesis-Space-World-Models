#!/usr/bin/env python3
"""Learn only the finite neighboring lower-hull slope operation."""

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

import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hull_neighbor_slope_smoke")
AUDIT_PATH = OUTPUT_DIR / "hull_slope_audit.txt"
SMOKE_PATH = OUTPUT_DIR / "hull_slope_smoke.csv"
FULL_PATH = OUTPUT_DIR / "hull_slope_full_pool.csv"

TRAIN_TARGET = 5_000
VALIDATION_TARGET = 1_000
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 256
LEARNING_RATE = 1.0e-3
HIDDEN_DIM = 64
MODEL_SEED = smoke.MODEL_SEED + 3_141_592
NORMALIZED_RMSE_GATE = 1.0e-5
CORRELATION_GATE = 0.999999
RELATIVE_ERROR_P99_GATE = 1.0e-2


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


def _build_slope_pool(states: list[dict[str, Any]], split: str) -> dict[str, np.ndarray]:
    delta_d: list[float] = []
    delta_f: list[float] = []
    target: list[float] = []
    state_position: list[int] = []
    left_index: list[int] = []
    right_index: list[int] = []
    role: list[str] = []
    unique_edge_id: list[int] = []
    edge_id = 0
    for position, state in enumerate(states):
        hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        sizes = state["sizes"]
        losses = state["losses"]
        for pair_position in range(len(hull) - 1):
            left = int(hull[pair_position])
            right = int(hull[pair_position + 1])
            dd = float(sizes[right] - sizes[left])
            df = float(losses[right] - losses[left])
            if not dd > 0.0:
                raise RuntimeError(
                    f"delta_d <= 0 in {split} state={position} edge={pair_position}: {dd}"
                )
            if not math.isfinite(dd) or not math.isfinite(df):
                raise RuntimeError("non-finite adjacent-hull delta")
            slope = df / dd
            if not math.isfinite(slope):
                raise RuntimeError("non-finite finite-neighbor teacher slope")
            # The same edge is K_next for L and K_prev for R. Keep both real roles.
            for edge_role in ("K_next_of_left", "K_prev_of_right"):
                delta_d.append(dd)
                delta_f.append(df)
                target.append(slope)
                state_position.append(position)
                left_index.append(left)
                right_index.append(right)
                role.append(edge_role)
                unique_edge_id.append(edge_id)
            edge_id += 1
    return {
        "inputs": np.column_stack(
            (np.asarray(delta_d, dtype=np.float64), np.asarray(delta_f, dtype=np.float64))
        ),
        "target": np.asarray(target, dtype=np.float64),
        "state_position": np.asarray(state_position, dtype=np.int64),
        "left_index": np.asarray(left_index, dtype=np.int64),
        "right_index": np.asarray(right_index, dtype=np.int64),
        "role": np.asarray(role, dtype=object),
        "unique_edge_id": np.asarray(unique_edge_id, dtype=np.int64),
        "unique_edge_count": np.asarray([edge_id], dtype=np.int64),
    }


def _select(pool: dict[str, np.ndarray], count: int, seed: int) -> dict[str, np.ndarray]:
    total = len(pool["target"])
    if total <= count:
        indices = np.arange(total, dtype=np.int64)
    else:
        generator = np.random.default_rng(seed)
        indices = generator.choice(total, size=count, replace=False)
    return {
        key: values[indices] if key != "unique_edge_count" else values.copy()
        for key, values in pool.items()
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.percentile(values, 1.0)),
        "p05": float(np.percentile(values, 5.0)),
        "p25": float(np.percentile(values, 25.0)),
        "p50": float(np.percentile(values, 50.0)),
        "p75": float(np.percentile(values, 75.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "maximum": float(np.max(values)),
    }


def _write_audit(
    train_pool: dict[str, np.ndarray], validation_pool: dict[str, np.ndarray]
) -> None:
    combined_delta_d = np.concatenate(
        (train_pool["inputs"][:, 0], validation_pool["inputs"][:, 0])
    )
    combined_delta_f = np.concatenate(
        (train_pool["inputs"][:, 1], validation_pool["inputs"][:, 1])
    )
    combined_k = np.concatenate((train_pool["target"], validation_pool["target"]))

    def lines(name: str, values: np.ndarray) -> str:
        stats = _distribution(values)
        return "\n".join(
            [f"  {name}:"]
            + [f"    {key} = {value:.17g}" for key, value in stats.items()]
        )

    text = f"""Finite neighboring-hull slope audit
====================================

Scope
-----
No locked network was loaded, changed, or retrained. No improvement PASS/FAIL model,
hull reconstruction, or trajectory search was run.

Exact audited teacher operation
-------------------------------
The implementation orders same-size-group representatives by strictly increasing cell
size d. For every adjacent pair L,R on the completed lower hull it computes exactly:

    delta_d = d_R - d_L
    delta_f = f_R - f_L
    K       = delta_f / delta_d

This same quotient is `K_next` for L and `K_prev` for R. There is no epsilon, clamp,
tolerance, stabilization, or special floating-point branch in the finite division.

Same-size grouping starts a new group only when the size differs by more than 1e-14.
All {int(train_pool['unique_edge_count'][0]) + int(validation_pool['unique_edge_count'][0])}
audited adjacent edges satisfied `delta_d > 0`; no zero or non-positive denominator
occurred, and no epsilon was inserted.

Endpoint handling
-----------------
The first hull representative has no previous neighbor, so the improvement code assigns
`K_prev = -inf` directly. The last hull representative has no next neighbor, so it
assigns `K_next = +inf` directly. Those {2 * (turn.TRAIN_SOURCE_STATES + turn.VALIDATION_SOURCE_STATES)}
endpoint sentinels are not quotient events and are excluded from this regression.
Every actual adjacent edge contributes twice to the shared dataset: once in its
`K_next_of_left` role and once in its `K_prev_of_right` role. State-level train and
validation splits prevent duplicate-role leakage.

Available real data
-------------------
Training states: {turn.TRAIN_SOURCE_STATES}
Held-out validation states: {turn.VALIDATION_SOURCE_STATES}
Training unique adjacent edges: {int(train_pool['unique_edge_count'][0])}
Training role events: {len(train_pool['target'])}
Held-out unique adjacent edges: {int(validation_pool['unique_edge_count'][0])}
Held-out role events: {len(validation_pool['target'])}

Pre-training distributions (train and held-out pools combined; duplicate roles retained)
---------------------------------------------------------------------------------------
{lines('delta_d', combined_delta_d)}
{lines('delta_f', combined_delta_f)}
{lines('K_teacher', combined_k)}
"""
    AUDIT_PATH.write_text(text, encoding="utf-8")


class Standardization:
    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        self.input_mean = np.mean(inputs, axis=0)
        self.input_scale = np.std(inputs, axis=0)
        self.target_mean = float(np.mean(target))
        self.target_scale = float(np.std(target))
        if np.any(self.input_scale <= 0.0) or not np.all(np.isfinite(self.input_scale)):
            raise RuntimeError("invalid input standardization")
        if not self.target_scale > 0.0 or not math.isfinite(self.target_scale):
            raise RuntimeError("invalid target standardization")

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return (values - self.input_mean) / self.input_scale

    def target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_scale

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_scale + self.target_mean


class SlopeMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(-1)


def _predict(model: SlopeMLP, scaling: Standardization, inputs: np.ndarray) -> np.ndarray:
    values = torch.as_tensor(scaling.inputs(inputs), dtype=torch.float64)
    parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 4096):
            parts.append(model(values[start : start + 4096]))
    return scaling.inverse_target(torch.cat(parts).cpu().numpy())


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    target_rms = float(np.sqrt(np.mean(target**2)))
    meaningful = np.abs(target) > np.finfo(np.float64).tiny
    relative = absolute[meaningful] / np.abs(target[meaningful])
    predicted_positive = prediction > 0.0
    teacher_positive = target > 0.0
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "target_rms": target_rms,
        "normalized_rmse_by_target_rms": (
            float(np.sqrt(np.mean(error**2))) / target_rms if target_rms else math.inf
        ),
        "correlation": float(np.corrcoef(target, prediction)[0, 1]),
        "relative_error_p50": float(np.percentile(relative, 50.0)),
        "relative_error_p90": float(np.percentile(relative, 90.0)),
        "relative_error_p99": float(np.percentile(relative, 99.0)),
        "maximum_relative_error": float(np.max(relative)),
        "sign_agreement": float(np.mean(predicted_positive == teacher_positive)),
        "slope_sign_mistakes": int(np.sum(predicted_positive != teacher_positive)),
        "smallest_absolute_K": float(np.min(np.abs(target))),
    }


def _subset_row(
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
    mask: np.ndarray,
    subset_type: str,
    subset: str,
    split: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "record_type": "subset",
        "split": split,
        "subset_type": subset_type,
        "subset": subset,
        "event_count": int(np.sum(mask)),
    }
    if np.any(mask):
        row.update(_metrics(dataset["target"][mask], prediction[mask]))
    return row


def _diagnostic_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantities = {
        "delta_d": dataset["inputs"][:, 0],
        "abs_delta_f": np.abs(dataset["inputs"][:, 1]),
        "abs_K": np.abs(dataset["target"]),
    }
    for name, values in quantities.items():
        boundaries = np.quantile(values, (0.0, 0.25, 0.50, 0.75, 1.0))
        for quartile in range(4):
            if quartile == 3:
                mask = (values >= boundaries[quartile]) & (
                    values <= boundaries[quartile + 1]
                )
            else:
                mask = (values >= boundaries[quartile]) & (
                    values < boundaries[quartile + 1]
                )
            rows.append(
                _subset_row(
                    dataset,
                    prediction,
                    mask,
                    f"{name}_quartile",
                    f"Q{quartile + 1}",
                    split,
                )
            )
    target = dataset["target"]
    rows.append(_subset_row(dataset, prediction, target > 0.0, "sign", "positive_K", split))
    rows.append(_subset_row(dataset, prediction, target < 0.0, "sign", "negative_K", split))
    rows.append(
        _subset_row(
            dataset,
            prediction,
            np.abs(target) < 1.0e-2,
            "magnitude",
            "abs_K_lt_1e-2",
            split,
        )
    )
    cutoff = float(np.quantile(np.abs(target), 0.10))
    rows.append(
        _subset_row(
            dataset,
            prediction,
            np.abs(target) <= cutoff,
            "magnitude",
            "lowest_abs_K_decile",
            split,
        )
    )
    return rows


def _train(
    training: dict[str, np.ndarray], validation_set: dict[str, np.ndarray]
) -> tuple[SlopeMLP, Standardization, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    generator = np.random.default_rng(MODEL_SEED + 97)
    scaling = Standardization(training["inputs"], training["target"])
    train_inputs = torch.as_tensor(scaling.inputs(training["inputs"]), dtype=torch.float64)
    train_target = torch.as_tensor(scaling.target(training["target"]), dtype=torch.float64)
    model = SlopeMLP().to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best_mse = math.inf
    best_epoch = 0
    stale = 0
    stopping_reason = "maximum_epochs"
    rows: list[dict[str, Any]] = []
    for epoch in range(1, MAX_EPOCHS + 1):
        order = generator.permutation(len(train_target))
        model.train()
        train_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = torch.as_tensor(order[start : start + BATCH_SIZE], dtype=torch.long)
            output = model(train_inputs[indices])
            loss = nn.functional.mse_loss(output, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(indices)
        validation_prediction = _predict(model, scaling, validation_set["inputs"])
        validation_metrics = _metrics(validation_set["target"], validation_prediction)
        validation_mse = validation_metrics["rmse"] ** 2
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        rows.append(
            {
                "record_type": "epoch",
                "epoch": epoch,
                "training_standardized_mse": train_loss / len(order),
                "validation_mse": validation_mse,
                "stale_epochs": stale,
                **validation_metrics,
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"slope epoch={epoch} nrmse={validation_metrics['normalized_rmse_by_target_rms']:.6e} "
                f"corr={validation_metrics['correlation']:.8f} "
                f"sign_mistakes={validation_metrics['slope_sign_mistakes']} stale={stale}",
                flush=True,
            )
        if stale >= PATIENCE:
            stopping_reason = "validation_plateau"
            break
    model.load_state_dict(best_state)
    prediction = _predict(model, scaling, validation_set["inputs"])
    metrics = _metrics(validation_set["target"], prediction)
    passed = bool(
        metrics["normalized_rmse_by_target_rms"] <= NORMALIZED_RMSE_GATE
        and metrics["correlation"] >= CORRELATION_GATE
        and metrics["relative_error_p99"] <= RELATIVE_ERROR_P99_GATE
        and metrics["sign_agreement"] == 1.0
        and metrics["slope_sign_mistakes"] == 0
    )
    summary = {
        "record_type": "final_summary",
        "best_epoch": best_epoch,
        "stop_epoch": len(rows),
        "stopping_reason": stopping_reason,
        "training_events": len(training["target"]),
        "validation_events": len(validation_set["target"]),
        **metrics,
        "normalized_rmse_gate": NORMALIZED_RMSE_GATE,
        "correlation_gate": CORRELATION_GATE,
        "relative_error_p99_gate": RELATIVE_ERROR_P99_GATE,
        "continuous_supervision_only": True,
        "sign_supervision_used": False,
        "smoke_gate_passed": passed,
    }
    rows.append(summary)
    rows.extend(_diagnostic_rows(validation_set, prediction, "validation"))
    return model, scaling, rows, summary


def _full_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "record_type": "final_summary",
            "event_count": len(dataset["target"]),
            **_metrics(dataset["target"], prediction),
        }
    ]
    rows.extend(_diagnostic_rows(dataset, prediction, "full_held_out_pool"))
    teacher_positive = dataset["target"] > 0.0
    predicted_positive = prediction > 0.0
    for index in np.flatnonzero(teacher_positive != predicted_positive):
        rows.append(
            {
                "record_type": "sign_mistake",
                "event_index": int(index),
                "state_position": int(dataset["state_position"][index]),
                "role": dataset["role"][index],
                "delta_d": float(dataset["inputs"][index, 0]),
                "delta_f": float(dataset["inputs"][index, 1]),
                "K_teacher": float(dataset["target"][index]),
                "K_hat": float(prediction[index]),
                "absolute_error": float(abs(prediction[index] - dataset["target"][index])),
            }
        )
    return rows


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if FULL_PATH.exists():
        FULL_PATH.unlink()
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
    train_states = turn._build_states(payload, train_indices, workers, "slope train")
    validation_states = turn._build_states(
        payload, validation_indices, workers, "slope validation"
    )
    train_pool = _build_slope_pool(train_states, "train")
    validation_pool = _build_slope_pool(validation_states, "validation")
    _write_audit(train_pool, validation_pool)
    training = _select(train_pool, TRAIN_TARGET, MODEL_SEED + 11)
    validation_set = _select(validation_pool, VALIDATION_TARGET, MODEL_SEED + 12)
    model, scaling, rows, summary = _train(training, validation_set)
    _write_csv(SMOKE_PATH, rows)
    if not bool(summary["smoke_gate_passed"]):
        print("slope smoke gate failed; stopped before full held-out pool", flush=True)
        return
    full_prediction = _predict(model, scaling, validation_pool["inputs"])
    full_rows = _full_rows(validation_pool, full_prediction)
    _write_csv(FULL_PATH, full_rows)
    print(
        f"slope smoke passed; full events={len(validation_pool['target'])} "
        f"sign_mistakes={full_rows[0]['slope_sign_mistakes']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
