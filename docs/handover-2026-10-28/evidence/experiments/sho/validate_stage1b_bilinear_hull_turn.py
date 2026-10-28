#!/usr/bin/env python3
"""Validate and lock the frozen epoch-259 bilinear hull-turn score."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import experiments.sho.run_stage1b_edirect_vs_direct as edirect
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/bilinear_hull_turn_lock")
VALIDATION_PATH = OUTPUT_DIR / "bilinear_full_validation.csv"
RECONSTRUCTION_PATH = OUTPUT_DIR / "bilinear_hull_reconstruction.csv"
LOCK_SUMMARY_PATH = OUTPUT_DIR / "bilinear_checkpoint_lock.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "bilinear_hull_turn_epoch259_locked.pt"

SOURCE_RESULTS_PATH = Path(
    "outputs/sho/stage1B/hull_multiplicative_smoke/hull_bilinear_smoke.csv"
)
EXPECTED_SOURCE_EPOCH = 259
HELD_OUT_STATES = turn.VALIDATION_SOURCE_STATES
RECONSTRUCTION_STATES = 200
STAGE1_RMSE_GATE = 1.0e-12
STAGE1_MAX_ERROR_GATE = 1.0e-10


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


def _load_frozen_source() -> tuple[np.ndarray, dict[str, str], dict[str, str]]:
    with SOURCE_RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = next(row for row in rows if row["record_type"] == "final_summary")
    matrix = next(row for row in rows if row["record_type"] == "learned_matrix")
    if int(summary["best_epoch"]) != EXPECTED_SOURCE_EPOCH:
        raise RuntimeError("source is not the frozen epoch-259 checkpoint result")
    if summary["success"] != "True":
        raise RuntimeError("source epoch-259 result was not marked successful")
    weight = np.asarray(
        [
            [float(matrix["W_00"]), float(matrix["W_01"])],
            [float(matrix["W_10"]), float(matrix["W_11"])],
        ],
        dtype=np.float64,
    )
    return weight, summary, matrix


def _event_dataset(
    pool: dict[str, np.ndarray], states: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    count = len(pool["label"])
    inputs = np.empty((count, 4), dtype=np.float64)
    target = np.empty(count, dtype=np.float64)
    tau_geom = np.empty(count, dtype=np.float64)
    for row in range(count):
        state = states[int(pool["state_position"][row])]
        first, second, third = (
            int(value) for value in pool["triple_indices"][row]
        )
        sizes = state["sizes"]
        losses = state["losses"]
        delta_d_1 = float(sizes[second] - sizes[first])
        delta_f_1 = float(losses[second] - losses[first])
        delta_d_2 = float(sizes[third] - sizes[second])
        delta_f_2 = float(losses[third] - losses[second])
        norm_1 = math.hypot(delta_d_1, delta_f_1)
        norm_2 = math.hypot(delta_d_2, delta_f_2)
        if not norm_1 > 0.0 or not norm_2 > 0.0:
            raise RuntimeError("non-positive local hull segment norm")
        product = norm_1 * norm_2
        cross = delta_d_1 * delta_f_2 - delta_f_1 * delta_d_2
        if cross != float(pool["cross"][row]):
            raise RuntimeError("reconstructed cross differs from exact teacher")
        inputs[row] = (
            delta_d_1 / norm_1,
            delta_f_1 / norm_1,
            delta_d_2 / norm_2,
            delta_f_2 / norm_2,
        )
        target[row] = cross / product
        tau_geom[row] = float(pool["tolerance"][row]) / product
    exact_label = (target >= -tau_geom).astype(np.int64)
    if not np.array_equal(exact_label, pool["label"]):
        raise RuntimeError("normalized continuous teacher changed KEEP/POP labels")
    return {
        "inputs": inputs,
        "target": target,
        "tau_geom": tau_geom,
        "tau_raw": pool["tolerance"].copy(),
        "label": pool["label"].copy(),
        "state_position": pool["state_position"].copy(),
        "triple_indices": pool["triple_indices"].copy(),
    }


def _predict(weight: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    return np.einsum("bi,ij,bj->b", inputs[:, :2], weight, inputs[:, 2:])


def _continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    absolute = np.abs(error)
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 0.0 and np.std(prediction) > 0.0
        else 0.0
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(absolute)),
        "maximum_absolute_error": float(np.max(absolute)),
        "correlation": correlation,
        "absolute_error_p50": float(np.quantile(absolute, 0.50)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
        "absolute_error_p99": float(np.quantile(absolute, 0.99)),
        "absolute_error_p999": float(np.quantile(absolute, 0.999)),
        "absolute_error_p100": float(np.max(absolute)),
    }


def _decision_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    tolerance: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    predicted = prediction >= -tolerance
    teacher = labels.astype(bool)
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
        "teacher_recomputed_accuracy": float(
            np.mean((target >= -tolerance) == teacher)
        ),
    }


def _metric_row(
    record_type: str,
    subset: str,
    mask: np.ndarray,
    dataset: dict[str, np.ndarray],
    prediction: np.ndarray,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    row: dict[str, Any] = {
        "record_type": record_type,
        "subset": subset,
        "event_count": len(indices),
        "event_fraction": len(indices) / len(mask),
    }
    if len(indices):
        row.update(_continuous_metrics(dataset["target"][indices], prediction[indices]))
        row.update(
            _decision_metrics(
                dataset["target"][indices],
                prediction[indices],
                dataset["tau_geom"][indices],
                dataset["label"][indices],
            )
        )
        row["minimum_abs_y_geom"] = float(
            np.min(np.abs(dataset["target"][indices]))
        )
        row["maximum_abs_y_geom"] = float(
            np.max(np.abs(dataset["target"][indices]))
        )
    return row


def _full_validation_rows(
    dataset: dict[str, np.ndarray], prediction: np.ndarray
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    all_mask = np.ones(len(prediction), dtype=np.bool_)
    summary = _metric_row("final_summary", "all", all_mask, dataset, prediction)
    stage_pass = bool(
        summary["rmse"] <= STAGE1_RMSE_GATE
        and summary["maximum_absolute_error"] <= STAGE1_MAX_ERROR_GATE
        and summary["decision_accuracy"] == 1.0
        and summary["keep_agreement"] == 1.0
        and summary["pop_agreement"] == 1.0
    )
    summary.update(
        {
            "rmse_gate": STAGE1_RMSE_GATE,
            "maximum_absolute_error_gate": STAGE1_MAX_ERROR_GATE,
            "stage1_passed": stage_pass,
            "training_performed": False,
        }
    )
    rows = [summary]
    absolute_target = np.abs(dataset["target"])
    for threshold in (1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40):
        rows.append(
            _metric_row(
                "near_collinear_subset",
                f"abs_y_geom_lt_{threshold:.0e}",
                absolute_target < threshold,
                dataset,
                prediction,
            )
        )
    teacher = dataset["label"].astype(bool)
    predicted = prediction >= -dataset["tau_geom"]
    for index in np.flatnonzero(predicted != teacher):
        triple = dataset["triple_indices"][index]
        geometric_margin = dataset["target"][index] + dataset["tau_geom"][index]
        rows.append(
            {
                "record_type": "mistake",
                "event_index": int(index),
                "state_position": int(dataset["state_position"][index]),
                "triple_first": int(triple[0]),
                "triple_second": int(triple[1]),
                "triple_third": int(triple[2]),
                "teacher_decision": "KEEP" if teacher[index] else "POP",
                "learned_decision": "KEEP" if predicted[index] else "POP",
                "exact_y_geom": float(dataset["target"][index]),
                "learned_y_geom": float(prediction[index]),
                "raw_teacher_tolerance": float(dataset["tau_raw"][index]),
                "normalized_tolerance": float(dataset["tau_geom"][index]),
                "teacher_decision_boundary": float(-dataset["tau_geom"][index]),
                "signed_distance_from_boundary": float(geometric_margin),
                "absolute_distance_from_boundary": float(abs(geometric_margin)),
                "absolute_prediction_error": float(
                    abs(prediction[index] - dataset["target"][index])
                ),
            }
        )
    return rows, stage_pass, summary


def _frozen_turn(
    weight: np.ndarray,
    sizes: np.ndarray,
    losses: np.ndarray,
    first: int,
    second: int,
    third: int,
) -> tuple[float, float, bool]:
    delta_d_1 = float(sizes[second] - sizes[first])
    delta_f_1 = float(losses[second] - losses[first])
    delta_d_2 = float(sizes[third] - sizes[second])
    delta_f_2 = float(losses[third] - losses[second])
    norm_1 = math.hypot(delta_d_1, delta_f_1)
    norm_2 = math.hypot(delta_d_2, delta_f_2)
    if not norm_1 > 0.0 or not norm_2 > 0.0:
        raise RuntimeError("non-positive segment norm during hull reconstruction")
    first_unit = np.asarray((delta_d_1 / norm_1, delta_f_1 / norm_1))
    second_unit = np.asarray((delta_d_2 / norm_2, delta_f_2 / norm_2))
    predicted = float(first_unit @ weight @ second_unit)
    scale = max(
        1.0,
        abs(float(losses[first])),
        abs(float(losses[second])),
        abs(float(losses[third])),
    )
    normalized_tolerance = turn.HULL_TOLERANCE * scale / (norm_1 * norm_2)
    return predicted, normalized_tolerance, predicted >= -normalized_tolerance


def _frozen_hull(
    weight: np.ndarray, sizes: np.ndarray, losses: np.ndarray
) -> tuple[np.ndarray, int]:
    stack: list[int] = []
    decision_count = 0
    for third in range(len(sizes)):
        while len(stack) >= 2:
            _, _, keep = _frozen_turn(
                weight, sizes, losses, stack[-2], stack[-1], third
            )
            decision_count += 1
            if keep:
                break
            stack.pop()
        stack.append(third)
    return np.asarray(stack, dtype=np.int64), decision_count


def _improvement_filter(
    hull: np.ndarray, sizes: np.ndarray, losses: np.ndarray
) -> np.ndarray:
    target = float(np.min(losses)) - edirect.EDIRECT_EPSILON * abs(
        float(np.min(losses))
    )
    selected: list[int] = []
    for position, index_value in enumerate(hull):
        index = int(index_value)
        size = float(sizes[index])
        loss = float(losses[index])
        lower_slope = -math.inf
        upper_slope = math.inf
        if position > 0:
            previous = int(hull[position - 1])
            lower_slope = (loss - float(losses[previous])) / (
                size - float(sizes[previous])
            )
        if position < len(hull) - 1:
            following = int(hull[position + 1])
            upper_slope = (float(losses[following]) - loss) / (
                float(sizes[following]) - size
            )
        required = (loss - target) / size if size > 0.0 else math.inf
        lower = max(lower_slope, required, 0.0)
        if upper_slope > 0.0 and lower <= upper_slope + turn.HULL_TOLERANCE:
            selected.append(index)
    if not selected:
        selected.append(int(np.argmax(sizes)))
    return np.asarray(
        sorted(selected, key=lambda i: (sizes[i], losses[i], i)), dtype=np.int64
    )


def _state_minimum_abs_y(state: dict[str, Any]) -> float:
    values: list[float] = []
    for triple, cross in zip(
        state["triples"]["triple_indices"], state["triples"]["cross"]
    ):
        first, second, third = (int(value) for value in triple)
        sizes = state["sizes"]
        losses = state["losses"]
        norm_1 = math.hypot(
            float(sizes[second] - sizes[first]),
            float(losses[second] - losses[first]),
        )
        norm_2 = math.hypot(
            float(sizes[third] - sizes[second]),
            float(losses[third] - losses[second]),
        )
        values.append(abs(float(cross)) / (norm_1 * norm_2))
    return min(values) if values else math.inf


def _hull_size_bin(size: int) -> str:
    if size == 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 4:
        return "3_4"
    if size <= 8:
        return "5_8"
    return "9_plus"


def _reconstruction_summary(
    record_type: str, subset: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "subset": subset,
        "state_count": len(rows),
        "exact_hull_states": sum(bool(row["exact_hull"]) for row in rows),
        "exact_hull_fraction": float(np.mean([row["exact_hull"] for row in rows])),
        "false_hull_additions": sum(int(row["false_hull_additions"]) for row in rows),
        "false_hull_removals": sum(int(row["false_hull_removals"]) for row in rows),
        "maximum_false_hull_additions": max(
            int(row["false_hull_additions"]) for row in rows
        ),
        "maximum_false_hull_removals": max(
            int(row["false_hull_removals"]) for row in rows
        ),
        "exact_po_states": sum(bool(row["exact_po_set"]) for row in rows),
        "exact_po_fraction": float(np.mean([row["exact_po_set"] for row in rows])),
        "false_po_additions": sum(int(row["false_po_additions"]) for row in rows),
        "false_po_removals": sum(int(row["false_po_removals"]) for row in rows),
        "deterministic_teacher_exact_states": sum(
            bool(row["deterministic_teacher_exact"]) for row in rows
        ),
        "states_with_near_collinear_turn": sum(
            bool(row["contains_near_collinear_turn"]) for row in rows
        ),
        "minimum_abs_y_geom": min(float(row["minimum_abs_y_geom"]) for row in rows),
    }


def _reconstruction_rows(
    weight: np.ndarray, states: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    per_state: list[dict[str, Any]] = []
    for state in states:
        sizes = state["sizes"]
        losses = state["losses"]
        teacher_hull = np.asarray(state["teacher_hull"], dtype=np.int64)
        deterministic_hull, _ = turn._teacher_hull_and_triples(sizes, losses)
        learned_hull, decision_count = _frozen_hull(weight, sizes, losses)
        teacher_po = edirect._potentially_optimal_cells(sizes, losses)
        reconstructed_teacher_po = _improvement_filter(
            teacher_hull, sizes, losses
        )
        learned_po = _improvement_filter(learned_hull, sizes, losses)
        teacher_hull_set = set(int(value) for value in teacher_hull)
        learned_hull_set = set(int(value) for value in learned_hull)
        teacher_po_set = set(int(value) for value in teacher_po)
        learned_po_set = set(int(value) for value in learned_po)
        minimum_abs_y = _state_minimum_abs_y(state)
        per_state.append(
            {
                "record_type": "state",
                "source_state_index": state["source_state_index"],
                "trajectory_id": state["trajectory_id"],
                "iteration": state["iteration"],
                "history_count": state["history_count"],
                "state_bin": state["state_bin"],
                "representative_cell_count": len(sizes),
                "teacher_hull_size": len(teacher_hull),
                "hull_size_bin": _hull_size_bin(len(teacher_hull)),
                "learned_hull_size": len(learned_hull),
                "learned_turn_decisions": decision_count,
                "false_hull_additions": len(learned_hull_set - teacher_hull_set),
                "false_hull_removals": len(teacher_hull_set - learned_hull_set),
                "exact_hull": learned_hull_set == teacher_hull_set,
                "teacher_po_size": len(teacher_po),
                "learned_po_size": len(learned_po),
                "false_po_additions": len(learned_po_set - teacher_po_set),
                "false_po_removals": len(teacher_po_set - learned_po_set),
                "exact_po_set": learned_po_set == teacher_po_set,
                "deterministic_teacher_exact": bool(
                    np.array_equal(teacher_hull, deterministic_hull)
                    and np.array_equal(teacher_po, reconstructed_teacher_po)
                ),
                "teacher_near_collinear_turn_count": int(
                    np.sum(state["triples"]["near_collinear"])
                ),
                "teacher_exact_collinear_turn_count": int(
                    np.sum(state["triples"]["exact_collinear"])
                ),
                "contains_near_collinear_turn": bool(
                    np.any(state["triples"]["near_collinear"])
                ),
                "minimum_abs_y_geom": minimum_abs_y,
            }
        )
    summary = _reconstruction_summary("final_summary", "all", per_state)
    rows: list[dict[str, Any]] = [summary]
    for name, _, _ in turn.STATE_BINS:
        subset = [row for row in per_state if row["state_bin"] == name]
        if subset:
            rows.append(
                _reconstruction_summary(
                    "by_representative_cell_count", name, subset
                )
            )
    for name in ("1", "2", "3_4", "5_8", "9_plus"):
        subset = [row for row in per_state if row["hull_size_bin"] == name]
        if subset:
            rows.append(_reconstruction_summary("by_hull_size", name, subset))
    near_subset = [row for row in per_state if row["contains_near_collinear_turn"]]
    if near_subset:
        rows.append(
            _reconstruction_summary(
                "near_collinear_states", "contains_near_collinear_turn", near_subset
            )
        )
    for threshold in (1.0e-6, 1.0e-12, 1.0e-24, 1.0e-40):
        subset = [
            row for row in per_state if float(row["minimum_abs_y_geom"]) < threshold
        ]
        if subset:
            rows.append(
                _reconstruction_summary(
                    "extreme_near_collinear_states",
                    f"minimum_abs_y_geom_lt_{threshold:.0e}",
                    subset,
                )
            )
    rows.extend(per_state)
    passed = bool(
        summary["exact_hull_fraction"] == 1.0
        and summary["exact_po_fraction"] == 1.0
        and summary["deterministic_teacher_exact_states"] == len(per_state)
    )
    summary["stage2_passed"] = passed
    return rows, passed, summary


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weight, source_summary, source_matrix = _load_frozen_source()
    source_hash = _sha256(SOURCE_RESULTS_PATH)
    payload = torch.load(turn.RAW_TRACE_PATH, map_location="cpu", weights_only=False)
    validation_indices = turn._stratified_state_indices(
        payload["states"],
        "validation",
        HELD_OUT_STATES,
        turn.DATA_SEED + 1,
    )
    validation_states = turn._build_states(
        payload, validation_indices, workers, "full validation"
    )
    validation_pool = turn._flatten_triples(validation_states)
    dataset = _event_dataset(validation_pool, validation_states)
    prediction = _predict(weight, dataset["inputs"])
    validation_rows, stage1_passed, validation_summary = _full_validation_rows(
        dataset, prediction
    )
    _write_csv(VALIDATION_PATH, validation_rows)

    stage2_run = False
    stage2_passed = False
    reconstruction_summary: dict[str, Any] = {}
    if stage1_passed:
        reconstruction_indices = turn._stratified_state_indices(
            payload["states"],
            "reconstruction",
            RECONSTRUCTION_STATES,
            turn.DATA_SEED + 2,
        )
        reconstruction_states = turn._build_states(
            payload, reconstruction_indices, workers, "hull reconstruction"
        )
        reconstruction_rows, stage2_passed, reconstruction_summary = (
            _reconstruction_rows(weight, reconstruction_states)
        )
        stage2_run = True
        _write_csv(RECONSTRUCTION_PATH, reconstruction_rows)
    else:
        _write_csv(
            RECONSTRUCTION_PATH,
            [
                {
                    "record_type": "not_run",
                    "reason": "full held-out continuous validation failed hard gate",
                    "stage1_passed": False,
                }
            ],
        )

    locked = bool(stage1_passed and stage2_passed)
    checkpoint_hash = ""
    checkpoint_readback_exact = False
    if locked:
        checkpoint = {
            "format_version": 1,
            "component": "hull_turn_continuous_score",
            "lock_status": "LOCKED",
            "architecture": "y_hat = u1^T W u2",
            "source_epoch": EXPECTED_SOURCE_EPOCH,
            "state_dict": {
                "weight": torch.as_tensor(weight.copy(), dtype=torch.float64)
            },
            "source_results_path": str(SOURCE_RESULTS_PATH),
            "source_results_sha256": source_hash,
            "training_performed_during_locking": False,
            "full_validation_metrics": validation_summary,
            "hull_reconstruction_metrics": reconstruction_summary,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        checkpoint_hash = _sha256(CHECKPOINT_PATH)
        restored = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        restored_weight = restored["state_dict"]["weight"].cpu().numpy()
        checkpoint_readback_exact = bool(np.array_equal(restored_weight, weight))
        if not checkpoint_readback_exact:
            raise RuntimeError("locked checkpoint failed exact readback")

    lock_rows = [
        {
            "record_type": "lock_summary",
            "hull_turn_continuous_score": "LOCKED" if locked else "NOT_LOCKED",
            "locked": locked,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_epoch": EXPECTED_SOURCE_EPOCH,
            "source_results_path": str(SOURCE_RESULTS_PATH),
            "source_results_sha256": source_hash,
            "checkpoint_path": str(CHECKPOINT_PATH) if locked else "",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_readback_exact": checkpoint_readback_exact,
            "training_performed": False,
            "keep_pop_network_trained": False,
            "stage1_passed": stage1_passed,
            "stage2_run": stage2_run,
            "stage2_passed": stage2_passed,
            "full_validation_event_count": validation_summary["event_count"],
            "full_validation_rmse": validation_summary["rmse"],
            "full_validation_maximum_absolute_error": validation_summary[
                "maximum_absolute_error"
            ],
            "full_validation_decision_accuracy": validation_summary[
                "decision_accuracy"
            ],
            "full_validation_mistake_count": validation_summary["mistake_count"],
            "reconstruction_state_count": reconstruction_summary.get(
                "state_count", ""
            ),
            "exact_hull_fraction": reconstruction_summary.get(
                "exact_hull_fraction", ""
            ),
            "exact_po_fraction": reconstruction_summary.get(
                "exact_po_fraction", ""
            ),
            "W_00": source_matrix["W_00"],
            "W_01": source_matrix["W_01"],
            "W_10": source_matrix["W_10"],
            "W_11": source_matrix["W_11"],
            "source_validation_rmse": source_summary["rmse"],
            "source_validation_maximum_absolute_error": source_summary[
                "maximum_absolute_error"
            ],
        }
    ]
    _write_csv(LOCK_SUMMARY_PATH, lock_rows)
    print(
        f"stage1_passed={stage1_passed} events={validation_summary['event_count']} "
        f"rmse={validation_summary['rmse']:.6e} "
        f"mistakes={validation_summary['mistake_count']}",
        flush=True,
    )
    print(
        f"stage2_run={stage2_run} stage2_passed={stage2_passed} locked={locked}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    arguments = parser.parse_args()
    run(arguments.workers)


if __name__ == "__main__":
    main()
