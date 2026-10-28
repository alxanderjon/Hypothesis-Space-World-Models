#!/usr/bin/env python3
"""Learn only the continuous normalized representative-selection margin."""

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


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/representative_margin")
SMOKE_PATH = OUTPUT_DIR / "representative_margin_smoke.csv"
FULL_PATH = OUTPUT_DIR / "representative_margin_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "representative_margin_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

MODEL_SEED = smoke.MODEL_SEED + 11_200_003
LEARNING_RATE = 2.0e-2
MAX_EPOCHS = 10_000
STABILITY_EPOCHS = 50
COEFFICIENT_GATE = 1.0e-12
EXPECTED_FULL_EVENT_COUNT = 10_406


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


def _flatten_comparisons(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {
        "features": [],
        "target": [],
        "replace": [],
        "f_best": [],
        "f_i": [],
        "scale": [],
        "state_position": [],
        "source_state_index": [],
        "group_index": [],
        "comparison_position": [],
        "current_best_index": [],
        "challenger_index": [],
    }
    for state_position, state in enumerate(states):
        trace = grouping_audit._teacher_trace(state["sizes"], state["losses"])
        losses = state["losses"]
        for group_index, group in enumerate(trace["groups"]):
            members = [int(value) for value in group["members"]]
            best_index = members[0]
            for comparison_position, challenger_index in enumerate(members[1:], start=1):
                f_best = float(losses[best_index])
                f_i = float(losses[challenger_index])
                scale = max(abs(f_best), abs(f_i))
                if not scale > 0.0:
                    raise RuntimeError(
                        "representative margin encountered nonpositive scale in "
                        f"state {state['source_state_index']} group {group_index}"
                    )
                x_best = f_best / scale
                x_i = f_i / scale
                target = x_best - x_i
                replace = f_i < f_best
                values["features"].append((x_best, x_i))
                values["target"].append(target)
                values["replace"].append(replace)
                values["f_best"].append(f_best)
                values["f_i"].append(f_i)
                values["scale"].append(scale)
                values["state_position"].append(state_position)
                values["source_state_index"].append(state["source_state_index"])
                values["group_index"].append(group_index)
                values["comparison_position"].append(comparison_position)
                values["current_best_index"].append(best_index)
                values["challenger_index"].append(challenger_index)
                if replace:
                    best_index = challenger_index
    return {
        "features": np.asarray(values["features"], dtype=np.float64),
        "target": np.asarray(values["target"], dtype=np.float64),
        "replace": np.asarray(values["replace"], dtype=bool),
        "f_best": np.asarray(values["f_best"], dtype=np.float64),
        "f_i": np.asarray(values["f_i"], dtype=np.float64),
        "scale": np.asarray(values["scale"], dtype=np.float64),
        "state_position": np.asarray(values["state_position"], dtype=np.int64),
        "source_state_index": np.asarray(
            values["source_state_index"], dtype=np.int64
        ),
        "group_index": np.asarray(values["group_index"], dtype=np.int64),
        "comparison_position": np.asarray(
            values["comparison_position"], dtype=np.int64
        ),
        "current_best_index": np.asarray(
            values["current_best_index"], dtype=np.int64
        ),
        "challenger_index": np.asarray(values["challenger_index"], dtype=np.int64),
    }


def _balanced_indices(pool: dict[str, np.ndarray], seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    replace = np.flatnonzero(pool["replace"])
    keep = np.flatnonzero(~pool["replace"])
    count = min(len(replace), len(keep))
    if count == 0:
        raise RuntimeError("cannot balance REPLACE and KEEP from real comparisons")
    selected = np.concatenate(
        (replace, generator.choice(keep, size=count, replace=False))
    )
    generator.shuffle(selected)
    return selected


def _predict(model: nn.Linear, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            model(torch.as_tensor(features, dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )


def _metrics(
    model: nn.Linear, pool: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    teacher = pool["target"][indices]
    replace = pool["replace"][indices]
    predicted = _predict(model, pool["features"][indices])
    errors = predicted - teacher
    predicted_replace = predicted > 0.0
    mistakes = predicted_replace != replace
    replace_errors = errors[replace]
    keep_errors = errors[~replace]
    correlation = float(np.corrcoef(predicted, teacher)[0, 1])
    return {
        "event_count": len(indices),
        "REPLACE_count": int(np.sum(replace)),
        "KEEP_count": int(np.sum(~replace)),
        "RMSE": float(np.sqrt(np.mean(errors * errors))),
        "MAE": float(np.mean(np.abs(errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "correlation": correlation,
        "REPLACE_RMSE": float(
            np.sqrt(np.mean(replace_errors * replace_errors))
        ),
        "REPLACE_MAE": float(np.mean(np.abs(replace_errors))),
        "REPLACE_max_absolute_error": float(np.max(np.abs(replace_errors))),
        "KEEP_RMSE": float(np.sqrt(np.mean(keep_errors * keep_errors))),
        "KEEP_MAE": float(np.mean(np.abs(keep_errors))),
        "KEEP_max_absolute_error": float(np.max(np.abs(keep_errors))),
        "overall_accuracy": float(np.mean(~mistakes)),
        "REPLACE_accuracy": float(np.mean(~mistakes[replace])),
        "KEEP_accuracy": float(np.mean(~mistakes[~replace])),
        "mistake_count": int(np.sum(mistakes)),
        "predicted_margin": predicted,
        "errors": errors,
        "predicted_replace": predicted_replace,
        "mistakes": mistakes,
    }


def _train(
    train_pool: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_pool: dict[str, np.ndarray],
    validation_indices: np.ndarray,
) -> tuple[nn.Linear, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(2, 1, bias=False).to(torch.float64)
    initial_weights = model.weight.detach().cpu().numpy()[0].copy()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_features = torch.as_tensor(
        train_pool["features"][train_indices], dtype=torch.float64
    )
    train_target = torch.as_tensor(
        train_pool["target"][train_indices], dtype=torch.float64
    )
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    exact_streak = 0
    stop_epoch = MAX_EPOCHS
    stopping_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        prediction = model(train_features).squeeze(-1)
        loss = torch.mean(torch.square(prediction - train_target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_metrics = _metrics(
            model, validation_pool, validation_indices
        )
        weights = model.weight.detach().cpu().numpy()[0]
        coefficient_error = max(abs(float(weights[0]) - 1.0), abs(float(weights[1]) + 1.0))
        key = (validation_metrics["mistake_count"], coefficient_error)
        if key < best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        exact_now = bool(
            validation_metrics["mistake_count"] == 0
            and coefficient_error <= COEFFICIENT_GATE
        )
        exact_streak = exact_streak + 1 if exact_now else 0
        if exact_streak >= STABILITY_EPOCHS:
            stop_epoch = epoch
            stopping_reason = "stable_numerical_convergence"
            break
        if epoch == 1 or epoch % 500 == 0:
            print(
                f"representative margin epoch={epoch} "
                f"w_best={weights[0]:.17g} w_i={weights[1]:.17g} "
                f"coefficient_error={coefficient_error:.3e} "
                f"mistakes={validation_metrics['mistake_count']}",
                flush=True,
            )
    model.load_state_dict(best_state)
    metrics = _metrics(model, validation_pool, validation_indices)
    weights = model.weight.detach().cpu().numpy()[0]
    summary = {
        "record_type": "summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "training_event_count": len(train_indices),
        "validation_event_count": len(validation_indices),
        "training_REPLACE_count": int(np.sum(train_pool["replace"][train_indices])),
        "training_KEEP_count": int(np.sum(~train_pool["replace"][train_indices])),
        "validation_REPLACE_count": metrics["REPLACE_count"],
        "validation_KEEP_count": metrics["KEEP_count"],
        "learned_w_best": float(weights[0]),
        "learned_w_i": float(weights[1]),
        "bias": 0.0,
        "initial_w_best": float(initial_weights[0]),
        "initial_w_i": float(initial_weights[1]),
        "coefficient_max_error": max(
            abs(float(weights[0]) - 1.0), abs(float(weights[1]) + 1.0)
        ),
        **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)},
    }
    summary["smoke_gate_passed"] = bool(
        metrics["mistake_count"] == 0
        and summary["coefficient_max_error"] <= COEFFICIENT_GATE
    )
    return model, summary


def _mistake_rows(
    pool: dict[str, np.ndarray],
    indices: np.ndarray,
    metrics: dict[str, Any],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_index in np.flatnonzero(metrics["mistakes"]):
        event_index = int(indices[local_index])
        rows.append(
            {
                "record_type": "mistake",
                "split": split,
                "event_index": event_index,
                "source_state_index": int(pool["source_state_index"][event_index]),
                "group_index": int(pool["group_index"][event_index]),
                "comparison_position": int(
                    pool["comparison_position"][event_index]
                ),
                "f_best": float(pool["f_best"][event_index]),
                "f_i": float(pool["f_i"][event_index]),
                "scale": float(pool["scale"][event_index]),
                "x_best": float(pool["features"][event_index, 0]),
                "x_i": float(pool["features"][event_index, 1]),
                "teacher_margin": float(pool["target"][event_index]),
                "learned_margin": float(metrics["predicted_margin"][local_index]),
                "teacher_REPLACE": bool(pool["replace"][event_index]),
                "predicted_REPLACE": bool(
                    metrics["predicted_replace"][local_index]
                ),
            }
        )
    return rows


def _bookkeeping_reconstruction(
    model: nn.Linear, states: list[dict[str, Any]], split: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    weights = model.weight.detach().cpu().numpy()[0]
    rows: list[dict[str, Any]] = []
    exact_groups = 0
    total_groups = 0
    decision_changes = 0
    exact_states = 0
    for state_position, state in enumerate(states):
        trace = grouping_audit._teacher_trace(state["sizes"], state["losses"])
        losses = state["losses"]
        learned_representatives: list[int] = []
        state_exact = True
        for group in trace["groups"]:
            total_groups += 1
            members = [int(value) for value in group["members"]]
            best_index = members[0]
            for challenger_index in members[1:]:
                f_best = float(losses[best_index])
                f_i = float(losses[challenger_index])
                scale = max(abs(f_best), abs(f_i))
                if not scale > 0.0:
                    raise RuntimeError("nonpositive representative comparison scale")
                learned_margin = (
                    float(weights[0]) * (f_best / scale)
                    + float(weights[1]) * (f_i / scale)
                )
                teacher_replace = f_i < f_best
                learned_replace = learned_margin > 0.0
                decision_changes += int(learned_replace != teacher_replace)
                if learned_replace:
                    best_index = challenger_index
            teacher_index = int(group["representative_index"])
            exact = best_index == teacher_index
            exact_groups += int(exact)
            state_exact = state_exact and exact
            learned_representatives.append(best_index)
        ordered_exact = bool(
            np.array_equal(
                np.asarray(learned_representatives, dtype=np.int64),
                trace["representatives"],
            )
        )
        state_exact = state_exact and ordered_exact
        exact_states += int(state_exact)
        rows.append(
            {
                "record_type": "state_reconstruction",
                "split": split,
                "state_position": state_position,
                "source_state_index": state["source_state_index"],
                "group_count": len(trace["groups"]),
                "exact_ordered_representative_list": ordered_exact,
                "exact_state": state_exact,
            }
        )
    return rows, {
        f"{split}_state_count": len(states),
        f"{split}_group_count": total_groups,
        f"{split}_exact_final_representative_groups": exact_groups,
        f"{split}_exact_ordered_representative_states": exact_states,
        f"{split}_replacement_decision_changes": decision_changes,
    }


def run(workers: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    train_pool = _flatten_comparisons(train_states)
    validation_pool = _flatten_comparisons(validation_states)
    train_indices = _balanced_indices(train_pool, MODEL_SEED + 1)
    validation_indices = _balanced_indices(validation_pool, MODEL_SEED + 2)
    model, smoke_summary = _train(
        train_pool, train_indices, validation_pool, validation_indices
    )
    smoke_metrics = _metrics(model, validation_pool, validation_indices)
    smoke_rows: list[dict[str, Any]] = [smoke_summary]
    smoke_rows.extend(
        _mistake_rows(
            validation_pool,
            validation_indices,
            smoke_metrics,
            "balanced_held_out",
        )
    )
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        print("representative continuous margin smoke gate failed; stopped", flush=True)
        return

    full_pool = {
        key: np.concatenate((train_pool[key], validation_pool[key]), axis=0)
        for key in train_pool
    }
    if len(full_pool["target"]) != EXPECTED_FULL_EVENT_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_FULL_EVENT_COUNT} audited comparisons, got "
            f"{len(full_pool['target'])}"
        )
    full_indices = np.arange(len(full_pool["target"]), dtype=np.int64)
    full_metrics = _metrics(model, full_pool, full_indices)
    train_state_rows, train_reconstruction = _bookkeeping_reconstruction(
        model, train_states, "training"
    )
    held_state_rows, held_reconstruction = _bookkeeping_reconstruction(
        model, validation_states, "held_out"
    )
    full_summary = {
        "record_type": "summary",
        "full_event_count": len(full_indices),
        "learned_w_best": float(model.weight.detach().cpu().numpy()[0, 0]),
        "learned_w_i": float(model.weight.detach().cpu().numpy()[0, 1]),
        "bias": 0.0,
        **{key: value for key, value in full_metrics.items() if not isinstance(value, np.ndarray)},
        **train_reconstruction,
        **held_reconstruction,
    }
    full_summary["all_group_count"] = (
        train_reconstruction["training_group_count"]
        + held_reconstruction["held_out_group_count"]
    )
    full_summary["all_exact_final_representative_groups"] = (
        train_reconstruction["training_exact_final_representative_groups"]
        + held_reconstruction["held_out_exact_final_representative_groups"]
    )
    full_summary["all_replacement_decision_changes"] = (
        train_reconstruction["training_replacement_decision_changes"]
        + held_reconstruction["held_out_replacement_decision_changes"]
    )
    full_summary["full_pool_gate_passed"] = bool(
        full_metrics["mistake_count"] == 0
        and full_summary["all_replacement_decision_changes"] == 0
        and full_summary["all_exact_final_representative_groups"]
        == full_summary["all_group_count"]
        and held_reconstruction["held_out_exact_ordered_representative_states"]
        == held_reconstruction["held_out_state_count"]
    )
    full_rows: list[dict[str, Any]] = [
        full_summary,
        *train_state_rows,
        *held_state_rows,
    ]
    full_rows.extend(
        _mistake_rows(full_pool, full_indices, full_metrics, "full_audited_pool")
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("representative continuous margin full-pool gate failed; stopped", flush=True)
        return

    checkpoint = {
        "format_version": 1,
        "component": "representative_margin",
        "status": "FROZEN_CANDIDATE",
        "architecture": "m_hat=w_best*(f_best/S)+w_i*(f_i/S); no bias",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_w_best": torch.tensor(
            float(model.weight.detach().cpu().numpy()[0, 0]), dtype=torch.float64
        ),
        "learned_w_i": torch.tensor(
            float(model.weight.detach().cpu().numpy()[0, 1]), dtype=torch.float64
        ),
        "bias": torch.tensor(0.0, dtype=torch.float64),
        "continuous_regression_only": True,
        "threshold_network_trained": False,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("representative margin checkpoint readback failed")
    checkpoint_hash = _sha256(CANDIDATE_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
    )
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["representative_margin_status"] = "FROZEN_CANDIDATE"
    _write_csv(FULL_PATH, full_rows)
    print(
        "representative margin FROZEN_CANDIDATE "
        f"w_best={float(model.weight.detach().cpu().numpy()[0, 0]):.17g} "
        f"w_i={float(model.weight.detach().cpu().numpy()[0, 1]):.17g} "
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
