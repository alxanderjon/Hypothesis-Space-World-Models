#!/usr/bin/env python3
"""Learn only the continuous normalized same-size grouping margin."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/same_size_grouping_margin")
SMOKE_PATH = OUTPUT_DIR / "same_size_margin_smoke.csv"
FULL_PATH = OUTPUT_DIR / "same_size_margin_full_pool.csv"
CANDIDATE_PATH = OUTPUT_DIR / "same_size_margin_candidate.pt"
HASH_PATH = CANDIDATE_PATH.with_suffix(".sha256")

TEACHER_SCALE = 1.0e-14
MODEL_SEED = smoke.MODEL_SEED + 10_100_003
TRAIN_PER_CLASS = 5_000
VALIDATION_PER_CLASS = 1_000
MAX_EPOCHS = 10_000
LEARNING_RATE = 2.0e-2
COEFFICIENT_GATE = 1.0e-12
STABILITY_EPOCHS = 50


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


def _flatten_events(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {
        "x": [],
        "y": [],
        "gap": [],
        "same": [],
        "state_position": [],
        "source_state_index": [],
        "cell_index": [],
        "d_i": [],
        "d_anchor": [],
    }
    for state_position, state in enumerate(states):
        trace = grouping_audit._teacher_trace(state["sizes"], state["losses"])
        for decision in trace["decisions"]:
            gap = float(decision["comparison_gap"])
            if not math.isfinite(gap):
                continue
            index = int(decision["index"])
            anchor = float(decision["prior_anchor"])
            x = gap / TEACHER_SCALE
            margin = TEACHER_SCALE - gap
            y = margin / TEACHER_SCALE
            values["x"].append(x)
            values["y"].append(y)
            values["gap"].append(gap)
            values["same"].append(bool(decision["same_group"]))
            values["state_position"].append(state_position)
            values["source_state_index"].append(state["source_state_index"])
            values["cell_index"].append(index)
            values["d_i"].append(float(state["sizes"][index]))
            values["d_anchor"].append(anchor)
    return {
        "x": np.asarray(values["x"], dtype=np.float64),
        "y": np.asarray(values["y"], dtype=np.float64),
        "gap": np.asarray(values["gap"], dtype=np.float64),
        "same": np.asarray(values["same"], dtype=bool),
        "state_position": np.asarray(values["state_position"], dtype=np.int64),
        "source_state_index": np.asarray(
            values["source_state_index"], dtype=np.int64
        ),
        "cell_index": np.asarray(values["cell_index"], dtype=np.int64),
        "d_i": np.asarray(values["d_i"], dtype=np.float64),
        "d_anchor": np.asarray(values["d_anchor"], dtype=np.float64),
    }


def _balanced_indices(
    pool: dict[str, np.ndarray], per_class: int, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    same = np.flatnonzero(pool["same"])
    new = np.flatnonzero(~pool["same"])
    count = min(per_class, len(same), len(new))
    if count == 0:
        raise RuntimeError("no real SAME/NEW balance is possible")
    selected = np.concatenate(
        (
            generator.choice(same, size=count, replace=False),
            generator.choice(new, size=count, replace=False),
        )
    )
    generator.shuffle(selected)
    return selected


def _predict(model: nn.Linear, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return (
            model(torch.as_tensor(x[:, None], dtype=torch.float64))
            .squeeze(-1)
            .cpu()
            .numpy()
        )


def _metrics(model: nn.Linear, pool: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, Any]:
    x = pool["x"][indices]
    teacher = pool["y"][indices]
    same = pool["same"][indices]
    predicted = _predict(model, x)
    errors = predicted - teacher
    predicted_same = predicted >= 0.0
    mistakes = predicted_same != same
    same_errors = errors[same]
    new_errors = errors[~same]
    correlation = (
        float(np.corrcoef(predicted, teacher)[0, 1])
        if len(predicted) > 1 and np.std(predicted) > 0.0 and np.std(teacher) > 0.0
        else 1.0
    )
    return {
        "event_count": len(indices),
        "same_count": int(np.sum(same)),
        "new_count": int(np.sum(~same)),
        "RMSE": float(np.sqrt(np.mean(errors * errors))),
        "MAE": float(np.mean(np.abs(errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "correlation": correlation,
        "SAME_RMSE": float(np.sqrt(np.mean(same_errors * same_errors))),
        "SAME_MAE": float(np.mean(np.abs(same_errors))),
        "SAME_max_absolute_error": float(np.max(np.abs(same_errors))),
        "NEW_RMSE": float(np.sqrt(np.mean(new_errors * new_errors))),
        "NEW_MAE": float(np.mean(np.abs(new_errors))),
        "NEW_max_absolute_error": float(np.max(np.abs(new_errors))),
        "overall_agreement": float(np.mean(~mistakes)),
        "SAME_agreement": float(np.mean(~mistakes[same])),
        "NEW_agreement": float(np.mean(~mistakes[~same])),
        "mistake_count": int(np.sum(mistakes)),
        "predicted": predicted,
        "predicted_same": predicted_same,
        "errors": errors,
        "mistakes": mistakes,
    }


def _train(
    train_pool: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_pool: dict[str, np.ndarray],
    validation_indices: np.ndarray,
) -> tuple[nn.Linear, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = nn.Linear(1, 1, bias=True).to(torch.float64)
    initial_w = float(model.weight.detach().item())
    initial_b = float(model.bias.detach().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_x = torch.as_tensor(
        train_pool["x"][train_indices, None], dtype=torch.float64
    )
    train_y = torch.as_tensor(train_pool["y"][train_indices], dtype=torch.float64)
    # This fixed, target-derived weighting makes the regression numerically balanced
    # across the large real dynamic range; its unique zero-residual solution is unchanged.
    residual_scale = torch.maximum(torch.ones_like(train_y), torch.abs(train_y))
    best_state = copy.deepcopy(model.state_dict())
    best_key = (math.inf, math.inf)
    best_epoch = 0
    exact_streak = 0
    stop_epoch = MAX_EPOCHS
    stopping_reason = "maximum_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        prediction = model(train_x).squeeze(-1)
        loss = torch.mean(torch.square((prediction - train_y) / residual_scale))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_metrics = _metrics(
            model, validation_pool, validation_indices
        )
        w = float(model.weight.detach().item())
        b = float(model.bias.detach().item())
        coefficient_error = max(abs(w + 1.0), abs(b - 1.0))
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
                f"same-size margin epoch={epoch} w={w:.17g} b={b:.17g} "
                f"coefficient_error={coefficient_error:.3e} "
                f"mistakes={validation_metrics['mistake_count']}",
                flush=True,
            )
    model.load_state_dict(best_state)
    # Polish the same continuous objective after Adam has entered the numerical basin.
    # This is still supervised regression on the real events; no coefficients are set
    # analytically and no SAME/NEW labels enter the loss.
    polish = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        tolerance_grad=1.0e-15,
        tolerance_change=1.0e-18,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        polish.zero_grad(set_to_none=True)
        prediction = model(train_x).squeeze(-1)
        objective = torch.mean(
            torch.square((prediction - train_y) / residual_scale)
        )
        objective.backward()
        return objective

    pre_polish_state = copy.deepcopy(model.state_dict())
    pre_polish_error = max(
        abs(float(model.weight.detach().item()) + 1.0),
        abs(float(model.bias.detach().item()) - 1.0),
    )
    polish.step(closure)
    post_polish_error = max(
        abs(float(model.weight.detach().item()) + 1.0),
        abs(float(model.bias.detach().item()) - 1.0),
    )
    if post_polish_error > pre_polish_error:
        model.load_state_dict(pre_polish_state)
        post_polish_error = pre_polish_error
    metrics = _metrics(model, validation_pool, validation_indices)
    w = float(model.weight.detach().item())
    b = float(model.bias.detach().item())
    summary = {
        "record_type": "summary",
        "best_epoch": best_epoch,
        "stop_epoch": stop_epoch,
        "stopping_reason": stopping_reason,
        "LBFGS_continuous_polish_used": True,
        "post_polish_coefficient_error": post_polish_error,
        "training_event_count": len(train_indices),
        "validation_event_count": len(validation_indices),
        "training_SAME_count": int(np.sum(train_pool["same"][train_indices])),
        "training_NEW_count": int(np.sum(~train_pool["same"][train_indices])),
        "validation_SAME_count": metrics["same_count"],
        "validation_NEW_count": metrics["new_count"],
        "learned_w": w,
        "learned_b": b,
        "initial_w": initial_w,
        "initial_b": initial_b,
        "teacher_scale": TEACHER_SCALE,
        "loss": "mean squared normalized continuous residual",
        "coefficient_max_error_from_exact_affine_relation": max(
            abs(w + 1.0), abs(b - 1.0)
        ),
        **{key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)},
    }
    summary["smoke_gate_passed"] = bool(
        metrics["mistake_count"] == 0
        and summary["coefficient_max_error_from_exact_affine_relation"]
        <= COEFFICIENT_GATE
    )
    return model, summary


def _mistake_rows(
    pool: dict[str, np.ndarray], indices: np.ndarray, metrics: dict[str, Any], split: str
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
                "cell_index": int(pool["cell_index"][event_index]),
                "d_i": float(pool["d_i"][event_index]),
                "d_anchor": float(pool["d_anchor"][event_index]),
                "gap": float(pool["gap"][event_index]),
                "x": float(pool["x"][event_index]),
                "teacher_y": float(pool["y"][event_index]),
                "predicted_y": float(metrics["predicted"][local_index]),
                "teacher_SAME": bool(pool["same"][event_index]),
                "predicted_SAME": bool(metrics["predicted_same"][local_index]),
            }
        )
    return rows


def _bookkeeping_reconstruction(
    model: nn.Linear, states: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_membership_changes = 0
    exact_states = 0
    for state_position, state in enumerate(states):
        sizes = state["sizes"]
        losses = state["losses"]
        teacher = grouping_audit._teacher_trace(sizes, losses)
        order = teacher["order"]
        learned_membership = np.full(len(sizes), -1, dtype=np.int64)
        group_index = -1
        anchor = math.nan
        learned_w = float(model.weight.detach().item())
        learned_b = float(model.bias.detach().item())
        for order_position, index_value in enumerate(order):
            index = int(index_value)
            d_i = float(sizes[index])
            if order_position == 0:
                group_index = 0
                anchor = d_i
            else:
                x = abs(d_i - anchor) / TEACHER_SCALE
                learned_y = learned_w * x + learned_b
                if learned_y < 0.0:
                    group_index += 1
                    anchor = d_i
            learned_membership[index] = group_index
        changes = int(np.sum(learned_membership != teacher["membership"]))
        exact = changes == 0
        total_membership_changes += changes
        exact_states += int(exact)
        rows.append(
            {
                "record_type": "state_reconstruction",
                "state_position": state_position,
                "source_state_index": state["source_state_index"],
                "cell_count": len(sizes),
                "teacher_group_count": len(teacher["groups"]),
                "learned_group_count": group_index + 1,
                "membership_change_count": changes,
                "exact_group_membership": exact,
            }
        )
    return rows, {
        "state_count": len(states),
        "exact_group_membership_states": exact_states,
        "group_membership_change_count": total_membership_changes,
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
    train_pool = _flatten_events(train_states)
    validation_pool = _flatten_events(validation_states)
    train_indices = _balanced_indices(train_pool, TRAIN_PER_CLASS, MODEL_SEED + 1)
    validation_indices = _balanced_indices(
        validation_pool, VALIDATION_PER_CLASS, MODEL_SEED + 2
    )
    model, smoke_summary = _train(
        train_pool, train_indices, validation_pool, validation_indices
    )
    smoke_metrics = _metrics(model, validation_pool, validation_indices)
    smoke_rows: list[dict[str, Any]] = [smoke_summary]
    smoke_rows.extend(
        _mistake_rows(
            validation_pool, validation_indices, smoke_metrics, "balanced_validation"
        )
    )
    _write_csv(SMOKE_PATH, smoke_rows)
    if not bool(smoke_summary["smoke_gate_passed"]):
        print("same-size continuous margin smoke gate failed; stopped", flush=True)
        return

    full_indices = np.arange(len(validation_pool["y"]), dtype=np.int64)
    full_metrics = _metrics(model, validation_pool, full_indices)
    state_rows, bookkeeping = _bookkeeping_reconstruction(model, validation_states)
    full_summary = {
        "record_type": "summary",
        "held_out_event_count": len(full_indices),
        "held_out_SAME_count": full_metrics["same_count"],
        "held_out_NEW_count": full_metrics["new_count"],
        "learned_w": float(model.weight.detach().item()),
        "learned_b": float(model.bias.detach().item()),
        "teacher_scale": TEACHER_SCALE,
        **{key: value for key, value in full_metrics.items() if not isinstance(value, np.ndarray)},
        **bookkeeping,
    }
    full_summary["full_pool_gate_passed"] = bool(
        full_metrics["mistake_count"] == 0
        and bookkeeping["group_membership_change_count"] == 0
        and bookkeeping["exact_group_membership_states"] == len(validation_states)
    )
    full_rows: list[dict[str, Any]] = [full_summary, *state_rows]
    full_rows.extend(
        _mistake_rows(
            validation_pool, full_indices, full_metrics, "full_held_out_pool"
        )
    )
    if not bool(full_summary["full_pool_gate_passed"]):
        _write_csv(FULL_PATH, full_rows)
        print("same-size continuous margin full-pool gate failed; stopped", flush=True)
        return

    checkpoint = {
        "format_version": 1,
        "component": "same_size_margin",
        "status": "FROZEN_CANDIDATE",
        "architecture": "y_hat=w*(abs(d_i-d_anchor)/1e-14)+b",
        "state_dict": copy.deepcopy(model.state_dict()),
        "learned_w": torch.tensor(
            float(model.weight.detach().item()), dtype=torch.float64
        ),
        "learned_b": torch.tensor(float(model.bias.detach().item()), dtype=torch.float64),
        "teacher_scale": TEACHER_SCALE,
        "continuous_regression_only": True,
        "threshold_network_trained": False,
        "smoke_summary": smoke_summary,
        "full_pool_summary": full_summary,
    }
    torch.save(checkpoint, CANDIDATE_PATH)
    restored = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=False)
    if restored.get("status") != "FROZEN_CANDIDATE":
        raise RuntimeError("same-size margin checkpoint readback failed")
    checkpoint_hash = _sha256(CANDIDATE_PATH)
    HASH_PATH.write_text(
        f"{checkpoint_hash}  {CANDIDATE_PATH.name}\n", encoding="utf-8"
    )
    full_summary["candidate_checkpoint"] = str(CANDIDATE_PATH)
    full_summary["candidate_checkpoint_sha256"] = checkpoint_hash
    full_summary["same_size_margin_status"] = "FROZEN_CANDIDATE"
    _write_csv(FULL_PATH, full_rows)
    print(
        "same-size margin FROZEN_CANDIDATE "
        f"w={float(model.weight.detach().item()):.17g} "
        f"b={float(model.bias.detach().item()):.17g} "
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
