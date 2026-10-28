"""Diagnose source-supported resource termination for continuous E-DIRECT.

This module changes only N1 termination: the historical 1e-4 crossing is
recorded, while search continues to the existing 5,000-evaluation cap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as base
from experiments.sho.deterministic_full_k_framework import deterministic_finish, hashes, prior


OUT = Path("outputs/sho/stage1B/resource_limited_edirect")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
CAP = 5_000
OLD_GATE = 1.0e-4
TARGET_NAME = "interior_51"


def resource_limited_search(objective, lower, upper, case_id, cap=CAP):
    """Run the existing continuous E-DIRECT loop without objective early exit."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    dimension = len(lower)
    traces: list[dict] = []
    api = base.composition(dimension)
    api["_greedy_proposal"] = (
        lambda points, losses, current_case, iteration:
        continuous.proposal(points, losses, current_case, iteration, traces)
    )
    points: list[np.ndarray] = []
    losses: list[float] = []
    events: list[dict] = []
    rounds = 0
    first_crossing = None

    def evaluate(unit_point, source):
        nonlocal first_crossing
        value = float(objective(lower + (upper - lower) * unit_point))
        if not np.isfinite(value):
            raise ValueError("nonfinite objective")
        points.append(np.asarray(unit_point, dtype=np.float64).copy())
        losses.append(value)
        best_index = int(np.argmin(losses))
        event = dict(
            evaluation=len(losses), iteration=rounds, source=source,
            point=unit_point.tolist(), loss=value, best_loss=float(losses[best_index]),
            boundary=bool(np.any((unit_point == 0.0) | (unit_point == 1.0))),
        )
        events.append(event)
        if first_crossing is None and losses[best_index] <= OLD_GATE:
            first_crossing = dict(
                evaluation=len(losses), iteration=rounds,
                incumbent_index=best_index,
                theta=(lower + (upper - lower) * points[best_index]).tolist(),
                loss=float(losses[best_index]), source=source,
            )

    design = base.teacher._initial_design() if dimension == 2 else base.initial_design(dimension)
    for unit_point in design:
        if len(losses) >= cap:
            break
        evaluate(unit_point, "initial")

    while len(losses) < cap:
        before = len(losses)
        proposals = api["_cell_proposals"](
            np.asarray(points), np.asarray(losses), case_id, rounds
        )
        for unit_point in proposals:
            if len(losses) >= cap:
                break
            evaluate(unit_point, "cell")
        if len(losses) < cap:
            unit_point = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), case_id, rounds
            )
            if unit_point is not None:
                evaluate(unit_point, "greedy")
        rounds += 1
        if len(losses) == before:
            break

    values = np.asarray(losses)
    normalized = np.asarray(points)
    best_index = int(np.argmin(values))
    theta = lower + (upper - lower) * normalized[best_index]
    radius = (upper - lower) / len(values) ** (1.0 / dimension)
    return dict(
        theta=theta, loss=float(values[best_index]), evaluations=len(values),
        iterations=rounds, points=normalized, values=values, events=events,
        lower=np.maximum(lower, theta - radius),
        upper=np.minimum(upper, theta + radius),
        first_crossing=first_crossing, continuous_traces=traces,
        exhausted_budget=len(values) == cap,
    )


def case_objectives(states):
    x = states[:-1]
    y = states[1:]
    torch_x = torch.from_numpy(x.copy())
    torch_y = torch.from_numpy(y.copy())

    def objective(theta):
        matrix = np.asarray(theta).reshape(2, 2)
        return float(np.mean((np.einsum("ij,nj->ni", matrix, x) - y) ** 2))

    def torch_objective(theta):
        return torch.mean((torch_x @ theta.reshape(2, 2).T - torch_y) ** 2)

    return objective, torch_objective


def validate_case(data, index):
    name = str(data["names"][index])
    category = str(data["categories"][index])
    states = data[f"prefix_{index}"]
    teacher = data["teachers"][index]
    objective, torch_objective = case_objectives(states)
    started = time.time()
    result = resource_limited_search(objective, prior.LOWER, prior.UPPER, index)
    truth = teacher.ravel()
    contained = bool(np.all(truth >= result["lower"]) and np.all(truth <= result["upper"]))
    row = dict(
        case=name, case_id=index, category=category,
        first_crossing=result["first_crossing"],
        n1_theta=result["theta"].tolist(), n1_loss=result["loss"],
        n1_evaluations=result["evaluations"], rounds=result["iterations"],
        exhausted_budget=result["exhausted_budget"],
        local_lower=result["lower"].tolist(), local_upper=result["upper"].tolist(),
        teacher_contained=contained,
        teacher_distance=float(np.linalg.norm(result["theta"] - truth)),
        teacher_coordinate_distance=np.abs(result["theta"] - truth).tolist(),
        continuous_calls=len(result["continuous_traces"]),
        continuous_physical_evaluations=sum(
            event["source"] == "greedy" for event in result["events"]
        ),
        duplicate_rejections=sum(
            trace["exact_duplicate_rejected"] for trace in result["continuous_traces"]
        ),
        surrogate_evaluations=sum(
            start["nfev"] for trace in result["continuous_traces"]
            for start in trace["starts"]
        ),
        seconds=time.time() - started,
    )
    np.savez_compressed(OUT / f"{name}_samples.npz", points=result["points"], losses=result["values"])
    (OUT / f"{name}_events.json").write_text(json.dumps(result["events"]))
    (OUT / f"{name}_continuous.json").write_text(json.dumps(result["continuous_traces"]))
    if contained:
        row.update(deterministic_finish(objective, torch_objective, result))
        final = np.asarray(row["final_theta"]).reshape(2, 2)
        delta = final - teacher
        true_state = states[-1].copy()
        predicted_state = true_state.copy()
        errors = []
        for _ in range(4096):
            true_state = teacher @ true_state
            predicted_state = final @ predicted_state
            errors.append(predicted_state - true_state)
        errors = np.asarray(errors)
        row.update(
            k_errors=delta.tolist(), matrix_error=float(np.linalg.norm(delta)),
            rollout_max_error=float(np.max(np.abs(errors))),
            rollout_mse=float(np.mean(errors ** 2)),
        )
    return row


def validate_index(index):
    with np.load(DATA) as data:
        return validate_case(data, index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-only", action="store_true")
    parser.add_argument("--resume-passed-target", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA)
    target_index = next(i for i, name in enumerate(data["names"]) if str(name) == TARGET_NAME)
    before = hashes()
    (OUT / "checkpoint_hashes_before.json").write_text(json.dumps(before, indent=2))
    previous_report = OUT / "report.json"
    partial_report = OUT / "results_partial.json"
    if args.resume_partial and partial_report.exists():
        rows = json.loads(partial_report.read_text())
        target_rows = [row for row in rows if row["case"] == TARGET_NAME]
        if len(target_rows) != 1 or not target_rows[0]["teacher_contained"]:
            raise RuntimeError("partial results do not contain a passed target")
    elif args.resume_passed_target and previous_report.exists():
        saved = json.loads(previous_report.read_text())
        rows = [row for row in saved["cases"] if row["case"] == TARGET_NAME]
        if len(rows) != 1 or not rows[0]["teacher_contained"]:
            raise RuntimeError("no passed target result available to resume")
    else:
        rows = [validate_case(data, target_index)]
    (OUT / "results_partial.json").write_text(json.dumps(rows, indent=2))
    target = rows[0]
    print(
        TARGET_NAME, "cross", target["first_crossing"], "final",
        target["n1_evaluations"], target["n1_loss"],
        "contained", target["teacher_contained"], flush=True,
    )
    stop = None if target["teacher_contained"] else "TARGET_CONTAINMENT_FAILURE"
    if stop is None and not args.target_only:
        # Exercise the 171 cases left unexecuted by the preceding validation
        # before repeating its already completed cases.
        old_report = Path("outputs/sho/stage1B/continuous_surrogate_edirect/report.json")
        old_names = set()
        if old_report.exists():
            old_names = {row["case"] for row in json.loads(old_report.read_text())["cases"]}
        remaining = [
            i for i, name in enumerate(data["names"])
            if i != target_index and str(name) not in old_names
        ]
        previously_run = [
            i for i, name in enumerate(data["names"])
            if i != target_index and str(name) in old_names
        ]
        order = remaining + previously_run
        completed = {row["case"] for row in rows}
        order = [index for index in order if str(data["names"][index]) not in completed]
        workers = max(1, args.workers)
        for offset in range(0, len(order), workers):
            batch = order[offset:offset + workers]
            if workers == 1:
                batch_rows = [validate_case(data, index) for index in batch]
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    batch_rows = list(executor.map(validate_index, batch))
            for row in batch_rows:
                rows.append(row)
                (OUT / "results_partial.json").write_text(json.dumps(rows, indent=2))
                print(
                    row["case"], row["category"], row["n1_loss"],
                    "contained", row["teacher_contained"], flush=True,
                )
                if not row["teacher_contained"]:
                    stop = "SUITE_CONTAINMENT_FAILURE"
                    break
            if stop is not None:
                break
    after = hashes()
    if before != after:
        raise RuntimeError("protected checkpoint changed")
    report = dict(
        status=("VALIDATED_DETERMINISTIC_TEACHER" if stop is None and not args.target_only
                else "TARGET_ONLY_PASSED" if stop is None else "STOPPED_" + stop),
        stopping_policy="exactly 5,000 physical evaluations; 1e-4 crossing recorded only",
        cap=CAP, old_gate=OLD_GATE, cases=rows,
        tested_cases=len(rows), planned_cases=len(data["names"]),
        checkpoints_unchanged=True, first_failure=stop,
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    if report["status"] == "VALIDATED_DETERMINISTIC_TEACHER":
        source = Path(__file__)
        (OUT / "dimension_general_edirect_teacher.json").write_text(json.dumps({
            "status": report["status"], "source": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "stopping": report["stopping_policy"],
        }, indent=2))
    print(report["status"], flush=True)


if __name__ == "__main__":
    main()
