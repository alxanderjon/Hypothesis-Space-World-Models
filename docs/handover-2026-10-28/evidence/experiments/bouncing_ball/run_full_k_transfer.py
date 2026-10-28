"""Fifty-case full 2x2 K_ball transfer through deterministic E-DIRECT/N2/N3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import deterministic_finish, hashes
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


OUT = Path("outputs/bouncing_ball/full_k_50_transfer")
CASES = OUT / "cases"
LOWER = np.full(4, -1.99, dtype=np.float64)
UPPER = np.full(4, 1.99, dtype=np.float64)
WIDTH = UPPER - LOWER
TEACHER = np.asarray((0.0, 1.0, 0.0, 1.0), dtype=np.float64)
COUNT = 50
SEED = 20261001
CAP = 5_000
BLOCK = 100
LOSS_GATE = 1.0e-4
FINAL_GATE = 1.0e-20


def _atomic_text(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def numpy_step(state: np.ndarray, theta: np.ndarray) -> np.ndarray:
    matrix = np.asarray(theta, dtype=np.float64).reshape(2, 2)
    x, y, vx, vy = (np.float64(value) for value in state)
    ux, wx = matrix @ np.asarray((x, vx), dtype=np.float64)
    uy, wy = matrix @ np.asarray((y, vy), dtype=np.float64)
    ux, wx, uy, wy = map(np.float64, (ux, wx, uy, wy))
    sign_x = np.float64(1.0)
    sign_y = np.float64(1.0)
    remaining = np.float64(ball.DT)
    for _ in range(ball.MAX_COLLISIONS):
        if remaining <= ball.EPS:
            break
        tx = np.float64(np.inf)
        ty = np.float64(np.inf)
        if ux > ball.EPS:
            tx = np.float64((ball.XMAX - x) / ux)
        elif ux < -ball.EPS:
            tx = np.float64((ball.XMIN - x) / ux)
        if uy > ball.EPS:
            ty = np.float64((ball.YMAX - y) / uy)
        elif uy < -ball.EPS:
            ty = np.float64((ball.YMIN - y) / uy)
        hit_time = min(tx, ty)
        if hit_time > remaining or np.isinf(hit_time):
            x = np.float64(x + ux * remaining)
            y = np.float64(y + uy * remaining)
            remaining = np.float64(0.0)
            break
        x = np.float64(x + ux * hit_time)
        y = np.float64(y + uy * hit_time)
        remaining = np.float64(remaining - hit_time)
        hit_x = abs(tx - hit_time) < ball.EPS
        hit_y = abs(ty - hit_time) < ball.EPS
        if hit_x:
            x = np.float64(ball.XMAX if ux > 0.0 else ball.XMIN)
            ux = np.float64(-ux)
            sign_x = np.float64(-sign_x)
        if hit_y:
            y = np.float64(ball.YMAX if uy > 0.0 else ball.YMIN)
            uy = np.float64(-uy)
            sign_y = np.float64(-sign_y)
    return np.asarray((x, y, sign_x * wx, sign_y * wy), dtype=np.float64)


def event_signature(state: np.ndarray, theta: np.ndarray) -> tuple[int, ...]:
    matrix = np.asarray(theta, dtype=np.float64).reshape(2, 2)
    x, y, vx, vy = (np.float64(value) for value in state)
    ux = np.float64(matrix[0, 0] * x + matrix[0, 1] * vx)
    uy = np.float64(matrix[0, 0] * y + matrix[0, 1] * vy)
    remaining = np.float64(ball.DT)
    signature = []
    for _ in range(ball.MAX_COLLISIONS):
        if remaining <= ball.EPS:
            break
        tx = np.float64(np.inf)
        ty = np.float64(np.inf)
        if ux > ball.EPS:
            tx = np.float64((ball.XMAX - x) / ux)
        elif ux < -ball.EPS:
            tx = np.float64((ball.XMIN - x) / ux)
        if uy > ball.EPS:
            ty = np.float64((ball.YMAX - y) / uy)
        elif uy < -ball.EPS:
            ty = np.float64((ball.YMIN - y) / uy)
        hit_time = min(tx, ty)
        if hit_time > remaining or np.isinf(hit_time):
            break
        x = np.float64(x + ux * hit_time)
        y = np.float64(y + uy * hit_time)
        remaining = np.float64(remaining - hit_time)
        hit_x = abs(tx - hit_time) < ball.EPS
        hit_y = abs(ty - hit_time) < ball.EPS
        signature.append(int(hit_x) + 2 * int(hit_y))
        if hit_x:
            x = np.float64(ball.XMAX if ux > 0.0 else ball.XMIN)
            ux = np.float64(-ux)
        if hit_y:
            y = np.float64(ball.YMAX if uy > 0.0 else ball.YMIN)
            uy = np.float64(-uy)
    return tuple(signature)


def torch_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    matrix = theta.reshape(2, 2)
    x, y, vx, vy = state.unbind()
    ux, wx = matrix @ torch.stack((x, vx))
    uy, wy = matrix @ torch.stack((y, vy))
    sign_x = 1.0
    sign_y = 1.0
    remaining: float | torch.Tensor = ball.DT
    for _ in range(ball.MAX_COLLISIONS):
        if float(remaining) <= ball.EPS:
            break
        tx: float | torch.Tensor = math.inf
        ty: float | torch.Tensor = math.inf
        if float(ux.detach()) > ball.EPS:
            tx = (ball.XMAX - x) / ux
        elif float(ux.detach()) < -ball.EPS:
            tx = (ball.XMIN - x) / ux
        if float(uy.detach()) > ball.EPS:
            ty = (ball.YMAX - y) / uy
        elif float(uy.detach()) < -ball.EPS:
            ty = (ball.YMIN - y) / uy
        tx_value = float(tx)
        ty_value = float(ty)
        hit_time: float | torch.Tensor = tx if tx_value <= ty_value else ty
        if float(hit_time) > float(remaining) or math.isinf(float(hit_time)):
            x = x + ux * remaining
            y = y + uy * remaining
            remaining = 0.0
            break
        x = x + ux * hit_time
        y = y + uy * hit_time
        remaining = remaining - hit_time
        hit_x = abs(tx_value - float(hit_time)) < ball.EPS
        hit_y = abs(ty_value - float(hit_time)) < ball.EPS
        if hit_x:
            x = torch.as_tensor(ball.XMAX if float(ux.detach()) > 0.0 else ball.XMIN)
            ux = -ux
            sign_x = -sign_x
        if hit_y:
            y = torch.as_tensor(ball.YMAX if float(uy.detach()) > 0.0 else ball.YMIN)
            uy = -uy
            sign_y = -sign_y
    return torch.stack((x, y, sign_x * wx, sign_y * wy))


def numpy_rollout(initial: np.ndarray, theta: np.ndarray, steps: int):
    values = np.empty((steps + 1, 4), dtype=np.float64)
    values[0] = initial
    for index in range(steps):
        values[index + 1] = numpy_step(values[index], theta)
    return values


def objectives(trajectory: np.ndarray):
    prefix = np.asarray(trajectory[:ball.PREFIX_STATES], dtype=np.float64)
    torch_prefix = torch.from_numpy(prefix.copy())

    def numpy_objective(theta):
        squared = np.float64(0.0)
        for index in range(ball.PREFIX_STATES - 1):
            error = numpy_step(prefix[index], theta) - prefix[index + 1]
            squared = np.float64(squared + np.dot(error, error))
        return float(squared / np.float64((ball.PREFIX_STATES - 1) * 4))

    def torch_objective(theta):
        errors = [torch_step(torch_prefix[index], theta) - torch_prefix[index + 1]
                  for index in range(ball.PREFIX_STATES - 1)]
        return torch.stack(errors).square().mean()

    return numpy_objective, torch_objective


def validate_teacher(trajectories: np.ndarray):
    controls = np.asarray((
        (4.9, 0.25, 3.0, -0.5),
        (5.0, 0.0, 1.0, 0.0),
        (4.9, 4.9, 2.0, 2.0),
        (5.0, -5.0, 1.0, -1.0),
        (0.0, 0.0, 250.0, -175.0),
        (-4.999999999999, 2.0, -2.5, 1.25),
    ), dtype=np.float64)
    teacher_controls = np.asarray([ball.TEACHER.step_bouncing_ball(*state, ball.DT)
                                   for state in controls])
    numpy_controls = np.asarray([numpy_step(state, TEACHER) for state in controls])
    torch_controls = np.asarray([torch_step(torch.from_numpy(state), torch.from_numpy(TEACHER)).detach().numpy()
                                 for state in controls])
    maximum_numpy = float(np.max(np.abs(numpy_controls - teacher_controls)))
    maximum_torch = float(np.max(np.abs(torch_controls - teacher_controls)))
    rollout_maximum = 0.0
    exact_values = 0
    total_values = 0
    zero_losses = []
    for trajectory in trajectories:
        predicted = numpy_rollout(trajectory[0], TEACHER, len(trajectory) - 1)
        rollout_maximum = max(rollout_maximum, float(np.max(np.abs(predicted - trajectory))))
        exact_values += int(np.count_nonzero(predicted == trajectory))
        total_values += int(trajectory.size)
        numpy_objective, torch_objective = objectives(trajectory)
        zero_losses.append((numpy_objective(TEACHER),
                            float(torch_objective(torch.from_numpy(TEACHER)))))
    passed = bool(
        maximum_numpy <= 8 * np.finfo(np.float64).eps
        and maximum_torch <= 8 * np.finfo(np.float64).eps
        and rollout_maximum <= 8 * np.finfo(np.float64).eps
        and all(a == 0.0 and b == 0.0 for a, b in zero_losses)
    )
    return dict(
        passed=passed, bounds=[LOWER.tolist(), UPPER.tolist()],
        controls=len(controls), control_labels=[
            "ordinary", "endpoint", "corner", "endpoint_corner",
            "multiple_impact", "near_endpoint",
        ],
        numpy_control_max_abs=maximum_numpy, torch_control_max_abs=maximum_torch,
        rollout_max_abs=rollout_maximum, exact_rollout_values=exact_values,
        total_rollout_values=total_values,
        numpy_zero_losses=[value[0] for value in zero_losses],
        torch_zero_losses=[value[1] for value in zero_losses],
    )


def _checkpoint(points, losses, iteration, traces):
    best = int(np.argmin(losses))
    theta = LOWER + WIDTH * points[best]
    radius = WIDTH / len(losses) ** 0.25
    box_lower = np.maximum(LOWER, theta - radius)
    box_upper = np.minimum(UPPER, theta + radius)
    outside = np.maximum(np.maximum(box_lower - TEACHER, TEACHER - box_upper), 0.0)
    loss_ok = bool(losses[best] <= LOSS_GATE)
    containment = bool(np.linalg.norm(outside) == 0.0)
    return dict(
        evaluation=len(losses), iteration=iteration, best_theta=theta.tolist(),
        best_loss=float(losses[best]), loss_ok=loss_ok,
        containment_ok=containment, both_ok=bool(loss_ok and containment),
        box_lower=box_lower.tolist(), box_upper=box_upper.tolist(),
        teacher_outside_l2=float(np.linalg.norm(outside)),
        continuous_calls=len(traces),
        continuous_duplicate_rejections=int(sum(t["exact_duplicate_rejected"] for t in traces)),
    )


def _save_resume(case_dir, points, losses, events, traces, checkpoints,
                 iteration, phase, pending, next_pending):
    temporary = case_dir / "resume.tmp.npz"
    np.savez_compressed(temporary, points=np.asarray(points), losses=np.asarray(losses),
                        pending=np.asarray(pending, dtype=np.float64))
    temporary.replace(case_dir / "resume.npz")
    _atomic_text(case_dir / "resume.json", json.dumps(dict(
        events=events, traces=traces, checkpoints=checkpoints,
        iteration=iteration, phase=phase, next_pending=next_pending,
    )))


def _first_newton(torch_objective, theta_value):
    theta = torch.tensor(theta_value, dtype=torch.float64, requires_grad=True)
    loss = torch_objective(theta)
    gradient = torch.autograd.grad(loss, theta, create_graph=True)[0]
    hessian = torch.stack([
        torch.autograd.grad(gradient[index], theta, retain_graph=True)[0]
        for index in range(len(theta))
    ])
    hessian = 0.5 * (hessian + hessian.T)
    singular = torch.linalg.svdvals(hessian)
    smallest = float(torch.min(singular))
    largest = float(torch.max(singular))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum = float(torch.min(eigenvalues))
    scale = max(float(torch.max(torch.abs(eigenvalues))), torch.finfo(torch.float64).tiny)
    direct = minimum > 0.0 and condition <= polish.CONDITION_LIMIT
    initial = 0.0 if direct else max(
        scale * polish.DAMPING_RELATIVE_FLOOR,
        -minimum + scale * polish.DAMPING_RELATIVE_FLOOR,
    )
    identity = torch.eye(len(theta), dtype=torch.float64)
    direction = None
    damping = math.nan
    for attempt in range(polish.MAX_DAMPING_TRIALS):
        if attempt == 0 and direct:
            damping = 0.0
        else:
            base_value = initial if initial > 0.0 else scale * polish.DAMPING_RELATIVE_FLOOR
            damping = base_value * (10.0 ** (attempt if not direct else attempt - 1))
        try:
            candidate = torch.linalg.solve(hessian.detach() + damping * identity, -gradient.detach())
        except torch.linalg.LinAlgError:
            continue
        if bool(torch.isfinite(candidate).all()) and float(torch.dot(gradient.detach(), candidate)) < 0.0:
            direction = candidate.numpy()
            break
    return gradient.detach().numpy(), direction, condition, damping


def _collision_diagnostics(trajectory, final_theta):
    start = ball.PREFIX_STATES - 1
    steps = ball.TOTAL_STEPS - start
    predicted = numpy_rollout(trajectory[start], final_theta, steps)
    truth = trajectory[start:]
    topology_mismatches = sum(
        event_signature(predicted[index], final_theta) != event_signature(truth[index], TEACHER)
        for index in range(steps)
    )
    prefix_signatures = [event_signature(trajectory[index], TEACHER)
                         for index in range(ball.PREFIX_STATES - 1)]
    prefix_events = sum(len(value) for value in prefix_signatures)
    prefix_corners = sum(any(mask == 3 for mask in value) for value in prefix_signatures)
    prefix_multiple = sum(len(value) > 1 for value in prefix_signatures)
    error = predicted - truth
    return dict(
        continuation_max_error=float(np.max(np.abs(error))),
        continuation_mse=float(np.mean(error * error)),
        collision_topology_regressions=int(topology_mismatches),
        prefix_collision_events=int(prefix_events), prefix_corner_steps=int(prefix_corners),
        prefix_multiple_impact_steps=int(prefix_multiple),
        rollout_finite=bool(np.isfinite(predicted).all()),
        rollout_positions_in_box=bool(np.all(predicted[:, :2] >= ball.XMIN - ball.EPS)
                                      and np.all(predicted[:, :2] <= ball.XMAX + ball.EPS)),
    )


def run_case(index: int, trajectory: np.ndarray):
    torch.set_num_threads(1)
    case_dir = CASES / f"trajectory_{index:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    started = time.time()
    numpy_objective, torch_objective = objectives(trajectory)
    api = edirect.composition(4)
    traces = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
    points, losses, events, checkpoints = [], [], [], []
    iteration, phase, next_pending = 0, "initial", 0
    pending = np.empty((0, 4), dtype=np.float64)
    resume_npz, resume_json = case_dir / "resume.npz", case_dir / "resume.json"
    if resume_npz.exists() and resume_json.exists():
        saved = np.load(resume_npz)
        points = [value.copy() for value in saved["points"]]
        losses = saved["losses"].tolist()
        pending = saved["pending"].copy()
        meta = json.loads(resume_json.read_text())
        events, traces, checkpoints = meta["events"], meta["traces"], meta["checkpoints"]
        iteration, phase, next_pending = meta["iteration"], meta["phase"], meta["next_pending"]

    def evaluate(unit, source):
        value = numpy_objective(LOWER + WIDTH * unit)
        if not np.isfinite(value):
            raise RuntimeError("nonfinite bouncing-ball objective")
        points.append(np.asarray(unit).copy())
        losses.append(value)
        events.append(dict(evaluation=len(losses), iteration=iteration, source=source,
                           point=np.asarray(unit).tolist(), loss=value))

    def checkpoint_if_due(current_phase, current_pending, current_next):
        if not losses or len(losses) % BLOCK:
            return False
        record = _checkpoint(np.asarray(points), np.asarray(losses), iteration, traces)
        checkpoints.append(record)
        _save_resume(case_dir, points, losses, events, traces, checkpoints,
                     iteration, current_phase, current_pending, current_next)
        return record["both_ok"] or len(losses) >= CAP

    if phase == "initial":
        design = edirect.initial_design(4)
        for unit in design[len(points):]:
            evaluate(unit, "initial")
        phase = "next_round"
    stop = False
    while len(losses) < CAP and not stop:
        if phase == "next_round":
            pending = api["_cell_proposals"](np.asarray(points), np.asarray(losses),
                                               2_000_000 + index, iteration)
            next_pending = 0
            phase = "cell"
        if phase == "cell":
            while next_pending < len(pending) and len(losses) < CAP:
                unit = pending[next_pending]
                next_pending += 1
                evaluate(unit, "cell")
                if checkpoint_if_due("cell", pending, next_pending):
                    stop = True
                    break
            if stop:
                break
            phase = "greedy"
        if phase == "greedy" and len(losses) < CAP:
            unit = api["_greedy_proposal"](np.asarray(points), np.asarray(losses),
                                             2_000_000 + index, iteration)
            if unit is not None:
                evaluate(unit, "continuous_surrogate")
            iteration += 1
            phase = "next_round"
            pending = np.empty((0, 4), dtype=np.float64)
            next_pending = 0
            if checkpoint_if_due("next_round", pending, 0):
                stop = True

    checkpoint = next((value for value in checkpoints if value["both_ok"]), checkpoints[-1])
    start = dict(theta=np.asarray(checkpoint["best_theta"]), loss=checkpoint["best_loss"],
                 lower=np.asarray(checkpoint["box_lower"]), upper=np.asarray(checkpoint["box_upper"]))
    finish = deterministic_finish(numpy_objective, torch_objective, start)
    n2 = np.asarray(finish["n2_theta"])
    final = np.asarray(finish["final_theta"])
    gradient, direction, condition, damping = _first_newton(torch_objective, n2)
    displacement = TEACHER - n2
    n2_distance = float(np.linalg.norm(displacement))
    direction_norm = float(np.linalg.norm(direction)) if direction is not None else None
    cosine = (float(np.dot(direction, displacement) / (direction_norm * n2_distance))
              if direction is not None and direction_norm > 0.0 and n2_distance > 0.0 else None)
    final_error = final - TEACHER
    collision = _collision_diagnostics(trajectory, final)
    result = dict(
        trajectory_id=index, bounds=[LOWER.tolist(), UPPER.tolist()],
        n1_evaluations=len(losses), n1_iterations=iteration,
        n1_loss_ok=any(value["loss_ok"] for value in checkpoints),
        containment_ok=any(value["containment_ok"] for value in checkpoints),
        both_ok=any(value["both_ok"] for value in checkpoints),
        handoff_checkpoint=checkpoint["evaluation"],
        n1_theta=checkpoint["best_theta"], n1_loss=checkpoint["best_loss"],
        n1_cell_rounds=iteration,
        n1_cell_proposals=sum(event["source"] == "cell" for event in events),
        n1_continuous_proposals=sum(event["source"] == "continuous_surrogate" for event in events),
        continuous_surrogate_calls=len(traces),
        continuous_duplicate_rejections=int(sum(t["exact_duplicate_rejected"] for t in traces)),
        learned_n1_decisions=0,
        n2_theta=finish["n2_theta"], n2_loss=finish["n2_loss"],
        n2_evaluations=finish["n2_evaluations"], n2_gate=finish["n2_gate"],
        n2_message=finish["n2_message"],
        n2_entrywise_error=(n2 - TEACHER).tolist(),
        n2_max_entry_error=float(np.max(np.abs(n2 - TEACHER))),
        n2_frobenius_error=n2_distance,
        n2_gradient=gradient.tolist(), n2_gradient_norm=float(np.linalg.norm(gradient)),
        first_newton_direction=(direction.tolist() if direction is not None else None),
        first_newton_step_norm=direction_norm,
        truth_displacement=displacement.tolist(), newton_truth_cosine=cosine,
        newton_step_truth_ratio=(direction_norm / n2_distance
                                 if direction_norm is not None and n2_distance > 0.0 else None),
        first_newton_condition=condition, first_newton_damping=damping,
        n3_accepted_steps=finish["n3_accepted_steps"],
        n3_evaluations=finish["n3_evaluations"], n3_stop=finish["n3_stop"],
        final_theta=finish["final_theta"], final_loss=finish["final_loss"],
        final_entrywise_error=final_error.tolist(),
        final_max_entry_error=float(np.max(np.abs(final_error))),
        final_frobenius_error=float(np.linalg.norm(final_error)),
        recovered=bool(finish["final_loss"] < FINAL_GATE),
        seconds=time.time() - started, checkpoints=checkpoints, **collision,
    )
    np.savez_compressed(case_dir / "n1_state.npz", points=np.asarray(points), losses=np.asarray(losses))
    _atomic_text(case_dir / "events.json", json.dumps(events))
    _atomic_text(case_dir / "continuous.json", json.dumps(traces))
    _atomic_text(result_path, json.dumps(result, indent=2))
    return result


def _distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return dict(count=0, minimum=None, p10=None, median=None, p90=None,
                    maximum=None, total=None)
    return dict(count=int(values.size), minimum=float(np.min(values)),
                p10=float(np.quantile(values, 0.10, method="lower")),
                median=float(np.median(values)),
                p90=float(np.quantile(values, 0.90, method="higher")),
                maximum=float(np.max(values)), total=float(np.sum(values)))


def _write_csv(rows):
    flattened = []
    for row in rows:
        flattened.append({key: value for key, value in row.items() if key != "checkpoints"})
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (OUT / "per_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _report(rows, validation, before, after, started, workers):
    recovered = [row for row in rows if row["recovered"]]
    steps = sorted({row["n3_accepted_steps"] for row in rows})
    impact_counts = [row["prefix_collision_events"] for row in rows]
    return dict(
        status="COMPLETE" if len(rows) == COUNT else "INCOMPLETE",
        teacher_validation=validation, trajectories=len(rows),
        completed=len(rows), recoveries=len(recovered), recovery_rate=len(recovered) / len(rows),
        bounds=[LOWER.tolist(), UPPER.tolist()], teacher=TEACHER.tolist(),
        n1_loss_gate_successes=sum(row["n1_loss_ok"] for row in rows),
        n1_containment_successes=sum(row["containment_ok"] for row in rows),
        n1_both_successes=sum(row["both_ok"] for row in rows),
        n2_gate_successes=sum(row["n2_gate"] for row in rows),
        n1_evaluations=_distribution([row["n1_evaluations"] for row in rows]),
        n2_evaluations=_distribution([row["n2_evaluations"] for row in rows]),
        n3_evaluations=_distribution([row["n3_evaluations"] for row in rows]),
        n2_truth_distance=_distribution([row["n2_frobenius_error"] for row in rows]),
        newton_alignment=_distribution([row["newton_truth_cosine"] for row in rows]),
        newton_step_truth_ratio=_distribution([row["newton_step_truth_ratio"] for row in rows]),
        n3_accepted_steps={str(step): sum(row["n3_accepted_steps"] == step for row in rows)
                           for step in steps},
        largest_n2_recovered=max(
            (row for row in recovered), key=lambda row: row["n2_frobenius_error"], default=None),
        worst_final_max_entry_error=max(row["final_max_entry_error"] for row in rows),
        worst_final_frobenius_error=max(row["final_frobenius_error"] for row in rows),
        worst_continuation_error=max(row["continuation_max_error"] for row in rows),
        worst_continuation_mse=max(row["continuation_mse"] for row in rows),
        collision_topology_regressions=sum(row["collision_topology_regressions"] for row in rows),
        prefix_collision_events=dict(minimum=min(impact_counts), median=float(np.median(impact_counts)),
                                     maximum=max(impact_counts), total=sum(impact_counts)),
        stable_rollouts=sum(row["rollout_finite"] and row["rollout_positions_in_box"] for row in rows),
        checkpoint_hashes_unchanged=before == after, checkpoint_count=len(before),
        workers=workers, elapsed_seconds=time.time() - started,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case-index", type=int, default=None)
    args = parser.parse_args()
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=COUNT, n_timesteps=ball.TOTAL_STEPS, dt=ball.DT, seed=SEED)
    if args.case_index is not None:
        result = run_case(args.case_index, trajectories[args.case_index])
        print(json.dumps(dict(case=args.case_index, n1=result["n1_evaluations"],
                              n2=result["n2_evaluations"], recovered=result["recovered"])), flush=True)
        return
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    validation = validate_teacher(trajectories)
    _atomic_text(OUT / "teacher_validation.json", json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise RuntimeError("full-K bouncing-ball teacher containment failed")
    before = hashes()
    _atomic_text(OUT / "checkpoint_hashes_before.json", json.dumps(before, indent=2))
    workers = max(1, min(args.workers, 6))
    rows = []
    for offset in range(0, COUNT, workers):
        indices = list(range(offset, min(offset + workers, COUNT)))
        children = []
        for index in indices:
            command = [sys.executable, "-m", "experiments.bouncing_ball.run_full_k_transfer",
                       "--case-index", str(index)]
            children.append((index, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(index, child.wait()) for index, child in children]
        failures = [(index, code) for index, code in failures if code]
        if failures:
            raise RuntimeError(f"case process failures: {failures}")
        batch = [json.loads((CASES / f"trajectory_{index:02d}" / "result.json").read_text())
                 for index in indices]
        rows.extend(batch)
        rows.sort(key=lambda row: row["trajectory_id"])
        _write_csv(rows)
        _atomic_text(OUT / "results_partial.json", json.dumps(rows, indent=2))
        for row in batch:
            print(f"case {row['trajectory_id']:02d}: N1={row['n1_evaluations']} "
                  f"N2={row['n2_evaluations']} N3={row['n3_accepted_steps']} "
                  f"loss={row['final_loss']:.3e} recovered={row['recovered']}", flush=True)
    after = hashes()
    _atomic_text(OUT / "checkpoint_hashes_after.json", json.dumps(after, indent=2))
    report = _report(rows, validation, before, after, started, workers)
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    print(report["status"], report["recoveries"], flush=True)


if __name__ == "__main__":
    main()
