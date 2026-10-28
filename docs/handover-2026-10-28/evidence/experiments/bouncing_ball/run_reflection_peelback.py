"""Deterministic five-parameter bouncing-ball reflection peelback.

The exact collision-time and wall-identity scaffold is retained.  A shared
continuous coefficient rho supplies the impact update for both the working
transport component and the sampled-velocity component.
"""
from __future__ import annotations

import argparse
import hashlib
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
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.kepler import run_four_parameter_peelback as staged


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/bouncing_ball/reflection_peelback")
CASES = OUT / "cases"
DIMENSION = 5
LOWER = np.full(DIMENSION, -1.99, dtype=np.float64)
UPPER = np.full(DIMENSION, 1.99, dtype=np.float64)
WIDTH = UPPER - LOWER
TEACHER = np.asarray((0.0, 1.0, 0.0, 1.0, -1.0), dtype=np.float64)
TRAJECTORY_SEED = 20_261_002
SOURCE_TRAJECTORIES = 100
SELECTED = (1, 6, 13, 3, 4, 8, 11, 12)  # 3x one, 3x two, 2x three prefix impacts.
CAP = 5_000
BLOCK = 100
LOSS_GATE = 1.0e-4
FINAL_GATE = 1.0e-20


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def numpy_step(state: np.ndarray, theta: np.ndarray) -> np.ndarray:
    matrix = np.asarray(theta[:4], dtype=np.float64).reshape(2, 2)
    rho = np.float64(theta[4])
    x, y, vx, vy = (np.float64(value) for value in state)
    ux, wx = matrix @ np.asarray((x, vx), dtype=np.float64)
    uy, wy = matrix @ np.asarray((y, vy), dtype=np.float64)
    ux, wx, uy, wy = map(np.float64, (ux, wx, uy, wy))
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
            ux = np.float64(rho * ux)
            wx = np.float64(rho * wx)
        if hit_y:
            y = np.float64(ball.YMAX if uy > 0.0 else ball.YMIN)
            uy = np.float64(rho * uy)
            wy = np.float64(rho * wy)
    return np.asarray((x, y, wx, wy), dtype=np.float64)


def torch_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    matrix = theta[:4].reshape(2, 2)
    rho = theta[4]
    x, y, vx, vy = state.unbind()
    ux, wx = matrix @ torch.stack((x, vx))
    uy, wy = matrix @ torch.stack((y, vy))
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
            ux = rho * ux
            wx = rho * wx
        if hit_y:
            y = torch.as_tensor(ball.YMAX if float(uy.detach()) > 0.0 else ball.YMIN)
            uy = rho * uy
            wy = rho * wy
    return torch.stack((x, y, wx, wy))


def event_signature(state: np.ndarray, theta: np.ndarray) -> tuple[int, ...]:
    matrix = np.asarray(theta[:4], dtype=np.float64).reshape(2, 2)
    rho = np.float64(theta[4])
    x, y, vx, vy = (np.float64(value) for value in state)
    ux = np.float64(matrix[0, 0] * x + matrix[0, 1] * vx)
    uy = np.float64(matrix[0, 0] * y + matrix[0, 1] * vy)
    remaining = np.float64(ball.DT)
    signature: list[int] = []
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
            ux = np.float64(rho * ux)
        if hit_y:
            y = np.float64(ball.YMAX if uy > 0.0 else ball.YMIN)
            uy = np.float64(rho * uy)
    return tuple(signature)


def numpy_rollout(initial: np.ndarray, theta: np.ndarray, steps: int) -> np.ndarray:
    values = np.empty((steps + 1, 4), dtype=np.float64)
    values[0] = initial
    for index in range(steps):
        values[index + 1] = numpy_step(values[index], theta)
    return values


def objectives(trajectory: np.ndarray):
    prefix = np.asarray(trajectory[:ball.PREFIX_STATES], dtype=np.float64)
    torch_prefix = torch.from_numpy(prefix.copy())

    def numpy_objective(theta: np.ndarray) -> float:
        squared = np.float64(0.0)
        for index in range(ball.PREFIX_STATES - 1):
            error = numpy_step(prefix[index], theta) - prefix[index + 1]
            squared = np.float64(squared + np.dot(error, error))
        return float(squared / np.float64((ball.PREFIX_STATES - 1) * 4))

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        errors = [torch_step(torch_prefix[index], theta) - torch_prefix[index + 1]
                  for index in range(ball.PREFIX_STATES - 1)]
        return torch.stack(errors).square().mean()

    return numpy_objective, torch_objective


def _trajectories() -> np.ndarray:
    values, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=SOURCE_TRAJECTORIES,
        n_timesteps=ball.TOTAL_STEPS,
        dt=ball.DT,
        seed=TRAJECTORY_SEED,
    )
    return values


def validate_teacher(trajectories: np.ndarray) -> dict:
    controls = np.asarray((
        (0.0, 0.0, 1.0, -0.5),            # ordinary no-impact
        (5.0, 0.0, 1.0, 0.0),             # endpoint impact
        (4.9, 4.9, 2.0, 2.0),             # corner impact
        (5.0, -5.0, 1.0, -1.0),           # endpoint corner
        (0.0, 0.0, 250.0, -175.0),        # multiple impacts
        (1.0, -2.0, 0.0, 0.0),            # zero velocity
    ), dtype=np.float64)
    authority = np.asarray([
        ball.TEACHER.step_bouncing_ball(*state, ball.DT) for state in controls
    ])
    numpy_values = np.asarray([numpy_step(state, TEACHER) for state in controls])
    torch_values = np.asarray([
        torch_step(torch.from_numpy(state), torch.from_numpy(TEACHER)).detach().numpy()
        for state in controls
    ])
    rollout_error = 0.0
    exact_values = 0
    total_values = 0
    zero_losses = []
    for trajectory in trajectories:
        prediction = numpy_rollout(trajectory[0], TEACHER, len(trajectory) - 1)
        rollout_error = max(rollout_error, float(np.max(np.abs(prediction - trajectory))))
        exact_values += int(np.count_nonzero(prediction == trajectory))
        total_values += int(trajectory.size)
        numpy_objective, torch_objective = objectives(trajectory)
        zero_losses.append((numpy_objective(TEACHER),
                            float(torch_objective(torch.from_numpy(TEACHER)))))
    numpy_error = float(np.max(np.abs(numpy_values - authority)))
    torch_error = float(np.max(np.abs(torch_values - authority)))
    passed = bool(
        numpy_error <= 8 * np.finfo(np.float64).eps
        and torch_error <= 8 * np.finfo(np.float64).eps
        and rollout_error <= 8 * np.finfo(np.float64).eps
        and all(left == 0.0 and right == 0.0 for left, right in zero_losses)
    )
    return dict(
        passed=passed,
        hypothesis="theta=(k11,k12,k21,k22,rho); on each impacted coordinate: u<-rho*u, w<-rho*w",
        teacher=TEACHER.tolist(),
        bounds=[LOWER.tolist(), UPPER.tolist()],
        control_labels=["ordinary", "endpoint", "corner", "endpoint_corner",
                        "multiple_impact", "zero_velocity"],
        numpy_control_max_abs=numpy_error,
        torch_control_max_abs=torch_error,
        uniform_trajectories=len(trajectories),
        rollout_max_abs=rollout_error,
        exact_rollout_values=exact_values,
        total_rollout_values=total_values,
        zero_objectives=sum(left == 0.0 and right == 0.0 for left, right in zero_losses),
    )


def _checkpoint(points, losses, iteration, traces) -> dict:
    best = int(np.argmin(losses))
    theta = LOWER + WIDTH * points[best]
    radius = WIDTH / len(losses) ** (1.0 / DIMENSION)
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
        continuous_duplicate_rejections=int(sum(
            row["exact_duplicate_rejected"] for row in traces
        )),
    )


def _search(objective, case_id: int) -> dict:
    api = edirect.composition(DIMENSION)
    traces: list[dict] = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
    points: list[np.ndarray] = []
    losses: list[float] = []
    events: list[dict] = []
    checkpoints: list[dict] = []
    iteration = 0

    def evaluate(unit, source):
        theta = LOWER + WIDTH * np.asarray(unit, dtype=np.float64)
        value = float(objective(theta))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite reflection-peelback objective")
        points.append(np.asarray(unit, dtype=np.float64).copy())
        losses.append(value)
        events.append(dict(evaluation=len(losses), iteration=iteration,
                           source=source, point=np.asarray(unit).tolist(), loss=value))
        if len(losses) % BLOCK == 0:
            checkpoints.append(_checkpoint(np.asarray(points), np.asarray(losses),
                                           iteration, traces))

    for unit in edirect.initial_design(DIMENSION):
        evaluate(unit, "initial")
    initial_best = int(np.argmin(losses))
    initial_state = dict(
        evaluation=len(losses),
        theta=(LOWER + WIDTH * points[initial_best]).tolist(),
        loss=float(losses[initial_best]),
    )
    stop = False
    while len(losses) < CAP and not stop:
        proposals = api["_cell_proposals"](
            np.asarray(points), np.asarray(losses), 3_000_000 + case_id, iteration
        )
        for unit in proposals:
            if len(losses) >= CAP:
                break
            evaluate(unit, "cell")
            if checkpoints and checkpoints[-1]["evaluation"] == len(losses):
                if checkpoints[-1]["both_ok"]:
                    stop = True
                    break
        if stop or len(losses) >= CAP:
            break
        proposal = api["_greedy_proposal"](
            np.asarray(points), np.asarray(losses), 3_000_000 + case_id, iteration
        )
        if proposal is not None:
            evaluate(proposal, "continuous_surrogate")
            if checkpoints and checkpoints[-1]["evaluation"] == len(losses):
                stop = checkpoints[-1]["both_ok"]
        iteration += 1
    if not checkpoints or checkpoints[-1]["evaluation"] != len(losses):
        checkpoints.append(_checkpoint(np.asarray(points), np.asarray(losses),
                                       iteration, traces))
    selected = next((row for row in checkpoints if row["both_ok"]), checkpoints[-1])
    return dict(
        points=np.asarray(points), losses=np.asarray(losses), events=events,
        traces=traces, checkpoints=checkpoints, selected=selected,
        initial_state=initial_state, iterations=iteration,
    )


def _collision_diagnostics(trajectory: np.ndarray, theta: np.ndarray) -> dict:
    start = ball.PREFIX_STATES - 1
    steps = ball.TOTAL_STEPS - start
    prediction = numpy_rollout(trajectory[start], theta, steps)
    truth = trajectory[start:]
    topology = sum(
        event_signature(prediction[index], theta)
        != event_signature(truth[index], TEACHER)
        for index in range(steps)
    )
    prefix_signatures = [event_signature(trajectory[index], TEACHER)
                         for index in range(ball.PREFIX_STATES - 1)]
    error = prediction - truth
    return dict(
        prefix_collision_events=int(sum(len(value) for value in prefix_signatures)),
        prefix_corner_steps=int(sum(any(mask == 3 for mask in value)
                                    for value in prefix_signatures)),
        prefix_multiple_impact_steps=int(sum(len(value) > 1
                                             for value in prefix_signatures)),
        continuation_max_error=float(np.max(np.abs(error))),
        continuation_mse=float(np.mean(error * error)),
        collision_topology_regressions=int(topology),
        rollout_finite=bool(np.isfinite(prediction).all()),
    )


def run_case(case_id: int, trajectory: np.ndarray) -> dict:
    case_dir = CASES / f"case_{case_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    started = time.time()
    numpy_objective, torch_objective = objectives(trajectory)
    derivatives, direction_fn, run_n3 = staged._make_n3(torch_objective, LOWER, UPPER)
    search = _search(numpy_objective, case_id)
    handoff = search["selected"]

    initial_n3 = staged._run_n3(
        np.asarray(search["initial_state"]["theta"]),
        float(search["initial_state"]["loss"]), "initial_design", run_n3,
    )
    n1_n3 = staged._run_n3(
        np.asarray(handoff["best_theta"]), float(handoff["best_loss"]),
        "n1_endpoint", run_n3,
    )
    n2 = staged._run_n2(
        numpy_objective, np.asarray(handoff["best_theta"]), float(handoff["best_loss"]),
        np.asarray(handoff["box_lower"]), np.asarray(handoff["box_upper"]),
    )
    n3 = staged._run_n3(n2["theta"], n2["loss"], "n2_endpoint", run_n3)
    first_newton = staged._first_newton(n2["theta"], derivatives, direction_fn, TEACHER)
    final = np.asarray(n3["final_theta"], dtype=np.float64)
    error = final - TEACHER
    collision = _collision_diagnostics(trajectory, final)
    checkpoints = search["checkpoints"]
    result = dict(
        case_id=case_id, source_trajectory=int(SELECTED[case_id]),
        bounds=[LOWER.tolist(), UPPER.tolist()], teacher=TEACHER.tolist(),
        n1_evaluations=int(len(search["losses"])),
        n1_iterations=int(search["iterations"]),
        n1_cell_proposals=sum(row["source"] == "cell" for row in search["events"]),
        n1_continuous_proposals=sum(row["source"] == "continuous_surrogate"
                                    for row in search["events"]),
        continuous_surrogate_calls=len(search["traces"]),
        continuous_duplicate_rejections=int(sum(
            row["exact_duplicate_rejected"] for row in search["traces"]
        )),
        n1_loss_ok=any(row["loss_ok"] for row in checkpoints),
        containment_ok=any(row["containment_ok"] for row in checkpoints),
        both_ok=any(row["both_ok"] for row in checkpoints),
        handoff_evaluation=int(handoff["evaluation"]),
        n1_theta=handoff["best_theta"], n1_loss=handoff["best_loss"],
        n1_box_lower=handoff["box_lower"], n1_box_upper=handoff["box_upper"],
        initial_n3_correctable=initial_n3["recovered"],
        n1_endpoint_n3_correctable=n1_n3["recovered"],
        n2_theta=n2["theta"].tolist(), n2_loss=float(n2["loss"]),
        n2_evaluations=n2["evaluations"], n2_gate=n2["gate"],
        n2_message=n2["message"],
        n3_accepted_steps=n3["accepted_steps"],
        n3_evaluations=n3["objective_evaluations"], n3_stop=n3["stop_reason"],
        final_theta=n3["final_theta"], final_loss=n3["final_loss"],
        final_entrywise_error=error.tolist(),
        final_max_parameter_error=float(np.max(np.abs(error))),
        final_parameter_l2_error=float(np.linalg.norm(error)),
        recovered=bool(n3["recovered"]),
        first_newton=first_newton, seconds=time.time() - started,
        checkpoints=checkpoints, **collision,
    )
    np.savez_compressed(case_dir / "n1_state.npz",
                        points=search["points"], losses=search["losses"])
    _atomic_text(case_dir / "events.json", json.dumps(search["events"]))
    _atomic_text(case_dir / "continuous.json", json.dumps(search["traces"]))
    _atomic_text(result_path, json.dumps(result, indent=2))
    return result


def _distribution(values) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return dict(minimum=float(np.min(data)), median=float(np.median(data)),
                p90=float(np.quantile(data, 0.9, method="higher")),
                maximum=float(np.max(data)), total=float(np.sum(data)))


def _report(rows, validation, before, after, elapsed) -> dict:
    return dict(
        status=("VALIDATED_DETERMINISTIC_HYPOTHESIS" if
                validation["passed"] and all(row["recovered"] for row in rows)
                and before == after else "FAILED_DETERMINISTIC_RECOVERY"),
        hypothesis=validation["hypothesis"], teacher=TEACHER.tolist(),
        parameter_order=["k11", "k12", "k21", "k22", "rho"],
        total_dimension=DIMENSION, shared_event_parameter=True,
        bounds=[LOWER.tolist(), UPPER.tolist()], teacher_validation=validation,
        cases=len(rows), recoveries=sum(row["recovered"] for row in rows),
        n1_loss_gate_successes=sum(row["n1_loss_ok"] for row in rows),
        containment_successes=sum(row["containment_ok"] for row in rows),
        n2_gate_successes=sum(row["n2_gate"] for row in rows),
        initial_n3_correctable=sum(row["initial_n3_correctable"] for row in rows),
        n1_endpoint_n3_correctable=sum(row["n1_endpoint_n3_correctable"] for row in rows),
        n1_evaluations=_distribution([row["n1_evaluations"] for row in rows]),
        n2_evaluations=_distribution([row["n2_evaluations"] for row in rows]),
        n3_evaluations=_distribution([row["n3_evaluations"] for row in rows]),
        n3_accepted_steps=_distribution([row["n3_accepted_steps"] for row in rows]),
        worst_final_parameter_error=max(row["final_max_parameter_error"] for row in rows),
        worst_final_parameter_l2_error=max(row["final_parameter_l2_error"] for row in rows),
        worst_continuation_error=max(row["continuation_max_error"] for row in rows),
        worst_continuation_mse=max(row["continuation_mse"] for row in rows),
        collision_topology_regressions=sum(row["collision_topology_regressions"]
                                           for row in rows),
        checkpoint_hashes_unchanged=before == after,
        protected_checkpoint_count=len(before), elapsed_seconds=elapsed,
        cases_detail=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    trajectories = _trajectories()
    validation = validate_teacher(trajectories)
    _atomic_text(OUT / "teacher_validation.json", json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise RuntimeError("reflection-peelback teacher containment failed")
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    if args.case_index is not None:
        source = SELECTED[args.case_index]
        result = run_case(args.case_index, trajectories[source])
        print(json.dumps(dict(case=args.case_index, source=source,
                              n1=result["n1_evaluations"],
                              n2=result["n2_evaluations"],
                              recovered=result["recovered"]), indent=2), flush=True)
        return
    started = time.time()
    before = hashes()
    _atomic_text(OUT / "checkpoint_hashes_before.json", json.dumps(before, indent=2))
    workers = max(1, min(args.workers, len(SELECTED)))
    for offset in range(0, len(SELECTED), workers):
        indices = list(range(offset, min(offset + workers, len(SELECTED))))
        children = []
        for index in indices:
            command = [sys.executable, "-m",
                       "experiments.bouncing_ball.run_reflection_peelback",
                       "--case-index", str(index)]
            children.append((index, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(index, child.wait()) for index, child in children]
        failures = [(index, code) for index, code in failures if code]
        if failures:
            raise RuntimeError(f"case process failures: {failures}")
    rows = [json.loads((CASES / f"case_{index:02d}" / "result.json").read_text())
            for index in range(len(SELECTED))]
    after = hashes()
    _atomic_text(OUT / "checkpoint_hashes_after.json", json.dumps(after, indent=2))
    report = _report(rows, validation, before, after, time.time() - started)
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in (
        "status", "cases", "recoveries", "n1_loss_gate_successes",
        "n2_gate_successes", "initial_n3_correctable",
        "worst_final_parameter_error", "worst_continuation_error",
        "collision_topology_regressions", "checkpoint_hashes_unchanged",
    )}, indent=2))


if __name__ == "__main__":
    main()
