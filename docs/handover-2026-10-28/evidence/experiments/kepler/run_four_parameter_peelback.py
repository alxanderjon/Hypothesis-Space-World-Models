"""Deterministic four-parameter Kepler E-DIRECT -> N2 -> N3 experiment.

The four parameters free genuine coefficients of velocity Verlet:

    v_half = v + kick_1 * dt * a_p(r)
    r_next = r + drift * dt * v_half
    v_next = v_half + kick_2 * dt * a_p(r_next)

where a_p retains the exact shared central-force structure and masses.  The
teacher is (kick_1, drift, kick_2, p) = (0.5, 1, 0.5, 3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.stats import qmc
import torch

from experiments.kepler import deterministic_teacher as kepler
from experiments.kepler.run_first_transfer import checkpoint_paths, manifest
from experiments.n2_exact_powell_assembly import minimize_powell_exact
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/kepler/four_parameter_deterministic_peelback")
CASES = OUT / "cases"
LOWER = np.asarray((-1.0, -0.5, -1.0, 1.5), dtype=np.float64)
UPPER = np.asarray((2.0, 2.5, 2.0, 4.5), dtype=np.float64)
WIDTH = UPPER - LOWER
TEACHER = np.asarray((0.5, 1.0, 0.5, 3.0), dtype=np.float64)
CAP = 5_000
BLOCK = 100
N2_CAP = 1_500
FINAL_GATE = 1.0e-20
LOSS_GATE = 1.0e-4

# Disjoint from the earlier eight two-parameter transfer cases.
FRESH_CASES = (
    (0.22, 0.78, 0.41, 17),
    (0.31, 0.96, 1.11, 37),
    (0.39, 1.12, 1.76, 61),
    (0.51, 0.86, 2.39, 83),
    (0.64, 1.24, 2.94, 109),
    (0.27, 1.42, 3.57, 137),
    (0.46, 0.92, 4.46, 163),
    (0.59, 1.52, 5.63, 211),
)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def trajectories() -> list[np.ndarray]:
    output = []
    for eccentricity, periapsis, orientation, phase_steps in FRESH_CASES:
        initial = kepler.initial_state(eccentricity, periapsis, orientation)
        phased = kepler.numpy_rollout(initial, steps=phase_steps)[-1]
        output.append(kepler.numpy_rollout(phased, steps=kepler.TOTAL_STEPS))
    return output


def numpy_acceleration_batch(states: np.ndarray, exponent: np.float64) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    dx = values[..., 4] - values[..., 0]
    dy = values[..., 5] - values[..., 1]
    radius_squared = dx * dx + dy * dy
    radius = np.sqrt(radius_squared)
    denominator = np.power(radius, exponent)
    common = np.float64(kepler.G) / denominator
    scale_1 = common * np.float64(kepler.MASS_2)
    scale_2 = common * np.float64(kepler.MASS_1)
    return np.stack((scale_1 * dx, scale_1 * dy,
                     -scale_2 * dx, -scale_2 * dy), axis=-1)


def numpy_step_batch(states: np.ndarray, theta: np.ndarray) -> np.ndarray:
    kick_1, drift, kick_2, exponent = np.asarray(theta, dtype=np.float64)
    values = np.asarray(states, dtype=np.float64)
    dt = np.float64(kepler.DT)
    first_scale = np.float64(kick_1 * dt)
    drift_scale = np.float64(drift * dt)
    final_scale = np.float64(kick_2 * dt)
    acceleration = numpy_acceleration_batch(values, exponent)
    v1hx = values[..., 2] + first_scale * acceleration[..., 0]
    v1hy = values[..., 3] + first_scale * acceleration[..., 1]
    v2hx = values[..., 6] + first_scale * acceleration[..., 2]
    v2hy = values[..., 7] + first_scale * acceleration[..., 3]
    x1 = values[..., 0] + drift_scale * v1hx
    y1 = values[..., 1] + drift_scale * v1hy
    x2 = values[..., 4] + drift_scale * v2hx
    y2 = values[..., 5] + drift_scale * v2hy
    drifted = np.stack((x1, y1, v1hx, v1hy, x2, y2, v2hx, v2hy), axis=-1)
    next_acceleration = numpy_acceleration_batch(drifted, exponent)
    return np.stack((x1, y1,
                     v1hx + final_scale * next_acceleration[..., 0],
                     v1hy + final_scale * next_acceleration[..., 1],
                     x2, y2,
                     v2hx + final_scale * next_acceleration[..., 2],
                     v2hy + final_scale * next_acceleration[..., 3]), axis=-1)


def numpy_step(state: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return numpy_step_batch(np.asarray(state, dtype=np.float64)[None], theta)[0]


def numpy_rollout(initial: np.ndarray, theta: np.ndarray, steps: int) -> np.ndarray:
    output = np.empty((steps + 1, 8), dtype=np.float64)
    output[0] = initial
    for index in range(steps):
        output[index + 1] = numpy_step(output[index], theta)
    return output


def torch_acceleration_batch(states: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    dx = states[..., 4] - states[..., 0]
    dy = states[..., 5] - states[..., 1]
    radius = torch.sqrt(dx * dx + dy * dy)
    denominator = torch.pow(radius, exponent)
    common = torch.as_tensor(float(kepler.G)) / denominator
    scale_1 = common * torch.as_tensor(float(kepler.MASS_2))
    scale_2 = common * torch.as_tensor(float(kepler.MASS_1))
    return torch.stack((scale_1 * dx, scale_1 * dy,
                        -scale_2 * dx, -scale_2 * dy), dim=-1)


def torch_step_batch(states: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    kick_1, drift, kick_2, exponent = theta.unbind()
    dt = torch.as_tensor(float(kepler.DT))
    acceleration = torch_acceleration_batch(states, exponent)
    v1hx = states[..., 2] + kick_1 * dt * acceleration[..., 0]
    v1hy = states[..., 3] + kick_1 * dt * acceleration[..., 1]
    v2hx = states[..., 6] + kick_1 * dt * acceleration[..., 2]
    v2hy = states[..., 7] + kick_1 * dt * acceleration[..., 3]
    x1 = states[..., 0] + drift * dt * v1hx
    y1 = states[..., 1] + drift * dt * v1hy
    x2 = states[..., 4] + drift * dt * v2hx
    y2 = states[..., 5] + drift * dt * v2hy
    drifted = torch.stack((x1, y1, v1hx, v1hy, x2, y2, v2hx, v2hy), dim=-1)
    next_acceleration = torch_acceleration_batch(drifted, exponent)
    return torch.stack((x1, y1,
                        v1hx + kick_2 * dt * next_acceleration[..., 0],
                        v1hy + kick_2 * dt * next_acceleration[..., 1],
                        x2, y2,
                        v2hx + kick_2 * dt * next_acceleration[..., 2],
                        v2hy + kick_2 * dt * next_acceleration[..., 3]), dim=-1)


def objectives(trajectory: np.ndarray):
    prefix = np.asarray(trajectory[:kepler.PREFIX_STATES], dtype=np.float64)
    inputs = prefix[:-1]
    targets = prefix[1:]
    torch_inputs = torch.from_numpy(inputs.copy())
    torch_targets = torch.from_numpy(targets.copy())

    def numpy_objective(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            predicted = numpy_step_batch(inputs, theta)
            error = predicted - targets
            value = float(np.mean(error * error))
        return value if math.isfinite(value) else 1.0e100

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        predicted = torch_step_batch(torch_inputs, theta)
        return torch.mean((predicted - torch_targets) ** 2)

    return numpy_objective, torch_objective


def validate_teacher(values: list[np.ndarray]) -> dict:
    one_step_max = 0.0
    torch_max = 0.0
    rollout_max = 0.0
    exact_one_step = 0
    total_one_step = 0
    zero_losses = []
    for trajectory in values:
        teacher_next = np.asarray([
            kepler.numpy_step(state) for state in trajectory[:kepler.PREFIX_STATES - 1]
        ])
        candidate_next = numpy_step_batch(
            trajectory[:kepler.PREFIX_STATES - 1], TEACHER
        )
        torch_next = torch_step_batch(
            torch.from_numpy(trajectory[:kepler.PREFIX_STATES - 1].copy()),
            torch.from_numpy(TEACHER.copy()),
        ).detach().numpy()
        one_step_max = max(one_step_max, float(np.max(np.abs(candidate_next - teacher_next))))
        torch_max = max(torch_max, float(np.max(np.abs(torch_next - teacher_next))))
        exact_one_step += int(np.count_nonzero(candidate_next == teacher_next))
        total_one_step += int(candidate_next.size)
        predicted = numpy_rollout(trajectory[0], TEACHER, len(trajectory) - 1)
        rollout_max = max(rollout_max, float(np.max(np.abs(predicted - trajectory))))
        numpy_objective, torch_objective = objectives(trajectory)
        zero_losses.append((numpy_objective(TEACHER),
                            float(torch_objective(torch.from_numpy(TEACHER.copy())))))

    controls = qmc.Sobol(4, scramble=True, seed=20261002).random_base2(9)
    candidates = LOWER + WIDTH * controls
    invalid_one_step = 0
    objective_values = []
    for trajectory in values:
        objective, _ = objectives(trajectory)
        for theta in candidates:
            value = objective(theta)
            objective_values.append(value)
            invalid_one_step += int(not math.isfinite(value) or value >= 1.0e100)
    passed = bool(
        one_step_max == 0.0
        and torch_max == 0.0
        and rollout_max == 0.0
        and all(a == 0.0 and b == 0.0 for a, b in zero_losses)
    )
    return dict(
        passed=passed,
        exact_numpy_one_step_values=exact_one_step,
        total_numpy_one_step_values=total_one_step,
        numpy_one_step_max_abs=one_step_max,
        torch_one_step_max_abs=torch_max,
        long_rollout_max_abs=rollout_max,
        numpy_ground_truth_losses=[value[0] for value in zero_losses],
        torch_ground_truth_losses=[value[1] for value in zero_losses],
        safety_candidates_per_case=len(candidates),
        safety_objective_evaluations=len(objective_values),
        invalid_one_step_objectives=invalid_one_step,
        safety_objective_min=float(np.min(objective_values)),
        safety_objective_max=float(np.max(objective_values)),
    )


def _make_n3(torch_objective, lower: np.ndarray, upper: np.ndarray):
    def derivatives(theta: torch.Tensor, _position: torch.Tensor):
        differentiable = theta.detach().clone().requires_grad_(True)
        loss = torch_objective(differentiable)
        gradient = torch.autograd.grad(loss, differentiable, create_graph=True)[0]
        hessian = torch.stack([
            torch.autograd.grad(gradient[index], differentiable, retain_graph=True)[0]
            for index in range(len(differentiable))
        ])
        hessian = 0.5 * (hessian + hessian.T)
        return loss.detach(), gradient.detach(), hessian.detach()

    namespace = dict(polish.__dict__)
    namespace["LOWER"] = torch.from_numpy(lower.copy())
    namespace["UPPER"] = torch.from_numpy(upper.copy())
    namespace["_scalar_loss"] = lambda theta, _position: torch_objective(theta)
    namespace["_gradient_hessian"] = derivatives
    source = inspect.getsource(polish._newton_direction).replace(
        "torch.eye(2, dtype=DTYPE)",
        "torch.eye(len(gradient), dtype=DTYPE)",
    )
    exec(compile(source, polish.__file__, "exec"), namespace)
    run = types.FunctionType(polish._polish.__code__, namespace)
    return derivatives, namespace["_newton_direction"], run


def _first_newton(theta: np.ndarray, derivatives, direction_fn, teacher: np.ndarray) -> dict:
    _, gradient_tensor, hessian_tensor = derivatives(
        torch.from_numpy(theta.copy()), torch.empty(0)
    )
    direction_tensor, condition, damping, status = direction_fn(
        gradient_tensor, hessian_tensor
    )
    gradient = gradient_tensor.numpy()
    direction = None if direction_tensor is None else direction_tensor.numpy()
    displacement = teacher - theta
    distance = float(np.linalg.norm(displacement))
    step_norm = float(np.linalg.norm(direction)) if direction is not None else None
    alignment = None
    ratio = None
    if direction is not None and step_norm > 0.0 and distance > 0.0:
        alignment = float(np.dot(direction, displacement) / (step_norm * distance))
        ratio = step_norm / distance
    eigenvalues = np.linalg.eigvalsh(hessian_tensor.numpy())
    return dict(
        gradient=gradient.tolist(),
        gradient_norm=float(np.linalg.norm(gradient)),
        hessian_eigenvalues=eigenvalues.tolist(),
        hessian_condition=condition,
        newton_direction=(direction.tolist() if direction is not None else None),
        newton_step_norm=step_norm,
        newton_truth_alignment=alignment,
        newton_step_truth_distance_ratio=ratio,
        newton_route=status,
        newton_damping=damping,
        truth_distance=distance,
    )


def _run_n3(theta: np.ndarray, loss: float, category: str, run_n3) -> dict:
    endpoint = polish.N2Endpoint(
        0, theta.copy(), float(loss), "deterministic_kepler_4d", category, False
    )
    result = run_n3(endpoint, torch.empty(0))
    return dict(
        final_theta=result.theta_history[-1].tolist(),
        final_loss=float(result.loss_history[-1]),
        accepted_steps=int(result.accepted_steps),
        objective_evaluations=int(result.actual_loss_evaluations),
        stop_reason=result.stop_reason,
        backtracking_rejections=int(result.backtracking_rejections),
        damping_steps=int(result.damping_steps),
        recovered=bool(result.loss_history[-1] < FINAL_GATE),
    )


def _run_n2(numpy_objective, start_theta, start_loss, lower, upper) -> dict:
    """Run the SciPy-1.13.0-exact learned Powell assembly and return strict best-ever."""
    best_theta = np.asarray(start_theta, dtype=np.float64).copy()
    best_loss = float(start_loss)
    trace = []

    def counted(theta):
        nonlocal best_theta, best_loss
        value = float(numpy_objective(np.asarray(theta, dtype=np.float64)))
        trace.append(dict(theta=np.asarray(theta).tolist(), loss=value))
        if value < best_loss:
            best_theta = np.asarray(theta, dtype=np.float64).copy()
            best_loss = value
        return value

    result = minimize_powell_exact(
        counted,
        best_theta,
        bounds=list(zip(lower, upper)),
        xtol=1e-12,
        ftol=1e-15,
        maxiter=200,
        maxfev=N2_CAP,
    )
    return dict(
        theta=best_theta,
        loss=best_loss,
        evaluations=int(result["nfev"]),
        message=str(result["message"]),
        gate=bool(best_loss < FINAL_GATE),
        trace=trace,
    )


def _checkpoint(points, losses, iteration, traces):
    best = int(np.argmin(losses))
    theta = LOWER + WIDTH * points[best]
    radius = WIDTH / len(losses) ** 0.25
    return dict(
        evaluation=len(losses),
        iteration=iteration,
        best_theta=theta.tolist(),
        best_loss=float(losses[best]),
        old_loss_gate=bool(losses[best] <= LOSS_GATE),
        box_lower=np.maximum(LOWER, theta - radius).tolist(),
        box_upper=np.minimum(UPPER, theta + radius).tolist(),
        continuous_calls=len(traces),
    )


def _save_resume(case_dir, points, losses, events, traces, checkpoints,
                 iteration, phase, pending, next_pending):
    temporary = case_dir / "resume.tmp.npz"
    np.savez_compressed(
        temporary,
        points=np.asarray(points),
        losses=np.asarray(losses),
        pending=np.asarray(pending, dtype=np.float64),
    )
    temporary.replace(case_dir / "resume.npz")
    _atomic_text(case_dir / "resume.json", json.dumps(dict(
        events=events,
        traces=traces,
        checkpoints=checkpoints,
        iteration=iteration,
        phase=phase,
        next_pending=next_pending,
    )))


def _run_n1(case_id: int, numpy_objective, case_dir: Path) -> dict:
    api = edirect.composition(4)
    traces = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
    points: list[np.ndarray] = []
    losses: list[float] = []
    events: list[dict] = []
    checkpoints: list[dict] = []
    iteration, phase, next_pending = 0, "initial", 0
    pending = np.empty((0, 4), dtype=np.float64)
    resume_npz = case_dir / "resume.npz"
    resume_json = case_dir / "resume.json"
    if resume_npz.exists() and resume_json.exists():
        saved = np.load(resume_npz)
        points = [point.copy() for point in saved["points"]]
        losses = saved["losses"].tolist()
        pending = saved["pending"].copy()
        metadata = json.loads(resume_json.read_text())
        events = metadata["events"]
        traces = metadata["traces"]
        checkpoints = metadata["checkpoints"]
        iteration = metadata["iteration"]
        phase = metadata["phase"]
        next_pending = metadata["next_pending"]
        api["_greedy_proposal"] = (
            lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
        )

    def evaluate(unit, source):
        theta = LOWER + WIDTH * np.asarray(unit)
        value = float(numpy_objective(theta))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite four-parameter Kepler objective")
        points.append(np.asarray(unit).copy())
        losses.append(value)
        events.append(dict(
            evaluation=len(losses), iteration=iteration, source=source,
            point=np.asarray(unit).tolist(), theta=theta.tolist(), loss=value,
        ))

    def save_if_due(current_phase, current_pending, current_next):
        if not losses or len(losses) % BLOCK:
            return
        checkpoints.append(_checkpoint(
            np.asarray(points), np.asarray(losses), iteration, traces
        ))
        _save_resume(
            case_dir, points, losses, events, traces, checkpoints,
            iteration, current_phase, current_pending, current_next,
        )

    if phase == "initial":
        design = edirect.initial_design(4)
        for unit in design[len(points):]:
            evaluate(unit, "initial")
        phase = "next_round"
    while len(losses) < CAP:
        if phase == "next_round":
            pending = api["_cell_proposals"](
                np.asarray(points), np.asarray(losses), 4_000_000 + case_id, iteration
            )
            next_pending = 0
            phase = "cell"
        if phase == "cell":
            while next_pending < len(pending) and len(losses) < CAP:
                evaluate(pending[next_pending], "cell")
                next_pending += 1
                save_if_due("cell", pending, next_pending)
            phase = "greedy"
        if phase == "greedy" and len(losses) < CAP:
            proposal = api["_greedy_proposal"](
                np.asarray(points), np.asarray(losses), 4_000_000 + case_id, iteration
            )
            if proposal is not None:
                evaluate(proposal, "continuous_surrogate")
            iteration += 1
            phase = "next_round"
            pending = np.empty((0, 4), dtype=np.float64)
            next_pending = 0
            save_if_due("next_round", pending, 0)

    if not checkpoints or checkpoints[-1]["evaluation"] != len(losses):
        checkpoints.append(_checkpoint(
            np.asarray(points), np.asarray(losses), iteration, traces
        ))
    best = int(np.argmin(losses))
    theta = LOWER + WIDTH * points[best]
    radius = WIDTH / len(losses) ** 0.25
    np.savez_compressed(
        case_dir / "n1_state.npz", points=np.asarray(points), losses=np.asarray(losses)
    )
    _atomic_text(case_dir / "events.json", json.dumps(events))
    _atomic_text(case_dir / "continuous.json", json.dumps(traces))
    return dict(
        theta=theta,
        loss=float(losses[best]),
        lower=np.maximum(LOWER, theta - radius),
        upper=np.minimum(UPPER, theta + radius),
        evaluations=len(losses),
        iterations=iteration,
        events=events,
        checkpoints=checkpoints,
        traces=traces,
    )


def _correctability_samples(case_id, trajectory, n1_result, n2_result,
                            torch_objective, derivatives, direction_fn, run_n3):
    desired_evaluations = {1, 100, 500, 1000, 2500, 5000}
    samples = []
    events = n1_result["events"]
    best_event = None
    best_loss = math.inf
    for event in events:
        if event["loss"] < best_loss:
            best_loss = event["loss"]
            best_event = event
        if event["evaluation"] in desired_evaluations:
            samples.append((
                "n1_state", int(event["evaluation"]),
                np.asarray(best_event["theta"], dtype=np.float64), float(best_loss),
            ))
    samples.append((
        "n2_endpoint", int(n2_result["evaluations"]),
        np.asarray(n2_result["theta"], dtype=np.float64), float(n2_result["loss"]),
    ))
    output = []
    seen = set()
    for source, evaluation, theta, loss in samples:
        key = (source, theta.tobytes())
        if key in seen:
            continue
        seen.add(key)
        first = _first_newton(theta, derivatives, direction_fn, TEACHER)
        finish = _run_n3(theta, loss, "kepler_4d_correctability", run_n3)
        final = np.asarray(finish["final_theta"])
        output.append(dict(
            case_id=case_id,
            source=source,
            evaluation=evaluation,
            theta=theta.tolist(),
            loss=loss,
            final_parameter_error=float(np.linalg.norm(final - TEACHER)),
            **first,
            **{f"n3_{key}": value for key, value in finish.items()},
        ))
    return output


def _two_parameter_reference(values: list[np.ndarray]) -> list[dict]:
    lower = kepler.LOWER.copy()
    upper = kepler.UPPER.copy()
    design = edirect.initial_design(2)
    candidates = lower + (upper - lower) * design
    output = []
    for case_id, trajectory in enumerate(values):
        torch_trajectory = torch.from_numpy(trajectory.copy())
        numpy_losses = kepler.prefix_loss_numpy(candidates, trajectory)

        def torch_objective(theta):
            return kepler.prefix_loss_torch(theta, torch_trajectory)

        derivatives, direction_fn, run_n3 = _make_n3(torch_objective, lower, upper)
        selected = sorted(set((0, len(candidates) // 2, len(candidates) - 1,
                               int(np.argmin(numpy_losses)))))
        for candidate_id in selected:
            theta = candidates[candidate_id]
            first = _first_newton(theta, derivatives, direction_fn, kepler.TEACHER_THETA)
            finish = _run_n3(
                theta, float(numpy_losses[candidate_id]), "kepler_2d_reference", run_n3
            )
            output.append(dict(
                case_id=case_id,
                candidate_id=candidate_id,
                theta=theta.tolist(),
                loss=float(numpy_losses[candidate_id]),
                final_parameter_error=float(np.linalg.norm(
                    np.asarray(finish["final_theta"]) - kepler.TEACHER_THETA
                )),
                **first,
                **{f"n3_{key}": value for key, value in finish.items()},
            ))
    return output


def run_case(case_id: int, trajectory: np.ndarray) -> dict:
    torch.set_num_threads(1)
    case_dir = CASES / f"trajectory_{case_id:02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    started = time.time()
    numpy_objective, torch_objective = objectives(trajectory)
    n1_result = _run_n1(case_id, numpy_objective, case_dir)
    n2_result = _run_n2(
        numpy_objective,
        n1_result["theta"],
        n1_result["loss"],
        n1_result["lower"],
        n1_result["upper"],
    )
    derivatives, direction_fn, run_n3 = _make_n3(torch_objective, LOWER, UPPER)
    first_n2 = _first_newton(n2_result["theta"], derivatives, direction_fn, TEACHER)
    n3_result = _run_n3(
        n2_result["theta"], n2_result["loss"], "kepler_4d_pipeline", run_n3
    )
    final = np.asarray(n3_result["final_theta"], dtype=np.float64)
    continuation_start = kepler.PREFIX_STATES - 1
    continuation = numpy_rollout(
        trajectory[continuation_start], final, len(trajectory) - continuation_start - 1
    )
    truth = trajectory[continuation_start:]
    rollout_error = continuation - truth
    correctability = _correctability_samples(
        case_id, trajectory, n1_result, n2_result,
        torch_objective, derivatives, direction_fn, run_n3,
    )
    _atomic_text(case_dir / "correctability.json", json.dumps(correctability, indent=2))
    _atomic_text(case_dir / "n2_trace.json", json.dumps(n2_result["trace"]))
    result = dict(
        trajectory_id=case_id,
        case_definition=dict(
            eccentricity=FRESH_CASES[case_id][0],
            periapsis=FRESH_CASES[case_id][1],
            orientation=FRESH_CASES[case_id][2],
            phase_steps=FRESH_CASES[case_id][3],
        ),
        n1_theta=n1_result["theta"].tolist(),
        n1_loss=n1_result["loss"],
        n1_evaluations=n1_result["evaluations"],
        n1_voronoi_rounds=n1_result["iterations"],
        n1_cell_proposals=sum(event["source"] == "cell" for event in n1_result["events"]),
        n1_continuous_proposals=sum(
            event["source"] == "continuous_surrogate" for event in n1_result["events"]
        ),
        n1_duplicate_rejections=sum(
            trace["exact_duplicate_rejected"] for trace in n1_result["traces"]
        ),
        n1_old_loss_gate=bool(n1_result["loss"] <= LOSS_GATE),
        n2_theta=n2_result["theta"].tolist(),
        n2_loss=n2_result["loss"],
        n2_evaluations=n2_result["evaluations"],
        n2_old_gate=n2_result["gate"],
        n2_message=n2_result["message"],
        n2_truth_distance=float(np.linalg.norm(n2_result["theta"] - TEACHER)),
        first_n2_newton=first_n2,
        n3_accepted_steps=n3_result["accepted_steps"],
        n3_evaluations=n3_result["objective_evaluations"],
        n3_stop_reason=n3_result["stop_reason"],
        final_theta=final.tolist(),
        final_loss=n3_result["final_loss"],
        final_entrywise_error=(final - TEACHER).tolist(),
        final_parameter_error=float(np.linalg.norm(final - TEACHER)),
        recovered=bool(n3_result["recovered"]),
        continuation_max_error=float(np.max(np.abs(rollout_error))),
        continuation_mse=float(np.mean(rollout_error * rollout_error)),
        continuation_finite=bool(np.isfinite(continuation).all()),
        correctability_states=len(correctability),
        correctability_recoveries=sum(row["n3_recovered"] for row in correctability),
        seconds=time.time() - started,
    )
    _atomic_text(result_path, json.dumps(result, indent=2))
    return result


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return dict(
        count=int(len(values)),
        minimum=float(np.min(values)),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.90, method="higher")),
        maximum=float(np.max(values)),
        total=float(np.sum(values)),
    )


def _write_csv(rows: list[dict]) -> None:
    flattened = []
    for row in rows:
        flat = dict(row)
        flat["case_definition"] = json.dumps(flat["case_definition"])
        flat["first_n2_newton"] = json.dumps(flat["first_n2_newton"])
        flattened.append(flat)
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (OUT / "per_trajectory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def summarize(rows: list[dict], validation: dict, before: dict, after: dict,
              elapsed: float, workers: int, two_parameter: list[dict]) -> dict:
    correctability = []
    for row in rows:
        path = CASES / f"trajectory_{row['trajectory_id']:02d}" / "correctability.json"
        correctability.extend(json.loads(path.read_text()))
    _atomic_text(OUT / "correctability_states.json", json.dumps(correctability, indent=2))
    _atomic_text(OUT / "two_parameter_reference.json", json.dumps(two_parameter, indent=2))
    return dict(
        status="COMPLETE" if len(rows) == len(FRESH_CASES) else "INCOMPLETE",
        parameterization=dict(
            order=("kick_1", "drift", "kick_2", "radial_exponent"),
            equations=(
                "v_half=v+kick_1*dt*a_p(r)",
                "r_next=r+drift*dt*v_half",
                "v_next=v_half+kick_2*dt*a_p(r_next)",
            ),
            teacher=TEACHER.tolist(),
            bounds=np.column_stack((LOWER, UPPER)).tolist(),
        ),
        teacher_validation=validation,
        trajectories=len(rows),
        n1_old_gate_successes=sum(row["n1_old_loss_gate"] for row in rows),
        n1_evaluations=_distribution([row["n1_evaluations"] for row in rows]),
        n1_voronoi_rounds=_distribution([row["n1_voronoi_rounds"] for row in rows]),
        n1_continuous_proposals=_distribution([
            row["n1_continuous_proposals"] for row in rows
        ]),
        n1_duplicate_rejections=sum(row["n1_duplicate_rejections"] for row in rows),
        n2_old_gate_successes=sum(row["n2_old_gate"] for row in rows),
        n2_evaluations=_distribution([row["n2_evaluations"] for row in rows]),
        n3_recoveries=sum(row["recovered"] for row in rows),
        n3_accepted_steps=_distribution([row["n3_accepted_steps"] for row in rows]),
        final_parameter_error=_distribution([row["final_parameter_error"] for row in rows]),
        continuation_max_error=_distribution([row["continuation_max_error"] for row in rows]),
        continuation_mse=_distribution([row["continuation_mse"] for row in rows]),
        four_parameter_correctability=dict(
            states=len(correctability),
            recovered=sum(row["n3_recovered"] for row in correctability),
            fraction=float(np.mean([row["n3_recovered"] for row in correctability])),
            truth_distance=_distribution([row["truth_distance"] for row in correctability]),
            alignment=_distribution([
                row["newton_truth_alignment"] for row in correctability
                if row["newton_truth_alignment"] is not None
            ]),
            accepted_steps=_distribution([row["n3_accepted_steps"] for row in correctability]),
        ),
        two_parameter_reference=dict(
            states=len(two_parameter),
            recovered=sum(row["n3_recovered"] for row in two_parameter),
            fraction=float(np.mean([row["n3_recovered"] for row in two_parameter])),
            truth_distance=_distribution([row["truth_distance"] for row in two_parameter]),
            alignment=_distribution([
                row["newton_truth_alignment"] for row in two_parameter
                if row["newton_truth_alignment"] is not None
            ]),
            accepted_steps=_distribution([row["n3_accepted_steps"] for row in two_parameter]),
        ),
        checkpoint_hashes_unchanged=before == after,
        checkpoint_manifest=after,
        workers=workers,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case-index", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    values = trajectories()
    if args.validate_only:
        print(json.dumps(validate_teacher(values), indent=2))
        return
    if args.case_index is not None:
        result = run_case(args.case_index, values[args.case_index])
        print(json.dumps(dict(
            case=args.case_index,
            n1_loss=result["n1_loss"],
            n2_loss=result["n2_loss"],
            final_loss=result["final_loss"],
            recovered=result["recovered"],
        )), flush=True)
        return

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    validation = validate_teacher(values)
    _atomic_text(OUT / "teacher_validation.json", json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise RuntimeError("four-parameter Kepler teacher containment failed")
    paths = checkpoint_paths()
    before = manifest(paths)
    _atomic_text(OUT / "checkpoint_hashes_before.json", json.dumps(before, indent=2))
    workers = max(1, min(args.workers, 6, len(values)))
    rows = []
    for offset in range(0, len(values), workers):
        indices = list(range(offset, min(offset + workers, len(values))))
        children = []
        for case_id in indices:
            command = [
                sys.executable, "-m", "experiments.kepler.run_four_parameter_peelback",
                "--case-index", str(case_id),
            ]
            children.append((case_id, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(case_id, child.wait()) for case_id, child in children]
        failures = [(case_id, code) for case_id, code in failures if code]
        if failures:
            raise RuntimeError(f"case subprocess failures: {failures}")
        batch = [
            json.loads((CASES / f"trajectory_{case_id:02d}" / "result.json").read_text())
            for case_id in indices
        ]
        rows.extend(batch)
        rows.sort(key=lambda row: row["trajectory_id"])
        _write_csv(rows)
        _atomic_text(OUT / "results_partial.json", json.dumps(rows, indent=2))
        for row in batch:
            print(
                f"case {row['trajectory_id']}: N1={row['n1_loss']:.3e} "
                f"N2={row['n2_loss']:.3e} N3={row['final_loss']:.3e} "
                f"recovered={row['recovered']}", flush=True,
            )
    two_parameter = _two_parameter_reference(values)
    after = manifest(paths)
    _atomic_text(OUT / "checkpoint_hashes_after.json", json.dumps(after, indent=2))
    report = summarize(
        rows, validation, before, after, time.time() - started, workers, two_parameter
    )
    _atomic_text(OUT / "report.json", json.dumps(report, indent=2))
    print(json.dumps(dict(
        status=report["status"],
        trajectories=report["trajectories"],
        recoveries=report["n3_recoveries"],
        correctability=report["four_parameter_correctability"],
        checkpoints_unchanged=report["checkpoint_hashes_unchanged"],
    ), indent=2))


if __name__ == "__main__":
    main()
