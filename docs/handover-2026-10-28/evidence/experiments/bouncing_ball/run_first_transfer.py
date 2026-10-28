#!/usr/bin/env python3
"""First frozen N1 -> N2 -> N3 transfer to the elastic bouncing ball.

The search hypothesis is K_ball(a,b).  ``a`` multiplies the sampled velocity
used for free-flight transport, while ``b`` multiplies the post-reflection
sampled velocity.  Collision detection, exact impact-time propagation, wall
snapping, simultaneous corner reflection, and the ten-event safety bound are
the unchanged deterministic teacher scaffold.

This adapter intentionally uses one-step prefix transition reconstruction.  It
does not import or evaluate the older SHO J_ALL multihorizon loss.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import torch

from experiments.sho import run_stage1b_fifo_vs_ranker as n1_current
from experiments.sho import run_stage1b_learned_global_search_imitation as n1_ranker
from experiments.sho import run_stage1b_learned_powell_2d as n2_scalar
from experiments.sho import run_stage1b_n3_gradient_polishing as n3_current
from experiments.sho import train_n2_direction_controller as n2_direction


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / os.environ.get(
    "BALL_TRANSFER_OUTPUT", "outputs/bouncing_ball/first_n1_n2_n3_transfer"
)
TEACHER_PATH = Path(
    "/Users/alexanderrichardson/Documents/conserved-quantities/"
    "conserved/data/bouncing_ball.py"
)

DT = 0.05
XMIN = YMIN = -5.0
XMAX = YMAX = 5.0
EPS = 1.0e-12
MAX_COLLISIONS = 10
PREFIX_STATES = 128
TOTAL_STEPS = 384
TRANSFER_SEED = int(os.environ.get("BALL_TRANSFER_SEED", "20260930"))
TRAJECTORY_COUNT = int(os.environ.get("BALL_TRANSFER_TRAJECTORIES", "8"))
DOMAIN_LOWER = float(os.environ.get("BALL_TRANSFER_DOMAIN_LOWER", "0.95"))
DOMAIN_UPPER = float(os.environ.get("BALL_TRANSFER_DOMAIN_UPPER", "1.05"))
LOWER = np.asarray((DOMAIN_LOWER, DOMAIN_LOWER), dtype=np.float64)
UPPER = np.asarray((DOMAIN_UPPER, DOMAIN_UPPER), dtype=np.float64)
WIDTH = UPPER - LOWER
TEACHER_THETA = np.asarray((1.0, 1.0), dtype=np.float64)
N1_GATE = n1_current.GATE
N1_CAP = int(os.environ.get("BALL_TRANSFER_N1_CAP", str(n1_ranker.NORMAL_CAP)))
N2_CAP = int(os.environ.get("BALL_TRANSFER_N2_CAP", "1500"))
N2_GATE = n2_scalar.SUCCESS_LOSS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_numpy_teacher():
    spec = importlib.util.spec_from_file_location("authoritative_bouncing_ball", TEACHER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load bouncing-ball teacher at {TEACHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TEACHER = load_numpy_teacher()


def numpy_k_step(state: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """K_ball step with production collision semantics in NumPy float64."""
    a, b = np.asarray(theta, dtype=np.float64)
    x, y, vx, vy = (np.float64(value) for value in state)
    ux = np.float64(a * vx)
    uy = np.float64(a * vy)
    t_remain = np.float64(DT)
    for _ in range(MAX_COLLISIONS):
        if t_remain <= EPS:
            break
        tx = np.float64(np.inf)
        ty = np.float64(np.inf)
        if ux > EPS:
            tx = np.float64((XMAX - x) / ux)
        elif ux < -EPS:
            tx = np.float64((XMIN - x) / ux)
        if uy > EPS:
            ty = np.float64((YMAX - y) / uy)
        elif uy < -EPS:
            ty = np.float64((YMIN - y) / uy)
        t_hit = min(tx, ty)
        if t_hit > t_remain or np.isinf(t_hit):
            x = np.float64(x + ux * t_remain)
            y = np.float64(y + uy * t_remain)
            t_remain = np.float64(0.0)
            break
        x = np.float64(x + ux * t_hit)
        y = np.float64(y + uy * t_hit)
        t_remain = np.float64(t_remain - t_hit)
        hit_x = abs(tx - t_hit) < EPS
        hit_y = abs(ty - t_hit) < EPS
        if hit_x:
            x = np.float64(XMAX if ux > 0.0 else XMIN)
            ux = np.float64(-ux)
            vx = np.float64(-vx)
        if hit_y:
            y = np.float64(YMAX if uy > 0.0 else YMIN)
            uy = np.float64(-uy)
            vy = np.float64(-vy)
    return np.asarray((x, y, b * vx, b * vy), dtype=np.float64)


def numpy_k_event_signature(state: np.ndarray, theta: np.ndarray) -> tuple[int, ...]:
    """Return hit-axis masks without changing the production step semantics."""
    a, _ = np.asarray(theta, dtype=np.float64)
    x, y, vx, vy = (np.float64(value) for value in state)
    ux = np.float64(a * vx)
    uy = np.float64(a * vy)
    t_remain = np.float64(DT)
    signature: list[int] = []
    for _ in range(MAX_COLLISIONS):
        if t_remain <= EPS:
            break
        tx = np.float64(np.inf)
        ty = np.float64(np.inf)
        if ux > EPS:
            tx = np.float64((XMAX-x)/ux)
        elif ux < -EPS:
            tx = np.float64((XMIN-x)/ux)
        if uy > EPS:
            ty = np.float64((YMAX-y)/uy)
        elif uy < -EPS:
            ty = np.float64((YMIN-y)/uy)
        t_hit = min(tx, ty)
        if t_hit > t_remain or np.isinf(t_hit):
            break
        x = np.float64(x + ux*t_hit)
        y = np.float64(y + uy*t_hit)
        t_remain = np.float64(t_remain-t_hit)
        hit_x = abs(tx-t_hit) < EPS
        hit_y = abs(ty-t_hit) < EPS
        signature.append(int(hit_x) + 2*int(hit_y))
        if hit_x:
            x = np.float64(XMAX if ux > 0 else XMIN)
            ux = np.float64(-ux)
        if hit_y:
            y = np.float64(YMAX if uy > 0 else YMIN)
            uy = np.float64(-uy)
    return tuple(signature)


def topology_summary(thetas: list[np.ndarray], trajectory: np.ndarray) -> dict[str, int]:
    teacher = [numpy_k_event_signature(trajectory[index], TEACHER_THETA)
               for index in range(PREFIX_STATES-1)]
    points_changed = 0
    changed_transitions = 0
    maximum = 0
    for theta in thetas:
        changed = sum(
            numpy_k_event_signature(trajectory[index], theta) != teacher[index]
            for index in range(PREFIX_STATES-1)
        )
        points_changed += int(changed > 0)
        changed_transitions += changed
        maximum = max(maximum, changed)
    return dict(
        topology_points_evaluated=len(thetas),
        topology_points_with_changes=points_changed,
        topology_changed_transitions=changed_transitions,
        topology_max_changed_transitions_per_point=maximum,
    )


def torch_k_step(state: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Differentiable-on-the-active-event-branch float64 K_ball step."""
    a, b = theta.unbind()
    x, y, vx, vy = state.unbind()
    ux = a * vx
    uy = a * vy
    t_remain: float | torch.Tensor = DT
    for _ in range(MAX_COLLISIONS):
        if float(t_remain) <= EPS:
            break
        tx: float | torch.Tensor = math.inf
        ty: float | torch.Tensor = math.inf
        if float(ux.detach()) > EPS:
            tx = (XMAX - x) / ux
        elif float(ux.detach()) < -EPS:
            tx = (XMIN - x) / ux
        if float(uy.detach()) > EPS:
            ty = (YMAX - y) / uy
        elif float(uy.detach()) < -EPS:
            ty = (YMIN - y) / uy
        tx_value = float(tx)
        ty_value = float(ty)
        t_hit: float | torch.Tensor = tx if tx_value <= ty_value else ty
        if float(t_hit) > float(t_remain) or math.isinf(float(t_hit)):
            x = x + ux * t_remain
            y = y + uy * t_remain
            t_remain = 0.0
            break
        x = x + ux * t_hit
        y = y + uy * t_hit
        t_remain = t_remain - t_hit
        hit_x = abs(tx_value - float(t_hit)) < EPS
        hit_y = abs(ty_value - float(t_hit)) < EPS
        if hit_x:
            x = torch.as_tensor(XMAX if float(ux.detach()) > 0.0 else XMIN)
            ux = -ux
            vx = -vx
        if hit_y:
            y = torch.as_tensor(YMAX if float(uy.detach()) > 0.0 else YMIN)
            uy = -uy
            vy = -vy
    return torch.stack((x, y, b * vx, b * vy))


def torch_rollout(initial: torch.Tensor, theta: torch.Tensor, steps: int) -> torch.Tensor:
    states = [initial]
    state = initial
    for _ in range(steps):
        state = torch_k_step(state, theta)
        states.append(state)
    return torch.stack(states)


def numpy_rollout(initial: np.ndarray, theta: np.ndarray, steps: int) -> np.ndarray:
    result = np.empty((steps + 1, 4), dtype=np.float64)
    result[0] = initial
    for index in range(steps):
        result[index + 1] = numpy_k_step(result[index], theta)
    return result


def prefix_loss_numpy(theta: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """One-step state-transition reconstruction over the observed prefix."""
    values = np.atleast_2d(theta).astype(np.float64, copy=False)
    prefix = np.asarray(trajectory[:PREFIX_STATES], dtype=np.float64)
    output = np.empty(len(values), dtype=np.float64)
    for candidate_index, candidate in enumerate(values):
        squared = np.float64(0.0)
        for index in range(PREFIX_STATES - 1):
            error = numpy_k_step(prefix[index], candidate) - prefix[index + 1]
            squared = np.float64(squared + np.dot(error, error))
        output[candidate_index] = squared / np.float64((PREFIX_STATES - 1) * 4)
    return output


def prefix_loss_torch(theta: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
    prefix = trajectory[:PREFIX_STATES]
    errors = [torch_k_step(prefix[index], theta) - prefix[index + 1]
              for index in range(PREFIX_STATES - 1)]
    return torch.stack(errors).square().mean()


def validate_teacher() -> dict[str, object]:
    """Validate Torch event semantics against archived and fresh NumPy states."""
    pools = []
    for label, seed, count in (("archived_seed42", 42, 4),
                               ("fresh", TRANSFER_SEED, TRAJECTORY_COUNT)):
        trajectories, _, _, _ = TEACHER.build_bouncing_ball_trajectories(
            n_trajectories=count, n_timesteps=TOTAL_STEPS, dt=DT, seed=seed
        )
        pools.append((label, trajectories))
    controls = np.asarray(
        (
            (4.9, 0.25, 3.0, -0.5),
            (4.9, 4.9, 2.0, 2.0),
            (5.0, -5.0, 1.0, -1.0),
            (0.0, 0.0, 250.0, -175.0),
            (-4.999999999999, 2.0, -2.5, 1.25),
        ),
        dtype=np.float64,
    )
    control_truth = []
    control_torch = []
    for state in controls:
        control_truth.append(TEACHER.step_bouncing_ball(*state, DT))
        control_torch.append(torch_k_step(torch.from_numpy(state), torch.ones(2)).detach().numpy())
    comparisons = []
    all_exact = 0
    all_values = 0
    global_max = 0.0
    for label, truth in pools:
        predicted = []
        for trajectory in truth:
            predicted.append(torch_rollout(torch.from_numpy(trajectory[0]), torch.ones(2),
                                           TOTAL_STEPS).detach().numpy())
        predicted_array = np.asarray(predicted)
        difference = np.abs(predicted_array - truth)
        exact = int(np.count_nonzero(predicted_array == truth))
        values = int(truth.size)
        maximum = float(np.max(difference))
        comparisons.append(dict(pool=label, trajectories=len(truth), values=values,
                                exact_values=exact, max_abs_error=maximum,
                                allclose_machine_precision=bool(np.allclose(
                                    predicted_array, truth, rtol=8*np.finfo(np.float64).eps,
                                    atol=8*np.finfo(np.float64).eps))))
        all_exact += exact
        all_values += values
        global_max = max(global_max, maximum)
    control_truth_array = np.asarray(control_truth)
    control_torch_array = np.asarray(control_torch)
    control_error = np.abs(control_truth_array - control_torch_array)
    # Verify the NumPy and Torch objective adapters share the same surface.
    theta_controls = np.asarray(((1.0, 1.0), (0.95, 1.05), (1.05, 0.95),
                                 (0.999, 1.001)), dtype=np.float64)
    objective_error = 0.0
    for _, trajectories in pools:
        for trajectory in trajectories:
            native = prefix_loss_numpy(theta_controls, trajectory)
            differentiable = np.asarray([
                float(prefix_loss_torch(torch.from_numpy(theta), torch.from_numpy(trajectory)))
                for theta in theta_controls
            ])
            objective_error = max(objective_error, float(np.max(np.abs(native-differentiable))))
    passed = bool(
        all(item["allclose_machine_precision"] for item in comparisons)
        and bool(np.allclose(control_torch_array, control_truth_array,
                             rtol=8*np.finfo(np.float64).eps,
                             atol=8*np.finfo(np.float64).eps))
        and objective_error <= 64*np.finfo(np.float64).eps
    )
    return dict(
        passed=passed,
        authoritative_teacher=str(TEACHER_PATH),
        teacher_sha256=sha256(TEACHER_PATH),
        pools=comparisons,
        exact_values=all_exact,
        total_values=all_values,
        max_abs_trajectory_error=global_max,
        collision_control_values=int(control_truth_array.size),
        collision_control_exact=int(np.count_nonzero(control_truth_array == control_torch_array)),
        collision_control_max_abs_error=float(np.max(control_error)),
        numpy_torch_objective_max_abs_error=objective_error,
    )


def normalized_to_physical(points: np.ndarray) -> np.ndarray:
    return LOWER + np.asarray(points, dtype=np.float64) * WIDTH


@dataclass
class N1Result:
    success: bool
    theta: np.ndarray
    loss: float
    evaluations: int
    iterations: int
    learned_ranker_decisions: int
    initial_theta: np.ndarray
    initial_loss: float
    initial_crossed_gate: bool
    cell_selection_calls: int
    cell_proposals_considered: int
    voronoi_probe_count: int
    potentially_optimal_cells: int
    evaluated_thetas: list[np.ndarray]
    first_failure: str = ""


def run_n1(trajectory_id: int, trajectory: np.ndarray,
           frozen: n1_current.FrozenProposals) -> N1Result:
    sites = n1_current.edirect._initial_design().copy()
    losses = prefix_loss_numpy(normalized_to_physical(sites), trajectory)
    initial_best = int(np.argmin(losses))
    initial_theta = normalized_to_physical(sites[initial_best])
    initial_loss = float(losses[initial_best])
    evaluated_thetas = [point.copy() for point in normalized_to_physical(sites)]
    iteration = 0
    learned_ranker_decisions = 0
    cell_selection_calls = 0
    cell_proposals_considered = 0
    voronoi_probe_count = 0
    potentially_optimal_cells = 0
    case = dict(generator_id=800_000 + trajectory_id, trajectory_id=trajectory_id)
    # The learned N1 model was trained on normalized coordinates.  Its domain
    # feature therefore retains the canonical normalized SHO interface.
    feature_bounds = n1_current.ordinary.FULL_BOUNDS
    while len(sites) < N1_CAP and float(np.min(losses)) > N1_GATE:
        pending, meta = frozen(sites, losses, case, iteration)
        cell_selection_calls += 1
        cell_proposals_considered += len(pending)
        voronoi_probe_count += int(meta["probe_count"])
        potentially_optimal_cells += int(meta["PO_cells"])
        while len(pending) and len(sites) < N1_CAP and float(np.min(losses)) > N1_GATE:
            global_features = n1_ranker._global_features(
                sites, losses, len(pending), len(sites), N1_CAP, iteration,
                feature_bounds, "normal_edirect"
            )
            choice, _ = n1_ranker._model_choice(
                frozen.ranker, sites, losses, pending, "cell", global_features
            )
            learned_ranker_decisions += 1
            point = pending[choice].copy()
            loss = float(prefix_loss_numpy(normalized_to_physical(point[None, :]), trajectory)[0])
            evaluated_thetas.append(normalized_to_physical(point))
            sites = np.vstack((sites, point))
            losses = np.append(losses, loss)
            pending = np.delete(pending, choice, axis=0)
        if len(sites) < N1_CAP and float(np.min(losses)) > N1_GATE:
            greedy = n1_current.edirect._greedy_proposal(
                sites, losses, case["generator_id"], iteration
            )
            if greedy is not None:
                loss = float(prefix_loss_numpy(
                    normalized_to_physical(greedy[None, :]), trajectory
                )[0])
                evaluated_thetas.append(normalized_to_physical(greedy))
                sites = np.vstack((sites, greedy))
                losses = np.append(losses, loss)
        iteration += 1
    best = int(np.argmin(losses))
    return N1Result(
        success=float(losses[best]) <= N1_GATE,
        theta=normalized_to_physical(sites[best]),
        loss=float(losses[best]),
        evaluations=len(losses),
        iterations=iteration,
        learned_ranker_decisions=learned_ranker_decisions,
        initial_theta=initial_theta,
        initial_loss=initial_loss,
        initial_crossed_gate=initial_loss <= N1_GATE,
        cell_selection_calls=cell_selection_calls,
        cell_proposals_considered=cell_proposals_considered,
        voronoi_probe_count=voronoi_probe_count,
        potentially_optimal_cells=potentially_optimal_cells,
        evaluated_thetas=evaluated_thetas,
        first_failure="" if float(losses[best]) <= N1_GATE else "N1_HANDOFF_GATE",
    )


@dataclass
class TransferEvaluationState:
    trajectory: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    best_theta: np.ndarray
    best_loss: float
    max_evaluations: int
    count: int = 0
    evaluated_thetas: list[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.evaluated_thetas is None:
            self.evaluated_thetas = []

    def evaluate(self, theta: np.ndarray) -> float:
        if self.count >= self.max_evaluations:
            raise RuntimeError("N2 objective evaluation cap exceeded")
        if not bool(np.all(theta >= self.lower - 1e-13) and
                    np.all(theta <= self.upper + 1e-13)):
            raise RuntimeError("Frozen N2 line controller proposed outside transfer bounds")
        theta = np.minimum(np.maximum(theta, self.lower), self.upper)
        loss = float(prefix_loss_numpy(theta[None, :], self.trajectory)[0])
        self.evaluated_thetas.append(theta.copy())
        self.count += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_theta = theta.copy()
        return loss


@dataclass
class N2Result:
    success: bool
    theta: np.ndarray
    loss: float
    evaluations: int
    cycles: int
    line_searches: int
    direction_updates: int
    stop_reason: str
    evaluated_thetas: list[np.ndarray]


def run_n2(trajectory_id: int, trajectory: np.ndarray, n1: N1Result) -> N2Result:
    spacing = WIDTH / math.sqrt(max(1, n1.evaluations))
    lower = np.maximum(LOWER, n1.theta - spacing)
    upper = np.minimum(UPPER, n1.theta + spacing)
    if not bool(np.all(TEACHER_THETA >= lower) and np.all(TEACHER_THETA <= upper)):
        return N2Result(False, n1.theta.copy(), n1.loss, 0, 0, 0, 0,
                        "N1_LOCAL_BOX_EXCLUDES_TEACHER", [])
    state = TransferEvaluationState(trajectory, lower, upper, n1.theta.copy(),
                                    n1.loss, N2_CAP)
    controller = n2_direction.load_controller()
    theta = n1.theta.copy()
    theta_loss = n1.loss
    directions = controller.initial_directions(2)
    cycles = line_searches = direction_updates = line_serial = 0
    stagnant_cycles = 0
    stop_reason = "evaluation_cap"
    while state.count < state.max_evaluations and state.best_loss >= N2_GATE:
        cycles += 1
        theta_start = theta.copy()
        loss_start = theta_loss
        decreases: list[float] = []
        for direction in directions:
            if state.count >= state.max_evaluations or state.best_loss < N2_GATE:
                break
            before = theta_loss
            line_serial += 1
            result = n2_scalar._line_search(
                state, theta, theta_loss, direction,
                n2_scalar.LINE_SEED + 100_003 * (900_000 + trajectory_id) + line_serial,
            )
            line_searches += 1
            theta, theta_loss = result.theta, result.loss
            decreases.append(max(0.0, before - theta_loss))
        if state.best_loss < N2_GATE:
            stop_reason = "success_gate"
            break
        if state.count >= state.max_evaluations or len(decreases) < len(directions):
            break
        theta_end = theta.copy()
        loss_end = theta_loss
        displacement = theta_end - theta_start
        displacement_norm = float(np.linalg.norm(displacement))
        biggest = controller.largest_decrease_index(decreases)
        biggest_decrease = decreases[biggest]
        significant = displacement_norm > n2_scalar.DISPLACEMENT_TOLERANCE * max(
            1.0, float(np.linalg.norm(upper-lower)))
        extrapolated = theta_end + displacement
        inside = bool(np.all(extrapolated >= lower) and np.all(extrapolated <= upper))
        if controller.conjunction(significant, inside):
            extrapolated_loss = state.evaluate(extrapolated)
            if state.best_loss < N2_GATE:
                theta, theta_loss = extrapolated, extrapolated_loss
                stop_reason = "success_gate"
                break
            better = extrapolated_loss < loss_start
            first = loss_start - loss_end - biggest_decrease
            second = loss_start - extrapolated_loss
            test = (2.0 * (loss_start - 2.0*loss_end + extrapolated_loss)
                    * first * first - biggest_decrease * second * second)
            if controller.conjunction(better, controller.should_replace(test)) and state.count < state.max_evaluations:
                line_serial += 1
                result = n2_scalar._line_search(
                    state, theta, theta_loss, displacement,
                    n2_scalar.LINE_SEED + 100_003 * (900_000 + trajectory_id) + line_serial,
                )
                line_searches += 1
                theta, theta_loss = result.theta, result.loss
                directions = controller.update(directions, biggest, displacement)
                direction_updates += 1
        cycle_relative = (loss_start-theta_loss)/max(abs(loss_start), 1e-300)
        if (cycle_relative <= 1e-10 and displacement_norm <= 1e-10 *
                max(1.0, float(np.linalg.norm(upper-lower)))):
            stagnant_cycles += 1
        else:
            stagnant_cycles = 0
        if stagnant_cycles >= 2:
            stop_reason = "direction_update_stagnation_observed_but_continued"
    return N2Result(state.best_loss < N2_GATE, state.best_theta, state.best_loss,
                    state.count, cycles, line_searches, direction_updates,
                    stop_reason if state.best_loss >= N2_GATE else "success_gate",
                    state.evaluated_thetas)


@dataclass
class N3Result:
    theta: np.ndarray
    loss: float
    accepted_steps: int
    objective_evaluations: int
    stop_reason: str
    evaluated_thetas: list[np.ndarray]


def run_n3(trajectory: np.ndarray, start: N2Result) -> N3Result:
    observed = torch.from_numpy(trajectory)
    theta = torch.from_numpy(start.theta.copy())
    with torch.no_grad():
        current_loss = float(prefix_loss_torch(theta, observed))
    accepted_steps = objective_evaluations = 0
    evaluated_thetas = [theta.detach().numpy().copy()]
    stop_reason = "max_accepted_steps"
    while accepted_steps < n3_current.MAX_ACCEPTED_STEPS:
        differentiable = theta.detach().clone().requires_grad_(True)
        loss = prefix_loss_torch(differentiable, observed)
        gradient = torch.autograd.grad(loss, differentiable, create_graph=True)[0]
        hessian = torch.stack([
            torch.autograd.grad(gradient[index], differentiable, retain_graph=True)[0]
            for index in range(2)
        ])
        hessian = 0.5 * (hessian + hessian.T)
        gradient = gradient.detach()
        hessian = hessian.detach()
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        if not math.isfinite(gradient_norm):
            stop_reason = "nonfinite_gradient"
            break
        if gradient_norm == 0.0:
            stop_reason = "floating_point_zero_gradient"
            break
        direction, _, _, direction_status = n3_current._newton_direction(gradient, hessian)
        if direction is None:
            stop_reason = direction_status
            break
        directional_derivative = float(torch.dot(gradient, direction))
        accepted = False
        unchanged = False
        for backtrack in range(n3_current.MAX_BACKTRACKS + 1):
            candidate = theta + (2.0 ** (-backtrack)) * direction
            if bool(torch.equal(candidate, theta)):
                unchanged = True
                break
            if not bool(torch.all(candidate >= torch.from_numpy(LOWER)) and
                        torch.all(candidate <= torch.from_numpy(UPPER))):
                continue
            with torch.no_grad():
                candidate_loss = float(prefix_loss_torch(candidate, observed))
            objective_evaluations += 1
            evaluated_thetas.append(candidate.detach().numpy().copy())
            if candidate_loss < current_loss:
                theta = candidate.detach()
                current_loss = candidate_loss
                accepted_steps += 1
                accepted = True
                break
        if not accepted:
            predicted = max(0.0, -directional_derivative)
            spacing = abs(float(np.spacing(np.float64(current_loss))))
            stop_reason = ("floating_point_no_further_improvement"
                           if unchanged or predicted <= 32.0*spacing
                           else "backtracking_no_improvement")
            break
    return N3Result(theta.detach().numpy(), current_loss, accepted_steps,
                    objective_evaluations, stop_reason, evaluated_thetas)


def collision_count(trajectory: np.ndarray, end: int) -> int:
    velocity = trajectory[:end, 2:]
    return int(np.count_nonzero(np.signbit(velocity[1:]) != np.signbit(velocity[:-1])))


def domain_safety_check(trajectories: np.ndarray) -> dict[str, object]:
    """Quick finite/box check at corners and center; no acceptance metric added."""
    controls = np.asarray((
        (LOWER[0], LOWER[1]), (LOWER[0], UPPER[1]),
        (UPPER[0], LOWER[1]), (UPPER[0], UPPER[1]),
        tuple((LOWER+UPPER)/2.0),
    ), dtype=np.float64)
    finite_one_step = True
    finite_rollout = True
    positions_in_box = True
    maximum_abs_value = 0.0
    for trajectory in trajectories:
        for theta in controls:
            one_step = np.asarray([
                numpy_k_step(trajectory[index], theta)
                for index in range(PREFIX_STATES-1)
            ])
            finite_one_step &= bool(np.isfinite(one_step).all())
            candidate = numpy_rollout(trajectory[0], theta, TOTAL_STEPS)
            finite_rollout &= bool(np.isfinite(candidate).all())
            positions_in_box &= bool(np.all(candidate[:, :2] >= XMIN-EPS) and
                                     np.all(candidate[:, :2] <= XMAX+EPS))
            maximum_abs_value = max(maximum_abs_value, float(np.max(np.abs(candidate))))
    return dict(
        passed=bool(finite_one_step and finite_rollout and positions_in_box),
        theta_controls=len(controls),
        trajectory_controls=len(trajectories),
        finite_one_step=bool(finite_one_step),
        finite_recursive_rollout=bool(finite_rollout),
        positions_remain_in_box=bool(positions_in_box),
        maximum_abs_candidate_state=maximum_abs_value,
    )


def checkpoint_manifest() -> dict[str, str]:
    paths = {
        "n1_candidate_ranker": n1_ranker.CHECKPOINT_PATH,
        "n1_ordinary_score": n1_current.ordinary.CHECKPOINT_PATH,
        "n2_scalar_line_controller": n2_scalar.LINE_CHECKPOINT,
        "n2_direction_routers": n2_direction.LOCAL_CHECKPOINT,
    }
    return {name: sha256(Path(path)) for name, path in paths.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = checkpoint_manifest()
    validation = validate_teacher()
    (OUTPUT / "teacher_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    if not validation["passed"]:
        report = dict(status="STOPPED_TORCH_TEACHER_MISMATCH", teacher_validation=validation,
                      elapsed_seconds=time.time()-started)
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return

    trajectories, energies, speeds, _ = TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=TRAJECTORY_COUNT, n_timesteps=TOTAL_STEPS, dt=DT,
        seed=TRANSFER_SEED
    )
    safety = domain_safety_check(trajectories)
    (OUTPUT/"domain_safety.json").write_text(json.dumps(safety, indent=2)+"\n")
    if not safety["passed"]:
        report = dict(status="STOPPED_DOMAIN_SAFETY_FAILURE", domain_safety=safety,
                      teacher_validation=validation,
                      checkpoints_unchanged=before == checkpoint_manifest(),
                      elapsed_seconds=time.time()-started)
        (OUTPUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2)); return
    frozen = n1_current.FrozenProposals()
    rows: list[dict[str, object]] = []
    n1_results: list[N1Result] = []
    for trajectory_id, trajectory in enumerate(trajectories):
        result = run_n1(trajectory_id, trajectory, frozen)
        n1_results.append(result)
        print(f"N1 {trajectory_id+1}/{TRAJECTORY_COUNT}: eval={result.evaluations} "
              f"loss={result.loss:.3e} theta={result.theta}", flush=True)
    n1_failures = [index for index, result in enumerate(n1_results) if not result.success]
    if n1_failures:
        for trajectory_id, result in enumerate(n1_results):
            rows.append(dict(trajectory_id=trajectory_id, speed=float(speeds[trajectory_id]),
                             energy=float(energies[trajectory_id]),
                             prefix_component_impacts=collision_count(trajectories[trajectory_id], PREFIX_STATES),
                             n1_success=int(result.success), n1_evaluations=result.evaluations,
                             n1_learned_ranker_decisions=result.learned_ranker_decisions,
                             n1_initial_a=result.initial_theta[0], n1_initial_b=result.initial_theta[1],
                             n1_initial_loss=result.initial_loss,
                             n1_initial_crossed_gate=int(result.initial_crossed_gate),
                             n1_a=result.theta[0], n1_b=result.theta[1], n1_loss=result.loss,
                             first_failure="N1" if not result.success else ""))
        write_csv(OUTPUT/"per_trajectory.csv", rows)
        report = dict(status="STOPPED_FIRST_TRANSFER_FAILURE", first_failing_stage="N1",
                      failing_trajectories=n1_failures, teacher_validation=validation,
                      checkpoints_unchanged=before == checkpoint_manifest(),
                      elapsed_seconds=time.time()-started)
        (OUTPUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2)); return

    n2_results: list[N2Result] = []
    for trajectory_id, (trajectory, n1) in enumerate(zip(trajectories, n1_results)):
        result = run_n2(trajectory_id, trajectory, n1)
        n2_results.append(result)
        print(f"N2 {trajectory_id+1}/{TRAJECTORY_COUNT}: eval={result.evaluations} "
              f"loss={result.loss:.3e} theta={result.theta} stop={result.stop_reason}", flush=True)
    n2_failures = [index for index, result in enumerate(n2_results) if not result.success]
    if n2_failures:
        first = "N2"
        n3_results: list[N3Result | None] = [None] * TRAJECTORY_COUNT
    else:
        first = ""
        n3_results = []
        for trajectory_id, (trajectory, n2) in enumerate(zip(trajectories, n2_results)):
            result = run_n3(trajectory, n2)
            n3_results.append(result)
            print(f"N3 {trajectory_id+1}/{TRAJECTORY_COUNT}: steps={result.accepted_steps} "
                  f"loss={result.loss:.3e} theta={result.theta} stop={result.stop_reason}", flush=True)

    for trajectory_id, trajectory in enumerate(trajectories):
        n1 = n1_results[trajectory_id]
        n2 = n2_results[trajectory_id]
        n3 = n3_results[trajectory_id]
        theta = n2.theta if n3 is None else n3.theta
        final_loss = n2.loss if n3 is None else n3.loss
        continuation_steps = TOTAL_STEPS - (PREFIX_STATES - 1)
        predicted = numpy_rollout(trajectory[PREFIX_STATES-1], theta, continuation_steps)
        truth = trajectory[PREFIX_STATES-1:]
        error = predicted - truth
        parameter_error = theta - TEACHER_THETA
        n1_topology = topology_summary(n1.evaluated_thetas, trajectory)
        n2_topology = topology_summary(n2.evaluated_thetas, trajectory)
        n3_topology = topology_summary([] if n3 is None else n3.evaluated_thetas, trajectory)
        rows.append(dict(
            trajectory_id=trajectory_id,
            initial_x=float(trajectory[0,0]), initial_y=float(trajectory[0,1]),
            initial_vx=float(trajectory[0,2]), initial_vy=float(trajectory[0,3]),
            speed=float(speeds[trajectory_id]), energy=float(energies[trajectory_id]),
            prefix_component_impacts=collision_count(trajectory, PREFIX_STATES),
            continuation_component_impacts=collision_count(trajectory[PREFIX_STATES-1:], continuation_steps+1),
            n1_success=int(n1.success), n1_evaluations=n1.evaluations,
            n1_learned_ranker_decisions=n1.learned_ranker_decisions,
            n1_initial_a=float(n1.initial_theta[0]), n1_initial_b=float(n1.initial_theta[1]),
            n1_initial_loss=n1.initial_loss,
            n1_initial_crossed_gate=int(n1.initial_crossed_gate),
            n1_cell_selection_calls=n1.cell_selection_calls,
            n1_cell_proposals_considered=n1.cell_proposals_considered,
            n1_voronoi_probe_count=n1.voronoi_probe_count,
            n1_potentially_optimal_cells=n1.potentially_optimal_cells,
            n1_a=float(n1.theta[0]), n1_b=float(n1.theta[1]), n1_loss=n1.loss,
            n2_success=int(n2.success), n2_evaluations=n2.evaluations, n2_cycles=n2.cycles,
            n2_line_searches=n2.line_searches, n2_direction_updates=n2.direction_updates,
            n2_stop_reason=n2.stop_reason, n2_a=float(n2.theta[0]), n2_b=float(n2.theta[1]),
            n2_loss=n2.loss,
            n3_ran=int(n3 is not None), n3_steps="" if n3 is None else n3.accepted_steps,
            n3_objective_evaluations="" if n3 is None else n3.objective_evaluations,
            n3_stop_reason="" if n3 is None else n3.stop_reason,
            final_a=float(theta[0]), final_b=float(theta[1]), final_loss=final_loss,
            parameter_l2_error=float(np.linalg.norm(parameter_error)),
            parameter_max_error=float(np.max(np.abs(parameter_error))),
            continuation_mse=float(np.mean(error*error)),
            continuation_max_abs=float(np.max(np.abs(error))),
            continuation_exact_values=int(np.count_nonzero(predicted == truth)),
            continuation_total_values=int(truth.size),
            n1_topology_points_with_changes=n1_topology["topology_points_with_changes"],
            n1_topology_changed_transitions=n1_topology["topology_changed_transitions"],
            n2_topology_points_with_changes=n2_topology["topology_points_with_changes"],
            n2_topology_changed_transitions=n2_topology["topology_changed_transitions"],
            n3_topology_points_with_changes=n3_topology["topology_points_with_changes"],
            n3_topology_changed_transitions=n3_topology["topology_changed_transitions"],
            first_failure="N2" if not n2.success else "",
        ))
    write_csv(OUTPUT/"per_trajectory.csv", rows)
    after = checkpoint_manifest()
    successful = not n2_failures
    report = dict(
        status="COMPLETED" if successful else "STOPPED_FIRST_TRANSFER_FAILURE",
        first_failing_stage=first,
        objective=dict(
            interface="continuous scalar float64 objective callable shared by N1/N2/N3",
            active_sho_sources=(
                "run_stage1b_edirect_vs_direct._prefix_losses_numpy",
                "run_stage1b_two_parameter_learned_search._prefix_losses",
            ),
            transfer_loss="mean squared one-step K_ball transition reconstruction over 127 observed prefix transitions",
            prefix_states=PREFIX_STATES,
            explicitly_excluded="SHO J_ALL multihorizon rollout loss",
        ),
        hypothesis=dict(parameters=("free_flight_transport_gain_a",
                                    "sampled_velocity_persistence_gain_b"),
                        teacher=(1.0,1.0),
                        bounds=((DOMAIN_LOWER,DOMAIN_UPPER),(DOMAIN_LOWER,DOMAIN_UPPER)),
                        collision_scaffold="authoritative exact event handling"),
        teacher_validation=validation,
        domain_safety=safety,
        trajectories=TRAJECTORY_COUNT,
        n1_successes=sum(result.success for result in n1_results),
        n1_initial_design_handoffs=sum(result.evaluations == 10 for result in n1_results),
        n1_learned_ranker_decisions=sum(result.learned_ranker_decisions for result in n1_results),
        n1_cell_selection_calls=sum(result.cell_selection_calls for result in n1_results),
        n1_voronoi_probe_count=sum(result.voronoi_probe_count for result in n1_results),
        n2_successes=sum(result.success for result in n2_results),
        n3_runs=sum(result is not None for result in n3_results),
        median_parameter_l2_error=float(np.median([row["parameter_l2_error"] for row in rows])),
        max_parameter_l2_error=float(np.max([row["parameter_l2_error"] for row in rows])),
        median_continuation_mse=float(np.median([row["continuation_mse"] for row in rows])),
        max_continuation_mse=float(np.max([row["continuation_mse"] for row in rows])),
        max_continuation_abs=float(np.max([row["continuation_max_abs"] for row in rows])),
        total_n1_evaluations=sum(result.evaluations for result in n1_results),
        total_n2_evaluations=sum(result.evaluations for result in n2_results),
        total_n3_objective_evaluations=sum(0 if result is None else result.objective_evaluations
                                           for result in n3_results),
        checkpoint_sha256_before=before,
        checkpoint_sha256_after=after,
        checkpoints_unchanged=before == after,
        elapsed_seconds=time.time()-started,
    )
    (OUTPUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
