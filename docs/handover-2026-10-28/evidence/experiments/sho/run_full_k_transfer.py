#!/usr/bin/env python3
"""First full 2x2 SHO K recovery with the frozen learned N1 -> N2 -> N3 stack.

Only parameter-dimension adapters are generalized here.  Learned checkpoints and
their weights remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import qmc

from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as hessian
from experiments.sho import assemble_n3_learned_polisher as n3
from experiments.sho import run_stage1b_fifo_vs_ranker as active_n1
from experiments.sho import run_stage1b_learned_global_search_imitation as ranker
from experiments.sho import run_stage1b_learned_powell_2d as scalar_n2
from experiments.sho import run_stage1b_n3_gradient_polishing as n3_teacher
from experiments.sho import run_stage1b_n3_rollout_precision as rollout
from experiments.sho import train_n2_direction_controller as direction_n2
from experiments.sho.train_n3_derivative_significance import State as N3State

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/full_k_transfer")
DIMENSION = 4
LOWER = np.full(DIMENSION, -1.99, dtype=np.float64)
UPPER = np.full(DIMENSION, 1.99, dtype=np.float64)
WIDTH = UPPER - LOWER
N1_GATE = active_n1.GATE
N1_CAP = ranker.NORMAL_CAP
N2_GATE = scalar_n2.SUCCESS_LOSS
N2_CAP = 1500
CASE_COUNT = 4
DATA_SEED = 20_261_030
PREFIX_STATES = 128
ROLLOUT_STEPS = 4096
SEED_BASE = 20_261_031


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_paths() -> list[Path]:
    paths = [
        Path("outputs/sho/stage1B/distance_metric_assembled_validation/geometric_distance_metric_locked.pt"),
        Path("outputs/sho/stage1B/nearest_argmin_assembled/nearest_argmin_operator_locked.pt"),
        Path("outputs/sho/stage1B/learned_cell_size_assembled/half_spacing_operator_locked.pt"),
        direction_n2.LOCAL_CHECKPOINT,
        direction_n2.LINE_CHECKPOINT,
        Path("outputs/sho/stage1B/n3_learned_polisher/gradient_operator/n3_gradient_operator.pt"),
        Path("outputs/sho/stage1B/n3_learned_polisher/hessian_operator/n3_hessian_operator.pt"),
        Path("outputs/sho/stage1B/n3_learned_polisher/curvature_router/n3_curvature_router.pt"),
        Path("outputs/sho/stage1B/n3_learned_polisher/control_and_closed_loop/n3_learned_polisher.pt"),
    ]
    return [p for p in paths if p.exists()]


def theta_to_k(theta: np.ndarray | torch.Tensor):
    return theta.reshape(2, 2)


def objective_numpy(theta: np.ndarray, position: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(theta).astype(np.float64, copy=False)
    inputs = np.column_stack((position[:PREFIX_STATES-2], position[1:PREFIX_STATES-1]))
    targets = np.column_stack((position[1:PREFIX_STATES-1], position[2:PREFIX_STATES]))
    matrices = values.reshape(-1, 2, 2)
    predicted = np.einsum("nij,tj->nti", matrices, inputs)
    return np.mean(np.square(predicted - targets[None, :, :]), axis=(1, 2))


def objective_torch_native(theta: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    inputs = torch.stack((position[:PREFIX_STATES-2], position[1:PREFIX_STATES-1]), dim=1)
    targets = torch.stack((position[1:PREFIX_STATES-1], position[2:PREFIX_STATES]), dim=1)
    predicted = inputs @ theta.reshape(2, 2).T
    return torch.mean(torch.square(predicted - targets))


def objective_torch_learned(theta: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    previous = position[:PREFIX_STATES-2]
    current = position[1:PREFIX_STATES-1]
    target_first = position[1:PREFIX_STATES-1]
    target_second = position[2:PREFIX_STATES]
    k11, k12, k21, k22 = theta.unbind()
    first = hessian.add(hessian.multiply(k11, previous), hessian.multiply(k12, current))
    second = hessian.add(hessian.multiply(k21, previous), hessian.multiply(k22, current))
    residual = hessian.subtract(torch.stack((first, second), dim=1),
                                torch.stack((target_first, target_second), dim=1))
    return hessian.mean(hessian.square(residual))


def initial_design(dimension: int) -> np.ndarray:
    count = 5 * dimension
    multipliers = []
    candidate = 1
    while len(multipliers) < dimension:
        if math.gcd(candidate, count) == 1:
            multipliers.append(candidate)
        candidate += 1
    base = np.arange(count, dtype=np.float64)
    return np.column_stack([((m * base) % count + 0.5) / count for m in multipliers])


def physical(points: np.ndarray) -> np.ndarray:
    return LOWER + points * WIDTH


def candidate_pool(sample_count: int, case_id: int, iteration: int) -> np.ndarray:
    requested = min(50 * DIMENSION * sample_count, active_n1.edirect.MC_CANDIDATE_CAP)
    exponent = int(math.ceil(math.log2(max(2, requested))))
    seed = SEED_BASE + 104729 * case_id + 1009 * iteration
    interior = qmc.Sobol(d=DIMENSION, scramble=True, seed=seed).random_base2(exponent)[:requested]
    # The production 2-D edge construction generalizes structurally to 2*d faces.
    face_count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    faces = []
    for axis in range(DIMENSION):
        engine = qmc.Sobol(d=DIMENSION-1, scramble=True, seed=seed + 7919 * (axis + 1))
        count_exp = int(math.ceil(math.log2(max(2, face_count))))
        free = engine.random_base2(count_exp)[:face_count]
        for side in (0.0, 1.0):
            face = np.empty((face_count, DIMENSION), dtype=np.float64)
            face[:, axis] = side
            face[:, np.arange(DIMENSION) != axis] = free
            faces.append(face)
    return np.unique(np.concatenate((interior, *faces), axis=0), axis=0)


def block_learned_distance(model, delta: np.ndarray) -> np.ndarray:
    """Dimension-generic composition of the locked exact 2-D Euclidean metric."""
    delta = np.asarray(delta, dtype=np.float64)
    shape = delta.shape[:-1]
    flat = delta.reshape(-1, DIMENSION)
    pair0 = active_n1.cells.base.distance_metric._predict(model, flat[:, :2])
    pair1 = active_n1.cells.base.distance_metric._predict(model, flat[:, 2:])
    return np.sqrt(pair0 * pair0 + pair1 * pair1).reshape(shape)


class FrozenProposals4D:
    def __init__(self):
        base = active_n1.FrozenProposals()
        self.__dict__.update(base.__dict__)
        metric = self.distance.metric_matrix().detach().numpy()
        if not np.array_equal(metric, np.eye(2)):
            raise RuntimeError("locked learned distance is not the exact Euclidean primitive")

    def nearest(self, sites: np.ndarray, probes: np.ndarray):
        tree = cKDTree(sites)
        reference_distance, candidates = tree.query(probes, k=2)
        d0 = block_learned_distance(self.distance, probes-sites[candidates[:, 0]])
        d1 = block_learned_distance(self.distance, probes-sites[candidates[:, 1]])
        tied = reference_distance[:, 0] == reference_distance[:, 1]
        replace = np.zeros(len(probes), dtype=bool)
        active = ~tied
        scale = np.maximum(d0[active], d1[active])
        inputs = np.column_stack((d0[active]/scale, d1[active]/scale))
        margin = active_n1.nearest.margin.predict(self.margin, inputs)
        logits = float(self.threshold.alpha()) * (margin-float(self.threshold.theta()))
        replace[active] = logits > 0.0
        owners = np.where(replace, candidates[:, 1], candidates[:, 0]).astype(np.int64)
        distances = np.where(replace, d1, d0)
        # Production cKDTree ownership is authoritative only on exact ties.
        owners[tied] = tree.query(probes[tied], k=1)[1]
        distances[tied] = tree.query(probes[tied], k=1)[0]
        return owners, distances

    def proposals(self, sites: np.ndarray, losses: np.ndarray, case: dict, iteration: int):
        probes = candidate_pool(len(sites), case["generator_id"], iteration)
        owners, distances = self.nearest(sites, probes)
        radius, _, stats = active_n1.cells.assigned_radius(
            owners, distances, len(sites), self.margin, self.threshold)
        # Generic nearest-neighbor spacing, evaluated with the same learned metric.
        tree = cKDTree(sites)
        _, near = tree.query(sites, k=2)
        spacing = block_learned_distance(self.distance, sites-sites[near[:, 1]])
        half = active_n1.cells.predict_half(self.half, spacing)
        replace, _, invalid, wrong = active_n1.cells.larger_candidate(
            self.margin, self.threshold, radius, half)
        if invalid or wrong or stats["radius_pairwise_mistakes"] or stats["radius_invalid_logits"]:
            raise RuntimeError("locked learned cell-size reduction diverged in 4-D")
        sizes = np.where(replace, half, radius)
        grouped = active_n1.down._learned_group_and_represent(sizes, losses, self.locked)
        hull = active_n1.down.hull_threshold._threshold_hull(
            self.locked["skew_weight"], self.locked["homogeneous_weight"],
            self.locked["hull_theta"], self.locked["hull_alpha"],
            grouped["grouped_sizes"], grouped["grouped_losses"])
        pool = active_n1.down.improvement_and._build_pool([dict(
            source_state_index=iteration, trajectory_id=case["trajectory_id"], iteration=iteration,
            state_bin=active_n1.cells.base.turn._state_bin(len(sizes)),
            sizes=grouped["grouped_sizes"], losses=grouped["grouped_losses"], teacher_hull=hull)],
            "full_k_transfer")
        with torch.no_grad():
            passed = self.locked["and_model"](torch.from_numpy(pool["inputs"])).numpy().ravel() >= 0
        selected = hull[pool["hull_position"][passed]]
        if not len(selected):
            selected = np.asarray([np.argmax(grouped["grouped_sizes"])])
        po = sorted(grouped["representatives"][selected].tolist(),
                    key=lambda i: (sizes[i], losses[i], i))
        surface = active_n1.edirect.GaussianLocalResponse(sites, losses)
        weights = .5*active_n1.edirect._normalize(sizes[po]) + .5*active_n1.edirect._normalize(
            surface.loo_errors(np.asarray(po)))
        largest = int(np.argmax(sizes[po])); proposals = []
        for j, site in enumerate(po):
            ix = np.flatnonzero(owners == site)
            if not len(ix):
                continue
            if j == largest:
                winner = int(np.argmax(distances[ix]))
            else:
                features = np.column_stack((active_n1.edirect._normalize(distances[ix]),
                    active_n1.edirect._normalize(surface.predict(probes[ix])),
                    np.full(len(ix), weights[j])))
                with torch.no_grad():
                    winner = int(np.argmax(self.score(torch.from_numpy(features)).numpy()))
            point = probes[ix[winner]]
            if np.min(np.linalg.norm(sites-point, axis=1)) > 1e-12:
                proposals.append(point)
        if not proposals:
            for index in np.argsort(distances)[::-1]:
                if distances[index] > 1e-12:
                    proposals.append(probes[index]); break
        return np.asarray(proposals, dtype=np.float64).reshape(-1, DIMENSION), dict(
            probe_count=len(probes), po_cells=len(po), cell_size_min=float(np.min(sizes)),
            cell_size_max=float(np.max(sizes)))


@dataclass
class N1Result:
    success: bool
    theta: np.ndarray
    loss: float
    evaluations: int
    decisions: int
    rounds: int
    initial_theta: np.ndarray
    initial_loss: float
    lower: np.ndarray
    upper: np.ndarray
    failure: str


def run_n1(case: dict, position: np.ndarray, frozen: FrozenProposals4D) -> N1Result:
    sites = initial_design(DIMENSION)
    losses = objective_numpy(physical(sites), position)
    initial = int(np.argmin(losses)); decisions = rounds = 0
    failure = ""
    try:
        while len(sites) < N1_CAP and float(np.min(losses)) > N1_GATE:
            pending, _ = frozen.proposals(sites, losses, case, rounds)
            while len(pending) and len(sites) < N1_CAP and float(np.min(losses)) > N1_GATE:
                # Preserve the frozen ranker feature semantics; its raw coordinate slots
                # remain the first two coordinates while all geometry features are d-generic.
                bounds2 = np.asarray(((-1.99, 1.99), (-1.99, 1.99)))
                global_features = ranker._global_features(
                    sites, losses, len(pending), len(sites), N1_CAP, rounds,
                    bounds2, "normal_edirect")
                choice, _ = ranker._model_choice(
                    frozen.ranker, sites, losses, pending, "cell", global_features)
                point = pending[choice].copy()
                value = float(objective_numpy(physical(point[None, :]), position)[0])
                sites = np.vstack((sites, point)); losses = np.r_[losses, value]
                pending = np.delete(pending, choice, axis=0); decisions += 1
            rounds += 1
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    best = int(np.argmin(losses)); theta = physical(sites[best:best+1])[0]
    spacing = WIDTH / math.sqrt(len(sites))
    lower = np.maximum(LOWER, theta-spacing); upper = np.minimum(UPPER, theta+spacing)
    return N1Result(float(losses[best]) <= N1_GATE and not failure, theta, float(losses[best]),
                    len(sites), decisions, rounds, physical(sites[initial:initial+1])[0],
                    float(losses[initial]), lower, upper, failure or ("" if losses[best] <= N1_GATE else "N1_GATE"))


class GenericEvaluationState(scalar_n2.EvaluationState):
    def evaluate(self, theta: np.ndarray) -> float:
        if self.count >= self.max_evaluations:
            raise RuntimeError("Objective evaluation cap exceeded")
        if not bool(np.all(theta >= self.lower-1e-13) and np.all(theta <= self.upper+1e-13)):
            raise RuntimeError("Frozen line controller proposed outside N2 bounds")
        theta = np.minimum(np.maximum(theta, self.lower), self.upper)
        loss = float(objective_numpy(theta[None, :], self.position)[0]); self.count += 1
        if loss < self.best_loss:
            self.best_loss = loss; self.best_theta = theta.copy(); self.last_best_evaluation = self.count
        for threshold in scalar_n2.THRESHOLDS:
            if self.first_threshold[threshold] is None and self.best_loss < threshold:
                self.first_threshold[threshold] = self.count
        self.best_loss_history.append(self.best_loss)
        return loss


@dataclass
class N2Result:
    success: bool
    theta: np.ndarray
    loss: float
    evaluations: int
    cycles: int
    direction_updates: int
    routing_errors: int
    failure: str


def run_n2(case_id: int, position: np.ndarray, start: N1Result, teacher: np.ndarray) -> N2Result:
    if not start.success:
        return N2Result(False, start.theta, start.loss, 0, 0, 0, 0, "N1_FAILED")
    if not bool(np.all(teacher >= start.lower) and np.all(teacher <= start.upper)):
        return N2Result(False, start.theta, start.loss, 0, 0, 0, 0, "N1_LOCAL_BOX_EXCLUDES_TEACHER")
    control = direction_n2.load_controller()
    state = GenericEvaluationState(position, start.lower, start.upper, start.theta.copy(),
                                   start.loss, max_evaluations=N2_CAP)
    theta, theta_loss = start.theta.copy(), start.loss
    directions = control.initial_directions(DIMENSION)
    cycles = direction_updates = line_serial = stagnant = 0
    original_interval = scalar_n2._alpha_interval
    scalar_n2._alpha_interval = direction_n2.alpha_interval
    try:
        while state.count < state.max_evaluations and state.best_loss >= N2_GATE:
            cycles += 1; theta_start, loss_start = theta.copy(), theta_loss; decreases = []
            for direction in directions:
                if state.count >= state.max_evaluations or state.best_loss < N2_GATE: break
                before = theta_loss; line_serial += 1
                result = scalar_n2._line_search(state, theta, theta_loss, direction,
                    scalar_n2.LINE_SEED + 100_003*case_id + line_serial)
                theta, theta_loss = result.theta, result.loss
                decreases.append(max(0.0, before-theta_loss))
            if state.best_loss < N2_GATE or state.count >= state.max_evaluations or len(decreases) < DIMENSION: break
            theta_end, loss_end = theta.copy(), theta_loss
            displacement = theta_end-theta_start; displacement_norm = float(np.linalg.norm(displacement))
            biggest = control.largest_decrease_index(decreases); biggest_decrease = decreases[biggest]
            significant = displacement_norm > scalar_n2.DISPLACEMENT_TOLERANCE*max(1.0, float(np.linalg.norm(start.upper-start.lower)))
            extrapolated = theta_end+displacement
            inside = bool(np.all(extrapolated >= start.lower) and np.all(extrapolated <= start.upper))
            if control.conjunction(significant, inside):
                extrapolated_loss = state.evaluate(extrapolated)
                if state.best_loss < N2_GATE: theta, theta_loss = extrapolated, extrapolated_loss; break
                better = extrapolated_loss < loss_start
                first = loss_start-loss_end-biggest_decrease; second = loss_start-extrapolated_loss
                test = 2.0*(loss_start-2.0*loss_end+extrapolated_loss)*first*first-biggest_decrease*second*second
                if control.conjunction(better, control.should_replace(test)) and state.count < state.max_evaluations:
                    line_serial += 1
                    result = scalar_n2._line_search(state, theta, theta_loss, displacement,
                        scalar_n2.LINE_SEED + 100_003*case_id + line_serial)
                    theta, theta_loss = result.theta, result.loss
                    directions = control.update(directions, biggest, displacement); direction_updates += 1
            relative = (loss_start-theta_loss)/max(abs(loss_start), 1e-300)
            if relative <= 1e-10 and displacement_norm <= 1e-10*max(1.0,float(np.linalg.norm(start.upper-start.lower))): stagnant += 1
            else: stagnant = 0
            if stagnant >= 2: break
    finally:
        scalar_n2._alpha_interval = original_interval
    errors = sum(value for key, value in control.counts.items() if key.endswith("_errors"))
    return N2Result(state.best_loss < N2_GATE, state.best_theta, state.best_loss,
                    state.count, cycles, direction_updates, errors,
                    "" if state.best_loss < N2_GATE else ("N2_STAGNATION" if stagnant >= 2 else "N2_GATE"))


@dataclass
class N3Result:
    theta: np.ndarray
    loss: float
    accepted_steps: int
    evaluations: int
    stop_reason: str
    routing_errors: int


def run_n3(position: np.ndarray, start: N2Result, lower: np.ndarray, upper: np.ndarray) -> N3Result:
    if not start.success:
        return N3Result(start.theta, start.loss, 0, 0, "N2_FAILED", 0)
    p = torch.from_numpy(position)
    state = N3State("full_sho_k", "fresh", "", torch.from_numpy(start.theta),
                    lambda value: objective_torch_native(value, p), None)
    controls = n3.ControlAssembly()
    theta = torch.as_tensor(start.theta, dtype=torch.float64).clone()
    current_loss = float(state.objective(theta)); accepted = evaluations = rejections = 0
    routing = Counter(); stop = "max_accepted_steps"
    while accepted < n3_teacher.MAX_ACCEPTED_STEPS:
        _, g, _, H = hessian.derivatives(lambda value: objective_torch_learned(value, p), theta, learned=True)
        norm = float(torch.linalg.vector_norm(g)); route = controls.pre_step_route(accepted, norm)
        if route != "CONTINUE": stop = route; break
        step = controls.newton_direction(g, H)
        if step["direction"] is None:
            stop = controls.pre_step_route(accepted, norm, step["status"]); break
        direction = step["direction"]; derivative = float(torch.dot(g, direction))
        accepted_this = unchanged = False
        for k in range(n3_teacher.MAX_BACKTRACKS+1):
            candidate = controls.candidate(theta, direction, k)
            if bool(torch.equal(candidate, theta)): unchanged = True; break
            if not controls.in_bounds(candidate, lower, upper): rejections += 1; continue
            value = float(state.objective(candidate)); evaluations += 1
            if controls.accepts(value, current_loss):
                theta = candidate.detach(); current_loss = value; accepted += 1; accepted_this = True; break
            rejections += 1
        if not accepted_this:
            stop = controls.exhausted_route(unchanged, derivative, current_loss); break
    return N3Result(theta.numpy().copy(), current_loss, accepted, evaluations, stop,
                    sum(routing.values()))


def rollout_matrix(k: np.ndarray, state: np.ndarray, steps: int) -> np.ndarray:
    output = np.empty((steps+1, 2), dtype=np.float64); output[0] = state
    for i in range(steps): output[i+1] = k @ output[i]
    return output


def validate_teacher(positions: np.ndarray, coefficients: np.ndarray) -> list[dict]:
    rows = []
    for i, (position, c) in enumerate(zip(positions, coefficients)):
        teacher = np.asarray([0.0, 1.0, -1.0, float(c)])
        full_loss = float(objective_numpy(teacher[None, :], position)[0])
        # The compact one-step baseline is the second-row residual.  Full-state MSE
        # includes an identically-zero shift row and hence is exactly half of it.
        compact = np.mean(np.square(-position[:PREFIX_STATES-2] + c*position[1:PREFIX_STATES-1]
                                    - position[2:PREFIX_STATES]))
        predicted = rollout_matrix(teacher.reshape(2,2), position[:2], PREFIX_STATES-2)[:, 1]
        phase = math.acos(float(c)/2.0)
        sine_coefficient = (position[1]-position[0]*math.cos(phase))/math.sin(phase)
        long_index = np.arange(ROLLOUT_STEPS+2, dtype=np.float64)
        analytic = position[0]*np.cos(long_index*phase) + sine_coefficient*np.sin(long_index*phase)
        long_prediction = rollout_matrix(teacher.reshape(2,2), position[:2], ROLLOUT_STEPS)[:, 1]
        rows.append(dict(case=i, c=float(c), full_loss=full_loss, compact_one_step_loss=float(compact),
                         baseline_relation_exact=bool(full_loss == 0.5*compact),
                         prefix_rollout_max_error=float(np.max(np.abs(predicted-position[1:PREFIX_STATES]))),
                         long_rollout_steps=ROLLOUT_STEPS,
                         long_rollout_max_error=float(np.max(np.abs(long_prediction-analytic[1:])))))
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows: return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = checkpoint_paths(); before = {str(p): sha256(p) for p in checkpoints}
    positions, _, coefficients = active_n1.im._make_exact_labeled_split(CASE_COUNT, DATA_SEED)
    positions = positions.numpy(); coefficients = coefficients.numpy()
    teacher_checks = validate_teacher(positions, coefficients)
    teacher_pass = all(r["baseline_relation_exact"] and r["prefix_rollout_max_error"] <= 5e-14
                       and r["long_rollout_max_error"] <= 1e-12
                       for r in teacher_checks)
    if not teacher_pass:
        report = dict(status="STOPPED_TEACHER_CONTAINMENT_FAILURE", teacher_checks=teacher_checks,
                      bounds=[LOWER.tolist(), UPPER.tolist()])
        (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n"); print(json.dumps(report, indent=2)); return
    gradient.OPS = gradient.LearnedArithmetic(); hessian.activate_second_order_gradient()
    frozen = FrozenProposals4D(); rows=[]; first_failure=""
    for i, (position, c) in enumerate(zip(positions, coefficients)):
        teacher = np.asarray([0.,1.,-1.,float(c)])
        case = dict(trajectory_id=i, generator_id=DATA_SEED+i)
        n1_result = run_n1(case, position, frozen)
        n2_result = run_n2(i, position, n1_result, teacher)
        n3_result = run_n3(position, n2_result, n1_result.lower, n1_result.upper)
        recovered = n3_result.theta; matrix_error = recovered-teacher
        teacher_future = rollout_matrix(teacher.reshape(2,2), position[PREFIX_STATES-2:PREFIX_STATES], ROLLOUT_STEPS)
        learned_future = rollout_matrix(recovered.reshape(2,2), position[PREFIX_STATES-2:PREFIX_STATES], ROLLOUT_STEPS)
        error = learned_future-teacher_future
        failure = n1_result.failure or n2_result.failure or ("" if n3_result.loss < n2_result.loss else "N3_NO_IMPROVEMENT")
        if failure and not first_failure: first_failure = "N1" if n1_result.failure else "N2" if n2_result.failure else "N3"
        rows.append(dict(case=i, c_teacher=float(c), teacher_k=json.dumps(teacher.reshape(2,2).tolist()),
            n1_initial_theta=json.dumps(n1_result.initial_theta.tolist()), n1_initial_loss=n1_result.initial_loss,
            n1_theta=json.dumps(n1_result.theta.tolist()), n1_loss=n1_result.loss,
            n1_evaluations=n1_result.evaluations, n1_learned_decisions=n1_result.decisions,
            n1_voronoi_rounds=n1_result.rounds, n1_success=int(n1_result.success),
            n2_theta=json.dumps(n2_result.theta.tolist()), n2_loss=n2_result.loss,
            n2_evaluations=n2_result.evaluations, n2_cycles=n2_result.cycles,
            n2_direction_updates=n2_result.direction_updates, n2_routing_errors=n2_result.routing_errors,
            n2_success=int(n2_result.success), n3_theta=json.dumps(recovered.tolist()),
            recovered_k=json.dumps(recovered.reshape(2,2).tolist()), n3_loss=n3_result.loss,
            n3_accepted_steps=n3_result.accepted_steps, n3_evaluations=n3_result.evaluations,
            n3_stop=n3_result.stop_reason, k11_error=float(matrix_error[0]),
            k12_error=float(matrix_error[1]), k21_error=float(matrix_error[2]),
            k22_error=float(matrix_error[3]), matrix_frobenius_error=float(np.linalg.norm(matrix_error)),
            matrix_max_error=float(np.max(np.abs(matrix_error))), rollout_steps=ROLLOUT_STEPS,
            rollout_max_error=float(np.max(np.abs(error))), rollout_mse=float(np.mean(error*error)), failure=failure))
        print(f"full-K {i+1}/{CASE_COUNT}: N1 {n1_result.evaluations}/{n1_result.loss:.3e} "
              f"N2 {n2_result.evaluations}/{n2_result.loss:.3e} N3 {n3_result.accepted_steps}/{n3_result.loss:.3e} "
              f"Kerr={np.max(np.abs(matrix_error)):.3e} failure={failure or 'none'}", flush=True)
    after = {str(p): sha256(p) for p in checkpoints}; unchanged = before == after
    write_csv(OUT/"per_trajectory.csv", rows)
    success_rows = [r for r in rows if not r["failure"]]
    report = dict(status="PASSED" if len(success_rows)==CASE_COUNT and unchanged else "STOPPED_FIRST_STAGE_FAILURE",
        first_failing_stage=first_failure, teacher_k_form="[[0,1],[-1,c_teacher]]",
        parameter_order=["k11","k12","k21","k22"], parameter_dimension=DIMENSION,
        bounds={name:[-1.99,1.99] for name in ("k11","k12","k21","k22")},
        objective="mean squared one-step reconstruction of full delay-state transition z_t -> z_(t+1)",
        objective_changed=False, fixed_matrix_entries=0, teacher_checks=teacher_checks,
        interface_changes=["5*d deterministic Latin-hypercube initial design",
            "d-dimensional scrambled Sobol interior reservoir",
            "2*d face generalization of production boundary support",
            "block composition of locked exact 2-D Euclidean metric for d=4",
            "runtime-dimension N2 box-line intersection and learned direction set",
            "runtime-dimension N3 gradient/Hessian indexing and identity"],
        unchanged_learned_components=["Sobol digital generator", "distance arithmetic", "nearest comparison",
            "cell-size reductions", "grouping/hull/improvement", "ordinary cell score", "candidate ranker",
            "N2 scalar line controller", "N2 direction routers", "N3 arithmetic/gradient/Hessian/control routers"],
        trajectories_requested=CASE_COUNT, trajectories_tested=len(rows), n1_successes=sum(r["n1_success"] for r in rows),
        n2_successes=sum(r["n2_success"] for r in rows), n3_successes=len(success_rows),
        learned_n1_exercised=sum(r["n1_learned_decisions"]>0 for r in rows),
        total_n1_evaluations=sum(r["n1_evaluations"] for r in rows),
        total_n2_evaluations=sum(r["n2_evaluations"] for r in rows),
        total_n3_evaluations=sum(r["n3_evaluations"] for r in rows),
        worst_matrix_max_error=max((r["matrix_max_error"] for r in rows), default=None),
        worst_matrix_frobenius_error=max((r["matrix_frobenius_error"] for r in rows), default=None),
        worst_rollout_max_error=max((r["rollout_max_error"] for r in rows), default=None),
        checkpoint_hashes_before=before, checkpoint_hashes_after=after, checkpoint_hashes_unchanged=unchanged,
        remaining_deterministic_scaffolding=["Sobol direction-number/RNG data", "candidate indexing and loop bookkeeping",
            "exact-tie ownership", "generic torch.linalg.solve"], elapsed_seconds=time.time()-started)
    (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
