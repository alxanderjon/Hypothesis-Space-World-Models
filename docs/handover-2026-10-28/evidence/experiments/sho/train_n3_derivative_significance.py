#!/usr/bin/env python3
"""Learn only N3 derivative significance and nonzero gradient magnitude."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import run_stage1b_n3_gradient_polishing as n3
from experiments.sho import train_n3_derivative_representation as prior
from experiments.sho import train_n3_stencil_scale_selector as scale_audit


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/n3_learned_polisher/derivative_significance")
BALL_ROWS = Path("outputs/bouncing_ball/second_broad_n1_n2_n3_transfer/per_trajectory.csv")
EXPONENTS = scale_audit.EXPONENTS
SEED = 20_261_004
SCALE_FEATURES = 8


@dataclass
class State:
    family: str
    split: str
    case_id: str
    theta: torch.Tensor
    objective: Callable[[torch.Tensor], torch.Tensor]
    topology: Callable[[np.ndarray], tuple] | None


@dataclass
class Coordinate:
    state: State
    axis: int
    target: float
    center_value: float
    hessian: np.ndarray
    estimates: np.ndarray
    numerators: np.ndarray
    cancellation: np.ndarray
    topology_valid: np.ndarray
    resolved: np.ndarray
    stable_member: np.ndarray
    baseline: float
    reliable_nonzero: bool
    scale_features: torch.Tensor


class SetEncoder(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layer = nn.Linear(SCALE_FEATURES, width, dtype=torch.float64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = torch.tanh(self.layer(values))
        return torch.cat((encoded.mean(dim=-2), encoded.amax(dim=-2)), dim=-1)


class SignificanceRouter(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.encoder = SetEncoder(width)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width, dtype=torch.float64), nn.Tanh(),
            nn.Linear(width, 2, dtype=torch.float64),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(values))


class NonzeroEstimator(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.encoder = SetEncoder(width)
        self.head = nn.Sequential(
            nn.Linear(2 * width + 2, width, dtype=torch.float64), nn.Tanh(),
            nn.Linear(width, 1, dtype=torch.float64),
        )

    def correction(self, values: torch.Tensor, summaries: torch.Tensor) -> torch.Tensor:
        # Positive factor preserves the sign of the resolved multiscale aggregate.
        raw = self.head(torch.cat((self.encoder(values), summaries), dim=-1)).squeeze(-1)
        return torch.exp2(2.0 * torch.tanh(raw))


def build_states() -> list[State]:
    result = [State(s.family, s.split, s.case_id, s.theta, s.objective, s.topology)
              for s in scale_audit.states()]

    # Add nearby ball states, grouped by whole trajectory just like the endpoints.
    with BALL_ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=len(rows), n_timesteps=ball.TOTAL_STEPS,
        dt=ball.DT, seed=20_261_001,
    )
    for row, trajectory in zip(rows, trajectories):
        trajectory_id = int(row["trajectory_id"])
        split = "train" if trajectory_id < 4 else "heldout" if trajectory_id < 6 else "fresh"
        trajectory_np = np.asarray(trajectory, dtype=np.float64)
        trajectory_t = torch.from_numpy(trajectory_np)
        base = torch.tensor((float(row["n2_a"]), float(row["n2_b"])))
        perturb = torch.tensor((1.0, -1.0)) * ((trajectory_id + 1) * 2.0**-20)
        theta = base + perturb

        def topology(value: np.ndarray, p=trajectory_np) -> tuple:
            return tuple(ball.numpy_k_event_signature(p[index], value)
                         for index in range(ball.PREFIX_STATES - 1))

        result.append(State(
            "ball", split, f"{trajectory_id}_near", theta,
            lambda value, p=trajectory_t: ball.prefix_loss_torch(value, p), topology,
        ))
    return result


def exact(state: State) -> tuple[np.ndarray, np.ndarray, float]:
    theta = state.theta.detach().clone().requires_grad_(True)
    value = state.objective(theta)
    gradient = torch.autograd.grad(value, theta, create_graph=True)[0]
    hessian = torch.stack([
        torch.autograd.grad(gradient[index], theta, retain_graph=True)[0]
        for index in range(len(theta))
    ])
    hessian = 0.5 * (hessian + hessian.T)
    return gradient.detach().numpy(), hessian.detach().numpy(), float(value.detach())


def same_topology(state: State, points: list[torch.Tensor]) -> bool:
    if state.topology is None:
        return True
    center = state.topology(state.theta.detach().numpy())
    return all(state.topology(point.detach().numpy()) == center for point in points)


def safe_log2(value: float) -> float:
    return float(np.log2(max(abs(value), np.finfo(np.float64).tiny)))


def materialize(state: State) -> list[Coordinate]:
    theta = state.theta.detach()
    gradient, hessian, f0_float = exact(state)
    f0 = torch.tensor(f0_float, dtype=torch.float64)
    dimension = len(theta)
    estimates = [[] for _ in range(dimension)]
    numerators = [[] for _ in range(dimension)]
    cancellation = [[] for _ in range(dimension)]
    topology_valid = [[] for _ in range(dimension)]
    resolved = [[] for _ in range(dimension)]
    with torch.no_grad():
        for exponent in EXPONENTS:
            steps = (2.0**(-exponent)) * torch.maximum(torch.ones_like(theta), torch.abs(theta))
            for axis in range(dimension):
                offset = torch.zeros_like(theta); offset[axis] = steps[axis]
                plus_point, minus_point = theta + offset, theta - offset
                plus, minus = state.objective(plus_point), state.objective(minus_point)
                numerator = float(plus - minus)
                estimate = numerator / (2.0 * float(steps[axis]))
                denominator = abs(float(plus)) + abs(float(minus)) + 2.0*abs(f0_float) + 1e-300
                spacing = max(abs(float(np.spacing(np.float64(float(plus))))),
                              abs(float(np.spacing(np.float64(float(minus))))),
                              abs(float(np.spacing(np.float64(f0_float)))))
                estimates[axis].append(estimate)
                numerators[axis].append(numerator)
                cancellation[axis].append(abs(numerator) / denominator)
                topology_valid[axis].append(same_topology(state, [plus_point, minus_point]))
                resolved[axis].append(abs(numerator) > 32.0 * spacing)

    output = []
    for axis in range(dimension):
        estimate = np.asarray(estimates[axis], dtype=np.float64)
        numerator = np.asarray(numerators[axis], dtype=np.float64)
        cancel = np.asarray(cancellation[axis], dtype=np.float64)
        topology = np.asarray(topology_valid[axis], dtype=bool)
        resolve = np.asarray(resolved[axis], dtype=bool)
        stable = np.zeros(len(EXPONENTS), dtype=bool)
        pair_signs = []
        for index in range(len(EXPONENTS)-1):
            left, right = estimate[index], estimate[index+1]
            if not (topology[index] and topology[index+1] and resolve[index] and resolve[index+1]):
                continue
            if left == 0.0 or right == 0.0 or np.signbit(left) != np.signbit(right):
                continue
            # The disagreement interval must not touch zero. This has no fitted tolerance.
            if abs(left-right) >= min(abs(left), abs(right)):
                continue
            stable[index] = stable[index+1] = True
            pair_signs.append(-1.0 if np.signbit(left) else 1.0)
        baseline = float(np.median(estimate[stable])) if stable.any() else 0.0
        target = float(gradient[axis])
        no_sign_conflict = len(set(pair_signs)) <= 1
        better_than_zero = target != 0.0 and abs(baseline-target) < abs(target)
        reliable = bool(stable.any() and no_sign_conflict and better_than_zero
                        and np.signbit(baseline) == np.signbit(target))

        rows = []
        magnitude = max(float(np.max(np.abs(estimate[topology]))) if topology.any() else 0.0,
                        np.finfo(np.float64).tiny)
        for index, exponent in enumerate(EXPONENTS):
            neighbor = estimate[index+1] if index+1 < len(estimate) else estimate[index]
            neighbor_relative = abs(neighbor-estimate[index]) / max(abs(estimate[index]),
                                                                     np.finfo(np.float64).tiny)
            rows.append((
                exponent / 40.0,
                float(np.sign(estimate[index])),
                float(np.clip(safe_log2(estimate[index])/1024.0, -1.0, 1.0)),
                float(np.clip(safe_log2(cancel[index])/1024.0, -1.0, 1.0)),
                float(resolve[index]), float(topology[index]),
                float(np.clip(safe_log2(neighbor_relative)/64.0, -1.0, 1.0)),
                float(stable[index]),
            ))
        output.append(Coordinate(
            state, axis, target, f0_float, hessian, estimate, numerator, cancel,
            topology, resolve, stable, baseline, reliable,
            torch.tensor(rows, dtype=torch.float64),
        ))
    return output


def train_router(coordinates: list[Coordinate]) -> tuple[SignificanceRouter, dict]:
    train = [item for item in coordinates if item.state.split == "train"]
    features = torch.stack([item.scale_features for item in train])
    targets = torch.tensor([int(item.reliable_nonzero) for item in train], dtype=torch.long)
    attempts = []
    for width in (2, 8):
        torch.manual_seed(SEED + width)
        model = SignificanceRouter(width)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        best, best_errors, stale = None, 10**9, 0
        for epoch in range(5000):
            optimizer.zero_grad()
            logits = model(features)
            counts = torch.bincount(targets, minlength=2).clamp_min(1)
            weights = len(targets)/(2.0*counts)
            loss = nn.functional.cross_entropy(logits, targets, weight=weights)
            loss.backward(); optimizer.step()
            if epoch % 25 == 0:
                with torch.no_grad(): errors = int((model(features).argmax(1) != targets).sum())
                if errors < best_errors:
                    best_errors = errors
                    best = {key:value.detach().clone() for key,value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 25
                if errors == 0 or stale >= 750:
                    break
        model.load_state_dict(best)
        attempts.append(dict(width=width, parameters=sum(p.numel() for p in model.parameters()),
                             epochs=epoch+1, training_errors=best_errors))
        if best_errors == 0:
            return model.eval().requires_grad_(False), dict(attempts=attempts,
                selected_width=width, parameters=sum(p.numel() for p in model.parameters()))
    return model.eval().requires_grad_(False), dict(attempts=attempts,
        selected_width=width, parameters=sum(p.numel() for p in model.parameters()))


def estimator_summaries(items: list[Coordinate]) -> torch.Tensor:
    return torch.tensor([
        (np.clip(safe_log2(item.baseline)/1024.0, -1.0, 1.0),
         float(np.count_nonzero(item.stable_member))/len(EXPONENTS))
        for item in items
    ], dtype=torch.float64)


def train_estimator(coordinates: list[Coordinate]) -> tuple[NonzeroEstimator, dict]:
    train = [item for item in coordinates if item.state.split == "train" and item.reliable_nonzero]
    features = torch.stack([item.scale_features for item in train])
    summaries = estimator_summaries(train)
    baseline = torch.tensor([item.baseline for item in train])
    targets = torch.tensor([item.target for item in train])
    log_ratio = torch.log2(torch.abs(targets / baseline))
    attempts = []
    for width in (2, 8):
        torch.manual_seed(SEED + 100 + width)
        model = NonzeroEstimator(width)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=600,
                                      tolerance_grad=1e-15, tolerance_change=1e-16,
                                      line_search_fn="strong_wolfe")
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            raw = model.head(torch.cat((model.encoder(features), summaries), dim=1)).squeeze(-1)
            prediction = 2.0*torch.tanh(raw)
            loss = torch.mean((prediction-log_ratio)**2)
            loss.backward(); return loss
        optimizer.step(closure)
        with torch.no_grad():
            prediction = baseline*model.correction(features, summaries)
            relative = torch.abs(prediction-targets)/torch.abs(targets)
            failures = int(torch.count_nonzero(~torch.isfinite(relative) | (relative >= 1.0)))
            max_relative = float(torch.max(relative))
        attempts.append(dict(width=width, parameters=sum(p.numel() for p in model.parameters()),
                             training_nonzero=len(train), useful_estimate_failures=failures,
                             max_relative_error=max_relative))
        if failures == 0:
            return model.eval().requires_grad_(False), dict(attempts=attempts,
                selected_width=width, parameters=sum(p.numel() for p in model.parameters()))
    return model.eval().requires_grad_(False), dict(attempts=attempts,
        selected_width=width, parameters=sum(p.numel() for p in model.parameters()))


def predict_estimate(model: NonzeroEstimator, item: Coordinate) -> float:
    summaries = estimator_summaries([item])
    with torch.no_grad(): factor = float(model.correction(item.scale_features[None], summaries)[0])
    return item.baseline * factor


def progress_needed(item: Coordinate) -> bool:
    gradient = torch.tensor([entry.target for entry in STATE_COORDINATES[id(item.state)]])
    hessian = torch.from_numpy(item.hessian)
    _, direction = prior.direct_route(gradient, hessian)
    if direction is None:
        return False
    predicted = max(0.0, -float(torch.dot(gradient, direction)))
    spacing = abs(float(np.spacing(np.float64(item.center_value))))
    return predicted > 32.0*spacing


STATE_COORDINATES: dict[int, list[Coordinate]] = {}


def evaluate(router: SignificanceRouter, estimator: NonzeroEstimator,
             coordinates: list[Coordinate]) -> dict:
    groups = defaultdict(lambda: dict(components=0, zero_teacher=0, nonzero_teacher=0,
        router_errors=0, false_nonzero=0, false_zero=0, trusted_wrong_sign=0,
        estimator_useful_failures=0, max_nonzero_relative_error=0.0))
    routed = {}
    estimates = {}
    failures = []
    oracle_estimator = defaultdict(lambda: dict(components=0, wrong_sign=0,
        useful_failures=0, max_relative_error=0.0))
    for item in coordinates:
        key = f"{item.state.family}_{item.state.split}"; group = groups[key]
        group["components"] += 1
        group["nonzero_teacher" if item.reliable_nonzero else "zero_teacher"] += 1
        with torch.no_grad(): decision = bool(router(item.scale_features[None]).argmax(1)[0])
        routed[id(item)] = decision
        group["router_errors"] += int(decision != item.reliable_nonzero)
        group["false_nonzero"] += int(decision and not item.reliable_nonzero)
        group["false_zero"] += int(not decision and item.reliable_nonzero)
        if decision != item.reliable_nonzero:
            failures.append(dict(kind="router", family=item.state.family,
                split=item.state.split, case_id=item.state.case_id, axis=item.axis,
                teacher="NONZERO" if item.reliable_nonzero else "ZERO_UNRESOLVED",
                predicted="NONZERO" if decision else "ZERO_UNRESOLVED",
                target=item.target, baseline=item.baseline,
                stable_scales=int(np.count_nonzero(item.stable_member))))
        if decision:
            estimate = predict_estimate(estimator, item); estimates[id(item)] = estimate
            if item.target != 0.0:
                group["trusted_wrong_sign"] += int(np.signbit(estimate) != np.signbit(item.target))
                relative = abs(estimate-item.target)/abs(item.target)
                group["estimator_useful_failures"] += int(not math.isfinite(relative) or relative >= 1.0)
                group["max_nonzero_relative_error"] = max(group["max_nonzero_relative_error"], relative)
                if np.signbit(estimate) != np.signbit(item.target) or not math.isfinite(relative) or relative >= 1.0:
                    failures.append(dict(kind="trusted_estimate", family=item.state.family,
                        split=item.state.split, case_id=item.state.case_id, axis=item.axis,
                        target=item.target, baseline=item.baseline, prediction=estimate,
                        relative_error=relative))
        if item.reliable_nonzero:
            oracle_prediction = predict_estimate(estimator, item)
            oracle_relative = abs(oracle_prediction-item.target)/abs(item.target)
            oracle = oracle_estimator[key]; oracle["components"] += 1
            oracle["wrong_sign"] += int(np.signbit(oracle_prediction) != np.signbit(item.target))
            oracle["useful_failures"] += int(not math.isfinite(oracle_relative) or oracle_relative >= 1.0)
            oracle["max_relative_error"] = max(oracle["max_relative_error"], oracle_relative)

    diagnostic = defaultdict(lambda: dict(states=0, gradient_sign_errors_on_trusted=0,
        curvature_route_errors=0, progress_states_without_nonzero=0,
        worst_gradient_relative_error=0.0, worst_direction_relative_error=0.0))
    for state_items in STATE_COORDINATES.values():
        state = state_items[0].state
        predicted = np.zeros(len(state.theta), dtype=np.float64)
        for item in state_items:
            if routed[id(item)]: predicted[item.axis] = estimates[id(item)]
        true = np.asarray([item.target for item in state_items])
        hessian = state_items[0].hessian
        key = f"{state.family}_{state.split}"; summary = diagnostic[key]; summary["states"] += 1
        trusted_indices = [item.axis for item in state_items if routed[id(item)]]
        summary["gradient_sign_errors_on_trusted"] += sum(
            int(np.signbit(predicted[index]) != np.signbit(true[index])) for index in trusted_indices)
        true_route, true_direction = prior.direct_route(torch.from_numpy(true), torch.from_numpy(hessian))
        route, direction = prior.direct_route(torch.from_numpy(predicted), torch.from_numpy(hessian))
        summary["curvature_route_errors"] += int(route != true_route)
        if progress_needed(state_items[0]) and not trusted_indices:
            summary["progress_states_without_nonzero"] += 1
            failures.append(dict(kind="progress_without_nonzero", family=state.family,
                split=state.split, case_id=state.case_id,
                teacher_gradient=true.tolist(), center_value=state_items[0].center_value))
        summary["worst_gradient_relative_error"] = max(summary["worst_gradient_relative_error"],
            float(np.linalg.norm(predicted-true)/max(np.linalg.norm(true), np.finfo(np.float64).tiny)))
        direction_relative = math.inf if direction is None or true_direction is None else float(
            torch.linalg.vector_norm(direction-true_direction)/
            max(float(torch.linalg.vector_norm(true_direction)), np.finfo(np.float64).tiny))
        summary["worst_direction_relative_error"] = max(summary["worst_direction_relative_error"],
                                                           direction_relative)

    totals = dict(
        router_errors=sum(value["router_errors"] for value in groups.values()),
        false_nonzero=sum(value["false_nonzero"] for value in groups.values()),
        false_zero=sum(value["false_zero"] for value in groups.values()),
        trusted_wrong_sign=sum(value["trusted_wrong_sign"] for value in groups.values()),
        estimator_useful_failures=sum(value["estimator_useful_failures"] for value in groups.values()),
        curvature_route_errors=sum(value["curvature_route_errors"] for value in diagnostic.values()),
        progress_states_without_nonzero=sum(value["progress_states_without_nonzero"] for value in diagnostic.values()),
    )
    passed = all(value == 0 for value in totals.values())
    return dict(passed=passed, totals=totals, groups=dict(groups),
        oracle_estimator_on_teacher_nonzero=dict(oracle_estimator),
        diagnostic=dict(diagnostic), failures=failures)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    states = build_states(); coordinates = []
    for index, state in enumerate(states):
        items = materialize(state); STATE_COORDINATES[id(state)] = items; coordinates.extend(items)
        if (index+1) % 10 == 0: print(f"stencils {index+1}/{len(states)}", flush=True)

    router, router_training = train_router(coordinates)
    estimator, estimator_training = train_estimator(coordinates)
    validation = evaluate(router, estimator, coordinates)
    passed = validation["passed"]
    checkpoints = {}
    if passed:
        topology_model, topology_report = scale_audit.fit_topology()
        topology_path = OUT/"n3_stencil_topology_validity.pt"
        router_path = OUT/"n3_derivative_significance.pt"
        estimator_path = OUT/"n3_nonzero_derivative_estimator.pt"
        torch.save(dict(status="LOCKED", architecture="Linear(1,2)",
                        state_dict=topology_model.state_dict(), report=topology_report), topology_path)
        torch.save(dict(status="FROZEN_CANDIDATE", architecture=type(router).__name__,
                        exponents=EXPONENTS, state_dict=router.state_dict()), router_path)
        torch.save(dict(status="FROZEN_CANDIDATE", architecture=type(estimator).__name__,
                        exponents=EXPONENTS, state_dict=estimator.state_dict()), estimator_path)
        checkpoints = {str(path):sha256(path) for path in (topology_path,router_path,estimator_path)}

    label_counts = Counter((item.state.family, item.state.split,
                            "NONZERO" if item.reliable_nonzero else "ZERO_UNRESOLVED")
                           for item in coordinates)
    hard_ids = {"9","10","17","132","230","236"}
    difficult = {}
    for item in coordinates:
        if (item.state.family == "sho" and item.state.case_id in hard_ids) or item.state.family == "ball":
            difficult[f"{item.state.family}_{item.state.case_id}_axis{item.axis}"] = dict(
                label="NONZERO" if item.reliable_nonzero else "ZERO_UNRESOLVED",
                target=item.target, baseline=item.baseline,
                topology_valid_scales=int(np.count_nonzero(item.topology_valid)),
                stable_scales=int(np.count_nonzero(item.stable_member)))
    report = dict(
        status="PASSED" if passed else "STOPPED_SIGNIFICANCE_OR_ESTIMATOR_FAILED",
        teacher_semantics=dict(
            production_zero="the existing N3 stops only when the float64 gradient norm equals exactly 0.0",
            distinguishability="a stencil numerator is resolved only when it exceeds 32 times the largest local objective-value ULP, reusing the existing N3 no-further-improvement factor",
            stable_pair="adjacent topology-valid, resolved estimates must have the same nonzero sign and disagreement smaller than the smaller magnitude",
            significance="RELIABLY_NONZERO iff stable scales have no sign conflict, their median has teacher sign, and it is closer to the autograd derivative than the zero estimate; otherwise ZERO_UNRESOLVED",
            estimator="shared set encoder predicts a positive multiplicative correction to the median of all stable topology-valid estimates; no stencil index is predicted"),
        systems=dict(states=len(states), coordinates=len(coordinates), dimensions="2 and synthetic 3-6",
                     exponents=EXPONENTS),
        label_counts={"|".join(key):value for key,value in sorted(label_counts.items())},
        significance_router=router_training,
        nonzero_estimator=estimator_training,
        validation=validation,
        difficult_cases=difficult,
        checkpoints=checkpoints,
        existing_checkpoints_modified=False,
        next_step_safe=bool(passed),
        diagnosis="" if passed else "The first failing gate is in derivative significance or nonzero magnitude; stop before Hessian/curvature learning.",
        elapsed_seconds=time.time()-started,
    )
    (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
