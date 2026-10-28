#!/usr/bin/env python3
"""Quick screen for the first learned N3 component: local derivatives.

The screen is deliberately structured and dimension agnostic.  A shared
float64 linear head combines multiscale symmetric-stencil estimates for every
gradient coordinate and every entry of a symmetric Hessian.  It is trained
from random initialization against the existing autograd Newton teacher.

No checkpoint is promoted unless derivative signs, curvature routing, Newton
directions, and closed-loop polishing all preserve the teacher gates.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import run_stage1b_n3_gradient_polishing as teacher
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/n3_learned_polisher/derivative_screen")
BALL_ROWS = Path("outputs/bouncing_ball/second_broad_n1_n2_n3_transfer/per_trajectory.csv")
H_EXPONENTS = (8, 12, 16, 20, 24)
SEED = 20_261_002


@dataclass
class State:
    family: str
    split: str
    case_id: str
    theta: torch.Tensor
    objective: Callable[[torch.Tensor], torch.Tensor]


class SharedStencilHead(nn.Module):
    """Ten shared parameters; independent of objective and dimension."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(SEED)
        self.gradient_weight = nn.Parameter(torch.randn(len(H_EXPONENTS), generator=generator))
        self.hessian_weight = nn.Parameter(torch.randn(len(H_EXPONENTS), generator=generator))

    def gradient(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.gradient_weight

    def hessian(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.hessian_weight


def exact_derivatives(state: State) -> tuple[torch.Tensor, torch.Tensor, float]:
    theta = state.theta.detach().clone().requires_grad_(True)
    value = state.objective(theta)
    gradient = torch.autograd.grad(value, theta, create_graph=True)[0]
    hessian = torch.stack([
        torch.autograd.grad(gradient[index], theta, retain_graph=True)[0]
        for index in range(len(theta))
    ])
    return gradient.detach(), (0.5*(hessian+hessian.T)).detach(), float(value.detach())


def stencil_features(state: State) -> tuple[torch.Tensor, torch.Tensor]:
    theta = state.theta.detach()
    dimension = len(theta)
    gradient_scales: list[torch.Tensor] = []
    hessian_scales: list[torch.Tensor] = []
    with torch.no_grad():
        f0 = state.objective(theta)
        for exponent in H_EXPONENTS:
            base = 2.0 ** (-exponent)
            steps = base * torch.maximum(torch.ones_like(theta), torch.abs(theta))
            plus, minus = [], []
            for axis in range(dimension):
                offset = torch.zeros_like(theta)
                offset[axis] = steps[axis]
                plus.append(state.objective(theta+offset))
                minus.append(state.objective(theta-offset))
            gradient = torch.stack([
                (plus[axis]-minus[axis])/(2.0*steps[axis])
                for axis in range(dimension)
            ])
            hessian = torch.empty((dimension, dimension), dtype=torch.float64)
            for axis in range(dimension):
                hessian[axis, axis] = (
                    plus[axis]-2.0*f0+minus[axis]
                )/(steps[axis]*steps[axis])
            for left in range(dimension):
                for right in range(left+1, dimension):
                    oi = torch.zeros_like(theta); oi[left] = steps[left]
                    oj = torch.zeros_like(theta); oj[right] = steps[right]
                    value = (
                        state.objective(theta+oi+oj)-state.objective(theta+oi-oj)
                        -state.objective(theta-oi+oj)+state.objective(theta-oi-oj)
                    )/(4.0*steps[left]*steps[right])
                    hessian[left, right] = hessian[right, left] = value
            gradient_scales.append(gradient)
            hessian_scales.append(hessian)
    # Rows are generic coordinate / symmetric-matrix-entry instances.
    gradient_features = torch.stack(gradient_scales, dim=1)
    indices = [(i, j) for i in range(dimension) for j in range(i, dimension)]
    hessian_features = torch.stack([
        torch.stack([value[i, j] for value in hessian_scales])
        for i, j in indices
    ])
    return gradient_features, hessian_features


def make_states() -> list[State]:
    endpoints = teacher._load_endpoints()
    positions, _, _ = _make_exact_labeled_split(teacher.TRAJECTORY_COUNT, FRESH_TEST_SEED)
    plateau = {item.trajectory_id for item in endpoints if item.previous_plateau_case}
    states: list[State] = []
    train_ids = [i for i in range(256) if i % 4 != 0 and i not in plateau][:64]
    held_ids = sorted(plateau) + [i for i in range(256) if i % 4 == 0 and i not in plateau][:26]
    for split, ids in (("train", train_ids), ("heldout", held_ids)):
        for index in ids:
            endpoint = endpoints[index]
            position = positions[index].detach().clone()
            states.append(State("sho", split, str(index),
                                torch.tensor(endpoint.theta),
                                lambda x, p=position: teacher._scalar_loss(x, p)))
    # Fresh nearby SHO states are grouped separately, not mixed into fitting.
    for serial, index in enumerate(held_ids[:16]):
        endpoint = endpoints[index]
        position = positions[index].detach().clone()
        perturb = torch.tensor((1.0, -1.0))*((serial+1)*1e-7)
        states.append(State("sho", "fresh", f"{index}_fresh",
                            torch.tensor(endpoint.theta)+perturb,
                            lambda x, p=position: teacher._scalar_loss(x, p)))

    with BALL_ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=len(rows), n_timesteps=ball.TOTAL_STEPS,
        dt=ball.DT, seed=20_261_001
    )
    for row, trajectory in zip(rows, trajectories):
        observed = torch.from_numpy(trajectory)
        theta = torch.tensor((float(row["n2_a"]), float(row["n2_b"])))
        states.append(State("ball", "fresh", row["trajectory_id"], theta,
                            lambda x, p=observed: ball.prefix_loss_torch(x, p)))

    # Dimension controls use the same scalar objective semantics with generic
    # positive-definite quadratic plus a small separable quartic term.
    generator = torch.Generator().manual_seed(SEED+1)
    for dimension in range(3, 7):
        for serial in range(6):
            matrix = torch.randn((dimension, dimension), generator=generator)
            matrix = matrix.T@matrix + 0.25*torch.eye(dimension)
            center = torch.randn(dimension, generator=generator)*0.2
            theta = center + torch.randn(dimension, generator=generator)*0.03
            split = "train" if serial < 4 else "fresh"
            def objective(x, a=matrix, c=center):
                delta = x-c
                return 0.5*delta@a@delta + 0.01*torch.sum(delta**4)
            states.append(State("synthetic", split, f"d{dimension}_{serial}",
                                theta, objective))
    return states


def materialize(states: list[State]) -> list[dict]:
    rows = []
    for index, state in enumerate(states):
        gradient, hessian, value = exact_derivatives(state)
        gradient_features, hessian_features = stencil_features(state)
        h_indices = [(i, j) for i in range(len(gradient)) for j in range(i, len(gradient))]
        rows.append(dict(state=state, value=value, gradient=gradient, hessian=hessian,
                         gradient_features=gradient_features,
                         hessian_features=hessian_features,
                         hessian_targets=torch.stack([hessian[i,j] for i,j in h_indices])))
        if (index+1) % 20 == 0:
            print(f"materialized {index+1}/{len(states)}", flush=True)
    return rows


def normalized_rows(features: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.max(torch.abs(features), dim=1).values.clamp_min(1e-300)
    return features/scale[:, None], target/scale


def fit(model: SharedStencilHead, records: list[dict]) -> dict:
    selected = [row for row in records if row["state"].split == "train"]
    gx, gy, hx, hy = [], [], [], []
    for row in selected:
        x, y = normalized_rows(row["gradient_features"], row["gradient"])
        gx.append(x); gy.append(y)
        x, y = normalized_rows(row["hessian_features"], row["hessian_targets"])
        hx.append(x); hy.append(y)
    gx, gy = torch.cat(gx), torch.cat(gy)
    hx, hy = torch.cat(hx), torch.cat(hy)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.5, max_iter=500,
                                  tolerance_grad=1e-15, tolerance_change=1e-16,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = torch.mean((model.gradient(gx)-gy)**2) + torch.mean((model.hessian(hx)-hy)**2)
        loss.backward()
        return loss
    optimizer.step(closure)
    return dict(
        training_rows=len(selected),
        gradient_instances=len(gx), hessian_instances=len(hx),
        initial_weights={k:v.tolist() for k,v in initial.items()},
        final_weights={k:v.detach().tolist() for k,v in model.state_dict().items()},
        parameters=sum(value.numel() for value in model.parameters()),
    )


def predict(model: SharedStencilHead, row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = model.gradient(row["gradient_features"])
    dimension = len(gradient)
    values = model.hessian(row["hessian_features"])
    hessian = torch.empty((dimension, dimension), dtype=torch.float64)
    serial = 0
    for left in range(dimension):
        for right in range(left, dimension):
            hessian[left,right] = hessian[right,left] = values[serial]
            serial += 1
    return gradient.detach(), hessian.detach()


def direct_route(gradient: torch.Tensor, hessian: torch.Tensor) -> tuple[str, torch.Tensor | None]:
    if not bool(torch.isfinite(gradient).all() and torch.isfinite(hessian).all()):
        return "nonfinite", None
    singular = torch.linalg.svdvals(hessian)
    condition = float(torch.max(singular)/torch.min(singular)) if float(torch.min(singular)) > 0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    route = "direct" if float(torch.min(eigenvalues)) > 0 and condition <= teacher.CONDITION_LIMIT else "damped"
    dimension = len(gradient)
    scale = max(float(torch.max(torch.abs(eigenvalues))), torch.finfo(torch.float64).tiny)
    minimum = float(torch.min(eigenvalues))
    damping = 0.0 if route == "direct" else max(scale*teacher.DAMPING_RELATIVE_FLOOR,
                                                  -minimum+scale*teacher.DAMPING_RELATIVE_FLOOR)
    try:
        direction = torch.linalg.solve(hessian+damping*torch.eye(dimension), -gradient)
    except torch.linalg.LinAlgError:
        direction = None
    return route, direction


def evaluate(model: SharedStencilHead, records: list[dict]) -> dict:
    groups: dict[str, dict] = {}
    failures = []
    for row in records:
        state = row["state"]
        gradient, hessian = predict(model, row)
        true_g, true_h = row["gradient"], row["hessian"]
        true_route, true_direction = direct_route(true_g, true_h)
        route, direction = direct_route(gradient, hessian)
        sign_errors = int(torch.count_nonzero(torch.signbit(gradient) != torch.signbit(true_g)))
        route_error = int(route != true_route)
        if direction is None or true_direction is None:
            direction_relative = math.inf if direction is not true_direction else 0.0
        else:
            direction_relative = float(torch.linalg.vector_norm(direction-true_direction) /
                                       max(float(torch.linalg.vector_norm(true_direction)), 1e-300))
        key = f"{state.family}_{state.split}"
        group = groups.setdefault(key, dict(states=0, gradient_values=0, sign_errors=0,
                                             route_errors=0, max_gradient_abs_error=0.0,
                                             max_hessian_abs_error=0.0,
                                             max_direction_relative_error=0.0))
        group["states"] += 1
        group["gradient_values"] += len(true_g)
        group["sign_errors"] += sign_errors
        group["route_errors"] += route_error
        group["max_gradient_abs_error"] = max(group["max_gradient_abs_error"],
                                                float(torch.max(torch.abs(gradient-true_g))))
        group["max_hessian_abs_error"] = max(group["max_hessian_abs_error"],
                                               float(torch.max(torch.abs(hessian-true_h))))
        group["max_direction_relative_error"] = max(group["max_direction_relative_error"],
                                                       direction_relative)
        if sign_errors or route_error or direction_relative > 1e-10:
            failures.append(dict(family=state.family, split=state.split, case_id=state.case_id,
                                 sign_errors=sign_errors, teacher_route=true_route,
                                 learned_route=route,
                                 direction_relative_error=direction_relative,
                                 teacher_gradient=true_g.tolist(), predicted_gradient=gradient.tolist()))
    total_sign = sum(value["sign_errors"] for value in groups.values())
    total_route = sum(value["route_errors"] for value in groups.values())
    max_direction = max(value["max_direction_relative_error"] for value in groups.values())
    # Exact routing is mandatory; a 1e-10 direction screen is already much
    # looser than the final no-regression replay and is used only to avoid an
    # expensive replay for a clearly inadequate representation.
    passed = total_sign == 0 and total_route == 0 and max_direction <= 1e-10
    return dict(passed=passed, groups=groups, total_sign_errors=total_sign,
                total_route_errors=total_route,
                max_direction_relative_error=max_direction,
                failures=failures[:100])


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    states = make_states()
    records = materialize(states)
    model = SharedStencilHead()
    training = fit(model, records)
    validation = evaluate(model, records)
    status = "QUICK_SCREEN_PASSED" if validation["passed"] else "STOPPED_FIRST_COMPONENT_FAILED"
    diagnosis = ""
    if not validation["passed"]:
        diagnosis = (
            "Multiscale objective samples do not preserve enough float64 local information "
            "to recover autograd gradient/Hessian signs and Newton directions across SHO, "
            "bouncing-ball, and higher-dimensional smooth states. The blocker is the "
            "derivative representation/input state, not downstream routing capacity."
        )
    report = dict(
        status=status,
        component="N3 derivative/curvature representation",
        architecture="shared bias-free float64 linear combination of five symmetric-stencil scales",
        h_exponents=H_EXPONENTS,
        training=training,
        validation=validation,
        promoted_checkpoint=None,
        existing_checkpoints_modified=False,
        diagnosis=diagnosis,
        elapsed_seconds=time.time()-started,
    )
    (OUT/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUT/"failures.csv").open("w", newline="") as handle:
        rows = validation["failures"]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
