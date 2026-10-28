"""Opt-in residual-Jacobian N3 direction for the generic SHO network.

This module is deliberately separate from the production recovery runner.  Its
default is the existing full-Hessian direction.  Selecting
``stabilized_residual`` changes only the direction returned after the existing
N3 derivative, curvature, and damping path has selected ``mu``.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from experiments.sho.resume_generic_update_targeted import Runtime


EXISTING_FULL_HESSIAN = "existing_full_hessian"
STABILIZED_RESIDUAL = "stabilized_residual"
DEFAULT_DIRECTION_MODE = EXISTING_FULL_HESSIAN


@dataclass
class DirectionBindingCounters:
    derivative_calls: int = 0
    existing_direction_outputs: int = 0
    stabilized_direction_outputs: int = 0


def residual_vector(adapter, trajectory: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    return (adapter.forward(theta, trajectory[:-1]) - trajectory[1:]).reshape(-1)


def residual_and_jacobian(adapter, trajectory: torch.Tensor, theta: torch.Tensor):
    residual = residual_vector(adapter, trajectory, theta).detach()
    jacobian = torch.autograd.functional.jacobian(
        lambda value: residual_vector(adapter, trajectory, value), theta
    )
    return residual, jacobian


def _bounded_residual_least_squares(
    matrix: torch.Tensor,
    residual: torch.Tensor,
    theta: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
):
    """Exact-active-set BVLS used by the validated isolated trajectory.

    All singular values participate.  Bounds are exact; there is no proximity
    tolerance or rank cutoff.
    """
    size = theta.numel()
    step_lower = lower - theta
    step_upper = upper - theta
    model_gradient = matrix.T @ residual
    direction = torch.zeros_like(theta)
    status = torch.zeros(size, dtype=torch.int64)
    status[(step_lower == 0) & (model_gradient >= 0)] = -1
    status[(step_upper == 0) & (model_gradient <= 0)] = 1
    seen = set()
    iterations = 0

    while True:
        key = (tuple(status.tolist()), direction.numpy().tobytes())
        if key in seen:
            raise RuntimeError("exact active-set cycle")
        seen.add(key)
        iterations += 1

        lower_active = torch.where(status == -1)[0]
        upper_active = torch.where(status == 1)[0]
        free = torch.where(status == 0)[0]
        active = torch.cat((lower_active, upper_active))
        trial = torch.zeros_like(theta)
        trial[lower_active] = step_lower[lower_active]
        trial[upper_active] = step_upper[upper_active]

        if len(free):
            u, singular, vh = torch.linalg.svd(matrix[:, free], full_matrices=False)
            if bool((singular <= 0).any()):
                raise RuntimeError("nonpositive singular value; no truncation authorized")
            active_residual = residual + matrix[:, active] @ trial[active]
            trial[free] = vh.T @ ((u.T @ (-active_residual)) / singular)

        delta = trial - direction
        blockers = []
        for coordinate in free.tolist():
            if trial[coordinate] < step_lower[coordinate]:
                blockers.append(
                    (float((step_lower[coordinate] - direction[coordinate]) / delta[coordinate]), coordinate, -1)
                )
            elif trial[coordinate] > step_upper[coordinate]:
                blockers.append(
                    (float((step_upper[coordinate] - direction[coordinate]) / delta[coordinate]), coordinate, 1)
                )
        if blockers:
            alpha = min(item[0] for item in blockers)
            direction = direction + alpha * delta
            for value, coordinate, sign in blockers:
                if value == alpha:
                    status[coordinate] = sign
                    direction[coordinate] = (
                        step_lower[coordinate] if sign == -1 else step_upper[coordinate]
                    )
            continue

        direction = trial
        projected_model_gradient = matrix.T @ (residual + matrix @ direction)
        violations = []
        for coordinate in lower_active.tolist():
            if projected_model_gradient[coordinate] < 0:
                violations.append((-float(projected_model_gradient[coordinate]), coordinate))
        for coordinate in upper_active.tolist():
            if projected_model_gradient[coordinate] > 0:
                violations.append((float(projected_model_gradient[coordinate]), coordinate))
        if violations:
            _, coordinate = min(violations, key=lambda item: (-item[0], item[1]))
            status[coordinate] = 0
            continue
        if not bool(((direction >= step_lower) & (direction <= step_upper)).all()):
            raise RuntimeError("infeasible stabilized residual direction")
        return direction, {
            "iterations": iterations,
            "active": status.tolist(),
            "free_model_gradient_norm": float(projected_model_gradient[free].norm()),
        }


class OptInResidualN3Runtime(Runtime):
    """Existing generic N3 runtime with one explicitly selected direction hook."""

    def __init__(self, adapter, trajectory, log, *, direction_mode=DEFAULT_DIRECTION_MODE):
        if direction_mode not in (EXISTING_FULL_HESSIAN, STABILIZED_RESIDUAL):
            raise ValueError(f"unknown direction mode: {direction_mode}")
        super().__init__(adapter, trajectory, log)
        self.direction_mode = direction_mode
        self.binding_counters = DirectionBindingCounters()
        self.last_residual_state = None

    def derivatives_and_direction(self, theta, count_route=True):
        self.binding_counters.derivative_calls += 1
        values = super().derivatives_and_direction(theta, count_route=count_route)
        result, gradient, hessian, gradient_norm, pre, route, direction, qp = values
        if self.direction_mode == EXISTING_FULL_HESSIAN or pre != "CONTINUE":
            self.binding_counters.existing_direction_outputs += int(pre == "CONTINUE")
            self.last_residual_state = None
            return values

        residual, jacobian = residual_and_jacobian(self.adapter, self.trajectory, theta)
        mu = float(qp["damping"])
        residual_count = residual.numel()
        regularization = torch.tensor(residual_count / 2 * mu, dtype=theta.dtype)
        augmented_matrix = torch.cat(
            (jacobian, torch.sqrt(regularization) * torch.eye(theta.numel(), dtype=theta.dtype))
        )
        augmented_residual = torch.cat((residual, torch.zeros_like(theta)))
        stabilized_direction, active_set = _bounded_residual_least_squares(
            augmented_matrix,
            augmented_residual,
            theta,
            self.lower,
            self.upper,
        )
        jd = jacobian @ stabilized_direction
        residual_predicted_decrease = float(
            -(2 * residual @ jd + jd @ jd) / residual_count
        )
        full_predicted_decrease = residual_predicted_decrease - mu / 2 * float(
            stabilized_direction @ stabilized_direction
        )
        stabilized_qp = {
            **active_set,
            "route_status": "ok",
            "damping": mu,
            "quadratic_value": -full_predicted_decrease,
        }
        self.binding_counters.stabilized_direction_outputs += 1
        self.last_residual_state = {
            "residual": residual,
            "jacobian": jacobian,
            "regularization": float(regularization),
            "residual_predicted_decrease": residual_predicted_decrease,
            "full_predicted_decrease": full_predicted_decrease,
            "existing_direction": direction,
            "existing_qp": qp,
        }
        return (
            result,
            gradient,
            hessian,
            gradient_norm,
            pre,
            route,
            stabilized_direction,
            stabilized_qp,
        )

