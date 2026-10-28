"""Dimension-general deterministic box-QP direction for safeguarded N3.

This module contains no learned state and no convergence tolerance.  It keeps
the existing N3 curvature/damping choice and solves the resulting convex box
quadratic by a primal active-set method.
"""
from __future__ import annotations

import torch


def _face(status: torch.Tensor) -> list[int]:
    return [int(value) for value in status.tolist()]


def dimension_general_box_qp_direction(theta, gradient, hessian, lower, upper,
                                       direction_fn):
    """Return the exact-bound active-set solution and deterministic diagnostics."""
    unconstrained, condition, damping, route_status = direction_fn(
        gradient, hessian)
    info = {
        "condition": float(condition), "damping": float(damping),
        "route_status": route_status,
        "used_unconstrained_feasible_shortcut": False,
        "active_set_iterations": 0, "face_solves": 0,
        "blocking_events": 0, "release_events": 0,
        "winning_face_status": None, "quadratic_value": None,
        "face_history": [],
    }
    if unconstrained is None:
        return None, info

    dimension = int(gradient.numel())
    lower = torch.as_tensor(lower, dtype=gradient.dtype,
                            device=gradient.device).expand_as(theta)
    upper = torch.as_tensor(upper, dtype=gradient.dtype,
                            device=gradient.device).expand_as(theta)
    step_lower = lower - theta
    step_upper = upper - theta
    identity = torch.eye(dimension, dtype=gradient.dtype,
                         device=gradient.device)
    matrix = hessian + damping * identity

    def quadratic_value(direction):
        return torch.dot(gradient, direction) + 0.5 * torch.dot(
            direction, matrix @ direction)

    # This shortcut is deliberately first and returns the pre-existing tensor
    # unchanged, preserving feasible unconstrained N3 bit-for-bit.
    if bool(torch.all(unconstrained >= step_lower) and
            torch.all(unconstrained <= step_upper)):
        info["used_unconstrained_feasible_shortcut"] = True
        info["winning_face_status"] = [0] * dimension
        info["quadratic_value"] = float(quadratic_value(unconstrained))
        return unconstrained, info

    direction = torch.zeros_like(gradient)
    status = torch.zeros(dimension, dtype=torch.int8, device=gradient.device)
    status[(step_lower == 0.0) & (gradient >= 0.0)] = -1
    status[(step_upper == 0.0) & (gradient <= 0.0)] = 1

    # There is no numerical or iteration tolerance.  Exact repetition is a
    # structural cycle/error path rather than an approximate convergence rule.
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    while True:
        info["active_set_iterations"] += 1
        key = (tuple(_face(status)), direction.detach().cpu().numpy().tobytes())
        if key in seen:
            info["route_status"] = "active_set_cycle"
            return None, info
        seen.add(key)
        info["face_history"].append(_face(status))

        lower_indices = torch.nonzero(status == -1).flatten()
        upper_indices = torch.nonzero(status == 1).flatten()
        free = torch.nonzero(status == 0).flatten()
        trial = torch.zeros_like(direction)
        if lower_indices.numel():
            trial[lower_indices] = step_lower[lower_indices]
        if upper_indices.numel():
            trial[upper_indices] = step_upper[upper_indices]
        if free.numel():
            # Lower-then-upper order matches the validated d=5 enumeration
            # oracle's reduced right-hand-side evaluation order.
            fixed = torch.cat((lower_indices, upper_indices))
            rhs = -gradient[free]
            if fixed.numel():
                rhs = rhs - matrix[free][:, fixed] @ trial[fixed]
            try:
                trial[free] = torch.linalg.solve(matrix[free][:, free], rhs)
            except torch.linalg.LinAlgError:
                info["route_status"] = "singular_reduced_system"
                return None, info
        info["face_solves"] += 1
        if not bool(torch.isfinite(trial).all()):
            info["route_status"] = "nonfinite_direction"
            return None, info

        delta = trial - direction
        blockers: list[tuple[float, int, int]] = []
        for index in free.tolist():
            if trial[index] < step_lower[index]:
                alpha = float((step_lower[index] - direction[index]) /
                              delta[index])
                blockers.append((alpha, index, -1))
            elif trial[index] > step_upper[index]:
                alpha = float((step_upper[index] - direction[index]) /
                              delta[index])
                blockers.append((alpha, index, 1))
        if blockers:
            alpha = min(item[0] for item in blockers)
            direction = direction + alpha * delta
            simultaneous = sorted((index, bound_status)
                                  for value, index, bound_status in blockers
                                  if value == alpha)
            for index, bound_status in simultaneous:
                status[index] = bound_status
                direction[index] = (step_lower[index] if bound_status == -1
                                    else step_upper[index])
            info["blocking_events"] += len(simultaneous)
            continue

        direction = trial
        reduced_gradient = gradient + matrix @ direction
        violations: list[tuple[float, int]] = []
        for index in lower_indices.tolist():
            if reduced_gradient[index] < 0.0:
                violations.append((-float(reduced_gradient[index]), index))
        for index in upper_indices.tolist():
            if reduced_gradient[index] > 0.0:
                violations.append((float(reduced_gradient[index]), index))
        if violations:
            _, index = min(violations, key=lambda item: (-item[0], item[1]))
            status[index] = 0
            info["release_events"] += 1
            continue

        info["winning_face_status"] = _face(status)
        info["quadratic_value"] = float(quadratic_value(direction))
        if (not bool(torch.isfinite(direction).all()) or
                float(torch.dot(gradient, direction)) >= 0.0):
            info["route_status"] = "no_descent_direction"
            return None, info
        return direction, info
