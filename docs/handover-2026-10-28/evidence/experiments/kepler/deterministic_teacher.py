#!/usr/bin/env python3
"""Float64 planar two-body teacher and exactly matched Torch adapter.

State order is (x1, y1, vx1, vy1, x2, y2, vx2, vy2).  The integrator is
velocity Verlet.  Masses and G are observed physical metadata; theta=(gamma,p)
controls the shared force law

    a1 = +gamma*G*m2*(r2-r1)/||r2-r1||**p
    a2 = -gamma*G*m1*(r2-r1)/||r2-r1||**p.

The Newtonian teacher is theta=(1,3), whose acceleration magnitude is inverse
square.  No softening is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

DT = np.float64(0.02)
G = np.float64(1.0)
MASS_1 = np.float64(1.0)
MASS_2 = np.float64(0.25)
TEACHER_THETA = np.asarray((1.0, 3.0), dtype=np.float64)
LOWER = np.asarray((0.25, 2.0), dtype=np.float64)
UPPER = np.asarray((2.0, 4.0), dtype=np.float64)
PREFIX_STATES = 128
TOTAL_STEPS = 1024


@dataclass(frozen=True)
class System:
    mass_1: float = float(MASS_1)
    mass_2: float = float(MASS_2)
    gravitational_constant: float = float(G)
    dt: float = float(DT)


DEFAULT_SYSTEM = System()


def initial_state(eccentricity: float = 0.35, periapsis: float = 1.0,
                  orientation: float = 0.0, system: System = DEFAULT_SYSTEM) -> np.ndarray:
    """Barycentric periapsis state for a bound eccentric orbit."""
    m1, m2 = np.float64(system.mass_1), np.float64(system.mass_2)
    total = np.float64(m1 + m2)
    e, rp, angle = np.float64(eccentricity), np.float64(periapsis), np.float64(orientation)
    relative_speed = np.sqrt(np.float64(system.gravitational_constant) * total * (1.0 + e) / rp)
    radial = np.asarray((np.cos(angle), np.sin(angle)), dtype=np.float64) * rp
    tangent = np.asarray((-np.sin(angle), np.cos(angle)), dtype=np.float64) * relative_speed
    r1 = -(m2 / total) * radial
    r2 = +(m1 / total) * radial
    v1 = -(m2 / total) * tangent
    v2 = +(m1 / total) * tangent
    return np.asarray((r1[0], r1[1], v1[0], v1[1],
                       r2[0], r2[1], v2[0], v2[1]), dtype=np.float64)


def numpy_acceleration(state: np.ndarray, theta: np.ndarray,
                       system: System = DEFAULT_SYSTEM) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64)
    strength, exponent = np.asarray(theta, dtype=np.float64)
    dx = np.float64(s[4] - s[0]); dy = np.float64(s[5] - s[1])
    radius_squared = np.float64(dx * dx + dy * dy)
    if not radius_squared > 0.0:
        raise FloatingPointError("two-body collision has no unsoftened acceleration")
    radius = np.sqrt(radius_squared)
    denominator = np.power(radius, exponent)
    common = np.float64(np.float64(system.gravitational_constant) * strength / denominator)
    scale_1 = np.float64(common * np.float64(system.mass_2))
    scale_2 = np.float64(common * np.float64(system.mass_1))
    return np.asarray((scale_1 * dx, scale_1 * dy,
                       -scale_2 * dx, -scale_2 * dy), dtype=np.float64)


def numpy_step(state: np.ndarray, theta: np.ndarray = TEACHER_THETA,
               system: System = DEFAULT_SYSTEM) -> np.ndarray:
    s = np.asarray(state, dtype=np.float64); dt = np.float64(system.dt)
    half_dt = np.float64(0.5 * dt)
    acceleration = numpy_acceleration(s, theta, system)
    v1hx = np.float64(s[2] + half_dt * acceleration[0])
    v1hy = np.float64(s[3] + half_dt * acceleration[1])
    v2hx = np.float64(s[6] + half_dt * acceleration[2])
    v2hy = np.float64(s[7] + half_dt * acceleration[3])
    x1 = np.float64(s[0] + dt * v1hx); y1 = np.float64(s[1] + dt * v1hy)
    x2 = np.float64(s[4] + dt * v2hx); y2 = np.float64(s[5] + dt * v2hy)
    drifted = np.asarray((x1, y1, v1hx, v1hy, x2, y2, v2hx, v2hy), dtype=np.float64)
    next_acceleration = numpy_acceleration(drifted, theta, system)
    v1x = np.float64(v1hx + half_dt * next_acceleration[0])
    v1y = np.float64(v1hy + half_dt * next_acceleration[1])
    v2x = np.float64(v2hx + half_dt * next_acceleration[2])
    v2y = np.float64(v2hy + half_dt * next_acceleration[3])
    return np.asarray((x1, y1, v1x, v1y, x2, y2, v2x, v2y), dtype=np.float64)


def numpy_rollout(initial: np.ndarray, steps: int = TOTAL_STEPS,
                  theta: np.ndarray = TEACHER_THETA,
                  system: System = DEFAULT_SYSTEM) -> np.ndarray:
    output = np.empty((steps + 1, 8), dtype=np.float64); output[0] = initial
    for index in range(steps):
        output[index + 1] = numpy_step(output[index], theta, system)
    return output


def torch_acceleration(state: torch.Tensor, theta: torch.Tensor,
                       system: System = DEFAULT_SYSTEM) -> torch.Tensor:
    strength, exponent = theta.unbind()
    dx = state[4] - state[0]; dy = state[5] - state[1]
    radius_squared = dx * dx + dy * dy
    radius = torch.sqrt(radius_squared)
    denominator = torch.pow(radius, exponent)
    common = torch.as_tensor(system.gravitational_constant) * strength / denominator
    scale_1 = common * torch.as_tensor(system.mass_2)
    scale_2 = common * torch.as_tensor(system.mass_1)
    return torch.stack((scale_1 * dx, scale_1 * dy, -scale_2 * dx, -scale_2 * dy))


def torch_step(state: torch.Tensor, theta: torch.Tensor,
               system: System = DEFAULT_SYSTEM) -> torch.Tensor:
    dt = torch.as_tensor(system.dt); half_dt = 0.5 * dt
    acceleration = torch_acceleration(state, theta, system)
    v1hx = state[2] + half_dt * acceleration[0]
    v1hy = state[3] + half_dt * acceleration[1]
    v2hx = state[6] + half_dt * acceleration[2]
    v2hy = state[7] + half_dt * acceleration[3]
    x1 = state[0] + dt * v1hx; y1 = state[1] + dt * v1hy
    x2 = state[4] + dt * v2hx; y2 = state[5] + dt * v2hy
    drifted = torch.stack((x1, y1, v1hx, v1hy, x2, y2, v2hx, v2hy))
    next_acceleration = torch_acceleration(drifted, theta, system)
    return torch.stack((x1, y1,
                        v1hx + half_dt * next_acceleration[0],
                        v1hy + half_dt * next_acceleration[1],
                        x2, y2,
                        v2hx + half_dt * next_acceleration[2],
                        v2hy + half_dt * next_acceleration[3]))


def torch_rollout(initial: torch.Tensor, steps: int = TOTAL_STEPS,
                  theta: torch.Tensor | None = None,
                  system: System = DEFAULT_SYSTEM) -> torch.Tensor:
    if theta is None:
        theta = torch.from_numpy(TEACHER_THETA)
    values = [initial]
    for _ in range(steps):
        values.append(torch_step(values[-1], theta, system))
    return torch.stack(values)


def prefix_loss_numpy(theta: np.ndarray, trajectory: np.ndarray,
                      prefix_states: int = PREFIX_STATES,
                      system: System = DEFAULT_SYSTEM) -> np.ndarray:
    candidates = np.atleast_2d(theta).astype(np.float64, copy=False)
    prefix = np.asarray(trajectory[:prefix_states], dtype=np.float64)
    output = np.empty(len(candidates), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        squared = np.float64(0.0)
        for index in range(prefix_states - 1):
            error = numpy_step(prefix[index], candidate, system) - prefix[index + 1]
            squared = np.float64(squared + np.dot(error, error))
        output[candidate_index] = squared / np.float64((prefix_states - 1) * 8)
    return output


def prefix_loss_torch(theta: torch.Tensor, trajectory: torch.Tensor,
                      prefix_states: int = PREFIX_STATES,
                      system: System = DEFAULT_SYSTEM) -> torch.Tensor:
    prefix = trajectory[:prefix_states]
    errors = [torch_step(prefix[index], theta, system) - prefix[index + 1]
              for index in range(prefix_states - 1)]
    return torch.stack(errors).square().mean()

