"""Simple harmonic oscillator data generation."""

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class SHOData:
    """Synthetic trajectories and ground-truth SHO coefficients."""

    initial_conditions: torch.Tensor
    times: torch.Tensor
    coefficients: torch.Tensor
    trajectories: torch.Tensor
    omega: float
    period: float
    num_periods: float


def coefficients_from_initial_conditions(
    initial_conditions: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Map initial conditions (x0, v0) to x(t) = a cos(wt) + b sin(wt)."""
    x0 = initial_conditions[..., 0]
    v0 = initial_conditions[..., 1]
    return torch.stack((x0, v0 / omega), dim=-1)


def decode_sho(coefficients: torch.Tensor, times: torch.Tensor, omega: float) -> torch.Tensor:
    """Construct SHO trajectories from coefficients and sample times."""
    a = coefficients[..., 0].unsqueeze(-1)
    b = coefficients[..., 1].unsqueeze(-1)
    basis = omega * times
    return a * torch.cos(basis) + b * torch.sin(basis)


def decode_sho_velocity(
    coefficients: torch.Tensor,
    times: torch.Tensor,
    omega: float,
) -> torch.Tensor:
    """Construct SHO velocities from coefficients and sample times."""
    a = coefficients[..., 0].unsqueeze(-1)
    b = coefficients[..., 1].unsqueeze(-1)
    basis = omega * times
    return -a * omega * torch.sin(basis) + b * omega * torch.cos(basis)


def decode_sho_phase_space(
    coefficients: torch.Tensor,
    times: torch.Tensor,
    omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct SHO position and velocity trajectories."""
    positions = decode_sho(coefficients, times, omega)
    velocities = decode_sho_velocity(coefficients, times, omega)
    return positions, velocities


def generate_sho_data(
    num_trajectories: int,
    num_time_steps: int,
    omega: float = 1.0,
    num_periods: float = 10.0,
    t_start: float = 0.0,
    initial_condition_scale: float = 1.0,
    seed: Optional[int] = None,
    device: torch.device | str = "cpu",
) -> SHOData:
    """Generate noise-free SHO trajectories from random initial conditions."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    if num_trajectories <= 0:
        raise ValueError("num_trajectories must be positive")
    if num_time_steps <= 1:
        raise ValueError("num_time_steps must be greater than 1")
    if num_periods <= 0:
        raise ValueError("num_periods must be positive")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    period = math.tau / omega
    t_max = t_start + num_periods * period
    times = torch.linspace(t_start, t_max, num_time_steps, device=device)
    initial_conditions = (
        2.0
        * initial_condition_scale
        * torch.rand(num_trajectories, 2, generator=generator, device=device)
        - initial_condition_scale
    )
    coefficients = coefficients_from_initial_conditions(initial_conditions, omega)
    trajectories = decode_sho(coefficients, times, omega)

    return SHOData(
        initial_conditions=initial_conditions,
        times=times,
        coefficients=coefficients,
        trajectories=trajectories,
        omega=omega,
        period=period,
        num_periods=num_periods,
    )
