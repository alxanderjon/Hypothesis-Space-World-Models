import torch
import math

from hypothesis_space_world_models.systems.sho import (
    coefficients_from_initial_conditions,
    decode_sho,
    decode_sho_phase_space,
    decode_sho_velocity,
    generate_sho_data,
)


def test_coefficients_from_initial_conditions() -> None:
    initial_conditions = torch.tensor([[1.0, 4.0], [-2.0, 6.0]])

    coefficients = coefficients_from_initial_conditions(initial_conditions, omega=2.0)

    assert torch.allclose(coefficients, torch.tensor([[1.0, 2.0], [-2.0, 3.0]]))


def test_generate_sho_data_shapes() -> None:
    data = generate_sho_data(
        num_trajectories=8,
        num_time_steps=21,
        omega=2.0,
        seed=0,
    )

    assert data.initial_conditions.shape == (8, 2)
    assert data.coefficients.shape == (8, 2)
    assert data.times.shape == (21,)
    assert data.trajectories.shape == (8, 21)


def test_generate_sho_data_uses_number_of_periods() -> None:
    data = generate_sho_data(
        num_trajectories=2,
        num_time_steps=11,
        omega=2.0,
        num_periods=3.0,
        seed=0,
    )

    expected_period = math.tau / 2.0
    assert data.period == expected_period
    assert data.num_periods == 3.0
    assert torch.allclose(data.times[-1], torch.tensor(3.0 * expected_period))


def test_decode_sho_matches_known_initial_state() -> None:
    times = torch.tensor([0.0])
    coefficients = torch.tensor([[3.0, 5.0]])

    trajectories = decode_sho(coefficients, times, omega=2.0)

    assert torch.allclose(trajectories, torch.tensor([[3.0]]))


def test_decode_sho_velocity_matches_known_initial_state() -> None:
    times = torch.tensor([0.0])
    coefficients = torch.tensor([[3.0, 5.0]])

    velocities = decode_sho_velocity(coefficients, times, omega=2.0)

    assert torch.allclose(velocities, torch.tensor([[10.0]]))


def test_decode_sho_phase_space_returns_position_and_velocity() -> None:
    times = torch.tensor([0.0])
    coefficients = torch.tensor([[3.0, 5.0]])

    positions, velocities = decode_sho_phase_space(coefficients, times, omega=2.0)

    assert torch.allclose(positions, torch.tensor([[3.0]]))
    assert torch.allclose(velocities, torch.tensor([[10.0]]))
