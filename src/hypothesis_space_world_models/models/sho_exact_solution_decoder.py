"""Stage 0 SHO model with a learned encoder and fixed analytic decoder."""

import torch
from torch import nn
from torch.nn import functional as F


def _make_activation(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unsupported activation: {name}")


def sho_target_weight_matrix(omega: float) -> torch.Tensor:
    """Return the exact fixed-omega SHO coefficient map."""
    if omega <= 0:
        raise ValueError("omega must be positive")
    return torch.tensor([[1.0, 0.0], [0.0, 1.0 / omega]])


def _normalize_hidden_dims(hidden_dims: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(hidden_dims, int):
        return [hidden_dims, hidden_dims]
    return list(hidden_dims)


def _make_mlp_encoder(
    hidden_dims: int | list[int] | tuple[int, ...],
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = 2
    for hidden_dim in _normalize_hidden_dims(hidden_dims):
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(_make_activation(activation))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 2))
    return nn.Sequential(*layers)


class SHOExactSolutionDecoder(nn.Module):
    """Fixed decoder for x(t) = a cos(omega t) + b sin(omega t)."""

    def __init__(self, omega: float) -> None:
        super().__init__()
        if omega <= 0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)

    def forward(self, coefficients: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = self.omega * times
        return a * torch.cos(phases) + b * torch.sin(phases)


def _inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value))))


class SHOExactSolutionModel(nn.Module):
    """Learn initial-condition-to-coefficient map, then decode analytically."""

    def __init__(
        self,
        omega: float,
        hidden_dim: int = 32,
        activation: str = "silu",
        hidden_dims: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.hidden_dims = _normalize_hidden_dims(hidden_dims or hidden_dim)
        self.hidden_dim = hidden_dim
        self.encoder = _make_mlp_encoder(self.hidden_dims, activation)
        self.decoder = SHOExactSolutionDecoder(omega)

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.encoder(initial_conditions)
        trajectories = self.decoder(coefficients, times)
        return coefficients, trajectories


class SHOTrainableLinearResidualModel(nn.Module):
    """Learn W x plus a nonlinear residual, then decode analytically."""

    def __init__(
        self,
        omega: float,
        hidden_dim: int = 32,
        activation: str = "silu",
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.hidden_dims = _normalize_hidden_dims(hidden_dims or hidden_dim)
        self.hidden_dim = hidden_dim
        self.residual_scale = float(residual_scale)
        self.linear_encoder = nn.Linear(2, 2, bias=False)
        self.residual_encoder = _make_mlp_encoder(self.hidden_dims, activation)
        self.decoder = SHOExactSolutionDecoder(omega)

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = (
            self.linear_encoder(initial_conditions)
            + self.residual(initial_conditions)
        )
        trajectories = self.decoder(coefficients, times)
        return coefficients, trajectories

    def residual(self, initial_conditions: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.residual_encoder(initial_conditions)


class SHOFixedLinearResidualModel(nn.Module):
    """Use the exact W x path plus a learned nonlinear residual."""

    def __init__(
        self,
        omega: float,
        hidden_dim: int = 32,
        activation: str = "silu",
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.hidden_dims = _normalize_hidden_dims(hidden_dims or hidden_dim)
        self.hidden_dim = hidden_dim
        self.residual_scale = float(residual_scale)
        self.register_buffer("exact_weight", sho_target_weight_matrix(omega))
        self.residual_encoder = _make_mlp_encoder(self.hidden_dims, activation)
        self.decoder = SHOExactSolutionDecoder(omega)

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exact_coefficients = initial_conditions @ self.exact_weight.T
        coefficients = exact_coefficients + self.residual(initial_conditions)
        trajectories = self.decoder(coefficients, times)
        return coefficients, trajectories

    def residual(self, initial_conditions: torch.Tensor) -> torch.Tensor:
        return self.residual_scale * self.residual_encoder(initial_conditions)


class SHOLinearExactSolutionModel(nn.Module):
    """Learn a linear coefficient map, then decode analytically."""

    def __init__(self, omega: float) -> None:
        super().__init__()
        self.encoder = nn.Linear(2, 2, bias=False)
        self.decoder = SHOExactSolutionDecoder(omega)

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.encoder(initial_conditions)
        trajectories = self.decoder(coefficients, times)
        return coefficients, trajectories


class SHOLearnedGlobalOmegaSolutionModel(nn.Module):
    """Learn a linear coefficient map and one global SHO frequency."""

    def __init__(self, initial_omega_hat: float = 1.0) -> None:
        super().__init__()
        if initial_omega_hat <= 0:
            raise ValueError("initial_omega_hat must be positive")
        self.encoder = nn.Linear(2, 2, bias=False)
        self.raw_omega = nn.Parameter(_inverse_softplus(initial_omega_hat - 1e-8))

    @property
    def omega_hat(self) -> torch.Tensor:
        return F.softplus(self.raw_omega) + 1e-8

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.encoder(initial_conditions)
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = self.omega_hat * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories
