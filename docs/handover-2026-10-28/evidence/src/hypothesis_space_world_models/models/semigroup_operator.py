"""Linear semigroup operators for continuous-time dynamics."""

import torch
from torch import nn


class LinearSemigroupOperator(nn.Module):
    """Continuous-time linear semigroup x(t) = exp(t A) x(0)."""

    def __init__(
        self,
        state_dim: int,
        generator: torch.Tensor | None = None,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        self.state_dim = int(state_dim)

        if generator is None:
            generator = torch.zeros(self.state_dim, self.state_dim)
        if tuple(generator.shape) != (self.state_dim, self.state_dim):
            raise ValueError(
                "generator must have shape "
                f"({self.state_dim}, {self.state_dim})",
            )

        generator = generator.detach().clone()
        if trainable:
            self.generator = nn.Parameter(generator)
        else:
            self.register_buffer("generator", generator)

    def transition_matrix(self, times: torch.Tensor) -> torch.Tensor:
        """Return exp(t A) for each requested time."""
        times = times.to(device=self.generator.device, dtype=self.generator.dtype)
        scaled_generators = times.unsqueeze(-1).unsqueeze(-1) * self.generator
        return torch.matrix_exp(scaled_generators)

    def forward(self, initial_states: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Roll initial states forward at each time.

        Args:
            initial_states: Tensor with shape ``(..., state_dim)``.
            times: One-dimensional tensor of sample times.

        Returns:
            Tensor with shape ``(..., num_times, state_dim)``.
        """
        if initial_states.shape[-1] != self.state_dim:
            raise ValueError(f"initial_states last dimension must be {self.state_dim}")
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")

        initial_states = initial_states.to(
            device=self.generator.device,
            dtype=self.generator.dtype,
        )
        transitions = self.transition_matrix(times)
        return torch.einsum("tij,...j->...ti", transitions, initial_states)


class SHOSemigroupOperator(LinearSemigroupOperator):
    """Exact SHO state-transition semigroup for state (x, v)."""

    def __init__(self, omega: float, *, trainable: bool = False) -> None:
        if omega <= 0:
            raise ValueError("omega must be positive")
        self.omega = float(omega)
        generator = torch.tensor(
            [[0.0, 1.0], [-(self.omega**2), 0.0]],
            dtype=torch.float32,
        )
        super().__init__(state_dim=2, generator=generator, trainable=trainable)
