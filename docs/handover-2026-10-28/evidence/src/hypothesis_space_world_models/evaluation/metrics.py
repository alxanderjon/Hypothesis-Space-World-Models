"""Metrics for Stage 0 SHO recovery."""

import torch


def mean_squared_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return scalar MSE as a Python float."""
    return torch.mean((prediction - target) ** 2).item()
