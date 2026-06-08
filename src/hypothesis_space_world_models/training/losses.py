"""Loss functions for recovery experiments."""

import torch
from torch.nn import functional as F


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    return F.mse_loss(prediction, target)
