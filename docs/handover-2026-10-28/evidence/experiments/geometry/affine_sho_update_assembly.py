"""Exact post-baseline affine SHO update and derivative assembly.

This module peels only the fixed-zero bias from the frozen full-K SHO law:

    z_next = K @ z + b

The matrix product, bias addition, and residual/MSE remain separate operations.
No search-domain choice is made here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.geometry.full_k_sho_hessian_assembly import (
    FullKSHOHessianAssembly,
)
from experiments.geometry.prefix_residual_mse_assembly import (
    PrefixResidualMSEAssembly,
)
from experiments.geometry.torch_matmul2_fma_assembly import (
    TorchMatmul2FMAAssembly,
)
from experiments.geometry.torch_mm_backward_dot_assembly import (
    TorchMMBackwardDotAssembly,
)
from experiments.sho import assemble_n3_gradient_operator as arithmetic


torch.set_default_dtype(torch.float64)


@dataclass
class AffineSHOResult:
    prediction_before_bias: torch.Tensor
    prediction: torch.Tensor
    target: torch.Tensor
    residual: torch.Tensor
    square: torch.Tensor
    loss: torch.Tensor
    prediction_adjoint: torch.Tensor
    gradient: torch.Tensor
    raw_hessian: torch.Tensor
    symmetric_hessian: torch.Tensor


class AffineSHOUpdateAssembly:
    """Teacher-order affine forward, gradient, and Hessian for float64 SHO."""

    parameter_order = (
        "K00", "K01", "K10", "K11", "b0", "b1",
    )
    _row_features = (
        (0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2),
    )

    def __init__(self) -> None:
        if arithmetic.OPS is None:
            arithmetic.OPS = arithmetic.LearnedArithmetic()
        self.forward_matmul = TorchMatmul2FMAAssembly()
        self.backward_dot = TorchMMBackwardDotAssembly()
        self.objective = PrefixResidualMSEAssembly()
        self.hessian_support = FullKSHOHessianAssembly()

    @staticmethod
    def _trajectory(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=torch.float64)
        return torch.from_numpy(np.asarray(value, np.float64).copy())

    @staticmethod
    def _theta(value) -> np.ndarray:
        result = np.asarray(value, np.float64)
        if result.shape != (6,):
            raise ValueError(f"affine theta must have shape [6], got {result.shape}")
        return result

    def forward_objective(self, trajectory, theta):
        """Evaluate only the exact affine forward objective.

        N1/N2 call the objective many times and do not consume derivatives.
        This preserves the same recovered operation order as ``evaluate``
        without constructing the adjoint and Hessian products.
        """
        states = self._trajectory(trajectory)
        if states.ndim != 2 or states.shape[1] != 2 or states.shape[0] < 2:
            raise ValueError("trajectory must have shape [T,2] with T>=2")
        parameters = self._theta(theta)
        inputs_np = states[:-1].numpy().copy()
        targets = states[1:]
        matrix = parameters[:4].reshape(2, 2)
        bias = parameters[4:]
        before_bias_np = self.forward_matmul(inputs_np, matrix)
        before_bias = torch.from_numpy(before_bias_np.copy())
        broadcast_bias = torch.from_numpy(
            np.broadcast_to(bias, before_bias_np.shape).copy()
        )
        prediction = arithmetic.OPS.add(before_bias, broadcast_bias)
        return self.objective.evaluate(
            prediction, targets, "full_k_sho", reduction_path="torch",
        )

    def evaluate(self, trajectory, theta) -> AffineSHOResult:
        states = self._trajectory(trajectory)
        if states.ndim != 2 or states.shape[1] != 2 or states.shape[0] < 2:
            raise ValueError("trajectory must have shape [T,2] with T>=2")
        parameters = self._theta(theta)
        inputs_np = states[:-1].numpy().copy()
        targets = states[1:]
        matrix = parameters[:4].reshape(2, 2)
        bias = parameters[4:]

        # Required source order: exact two-term matmul, then a distinct locked
        # coordinatewise addition of the broadcast bias.
        before_bias_np = self.forward_matmul(inputs_np, matrix)
        before_bias = torch.from_numpy(before_bias_np.copy())
        broadcast_bias = torch.from_numpy(
            np.broadcast_to(bias, before_bias_np.shape).copy()
        )
        prediction = arithmetic.OPS.add(before_bias, broadcast_bias)
        objective = self.objective.evaluate(
            prediction, targets, "full_k_sho", reduction_path="torch",
        )

        # The recovered residual/MSE assembly supplies the exact incoming
        # prediction adjoint. Graph traversal is scaffolding; final parameter
        # adjoints are assembled explicitly below.
        prediction_leaf = prediction.detach().clone().requires_grad_(True)
        leaf_objective = self.objective.evaluate(
            prediction_leaf, targets, "full_k_sho", reduction_path="torch",
        )
        incoming = torch.autograd.grad(
            leaf_objective.loss, prediction_leaf,
        )[0].detach()

        gradient_k = self.backward_dot(
            incoming.numpy().T, inputs_np,
        ).reshape(-1)
        # AddBackward's broadcast reduction is retained as the exact validated
        # ATen implementation-order scaffold. It introduces no parameter or
        # learned operation and exactly matches current Torch 2.2.2 semantics.
        gradient_bias = incoming.sum(dim=0).numpy()
        gradient = np.concatenate((gradient_k, gradient_bias))

        transition_count = len(inputs_np)
        second_scale = self.hessian_support.second_adjoint_scale(
            transition_count
        )
        features = np.column_stack((
            inputs_np, np.ones(transition_count, dtype=np.float64),
        ))
        rows = []
        for output_index, feature_index in self._row_features:
            propagated = np.zeros(
                (transition_count, 2), dtype=np.float64,
            )
            propagated[:, output_index] = features[:, feature_index]
            scaled = self.hessian_support._multiply(
                propagated, second_scale,
            )
            row_k = self.backward_dot(scaled.T, inputs_np).reshape(-1)
            row_bias = torch.from_numpy(scaled).sum(dim=0).numpy()
            rows.append(np.concatenate((row_k, row_bias)))
        raw = np.stack(rows)
        summed = self.hessian_support.forward._add(raw, raw.T)
        symmetric = self.hessian_support._multiply(np.float64(0.5), summed)

        return AffineSHOResult(
            prediction_before_bias=before_bias,
            prediction=prediction.detach(), target=targets.detach(),
            residual=objective.residual.detach(),
            square=objective.square.detach(), loss=objective.loss.detach(),
            prediction_adjoint=incoming,
            gradient=torch.from_numpy(gradient.copy()),
            raw_hessian=torch.from_numpy(raw.copy()),
            symmetric_hessian=torch.from_numpy(symmetric.copy()),
        )
