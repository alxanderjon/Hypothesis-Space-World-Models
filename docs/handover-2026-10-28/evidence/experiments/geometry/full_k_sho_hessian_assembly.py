"""Exact current full-K SHO Hessian from recovered Torch kernel assemblies.

This module replays the production second reverse pass for

    mean((X @ K.T - target)**2)

with runtime ``X`` shape ``[N,2]`` and free ``K`` shaped ``[2,2]``.  Each Hessian
row executes the same operations as Torch autograd:

1. propagate the one-hot row seed through the first ``MmBackward`` using the
   recovered ``[N,2] @ [2,2]`` forward matrix-product assembly;
2. multiply by the production square/mean second-adjoint scale;
3. propagate through the second ``MmBackward`` using the recovered ascending
   FMA dot assembly;
4. stack rows and perform production ``0.5 * (raw + raw.T)`` symmetrization.

All arithmetic is supplied by locked primitives and exact assemblies.  Row
iteration, one-hot construction, views, stacking, and transpose ownership are
deterministic implementation-order scaffolding.
"""
from __future__ import annotations

import numpy as np

from experiments.geometry.torch_mm_backward_dot_assembly import (
    TorchMMBackwardDotAssembly,
)


class FullKSHOHessianAssembly:
    """Bit-replay the production raw and symmetric full-K SHO Hessian."""

    def __init__(self, output_count: int = 2) -> None:
        if output_count != 2:
            raise ValueError("the recovered active path requires two outputs")
        self.output_count = int(output_count)
        self.dot = TorchMMBackwardDotAssembly()
        self.forward = self.dot.fma
        self.arithmetic = self.forward.arithmetic

    def second_adjoint_scale(self, transition_count: int) -> np.float64:
        """Replay MeanBackward then square double-backward at runtime length.

        Keep the two production roundings distinct: MeanBackward first
        materializes RN(1 / residual_element_count), then the square second
        adjoint multiplies that rounded value by 2.  This deliberately avoids
        replacing the path with the algebraically simplified 1 / N.
        """
        if transition_count <= 0:
            raise ValueError("the recovered active path requires N>0")
        # MeanBackward materializes RN(1 / (2*N)); the square second adjoint
        # first doubles that scalar, before it multiplies the propagated row.
        mean_scale = np.float64(1.0) / np.float64(
            self.output_count * transition_count
        )
        return np.float64(self._multiply(mean_scale, np.float64(2.0)))

    def _multiply(self, left, right):
        return self.arithmetic._call(self.arithmetic.multiply, left, right)

    def raw(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] <= 0:
            raise ValueError(f"inputs must have shape [N,2] with N>0, got {values.shape}")
        transition_count = int(values.shape[0])
        second_adjoint_scale = self.second_adjoint_scale(transition_count)
        rows = []
        parameter_count = self.output_count * 2
        for row_index in range(parameter_count):
            output_index, coordinate_index = divmod(row_index, 2)
            seed = np.zeros((self.output_count, 2), dtype=np.float64)
            seed[output_index, coordinate_index] = np.float64(1.0)

            # d(grad_K[row])/d(incoming_prediction_adjoint): X @ seed.T.
            propagated = self.forward(values, seed)
            # d(incoming_prediction_adjoint)/d(prediction).
            scaled = self._multiply(propagated, second_adjoint_scale)
            # Final d(prediction)/dK MmBackward, preserving its DGEMM order.
            row_matrix = self.dot(scaled.T, values)
            rows.append(row_matrix.reshape(-1))
        return np.stack(rows)

    def symmetrize(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (4, 4):
            raise ValueError("raw Hessian must have shape [4,4]")
        summed = self.forward._add(values, values.T)
        return self._multiply(np.float64(0.5), summed)

    def evaluate(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = self.raw(inputs)
        return raw, self.symmetrize(raw)

    __call__ = evaluate
