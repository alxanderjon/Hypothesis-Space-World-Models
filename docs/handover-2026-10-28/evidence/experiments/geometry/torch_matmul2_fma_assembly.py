"""Exact active float64 two-term Torch/Accelerate matrix-product arithmetic.

For the current full-K SHO objective, Torch 2.2.2 on macOS arm64 dispatches
the [N,2] x [2,2] product through its Accelerate-backed matrix kernel.  Its
two-row vector body executes each output dot product as

    p0 = RN(x0 * k0)
    y  = RN_fma(x1 * k1 + p0)

An odd final row is handled by the scalar tail as two rounded products followed
by a rounded add.  The fused body operation is reconstructed from already-
locked product-tail and TwoSum arithmetic.  This module is forward-only: it
deliberately does not claim the separate BLAS-backed matmul adjoint accumulation.
"""
from __future__ import annotations

import numpy as np

from experiments.geometry.compensated_division_residual import (
    CompensatedDivisionResidual,
)


class TorchMatmul2FMAAssembly:
    """Bit-replay [N,2] @ [M,2].T for the active float64 CPU kernel."""

    def __init__(self) -> None:
        self.arithmetic = CompensatedDivisionResidual()

    def _add(self, left, right):
        return self.arithmetic._call(self.arithmetic.add, left, right)

    def _subtract(self, left, right):
        return self.arithmetic._call(self.arithmetic.subtract, left, right)

    def _two_sum(self, left, right):
        total = self._add(left, right)
        right_virtual = self._subtract(total, left)
        left_virtual = self._subtract(total, right_virtual)
        left_tail = self._subtract(left, left_virtual)
        right_tail = self._subtract(right, right_virtual)
        return total, self._add(left_tail, right_tail)

    def fused_multiply_add(self, left, right, addend):
        product, product_tail = self.arithmetic.product_pair(left, right)
        total, sum_tail = self._two_sum(product, addend)
        correction = self._add(sum_tail, product_tail)
        return self._add(total, correction)

    def evaluate(self, inputs: np.ndarray, matrix_rows: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        rows = np.asarray(matrix_rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("inputs must have shape [N,2]")
        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError("matrix_rows must have shape [M,2]")
        first = self.arithmetic._call(
            self.arithmetic.multiply, values[:, 0, None], rows[None, :, 0]
        )
        second = self.arithmetic._call(
            self.arithmetic.multiply, values[:, 1, None], rows[None, :, 1]
        )
        result = self.fused_multiply_add(
            values[:, 1, None], rows[None, :, 1], first
        )
        if len(values) % 2:
            result[-1] = self._add(first[-1], second[-1])
        return result

    __call__ = evaluate
