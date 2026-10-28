"""Exact SciPy 1.13.0 cKDTree Euclidean-distance arithmetic.

SciPy's cKDTree does not evaluate ``sqrt(sum(delta * delta))`` as a generic
NumPy reduction.  Its p=2 kernel has four independent accumulators.  The
official 1.13.0 source spells the operation as a four-coordinate unrolled
loop, and the macOS arm64 wheel contracts every square-plus-accumulator into
an IEEE-754 fused multiply-add.  The four accumulators are reduced in order,
then any remaining coordinates are accumulated with the same fused operation.

This module reproduces that kernel with the already locked subtraction,
addition and multiplication operators, the validated exact product-tail
assembly, and the recovered correctly-rounded square root.  Loop/index state
and the four-lane accumulator layout are deterministic implementation-order
scaffolding.  No cKDTree ownership or neighbor ordering is implemented here.
"""
from __future__ import annotations

import numpy as np

from experiments.geometry.compensated_division_residual import (
    CompensatedDivisionResidual,
)
from experiments.geometry.n1_sqrt_rounding_assembly import N1CorrectlyRoundedSqrt


class CKDTreeEuclideanDistance:
    """Bit-replay the active d=4/d=5 branch of SciPy's p=2 kernel.

    The source-order loop is dimension-general.  The production arm64 binary
    introduces another vectorized code path at dimensions >= 12; that inactive
    compiler branch is deliberately outside this component's exact contract.
    """

    lane_count = 4

    def __init__(self) -> None:
        self.arithmetic = CompensatedDivisionResidual()
        self.sqrt = N1CorrectlyRoundedSqrt()

    def _add(self, left, right):
        return self.arithmetic._call(self.arithmetic.add, left, right)

    def _subtract(self, left, right):
        return self.arithmetic._call(self.arithmetic.subtract, left, right)

    def _two_sum(self, left, right):
        """Exact real expansion of RN(left + right), using locked arithmetic."""
        total = self._add(left, right)
        right_virtual = self._subtract(total, left)
        left_virtual = self._subtract(total, right_virtual)
        left_tail = self._subtract(left, left_virtual)
        right_tail = self._subtract(right, right_virtual)
        tail = self._add(left_tail, right_tail)
        return total, tail

    def _fused_square_add(self, delta, accumulator):
        """Correctly rounded RN(delta*delta + accumulator), without native FMA."""
        product, product_tail = self.arithmetic.product_pair(delta, delta)
        total, sum_tail = self._two_sum(product, accumulator)
        correction = self._add(sum_tail, product_tail)
        return self._add(total, correction)

    def squared(self, queries: np.ndarray, sites: np.ndarray) -> np.ndarray:
        query = np.asarray(queries, dtype=np.float64)
        site = np.asarray(sites, dtype=np.float64)
        if query.shape != site.shape or query.ndim != 2:
            raise ValueError("paired [n,d] float64 inputs required")
        if not np.isfinite(query).all() or not np.isfinite(site).all():
            raise ValueError("finite cKDTree coordinates required")

        delta = self._subtract(query, site)
        accumulators = [np.zeros(len(delta), dtype=np.float64) for _ in range(4)]
        dimension = delta.shape[1]
        if dimension >= 12:
            raise ValueError(
                "the SciPy arm64 wheel's separate d>=12 vectorized branch is outside "
                "the active d=4/d=5 exact-replay contract"
            )
        axis = 0
        while axis + 4 <= dimension:
            for lane in range(4):
                accumulators[lane] = self._fused_square_add(
                    delta[:, axis + lane], accumulators[lane]
                )
            axis += 4

        # This parenthesization is the order emitted by the SciPy 1.13.0
        # macOS arm64 wheel: ((acc[0] + acc[1]) + acc[2]) + acc[3].
        total = self._add(accumulators[0], accumulators[1])
        total = self._add(total, accumulators[2])
        total = self._add(total, accumulators[3])
        while axis < dimension:
            total = self._fused_square_add(delta[:, axis], total)
            axis += 1
        return total

    def evaluate(self, queries: np.ndarray, sites: np.ndarray) -> np.ndarray:
        return self.sqrt(self.squared(queries, sites))

    __call__ = evaluate
