"""Exact active float64 Torch/Accelerate ``MmBackward`` dot accumulation.

The full-K SHO backward multiply has shapes ``[2,K] @ [K,2]``.  On the
audited Torch 2.2.2 macOS arm64 build, the Accelerate DGEMM micro-kernel keeps
one accumulator for each of the four output entries and visits the reduction
coordinate in increasing memory order:

    accumulator = RN_fma(left[..., i], right[..., i], accumulator)

There is no reduction-axis lane split or partial-sum tree for this active
small-output kernel.  The last reduction coordinate is an ordinary final FMA.
The FMA is reconstructed from the already locked product-tail and TwoSum
assemblies; the reduction loop and accumulator storage are deterministic
implementation-order scaffolding.
"""
from __future__ import annotations

import numpy as np

from experiments.geometry.torch_matmul2_fma_assembly import (
    TorchMatmul2FMAAssembly,
)


class TorchMMBackwardDotAssembly:
    """Replay the active ``[2,K] @ [K,2]`` float64 DGEMM accumulation."""

    def __init__(self) -> None:
        self.fma = TorchMatmul2FMAAssembly()

    def evaluate(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the exact active-kernel product, with optional batch axes.

        ``left`` has shape ``[..., M, K]`` and ``right`` has shape
        ``[..., K, N]``.  Exact replay is validated for the production
        ``M=N=2`` kernel; supporting batch axes is only vectorized bookkeeping.
        """
        lhs = np.asarray(left, dtype=np.float64)
        rhs = np.asarray(right, dtype=np.float64)
        if lhs.ndim < 2 or rhs.ndim < 2:
            raise ValueError("left and right must each have at least two axes")
        if lhs.shape[:-2] != rhs.shape[:-2]:
            raise ValueError("batch shapes must agree")
        if lhs.shape[-1] != rhs.shape[-2]:
            raise ValueError("reduction dimensions must agree")
        if lhs.shape[-2] != 2 or rhs.shape[-1] != 2:
            raise ValueError("the recovered production kernel has M=N=2")

        result = np.zeros(
            lhs.shape[:-2] + (lhs.shape[-2], rhs.shape[-1]), dtype=np.float64
        )
        for reduction_index in range(lhs.shape[-1]):
            result = self.fma.fused_multiply_add(
                lhs[..., :, reduction_index, None],
                rhs[..., reduction_index, None, :],
                result,
            )
        return result

    __call__ = evaluate
