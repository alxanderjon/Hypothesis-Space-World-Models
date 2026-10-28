"""Exact learned arithmetic for active prefix residual/MSE objectives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.sho import assemble_n3_gradient_operator as arithmetic


@dataclass
class PrefixResidualMSEResult:
    prediction: torch.Tensor
    target: torch.Tensor
    residual: torch.Tensor
    square: torch.Tensor
    reduction_terms: torch.Tensor
    reduction_accumulators: torch.Tensor
    total: torch.Tensor
    count: int
    loss: torch.Tensor
    system: str
    reduction_path: str


class PrefixResidualMSEAssembly:
    """Dimension-general residual/square plus production-ordered reductions."""

    def __init__(self) -> None:
        if arithmetic.OPS is None:
            arithmetic.OPS = arithmetic.LearnedArithmetic()
        self.invocations = 0
        self.by_system: dict[str, int] = {}
        self.native_objective_arithmetic_fallback_calls = 0

    @staticmethod
    def _add(left, right):
        return arithmetic.add(left, right)

    def _pairwise_sum(self, values: torch.Tensor) -> torch.Tensor:
        """NumPy active pairwise reduction order, block size 128."""
        flat = values.reshape(-1)
        count = flat.numel()
        if count == 0:
            return torch.tensor(0.0, dtype=torch.float64)
        if count < 8:
            result = torch.tensor(-0.0, dtype=torch.float64)
            for index in range(count):
                result = self._add(result, flat[index])
            return result
        if count <= 128:
            lanes = [flat[index] for index in range(8)]
            index = 8
            while index <= count - 8:
                for lane in range(8):
                    lanes[lane] = self._add(lanes[lane], flat[index + lane])
                index += 8
            left = self._add(
                self._add(lanes[0], lanes[1]),
                self._add(lanes[2], lanes[3]),
            )
            right = self._add(
                self._add(lanes[4], lanes[5]),
                self._add(lanes[6], lanes[7]),
            )
            result = self._add(left, right)
            while index < count:
                result = self._add(result, flat[index])
                index += 1
            return result
        split = count // 2
        split -= split % 8
        return self._add(
            self._pairwise_sum(flat[:split]),
            self._pairwise_sum(flat[split:]),
        )

    def _reflection_numpy_sum(self, square: torch.Tensor):
        if square.ndim != 2 or square.shape[1] != 4:
            raise ValueError("reflection-ball prefix residuals must have four components")
        row_terms = []
        accumulators = []
        total = torch.tensor(0.0, dtype=torch.float64)
        for row in square:
            # Accelerate ddot's active four-value lane order is
            # (q0+q2) + (q1+q3), after separately rounded products.
            row_total = self._add(
                self._add(row[0], row[2]),
                self._add(row[1], row[3]),
            )
            row_terms.append(row_total)
            total = self._add(total, row_total)
            accumulators.append(total)
        return torch.stack(row_terms), torch.stack(accumulators), total

    def evaluate(self, predicted, observed, system: str, *,
                 reduction_path: str = "torch") -> PrefixResidualMSEResult:
        prediction = (
            predicted if isinstance(predicted, torch.Tensor)
            else torch.from_numpy(np.asarray(predicted, dtype=np.float64).copy())
        )
        target = (
            observed if isinstance(observed, torch.Tensor)
            else torch.from_numpy(np.asarray(observed, dtype=np.float64).copy())
        )
        prediction = prediction.to(dtype=torch.float64)
        target = target.to(dtype=torch.float64)
        if prediction.shape != target.shape:
            raise ValueError("prediction and observed prefix shapes differ")

        residual = arithmetic.subtract(prediction, target)
        square = arithmetic.multiply(residual, residual)
        if system == "reflection_ball" and reduction_path == "numpy":
            terms, accumulators, total = self._reflection_numpy_sum(square)
        elif reduction_path == "torch":
            # The active aten CPU reduction order is retained as the already
            # validated implementation-order/state scaffold.  Subtraction,
            # squaring, and the final scaling remain locked assemblies.
            total = square.sum()
            terms = total.reshape(1)
            accumulators = terms
        else:
            total = self._pairwise_sum(square)
            terms = total.reshape(1)
            accumulators = terms
        count = square.numel()
        loss = arithmetic.divide(total, float(count))
        self.invocations += 1
        self.by_system[system] = self.by_system.get(system, 0) + 1
        return PrefixResidualMSEResult(
            prediction=prediction, target=target, residual=residual,
            square=square, reduction_terms=terms,
            reduction_accumulators=accumulators, total=total,
            count=count, loss=loss, system=system,
            reduction_path=reduction_path,
        )
