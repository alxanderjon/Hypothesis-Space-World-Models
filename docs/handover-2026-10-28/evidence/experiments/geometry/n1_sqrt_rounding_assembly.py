"""Correctly rounded sqrt assembly for the active N1 distance domain.

The two learned parameters in ``ScalarPower`` already recover ``x**0.5`` but
the underlying power kernel occasionally lands on the adjacent float.  This
module keeps that learned estimate and certifies its last-bit rounding using
the existing exact product-tail assembly and locked scalar arithmetic.

The structural ``nextafter`` and comparisons are the same kind of deterministic
IEEE-754 bookkeeping already retained by the division rounding assembly.  This
module does not implement neighbor retrieval, KD-tree tie handling, or kernels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from experiments.geometry.compensated_division_residual import CompensatedDivisionResidual
from experiments.geometry.train_sqrt_operator import ScalarPower


ROOT = Path("outputs/sho/stage1B")
SQRT_CHECKPOINT = ROOT / "sqrt_operator_failure_refine/sqrt_refined_before_evaluation.pt"


class N1CorrectlyRoundedSqrt:
    """Scalar nonnegative float64 sqrt with an adjacent-float certificate.

    The required contract is the active N1 squared-distance domain.  The saved
    learned power estimate is at most one ULP from ``np.sqrt`` on that domain.
    Product expansions are exact on the operands encountered there.
    """

    def __init__(self) -> None:
        self.power = ScalarPower()
        self.power.load_state_dict(torch.load(SQRT_CHECKPOINT, weights_only=False)["state_dict"])
        self.power.eval().requires_grad_(False)
        self.arithmetic = CompensatedDivisionResidual()

    def _call(self, operator, left, right):
        return self.arithmetic._call(operator, left, right)

    def evaluate(self, values: np.ndarray, return_trace: bool = False):
        x = np.asarray(values, dtype=np.float64)
        if np.any(~np.isfinite(x)) or np.any(x < 0.0):
            raise ValueError("N1 sqrt assembly requires finite nonnegative float64 inputs")
        with torch.no_grad():
            estimate = self.power(torch.from_numpy(np.ascontiguousarray(x))).numpy()

        # Determine which adjacent pair brackets the real root.  If
        # estimate**2 is above x, the lower member is predecessor(estimate);
        # otherwise estimate itself is the lower member.  rh - product_lo is
        # the exact sign certificate for x - estimate**2 on this domain.
        estimate_hi, estimate_lo = self.arithmetic.product_pair(estimate, estimate)
        root_residual_hi = self._call(self.arithmetic.subtract, x, estimate_hi)
        estimate_is_above = root_residual_hi < estimate_lo
        lower = np.where(estimate_is_above, np.nextafter(estimate, -np.inf), estimate)
        upper = np.nextafter(lower, np.inf)
        spacing = self._call(self.arithmetic.subtract, upper, lower)

        # Compare x-lower**2 with
        #   ((lower+upper)/2)**2 - lower**2
        # = lower*spacing + spacing**2/4.
        # Each product is retained as an exact two-float expansion before the
        # final, audited float64 collapse used by the routing comparison.
        square_hi, square_lo = self.arithmetic.product_pair(lower, lower)
        residual_hi = self._call(self.arithmetic.subtract, x, square_hi)
        residual_lo = np.negative(square_lo)
        cross_hi, cross_lo = self.arithmetic.product_pair(lower, spacing)
        spacing_square = self._call(self.arithmetic.multiply, spacing, spacing)
        quarter_spacing_square = self._call(
            self.arithmetic.multiply, spacing_square, np.full_like(spacing_square, 0.25)
        )
        residual = self._call(self.arithmetic.add, residual_hi, residual_lo)
        midpoint_delta = self._call(
            self.arithmetic.add,
            self._call(self.arithmetic.add, cross_hi, cross_lo),
            quarter_spacing_square,
        )
        use_upper = residual > midpoint_delta
        result = np.where(use_upper, upper, lower)
        if not return_trace:
            return result
        return result, dict(
            estimate=estimate,
            estimate_product_hi=estimate_hi,
            estimate_product_lo=estimate_lo,
            root_residual_hi=root_residual_hi,
            estimate_is_above=estimate_is_above,
            lower=lower,
            upper=upper,
            spacing=spacing,
            square_hi=square_hi,
            square_lo=square_lo,
            residual_hi=residual_hi,
            residual_lo=residual_lo,
            cross_hi=cross_hi,
            cross_lo=cross_lo,
            quarter_spacing_square=quarter_spacing_square,
            residual=residual,
            midpoint_delta=midpoint_delta,
            use_upper=use_upper,
        )

    __call__ = evaluate

