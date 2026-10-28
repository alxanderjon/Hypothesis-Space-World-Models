"""Teacher-exact N3 Hessian spectrum and condition analysis.

The two spectral decompositions in the production Torch 2.2.2 arm64 path are
kept as explicit external deterministic primitives.  They are not
interchangeable at the bit level:

* ``torch.linalg.svdvals`` supplies the condition number through Accelerate
  ``dgesdd``;
* ``torch.linalg.eigvalsh(..., UPLO="L")`` supplies signed curvature through
  Accelerate ``dsyevd``.

The surrounding scalar division and strict comparisons reuse locked learned
primitives.  Finiteness checks, reductions, tensor storage, and calls into the
external LAPACK routines are deterministic scaffolding.  This module stops
before the positivity/condition predicates are combined into a curvature
route.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from experiments.sho.assemble_n3_gradient_operator import LearnedArithmetic
from experiments.sho.validate_representative_tie_selector import StrictLossSelector


@dataclass(frozen=True)
class EigenspectrumConditionResult:
    finite: bool
    singular_values: torch.Tensor | None
    smallest_singular: float | None
    largest_singular: float | None
    condition: float
    eigenvalues: torch.Tensor | None
    minimum_eigenvalue: float | None
    maximum_eigenvalue: float | None
    maximum_absolute_eigenvalue: float | None
    scale: float | None
    positive: bool
    condition_ok: bool


class N3EigenspectrumConditionAssembly:
    """Replay exactly the production operations preceding curvature routing."""

    def __init__(self, condition_limit: float = 1.0e12) -> None:
        self.condition_limit = float(condition_limit)
        self.arithmetic = LearnedArithmetic()
        self.compare = StrictLossSelector()
        self.svd_calls = 0
        self.eigh_calls = 0

    def _positive(self, value: float) -> bool:
        # StrictLossSelector returns candidate < best.  Here: 0 < value.
        answer, _, _ = self.compare.decide(
            np.asarray([value], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )
        return bool(answer[0])

    def _at_most(self, value: float, limit: float) -> bool:
        if math.isinf(value) and value > 0.0:
            return False
        # Equality passes: negate the locked strict comparison limit < value.
        above, _, _ = self.compare.decide(
            np.asarray([value], dtype=np.float64),
            np.asarray([limit], dtype=np.float64),
        )
        return not bool(above[0])

    def _divide(self, numerator: float, denominator: float) -> float:
        values = self.arithmetic.divide(
            torch.tensor(numerator, dtype=torch.float64),
            torch.tensor(denominator, dtype=torch.float64),
        )
        return float(values)

    def analyze(
        self, gradient: torch.Tensor, hessian: torch.Tensor
    ) -> EigenspectrumConditionResult:
        if gradient.dtype != torch.float64 or hessian.dtype != torch.float64:
            raise TypeError("the recovered production path requires float64 tensors")
        finite = bool(torch.isfinite(gradient).all() and torch.isfinite(hessian).all())
        if not finite:
            return EigenspectrumConditionResult(
                finite=False,
                singular_values=None,
                smallest_singular=None,
                largest_singular=None,
                condition=math.inf,
                eigenvalues=None,
                minimum_eigenvalue=None,
                maximum_eigenvalue=None,
                maximum_absolute_eigenvalue=None,
                scale=None,
                positive=False,
                condition_ok=False,
            )

        # Production uses two independent LAPACK paths.  Keep their call order.
        self.svd_calls += 1
        singular_values = torch.linalg.svdvals(hessian)
        smallest = float(torch.min(singular_values))
        largest = float(torch.max(singular_values))
        condition = self._divide(largest, smallest) if self._positive(smallest) else math.inf

        self.eigh_calls += 1
        eigenvalues = torch.linalg.eigvalsh(hessian, UPLO="L")
        minimum = float(torch.min(eigenvalues))
        maximum = float(torch.max(eigenvalues))
        maximum_absolute = float(torch.max(torch.abs(eigenvalues)))
        scale = max(maximum_absolute, torch.finfo(torch.float64).tiny)

        return EigenspectrumConditionResult(
            finite=True,
            singular_values=singular_values,
            smallest_singular=smallest,
            largest_singular=largest,
            condition=condition,
            eigenvalues=eigenvalues,
            minimum_eigenvalue=minimum,
            maximum_eigenvalue=maximum,
            maximum_absolute_eigenvalue=maximum_absolute,
            scale=scale,
            positive=self._positive(minimum),
            condition_ok=self._at_most(condition, self.condition_limit),
        )

