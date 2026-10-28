"""Exact N1 boundary-face resolution from locked arithmetic and routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.sho.assemble_n3_gradient_operator import LearnedArithmetic
from experiments.sho.validate_representative_tie_selector import StrictLossSelector


@dataclass(frozen=True)
class FaceResolutionTrace:
    requested: float
    states: tuple[dict[str, object], ...]
    unclamped_k: float
    lower_clamp_taken: bool
    upper_clamp_taken: bool
    multiplication_calls: int
    addition_calls: int
    comparison_calls: int
    fallback_calls: int


@dataclass(frozen=True)
class FaceResolutionBatchTrace:
    iterations: np.ndarray
    unclamped_k: np.ndarray
    lower_clamp_taken: np.ndarray
    upper_clamp_taken: np.ndarray
    multiplication_calls: int
    addition_calls: int
    comparison_calls: int
    fallback_calls: int


class BoundaryFaceResolutionAssembly:
    """Return min(257,max(9,ceil(sqrt(requested)))) without root/ceil."""

    def __init__(self) -> None:
        self.arithmetic = LearnedArithmetic()
        self.comparison = StrictLossSelector()

    def _multiply(self, left: float, right: float) -> float:
        with torch.no_grad():
            value = self.arithmetic.multiply(
                torch.as_tensor(left, dtype=torch.float64),
                torch.as_tensor(right, dtype=torch.float64),
            )
        return float(value)

    def _add(self, left: float, right: float) -> float:
        with torch.no_grad():
            value = self.arithmetic.add(
                torch.as_tensor(left, dtype=torch.float64),
                torch.as_tensor(right, dtype=torch.float64),
            )
        return float(value)

    def _less_than(self, left: float, right: float) -> bool:
        less, _, _ = self.comparison.decide(
            np.asarray([right], dtype=np.float64),
            np.asarray([left], dtype=np.float64),
        )
        return bool(less[0])

    def evaluate_batch(self, requested, *, return_trace: bool = False):
        values = np.asarray(requested, dtype=np.float64)
        flat = values.reshape(-1)
        if (
            not np.isfinite(flat).all()
            or np.any(flat <= 0.0)
            or np.any(flat != flat.astype(np.int64).astype(np.float64))
        ):
            raise ValueError("positive integer-valued float64 requested values required")

        current = np.zeros(len(flat), dtype=np.float64)
        iterations = np.zeros(len(flat), dtype=np.int64)
        active = np.arange(len(flat), dtype=np.int64)
        multiplication_calls = addition_calls = comparison_calls = 0
        while len(active):
            tensor = torch.from_numpy(current[active])
            with torch.no_grad():
                square = self.arithmetic.multiply(tensor, tensor).numpy()
            multiplication_calls += len(active)
            less, _, _ = self.comparison.decide(flat[active], square)
            comparison_calls += len(active)
            continuing = active[np.asarray(less, dtype=bool)]
            if not len(continuing):
                break
            with torch.no_grad():
                incremented = self.arithmetic.add(
                    torch.from_numpy(current[continuing]),
                    torch.ones(len(continuing), dtype=torch.float64),
                ).numpy()
            current[continuing] = incremented
            iterations[continuing] += 1
            addition_calls += len(continuing)
            active = continuing

        unclamped = current.copy()
        lower, _, _ = self.comparison.decide(
            np.full(len(flat), 9.0, dtype=np.float64), current
        )
        lower = np.asarray(lower, dtype=bool)
        comparison_calls += len(flat)
        current[lower] = 9.0
        upper, _, _ = self.comparison.decide(
            current, np.full(len(flat), 257.0, dtype=np.float64)
        )
        upper = np.asarray(upper, dtype=bool)
        comparison_calls += len(flat)
        current[upper] = 257.0
        output = current.astype(np.int64).reshape(values.shape)
        if not return_trace:
            return output
        return output, FaceResolutionBatchTrace(
            iterations=iterations.reshape(values.shape),
            unclamped_k=unclamped.reshape(values.shape),
            lower_clamp_taken=lower.reshape(values.shape),
            upper_clamp_taken=upper.reshape(values.shape),
            multiplication_calls=multiplication_calls,
            addition_calls=addition_calls,
            comparison_calls=comparison_calls,
            fallback_calls=0,
        )

    def __call__(self, requested, *, return_trace: bool = False):
        value = np.float64(requested)
        if (
            not np.isfinite(value)
            or value <= 0.0
            or value != np.float64(int(value))
        ):
            raise ValueError("positive integer-valued float64 requested value required")

        current = 0.0
        states: list[dict[str, object]] = []
        multiplication_calls = addition_calls = comparison_calls = 0
        while True:
            square = self._multiply(current, current)
            multiplication_calls += 1
            continue_loop = self._less_than(square, float(value))
            comparison_calls += 1
            if return_trace:
                states.append({
                    "k": current,
                    "square": square,
                    "square_less_than_requested": continue_loop,
                })
            if not continue_loop:
                break
            current = self._add(current, 1.0)
            addition_calls += 1

        unclamped = current
        lower_clamp = self._less_than(current, 9.0)
        comparison_calls += 1
        if lower_clamp:
            current = 9.0
        upper_clamp = self._less_than(257.0, current)
        comparison_calls += 1
        if upper_clamp:
            current = 257.0

        output = int(current)
        if not return_trace:
            return output
        return output, FaceResolutionTrace(
            requested=float(value),
            states=tuple(states),
            unclamped_k=unclamped,
            lower_clamp_taken=lower_clamp,
            upper_clamp_taken=upper_clamp,
            multiplication_calls=multiplication_calls,
            addition_calls=addition_calls,
            comparison_calls=comparison_calls,
            fallback_calls=0,
        )
