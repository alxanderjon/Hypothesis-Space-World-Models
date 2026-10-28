"""Exact arbitrary-modulus remainder for the N1 initial-design contract.

The mathematical state update is deliberately only repeated subtraction.  The
comparison and subtraction values come from the project's already locked
float64 operators; array masks, loop state, and counters are deterministic
scaffolding.  Inputs are assumed to satisfy the narrow caller contract:
nonnegative integer-valued float64 ``x`` and positive integer-valued float64
``n``.  No behavior is promised outside that contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.sho.validate_representative_tie_selector import StrictLossSelector


@dataclass(frozen=True)
class RemainderTrace:
    iterations: np.ndarray
    comparison_calls: int
    subtraction_calls: int
    fallback_calls: int


@dataclass(frozen=True)
class GCDTrace:
    iterations: np.ndarray
    nested_remainder_iterations: np.ndarray
    equality_calls: int
    remainder_calls: int
    fallback_calls: int
    state_traces: tuple[tuple[dict[str, object], ...], ...] | None = None


class InitialDesignRemainderAssembly:
    """Compute exact nonnegative remainder with locked operations."""

    def __init__(self) -> None:
        self.comparison = StrictLossSelector()

    def _less_than(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        # StrictLossSelector decides whether candidate < best.  Supplying
        # candidate=left and best=right therefore implements left < right.
        less, _, _ = self.comparison.decide(right, left)
        return np.asarray(less, dtype=bool)

    def _subtract(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        pairs = np.column_stack((left, right)).astype(np.float64, copy=False)
        with torch.no_grad():
            result = self.comparison.subtraction(torch.from_numpy(pairs)).numpy()
        return np.asarray(result, dtype=np.float64)

    def equal(self, left, right) -> np.ndarray:
        """Exact equality routed through the locked comparison state."""
        left_values, right_values = np.broadcast_arrays(
            np.asarray(left, dtype=np.float64),
            np.asarray(right, dtype=np.float64),
        )
        # decide(best=right, candidate=left) returns the exact sign state of
        # left-right.  State zero is exact float64 equality on this contract.
        _, _, state = self.comparison.decide(right_values, left_values)
        return np.asarray(state) == 0

    def __call__(self, x, n, *, return_trace: bool = False):
        values, moduli = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(n, dtype=np.float64)
        )
        if not np.isfinite(values).all() or not np.isfinite(moduli).all():
            raise ValueError("finite operands required")
        if np.any(values < 0.0) or np.any(moduli <= 0.0):
            raise ValueError("nonnegative x and positive n required")
        if np.any((values == 0.0) & np.signbit(values)):
            raise ValueError("negative zero is outside the initial-design contract")

        shape = values.shape
        remainder = values.reshape(-1).copy()
        divisor = moduli.reshape(-1)
        iterations = np.zeros(len(remainder), dtype=np.int64)
        comparison_calls = 0
        subtraction_calls = 0

        while len(remainder):
            less = self._less_than(remainder, divisor)
            comparison_calls += len(remainder)
            active = np.flatnonzero(~less)
            if not len(active):
                break
            remainder[active] = self._subtract(
                remainder[active], divisor[active]
            )
            iterations[active] += 1
            subtraction_calls += len(active)

        output = remainder.reshape(shape)
        if not return_trace:
            return output
        return output, RemainderTrace(
            iterations=iterations.reshape(shape),
            comparison_calls=comparison_calls,
            subtraction_calls=subtraction_calls,
            fallback_calls=0,
        )


class InitialDesignGCDAssembly:
    """Euclidean GCD built only from locked remainder/equality operations."""

    def __init__(
        self, remainder: InitialDesignRemainderAssembly | None = None
    ) -> None:
        self.remainder = remainder or InitialDesignRemainderAssembly()

    def equal(self, left, right) -> np.ndarray:
        return self.remainder.equal(left, right)

    def __call__(
        self,
        a,
        b,
        *,
        return_trace: bool = False,
        record_states: bool = False,
    ):
        left, right = np.broadcast_arrays(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("finite operands required")
        if np.any(left <= 0.0) or np.any(right <= 0.0):
            raise ValueError("positive operands required by the initial-design contract")

        shape = left.shape
        current_a = left.reshape(-1).copy()
        current_b = right.reshape(-1).copy()
        iterations = np.zeros(len(current_a), dtype=np.int64)
        nested_iterations = np.zeros(len(current_a), dtype=np.int64)
        equality_calls = 0
        remainder_calls = 0
        histories: list[list[dict[str, object]]] | None = (
            [[] for _ in range(len(current_a))] if record_states else None
        )

        while len(current_b):
            is_zero = self.equal(current_b, np.zeros_like(current_b))
            equality_calls += len(current_b)
            active = np.flatnonzero(~is_zero)
            if not len(active):
                break
            old_a = current_a[active].copy()
            old_b = current_b[active].copy()
            residues, remainder_trace = self.remainder(
                old_a, old_b, return_trace=True
            )
            remainder_calls += len(active)
            nested_iterations[active] += remainder_trace.iterations
            iterations[active] += 1
            if histories is not None:
                for local, global_index in enumerate(active):
                    histories[int(global_index)].append({
                        "a": float(old_a[local]),
                        "b": float(old_b[local]),
                        "r": float(residues[local]),
                        "remainder_iterations": int(
                            remainder_trace.iterations[local]
                        ),
                    })
            current_a[active] = old_b
            current_b[active] = residues

        output = current_a.reshape(shape)
        if not return_trace:
            return output
        frozen_histories = None
        if histories is not None:
            frozen_histories = tuple(tuple(history) for history in histories)
        return output, GCDTrace(
            iterations=iterations.reshape(shape),
            nested_remainder_iterations=nested_iterations.reshape(shape),
            equality_calls=equality_calls,
            remainder_calls=remainder_calls,
            fallback_calls=0,
            state_traces=frozen_histories,
        )


def multipliers_with_gcd_assembly(
    dimension: int,
    gcd: InitialDesignGCDAssembly | None = None,
    *,
    return_trace: bool = False,
):
    """Generate the teacher multiplier schedule with the assembled GCD."""
    count = 5 * dimension
    operator = gcd or InitialDesignGCDAssembly()
    multipliers: list[int] = []
    candidate = 1
    calls = []
    while len(multipliers) < dimension:
        value, trace = operator(
            np.float64(candidate), np.float64(count), return_trace=True
        )
        coprime = bool(operator.equal(value, np.float64(1.0)))
        calls.append({
            "candidate": candidate,
            "n": count,
            "gcd": int(value),
            "coprime": coprime,
            "euclidean_iterations": int(trace.iterations),
            "nested_remainder_iterations": int(
                trace.nested_remainder_iterations
            ),
        })
        if coprime:
            multipliers.append(candidate)
        candidate += 1
    if not return_trace:
        return multipliers
    return multipliers, {
        "calls": calls,
        "production_gcd_fallback_calls": 0,
        "production_remainder_fallback_calls": 0,
    }


def initial_design_with_remainder_assembly(
    dimension: int,
    remainder: InitialDesignRemainderAssembly | None = None,
    gcd: InitialDesignGCDAssembly | None = None,
    *,
    return_trace: bool = False,
):
    """Rebuild the teacher lattice with assembled remainder and Euclidean GCD."""
    count = 5 * dimension
    operator = remainder or InitialDesignRemainderAssembly()
    gcd_operator = gcd or InitialDesignGCDAssembly(operator)
    multipliers, gcd_trace = multipliers_with_gcd_assembly(
        dimension, gcd_operator, return_trace=True
    )
    base = np.arange(count, dtype=np.float64)
    columns = []
    traces = []
    for multiplier in multipliers:
        product = np.multiply(multiplier, base, dtype=np.float64)
        residue, trace = operator(
            product, np.float64(count), return_trace=True
        )
        centered = np.add(residue, np.float64(0.5), dtype=np.float64)
        columns.append(np.divide(centered, count, dtype=np.float64))
        traces.append(trace)
    sites = np.column_stack(columns)
    if not return_trace:
        return sites
    return sites, {
        "count": count,
        "multipliers": multipliers,
        "gcd_trace": gcd_trace,
        "remainder_traces": traces,
        "production_gcd_fallback_calls": 0,
        "production_remainder_fallback_calls": 0,
    }
