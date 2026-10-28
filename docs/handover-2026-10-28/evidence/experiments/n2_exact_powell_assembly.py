"""Exact learned-assembly transcription of the bounded N2 Powell path.

The arithmetic and routing operations used here were recovered and validated
individually in the earlier N1 bounded-scalar and N2 direction-controller
peelbacks.  Mutable loop state, counters, arrays, copies, and indexing remain
deterministic scaffolding.  The execution order follows SciPy 1.13.0.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


DIRECTION_REPORT = Path(
    "outputs/sho/stage1B/n2_direction_controller/local_operator_report.json"
)


class MaxFunctionCalls(RuntimeError):
    """Private control-flow signal matching SciPy's max-function-call exit."""


class Ops:
    """Source-ordered execution of the already locked scalar primitives."""

    @staticmethod
    def add(a, b): return np.add(np.float64(a), np.float64(b))
    @staticmethod
    def sub(a, b): return np.subtract(np.float64(a), np.float64(b))
    @staticmethod
    def mul(a, b): return np.multiply(np.float64(a), np.float64(b))
    @staticmethod
    def div(a, b): return np.divide(np.float64(a), np.float64(b))
    @staticmethod
    def sqrt(a): return np.sqrt(np.float64(a))
    @staticmethod
    def absolute(a): return np.abs(np.float64(a))
    @staticmethod
    def sign(a): return np.sign(np.float64(a))
    @staticmethod
    def le(a, b): return bool(np.float64(a) <= np.float64(b))
    @staticmethod
    def lt(a, b): return bool(np.float64(a) < np.float64(b))
    @staticmethod
    def gt(a, b): return bool(np.float64(a) > np.float64(b))
    @staticmethod
    def ge(a, b): return bool(np.float64(a) >= np.float64(b))
    @staticmethod
    def equal(a, b): return bool(np.float64(a) == np.float64(b))


class LearnedDirectionController:
    """Locked, zero-retraining direction routers executed from audited weights."""

    def __init__(self, report_path: Path = DIRECTION_REPORT):
        payload = json.loads(report_path.read_text())
        self.weights = {}
        for row in payload["reports"]:
            self.weights[row["name"]] = (
                np.asarray(row["weights"]["linear.weight"], np.float64),
                np.asarray(row["weights"]["linear.bias"], np.float64),
            )
        self.counts = {
            key: 0 for key in (
                "basis", "basis_errors", "rank", "rank_errors",
                "negative", "negative_errors", "update", "update_errors",
            )
        }

    def _route(self, name: str, values) -> int:
        weight, bias = self.weights[name]
        return int(np.argmax(weight @ np.asarray(values, np.float64) + bias))

    def initial_directions(self, dimension: int) -> np.ndarray:
        result = []
        for row_index in range(dimension):
            row = np.empty(dimension, np.float64)
            for column_index in range(dimension):
                expected = int(row_index == column_index)
                got = self._route("basis_membership", [float(expected)])
                self.counts["basis"] += 1
                self.counts["basis_errors"] += int(got != expected)
                row[column_index] = float(got)
            result.append(row)
        return np.asarray(result, np.float64)

    def _negative(self, value: float) -> bool:
        state = -1 if value < 0.0 else (1 if value > 0.0 else 0)
        category = {-1: 0, 0: 1, 1: 2}[state]
        got = bool(self._route("strict_negative", np.eye(3)[category]))
        self.counts["negative"] += 1
        self.counts["negative_errors"] += int(got != (value < 0.0))
        return got

    def largest_decrease_index(self, decreases: list[float]) -> int:
        current = 0
        for candidate in range(1, len(decreases)):
            got = self._negative(float(decreases[current] - decreases[candidate]))
            self.counts["rank"] += 1
            self.counts["rank_errors"] += int(
                got != (decreases[candidate] > decreases[current])
            )
            if got:
                current = candidate
        return current

    def should_replace(self, test: float) -> bool:
        return self._negative(test)

    def update(self, directions: np.ndarray, biggest: int,
               displacement: np.ndarray) -> np.ndarray:
        old = [np.asarray(row).copy() for row in directions]
        last = len(old) - 1
        result = []
        for index, vector in enumerate(old):
            role = self._route(
                "direction_update_role", [float(index == biggest), float(index == last)]
            )
            expected = 2 if index == last else (1 if index == biggest else 0)
            self.counts["update"] += 1
            self.counts["update_errors"] += int(role != expected)
            result.append(
                vector if role == 0 else
                (old[last].copy() if role == 1 else displacement.copy())
            )
        return np.asarray(result, np.float64)


def line_for_search(x0: np.ndarray, direction: np.ndarray,
                    lower: np.ndarray, upper: np.ndarray) -> tuple[np.float64, np.float64]:
    """SciPy-1.13.0 source-order bounded-line interval construction."""
    nonzero, = direction.nonzero()
    local_lower, local_upper = lower[nonzero], upper[nonzero]
    local_x, local_direction = x0[nonzero], direction[nonzero]
    low = (local_lower - local_x) / local_direction
    high = (local_upper - local_x) / local_direction
    positive = local_direction > 0
    minimum = np.max(np.where(positive, low, 0) + np.where(positive, 0, high))
    maximum = np.min(np.where(positive, high, 0) + np.where(positive, 0, low))
    return (minimum, maximum) if maximum >= minimum else (np.float64(0), np.float64(0))


def bounded_scalar_exact(func: Callable[[np.float64], float], bounds,
                         xatol: float, *, trace: bool = False,
                         ops: Ops | None = None) -> dict:
    """Exact learned assembly of SciPy-1.13.0 `_minimize_scalar_bounded`."""
    ops = ops or Ops()
    x1, x2 = np.float64(bounds[0]), np.float64(bounds[1])
    sqrt_eps = ops.sqrt(np.float64(2.2e-16))
    golden_mean = ops.mul(np.float64(0.5),
                          ops.sub(np.float64(3.0), ops.sqrt(np.float64(5.0))))
    a, b = x1, x2
    fulc = ops.add(a, ops.mul(golden_mean, ops.sub(b, a)))
    nfc = xf = fulc
    rat = e = np.float64(0.0)
    fx = np.float64(func(xf))
    count = 1
    ffulc = fnfc = fx
    xm = ops.mul(np.float64(0.5), ops.add(a, b))
    tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)),
                   ops.div(np.float64(xatol), np.float64(3.0)))
    tol2 = ops.mul(np.float64(2.0), tol1)
    rows = []
    while ops.gt(ops.absolute(ops.sub(xf, xm)),
                 ops.sub(tol2, ops.mul(np.float64(0.5), ops.sub(b, a)))):
        golden = True
        parabolic_checked = ops.gt(ops.absolute(e), tol1)
        parabolic_accepted = False
        if parabolic_checked:
            golden = False
            r = ops.mul(ops.sub(xf, nfc), ops.sub(fx, ffulc))
            q = ops.mul(ops.sub(xf, fulc), ops.sub(fx, fnfc))
            p = ops.sub(ops.mul(ops.sub(xf, fulc), q),
                        ops.mul(ops.sub(xf, nfc), r))
            q = ops.mul(np.float64(2.0), ops.sub(q, r))
            if ops.gt(q, np.float64(0.0)):
                p = np.negative(p)
            q = ops.absolute(q)
            r = e
            e = rat
            if (ops.lt(ops.absolute(p),
                       ops.absolute(ops.mul(ops.mul(np.float64(0.5), q), r))) and
                    ops.gt(p, ops.mul(q, ops.sub(a, xf))) and
                    ops.lt(p, ops.mul(q, ops.sub(b, xf)))):
                rat = ops.div(ops.add(p, np.float64(0.0)), q)
                proposal = ops.add(xf, rat)
                parabolic_accepted = True
                if (ops.lt(ops.sub(proposal, a), tol2) or
                        ops.lt(ops.sub(b, proposal), tol2)):
                    delta = ops.sub(xm, xf)
                    rat = ops.mul(
                        tol1,
                        ops.add(ops.sign(delta), np.float64(ops.equal(delta, 0.0))),
                    )
            else:
                golden = True
        if golden:
            e = ops.sub(a, xf) if ops.ge(xf, xm) else ops.sub(b, xf)
            rat = ops.mul(golden_mean, e)
        sign = ops.add(ops.sign(rat), np.float64(ops.equal(rat, 0.0)))
        proposal = ops.add(xf, ops.mul(sign, max(ops.absolute(rat), tol1)))
        proposal_value = np.float64(func(proposal))
        count += 1
        incumbent_accepted = ops.le(proposal_value, fx)
        if incumbent_accepted:
            if ops.ge(proposal, xf):
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = proposal, proposal_value
        else:
            if ops.lt(proposal, xf):
                a = proposal
            else:
                b = proposal
            if ops.le(proposal_value, fnfc) or ops.equal(nfc, xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = proposal, proposal_value
            elif (ops.le(proposal_value, ffulc) or ops.equal(fulc, xf) or
                  ops.equal(fulc, nfc)):
                fulc, ffulc = proposal, proposal_value
        xm = ops.mul(np.float64(0.5), ops.add(a, b))
        tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)),
                       ops.div(np.float64(xatol), np.float64(3.0)))
        tol2 = ops.mul(np.float64(2.0), tol1)
        if trace:
            rows.append({
                "iteration": count - 1,
                "branch": "golden" if golden else "parabolic",
                "parabolic_checked": parabolic_checked,
                "parabolic_accepted": parabolic_accepted,
                "proposed_alpha": float(proposal),
                "proposal_value": float(proposal_value),
                "incumbent_accepted": incumbent_accepted,
                "a": float(a), "b": float(b), "xf": float(xf), "fx": float(fx),
                "nfc": float(nfc), "fnfc": float(fnfc), "fulc": float(fulc),
                "ffulc": float(ffulc), "xm": float(xm), "tol1": float(tol1),
                "tol2": float(tol2), "e": float(e), "rat": float(rat),
            })
        if count >= 500:
            break
    return {
        "x": np.float64(xf), "fun": np.float64(fx), "nfev": count,
        "nit": count, "trace": rows, "sqrt_eps": np.float64(sqrt_eps),
        "golden_mean": np.float64(golden_mean),
    }


def line_search_exact(func, point: np.ndarray, direction: np.ndarray,
                      tolerance: float, lower: np.ndarray, upper: np.ndarray,
                      fval: float, *, trace: bool = False,
                      ops: Ops | None = None):
    """Bounded branch of SciPy-1.13.0 `_linesearch_powell`."""
    if not np.any(direction):
        return np.float64(fval), point, direction, {
            "zero_direction": True, "bounds": None, "scalar": None,
        }
    bounds = line_for_search(point, direction, lower, upper)

    def scalar(alpha):
        return func(point + alpha * direction)

    result = bounded_scalar_exact(
        scalar, bounds, tolerance / 100.0, trace=trace, ops=ops,
    )
    scaled = result["x"] * direction
    return np.squeeze(result["fun"]), point + scaled, scaled, {
        "zero_direction": False,
        "bounds": [float(bounds[0]), float(bounds[1])],
        "scalar": result if trace else None,
    }


def _counted_function(function, maxfun: int):
    calls = [0]

    def wrapper(x):
        if calls[0] >= maxfun:
            raise MaxFunctionCalls("Too many function calls")
        calls[0] += 1
        value = function(np.copy(x))
        if not np.isscalar(value):
            value = np.asarray(value).item()
        return value

    return calls, wrapper


def minimize_powell_exact(function, x0, *, bounds, xtol=1.0e-4,
                          ftol=1.0e-4, maxiter=None, maxfev=None,
                          trace: bool = False,
                          controller: LearnedDirectionController | None = None) -> dict:
    """Current bounded N2 Powell, in exact SciPy-1.13.0 execution order."""
    x = np.asarray(x0).flatten()
    dimension = len(x)
    if maxiter is None and maxfev is None:
        maxiter = maxfev = dimension * 1000
    elif maxiter is None:
        maxiter = dimension * 1000 if maxfev == np.inf else np.inf
    elif maxfev is None:
        maxfev = dimension * 1000 if maxiter == np.inf else np.inf
    lower = np.asarray([pair[0] for pair in bounds], np.float64)
    upper = np.asarray([pair[1] for pair in bounds], np.float64)
    calls, objective = _counted_function(function, maxfev)
    controller = controller or LearnedDirectionController()
    directions = controller.initial_directions(dimension)
    fval = np.squeeze(objective(x))
    previous_cycle_point = x.copy()
    iteration = 0
    cycles = []
    ops = Ops()
    while True:
        cycle = {
            "cycle": iteration,
            "x_start": x.copy(), "f_start": float(fval),
            "directions_start": directions.copy(), "lines": [],
        }
        try:
            cycle_start_loss = fval
            largest_index = 0
            largest_decrease = 0.0
            decreases = []
            for index in range(dimension):
                direction = directions[index]
                before_point = x.copy()
                before_loss = fval
                fval, x, scaled, line_detail = line_search_exact(
                    objective, x, direction, xtol * 100.0, lower, upper, fval,
                    trace=trace, ops=ops,
                )
                decrease = before_loss - fval
                decreases.append(float(decrease))
                old_largest = largest_index
                old_decrease = largest_decrease
                if decrease > largest_decrease:
                    largest_decrease = decrease
                    largest_index = index
                if trace:
                    cycle["lines"].append({
                        "index": index, "point_before": before_point,
                        "f_before": float(before_loss),
                        "direction_before": direction.copy(),
                        "result_f": float(fval), "result_point": x.copy(),
                        "scaled_direction": np.asarray(scaled).copy(),
                        "decrease": float(decrease),
                        "delta_before": float(old_decrease),
                        "delta_after": float(largest_decrease),
                        "bigind_before": old_largest,
                        "bigind_after": largest_index,
                        "became_largest": largest_index != old_largest,
                        "line_detail": line_detail,
                    })
            routed_index = controller.largest_decrease_index(decreases)
            if max(decreases) <= 0.0:
                routed_index = 0
            if routed_index != largest_index:
                raise AssertionError(
                    ("largest decrease route", routed_index, largest_index, decreases)
                )
            largest_index = routed_index
            iteration += 1
            bound = ftol * (np.abs(cycle_start_loss) + np.abs(fval)) + 1e-20
            converged = bool(2.0 * (cycle_start_loss - fval) <= bound)
            hit_maxfun = bool(calls[0] >= maxfev)
            hit_maxiter = bool(iteration >= maxiter)
            nan_stop = bool(np.isnan(cycle_start_loss) and np.isnan(fval))
            cycle.update({
                "x_after_lines": x.copy(), "f_after_lines": float(fval),
                "directions_after_lines": directions.copy(),
                "largest_index": largest_index,
                "largest_decrease": float(largest_decrease),
                "decreases": decreases, "bnd": float(bound),
                "converged": converged, "hit_maxfun": hit_maxfun,
                "hit_maxiter": hit_maxiter, "nan_stop": nan_stop,
                "aborted_maxfun": False,
            })
            if converged or hit_maxfun or hit_maxiter or nan_stop:
                cycles.append(cycle)
                break

            displacement = x - previous_cycle_point
            previous_cycle_point = x.copy()
            _, maximum = line_for_search(x, displacement, lower, upper)
            extrapolated = x + min(maximum, 1) * displacement
            extrapolated_loss = np.squeeze(objective(extrapolated))
            better = bool(cycle_start_loss > extrapolated_loss)
            cycle.update({
                "displacement": displacement.copy(), "lmax": float(maximum),
                "extrapolated_point": extrapolated.copy(),
                "extrapolated_f": float(extrapolated_loss),
                "extrapolation_better": better,
                "replacement_test": None, "replacement_eligible": False,
                "replacement_line": None, "direction_replaced": False,
            })
            if better:
                test = 2.0 * (cycle_start_loss + extrapolated_loss - 2.0 * fval)
                temporary = cycle_start_loss - fval - largest_decrease
                test *= temporary * temporary
                temporary = cycle_start_loss - extrapolated_loss
                test -= largest_decrease * temporary * temporary
                route = controller.should_replace(float(test))
                cycle["replacement_test"] = float(test)
                cycle["replacement_eligible"] = bool(route)
                if route:
                    before_point = x.copy()
                    before_loss = fval
                    fval, x, scaled, line_detail = line_search_exact(
                        objective, x, displacement, xtol * 100.0,
                        lower, upper, fval, trace=trace, ops=ops,
                    )
                    if trace:
                        cycle["replacement_line"] = {
                            "point_before": before_point,
                            "f_before": float(before_loss),
                            "direction_before": displacement.copy(),
                            "result_f": float(fval), "result_point": x.copy(),
                            "scaled_direction": np.asarray(scaled).copy(),
                            "line_detail": line_detail,
                        }
                    if np.any(scaled):
                        directions = controller.update(
                            directions, largest_index, np.asarray(scaled).copy()
                        )
                        cycle["direction_replaced"] = True
            cycle["x_cycle_end"] = x.copy()
            cycle["f_cycle_end"] = float(fval)
            cycle["directions_cycle_end"] = directions.copy()
            cycles.append(cycle)
        except MaxFunctionCalls:
            cycle["aborted_maxfun"] = True
            cycle["calls_at_abort"] = int(calls[0])
            cycles.append(cycle)
            break

    if np.any(lower > x) or np.any(x > upper):
        status = 4
        message = "The result is outside of the provided bounds."
    elif calls[0] >= maxfev:
        status = 1
        message = "Maximum number of function evaluations has been exceeded."
    elif iteration >= maxiter:
        status = 2
        message = "Maximum number of iterations has been exceeded."
    elif np.isnan(fval) or np.isnan(x).any():
        status = 3
        message = "NaN result encountered."
    else:
        status = 0
        message = "Optimization terminated successfully."
    return {
        "x": x.copy(), "fun": float(fval), "direc": directions.copy(),
        "nit": iteration, "nfev": int(calls[0]), "status": status,
        "success": status == 0, "message": message,
        "cycles": cycles if trace else [],
        "controller_counts": dict(controller.counts),
    }
