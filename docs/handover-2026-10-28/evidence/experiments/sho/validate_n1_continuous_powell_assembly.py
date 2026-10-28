"""Validate the complete bounded multistart Powell proposal used by N1.

This is deliberately a small replay: the first archived d=4 and d=5
continuous calls.  The production Gaussian surrogate is left untouched.  A
source-order Powell transcription replaces only its scalar line search with
the already recovered exact bounded-scalar assembly and its direction control
with the existing learned N2 direction routers.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import _optimize as scipy_opt
from scipy.spatial import cKDTree
from scipy.stats import qmc


SEARCH_SEED = 8_730_000


class GaussianLocalResponse:
    """Literal production callback, copied here to avoid loading unrelated Torch code."""
    def __init__(self, points, losses):
        self.points = points
        self.values = np.log10(np.maximum(losses, 1.0e-300))
        self.tree = cKDTree(points)

    def predict(self, query):
        query = np.atleast_2d(query)
        count = min(24, len(self.points))
        distances, indices = self.tree.query(query, k=count)
        if count == 1:
            distances = distances[:, None]; indices = indices[:, None]
        bandwidth = np.maximum(distances[:, -1:], 1.0e-12)
        weights = np.exp(-9.0 * np.square(distances / bandwidth)) + 1.0e-15
        return np.sum(weights * self.values[indices], axis=1) / np.sum(weights, axis=1)


OUT = Path("outputs/sho/stage1B/n1_continuous_powell_assembly")
CASES = (
    (Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases/companion_31"),
     "stopping_state.npz", "d4_sho"),
    (Path("outputs/bouncing_ball/reflection_peelback/cases/case_00"),
     "n1_state.npz", "d5_reflection_bouncing_ball"),
)
PROTECTED = {
    "bounded_scalar_report": Path("outputs/sho/stage1B/n1_bounded_scalar_minimizer/report.json"),
    "direction_router": Path("outputs/sho/stage1B/n2_direction_controller/local_direction_routers.pt"),
    "continuous_wrapper": Path("experiments/sho/continuous_surrogate_edirect.py"),
    "surrogate": Path("experiments/sho/run_stage1b_edirect_vs_direct.py"),
}


def bits_equal(a, b) -> bool:
    return bool(np.asarray([a], np.float64).view(np.uint64)[0] ==
                np.asarray([b], np.float64).view(np.uint64)[0])


class Ops:
    """Source-ordered NumPy execution of already validated scalar primitives."""
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


def bounded_trace(func, bounds, xatol: float, ops: Ops) -> dict:
    """Exact recovered SciPy-1.13 bounded-scalar state machine."""
    x1, x2 = np.float64(bounds[0]), np.float64(bounds[1])
    sqrt_eps = ops.sqrt(np.float64(2.2e-16))
    golden_mean = ops.mul(0.5, ops.sub(3.0, ops.sqrt(5.0)))
    a, b = x1, x2
    fulc = ops.add(a, ops.mul(golden_mean, ops.sub(b, a)))
    nfc = xf = fulc
    rat = e = np.float64(0.0)
    fx = np.float64(func(xf)); num = 1
    ffulc = fnfc = fx
    xm = ops.mul(0.5, ops.add(a, b))
    tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)), ops.div(xatol, 3.0))
    tol2 = ops.mul(2.0, tol1)
    rows = []
    while ops.gt(ops.absolute(ops.sub(xf, xm)),
                 ops.sub(tol2, ops.mul(0.5, ops.sub(b, a)))):
        golden = True
        parabolic_checked = ops.gt(ops.absolute(e), tol1)
        parabolic_accepted = False
        if parabolic_checked:
            golden = False
            r = ops.mul(ops.sub(xf, nfc), ops.sub(fx, ffulc))
            q = ops.mul(ops.sub(xf, fulc), ops.sub(fx, fnfc))
            p = ops.sub(ops.mul(ops.sub(xf, fulc), q), ops.mul(ops.sub(xf, nfc), r))
            q = ops.mul(2.0, ops.sub(q, r))
            if ops.gt(q, 0.0): p = np.negative(p)
            q = ops.absolute(q); r = e; e = rat
            if (ops.lt(ops.absolute(p), ops.absolute(ops.mul(ops.mul(0.5, q), r))) and
                    ops.gt(p, ops.mul(q, ops.sub(a, xf))) and
                    ops.lt(p, ops.mul(q, ops.sub(b, xf)))):
                rat = ops.div(ops.add(p, 0.0), q)
                x = ops.add(xf, rat); parabolic_accepted = True
                if ops.lt(ops.sub(x, a), tol2) or ops.lt(ops.sub(b, x), tol2):
                    delta = ops.sub(xm, xf)
                    rat = ops.mul(tol1, ops.add(ops.sign(delta), float(ops.equal(delta, 0.0))))
            else:
                golden = True
        if golden:
            e = ops.sub(a, xf) if ops.ge(xf, xm) else ops.sub(b, xf)
            rat = ops.mul(golden_mean, e)
        si = ops.add(ops.sign(rat), float(ops.equal(rat, 0.0)))
        x = ops.add(xf, ops.mul(si, max(ops.absolute(rat), tol1)))
        fu = np.float64(func(x)); num += 1
        accepted = ops.le(fu, fx)
        if accepted:
            if ops.ge(x, xf): a = xf
            else: b = xf
            fulc, ffulc = nfc, fnfc; nfc, fnfc = xf, fx; xf, fx = x, fu
        else:
            if ops.lt(x, xf): a = x
            else: b = x
            if ops.le(fu, fnfc) or ops.equal(nfc, xf):
                fulc, ffulc = nfc, fnfc; nfc, fnfc = x, fu
            elif ops.le(fu, ffulc) or ops.equal(fulc, xf) or ops.equal(fulc, nfc):
                fulc, ffulc = x, fu
        xm = ops.mul(0.5, ops.add(a, b))
        tol1 = ops.add(ops.mul(sqrt_eps, ops.absolute(xf)), ops.div(xatol, 3.0))
        tol2 = ops.mul(2.0, tol1)
        rows.append({"branch": "golden" if golden else "parabolic",
                     "parabolic_checked": parabolic_checked,
                     "parabolic_accepted": parabolic_accepted,
                     "incumbent_accepted": accepted})
        if num >= 500: break
    return {"x": float(xf), "fun": float(fx), "nfev": num, "nit": num,
            "trace": rows, "sqrt_eps": float(sqrt_eps), "golden_mean": float(golden_mean)}


class NumpyDirectionController:
    """Execute the locked tiny routers from their audited JSON weights."""
    def __init__(self):
        payload = json.loads(Path(
            "outputs/sho/stage1B/n2_direction_controller/local_operator_report.json"
        ).read_text())
        self.weights = {}
        for row in payload["reports"]:
            self.weights[row["name"]] = (
                np.asarray(row["weights"]["linear.weight"], np.float64),
                np.asarray(row["weights"]["linear.bias"], np.float64))
        self.counts = {k: 0 for k in ("basis", "basis_errors", "rank", "rank_errors",
                                      "negative", "negative_errors", "update", "update_errors")}

    def _route(self, name, values):
        weight, bias = self.weights[name]
        return int(np.argmax(weight @ np.asarray(values, np.float64) + bias))

    def initial_directions(self, dimension):
        result = []
        for i in range(dimension):
            row = np.empty(dimension, np.float64)
            for j in range(dimension):
                got = self._route("basis_membership", [float(i == j)])
                self.counts["basis"] += 1
                self.counts["basis_errors"] += int(got != int(i == j))
                row[j] = float(got)
            result.append(row)
        return result

    def _negative(self, value):
        state = -1 if value < 0.0 else (1 if value > 0.0 else 0)
        category = {-1: 0, 0: 1, 1: 2}[state]
        got = bool(self._route("strict_negative", np.eye(3)[category]))
        self.counts["negative"] += 1
        self.counts["negative_errors"] += int(got != (value < 0.0))
        return got

    def largest_decrease_index(self, decreases):
        current = 0
        for candidate in range(1, len(decreases)):
            got = self._negative(float(decreases[current] - decreases[candidate]))
            self.counts["rank"] += 1
            self.counts["rank_errors"] += int(got != (decreases[candidate] > decreases[current]))
            if got: current = candidate
        return current

    def should_replace(self, test): return self._negative(test)

    def update(self, directions, biggest, displacement):
        old = [np.asarray(v).copy() for v in directions]; last = len(old) - 1; result = []
        for index, vector in enumerate(old):
            role = self._route("direction_update_role", [float(index == biggest), float(index == last)])
            expected = 2 if index == last else (1 if index == biggest else 0)
            self.counts["update"] += 1
            self.counts["update_errors"] += int(role != expected)
            result.append(vector if role == 0 else (old[last].copy() if role == 1 else displacement.copy()))
        return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_assembled(func, p, xi, tol, lower, upper, fval, ops):
    if not np.any(xi):
        return np.float64(fval), p, xi, {"zero_direction": True, "scalar": None}
    bound = scipy_opt._line_for_search(p, xi, lower, upper)

    def scalar(alpha):
        return func(p + np.float64(alpha) * xi)

    result = bounded_trace(scalar, bound, tol / 100.0, ops)
    scaled = np.float64(result["x"]) * xi
    return (np.float64(result["fun"]), p + scaled, scaled,
            {"zero_direction": False, "scalar": result})


def _array_bits_equal(a, b) -> bool:
    aa, bb = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return aa.shape == bb.shape and bool(np.array_equal(aa.view(np.uint64), bb.view(np.uint64)))


def powell_trace(func, start, assembled: bool, control=None) -> dict:
    """SciPy 1.13 `_minimize_powell`, restricted to production bounded calls."""
    maxfun, maxiter, xtol, ftol = 2000, 100, np.float64(1e-10), np.float64(1e-12)
    x = np.asarray(start, np.float64).flatten()
    n = len(x)
    fcalls, wrapped = scipy_opt._wrap_scalar_function_maxfun_validation(func, (), maxfun)
    lower, upper = np.zeros(n, np.float64), np.ones(n, np.float64)
    if assembled:
        directions = np.asarray(control.initial_directions(n), np.float64)
    else:
        directions = np.eye(n, dtype=np.float64)
    fval = np.float64(np.squeeze(wrapped(x)))
    x1 = x.copy()
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
            fx = fval
            bigind = 0
            delta = np.float64(0.0)
            decreases = []
            for i in range(n):
                direction_before = directions[i].copy()
                point_before, fx2 = x.copy(), fval
                if assembled:
                    fval, x, scaled, line_detail = _line_assembled(
                        wrapped, x, direction_before, xtol * 100.0,
                        lower, upper, fval, ops)
                else:
                    fval, x, scaled = scipy_opt._linesearch_powell(
                        wrapped, x, direction_before, tol=xtol * 100.0,
                        lower_bound=lower, upper_bound=upper, fval=fval)
                    line_detail = None
                decrease = np.float64(fx2 - fval)
                decreases.append(float(decrease))
                previous_delta, previous_bigind = delta, bigind
                if decrease > delta:
                    delta, bigind = decrease, i
                cycle["lines"].append({
                    "index": i, "point_before": point_before,
                    "f_before": float(fx2), "direction_before": direction_before,
                    "result_f": float(fval), "result_point": x.copy(),
                    "scaled_direction": np.asarray(scaled).copy(),
                    "decrease": float(decrease), "delta_before": float(previous_delta),
                    "delta_after": float(delta), "bigind_before": previous_bigind,
                    "bigind_after": bigind, "became_largest": bigind != previous_bigind,
                    "line_detail": line_detail,
                })

            # The learned router implements the same strict, first-maximum
            # reduction.  A no-positive-decrease cycle retains production's 0.
            if assembled:
                routed_bigind = control.largest_decrease_index(decreases)
                if max(decreases) <= 0.0:
                    routed_bigind = 0
                if routed_bigind != bigind:
                    raise AssertionError(("largest decrease", routed_bigind, bigind, decreases))
                bigind = routed_bigind

            iteration += 1
            bnd = np.float64(ftol * (np.abs(fx) + np.abs(fval)) + 1e-20)
            converged = bool(2.0 * (fx - fval) <= bnd)
            hit_maxfun = bool(fcalls[0] >= maxfun)
            hit_maxiter = bool(iteration >= maxiter)
            nan_stop = bool(np.isnan(fx) and np.isnan(fval))
            cycle.update({
                "x_after_lines": x.copy(), "f_after_lines": float(fval),
                "directions_after_lines": directions.copy(), "largest_index": bigind,
                "largest_decrease": float(delta), "decreases": decreases,
                "bnd": float(bnd), "converged": converged,
                "hit_maxfun": hit_maxfun, "hit_maxiter": hit_maxiter,
                "nan_stop": nan_stop, "aborted_maxfun": False,
            })
            if converged or hit_maxfun or hit_maxiter or nan_stop:
                cycles.append(cycle)
                break

            displacement = x - x1
            x1 = x.copy()
            _, lmax = scipy_opt._line_for_search(x, displacement, lower, upper)
            extrapolated = x + min(lmax, 1) * displacement
            extrapolated_f = np.float64(np.squeeze(wrapped(extrapolated)))
            better = bool(fx > extrapolated_f)
            cycle.update({
                "displacement": displacement.copy(), "lmax": float(lmax),
                "extrapolated_point": extrapolated.copy(),
                "extrapolated_f": float(extrapolated_f), "extrapolation_better": better,
                "replacement_test": None, "replacement_eligible": False,
                "replacement_line": None, "direction_replaced": False,
            })
            if better:
                test = np.float64(2.0 * (fx + extrapolated_f - 2.0 * fval))
                temp = np.float64(fx - fval - delta)
                test = np.float64(test * temp * temp)
                temp = np.float64(fx - extrapolated_f)
                test = np.float64(test - delta * temp * temp)
                teacher_route = bool(test < 0.0)
                route = control.should_replace(float(test)) if assembled else teacher_route
                cycle["replacement_test"] = float(test)
                cycle["replacement_eligible"] = bool(route)
                if route:
                    point_before, f_before = x.copy(), fval
                    if assembled:
                        fval, x, scaled, line_detail = _line_assembled(
                            wrapped, x, displacement, xtol * 100.0,
                            lower, upper, fval, ops)
                    else:
                        fval, x, scaled = scipy_opt._linesearch_powell(
                            wrapped, x, displacement, tol=xtol * 100.0,
                            lower_bound=lower, upper_bound=upper, fval=fval)
                        line_detail = None
                    cycle["replacement_line"] = {
                        "point_before": point_before, "f_before": float(f_before),
                        "direction_before": displacement.copy(), "result_f": float(fval),
                        "result_point": x.copy(), "scaled_direction": np.asarray(scaled).copy(),
                        "line_detail": line_detail,
                    }
                    if np.any(scaled):
                        if assembled:
                            directions = np.asarray(control.update(
                                [row.copy() for row in directions], bigind,
                                np.asarray(scaled).copy()), np.float64)
                        else:
                            directions[bigind] = directions[-1]
                            directions[-1] = scaled
                        cycle["direction_replaced"] = True
            cycle["x_cycle_end"] = x.copy()
            cycle["f_cycle_end"] = float(fval)
            cycle["directions_cycle_end"] = directions.copy()
            cycles.append(cycle)
        except scipy_opt._MaxFuncCallError:
            cycle["aborted_maxfun"] = True
            cycle["calls_at_abort"] = int(fcalls[0])
            cycles.append(cycle)
            break

    status = 1 if fcalls[0] >= maxfun else (2 if iteration >= maxiter else 0)
    return {
        "x": x.copy(), "fun": float(fval), "directions": directions.copy(),
        "nit": iteration, "nfev": int(fcalls[0]), "status": status,
        "success": status == 0, "cycles": cycles,
    }


def compare_line(teacher, assembled, counts: Counter, prefix: str) -> None:
    float_keys = ("f_before", "result_f")
    array_keys = ("point_before", "direction_before", "result_point", "scaled_direction")
    for key in float_keys:
        counts[prefix + key + "_bit_errors"] += int(not bits_equal(teacher[key], assembled[key]))
    for key in array_keys:
        counts[prefix + key + "_bit_errors"] += int(not _array_bits_equal(teacher[key], assembled[key]))


def compare_powell(teacher: dict, assembled: dict) -> Counter:
    c = Counter(starts=1, teacher_cycles=len(teacher["cycles"]),
                assembled_cycles=len(assembled["cycles"]))
    for key in ("fun",):
        c[key + "_bit_errors"] += int(not bits_equal(teacher[key], assembled[key]))
    for key in ("x", "directions"):
        c[key + "_bit_errors"] += int(not _array_bits_equal(teacher[key], assembled[key]))
    for key in ("nit", "nfev", "status", "success"):
        c[key + "_errors"] += int(teacher[key] != assembled[key])
    c["cycle_length_errors"] += int(len(teacher["cycles"]) != len(assembled["cycles"]))
    for tc, ac in zip(teacher["cycles"], assembled["cycles"]):
        c["cycles"] += 1
        for key in ("x_start", "directions_start"):
            c[key + "_bit_errors"] += int(not _array_bits_equal(tc[key], ac[key]))
        c["f_start_bit_errors"] += int(not bits_equal(tc["f_start"], ac["f_start"]))
        c["line_count_errors"] += int(len(tc["lines"]) != len(ac["lines"]))
        for tl, al in zip(tc["lines"], ac["lines"]):
            c["coordinate_lines"] += 1
            compare_line(tl, al, c, "line_")
            for key in ("decrease", "delta_before", "delta_after"):
                c[key + "_bit_errors"] += int(not bits_equal(tl[key], al[key]))
            for key in ("index", "bigind_before", "bigind_after", "became_largest"):
                c[key + "_errors"] += int(tl[key] != al[key])
        for key in ("aborted_maxfun",):
            c[key + "_errors"] += int(tc.get(key) != ac.get(key))
        if tc.get("aborted_maxfun") or ac.get("aborted_maxfun"):
            c["maxfun_aborts"] += 1
            c["calls_at_abort_errors"] += int(tc.get("calls_at_abort") != ac.get("calls_at_abort"))
            continue
        for key in ("f_after_lines", "largest_decrease", "bnd"):
            c[key + "_bit_errors"] += int(not bits_equal(tc[key], ac[key]))
        for key in ("x_after_lines", "directions_after_lines"):
            c[key + "_bit_errors"] += int(not _array_bits_equal(tc[key], ac[key]))
        for key in ("largest_index", "converged", "hit_maxfun", "hit_maxiter", "nan_stop"):
            c[key + "_errors"] += int(tc[key] != ac[key])
        c["convergence_routes"] += 1
        if "displacement" not in tc or "displacement" not in ac:
            c["extrapolation_presence_errors"] += int(("displacement" in tc) != ("displacement" in ac))
            continue
        c["extrapolation_routes"] += 1
        for key in ("displacement", "extrapolated_point"):
            c[key + "_bit_errors"] += int(not _array_bits_equal(tc[key], ac[key]))
        for key in ("lmax", "extrapolated_f"):
            c[key + "_bit_errors"] += int(not bits_equal(tc[key], ac[key]))
        for key in ("extrapolation_better", "replacement_eligible", "direction_replaced"):
            c[key + "_errors"] += int(tc[key] != ac[key])
        if tc["replacement_test"] is not None or ac["replacement_test"] is not None:
            c["replacement_tests"] += 1
            c["replacement_test_bit_errors"] += int(not bits_equal(
                tc["replacement_test"], ac["replacement_test"]))
        if tc["replacement_line"] is not None or ac["replacement_line"] is not None:
            c["replacement_lines"] += 1
            if (tc["replacement_line"] is None) != (ac["replacement_line"] is None):
                c["replacement_line_presence_errors"] += 1
            else:
                compare_line(tc["replacement_line"], ac["replacement_line"], c, "replacement_line_")
        for key in ("x_cycle_end", "directions_cycle_end"):
            c[key + "_bit_errors"] += int(not _array_bits_equal(tc[key], ac[key]))
        c["f_cycle_end_bit_errors"] += int(not bits_equal(tc["f_cycle_end"], ac["f_cycle_end"]))
    return c


def run() -> None:
    started = time.monotonic()
    before = {name: sha256(path) for name, path in PROTECTED.items()}
    aggregate = Counter()
    case_reports = []
    control = NumpyDirectionController()
    for case_dir, state_name, label in CASES:
        saved = json.loads((case_dir / "continuous.json").read_text())[0]
        with np.load(case_dir / state_name, allow_pickle=False) as data:
            points = np.asarray(data["points"], np.float64)[:int(saved["site_count"])]
            losses = np.asarray(data["losses"], np.float64)[:int(saved["site_count"])]
        d = points.shape[1]
        surface = GaussianLocalResponse(points, losses)

        def surrogate(u):
            return float(surface.predict(np.asarray(u, np.float64)[None])[0])

        n = 2 * d
        sobol = qmc.Sobol(d, scramble=True, seed=SEARCH_SEED).random_base2(
            int(math.ceil(math.log2(n))))[:n]
        regenerated_starts = np.vstack((points[int(np.argmin(losses))], sobol))
        saved_starts = np.asarray([row["start"] for row in saved["starts"]], np.float64)
        start_bit_errors = int(np.count_nonzero(
            regenerated_starts.view(np.uint64) != saved_starts.view(np.uint64)))
        per_start = []
        endpoints = Counter()
        teacher_results, assembled_results = [], []
        for start, saved_result in zip(regenerated_starts, saved["starts"]):
            teacher = powell_trace(surrogate, start, assembled=False)
            assembled = powell_trace(surrogate, start, assembled=True, control=control)
            comparison = compare_powell(teacher, assembled)
            aggregate.update(comparison)
            teacher_results.append(teacher)
            assembled_results.append(assembled)
            expected_point = np.asarray(saved_result["point"], np.float64)
            endpoints["point_bit_errors"] += int(not _array_bits_equal(teacher["x"], expected_point))
            endpoints["prediction_bit_errors"] += int(not bits_equal(
                teacher["fun"], saved_result["prediction"]))
            for key in ("nfev", "nit", "success"):
                endpoints[key + "_errors"] += int(teacher[key] != saved_result[key])
            per_start.append({
                "start_id": saved_result["start_id"], "nfev": teacher["nfev"],
                "cycles": teacher["nit"], "success": teacher["success"],
                "coordinate_lines": comparison["coordinate_lines"],
                "replacement_lines": comparison["replacement_lines"],
                "maxfun_abort": bool(comparison["maxfun_aborts"]),
                "replay_errors": sum(v for k, v in comparison.items() if k.endswith("errors")),
            })
        teacher_winner = min(range(len(teacher_results)), key=lambda i: (teacher_results[i]["fun"], i))
        assembled_winner = min(range(len(assembled_results)), key=lambda i: (assembled_results[i]["fun"], i))
        teacher_point = teacher_results[teacher_winner]["x"]
        assembled_point = assembled_results[assembled_winner]["x"]
        teacher_duplicate = bool(np.any(np.all(points == teacher_point, axis=1)))
        assembled_duplicate = bool(np.any(np.all(points == assembled_point, axis=1)))
        final_errors = {
            "start_generation_bit_errors": start_bit_errors,
            "teacher_saved_endpoint_errors": int(sum(endpoints.values())),
            "winner_identity_errors": int(teacher_winner != assembled_winner),
            "saved_winner_identity_errors": int(teacher_winner != saved["chosen_start"]),
            "proposal_bit_errors": int(not _array_bits_equal(teacher_point, assembled_point)),
            "saved_proposal_bit_errors": int(not _array_bits_equal(
                teacher_point, np.asarray(saved["point"], np.float64))),
            "duplicate_route_errors": int(teacher_duplicate != assembled_duplicate),
            "saved_duplicate_route_errors": int(teacher_duplicate != saved["exact_duplicate_rejected"]),
        }
        case_reports.append({
            "label": label, "source": str(case_dir / "continuous.json"),
            "case_id": saved["case_id"], "iteration": saved["iteration"],
            "dimension": d, "site_count": len(points), "starts": len(saved_starts),
            "per_start": per_start, "saved_endpoint_comparison": dict(endpoints),
            "teacher_winner": teacher_winner, "assembled_winner": assembled_winner,
            "duplicate_rejected": teacher_duplicate, "final_comparison": final_errors,
        })

    after = {name: sha256(path) for name, path in PROTECTED.items()}
    state_errors = sum(v for k, v in aggregate.items() if k.endswith("errors"))
    final_errors = sum(sum(row["final_comparison"].values()) for row in case_reports)
    saved_endpoint_errors = sum(row["final_comparison"]["teacher_saved_endpoint_errors"]
                                for row in case_reports)
    controller_errors = sum(v for k, v in control.counts.items() if k.endswith("errors"))
    exact = (state_errors == final_errors == saved_endpoint_errors == controller_errors == 0
             and before == after)
    result = {
        "component": "n1_bounded_multistart_powell_continuous_proposal",
        "status": "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_STATE" if exact else "FIRST_MISMATCH",
        "representative_scope": {
            "calls": len(case_reports), "dimensions": [4, 5],
            "note": "One existing archived production call per dimension; no full-corpus regression.",
        },
        "replay": dict(aggregate),
        "total_internal_state_errors": state_errors,
        "controller_counts": control.counts,
        "controller_errors": controller_errors,
        "cases": case_reports,
        "remaining_deterministic_bookkeeping": [
            "Powell cycle/line-loop counters and maxfev/maxiter accounting",
            "variable-length vector/direction storage, copies, indexing, and mutation",
            "surrogate callback invocation and function-evaluation counter",
            "bounded interval indexing/reductions and scalar state storage",
            "multistart result list and canonical start IDs",
            "exact evaluated-site row-equality duplicate scan",
        ],
        "next_unresolved_original_audit": {
            "operation": 60,
            "name": "system-independent N1 handoff/stop route",
            "note": "Operations 55-59 are accepted assemblies or deterministic bookkeeping; the repository best_loss<=1e-4 wrapper remains unresolved.",
        },
        "protected_dependencies": {
            name: {"path": str(PROTECTED[name]), "sha256": before[name],
                   "unchanged": before[name] == after[name]} for name in PROTECTED
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if before != after:
        raise SystemExit("protected dependency changed")


if __name__ == "__main__":
    run()
