#!/usr/bin/env python3
"""Validate the recovered Powell assembly on current d=5 archive-N2 branches.

Run this under the recorded NumPy-1.26.4/SciPy-1.13.0 environment.  The
teacher path calls SciPy's private 1.13.0 line search, while the candidate path
uses only the recovered exact assembly.  Both receive the current physical
one-step objective and current archive-derived N2 box.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.optimize import _optimize as scipy_opt
import torch

from experiments import expand_chronological_archive_handoff_dataset as expanded
from experiments import validate_d5_production_pipeline as production
from experiments.n2_exact_powell_assembly import (
    LearnedDirectionController,
    minimize_powell_exact,
)


OUT = Path("outputs/n2_current_archive_powell_assembly")
LABELS = Path(
    "outputs/chronological_archive_handoff_supervision_expanded/"
    "expanded_archive_site_labels.jsonl"
)
PROTECTED = (
    Path("outputs/sho/stage1B/n2_direction_controller/local_direction_routers.pt"),
    Path("outputs/sho/stage1B/n2_direction_controller/local_operator_report.json"),
    Path("outputs/kepler/n3_variable_power_adjoint/n3_variable_power_adjoint.pt"),
)
XTOL = 1.0e-12
FTOL = 1.0e-15
MAXITER = 200
MAXFEV = 1500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a = np.asarray([left], np.float64).view(np.uint64)
    b = np.asarray([right], np.float64).view(np.uint64)
    return bool(np.array_equal(a, b))


def array_bits_equal(left, right) -> bool:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def native_powell_trace(function, x0, lower, upper) -> dict:
    """Traced SciPy-1.13.0 Powell outer loop with native private line search."""
    x = np.asarray(x0).flatten()
    dimension = len(x)
    calls, objective = scipy_opt._wrap_scalar_function_maxfun_validation(
        function, (), MAXFEV
    )
    directions = np.eye(dimension, dtype=float)
    fval = np.squeeze(objective(x))
    previous_cycle_point = x.copy()
    iteration = 0
    cycles = []
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
            for index in range(dimension):
                direction = directions[index]
                before_point = x.copy()
                before_loss = fval
                calls_before = int(calls[0])
                fval, x, scaled = scipy_opt._linesearch_powell(
                    objective, x, direction, tol=XTOL * 100.0,
                    lower_bound=lower, upper_bound=upper, fval=fval,
                )
                decrease = before_loss - fval
                old_index, old_decrease = largest_index, largest_decrease
                if decrease > largest_decrease:
                    largest_decrease = decrease
                    largest_index = index
                cycle["lines"].append({
                    "index": index, "point_before": before_point,
                    "f_before": float(before_loss),
                    "direction_before": direction.copy(),
                    "result_f": float(fval), "result_point": x.copy(),
                    "scaled_direction": np.asarray(scaled).copy(),
                    "decrease": float(decrease),
                    "delta_before": float(old_decrease),
                    "delta_after": float(largest_decrease),
                    "bigind_before": old_index, "bigind_after": largest_index,
                    "became_largest": largest_index != old_index,
                    "scalar_objective_evaluations": int(calls[0]) - calls_before,
                })
            iteration += 1
            bound = FTOL * (np.abs(cycle_start_loss) + np.abs(fval)) + 1e-20
            converged = bool(2.0 * (cycle_start_loss - fval) <= bound)
            hit_maxfun = bool(calls[0] >= MAXFEV)
            hit_maxiter = bool(iteration >= MAXITER)
            nan_stop = bool(np.isnan(cycle_start_loss) and np.isnan(fval))
            cycle.update({
                "x_after_lines": x.copy(), "f_after_lines": float(fval),
                "directions_after_lines": directions.copy(),
                "largest_index": largest_index,
                "largest_decrease": float(largest_decrease), "bnd": float(bound),
                "converged": converged, "hit_maxfun": hit_maxfun,
                "hit_maxiter": hit_maxiter, "nan_stop": nan_stop,
                "aborted_maxfun": False,
            })
            if converged or hit_maxfun or hit_maxiter or nan_stop:
                cycles.append(cycle)
                break
            displacement = x - previous_cycle_point
            previous_cycle_point = x.copy()
            _, maximum = scipy_opt._line_for_search(x, displacement, lower, upper)
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
                cycle["replacement_test"] = float(test)
                cycle["replacement_eligible"] = bool(test < 0.0)
                if test < 0.0:
                    before_point = x.copy()
                    before_loss = fval
                    calls_before = int(calls[0])
                    fval, x, scaled = scipy_opt._linesearch_powell(
                        objective, x, displacement, tol=XTOL * 100.0,
                        lower_bound=lower, upper_bound=upper, fval=fval,
                    )
                    cycle["replacement_line"] = {
                        "point_before": before_point, "f_before": float(before_loss),
                        "direction_before": displacement.copy(),
                        "result_f": float(fval), "result_point": x.copy(),
                        "scaled_direction": np.asarray(scaled).copy(),
                        "scalar_objective_evaluations": int(calls[0]) - calls_before,
                    }
                    if np.any(scaled):
                        directions[largest_index] = directions[-1]
                        directions[-1] = scaled
                        cycle["direction_replaced"] = True
            cycle["x_cycle_end"] = x.copy()
            cycle["f_cycle_end"] = float(fval)
            cycle["directions_cycle_end"] = directions.copy()
            cycles.append(cycle)
        except scipy_opt._MaxFuncCallError:
            cycle["aborted_maxfun"] = True
            cycle["calls_at_abort"] = int(calls[0])
            cycles.append(cycle)
            break
    status = 1 if calls[0] >= MAXFEV else (2 if iteration >= MAXITER else 0)
    message = (
        "Maximum number of function evaluations has been exceeded." if status == 1
        else "Maximum number of iterations has been exceeded." if status == 2
        else "Optimization terminated successfully."
    )
    return {
        "x": x.copy(), "fun": float(fval), "direc": directions.copy(),
        "nit": iteration, "nfev": int(calls[0]), "status": status,
        "success": status == 0, "message": message, "cycles": cycles,
    }


def compare_line(native: dict, learned: dict, counts: Counter, prefix: str) -> None:
    for key in ("f_before", "result_f", "decrease", "delta_before", "delta_after"):
        if key in native and key in learned:
            counts[prefix + key + "_bit_errors"] += int(
                not bits_equal(native[key], learned[key])
            )
    for key in ("point_before", "direction_before", "result_point", "scaled_direction"):
        counts[prefix + key + "_bit_errors"] += int(
            not array_bits_equal(native[key], learned[key])
        )
    for key in ("index", "bigind_before", "bigind_after", "became_largest"):
        if key in native and key in learned:
            counts[prefix + key + "_errors"] += int(native[key] != learned[key])
    scalar_count = learned.get("line_detail", {}).get("scalar")
    scalar_count = 0 if scalar_count is None else int(scalar_count["nfev"])
    counts[prefix + "scalar_count_errors"] += int(
        native["scalar_objective_evaluations"] != scalar_count
    )
    counts["scalar_proposals"] += native["scalar_objective_evaluations"]


def compare_powell(native: dict, learned: dict) -> Counter:
    counts = Counter(branches=1)
    for key in ("fun",):
        counts[key + "_bit_errors"] += int(not bits_equal(native[key], learned[key]))
    for key in ("x", "direc"):
        counts[key + "_bit_errors"] += int(
            not array_bits_equal(native[key], learned[key])
        )
    for key in ("nit", "nfev", "status", "success", "message"):
        counts[key + "_errors"] += int(native[key] != learned[key])
    counts["cycle_length_errors"] += int(len(native["cycles"]) != len(learned["cycles"]))
    for left, right in zip(native["cycles"], learned["cycles"]):
        counts["cycles"] += 1
        for key in ("x_start", "directions_start"):
            counts[key + "_bit_errors"] += int(
                not array_bits_equal(left[key], right[key])
            )
        counts["f_start_bit_errors"] += int(not bits_equal(left["f_start"], right["f_start"]))
        counts["line_count_errors"] += int(len(left["lines"]) != len(right["lines"]))
        for native_line, learned_line in zip(left["lines"], right["lines"]):
            counts["coordinate_lines"] += 1
            compare_line(native_line, learned_line, counts, "line_")
        counts["aborted_maxfun_errors"] += int(
            left.get("aborted_maxfun") != right.get("aborted_maxfun")
        )
        if left.get("aborted_maxfun") or right.get("aborted_maxfun"):
            counts["maxfun_aborts"] += 1
            counts["calls_at_abort_errors"] += int(
                left.get("calls_at_abort") != right.get("calls_at_abort")
            )
            continue
        for key in ("f_after_lines", "largest_decrease", "bnd"):
            counts[key + "_bit_errors"] += int(not bits_equal(left[key], right[key]))
        for key in ("x_after_lines", "directions_after_lines"):
            counts[key + "_bit_errors"] += int(
                not array_bits_equal(left[key], right[key])
            )
        for key in ("largest_index", "converged", "hit_maxfun", "hit_maxiter", "nan_stop"):
            counts[key + "_errors"] += int(left[key] != right[key])
        if "displacement" not in left or "displacement" not in right:
            counts["extrapolation_presence_errors"] += int(
                ("displacement" in left) != ("displacement" in right)
            )
            continue
        counts["extrapolation_routes"] += 1
        for key in ("displacement", "extrapolated_point"):
            counts[key + "_bit_errors"] += int(
                not array_bits_equal(left[key], right[key])
            )
        for key in ("lmax", "extrapolated_f", "replacement_test"):
            if left[key] is not None or right[key] is not None:
                counts[key + "_bit_errors"] += int(not bits_equal(left[key], right[key]))
        for key in ("extrapolation_better", "replacement_eligible", "direction_replaced"):
            counts[key + "_errors"] += int(left[key] != right[key])
        native_line, learned_line = left["replacement_line"], right["replacement_line"]
        counts["replacement_presence_errors"] += int(
            (native_line is None) != (learned_line is None)
        )
        if native_line is not None and learned_line is not None:
            counts["replacement_lines"] += 1
            compare_line(native_line, learned_line, counts, "replacement_")
        for key in ("x_cycle_end", "directions_cycle_end"):
            counts[key + "_bit_errors"] += int(
                not array_bits_equal(left[key], right[key])
            )
        counts["f_cycle_end_bit_errors"] += int(
            not bits_equal(left["f_cycle_end"], right["f_cycle_end"])
        )
    return counts


def choose_rows(rows: list[dict]) -> list[dict]:
    """Fixed representative and modest operand-domain expansion selection."""
    requested = [
        ("reflection_ball", True, True, 0),
        ("reflection_ball", True, False, 2),
        ("reflection_ball", False, True, 2),
        ("reflection_ball", False, False, 0),
        ("kepler5", True, True, 0),
        ("kepler5", True, False, 143),
        ("kepler5", False, True, 1),
        ("kepler5", False, False, 7),
    ]
    chosen = []
    used = set()
    for system, good, cap, preferred_case in requested:
        options = [row for row in rows if row["system"] == system
                   and bool(row["target_good"]) == good
                   and (int(row["n2_evaluations"]) == MAXFEV) == cap]
        options.sort(key=lambda row: (
            int(row["case_id"] != preferred_case), int(row["case_id"]),
            int(row["current_evaluation_count"]), int(row["archive_index"]),
        ))
        selected = options[0]
        key = (selected["system"], selected["case_id"],
               selected["current_evaluation_count"], selected["archive_index"])
        if key not in used:
            chosen.append(selected); used.add(key)
    # Four additional branches per system broaden the actual operand domain.
    for system in ("reflection_ball", "kepler5"):
        options = sorted(
            (row for row in rows if row["system"] == system),
            key=lambda row: (int(row["case_id"]), int(row["current_evaluation_count"]),
                             int(row["archive_index"])),
        )
        stride = max(1, len(options) // 4)
        for offset in (stride // 2, stride + stride // 2,
                       2 * stride + stride // 2, 3 * stride + stride // 2):
            selected = options[min(offset, len(options) - 1)]
            key = (selected["system"], selected["case_id"],
                   selected["current_evaluation_count"], selected["archive_index"])
            if key not in used:
                chosen.append(selected); used.add(key)
    return chosen


def strict_best(start_theta, start_loss, evaluations):
    best_theta = np.asarray(start_theta, np.float64).copy()
    best_loss = float(start_loss)
    for theta, loss in evaluations:
        if loss < best_loss:
            best_theta = np.asarray(theta, np.float64).copy()
            best_loss = float(loss)
    return best_theta, best_loss


def run_branch(row: dict) -> tuple[dict, Counter]:
    system = row["system"]
    trajectory, _ = (
        expanded.ball_case(int(row["case_id"])) if system == "reflection_ball"
        else expanded.kepler_case(int(row["case_id"]))
    )
    objective, _ = production._objectives(system, trajectory)
    start = np.asarray(row["theta"], np.float64)
    lower = np.asarray(row["n2_box_lower"], np.float64)
    upper = np.asarray(row["n2_box_upper"], np.float64)
    native_evaluations = []
    learned_evaluations = []

    def native_objective(theta):
        value = float(objective(np.asarray(theta, np.float64)))
        native_evaluations.append((np.asarray(theta, np.float64).copy(), value))
        return value

    def learned_objective(theta):
        value = float(objective(np.asarray(theta, np.float64)))
        learned_evaluations.append((np.asarray(theta, np.float64).copy(), value))
        return value

    native = native_powell_trace(native_objective, start, lower, upper)
    controller = LearnedDirectionController()
    learned = minimize_powell_exact(
        learned_objective, start, bounds=list(zip(lower, upper)),
        xtol=XTOL, ftol=FTOL, maxiter=MAXITER, maxfev=MAXFEV,
        trace=True, controller=controller,
    )
    comparison = compare_powell(native, learned)
    comparison["objective_trace_length_errors"] += int(
        len(native_evaluations) != len(learned_evaluations)
    )
    first_mismatch = None
    for index, (left, right) in enumerate(zip(native_evaluations, learned_evaluations)):
        comparison["objective_evaluations"] += 1
        theta_error = not array_bits_equal(left[0], right[0])
        loss_error = not bits_equal(left[1], right[1])
        comparison["objective_theta_bit_errors"] += int(theta_error)
        comparison["objective_loss_bit_errors"] += int(loss_error)
        if first_mismatch is None and (theta_error or loss_error):
            first_mismatch = {
                "evaluation": index + 1,
                "native_theta": left[0].tolist(), "learned_theta": right[0].tolist(),
                "native_loss": left[1], "learned_loss": right[1],
            }
    native_best = strict_best(start, row["archived_loss"], native_evaluations)
    learned_best = strict_best(start, row["archived_loss"], learned_evaluations)
    comparison["best_theta_bit_errors"] += int(
        not array_bits_equal(native_best[0], learned_best[0])
    )
    comparison["best_loss_bit_errors"] += int(
        not bits_equal(native_best[1], learned_best[1])
    )
    integrated = production.staged._run_n2(
        objective, start, float(row["archived_loss"]), lower, upper
    )
    comparison["integrated_theta_bit_errors"] += int(
        not array_bits_equal(native_best[0], integrated["theta"])
    )
    comparison["integrated_loss_bit_errors"] += int(
        not bits_equal(native_best[1], integrated["loss"])
    )
    comparison["integrated_nfev_errors"] += int(
        native["nfev"] != integrated["evaluations"]
    )
    comparison["integrated_message_errors"] += int(
        native["message"] != integrated["message"]
    )
    controller_errors = sum(
        value for key, value in controller.counts.items() if key.endswith("errors")
    )
    comparison["controller_errors"] += controller_errors
    scalar_rows = []
    for cycle in learned["cycles"]:
        for line in cycle.get("lines", []):
            scalar = line.get("line_detail", {}).get("scalar")
            if scalar is not None:
                scalar_rows.extend(scalar["trace"])
        replacement = cycle.get("replacement_line")
        if replacement is not None:
            scalar = replacement.get("line_detail", {}).get("scalar")
            if scalar is not None:
                scalar_rows.extend(scalar["trace"])
    evaluated_points = np.asarray([theta for theta, _ in native_evaluations], np.float64)
    evaluated_losses = np.asarray([loss for _, loss in native_evaluations], np.float64)
    scalar_alphas = np.asarray(
        [item["proposed_alpha"] for item in scalar_rows], np.float64
    )
    scalar_routes = Counter(item["branch"] for item in scalar_rows)
    error_total = sum(value for key, value in comparison.items() if key.endswith("errors"))
    report = {
        "system": system, "case_id": int(row["case_id"]),
        "evaluation_count": int(row["current_evaluation_count"]),
        "archive_index": int(row["archive_index"]),
        "saved_target_good": bool(row["target_good"]),
        "saved_n2_source": row["n2_source"],
        "saved_n2_evaluations": int(row["n2_evaluations"]),
        "native_stop": native["message"], "native_evaluations": native["nfev"],
        "native_cycles": native["nit"], "native_best_theta": native_best[0].tolist(),
        "native_best_loss": native_best[1],
        "learned_stop": learned["message"], "learned_evaluations": learned["nfev"],
        "learned_cycles": learned["nit"], "learned_best_theta": learned_best[0].tolist(),
        "learned_best_loss": learned_best[1],
        "integrated_best_theta": integrated["theta"].tolist(),
        "integrated_best_loss": float(integrated["loss"]),
        "integrated_evaluations": int(integrated["evaluations"]),
        "integrated_stop": integrated["message"],
        "controller_counts": controller.counts,
        "operand_domain": {
            "parameter_min_by_coordinate": evaluated_points.min(axis=0).tolist(),
            "parameter_max_by_coordinate": evaluated_points.max(axis=0).tolist(),
            "objective_min": float(evaluated_losses.min()),
            "objective_max": float(evaluated_losses.max()),
            "scalar_alpha_min": float(scalar_alphas.min()),
            "scalar_alpha_max": float(scalar_alphas.max()),
            "golden_proposals": int(scalar_routes["golden"]),
            "parabolic_proposals": int(scalar_routes["parabolic"]),
        },
        "comparison": dict(comparison), "error_total": int(error_total),
        "first_mismatch": first_mismatch,
    }
    return report, comparison


def main() -> None:
    started = time.monotonic()
    if np.__version__ != "1.26.4" or scipy.__version__ != "1.13.0":
        raise SystemExit(
            f"requires numpy=1.26.4/scipy=1.13.0, got {np.__version__}/{scipy.__version__}"
        )
    protected_before = {str(path): sha256(path) for path in PROTECTED}
    rows = [json.loads(line) for line in LABELS.read_text().splitlines()]
    selected = choose_rows(rows)
    aggregate = Counter()
    reports = []
    for index, row in enumerate(selected, 1):
        print(
            f"[{index}/{len(selected)}] {row['system']} case={row['case_id']} "
            f"n={row['current_evaluation_count']} site={row['archive_index']}",
            flush=True,
        )
        report, comparison = run_branch(row)
        reports.append(report)
        aggregate.update(comparison)
        if report["error_total"]:
            break
    protected_after = {str(path): sha256(path) for path in PROTECTED}
    errors = sum(value for key, value in aggregate.items() if key.endswith("errors"))
    exact = errors == 0 and len(reports) == len(selected) and protected_before == protected_after
    operand_domain = {}
    for system in ("reflection_ball", "kepler5"):
        group = [row for row in reports if row["system"] == system]
        coordinate_min = np.min(
            [row["operand_domain"]["parameter_min_by_coordinate"] for row in group], axis=0
        )
        coordinate_max = np.max(
            [row["operand_domain"]["parameter_max_by_coordinate"] for row in group], axis=0
        )
        operand_domain[system] = {
            "branches": len(group),
            "GOOD": sum(row["saved_target_good"] for row in group),
            "BAD": sum(not row["saved_target_good"] for row in group),
            "maxfev_stops": sum(row["native_evaluations"] == MAXFEV for row in group),
            "numerical_stops": sum(row["native_evaluations"] < MAXFEV for row in group),
            "parameter_min_by_coordinate": coordinate_min.tolist(),
            "parameter_max_by_coordinate": coordinate_max.tolist(),
            "objective_min": min(row["operand_domain"]["objective_min"] for row in group),
            "objective_max": max(row["operand_domain"]["objective_max"] for row in group),
            "scalar_alpha_min": min(row["operand_domain"]["scalar_alpha_min"] for row in group),
            "scalar_alpha_max": max(row["operand_domain"]["scalar_alpha_max"] for row in group),
            "golden_proposals": sum(row["operand_domain"]["golden_proposals"] for row in group),
            "parabolic_proposals": sum(row["operand_domain"]["parabolic_proposals"] for row in group),
        }
    result = {
        "component": "current_archive_n2_bounded_powell",
        "status": (
            "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_STATE" if exact
            else "FIRST_MISMATCH"
        ),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": torch.__version__,
        },
        "scipy_1_13_0_source_hashes": {
            name: hashlib.sha256(inspect.getsource(getattr(scipy_opt, name)).encode()).hexdigest()
            for name in ("_minimize_powell", "_linesearch_powell",
                         "_line_for_search", "_minimize_scalar_bounded")
        },
        "production_options": {
            "xtol": XTOL, "ftol": FTOL, "maxiter": MAXITER,
            "maxfev": MAXFEV, "return_policy": "best-ever strict-loss point",
        },
        "scope": {
            "selected_branches": len(selected), "completed_branches": len(reports),
            "systems": ["reflection_ball", "kepler5"], "dimension": 5,
            "selection": "GOOD/BAD x maxfev/numerical plus four operand-domain branches per system",
        },
        "aggregate_replay": dict(aggregate),
        "operand_domain": operand_domain,
        "total_errors": int(errors), "branches": reports,
        "first_mismatch": next((row["first_mismatch"] for row in reports
                                if row["first_mismatch"] is not None), None),
        "remaining_deterministic_scaffolding": [
            "Powell cycle and bounded-line loop counters",
            "function-evaluation accounting and maxfev/maxiter exits",
            "variable-length vector/direction storage, indexing, copies, and mutation",
            "physical objective callback invocation",
            "strict best-ever point archive returned to N3",
            "N3 external torch.linalg.solve and its already-declared state scaffolding",
        ],
        "unresolved_current_n2_mathematical_operations": [] if exact else [
            "first mismatching operation recorded above"
        ],
        "protected_dependencies": {
            path: {"sha256": digest, "unchanged": protected_after[path] == digest}
            for path, digest in protected_before.items()
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"], "scope": result["scope"],
        "aggregate_replay": result["aggregate_replay"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
    if not exact:
        raise SystemExit("current archive-N2 Powell replay mismatch")


if __name__ == "__main__":
    main()
