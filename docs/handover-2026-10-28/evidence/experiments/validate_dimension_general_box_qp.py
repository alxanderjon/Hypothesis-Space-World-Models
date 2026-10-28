#!/usr/bin/env python3
"""Validate a dimension-general box-QP active-set Newton direction.

Production is untouched.  The d=5 face-enumeration implementation remains the
oracle; this file replaces only that exponential subproblem solver in a
diagnostic N3 replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments.n3_box_qp import (
    dimension_general_box_qp_direction as integrated_box_qp_direction,
)
from experiments import test_n3_active_set_repair as active
from experiments import test_n3_box_constrained_newton as oracle
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
SOURCE = ROOT / "handoff_promotion_unresolved.jsonl"
ACTIVE_REPORT = ROOT / "active_set_n3_diagnostic" / "report.json"
CAP_REPORT = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = ROOT / "dimension_general_box_qp_validation"
CONTROL_KEYS = {
    ("kepler5", "12"), ("kepler5", "27"), ("kepler5", "45"),
    ("full_k_sho", "companion_14"), ("full_k_sho", "interior_49"),
    ("full_k_ball", "17"), ("reflection_ball", "0"),
}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    temporary.replace(path)


def _face(status: torch.Tensor) -> list[int]:
    return [int(value) for value in status.tolist()]


def dimension_general_box_qp_direction(theta, gradient, hessian, lower, upper,
                                       direction_fn):
    """Primal active-set solution of a convex box quadratic.

    Status is -1 at a lower bound, 0 when free, and +1 at an upper bound.
    All feasibility, blocking, and multiplier decisions use exact float64
    comparisons.  There is no distance-to-bound or KKT tolerance.
    """
    unconstrained, condition, damping, route_status = direction_fn(
        gradient, hessian)
    info = {
        "condition": float(condition), "damping": float(damping),
        "route_status": route_status,
        "used_unconstrained_feasible_shortcut": False,
        "active_set_iterations": 0, "face_solves": 0,
        "blocking_events": 0, "release_events": 0,
        "winning_face_status": None, "quadratic_value": None,
        "face_history": [],
    }
    if unconstrained is None:
        return None, info

    dimension = int(gradient.numel())
    step_lower = lower - theta
    step_upper = upper - theta
    identity = torch.eye(dimension, dtype=gradient.dtype,
                         device=gradient.device)
    matrix = hessian + damping * identity

    def quadratic_value(direction):
        return torch.dot(gradient, direction) + 0.5 * torch.dot(
            direction, matrix @ direction)

    if bool(torch.all(unconstrained >= step_lower) and
            torch.all(unconstrained <= step_upper)):
        info["used_unconstrained_feasible_shortcut"] = True
        info["winning_face_status"] = [0] * dimension
        info["quadratic_value"] = float(quadratic_value(unconstrained))
        return unconstrained, info

    # d=0 is always feasible.  Only coordinates exactly on a parameter bound
    # and satisfying the exact one-sided KKT sign begin fixed.
    direction = torch.zeros_like(gradient)
    status = torch.zeros(dimension, dtype=torch.int8, device=gradient.device)
    status[(step_lower == 0.0) & (gradient >= 0.0)] = -1
    status[(step_upper == 0.0) & (gradient <= 0.0)] = 1

    # A convex active-set QP has finitely many faces.  We do not impose an
    # iteration threshold; repeated full (face, direction) state is a
    # structural degeneracy/failure, rather than approximate convergence.
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    while True:
        info["active_set_iterations"] += 1
        key = (tuple(_face(status)), direction.detach().cpu().numpy().tobytes())
        if key in seen:
            info["route_status"] = "active_set_cycle"
            return None, info
        seen.add(key)
        info["face_history"].append(_face(status))

        lower_indices = torch.nonzero(status == -1).flatten()
        upper_indices = torch.nonzero(status == 1).flatten()
        free = torch.nonzero(status == 0).flatten()
        trial = torch.zeros_like(direction)
        if lower_indices.numel():
            trial[lower_indices] = step_lower[lower_indices]
        if upper_indices.numel():
            trial[upper_indices] = step_upper[upper_indices]
        if free.numel():
            # Match the oracle's lower-then-upper fixed-coordinate product
            # order so that the final reduced solve has identical float64
            # arithmetic when the active face agrees.
            fixed = torch.cat((lower_indices, upper_indices))
            rhs = -gradient[free]
            if fixed.numel():
                rhs = rhs - matrix[free][:, fixed] @ trial[fixed]
            try:
                trial[free] = torch.linalg.solve(matrix[free][:, free], rhs)
            except torch.linalg.LinAlgError:
                info["route_status"] = "singular_reduced_system"
                return None, info
        info["face_solves"] += 1
        if not bool(torch.isfinite(trial).all()):
            info["route_status"] = "nonfinite_direction"
            return None, info

        # If the face minimizer is outside the box, move from the current
        # feasible point to the first exactly encountered blocking bound.
        delta = trial - direction
        blockers: list[tuple[float, int, int]] = []
        for index in free.tolist():
            if trial[index] < step_lower[index]:
                alpha = float((step_lower[index] - direction[index]) /
                              delta[index])
                blockers.append((alpha, index, -1))
            elif trial[index] > step_upper[index]:
                alpha = float((step_upper[index] - direction[index]) /
                              delta[index])
                blockers.append((alpha, index, 1))
        if blockers:
            # Exact minimum and canonical coordinate order.  Simultaneous
            # blockers with the exact same alpha enter together.
            alpha = min(item[0] for item in blockers)
            direction = direction + alpha * delta
            simultaneous = sorted((index, bound_status)
                                  for value, index, bound_status in blockers
                                  if value == alpha)
            for index, bound_status in simultaneous:
                status[index] = bound_status
                direction[index] = (step_lower[index] if bound_status == -1
                                    else step_upper[index])
            info["blocking_events"] += len(simultaneous)
            continue

        direction = trial
        reduced_gradient = gradient + matrix @ direction
        violations: list[tuple[float, int]] = []
        for index in lower_indices.tolist():
            if reduced_gradient[index] < 0.0:
                violations.append((-float(reduced_gradient[index]), index))
        for index in upper_indices.tolist():
            if reduced_gradient[index] > 0.0:
                violations.append((float(reduced_gradient[index]), index))
        if violations:
            # Release the largest exact KKT violation; coordinate index is the
            # deterministic tie-break.  The next reduced solve can move it in
            # either feasible direction.
            _, index = min(violations, key=lambda item: (-item[0], item[1]))
            status[index] = 0
            info["release_events"] += 1
            continue

        info["winning_face_status"] = _face(status)
        info["quadratic_value"] = float(quadratic_value(direction))
        if (not bool(torch.isfinite(direction).all()) or
                float(torch.dot(gradient, direction)) >= 0.0):
            info["route_status"] = "no_descent_direction"
            return None, info
        return direction, info


def compare_direction(theta, gradient, hessian, lower, upper, direction_fn):
    enumerated, enumerated_info = oracle.box_qp_direction(
        theta, gradient, hessian, lower, upper, direction_fn)
    active_direction, active_info = integrated_box_qp_direction(
        theta, gradient, hessian, lower, upper, direction_fn)
    both_none = enumerated is None and active_direction is None
    same_direction = both_none or (
        enumerated is not None and active_direction is not None and
        torch.equal(enumerated, active_direction))
    direction_max_abs = None
    if enumerated is not None and active_direction is not None:
        direction_max_abs = float(torch.max(torch.abs(
            enumerated - active_direction)))
    return {
        "direction_exact": bool(same_direction),
        "direction_max_abs": direction_max_abs,
        "face_exact": (enumerated_info["winning_face_status"] ==
                       active_info["winning_face_status"]),
        "quadratic_exact": (enumerated_info["quadratic_value"] ==
                            active_info["quadratic_value"]),
        "oracle": enumerated_info, "active_set": active_info,
    }


def aliases():
    wanted = {row["example_id"] for row in
              json.loads(ACTIVE_REPORT.read_text())["aliases"]}
    return [row for row in map(json.loads, SOURCE.read_text().splitlines())
            if row["example_id"] in wanted]


def run_aliases(selected_index=None):
    rows = aliases()
    if selected_index is not None:
        rows = [rows[selected_index]]
    results = []
    for number, row in enumerate(rows, 1):
        ctx = active.alias_context(row)
        start = np.asarray(row["label_n2_endpoint"], np.float64)
        oracle_result = oracle.run_box(start, *ctx)
        active_result = oracle.run_box(
            start, *ctx, qp_solver=integrated_box_qp_direction)

        torch_objective, derivatives, direction_fn, lower_np, upper_np, _, _ = ctx
        lower = torch.from_numpy(lower_np.copy())
        upper = torch.from_numpy(upper_np.copy())
        state_comparisons = []
        for iteration in oracle_result["iterations"]:
            theta = torch.tensor(iteration["theta"], dtype=torch.float64)
            _, gradient, hessian = derivatives(theta, torch.empty(0))
            state_comparisons.append(compare_direction(
                theta, gradient, hessian, lower, upper, direction_fn))

        trajectory_comparison = {
            "theta_history_exact": bool(np.array_equal(
                np.asarray(oracle_result["theta_history"]),
                np.asarray(active_result["theta_history"]))),
            "loss_history_exact": (oracle_result["loss_history"] ==
                                   active_result["loss_history"]),
            "accepted_steps_exact": (oracle_result["accepted_steps"] ==
                                     active_result["accepted_steps"]),
            "stop_reason_exact": (oracle_result["stop_reason"] ==
                                  active_result["stop_reason"]),
            "final_theta_exact": (oracle_result["final_theta"] ==
                                  active_result["final_theta"]),
            "final_loss_exact": (oracle_result["final_loss"] ==
                                 active_result["final_loss"]),
        }
        record = {
            "case_id": row["case_id"], "state_id": row["state_id"],
            "example_id": row["example_id"],
            "oracle": oracle_result, "active_set": active_result,
            "intermediate_state_comparisons": state_comparisons,
            "trajectory_comparison": trajectory_comparison,
        }
        results.append(record)
        atomic_json(OUT / "aliases" / f'{row["example_id"]}.json', record)
        print("alias", number, len(rows), row["case_id"],
              all(trajectory_comparison.values()),
              sum(not comparison["direction_exact"]
                  for comparison in state_comparisons), flush=True)
    return results


def run_controls():
    cap_report = json.loads(CAP_REPORT.read_text())
    rows = [row for row in cap_report["cases"]
            if (row["system"], str(row["case_id"])) in CONTROL_KEYS]
    output = []
    for number, row in enumerate(rows, 1):
        ctx = active.control_context(row)
        start = np.asarray(row["cap_suppressed"]["theta_history"][0], np.float64)
        result = oracle.run_box(
            start, *ctx, qp_solver=integrated_box_qp_direction)
        production = row["cap_suppressed"]
        comparison = {
            "theta_history_exact": bool(np.array_equal(
                np.asarray(result["theta_history"]),
                np.asarray(production["theta_history"]))),
            "loss_history_exact": result["loss_history"] == production["loss_history"],
            "accepted_steps_exact": result["accepted_steps"] == production["accepted_steps"],
            "stop_reason_exact": result["stop_reason"] == production["stop_reason"],
            "final_theta_exact": result["final_theta"] == production["final_theta"],
            "final_loss_exact": result["final_loss"] == production["final_loss"],
        }
        record = {"system": row["system"], "case_id": row["case_id"],
                  "comparison": comparison, "result": result}
        output.append(record)
        atomic_json(OUT / "controls" /
                    f'{row["system"]}_{row["case_id"]}.json', record)
        print("control", number, len(rows), row["system"], row["case_id"],
              all(comparison.values()), flush=True)
    return output


def synthetic_direction_fn(gradient, hessian):
    direction = torch.linalg.solve(hessian, -gradient)
    singular_values = torch.linalg.svdvals(hessian)
    condition = float(torch.max(singular_values) / torch.min(singular_values))
    return direction, condition, 0.0, "ok"


def synthetic_checks():
    output = []
    generator = np.random.default_rng(74291)
    for dimension in (6, 8, 12, 20, 32):
        for control in range(8):
            raw = generator.normal(size=(dimension, dimension))
            matrix_np = raw.T @ raw + np.eye(dimension) * 0.25
            gradient_np = generator.normal(size=dimension)
            lower_np = generator.uniform(-2.0, -0.1, size=dimension)
            upper_np = generator.uniform(0.1, 2.0, size=dimension)
            theta_np = generator.uniform(lower_np, upper_np)
            # Exact-bound and coupled-release mechanics occur in deterministic
            # controls without changing the convex objective.
            if control % 2 == 0:
                theta_np[0] = lower_np[0]
            if control % 3 == 0:
                theta_np[-1] = upper_np[-1]
            theta = torch.from_numpy(theta_np.copy())
            gradient = torch.from_numpy(gradient_np.copy())
            hessian = torch.from_numpy(matrix_np.copy())
            lower = torch.from_numpy(lower_np.copy())
            upper = torch.from_numpy(upper_np.copy())
            direction, info = integrated_box_qp_direction(
                theta, gradient, hessian, lower, upper,
                synthetic_direction_fn)
            assert direction is not None
            subgradient = gradient + hessian @ direction
            at_lower = direction == lower - theta
            at_upper = direction == upper - theta
            free = ~(at_lower | at_upper)
            exact_kkt = bool(
                torch.all(direction >= lower - theta) and
                torch.all(direction <= upper - theta) and
                torch.all(subgradient[at_lower] >= 0.0) and
                torch.all(subgradient[at_upper] <= 0.0) and
                torch.all(subgradient[free] == 0.0))
            # torch.linalg.solve leaves roundoff residuals, so exact free KKT
            # is recorded separately from the standard scaled residual.
            scale = max(1.0, float(torch.linalg.vector_norm(gradient)))
            free_residual = (0.0 if not bool(torch.any(free)) else
                             float(torch.max(torch.abs(subgradient[free]))))
            output.append({
                "dimension": dimension, "control": control,
                "feasible_exact": bool(torch.all(direction >= lower - theta) and
                                       torch.all(direction <= upper - theta)),
                "active_signs_exact": bool(torch.all(subgradient[at_lower] >= 0.0) and
                                           torch.all(subgradient[at_upper] <= 0.0)),
                "free_residual": free_residual,
                "scaled_free_residual": free_residual / scale,
                "exact_kkt_including_free_zero": exact_kkt,
                "active_set_iterations": info["active_set_iterations"],
                "blocking_events": info["blocking_events"],
                "release_events": info["release_events"],
                "winning_face_status": info["winning_face_status"],
            })
    return output


def targeted_synthetic_checks():
    """Exact structural controls for simultaneous blocks and active release."""
    output = []

    def run(name, hessian_np, gradient_np, theta_np, lower_np, upper_np,
            expected_np, expected_min_blocking, expected_min_releases):
        tensors = [torch.from_numpy(np.asarray(value, np.float64).copy())
                   for value in (theta_np, gradient_np, hessian_np,
                                 lower_np, upper_np)]
        theta, gradient, hessian, lower, upper = tensors
        direction, info = integrated_box_qp_direction(
            theta, gradient, hessian, lower, upper, synthetic_direction_fn)
        expected = torch.from_numpy(np.asarray(expected_np, np.float64).copy())
        residual = gradient + hessian @ direction
        at_lower = direction == lower - theta
        at_upper = direction == upper - theta
        free = ~(at_lower | at_upper)
        output.append({
            "name": name, "dimension": len(gradient_np),
            "direction": direction.tolist(), "expected": expected.tolist(),
            "direction_max_abs": float(torch.max(torch.abs(direction - expected))),
            "feasible_exact": bool(torch.all(direction >= lower - theta) and
                                   torch.all(direction <= upper - theta)),
            "active_signs_exact": bool(torch.all(residual[at_lower] >= 0.0) and
                                       torch.all(residual[at_upper] <= 0.0)),
            "free_residual_max": (0.0 if not bool(torch.any(free)) else
                                  float(torch.max(torch.abs(residual[free])))),
            "blocking_events": info["blocking_events"],
            "release_events": info["release_events"],
            "blocking_requirement_met": (info["blocking_events"] >=
                                         expected_min_blocking),
            "release_requirement_met": (info["release_events"] >=
                                        expected_min_releases),
            "face_history": info["face_history"],
            "winning_face_status": info["winning_face_status"],
        })

    # Two coordinates hit opposing bounds at the identical alpha=0.5.
    dimension = 6
    run(
        "simultaneous_lower_upper",
        np.eye(dimension), np.array([-2., 2., 0., 0., 0., 0.]),
        np.zeros(dimension), -np.ones(dimension), np.ones(dimension),
        np.array([1., -1., 0., 0., 0., 0.]), 2, 0)

    # Distinct blocking fractions force sequential face changes.
    run(
        "sequential_upper_bounds",
        np.eye(dimension), np.array([-4., -2., 0., 0., 0., 0.]),
        np.zeros(dimension), -np.ones(dimension), np.ones(dimension),
        np.array([1., 1., 0., 0., 0., 0.]), 2, 0)

    # Coordinate zero begins exactly at its lower bound with g0>0.  Coupling to
    # coordinate one makes its multiplier negative after coordinate two blocks,
    # so the correct optimum releases it.
    hessian = np.eye(dimension)
    hessian[:2, :2] = np.array([[2., -1.], [-1., 2.]])
    gradient = np.array([.1, -1., -2., 0., 0., 0.])
    theta = np.zeros(dimension)
    lower = -np.ones(dimension)
    lower[0] = 0.0
    upper = np.ones(dimension)
    expected = np.array([0.8 / 3.0, 1.9 / 3.0, 1., 0., 0., 0.])
    run("release_initial_lower_bound", hessian, gradient, theta, lower,
        upper, expected, 1, 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias-index", type=int)
    parser.add_argument("--aliases-only", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    alias_rows = ([json.loads(path.read_text())
                   for path in sorted((OUT / "aliases").glob("*.json"))]
                  if args.compile_only else run_aliases(args.alias_index))
    if args.alias_index is not None:
        return
    controls = [] if args.aliases_only else run_controls()
    synthetic = [] if args.aliases_only else synthetic_checks()
    targeted_synthetic = ([] if args.aliases_only else
                          targeted_synthetic_checks())
    all_states = [comparison for row in alias_rows
                  for comparison in row["intermediate_state_comparisons"]]
    report = {
        "algorithm": "dimension-general primal active-set box-QP",
        "new_parameters": 0,
        "aliases": alias_rows, "controls": controls,
        "synthetic": synthetic, "targeted_synthetic": targeted_synthetic,
        "summary": {
            "alias_trajectories": len(alias_rows),
            "alias_trajectories_exact": sum(all(row["trajectory_comparison"].values())
                                            for row in alias_rows),
            "intermediate_states": len(all_states),
            "directions_exact": sum(row["direction_exact"] for row in all_states),
            "faces_exact": sum(row["face_exact"] for row in all_states),
            "quadratic_values_exact": sum(row["quadratic_exact"] for row in all_states),
            "controls": len(controls),
            "controls_exact": sum(all(row["comparison"].values()) for row in controls),
            "synthetic_controls": len(synthetic),
            "synthetic_feasible": sum(row["feasible_exact"] for row in synthetic),
            "synthetic_active_signs": sum(row["active_signs_exact"] for row in synthetic),
            "synthetic_max_scaled_free_residual": max(
                (row["scaled_free_residual"] for row in synthetic), default=0.0),
            "synthetic_release_controls": sum(row["release_events"] > 0
                                              for row in synthetic),
            "targeted_synthetic_controls": len(targeted_synthetic),
            "targeted_synthetic_requirements_met": sum(
                row["feasible_exact"] and row["active_signs_exact"] and
                row["blocking_requirement_met"] and
                row["release_requirement_met"]
                for row in targeted_synthetic),
        },
        "checkpoint_hashes_unchanged": before == hashes(),
    }
    atomic_json(OUT / "report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
