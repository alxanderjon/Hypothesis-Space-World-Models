#!/usr/bin/env python3
"""Exact d<=5 box-QP Newton diagnostic for the eight remaining aliases."""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_n3_accepted_step_cap as cap
from experiments import test_n3_active_set_repair as active
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
SOURCE = ROOT / "handoff_promotion_unresolved.jsonl"
ACTIVE_REPORT = ROOT / "active_set_n3_diagnostic" / "report.json"
CAP_REPORT = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = ROOT / "box_constrained_n3_diagnostic"
CONTROL_KEYS = {
    ("kepler5", "12"), ("kepler5", "27"), ("kepler5", "45"),
    ("full_k_sho", "companion_14"), ("full_k_sho", "interior_49"),
    ("full_k_ball", "17"), ("reflection_ball", "0"),
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Preserve the existing diagnostic representation for non-finite derivative
    # states in controls; these are data, not values used by the QP router.
    tmp.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
    tmp.replace(path)


def old_active_direction(theta, gradient, hessian, lower, upper, direction_fn):
    active_lower = (theta == lower) & (gradient >= 0.0)
    active_upper = (theta == upper) & (gradient <= 0.0)
    fixed = active_lower | active_upper
    free = ~fixed
    direction = torch.zeros_like(gradient)
    if not bool(torch.any(free)):
        return direction, active_lower, active_upper, "no_descent_direction"
    reduced, _, _, status = direction_fn(gradient[free], hessian[free][:, free])
    if reduced is None:
        return None, active_lower, active_upper, status
    direction[free] = reduced
    return direction, active_lower, active_upper, status


def box_qp_direction(theta, gradient, hessian, lower, upper, direction_fn):
    """Minimize the (possibly existing-damped) quadratic over exact box steps."""
    unconstrained, condition, damping, status = direction_fn(gradient, hessian)
    info = {
        "condition": float(condition), "damping": float(damping),
        "route_status": status, "used_unconstrained_feasible_shortcut": False,
        "enumerated_configurations": 0, "feasible_face_solutions": 0,
        "winning_face_status": None, "quadratic_value": None,
    }
    if unconstrained is None:
        return None, info
    step_lower = lower - theta
    step_upper = upper - theta
    if bool(torch.all(unconstrained >= step_lower) and
            torch.all(unconstrained <= step_upper)):
        info["used_unconstrained_feasible_shortcut"] = True
        matrix = hessian + damping * torch.eye(len(gradient), dtype=gradient.dtype)
        value = torch.dot(gradient, unconstrained) + 0.5 * torch.dot(
            unconstrained, matrix @ unconstrained)
        info["winning_face_status"] = [0] * len(gradient)
        info["quadratic_value"] = float(value)
        return unconstrained, info

    matrix = hessian + damping * torch.eye(len(gradient), dtype=gradient.dtype)
    best = None
    best_value = math.inf
    best_status = None
    dimension = len(gradient)
    for statuses in product((-1, 0, 1), repeat=dimension):
        info["enumerated_configurations"] += 1
        direction = torch.zeros_like(gradient)
        lower_indices = [i for i, value in enumerate(statuses) if value == -1]
        upper_indices = [i for i, value in enumerate(statuses) if value == 1]
        free_indices = [i for i, value in enumerate(statuses) if value == 0]
        if lower_indices:
            direction[lower_indices] = step_lower[lower_indices]
        if upper_indices:
            direction[upper_indices] = step_upper[upper_indices]
        if free_indices:
            free = torch.as_tensor(free_indices, dtype=torch.long)
            fixed_indices = lower_indices + upper_indices
            rhs = -gradient[free]
            if fixed_indices:
                fixed = torch.as_tensor(fixed_indices, dtype=torch.long)
                rhs = rhs - matrix[free][:, fixed] @ direction[fixed]
            try:
                direction[free] = torch.linalg.solve(matrix[free][:, free], rhs)
            except torch.linalg.LinAlgError:
                continue
        if not bool(torch.isfinite(direction).all()):
            continue
        if not bool(torch.all(direction >= step_lower) and
                    torch.all(direction <= step_upper)):
            continue
        info["feasible_face_solutions"] += 1
        value = torch.dot(gradient, direction) + 0.5 * torch.dot(
            direction, matrix @ direction)
        scalar = float(value)
        if scalar < best_value:
            best_value = scalar
            best = direction.clone()
            best_status = list(statuses)
    if best is None:
        info["route_status"] = "no_descent_direction"
        return None, info
    info["winning_face_status"] = best_status
    info["quadratic_value"] = best_value
    if not bool(torch.isfinite(best).all()) or float(torch.dot(gradient, best)) >= 0.0:
        info["route_status"] = "no_descent_direction"
        return None, info
    return best, info


def run_box(start_theta, torch_objective, derivatives, direction_fn,
            lower_np, upper_np, teacher, rollout_fn,
            qp_solver=box_qp_direction):
    lower = torch.from_numpy(lower_np.copy())
    upper = torch.from_numpy(upper_np.copy())
    theta = torch.from_numpy(np.asarray(start_theta, np.float64).copy())
    with torch.no_grad():
        current_loss = float(torch_objective(theta))
    theta_history = [theta.numpy().copy()]
    loss_history = [current_loss]
    iterations = []
    objective_evaluations = 0
    backtracking_rejections = 0
    accepted_with_backtracking = 0
    damping_steps = 0

    while True:
        _, gradient, hessian = derivatives(theta, torch.empty(0))
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        projected = active.projected_gradient(theta, gradient, lower, upper)
        old_direction, old_lower, old_upper, old_status = old_active_direction(
            theta, gradient, hessian, lower, upper, direction_fn)
        direction, qp = qp_solver(theta, gradient, hessian, lower, upper,
                                  direction_fn)
        if qp["damping"] > 0.0 and math.isfinite(qp["damping"]):
            damping_steps += 1
        old_fixed = old_lower | old_upper
        releases = [] if direction is None else torch.nonzero(
            old_fixed & (direction != 0.0)).flatten().tolist()
        record = {
            "accepted_step_before": len(loss_history) - 1,
            "theta": theta.tolist(), "loss": current_loss,
            "teacher_distance": float(np.linalg.norm(theta.numpy() - teacher)),
            "gradient": gradient.tolist(), "gradient_norm": gradient_norm,
            "projected_gradient": projected.tolist(),
            "projected_gradient_norm": float(torch.linalg.vector_norm(projected)),
            "old_active_lower": torch.nonzero(old_lower).flatten().tolist(),
            "old_active_upper": torch.nonzero(old_upper).flatten().tolist(),
            "old_active_direction": None if old_direction is None else old_direction.tolist(),
            "old_active_direction_status": old_status,
            "box_qp_direction": None if direction is None else direction.tolist(),
            "box_qp_direction_norm": None if direction is None else
                float(torch.linalg.vector_norm(direction)),
            "box_qp": qp, "released_old_active_coordinates": releases,
            "accepted": False, "accepted_backtrack": None,
            "accepted_alpha": None, "backtracking_attempts": 0,
            "actual_displacement": None, "actual_displacement_norm": None,
            "loss_decrease": None,
        }
        if not math.isfinite(gradient_norm):
            stop_reason = "nonfinite_gradient"
            iterations.append(record)
            break
        if gradient_norm == 0.0:
            stop_reason = "floating_point_zero_gradient"
            iterations.append(record)
            break
        if direction is None:
            stop_reason = qp["route_status"]
            iterations.append(record)
            break
        directional = float(torch.dot(gradient, direction))
        record["directional_derivative"] = directional
        accepted = False
        unchanged_candidate = False
        for backtrack in range(polish.MAX_BACKTRACKS + 1):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            record["backtracking_attempts"] += 1
            if bool(torch.equal(candidate, theta)):
                unchanged_candidate = True
                break
            if not bool(torch.all(candidate >= lower) and torch.all(candidate <= upper)):
                backtracking_rejections += 1
                continue
            with torch.no_grad():
                candidate_loss = float(torch_objective(candidate))
            objective_evaluations += 1
            if candidate_loss < current_loss:
                delta = candidate.numpy() - theta.numpy()
                record.update(
                    accepted=True, accepted_backtrack=backtrack,
                    accepted_alpha=alpha, actual_displacement=delta.tolist(),
                    actual_displacement_norm=float(np.linalg.norm(delta)),
                    loss_decrease=float(current_loss - candidate_loss))
                theta = candidate.detach()
                current_loss = candidate_loss
                theta_history.append(theta.numpy().copy())
                loss_history.append(current_loss)
                accepted_with_backtracking += int(backtrack > 0)
                accepted = True
                break
            backtracking_rejections += 1
        iterations.append(record)
        if not accepted:
            predicted = max(0.0, -directional)
            spacing = abs(float(np.spacing(np.float64(current_loss))))
            stop_reason = ("floating_point_no_further_improvement"
                           if unchanged_candidate or predicted <= 32.0 * spacing
                           else "backtracking_no_improvement")
            break

    final = theta.numpy().copy()
    rollout_final = rollout_fn(final)
    sample_indices = sorted(set((0, min(1, len(theta_history) - 1),
                                 (len(theta_history) - 1) // 2,
                                 len(theta_history) - 1)))
    rollout_samples = [{
        "accepted_step": index, "theta": theta_history[index].tolist(),
        "loss": loss_history[index],
        "teacher_distance": float(np.linalg.norm(theta_history[index] - teacher)),
        "rollout": rollout_fn(theta_history[index]),
    } for index in sample_indices]
    return {
        "accepted_steps": len(loss_history) - 1, "stop_reason": stop_reason,
        "objective_evaluations": objective_evaluations,
        "backtracking_rejections": backtracking_rejections,
        "accepted_with_backtracking": accepted_with_backtracking,
        "damping_steps": damping_steps,
        "theta_history": [x.tolist() for x in theta_history],
        "loss_history": loss_history, "iterations": iterations,
        "final_theta": final.tolist(), "final_loss": current_loss,
        "parameter_error": (final - teacher).tolist(),
        "parameter_l2_error": float(np.linalg.norm(final - teacher)),
        "parameter_max_error": float(np.max(np.abs(final - teacher))),
        "rollout": rollout_final, "rollout_samples": rollout_samples,
        "recovered": bool(current_loss < cap.staged.FINAL_GATE and
                          rollout_final.get("rollout_finite", True)),
    }


def target_examples():
    active_report = json.loads(ACTIVE_REPORT.read_text())
    lower = np.asarray((-.25, -1.0, -2.0, -.5, 1.5), np.float64)
    upper = np.asarray((.5, 2.0, 3.0, 2.5, 4.5), np.float64)
    ids = []
    for row in active_report["aliases"]:
        final = row["active_set"]["iterations"][-1]
        theta = np.asarray(final["theta"], np.float64)
        direction = np.asarray(final["full_direction"], np.float64)
        free = final["free"]
        if any(theta[index] + direction[index] < lower[index] or
               theta[index] + direction[index] > upper[index]
               for index in free):
            ids.append(row["example_id"])
    assert len(ids) == 8
    return set(ids)


def run_targets(index=None):
    ids = target_examples()
    source = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    rows = [row for row in source if row["example_id"] in ids]
    if index is not None:
        rows = [rows[index]]
    active_report = json.loads(ACTIVE_REPORT.read_text())
    previous = {row["example_id"]: row for row in active_report["aliases"]}
    output = []
    for number, row in enumerate(rows, 1):
        ctx = active.alias_context(row)
        result = run_box(np.asarray(row["label_n2_endpoint"], np.float64), *ctx)
        record = {
            "system": "kepler5", "case_id": int(row["case_id"]),
            "state_id": row["state_id"], "example_id": row["example_id"],
            "old_active_set": previous[row["example_id"]]["active_set"],
            "box_constrained": result,
            "released_any_old_active_coordinate": any(
                iteration["released_old_active_coordinates"]
                for iteration in result["iterations"]),
        }
        output.append(record)
        atomic_json(OUT / "targets" / f'{row["example_id"]}.json', record)
        print("target", number, len(rows), row["case_id"], result["accepted_steps"],
              result["stop_reason"], result["recovered"], flush=True)
    return output


def run_controls():
    report = json.loads(CAP_REPORT.read_text())
    rows = [row for row in report["cases"]
            if (row["system"], str(row["case_id"])) in CONTROL_KEYS]
    output = []
    for number, row in enumerate(rows, 1):
        ctx = active.control_context(row)
        start = np.asarray(row["cap_suppressed"]["theta_history"][0], np.float64)
        result = run_box(start, *ctx)
        production = row["cap_suppressed"]
        comparison = {
            "theta_history_exact": bool(np.array_equal(
                np.asarray(result["theta_history"]),
                np.asarray(production["theta_history"]))),
            "loss_history_exact": result["loss_history"] == production["loss_history"],
            "accepted_steps_exact": result["accepted_steps"] == production["accepted_steps"],
            "stop_reason_exact": result["stop_reason"] == production["stop_reason"],
            "final_theta_exact": bool(np.array_equal(
                np.asarray(result["final_theta"]),
                np.asarray(production["final_theta"]))),
            "final_loss_exact": result["final_loss"] == production["final_loss"],
        }
        record = {"system": row["system"], "case_id": row["case_id"],
                  "production": production, "box_constrained": result,
                  "comparison": comparison,
                  "trajectory_changed": not all(comparison.values())}
        output.append(record)
        atomic_json(OUT / "controls" / f'{row["system"]}_{row["case_id"]}.json', record)
        print("control", number, len(rows), row["system"], row["case_id"],
              record["trajectory_changed"], flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    if args.controls_only:
        targets = [json.loads(path.read_text())
                   for path in sorted((OUT / "targets").glob("*.json"))]
    else:
        targets = run_targets(args.target_index)
    if args.target_index is not None:
        return
    controls = [] if args.targets_only else run_controls()
    result = {
        "diagnostic": "exact face-enumerated box quadratic Newton, no tolerance or cap",
        "targets": targets, "controls": controls,
        "summary": {
            "targets": len(targets),
            "recoveries": sum(row["box_constrained"]["recovered"] for row in targets),
            "stop_reasons": dict(Counter(row["box_constrained"]["stop_reason"]
                                         for row in targets)),
            "old_active_releases": sum(row["released_any_old_active_coordinate"]
                                       for row in targets),
            "controls": len(controls),
            "changed_controls": sum(row["trajectory_changed"] for row in controls),
            "all_control_fields_exact": all(all(row["comparison"].values())
                                            for row in controls),
        },
        "checkpoint_hashes_unchanged": before == hashes(),
    }
    atomic_json(OUT / "report.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
