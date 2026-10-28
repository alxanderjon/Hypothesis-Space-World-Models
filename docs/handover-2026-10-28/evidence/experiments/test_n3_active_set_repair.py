#!/usr/bin/env python3
"""Targeted active-set Newton diagnostic; production N3 is untouched."""
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

from experiments import validate_d5_production_pipeline as d5
from experiments import validate_n3_accepted_step_cap as cap
from experiments import audit_n3_projected_kkt_geometry as kkt
from experiments.bouncing_ball import run_first_transfer as ball_teacher
from experiments.bouncing_ball import run_full_k_transfer as ball4
from experiments.bouncing_ball import run_reflection_peelback as ball5
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import generate_checkpointed_edirect_teacher_data as sho_data
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
UNRESOLVED = ROOT / "handoff_promotion_unresolved.jsonl"
CAP_REPORT = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = ROOT / "active_set_n3_diagnostic"
CONTROL_KEYS = {
    ("kepler5", "12"), ("kepler5", "27"), ("kepler5", "45"),
    ("kepler5", "3"), ("full_k_sho", "companion_14"),
    ("full_k_sho", "interior_49"), ("full_k_ball", "17"),
    ("full_k_ball", "46"), ("reflection_ball", "0"),
    ("reflection_ball", "5"),
}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def projected_gradient(theta, gradient, lower, upper):
    result = gradient.clone()
    result[(theta == lower) & (gradient >= 0.0)] = 0.0
    result[(theta == upper) & (gradient <= 0.0)] = 0.0
    return result


def active_run(start_theta, torch_objective, derivatives, reduced_direction_fn,
               lower_np, upper_np, teacher, rollout_fn):
    lower = torch.from_numpy(lower_np.copy())
    upper = torch.from_numpy(upper_np.copy())
    theta = torch.from_numpy(np.asarray(start_theta, np.float64).copy())
    with torch.no_grad():
        current_loss = float(torch_objective(theta))
    theta_history = [theta.numpy().copy()]
    loss_history = [current_loss]
    records = []
    objective_evaluations = 0
    backtracking_rejections = 0
    accepted_with_backtracking = 0
    damping_steps = 0
    stop_reason = None

    while True:
        _, gradient, hessian = derivatives(theta, torch.empty(0))
        gradient_norm = float(torch.linalg.vector_norm(gradient))
        projected = projected_gradient(theta, gradient, lower, upper)
        active_lower = (theta == lower) & (gradient >= 0.0)
        active_upper = (theta == upper) & (gradient <= 0.0)
        active = active_lower | active_upper
        free = ~active
        iteration = {
            "accepted_step_before": len(loss_history) - 1,
            "theta": theta.numpy().copy().tolist(), "loss": current_loss,
            "gradient": gradient.tolist(), "gradient_norm": gradient_norm,
            "projected_gradient": projected.tolist(),
            "projected_gradient_norm": float(torch.linalg.vector_norm(projected)),
            "active_lower": torch.nonzero(active_lower).flatten().tolist(),
            "active_upper": torch.nonzero(active_upper).flatten().tolist(),
            "free": torch.nonzero(free).flatten().tolist(),
            "full_hessian_eigenvalues": torch.linalg.eigvalsh(hessian).tolist(),
            "reduced_hessian_eigenvalues": None,
            "reduced_condition": None, "damping": None,
            "direction_status": None, "full_direction": None,
            "direction_norm": None, "directional_derivative": None,
            "accepted": False, "accepted_backtrack": None,
            "accepted_alpha": None, "backtracking_attempts": 0,
            "actual_displacement": None,
            "actual_displacement_local_ulps": None,
            "actual_displacement_norm": None, "loss_decrease": None,
            "teacher_distance": float(np.linalg.norm(theta.numpy() - teacher)),
        }
        if not math.isfinite(gradient_norm):
            stop_reason = "nonfinite_gradient"
            records.append(iteration)
            break
        if gradient_norm == 0.0:
            stop_reason = "floating_point_zero_gradient"
            records.append(iteration)
            break
        if not bool(torch.any(free)):
            iteration["direction_status"] = "no_descent_direction"
            stop_reason = "no_descent_direction"
            records.append(iteration)
            break

        reduced_gradient = gradient[free]
        reduced_hessian = hessian[free][:, free]
        iteration["reduced_hessian_eigenvalues"] = torch.linalg.eigvalsh(
            reduced_hessian).tolist()
        reduced_direction, condition, damping, status = reduced_direction_fn(
            reduced_gradient, reduced_hessian)
        iteration["reduced_condition"] = float(condition)
        iteration["damping"] = float(damping)
        iteration["direction_status"] = status
        if damping > 0.0 and math.isfinite(damping):
            damping_steps += 1
        if reduced_direction is None:
            stop_reason = status
            records.append(iteration)
            break
        direction = torch.zeros_like(gradient)
        direction[free] = reduced_direction
        iteration["full_direction"] = direction.tolist()
        iteration["direction_norm"] = float(torch.linalg.vector_norm(direction))
        directional = float(torch.dot(gradient, direction))
        iteration["directional_derivative"] = directional

        accepted = False
        unchanged_candidate = False
        for backtrack in range(polish.MAX_BACKTRACKS + 1):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            iteration["backtracking_attempts"] += 1
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
                old = theta.numpy().copy()
                new = candidate.numpy().copy()
                delta = new - old
                spacing = np.abs(np.spacing(old))
                ulps = np.divide(delta, spacing, out=np.zeros_like(delta),
                                 where=spacing != 0.0)
                iteration.update(
                    accepted=True, accepted_backtrack=backtrack,
                    accepted_alpha=alpha, actual_displacement=delta.tolist(),
                    actual_displacement_local_ulps=ulps.tolist(),
                    actual_displacement_norm=float(np.linalg.norm(delta)),
                    loss_decrease=float(current_loss - candidate_loss),
                )
                theta = candidate.detach()
                current_loss = candidate_loss
                theta_history.append(theta.numpy().copy())
                loss_history.append(current_loss)
                accepted_with_backtracking += int(backtrack > 0)
                accepted = True
                break
            backtracking_rejections += 1
        records.append(iteration)
        if not accepted:
            predicted_decrease = max(0.0, -directional)
            loss_spacing = abs(float(np.spacing(np.float64(current_loss))))
            stop_reason = ("floating_point_no_further_improvement"
                           if unchanged_candidate or
                           predicted_decrease <= 32.0 * loss_spacing
                           else "backtracking_no_improvement")
            break

    final = theta.numpy().copy()
    rollout_final = rollout_fn(final)
    sample_indices = sorted(set((0, min(1, len(theta_history) - 1),
                                 (len(theta_history) - 1) // 2,
                                 len(theta_history) - 1)))
    rollout_samples = [{
        "accepted_step": index,
        "theta": theta_history[index].tolist(),
        "loss": loss_history[index],
        "teacher_distance": float(np.linalg.norm(theta_history[index] - teacher)),
        "rollout": rollout_fn(theta_history[index]),
    } for index in sample_indices]
    return {
        "accepted_steps": len(loss_history) - 1,
        "stop_reason": stop_reason,
        "objective_evaluations": objective_evaluations,
        "backtracking_rejections": backtracking_rejections,
        "accepted_with_backtracking": accepted_with_backtracking,
        "damping_steps": damping_steps,
        "theta_history": [x.tolist() for x in theta_history],
        "loss_history": loss_history, "iterations": records,
        "final_theta": final.tolist(), "final_loss": current_loss,
        "parameter_error": (final - teacher).tolist(),
        "parameter_l2_error": float(np.linalg.norm(final - teacher)),
        "parameter_max_error": float(np.max(np.abs(final - teacher))),
        "rollout": rollout_final,
        "rollout_samples": rollout_samples,
        "recovered": bool(current_loss < staged.FINAL_GATE and
                          rollout_final.get("rollout_finite", True)),
    }


def alias_context(row):
    case_id = int(row["case_id"])
    trajectory, _ = d5._kepler_case(case_id)
    _, objective = d5._objectives("kepler5", trajectory)
    lower, upper, teacher = d5._config("kepler5")
    derivatives, direction_fn, _ = staged._make_n3(objective, lower, upper)
    rollout_fn = lambda theta: d5._rollout("kepler5", trajectory, theta)
    return objective, derivatives, direction_fn, lower, upper, teacher, rollout_fn


def control_context(row):
    objective, derivatives, direction_fn, lower, upper, teacher = kkt.context(row)
    system, case_id = row["system"], row["case_id"]
    source = json.loads(Path(row["source_path"]).read_text())
    if system == "kepler5":
        trajectory, _ = d5._kepler_case(int(case_id))
        rollout_fn = lambda theta: d5._rollout("kepler5", trajectory, theta)
    elif system == "full_k_sho":
        index = int(source["case_id"])
        corpus = np.load(cap.SHO_DATA)
        states = corpus[f"prefix_{index}"].copy()
        rollout_fn = lambda theta: cap.sho_rollout(states, teacher, theta)
    elif system == "full_k_ball":
        trajectories, _, _, _ = ball_teacher.TEACHER.build_bouncing_ball_trajectories(
            n_trajectories=ball4.COUNT, n_timesteps=ball_teacher.TOTAL_STEPS,
            dt=ball_teacher.DT, seed=ball4.SEED)
        trajectory = np.asarray(trajectories[int(case_id)], np.float64)
        rollout_fn = lambda theta: ball4._collision_diagnostics(trajectory, theta)
    elif system == "reflection_ball":
        trajectory = ball5._trajectories()[int(source["source_trajectory"])]
        rollout_fn = lambda theta: ball5._collision_diagnostics(trajectory, theta)
    else:
        raise ValueError(system)
    return objective, derivatives, direction_fn, lower, upper, teacher, rollout_fn


def summarize_iterations(result):
    active_sets = [tuple(row["active_lower"] + row["active_upper"])
                   for row in result["iterations"]]
    releases = 0
    for before, after in zip(active_sets, active_sets[1:]):
        releases += len(set(before) - set(after))
    return {
        "active_set_sequence": [list(x) for x in active_sets],
        "active_set_changes": sum(a != b for a, b in zip(active_sets, active_sets[1:])),
        "active_coordinate_releases": releases,
        "maximum_backtrack": max((row["accepted_backtrack"] or 0)
                                 for row in result["iterations"]),
        "median_accepted_alpha": float(np.median([
            row["accepted_alpha"] for row in result["iterations"] if row["accepted"]
        ])) if result["accepted_steps"] else None,
    }


def run_aliases(selected=None):
    rows = [json.loads(line) for line in UNRESOLVED.read_text().splitlines()]
    if selected is not None:
        rows = [rows[selected]]
    output = []
    for number, row in enumerate(rows, 1):
        ctx = alias_context(row)
        result = active_run(np.asarray(row["label_n2_endpoint"], np.float64),
                            *ctx)
        record = {
            "system": "kepler5", "case_id": int(row["case_id"]),
            "state_id": row["state_id"], "example_id": row["example_id"],
            "production": {
                "accepted_steps_observed": row["accepted_steps_so_far"],
                "latest_theta": row["uncapped_progress"]["theta"],
                "latest_loss": row["uncapped_progress"]["loss"],
                "stop_reason": "NO_GENUINE_STOP_OBSERVED",
            },
            "active_set": result,
            "active_set_summary": summarize_iterations(result),
        }
        output.append(record)
        atomic_json(OUT / "aliases" / f'{row["example_id"]}.json', record)
        print("alias", number, len(rows), row["case_id"], result["accepted_steps"],
              result["stop_reason"], result["recovered"], flush=True)
    return output


def run_controls():
    report = json.loads(CAP_REPORT.read_text())
    rows = [row for row in report["cases"]
            if (row["system"], str(row["case_id"])) in CONTROL_KEYS]
    output = []
    for number, row in enumerate(rows, 1):
        ctx = control_context(row)
        start = np.asarray(row["cap_suppressed"]["theta_history"][0], np.float64)
        result = active_run(start, *ctx)
        production = row["cap_suppressed"]
        same = {
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
        record = {
            "system": row["system"], "case_id": row["case_id"],
            "production": production, "active_set": result,
            "comparison": same, "trajectory_changed": not all(same.values()),
            "active_set_summary": summarize_iterations(result),
        }
        output.append(record)
        atomic_json(OUT / "controls" / f'{row["system"]}_{row["case_id"]}.json', record)
        print("control", number, len(rows), row["system"], row["case_id"],
              "changed", record["trajectory_changed"], flush=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias-index", type=int)
    parser.add_argument("--aliases-only", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    if args.controls_only:
        aliases = [json.loads(path.read_text())
                   for path in sorted((OUT / "aliases").glob("*.json"))]
    else:
        aliases = run_aliases(args.alias_index)
    if args.alias_index is not None:
        return
    controls = [] if args.aliases_only else run_controls()
    result = {
        "diagnostic": "exact active-set reduced Newton with no new cap or tolerance",
        "aliases": aliases, "controls": controls,
        "summary": {
            "aliases": len(aliases),
            "alias_recoveries": sum(row["active_set"]["recovered"] for row in aliases),
            "alias_stops": dict(Counter(row["active_set"]["stop_reason"] for row in aliases)),
            "alias_total_accepted_steps": sum(row["active_set"]["accepted_steps"] for row in aliases),
            "aliases_with_active_release": sum(
                row["active_set_summary"]["active_coordinate_releases"] > 0
                for row in aliases),
            "controls": len(controls),
            "changed_controls": sum(row["trajectory_changed"] for row in controls),
            "all_control_fields_exact": all(all(row["comparison"].values()) for row in controls),
        },
        "checkpoint_hashes_unchanged": before == hashes(),
    }
    atomic_json(OUT / "report.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
