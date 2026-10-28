#!/usr/bin/env python3
"""Audit exact projected/KKT geometry without changing N3."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as d5
from experiments import validate_n3_accepted_step_cap as cap
from experiments.bouncing_ball import run_first_transfer as ball_teacher
from experiments.bouncing_ball import run_full_k_transfer as ball4
from experiments.bouncing_ball import run_reflection_peelback as ball5
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import generate_checkpointed_edirect_teacher_data as sho_data
from experiments.diagnose_uncapped_n3_nontermination import exact_step
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
ALIAS_REPORT = ROOT / "nontermination_diagnosis" / "report.json"
CONTROL_REPORT = Path("outputs/n3_accepted_step_cap_validation/report.json")
OUT = ROOT / "projected_kkt_audit"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def projected(state, lower, upper, step=None):
    theta = np.asarray(state["theta"], np.float64)
    gradient = np.asarray(state["gradient"], np.float64)
    direction = (None if state["newton_direction"] is None else
                 np.asarray(state["newton_direction"], np.float64))
    active_lower = theta == lower
    active_upper = theta == upper
    free = ~(active_lower | active_upper)
    lower_satisfied = active_lower & (gradient >= 0.0)
    upper_satisfied = active_upper & (gradient <= 0.0)
    projected_gradient = gradient.copy()
    projected_gradient[lower_satisfied | upper_satisfied] = 0.0
    projected_direction = None
    if direction is not None:
        projected_direction = direction.copy()
        projected_direction[active_lower & (direction < 0.0)] = 0.0
        projected_direction[active_upper & (direction > 0.0)] = 0.0
    displacement = np.zeros_like(theta)
    ulps = np.zeros_like(theta)
    accepted = False
    if step is not None and step["accepted"]:
        accepted = True
        displacement = np.asarray(step["parameter_delta"], np.float64)
        ulps = np.asarray(step["parameter_delta_in_local_ulps"], np.float64)
    return {
        "theta": theta.tolist(), "loss": state["loss"],
        "active_lower": np.flatnonzero(active_lower).tolist(),
        "active_upper": np.flatnonzero(active_upper).tolist(),
        "lower_kkt_sign_satisfied": np.flatnonzero(lower_satisfied).tolist(),
        "upper_kkt_sign_satisfied": np.flatnonzero(upper_satisfied).tolist(),
        "active_sign_violations": np.flatnonzero(
            (active_lower & (gradient < 0.0)) |
            (active_upper & (gradient > 0.0))).tolist(),
        "free_coordinates": np.flatnonzero(free).tolist(),
        "gradient": gradient.tolist(),
        "raw_gradient_norm": float(np.linalg.norm(gradient)),
        "projected_gradient": projected_gradient.tolist(),
        "projected_kkt_residual_norm": float(np.linalg.norm(projected_gradient)),
        "projected_to_raw_gradient_ratio": (
            float(np.linalg.norm(projected_gradient) / np.linalg.norm(gradient))
            if np.linalg.norm(gradient) > 0.0 else 0.0),
        "free_gradient": np.where(free, gradient, 0.0).tolist(),
        "free_gradient_norm": float(np.linalg.norm(gradient[free])),
        "unconstrained_newton_direction": None if direction is None else direction.tolist(),
        "unconstrained_newton_direction_norm": None if direction is None else
            float(np.linalg.norm(direction)),
        "projected_newton_direction": None if projected_direction is None else
            projected_direction.tolist(),
        "projected_newton_direction_norm": None if projected_direction is None else
            float(np.linalg.norm(projected_direction)),
        "projected_to_raw_direction_ratio": (
            None if direction is None or np.linalg.norm(direction) == 0.0 else
            float(np.linalg.norm(projected_direction) / np.linalg.norm(direction))),
        "actual_step_accepted": accepted,
        "actual_feasible_displacement": displacement.tolist(),
        "actual_feasible_displacement_norm": float(np.linalg.norm(displacement)),
        "actual_feasible_displacement_local_ulps": ulps.tolist(),
        "actual_feasible_displacement_ulp_norm": float(np.linalg.norm(ulps)),
        "actual_feasible_displacement_max_abs_ulps": float(np.max(np.abs(ulps))),
        "accepted_backtrack": None if step is None else step["accepted_backtrack"],
        "accepted_alpha": None if step is None else step["accepted_alpha"],
        "loss_decrease": None if step is None else step["loss_decrease"],
        "loss_decrease_ulps": None if step is None else step["loss_decrease_ulps"],
    }


def context(row):
    system = row["system"]
    case_id = row["case_id"]
    source = json.loads(Path(row["source_path"]).read_text())
    if system == "kepler5":
        trajectory, _ = d5._kepler_case(int(case_id))
        _, objective = d5._objectives("kepler5", trajectory)
        lower, upper, teacher = d5._config("kepler5")
    elif system == "full_k_sho":
        saved = source
        index = int(saved["case_id"])
        corpus = np.load(cap.SHO_DATA)
        states = corpus[f"prefix_{index}"].copy()
        teacher = corpus["teachers"][index].reshape(-1).copy()
        _, objective = sho_data._case_objectives(states)
        lower, upper = cap.sho.LOWER.copy(), cap.sho.UPPER.copy()
    elif system == "full_k_ball":
        trajectories, _, _, _ = ball_teacher.TEACHER.build_bouncing_ball_trajectories(
            n_trajectories=ball4.COUNT, n_timesteps=ball_teacher.TOTAL_STEPS,
            dt=ball_teacher.DT, seed=ball4.SEED)
        trajectory = np.asarray(trajectories[int(case_id)], np.float64)
        _, objective = ball4.objectives(trajectory)
        lower, upper, teacher = ball4.LOWER.copy(), ball4.UPPER.copy(), ball4.TEACHER.copy()
    elif system == "reflection_ball":
        saved = source
        trajectory = ball5._trajectories()[int(saved["source_trajectory"])]
        _, objective = ball5.objectives(trajectory)
        lower, upper, teacher = ball5.LOWER.copy(), ball5.UPPER.copy(), ball5.TEACHER.copy()
    else:
        raise ValueError(system)
    derivatives, direction_fn, _ = staged._make_n3(objective, lower, upper)
    return objective, derivatives, direction_fn, lower, upper, teacher


def local(theta_np, derivatives, direction_fn, lower, upper):
    theta = torch.from_numpy(np.asarray(theta_np, np.float64).copy())
    loss, gradient, hessian = derivatives(theta, torch.empty(0))
    direction, condition, damping, status = direction_fn(gradient, hessian)
    eigenvalues = torch.linalg.eigvalsh(hessian)
    return {
        "theta": theta.tolist(), "loss": float(loss),
        "gradient": gradient.tolist(),
        "newton_direction": None if direction is None else direction.tolist(),
        "condition": float(condition), "damping": float(damping),
        "direction_status": status, "hessian_eigenvalues": eigenvalues.tolist(),
        "active_lower_bounds": np.flatnonzero(theta.numpy() == lower).tolist(),
        "active_upper_bounds": np.flatnonzero(theta.numpy() == upper).tolist(),
    }


def aliases():
    source = json.loads(ALIAS_REPORT.read_text())
    records = []
    lower, upper, _ = d5._config("kepler5")
    for case in source["cases_detail"]:
        for phase in ("early_exact_window", "late_exact_window"):
            for index, step in enumerate(case[phase]["steps"]):
                records.append({
                    "system": "kepler5_alias", "case_id": case["case_id"],
                    "state_id": case["state_id"], "phase": phase,
                    "window_index": index,
                    "geometry": projected(step["before"], lower, upper, step),
                })
    return records


def controls():
    report = json.loads(CONTROL_REPORT.read_text())
    output = []
    for row in report["cases"]:
        objective, derivatives, direction_fn, lower, upper, teacher = context(row)
        history = [np.asarray(x, np.float64)
                   for x in row["cap_suppressed"]["theta_history"]]
        first = max(0, len(history) - 4)
        # Include the actual N2 endpoint as an ordinary nonconverged control,
        # plus the final three accepted states and genuine stopping state.
        indices = sorted(set([0, *range(first, len(history))]))
        for index in indices:
            theta = history[index]
            state = local(theta, derivatives, direction_fn, lower, upper)
            step, candidate = exact_step(theta, objective, derivatives, direction_fn,
                                         lower, upper, teacher)
            matches_next = None
            if index + 1 < len(history):
                matches_next = bool(step["accepted"] and
                                    np.array_equal(candidate, history[index + 1]))
            output.append({
                "system": row["system"], "case_id": row["case_id"],
                "history_index": index, "accepted_steps_total": len(history) - 1,
                "is_initial_n2_endpoint": index == 0,
                "is_genuine_stopping_state": index == len(history) - 1,
                "stop_reason": row["cap_suppressed"]["stop_reason"]
                    if index == len(history) - 1 else None,
                "next_state_exact_match": matches_next,
                "geometry": projected(state, lower, upper, step),
            })
    return output


def ranges(records, key):
    values = [row["geometry"][key] for row in records
              if row["geometry"][key] is not None]
    return {"count": len(values), "minimum": min(values),
            "median": float(np.median(values)), "maximum": max(values)}


def main():
    before = hashes()
    alias = aliases()
    control = controls()
    terminal = [row for row in control if row["is_genuine_stopping_state"]]
    preterminal = [row for row in control if not row["is_genuine_stopping_state"]]
    initial = [row for row in control if row["is_initial_n2_endpoint"] and
               not row["is_genuine_stopping_state"]]
    near_terminal = [row for row in preterminal if not row["is_initial_n2_endpoint"]]
    by_system = {}
    for system in sorted({row["system"] for row in terminal}):
        rows = [row for row in terminal if row["system"] == system]
        by_system[system] = {
            "states": len(rows),
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "projected_gradient_norm": ranges(rows, "projected_kkt_residual_norm"),
            "projected_direction_norm": ranges(rows, "projected_newton_direction_norm"),
            "actual_displacement_norm": ranges(rows, "actual_feasible_displacement_norm"),
            "active_bound_states": sum(bool(row["geometry"]["active_lower"] or
                                            row["geometry"]["active_upper"])
                                       for row in rows),
        }
    result = {
        "kkt_sign_convention": {
            "lower_bound": "g_i >= 0; negative-gradient descent points outward",
            "upper_bound": "g_i <= 0; negative-gradient descent points outward",
            "free_coordinate": "g_i = 0",
            "projected_gradient": (
                "zero g_i exactly when x_i==lower and g_i>=0, or x_i==upper and g_i<=0; "
                "retain g_i otherwise"),
            "projected_newton_direction": (
                "zero d_i exactly when x_i==lower and d_i<0, or x_i==upper and d_i>0; "
                "retain d_i otherwise"),
        },
        "alias_states": alias, "control_states": control,
        "summary": {
            "alias_records": len(alias),
            "alias_active_sign_violations": sum(bool(row["geometry"]["active_sign_violations"])
                                                for row in alias),
            "alias_projected_gradient_norm": ranges(alias, "projected_kkt_residual_norm"),
            "alias_raw_gradient_norm": ranges(alias, "raw_gradient_norm"),
            "alias_projected_gradient_ratio": ranges(alias, "projected_to_raw_gradient_ratio"),
            "alias_projected_direction_norm": ranges(alias, "projected_newton_direction_norm"),
            "alias_projected_direction_ratio": ranges(alias, "projected_to_raw_direction_ratio"),
            "alias_actual_displacement_norm": ranges(alias, "actual_feasible_displacement_norm"),
            "alias_actual_displacement_ulp_norm": ranges(alias, "actual_feasible_displacement_ulp_norm"),
            "control_preterminal_records": len(preterminal),
            "control_preterminal_projected_gradient_norm": ranges(preterminal, "projected_kkt_residual_norm"),
            "control_preterminal_projected_direction_norm": ranges(preterminal, "projected_newton_direction_norm"),
            "ordinary_nonconverged_initial_records": len(initial),
            "ordinary_nonconverged_initial_projected_gradient_norm": ranges(initial, "projected_kkt_residual_norm"),
            "ordinary_nonconverged_initial_projected_direction_norm": ranges(initial, "projected_newton_direction_norm"),
            "near_terminal_accepted_records": len(near_terminal),
            "near_terminal_projected_gradient_norm": ranges(near_terminal, "projected_kkt_residual_norm"),
            "near_terminal_projected_direction_norm": ranges(near_terminal, "projected_newton_direction_norm"),
            "control_terminal_records": len(terminal),
            "control_terminal_projected_gradient_norm": ranges(terminal, "projected_kkt_residual_norm"),
            "control_terminal_projected_direction_norm": ranges(terminal, "projected_newton_direction_norm"),
            "control_terminal_actual_displacement_norm": ranges(terminal, "actual_feasible_displacement_norm"),
            "control_terminal_by_system": by_system,
            "control_next_state_replay_exact": all(
                row["next_state_exact_match"] is not False for row in preterminal),
        },
        "conclusion": {
            "aliases_are_kkt_stationary": False,
            "reason": (
                "active-bound signs satisfy KKT, but free-coordinate/projected gradients and "
                "projected Newton directions remain large; only the production scalar-backtracked "
                "actual displacement collapses"),
            "exact_alias_structure": (
                "an exactly active bound with KKT-compatible gradient sign and an outward raw "
                "Newton component forces scalar alpha down until that component rounds unchanged"),
            "structure_is_convergence_certificate": False,
            "smallest_unresolved_quantity": (
                "a numerical notion of stationarity/progress for the feasible projected problem; "
                "exact active-set logic alone cannot declare convergence because the projected "
                "gradient and projected direction are nonzero"),
        },
        "checkpoint_hashes_unchanged": before == hashes(),
    }
    atomic_json(OUT / "report.json", result)
    print(json.dumps({"summary": result["summary"],
                      "conclusion": result["conclusion"],
                      "checkpoint_hashes_unchanged": result["checkpoint_hashes_unchanged"]},
                     indent=2))


if __name__ == "__main__":
    main()
