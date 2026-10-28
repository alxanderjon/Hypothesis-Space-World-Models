#!/usr/bin/env python3
"""Diagnose uncapped five-parameter Kepler N3 nontermination.

No optimizer rule is changed.  Exact short windows are replayed from the saved
step-20 and latest checkpoint states so the accepted float64 tail can be
inspected without another long continuation.
"""
from __future__ import annotations

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

from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
UNRESOLVED = ROOT / "handoff_promotion_unresolved.jsonl"
COMPLETED = ROOT / "handoff_promotion_dataset_completed.jsonl"
OUT = ROOT / "nontermination_diagnosis"
WINDOW = 4


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def positive_ulp_decrease(before: float, after: float) -> int:
    assert before >= after >= 0.0
    a = int(np.asarray(np.float64(before)).view(np.uint64))
    b = int(np.asarray(np.float64(after)).view(np.uint64))
    return a - b


def rollout(trajectory, theta):
    value = production._rollout("kepler5", trajectory, np.asarray(theta, np.float64))
    return {"maximum_error": value["continuation_max_error"],
            "mse": value["continuation_mse"],
            "finite": value["rollout_finite"]}


def local_state(theta_np, torch_objective, derivatives, direction_fn,
                lower, upper, teacher):
    theta = torch.from_numpy(np.asarray(theta_np, np.float64).copy())
    loss, gradient, hessian = derivatives(theta, torch.empty(0))
    eigenvalues = torch.linalg.eigvalsh(hessian)
    direction, condition, damping, status = direction_fn(gradient, hessian)
    theta_array = theta.numpy().copy()
    gradient_array = gradient.numpy().copy()
    active_lower = np.flatnonzero(theta_array == lower)
    active_upper = np.flatnonzero(theta_array == upper)
    result = {
        "theta": theta_array.tolist(), "loss": float(loss),
        "teacher_distance": float(np.linalg.norm(theta_array - teacher)),
        "gradient": gradient_array.tolist(),
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "hessian_eigenvalues": eigenvalues.tolist(),
        "hessian_minimum_eigenvalue": float(torch.min(eigenvalues)),
        "hessian_maximum_eigenvalue": float(torch.max(eigenvalues)),
        "hessian_condition": float(condition), "damping": float(damping),
        "route": "NO_DIRECTION" if direction is None else (
            "DIRECT" if damping == 0.0 else "DAMPED"),
        "direction_status": status,
        "newton_direction": None if direction is None else direction.tolist(),
        "newton_direction_norm": None if direction is None else
            float(torch.linalg.vector_norm(direction)),
        "directional_derivative": None if direction is None else
            float(torch.dot(gradient, direction)),
        "lower_bound_margin": (theta_array - lower).tolist(),
        "upper_bound_margin": (upper - theta_array).tolist(),
        "active_lower_bounds": active_lower.tolist(),
        "active_upper_bounds": active_upper.tolist(),
        "active_bound_gradient": {}, "active_bound_direction": {},
        "loss_spacing": abs(float(np.spacing(np.float64(float(loss))))),
    }
    direction_array = None if direction is None else direction.numpy()
    for index in active_lower:
        result["active_bound_gradient"][str(index)] = float(gradient_array[index])
        result["active_bound_direction"][str(index)] = (
            None if direction_array is None else float(direction_array[index]))
    for index in active_upper:
        result["active_bound_gradient"][str(index)] = float(gradient_array[index])
        result["active_bound_direction"][str(index)] = (
            None if direction_array is None else float(direction_array[index]))
    return result


def exact_step(theta_np, torch_objective, derivatives, direction_fn, lower, upper,
               teacher):
    before = local_state(theta_np, torch_objective, derivatives, direction_fn,
                         lower, upper, teacher)
    theta = torch.from_numpy(np.asarray(theta_np, np.float64).copy())
    current_loss = before["loss"]
    _, gradient, hessian = derivatives(theta, torch.empty(0))
    direction, _, _, _ = direction_fn(gradient, hessian)
    record = {"before": before, "accepted": False, "accepted_backtrack": None,
              "accepted_alpha": None, "in_bounds_rejections": 0,
              "loss_rejections": 0, "unchanged_candidate": False,
              "candidate_loss": None, "loss_decrease": None,
              "loss_decrease_ulps": None, "parameter_delta": None,
              "parameter_delta_in_local_ulps": None, "actual_step_norm": None,
              "changed_coordinate_count": 0}
    if direction is None:
        return record, theta.numpy().copy()
    low = torch.from_numpy(lower)
    high = torch.from_numpy(upper)
    for backtrack in range(polish.MAX_BACKTRACKS + 1):
        alpha = 2.0 ** (-backtrack)
        candidate = theta + alpha * direction
        if bool(torch.equal(candidate, theta)):
            record["unchanged_candidate"] = True
            break
        if not bool(torch.all(candidate >= low) and torch.all(candidate <= high)):
            record["in_bounds_rejections"] += 1
            continue
        with torch.no_grad():
            candidate_loss = float(torch_objective(candidate))
        if candidate_loss < current_loss:
            candidate_np = candidate.numpy().copy()
            theta_array = theta.numpy().copy()
            delta = candidate_np - theta_array
            spacing = np.abs(np.spacing(theta_array))
            local_ulps = np.divide(delta, spacing, out=np.zeros_like(delta),
                                   where=spacing != 0.0)
            record.update(
                accepted=True, accepted_backtrack=backtrack, accepted_alpha=alpha,
                candidate_loss=candidate_loss,
                loss_decrease=float(current_loss - candidate_loss),
                loss_decrease_ulps=positive_ulp_decrease(current_loss, candidate_loss),
                parameter_delta=delta.tolist(),
                parameter_delta_in_local_ulps=local_ulps.tolist(),
                actual_step_norm=float(np.linalg.norm(delta)),
                changed_coordinate_count=int(np.count_nonzero(delta)),
            )
            return record, candidate_np
        record["loss_rejections"] += 1
    return record, theta.numpy().copy()


def window(theta, trajectory, torch_objective, derivatives, direction_fn,
           lower, upper, teacher):
    current = np.asarray(theta, np.float64).copy()
    initial_rollout = rollout(trajectory, current)
    records = []
    for _ in range(WINDOW):
        record, candidate = exact_step(current, torch_objective, derivatives,
                                       direction_fn, lower, upper, teacher)
        records.append(record)
        if not record["accepted"]:
            break
        current = candidate
    final_rollout = rollout(trajectory, current)
    return {"steps": records, "initial_rollout": initial_rollout,
            "final_rollout": final_rollout,
            "final_theta": current.tolist(),
            "final_teacher_distance": float(np.linalg.norm(current - teacher))}


def diagnose(row):
    case_id = int(row["case_id"])
    trajectory, _ = production._kepler_case(case_id)
    _, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    derivatives, direction_fn, _ = staged._make_n3(torch_objective, lower, upper)
    start = np.asarray(row["label_final_theta"], np.float64)
    latest = np.asarray(row["uncapped_progress"]["theta"], np.float64)
    start_local = local_state(start, torch_objective, derivatives, direction_fn,
                              lower, upper, teacher)
    latest_local = local_state(latest, torch_objective, derivatives, direction_fn,
                               lower, upper, teacher)
    early = window(start, trajectory, torch_objective, derivatives, direction_fn,
                   lower, upper, teacher)
    late = window(latest, trajectory, torch_objective, derivatives, direction_fn,
                  lower, upper, teacher)
    start_rollout = rollout(trajectory, start)
    latest_rollout = rollout(trajectory, latest)
    delta = latest - start
    active = latest_local["active_lower_bounds"] + latest_local["active_upper_bounds"]
    outward = []
    for index in latest_local["active_lower_bounds"]:
        if latest_local["active_bound_direction"][str(index)] < 0.0:
            outward.append(index)
    for index in latest_local["active_upper_bounds"]:
        if latest_local["active_bound_direction"][str(index)] > 0.0:
            outward.append(index)
    late_accepted = [step for step in late["steps"] if step["accepted"]]
    tiny = bool(late_accepted and all(step["actual_step_norm"] <= 1e-13
                                     for step in late_accepted))
    return {
        "example_id": row["example_id"], "case_id": case_id,
        "state_id": row["state_id"], "entity_index": row["entity_index"],
        "accepted_steps_saved": row["accepted_steps_so_far"],
        "step20": {"local": start_local, "rollout": start_rollout},
        "latest_checkpoint": {"local": latest_local, "rollout": latest_rollout},
        "step20_to_latest": {
            "parameter_delta": delta.tolist(),
            "parameter_delta_norm": float(np.linalg.norm(delta)),
            "loss_decrease": float(row["label_final_loss"] -
                                   row["uncapped_progress"]["loss"]),
            "teacher_distance_change": float(
                latest_local["teacher_distance"] - start_local["teacher_distance"]),
            "rollout_max_error_change": float(
                latest_rollout["maximum_error"] - start_rollout["maximum_error"]),
        },
        "early_exact_window": early, "late_exact_window": late,
        "active_bound_coordinates": active,
        "outward_newton_coordinates_at_active_bounds": outward,
        "late_steps_are_ulp_scale": tiny,
        "classification": (
            "DISTINCT_CONSTRAINED_BOUNDARY_ALIAS_WITH_ULP_TAIL"
            if active and outward and tiny else "OTHER_REVIEW_REQUIRED"),
    }


def control_rows():
    rows = [json.loads(line) for line in COMPLETED.read_text().splitlines()]
    candidates = [row for row in rows
                  if row["system"] == "kepler5" and
                  row["corrected_n3_stop_reason"] ==
                  "floating_point_no_further_improvement" and
                  row["corrected_n3_accepted_steps"] > 20 and
                  row["target_good"]]
    candidates.sort(key=lambda row: row["corrected_n3_accepted_steps"])
    picks = candidates[:2] + candidates[len(candidates)//2:len(candidates)//2+2] + candidates[-2:]
    result = []
    for row in picks:
        case_id = int(row["case_id"])
        trajectory, _ = production._kepler_case(case_id)
        _, objective = production._objectives("kepler5", trajectory)
        lower, upper, teacher = production._config("kepler5")
        derivatives, direction_fn, _ = staged._make_n3(objective, lower, upper)
        final = np.asarray(row["corrected_final_theta"], np.float64)
        result.append({
            "example_id": row["example_id"], "case_id": case_id,
            "accepted_steps": row["corrected_n3_accepted_steps"],
            "final_local": local_state(final, objective, derivatives, direction_fn,
                                       lower, upper, teacher),
            "final_rollout": rollout(trajectory, final),
            "terminal_probe": window(final, trajectory, objective, derivatives,
                                     direction_fn, lower, upper, teacher),
        })
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    rows = [json.loads(line) for line in UNRESOLVED.read_text().splitlines()]
    cases = []
    for index, row in enumerate(rows, 1):
        record = diagnose(row)
        cases.append(record)
        atomic_json(OUT / f"case_{int(row['case_id']):03d}_{row['example_id'][:12]}.json",
                    record)
        print(index, len(rows), row["case_id"], record["classification"], flush=True)
    controls = control_rows()
    late_steps = [step for case in cases for step in case["late_exact_window"]["steps"]
                  if step["accepted"]]
    early_steps = [step for case in cases for step in case["early_exact_window"]["steps"]
                   if step["accepted"]]
    active_counts = Counter(tuple(case["active_bound_coordinates"]) for case in cases)
    classifications = Counter(case["classification"] for case in cases)
    report = {
        "cases": len(cases), "window_steps": WINDOW,
        "classifications": dict(classifications),
        "active_bound_coordinate_patterns": {
            str(key): value for key, value in active_counts.items()},
        "outward_direction_at_active_bound_cases": sum(
            bool(case["outward_newton_coordinates_at_active_bounds"]) for case in cases),
        "ulp_scale_late_tail_cases": sum(case["late_steps_are_ulp_scale"] for case in cases),
        "teacher_distance_change_step20_to_latest": {
            "minimum": min(case["step20_to_latest"]["teacher_distance_change"] for case in cases),
            "median": float(np.median([case["step20_to_latest"]["teacher_distance_change"] for case in cases])),
            "maximum": max(case["step20_to_latest"]["teacher_distance_change"] for case in cases),
        },
        "rollout_max_error_change_step20_to_latest": {
            "minimum": min(case["step20_to_latest"]["rollout_max_error_change"] for case in cases),
            "median": float(np.median([case["step20_to_latest"]["rollout_max_error_change"] for case in cases])),
            "maximum": max(case["step20_to_latest"]["rollout_max_error_change"] for case in cases),
        },
        "late_accepted_step": {
            "count": len(late_steps),
            "backtrack_min": min(step["accepted_backtrack"] for step in late_steps),
            "backtrack_median": float(np.median([step["accepted_backtrack"] for step in late_steps])),
            "backtrack_max": max(step["accepted_backtrack"] for step in late_steps),
            "alpha_min": min(step["accepted_alpha"] for step in late_steps),
            "alpha_max": max(step["accepted_alpha"] for step in late_steps),
            "step_norm_min": min(step["actual_step_norm"] for step in late_steps),
            "step_norm_median": float(np.median([step["actual_step_norm"] for step in late_steps])),
            "step_norm_max": max(step["actual_step_norm"] for step in late_steps),
            "loss_decrease_min": min(step["loss_decrease"] for step in late_steps),
            "loss_decrease_median": float(np.median([step["loss_decrease"] for step in late_steps])),
            "loss_decrease_max": max(step["loss_decrease"] for step in late_steps),
            "loss_ulp_decrease_min": min(step["loss_decrease_ulps"] for step in late_steps),
            "loss_ulp_decrease_median": float(np.median([step["loss_decrease_ulps"] for step in late_steps])),
            "loss_ulp_decrease_max": max(step["loss_decrease_ulps"] for step in late_steps),
        },
        "early_accepted_step_count": len(early_steps),
        "normal_floating_point_stop_controls": len(controls),
        "cases_detail": cases,
        "controls": controls,
        "checkpoint_hashes_unchanged": before == hashes(),
        "data_scope_note": (
            "The relabel run retained exact step-20 and latest 1000-step checkpoint states, "
            "not every intervening accepted state. Exact early/late windows were replayed "
            "from those saved states; no long continuation was added."
        ),
    }
    atomic_json(OUT / "report.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("cases_detail", "controls")}, indent=2))


if __name__ == "__main__":
    main()
