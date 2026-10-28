#!/usr/bin/env python3
"""Diagnose six low-loss Kepler archive handoffs with weak parameter precision.

Production N1/N2/N3 are not changed.  The production result is replayed from
the saved archive handoff.  A second N3 call starts at the production final
state with only MAX_ACCEPTED_STEPS raised, because all audited production runs
terminated at that cap.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import types

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

SOURCE = Path("outputs/kepler_archive_absorbing_handoff/cases")
OUT = Path("outputs/kepler_archive_precision_outlier_audit")
OUTLIERS = (47, 31, 52, 54, 4, 40)
CONTROLS = (12, 27, 45, 3)
EXTENDED_ACCEPTED_CAP = 200


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def source(case_id: int) -> tuple[dict, dict]:
    path = SOURCE / f"case_{case_id:03d}.json"
    wrapper = json.loads(path.read_text())
    return wrapper, wrapper["successful_archived_handoff"]


def endpoint(theta, loss, label):
    return polish.N2Endpoint(0, np.asarray(theta, np.float64), float(loss),
                             label, label, False)


def extended_runner(run):
    namespace = dict(run.__globals__)
    namespace["MAX_ACCEPTED_STEPS"] = EXTENDED_ACCEPTED_CAP
    return types.FunctionType(run.__code__, namespace)


def derivative_state(theta: np.ndarray, derivatives, direction_fn,
                     torch_objective, lower: np.ndarray, upper: np.ndarray) -> dict:
    value = torch.from_numpy(np.asarray(theta, np.float64).copy())
    loss, gradient, hessian = derivatives(value, torch.empty(0))
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    eigenvalues = torch.linalg.eigvalsh(hessian)
    direction, condition, damping, status = direction_fn(gradient, hessian)
    record = dict(
        loss=float(loss), gradient=gradient.tolist(), gradient_norm=gradient_norm,
        hessian_eigenvalues=eigenvalues.tolist(), hessian_condition=float(condition),
        minimum_eigenvalue=float(torch.min(eigenvalues)),
        curvature_route=("NO_DIRECTION" if direction is None else
                         ("DIRECT" if damping == 0.0 else "DAMPED")),
        direction_status=status, damping=float(damping),
        newton_direction=None, newton_step_norm=None,
        directional_derivative=None, predicted_decrease=None,
        backtracking_attempts=[], accepted=False, accepted_backtrack=None,
        accepted_alpha=None, stop_diagnostic=None,
    )
    if direction is None:
        record["stop_diagnostic"] = status
        return record
    record["newton_direction"] = direction.tolist()
    record["newton_step_norm"] = float(torch.linalg.vector_norm(direction))
    directional = float(torch.dot(gradient, direction))
    predicted = max(0.0, -directional)
    record["directional_derivative"] = directional
    record["predicted_decrease"] = predicted
    unchanged = False
    current_loss = float(loss)
    for backtrack in range(polish.MAX_BACKTRACKS + 1):
        alpha = 2.0 ** (-backtrack)
        candidate = value + alpha * direction
        same = bool(torch.equal(candidate, value))
        in_bounds = bool(torch.all(candidate >= torch.from_numpy(lower)) and
                         torch.all(candidate <= torch.from_numpy(upper)))
        item = dict(backtrack=backtrack, alpha=alpha, unchanged=same,
                    in_bounds=in_bounds, candidate_loss=None, strict_decrease=False)
        if same:
            unchanged = True
            record["backtracking_attempts"].append(item)
            break
        if not in_bounds:
            record["backtracking_attempts"].append(item)
            continue
        with torch.no_grad():
            candidate_loss = float(torch_objective(candidate))
        item["candidate_loss"] = candidate_loss
        item["strict_decrease"] = bool(candidate_loss < current_loss)
        record["backtracking_attempts"].append(item)
        if candidate_loss < current_loss:
            record["accepted"] = True
            record["accepted_backtrack"] = backtrack
            record["accepted_alpha"] = alpha
            break
    if not record["accepted"]:
        spacing = abs(float(np.spacing(np.float64(current_loss))))
        record["loss_spacing"] = spacing
        record["predicted_decrease_over_spacing"] = (
            predicted / spacing if spacing > 0 else math.inf)
        record["stop_diagnostic"] = (
            "floating_point_no_further_improvement"
            if unchanged or predicted <= 32.0 * spacing
            else "backtracking_no_improvement")
    return record


def rollout(trajectory: np.ndarray, theta: np.ndarray) -> dict:
    result = production._rollout("kepler5", trajectory, theta)
    return dict(max_error=result["continuation_max_error"],
                mse=result["continuation_mse"],
                finite=result["rollout_finite"])


def result_details(result, trajectory, teacher, derivatives, direction_fn,
                   torch_objective, lower, upper) -> dict:
    final = np.asarray(result.theta_history[-1], np.float64)
    return dict(
        stop_reason=result.stop_reason,
        accepted_steps=int(result.accepted_steps),
        objective_evaluations=int(result.actual_loss_evaluations),
        rejected_newton_steps=int(result.rejected_newton_steps),
        backtracking_rejections=int(result.backtracking_rejections),
        accepted_with_backtracking=int(result.accepted_with_backtracking),
        hessian_evaluations=int(result.hessian_evaluations),
        damping_steps=int(result.damping_steps),
        condition_numbers=[float(x) for x in result.condition_numbers],
        damping_values=[float(x) for x in result.damping_values],
        gradient_norms=[float(x) for x in result.gradient_norms],
        theta_history=[np.asarray(x).tolist() for x in result.theta_history],
        loss_history=[float(x) for x in result.loss_history],
        final_theta=final.tolist(), final_loss=float(result.loss_history[-1]),
        parameter_error=(final-teacher).tolist(),
        parameter_l2_error=float(np.linalg.norm(final-teacher)),
        parameter_max_error=float(np.max(np.abs(final-teacher))),
        rollout=rollout(trajectory, final),
        final_local_state=derivative_state(final, derivatives, direction_fn,
                                           torch_objective, lower, upper),
    )


def audit_case(case_id: int, control: bool) -> dict:
    wrapper, saved = source(case_id)
    trajectory, initial = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    box_lower = np.asarray(saved["n2_box_lower"], np.float64)
    box_upper = np.asarray(saved["n2_box_upper"], np.float64)

    # Exact production N2 replay from the archived N1 site.
    n2 = staged._run_n2(objective, np.asarray(saved["theta"], np.float64),
                        float(saved["archived_loss"]), box_lower, box_upper)
    derivatives, direction_fn, run = staged._make_n3(torch_objective, lower, upper)
    reproduced = run(endpoint(n2["theta"], n2["loss"], "archive_replay"),
                     torch.empty(0))
    production_details = result_details(reproduced, trajectory, teacher,
        derivatives, direction_fn, torch_objective, lower, upper)
    production_match = dict(
        n2_theta_exact=bool(np.array_equal(np.asarray(n2["theta"]),
                                           np.asarray(saved["n2_endpoint"]))),
        n2_loss_exact=bool(float(n2["loss"]) == float(saved["n2_loss"])),
        final_theta_exact=bool(np.array_equal(np.asarray(production_details["final_theta"]),
                                              np.asarray(saved["final_theta"]))),
        final_loss_exact=bool(production_details["final_loss"] == float(saved["final_loss"])),
        accepted_steps_exact=bool(production_details["accepted_steps"] ==
                                  int(saved["n3_accepted_steps"])),
        stop_reason_exact=bool(production_details["stop_reason"] == saved["n3_stop_reason"]),
    )

    # Only the accepted-step cap is changed.  Start from the exact production
    # final state; all derivative, solve, bounds, and backtracking globals are
    # otherwise the production globals captured in `run`.
    continued_run = extended_runner(run)
    continued = continued_run(endpoint(production_details["final_theta"],
                                      production_details["final_loss"],
                                      "cap_only_continuation"), torch.empty(0))
    continuation_details = result_details(continued, trajectory, teacher,
        derivatives, direction_fn, torch_objective, lower, upper)
    initial_distance = production_details["parameter_l2_error"]
    final_distance = continuation_details["parameter_l2_error"]
    initial_rollout = production_details["rollout"]["max_error"]
    final_rollout = continuation_details["rollout"]["max_error"]

    if production_details["stop_reason"] == "max_accepted_steps" and \
       continuation_details["accepted_steps"] > 0 and \
       continuation_details["final_loss"] < production_details["final_loss"] and \
       final_distance < initial_distance:
        classification = "CORRECT_BASIN_EARLY_STOP"
    elif continuation_details["accepted_steps"] == 0:
        classification = "NUMERICAL_STALL"
    else:
        classification = "OTHER_EXPLAINED_CAUSE"

    return dict(
        case_id=case_id, control=control, initial_condition=initial,
        source_path=str(SOURCE/f"case_{case_id:03d}.json"),
        source_sha256=digest(SOURCE/f"case_{case_id:03d}.json"),
        archive_handoff=dict(current_evaluation=saved["current_evaluation"],
            insertion_evaluation=saved["insertion_evaluation"], theta=saved["theta"],
            archived_loss=saved["archived_loss"], n2_box_lower=saved["n2_box_lower"],
            n2_box_upper=saved["n2_box_upper"]),
        replay_n2=dict(theta=np.asarray(n2["theta"]).tolist(), loss=float(n2["loss"]),
                       evaluations=int(n2["evaluations"]), stop_reason=n2["message"]),
        production_match=production_match,
        production_n3=production_details,
        cap_only_continuation=continuation_details,
        changes=dict(additional_accepted_steps=continuation_details["accepted_steps"],
            loss_reduction_factor=(production_details["final_loss"] /
                continuation_details["final_loss"]
                if continuation_details["final_loss"] > 0 else math.inf),
            parameter_distance_reduction_factor=(initial_distance/final_distance
                if final_distance > 0 else math.inf),
            rollout_error_reduction_factor=(initial_rollout/final_rollout
                if final_rollout and initial_rollout is not None else math.inf)),
        classification=classification,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    rows=[]
    for case_id in OUTLIERS + CONTROLS:
        row=audit_case(case_id, case_id in CONTROLS)
        rows.append(row); atomic_json(OUT/f"case_{case_id:03d}.json", row)
        print(case_id, row["classification"],
              row["production_n3"]["final_loss"], "->",
              row["cap_only_continuation"]["final_loss"],
              "extra", row["changes"]["additional_accepted_steps"], flush=True)
    after=hashes()
    report=dict(outlier_cases=list(OUTLIERS), control_cases=list(CONTROLS),
                extended_accepted_cap=EXTENDED_ACCEPTED_CAP, rows=rows,
                before_checkpoint_hashes=before, after_checkpoint_hashes=after,
                checkpoint_hashes_unchanged=before==after)
    atomic_json(OUT/"report.json", report)


if __name__ == "__main__":
    main()
