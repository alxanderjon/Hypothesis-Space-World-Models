#!/usr/bin/env python3
"""Targeted diagnostic of removing only N3's accepted-step resource cap."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import types

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as d5
from experiments.bouncing_ball import run_first_transfer as ball_teacher
from experiments.bouncing_ball import run_full_k_transfer as ball4
from experiments.bouncing_ball import run_reflection_peelback as ball5
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import generate_checkpointed_edirect_teacher_data as sho_data
from experiments.sho import run_full_k_transfer as sho
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/n3_accepted_step_cap_validation")
KEPLER_SOURCE = Path("outputs/kepler_archive_absorbing_handoff/cases")
SHO_SOURCE = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases")
BALL4_SOURCE = Path("outputs/bouncing_ball/full_k_50_transfer/cases")
BALL5_SOURCE = Path("outputs/bouncing_ball/reflection_peelback/cases")
SHO_DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")

KEPLER_OUTLIERS = (47, 31, 52, 54, 4, 40)
KEPLER_CONTROLS = (12, 27, 45, 3)
SHO_CASES = ("companion_6", "companion_29", "companion_14", "interior_49")
BALL4_CASES = (17, 46, 33)
BALL5_CASES = (0, 4, 5)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def endpoint(theta, loss, label):
    return polish.N2Endpoint(0, np.asarray(theta, np.float64), float(loss),
                             label, label, False)


def without_accepted_step_cap(run):
    namespace = dict(run.__globals__)
    namespace["MAX_ACCEPTED_STEPS"] = math.inf
    return types.FunctionType(run.__code__, namespace)


def result_record(result, teacher, rollout_fn):
    theta = np.asarray(result.theta_history[-1], np.float64)
    damping = np.asarray(result.damping_values, np.float64)
    return dict(
        accepted_steps=int(result.accepted_steps),
        stop_reason=result.stop_reason,
        objective_evaluations=int(result.actual_loss_evaluations),
        rejected_newton_steps=int(result.rejected_newton_steps),
        backtracking_rejections=int(result.backtracking_rejections),
        accepted_with_backtracking=int(result.accepted_with_backtracking),
        hessian_evaluations=int(result.hessian_evaluations),
        curvature_routes=dict(
            direct=int(np.count_nonzero(damping == 0.0)),
            damped=int(np.count_nonzero(damping > 0.0)),
        ),
        damping_values=damping.tolist(),
        condition_numbers=[float(x) for x in result.condition_numbers],
        final_theta=theta.tolist(),
        final_loss=float(result.loss_history[-1]),
        parameter_error=(theta - teacher).tolist(),
        parameter_l2_error=float(np.linalg.norm(theta - teacher)),
        parameter_max_error=float(np.max(np.abs(theta - teacher))),
        rollout=rollout_fn(theta),
        theta_history=[np.asarray(x, np.float64).tolist() for x in result.theta_history],
        loss_history=[float(x) for x in result.loss_history],
    )


def replay(system, case_id, torch_objective, lower, upper, teacher,
           n2_theta, n2_loss, saved, rollout_fn, source_path):
    _, _, run = staged._make_n3(torch_objective, lower, upper)
    start = endpoint(n2_theta, n2_loss, f"{system}_{case_id}")
    production = run(start, torch.empty(0))
    natural = without_accepted_step_cap(run)(start, torch.empty(0))
    production_record = result_record(production, teacher, rollout_fn)
    natural_record = result_record(natural, teacher, rollout_fn)
    match = dict(
        final_theta_exact=bool(np.array_equal(
            np.asarray(production_record["final_theta"]), np.asarray(saved["final_theta"]))),
        final_loss_exact=bool(production_record["final_loss"] == float(saved["final_loss"])),
        accepted_steps_exact=bool(production_record["accepted_steps"] ==
                                  int(saved["n3_accepted_steps"])),
        stop_reason_exact=bool(production_record["stop_reason"] == saved["n3_stop"]),
    )
    return dict(
        system=system, case_id=str(case_id), source_path=str(source_path),
        production_match=match, production=production_record,
        cap_suppressed=natural_record,
        changes=dict(
            additional_accepted_steps=(natural_record["accepted_steps"] -
                                       production_record["accepted_steps"]),
            final_theta_unchanged=bool(np.array_equal(
                np.asarray(production_record["final_theta"]),
                np.asarray(natural_record["final_theta"]))),
            final_loss_unchanged=bool(production_record["final_loss"] ==
                                      natural_record["final_loss"]),
        ),
    )


def kepler_case(case_id):
    source_path = KEPLER_SOURCE / f"case_{case_id:03d}.json"
    saved = json.loads(source_path.read_text())["successful_archived_handoff"]
    trajectory, _ = d5._kepler_case(case_id)
    objective, torch_objective = d5._objectives("kepler5", trajectory)
    lower, upper, teacher = d5._config("kepler5")
    n2 = staged._run_n2(objective, np.asarray(saved["theta"], np.float64),
                        float(saved["archived_loss"]),
                        np.asarray(saved["n2_box_lower"], np.float64),
                        np.asarray(saved["n2_box_upper"], np.float64))
    canonical = dict(final_theta=saved["final_theta"], final_loss=saved["final_loss"],
                     n3_accepted_steps=saved["n3_accepted_steps"],
                     n3_stop=saved["n3_stop_reason"])
    return replay("kepler5", case_id, torch_objective, lower, upper, teacher,
                  n2["theta"], n2["loss"], canonical,
                  lambda theta: d5._rollout("kepler5", trajectory, theta), source_path)


def sho_rollout(states, teacher, theta):
    truth = np.empty((sho.ROLLOUT_STEPS + 1, 2), np.float64)
    prediction = np.empty_like(truth)
    truth[0] = prediction[0] = states[-1]
    teacher_matrix = teacher.reshape(2, 2)
    matrix = theta.reshape(2, 2)
    for index in range(sho.ROLLOUT_STEPS):
        truth[index + 1] = teacher_matrix @ truth[index]
        prediction[index + 1] = matrix @ prediction[index]
    error = prediction - truth
    return dict(continuation_max_error=float(np.max(np.abs(error))),
                continuation_mse=float(np.mean(error * error)),
                rollout_finite=bool(np.isfinite(prediction).all()))


def sho_cases():
    corpus = np.load(SHO_DATA)
    rows = []
    for name in SHO_CASES:
        source_path = SHO_SOURCE / name / "result.json"
        saved = json.loads(source_path.read_text())
        index = int(saved["case_id"])
        states = corpus[f"prefix_{index}"].copy()
        teacher = corpus["teachers"][index].reshape(-1).copy()
        _, torch_objective = sho_data._case_objectives(states)
        rows.append(replay(
            "full_k_sho", name, torch_objective, sho.LOWER, sho.UPPER, teacher,
            saved["n2_theta"], saved["n2_loss"], saved,
            lambda theta, s=states, t=teacher: sho_rollout(s, t, theta), source_path))
    return rows


def ball4_cases():
    trajectories, _, _, _ = ball_teacher.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=ball4.COUNT, n_timesteps=ball_teacher.TOTAL_STEPS,
        dt=ball_teacher.DT, seed=ball4.SEED)
    rows = []
    for case_id in BALL4_CASES:
        source_path = BALL4_SOURCE / f"trajectory_{case_id:02d}" / "result.json"
        saved = json.loads(source_path.read_text())
        trajectory = np.asarray(trajectories[case_id], np.float64)
        _, torch_objective = ball4.objectives(trajectory)
        rows.append(replay(
            "full_k_ball", case_id, torch_objective, ball4.LOWER, ball4.UPPER,
            ball4.TEACHER, saved["n2_theta"], saved["n2_loss"], saved,
            lambda theta, tr=trajectory: ball4._collision_diagnostics(tr, theta),
            source_path))
    return rows


def ball5_cases():
    trajectories = ball5._trajectories()
    rows = []
    for case_id in BALL5_CASES:
        source_path = BALL5_SOURCE / f"case_{case_id:02d}" / "result.json"
        saved = json.loads(source_path.read_text())
        trajectory = trajectories[int(saved["source_trajectory"])]
        _, torch_objective = ball5.objectives(trajectory)
        rows.append(replay(
            "reflection_ball", case_id, torch_objective, ball5.LOWER, ball5.UPPER,
            ball5.TEACHER, saved["n2_theta"], saved["n2_loss"], saved,
            lambda theta, tr=trajectory: ball5._collision_diagnostics(tr, theta),
            source_path))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = hashes()
    rows = []
    for case_id in KEPLER_OUTLIERS + KEPLER_CONTROLS:
        row = kepler_case(case_id)
        rows.append(row)
        atomic_json(OUT / f"kepler5_{case_id:03d}.json", row)
        print(row["system"], row["case_id"], row["production"]["accepted_steps"],
              row["production"]["stop_reason"], "->",
              row["cap_suppressed"]["accepted_steps"],
              row["cap_suppressed"]["stop_reason"], flush=True)
    for group in (sho_cases(), ball4_cases(), ball5_cases()):
        for row in group:
            rows.append(row)
            atomic_json(OUT / f'{row["system"]}_{row["case_id"]}.json', row)
            print(row["system"], row["case_id"], row["production"]["accepted_steps"],
                  row["production"]["stop_reason"], "->",
                  row["cap_suppressed"]["accepted_steps"],
                  row["cap_suppressed"]["stop_reason"], flush=True)
    after = hashes()
    max_steps = max(row["cap_suppressed"]["accepted_steps"] for row in rows)
    report = dict(
        diagnostic="Suppress only MAX_ACCEPTED_STEPS by setting it to mathematical infinity",
        cases=rows, maximum_naturally_required_accepted_steps=max_steps,
        all_production_replays_exact=all(all(row["production_match"].values()) for row in rows),
        all_previously_genuine_stops_unchanged=all(
            row["changes"]["final_theta_unchanged"] and
            row["changes"]["final_loss_unchanged"] and
            row["cap_suppressed"]["stop_reason"] == row["production"]["stop_reason"]
            for row in rows if row["production"]["stop_reason"] != "max_accepted_steps"),
        all_cap_limited_reach_genuine_stop=all(
            row["cap_suppressed"]["stop_reason"] != "max_accepted_steps" and
            row["cap_suppressed"]["accepted_steps"] >= row["production"]["accepted_steps"]
            for row in rows if row["production"]["stop_reason"] == "max_accepted_steps"),
        checkpoint_hashes_before=before, checkpoint_hashes_after=after,
        checkpoint_hashes_unchanged=before == after,
    )
    atomic_json(OUT / "report.json", report)


if __name__ == "__main__":
    main()
