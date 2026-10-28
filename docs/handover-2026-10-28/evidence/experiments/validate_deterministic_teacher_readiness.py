#!/usr/bin/env python3
"""Comprehensive teacher-readiness validation of the corrected recovery stack.

New cases run the actual chronological teacher:
N1 -> archive -> newest-first probes -> exact Powell N2 -> corrected box-QP N3.
Existing completed cohorts are retained as separately reported prior evidence.
Ground truth is used only after a branch completes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import qmc
import torch

from experiments import expand_chronological_archive_handoff_dataset as archive_data
from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.sho import run_full_k_transfer as full_k
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/deterministic_teacher_readiness")
CASES = OUT / "cases"
TRACES = OUT / "traces"
MANIFEST = OUT / "manifest.json"
NEW_PER_SYSTEM = 64
N1_CAP = 5_000
FINAL_GATE = 1.0e-20
SHO_SEED = 20_261_120
SYSTEMS = ("full_k_sho", "reflection_ball", "kepler5")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sobol_point(index: int, dimension: int, seed: int) -> np.ndarray:
    power = max(1, int(math.ceil(math.log2(index + 1))))
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(power)[index]


def sho_case(index: int) -> tuple[np.ndarray, dict, np.ndarray]:
    unit = sobol_point(index, 3, SHO_SEED)
    c = np.float64(-1.99) + np.float64(3.98) * np.float64(unit[0])
    initial = np.float64(-1.0) + np.float64(2.0) * unit[1:]
    teacher = np.asarray(((0.0, 1.0), (-1.0, c)), np.float64)
    states = np.empty((128, 2), np.float64)
    states[0] = initial
    for step in range(127):
        states[step + 1] = teacher @ states[step]
    return states, {
        "sobol_index": index, "sobol_unit": unit.tolist(), "seed": SHO_SEED,
        "c": float(c), "initial_delay_state": initial.tolist(),
    }, teacher.ravel()


def sho_objectives(states: np.ndarray):
    inputs = states[:-1].copy()
    targets = states[1:].copy()
    torch_inputs = torch.from_numpy(inputs)
    torch_targets = torch.from_numpy(targets)

    def objective(theta: np.ndarray) -> float:
        matrix = np.asarray(theta, np.float64).reshape(2, 2)
        residual = np.einsum("ij,nj->ni", matrix, inputs) - targets
        return float(np.mean(residual * residual))

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        residual = torch_inputs @ theta.reshape(2, 2).T - torch_targets
        return torch.mean(residual.square())

    return objective, torch_objective


def rollout_sho(states: np.ndarray, theta: np.ndarray, teacher: np.ndarray) -> dict:
    teacher_matrix = teacher.reshape(2, 2)
    recovered_matrix = theta.reshape(2, 2)
    truth = states[-1].copy()
    predicted = truth.copy()
    maximum = 0.0
    squared = 0.0
    values = 0
    exact = True
    for _ in range(4096):
        truth = teacher_matrix @ truth
        predicted = recovered_matrix @ predicted
        difference = predicted - truth
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared += float(np.sum(difference * difference))
        values += difference.size
        exact = exact and bool(np.array_equal(predicted.view(np.uint64), truth.view(np.uint64)))
    return {
        "continuation_max_error": maximum,
        "continuation_mse": squared / values,
        "collision_topology_regressions": 0,
        "rollout_finite": bool(np.isfinite(predicted).all()),
        "rollout_bit_exact": exact,
    }


def make_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    prior_manifest = json.loads(archive_data.MANIFEST.read_text())
    used_ball = [int(row["sobol_index"]) for row in prior_manifest["trajectories"]
                 if row["system"] == "reflection_ball"]
    rejected_ball = [int(row["sobol_index"]) for row in prior_manifest["ball_rejections"]]
    used_kepler = [int(row["sobol_index"]) for row in prior_manifest["trajectories"]
                   if row["system"] == "kepler5"]
    next_ball = max([63, *used_ball, *rejected_ball]) + 1
    ball_specs = []
    ball_rejections = []
    while len(ball_specs) < NEW_PER_SYSTEM:
        audit = archive_data.ball_identifiability(next_ball)
        if audit["classification"] == "ACCEPT_IDENTIFIABLE":
            ball_specs.append({
                "system": "reflection_ball", "case_id": next_ball,
                "sobol_index": next_ball, "identifiability": audit,
            })
        else:
            ball_rejections.append(audit)
        next_ball += 1
    first_kepler = max([63, *used_kepler]) + 1
    specs = [
        {"system": "full_k_sho", "case_id": index, "sobol_index": index}
        for index in range(NEW_PER_SYSTEM)
    ]
    specs.extend(ball_specs)
    specs.extend({
        "system": "kepler5", "case_id": index, "sobol_index": index,
    } for index in range(first_kepler, first_kepler + NEW_PER_SYSTEM))
    manifest = {
        "version": 1, "new_per_system": NEW_PER_SYSTEM,
        "systems": list(SYSTEMS), "cases": specs,
        "reflection_ball_rejections": ball_rejections,
        "disjoint_from_prior_archive_manifest": True,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def case_data(spec: dict):
    system = spec["system"]
    if system == "full_k_sho":
        trajectory, initial, teacher = sho_case(int(spec["sobol_index"]))
        objective, torch_objective = sho_objectives(trajectory)
        lower = np.full(4, -1.99, np.float64)
        upper = np.full(4, 1.99, np.float64)
        return trajectory, initial, objective, torch_objective, lower, upper, teacher
    if system == "reflection_ball":
        trajectory, initial = archive_data.ball_case(int(spec["sobol_index"]))
    else:
        trajectory, initial = archive_data.kepler_case(int(spec["sobol_index"]))
    objective, torch_objective = production._objectives(system, trajectory)
    lower, upper, teacher = production._config(system)
    return trajectory, initial, objective, torch_objective, lower, upper, teacher


def rollout(system: str, trajectory: np.ndarray, theta: np.ndarray,
            teacher: np.ndarray) -> dict:
    if system == "full_k_sho":
        return rollout_sho(trajectory, theta, teacher)
    value = production._rollout(system, trajectory, theta)
    value["rollout_bit_exact"] = bool(
        value["continuation_max_error"] == 0.0 and value["continuation_mse"] == 0.0
    )
    return value


def n3_geometry(theta_history, derivatives, direction_function, lower, upper) -> dict:
    constrained = 0
    shortcut = 0
    shortcut_errors = 0
    active_iterations = 0
    blocking = 0
    releases = 0
    routes = Counter()
    for value in theta_history[:-1]:
        theta = torch.from_numpy(np.asarray(value, np.float64).copy())
        _, gradient, hessian = derivatives(theta, torch.empty(0))
        unconstrained, _, _, _ = direction_function(gradient, hessian)
        direction, info = dimension_general_box_qp_direction(
            theta, gradient, hessian, lower, upper, direction_function
        )
        routes[info["route_status"]] += 1
        active_iterations += int(info["active_set_iterations"])
        blocking += int(info["blocking_events"])
        releases += int(info["release_events"])
        if info["used_unconstrained_feasible_shortcut"]:
            shortcut += 1
            shortcut_errors += int(
                unconstrained is None or direction is None or
                not bool(torch.equal(unconstrained, direction))
            )
        else:
            constrained += 1
    return {
        "box_qp_constrained_iterations": constrained,
        "unconstrained_feasible_shortcuts": shortcut,
        "unconstrained_shortcut_bit_errors": shortcut_errors,
        "active_set_iterations": active_iterations,
        "blocking_events": blocking, "release_events": releases,
        "box_qp_routes": dict(routes),
    }


def branch_path(spec: dict, n: int, index: int) -> Path:
    return (CASES / spec["system"] / f"case_{int(spec['case_id']):05d}" /
            "branches" / f"n{n:05d}_i{index + 1:05d}.json")


def run_branch(spec: dict, trajectory: np.ndarray, objective, torch_objective,
               lower: np.ndarray, upper: np.ndarray, teacher: np.ndarray,
               points: list[np.ndarray], losses: list[float], n: int, index: int) -> dict:
    path = branch_path(spec, n, index)
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    theta = lower + (upper - lower) * points[index]
    radius = (upper - lower) / float(n) ** (1.0 / len(lower))
    box_lower = np.maximum(lower, theta - radius)
    box_upper = np.minimum(upper, theta + radius)
    n2 = staged._run_n2(
        objective, theta, float(losses[index]), box_lower, box_upper
    )
    derivatives, direction_function, run_n3 = staged._make_n3(
        torch_objective, lower, upper
    )
    endpoint = polish.N2Endpoint(
        0, n2["theta"].copy(), float(n2["loss"]),
        "exact_learned_assembly_powell", "teacher_readiness", False,
    )
    n3 = run_n3(endpoint, torch.empty(0))
    final = np.asarray(n3.theta_history[-1], np.float64)
    parameter_error = final - teacher
    rollout_result = rollout(spec["system"], trajectory, final, teacher)
    recovered = bool(n3.loss_history[-1] < FINAL_GATE and rollout_result["rollout_finite"])
    geometry = n3_geometry(
        n3.theta_history, derivatives, direction_function, lower, upper
    )
    trace_path = (TRACES / spec["system"] / f"case_{int(spec['case_id']):05d}" /
                  f"n{n:05d}_i{index + 1:05d}.npz")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    n2_theta = np.asarray([row["theta"] for row in n2["trace"]], np.float64)
    n2_loss = np.asarray([row["loss"] for row in n2["trace"]], np.float64)
    np.savez_compressed(
        trace_path,
        n2_theta=n2_theta, n2_loss=n2_loss,
        n3_theta=np.asarray(n3.theta_history, np.float64),
        n3_loss=np.asarray(n3.loss_history, np.float64),
        n3_gradient_norm=np.asarray(n3.gradient_norms, np.float64),
        n3_condition=np.asarray(n3.condition_numbers, np.float64),
        n3_damping=np.asarray(n3.damping_values, np.float64),
    )
    result = {
        "system": spec["system"], "case_id": int(spec["case_id"]),
        "current_evaluation_count": n, "archive_index": index,
        "insertion_evaluation": index + 1,
        "archive_theta": theta.tolist(), "archive_loss": float(losses[index]),
        "n2_box_lower": box_lower.tolist(), "n2_box_upper": box_upper.tolist(),
        "n2_evaluations": int(n2["evaluations"]),
        "n2_stop_reason": n2["message"],
        "n2_endpoint": n2["theta"].tolist(), "n2_loss": float(n2["loss"]),
        "n3_accepted_steps": int(n3.accepted_steps),
        "n3_objective_evaluations": int(n3.actual_loss_evaluations),
        "n3_backtracking_rejections": int(n3.backtracking_rejections),
        "n3_damping_steps": int(n3.damping_steps),
        "n3_stop_reason": n3.stop_reason,
        "final_theta": final.tolist(), "final_loss": float(n3.loss_history[-1]),
        "entrywise_parameter_error": parameter_error.tolist(),
        "max_parameter_error": float(np.max(np.abs(parameter_error))),
        "parameter_l2_error": float(np.linalg.norm(parameter_error)),
        "parameter_bit_exact": bool(np.array_equal(final.view(np.uint64),
                                                    teacher.view(np.uint64))),
        "existing_recovery_criterion": recovered,
        "trace_npz": str(trace_path),
        "elapsed_seconds": float(time.monotonic() - started),
        **geometry, **rollout_result,
    }
    atomic_json(path, result)
    return result


def classify_no_success(branches: list[dict], hit_cap: bool,
                        implementation_error: str | None) -> str:
    if implementation_error:
        return "NUMERICAL/IMPLEMENTATION ERROR"
    if hit_cap:
        return "RESOURCE_INCOMPLETE"
    if branches and all(row["n2_evaluations"] == 0 for row in branches):
        return "N2 FAILURE"
    if branches and any(row["n3_stop_reason"] == "FLOAT64_STAGNATION_TAIL"
                        for row in branches):
        return "FLOAT64_STAGNATION_TAIL"
    if branches:
        return "N3 WRONG-BASIN / ALIAS"
    return "N1/ARCHIVE COVERAGE FAILURE"


def run_case(spec: dict) -> dict:
    result_path = CASES / spec["system"] / f"case_{int(spec['case_id']):05d}" / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    torch.set_num_threads(1)
    started = time.monotonic()
    trajectory, initial, objective, torch_objective, lower, upper, teacher = case_data(spec)
    runner = archive_data.ArchiveN1Runner(
        objective, lower, upper,
        {"full_k_sho": 8_100_000, "reflection_ball": 8_200_000,
         "kepler5": 8_300_000}[spec["system"]] + int(spec["case_id"]),
    )
    tested = []
    success = None
    implementation_error = None
    try:
        while len(runner.losses) < N1_CAP and success is None:
            runner.advance_one()
            n = len(runner.losses)
            order = [n - 1] + list(range(n - 1))
            for index in order:
                branch = run_branch(
                    spec, trajectory, objective, torch_objective,
                    lower, upper, teacher, runner.points, runner.losses, n, index,
                )
                tested.append(branch)
                if branch["existing_recovery_criterion"]:
                    success = branch
                    break
    except Exception as error:  # retained as a classified implementation result
        implementation_error = f"{type(error).__name__}: {error}"
    n = len(runner.losses)
    incumbent = int(np.argmin(runner.losses)) if runner.losses else None
    if success is not None:
        successful_index = int(success["archive_index"])
        classification = "RECOVERED"
    else:
        successful_index = None
        classification = classify_no_success(
            tested, n >= N1_CAP, implementation_error
        )
    result = {
        "system": spec["system"], "case_id": int(spec["case_id"]),
        "sobol_index": int(spec["sobol_index"]), "initial_condition": initial,
        "identifiability": spec.get("identifiability", {
            "classification": "IDENTIFIABLE_BY_CONSTRUCTION",
            "jacobian_rank": 4 if spec["system"] == "full_k_sho" else 5,
        }),
        "n1_evaluations": n, "n1_events": runner.events,
        "n1_continuous_traces": runner.traces,
        "archive_size": n, "archive_branches_tested": len(tested),
        "branch_sequence": [{
            "evaluation_count": row["current_evaluation_count"],
            "archive_index": row["archive_index"],
            "n2_evaluations": row["n2_evaluations"],
            "n3_accepted_steps": row["n3_accepted_steps"],
            "n3_stop_reason": row["n3_stop_reason"],
            "success": row["existing_recovery_criterion"],
        } for row in tested],
        "successful_handoff_site": successful_index,
        "successful_site_newest": bool(success is not None and successful_index == n - 1),
        "successful_site_incumbent": bool(success is not None and successful_index == incumbent),
        "successful_site_nonincumbent": bool(success is not None and successful_index != incumbent),
        "absorbing_success": bool(success is not None),
        "later_search_executed_after_success": False,
        "classification": classification,
        "implementation_error": implementation_error,
        "successful_branch": success,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    atomic_json(result_path, result)
    print(spec["system"], spec["case_id"], classification,
          f"N1={n}", f"branches={len(tested)}",
          f"loss={None if success is None else success['final_loss']}", flush=True)
    return result


def old_evidence() -> dict:
    sho_report = json.loads(Path(
        "outputs/sho/stage1B/checkpointed_resource_edirect/report.json"
    ).read_text())
    archive_states = [json.loads(line) for line in Path(
        "outputs/chronological_archive_handoff_supervision_expanded/expanded_archive_states.jsonl"
    ).read_text().splitlines()]
    archive_sites = [json.loads(line) for line in Path(
        "outputs/chronological_archive_handoff_supervision_expanded/expanded_archive_site_labels.jsonl"
    ).read_text().splitlines()]
    positive = {row["state_id"]: row for row in archive_states if row["target_any_good"]}
    site_by_state = defaultdict(list)
    for row in archive_sites:
        site_by_state[row["state_id"]].append(row)
    systems = {}
    for system in ("reflection_ball", "kepler5"):
        rows = [row for row in positive.values() if row["system"] == system]
        successful = []
        for state in rows:
            n = int(state["evaluation_count"])
            order = [n - 1] + list(range(n - 1))
            labels = {int(row["archive_index"]): row for row in site_by_state[state["state_id"]]}
            successful.append(next(labels[index] for index in order
                                   if labels[index]["target_good"]))
        systems[system] = {
            "old_cases": len(rows), "recoveries": len(successful),
            "max_parameter_error": [row["final_max_parameter_error"] for row in successful],
            "parameter_l2_error": [row["final_parameter_l2_error"] for row in successful],
            "rollout_max_error": [row["continuation_max_error"] for row in successful],
            "final_loss": [row["final_loss"] for row in successful],
        }
    sho_rows = json.loads(Path(
        "outputs/sho/stage1B/checkpointed_resource_edirect/results_partial.json"
    ).read_text())
    systems["full_k_sho"] = {
        "old_cases": len(sho_rows),
        "recoveries": int(sho_report["n3_successes"]),
        "max_parameter_error": [row["matrix_max_error"] for row in sho_rows],
        "parameter_l2_error": [row["matrix_error"] for row in sho_rows],
        "rollout_max_error": [row["rollout_max_error"] for row in sho_rows],
        "final_loss": [row["final_loss"] for row in sho_rows],
    }
    return systems


def distribution(values) -> dict:
    values = np.asarray(list(values), np.float64)
    return {
        "count": int(len(values)), "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9, method="higher")),
        "p99": float(np.quantile(values, 0.99, method="higher")),
        "maximum": float(np.max(values)),
    }


def summarize(results: list[dict], old: dict) -> dict:
    summary = {}
    for system in SYSTEMS:
        cases = [row for row in results if row["system"] == system]
        successes = [row for row in cases if row["classification"] == "RECOVERED"]
        branches = [row["successful_branch"] for row in successes]
        summary[system] = {
            "old_cases": old[system]["old_cases"], "new_cases": len(cases),
            "identifiable_new_cases": sum(
                row["identifiability"].get("jacobian_rank") in (4, 5) for row in cases
            ),
            "new_recoveries": len(successes),
            "failure_classifications": dict(Counter(
                row["classification"] for row in cases if row["classification"] != "RECOVERED"
            )),
            "parameter_max_error": distribution(
                row["max_parameter_error"] for row in branches
            ) if branches else None,
            "parameter_l2_error": distribution(
                row["parameter_l2_error"] for row in branches
            ) if branches else None,
            "rollout_max_error": distribution(
                row["continuation_max_error"] for row in branches
            ) if branches else None,
            "final_loss": distribution(row["final_loss"] for row in branches) if branches else None,
            "n1_evaluations_to_success": distribution(
                row["n1_evaluations"] for row in successes
            ) if successes else None,
            "archive_branches_tested": distribution(
                row["archive_branches_tested"] for row in successes
            ) if successes else None,
            "total_n2_evaluations_all_probes": distribution(
                sum(branch["n2_evaluations"] for branch in row["branch_sequence"])
                for row in successes
            ) if successes else None,
            "total_n3_accepted_steps_all_probes": distribution(
                sum(branch["n3_accepted_steps"] for branch in row["branch_sequence"])
                for row in successes
            ) if successes else None,
            "n2_evaluations_successful_branch": distribution(
                row["n2_evaluations"] for row in branches
            ) if branches else None,
            "n3_accepted_steps_successful_branch": distribution(
                row["n3_accepted_steps"] for row in branches
            ) if branches else None,
            "n3_stop_reasons": dict(Counter(row["n3_stop_reason"] for row in branches)),
            "bit_exact_parameters": sum(row["parameter_bit_exact"] for row in branches),
            "bit_exact_rollouts": sum(row["rollout_bit_exact"] for row in branches),
            "collision_topology_regressions": sum(
                row["collision_topology_regressions"] for row in branches
            ),
            "successful_site_newest": sum(row["successful_site_newest"] for row in successes),
            "successful_site_older": sum(
                not row["successful_site_newest"] for row in successes
            ),
            "successful_site_incumbent": sum(
                row["successful_site_incumbent"] for row in successes
            ),
            "successful_site_nonincumbent": sum(
                row["successful_site_nonincumbent"] for row in successes
            ),
            "box_qp_constrained_iterations": sum(
                row["box_qp_constrained_iterations"] for row in branches
            ),
            "unconstrained_shortcut_bit_errors": sum(
                row["unconstrained_shortcut_bit_errors"] for row in branches
            ),
            "prior_precision": {
                "recoveries": old[system]["recoveries"],
                "max_parameter_error": distribution(old[system]["max_parameter_error"]),
                "parameter_l2_error": distribution(old[system]["parameter_l2_error"]),
                "rollout_max_error": distribution(old[system]["rollout_max_error"]),
                "final_loss": distribution(old[system]["final_loss"]),
            },
        }
    return summary


def run_all(workers: int) -> None:
    manifest = make_manifest()
    specs = manifest["cases"]
    workers = max(1, min(int(workers), 4))
    if workers == 1:
        for spec in specs:
            result_path = (CASES / spec["system"] /
                           f"case_{int(spec['case_id']):05d}" / "result.json")
            if not result_path.exists():
                run_case(spec)
        return
    for offset in range(0, len(specs), workers):
        children = []
        for spec in specs[offset:offset + workers]:
            result_path = CASES / spec["system"] / f"case_{int(spec['case_id']):05d}" / "result.json"
            if result_path.exists():
                continue
            command = [sys.executable, "-m", __spec__.name, "--case",
                       spec["system"], str(spec["case_id"])]
            children.append((spec, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(spec["system"], spec["case_id"], child.wait())
                    for spec, child in children]
        failures = [row for row in failures if row[2] != 0]
        if failures:
            raise RuntimeError(failures)


def finalize(before_hashes: dict) -> dict:
    manifest = make_manifest()
    paths = [(spec, CASES / spec["system"] /
              f"case_{int(spec['case_id']):05d}" / "result.json")
             for spec in manifest["cases"]]
    missing = [spec for spec, path in paths if not path.exists()]
    results = [json.loads(path.read_text()) for _, path in paths if path.exists()]
    old = old_evidence()
    summary = summarize(results, old)
    after_hashes = hashes()
    site_labels = [json.loads(line) for line in Path(
        "outputs/chronological_archive_handoff_supervision_expanded/expanded_archive_site_labels.jsonl"
    ).read_text().splitlines()]
    alias_validation = json.loads(Path(
        "outputs/n1_candidate_handoff_supervision_uncapped_n3/"
        "integrated_box_qp_n3_validation/report.json"
    ).read_text())
    alias_controls = alias_validation["aliases"]
    ball_stagnation = [row for row in site_labels if row["n3_stop_reason"] ==
                       "FLOAT64_STAGNATION_TAIL"]
    all_recovered = not missing and all(
        row["classification"] == "RECOVERED" for row in results
    )
    topology_exact = all(
        row["successful_branch"]["collision_topology_regressions"] == 0
        for row in results if row["classification"] == "RECOVERED"
    )
    no_shortcut_errors = all(
        row["successful_branch"]["unconstrained_shortcut_bit_errors"] == 0
        for row in results if row["classification"] == "RECOVERED"
    )
    aliases_remain_bad = (
        alias_validation["summary"]["aliases"] == 15 and
        alias_validation["summary"]["aliases_exact"] == 15 and
        all(not row["target_good"] for row in ball_stagnation)
    )
    status = (
        "DETERMINISTIC_TEACHER_VALIDATED"
        if all_recovered and topology_exact and no_shortcut_errors and
        aliases_remain_bad and before_hashes == after_hashes
        else "DETERMINISTIC_TEACHER_NOT_READY"
    )
    report = {
        "status": status,
        "teacher": (
            "N1 -> chronological archive -> newest-first probes -> current-count box -> "
            "exact learned Powell N2 -> corrected box-QP uncapped N3 -> absorbing success"
        ),
        "old_evidence": old, "new_summary": summary,
        "new_cases_planned": len(manifest["cases"]),
        "new_cases_completed": len(results), "missing_cases": missing,
        "internal_consistency": {
            "exact_n2_report": "outputs/n2_current_archive_powell_assembly/report.json",
            "exact_n2_status": json.loads(Path(
                "outputs/n2_current_archive_powell_assembly/report.json"
            ).read_text())["status"],
            "box_qp_constrained_iterations": sum(
                row["successful_branch"]["box_qp_constrained_iterations"]
                for row in results if row["classification"] == "RECOVERED"
            ),
            "unconstrained_shortcut_bit_errors": sum(
                row["successful_branch"]["unconstrained_shortcut_bit_errors"]
                for row in results if row["classification"] == "RECOVERED"
            ),
            "absorbing_success_overwrite_errors": sum(
                row["later_search_executed_after_success"] for row in results
            ),
            "kepler_bad_alias_controls": len(alias_controls),
            "reflection_stagnation_bad_controls": len(ball_stagnation),
            "bad_alias_controls_mislabeled_good": sum(
                row["target_good"] for row in ball_stagnation
            ),
        },
        "failure_rows": [row for row in results if row["classification"] != "RECOVERED"],
        "checkpoint_hashes_unchanged": before_hashes == after_hashes,
        "checkpoint_count": len(before_hashes),
        "remaining_deterministic_scaffolding": [
            "N1 and archive loop/counter/index storage",
            "newest-first archive iteration and absorbing state bookkeeping",
            "N2 vector state, Powell loops, objective-evaluation counters, and strict-best archive",
            "N3 active-set loop and external torch.linalg.solve",
            "physical/event collision-time, wall identity, event order, and rollout scaffolding",
        ],
        "learned_or_exact_assembly": [
            "locked N1 arithmetic/geometry/routing and continuous Powell proposal assemblies",
            "EXACT_LEARNED_ASSEMBLY N2 bounded Powell arithmetic and direction routing",
            "frozen N3 gradient/Hessian/variable-power adjoints and learned control assemblies",
        ],
    }
    atomic_json(OUT / "report.json", report)
    atomic_json(OUT / "checkpoint_hashes_after.json", after_hashes)
    print(json.dumps({
        "status": status, "new_cases_completed": len(results),
        "missing": len(missing), "summary": summary,
        "internal_consistency": report["internal_consistency"],
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, metavar=("SYSTEM", "CASE_ID"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    before_path = OUT / "checkpoint_hashes_before.json"
    if not before_path.exists():
        atomic_json(before_path, hashes())
    before = json.loads(before_path.read_text())
    manifest = make_manifest()
    if args.case:
        system, case_id = args.case[0], int(args.case[1])
        spec = next(row for row in manifest["cases"]
                    if row["system"] == system and int(row["case_id"]) == case_id)
        run_case(spec)
        return
    if not args.finalize_only:
        run_all(args.workers)
    report = finalize(before)
    if report["status"] != "DETERMINISTIC_TEACHER_VALIDATED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
