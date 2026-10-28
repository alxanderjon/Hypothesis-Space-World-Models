#!/usr/bin/env python3
"""Audit accepted-step stagnation using existing N3 trajectories only.

This diagnostic does not alter or run the optimizer.  It reconstructs local
quadratic quantities at already-saved accepted states and records offline
parameter/rollout progress for four cohorts:

* reflection-ball case 186;
* normal terminating SHO/ball/Kepler controls;
* successful saved trajectories with the smallest accepted step fractions;
* the saved windows from the old Kepler boundary-alias ULP tails.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import analyze_early_n3_basin_trajectories as early
from experiments import diagnose_reflection_ball_case186_n3 as c186
from experiments import expand_chronological_archive_handoff_dataset as expanded
from experiments import map_direct_n3_correctability_geometry as basin
from experiments import test_n3_active_set_repair as active
from experiments import validate_d5_production_pipeline as production
from experiments.bouncing_ball import run_reflection_peelback as ball5
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/n3_accepted_step_stagnation_audit")
C186_REPORT = Path(
    "outputs/chronological_archive_handoff_supervision_expanded/"
    "case186_n3_diagnosis/report.json")
ACTIVE_ROOT = Path(
    "outputs/n1_candidate_handoff_supervision_uncapped_n3/"
    "active_set_n3_diagnostic")
NONTERM_ROOT = Path(
    "outputs/n1_candidate_handoff_supervision_uncapped_n3/"
    "nontermination_diagnosis")
UNRESOLVED = Path(
    "outputs/n1_candidate_handoff_supervision_uncapped_n3/"
    "handoff_promotion_unresolved.jsonl")
EARLY_TRACES = Path("outputs/early_n3_basin_trajectory_geometry/traces.jsonl")


def finite(value):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    return value


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def loss_ulps(before: float, after: float) -> int:
    if not (math.isfinite(before) and math.isfinite(after) and before >= after >= 0.0):
        return 0
    return int(np.float64(before).view(np.uint64)) - int(np.float64(after).view(np.uint64))


def rollout_scalars(value: dict) -> dict:
    maximum = value.get("continuation_max_error", value.get("rollout_max_error"))
    mse = value.get("continuation_mse", value.get("rollout_mse"))
    return {
        "maximum_error": maximum,
        "mse": mse,
        "finite": bool(value.get("rollout_finite", True)),
        "topology_regressions": value.get("collision_topology_regressions"),
    }


def step_metric(*, cohort: str, identity: str, step: int, theta: np.ndarray,
                next_theta: np.ndarray, loss: float, next_loss: float,
                gradient: np.ndarray, projected_gradient: np.ndarray,
                hessian: np.ndarray, direction: np.ndarray, damping: float,
                alpha: float, backtrack: int, teacher: np.ndarray,
                rollout_before: dict | None, rollout_after: dict | None,
                extra: dict | None = None) -> dict:
    displacement = next_theta - theta
    matrix = hessian + float(damping) * np.eye(len(theta), dtype=np.float64)
    full_model_change = (float(np.dot(gradient, direction)) +
                         0.5 * float(direction @ matrix @ direction))
    accepted_model_change = (float(np.dot(gradient, displacement)) +
                             0.5 * float(displacement @ matrix @ displacement))
    predicted_full = -full_model_change
    predicted_accepted = -accepted_model_change
    actual = float(loss - next_loss)
    direction_norm = float(np.linalg.norm(direction))
    displacement_norm = float(np.linalg.norm(displacement))
    spacing = np.abs(np.spacing(theta))
    displacement_ulps = np.divide(displacement, spacing, out=np.zeros_like(displacement),
                                  where=spacing != 0.0)
    distance_before = float(np.linalg.norm(theta - teacher))
    distance_after = float(np.linalg.norm(next_theta - teacher))
    result = {
        "cohort": cohort, "identity": identity, "accepted_step": int(step),
        "theta": theta.tolist(), "next_theta": next_theta.tolist(),
        "loss": float(loss), "next_loss": float(next_loss),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "projected_gradient_norm": float(np.linalg.norm(projected_gradient)),
        "direction_norm": direction_norm,
        "accepted_displacement_norm": displacement_norm,
        "accepted_alpha": float(alpha), "backtracking_count": int(backtrack),
        "displacement_over_direction": (displacement_norm / direction_norm
                                         if direction_norm else None),
        "displacement_local_ulps": displacement_ulps.tolist(),
        "displacement_local_ulps_l2": float(np.linalg.norm(displacement_ulps)),
        "displacement_local_ulps_max": float(np.max(np.abs(displacement_ulps))),
        "actual_loss_decrease": actual,
        "actual_improvement_loss_ulps": loss_ulps(loss, next_loss),
        "predicted_decrease_accepted_displacement": predicted_accepted,
        "predicted_decrease_full_direction": predicted_full,
        "actual_over_predicted_accepted": (actual / predicted_accepted
                                            if predicted_accepted > 0.0 else None),
        "actual_over_predicted_full": (actual / predicted_full
                                        if predicted_full > 0.0 else None),
        "accepted_model_over_full_model": (predicted_accepted / predicted_full
                                            if predicted_full > 0.0 else None),
        "teacher_distance_before": distance_before,
        "teacher_distance_after": distance_after,
        "teacher_distance_improvement": distance_before - distance_after,
        "rollout_before": None if rollout_before is None else rollout_scalars(rollout_before),
        "rollout_after": None if rollout_after is None else rollout_scalars(rollout_after),
    }
    if rollout_before is not None and rollout_after is not None:
        before = result["rollout_before"]["maximum_error"]
        after = result["rollout_after"]["maximum_error"]
        result["rollout_max_error_improvement"] = (
            None if before is None or after is None else float(before - after))
    if extra:
        result.update(extra)
    return result


def saved_newton_relation_metric(*, cohort: str, identity: str, step: int,
                                 theta: np.ndarray, next_theta: np.ndarray,
                                 loss: float, next_loss: float,
                                 gradient_norm: float, direction: np.ndarray,
                                 gradient_dot_direction: float, alpha: float,
                                 backtrack: int, teacher: np.ndarray,
                                 rollout_before: dict, rollout_after: dict,
                                 extra: dict | None = None) -> dict:
    """Metric for saved interior Newton states whose full tensors predate current code.

    For the exact saved DIRECT/DAMPED solve, B d = -g.  Consequently the
    quadratic-model decreases are -0.5*g.d at d and
    (-g.d)*(alpha-0.5*alpha^2) at alpha*d.  This avoids replaying a historical
    derivative implementation that has since been replaced.
    """
    displacement = next_theta - theta
    actual = float(loss - next_loss)
    direction_norm = float(np.linalg.norm(direction))
    displacement_norm = float(np.linalg.norm(displacement))
    spacing = np.abs(np.spacing(theta))
    displacement_ulps = np.divide(displacement, spacing, out=np.zeros_like(displacement),
                                  where=spacing != 0.0)
    predicted_full = -0.5 * float(gradient_dot_direction)
    predicted_accepted = (-float(gradient_dot_direction) *
                          (float(alpha) - 0.5 * float(alpha) ** 2))
    distance_before = float(np.linalg.norm(theta - teacher))
    distance_after = float(np.linalg.norm(next_theta - teacher))
    result = {
        "cohort": cohort, "identity": identity, "accepted_step": int(step),
        "theta": theta.tolist(), "next_theta": next_theta.tolist(),
        "loss": float(loss), "next_loss": float(next_loss),
        "gradient_norm": float(gradient_norm),
        "projected_gradient_norm": float(gradient_norm),
        "direction_norm": direction_norm,
        "accepted_displacement_norm": displacement_norm,
        "accepted_alpha": float(alpha), "backtracking_count": int(backtrack),
        "displacement_over_direction": displacement_norm / direction_norm,
        "displacement_local_ulps": displacement_ulps.tolist(),
        "displacement_local_ulps_l2": float(np.linalg.norm(displacement_ulps)),
        "displacement_local_ulps_max": float(np.max(np.abs(displacement_ulps))),
        "actual_loss_decrease": actual,
        "actual_improvement_loss_ulps": loss_ulps(loss, next_loss),
        "predicted_decrease_accepted_displacement": predicted_accepted,
        "predicted_decrease_full_direction": predicted_full,
        "actual_over_predicted_accepted": actual / predicted_accepted,
        "actual_over_predicted_full": actual / predicted_full,
        "accepted_model_over_full_model": predicted_accepted / predicted_full,
        "teacher_distance_before": distance_before,
        "teacher_distance_after": distance_after,
        "teacher_distance_improvement": distance_before - distance_after,
        "rollout_before": rollout_scalars(rollout_before),
        "rollout_after": rollout_scalars(rollout_after),
    }
    before = result["rollout_before"]["maximum_error"]
    after = result["rollout_after"]["maximum_error"]
    result["rollout_max_error_improvement"] = (
        None if before is None or after is None else float(before - after))
    if extra: result.update(extra)
    return result


def case186_metrics() -> tuple[list[dict], dict]:
    report = json.loads(C186_REPORT.read_text())
    expanded_root = Path("outputs/chronological_archive_handoff_supervision_expanded")
    manifest = json.loads((expanded_root / "manifest.json").read_text())
    spec = next(row for row in manifest["trajectories"]
                if row["system"] == "reflection_ball" and row["case_id"] == 186)
    trajectory, _ = expanded.trajectory_for(spec)
    _, objective = production._objectives("reflection_ball", trajectory)
    lower_np, upper_np, teacher = production._config("reflection_ball")
    derivatives, direction_fn, _ = staged._make_n3(objective, lower_np, upper_np)
    lower = torch.from_numpy(lower_np.copy()); upper = torch.from_numpy(upper_np.copy())
    theta = torch.tensor(report["n2"]["n2_endpoint"], dtype=torch.float64)
    compact = report["n3"]["compact_history"]
    rollout_cache: dict[bytes, dict] = {}

    def get_rollout(value: np.ndarray) -> dict:
        key = value.tobytes()
        if key not in rollout_cache:
            rollout_cache[key] = c186.rollout(trajectory, value)
        return rollout_cache[key]

    rows = []
    exact = []
    for index, saved in enumerate(compact):
        theta_np = theta.numpy().copy()
        _, gradient_t, hessian_t = derivatives(theta, torch.empty(0))
        direction_t, qp = dimension_general_box_qp_direction(
            theta, gradient_t, hessian_t, lower, upper, direction_fn)
        alpha = float(saved["accepted_alpha"])
        candidate = theta + alpha * direction_t
        next_theta = candidate.numpy().copy()
        next_loss = (float(compact[index + 1]["loss"]) if index + 1 < len(compact)
                     else float(report["n3"]["final_loss"]))
        projected = active.projected_gradient(theta, gradient_t, lower, upper).numpy()
        row = step_metric(
            cohort="case186_event_blocked", identity="reflection_ball_case_186",
            step=index, theta=theta_np, next_theta=next_theta,
            loss=float(saved["loss"]), next_loss=next_loss,
            gradient=gradient_t.numpy(), projected_gradient=projected,
            hessian=hessian_t.numpy(), direction=direction_t.numpy(),
            damping=float(qp["damping"]), alpha=alpha,
            backtrack=int(saved["accepted_backtrack"]), teacher=teacher,
            rollout_before=get_rollout(theta_np), rollout_after=get_rollout(next_theta),
            extra={"event_topology": saved["topology_sha256"],
                   "active_lower": saved["active_lower"],
                   "active_upper": saved["active_upper"]})
        rows.append(row)
        exact.append({
            # The saved value used torch.linalg.vector_norm; use the same
            # reduction here rather than NumPy's independently rounded norm.
            "direction_norm": float(torch.linalg.vector_norm(direction_t)) ==
                              saved["box_qp_direction_norm"],
            "displacement_norm": row["accepted_displacement_norm"] ==
                                 saved["actual_displacement_norm"],
            "loss_decrease": row["actual_loss_decrease"] == saved["loss_decrease"],
            "loss_ulps": row["actual_improvement_loss_ulps"] == saved["loss_decrease_ulps"],
        })
        theta = candidate.detach()
    return rows, {"steps": len(rows), "all_saved_fields_exact": all(
        all(item.values()) for item in exact), "field_checks": exact}


def normal_control_metrics() -> list[dict]:
    report = json.loads((ACTIVE_ROOT / "report.json").read_text())
    cap_report = json.loads(Path("outputs/n3_accepted_step_cap_validation/report.json").read_text())
    cap_rows = {(row["system"], str(row["case_id"])): row for row in cap_report["cases"]}
    output = []
    for control in report["controls"]:
        key = (control["system"], str(control["case_id"]))
        context = active.control_context(cap_rows[key])
        _, derivatives, _, lower_np, upper_np, teacher, rollout_fn = context
        lower = torch.from_numpy(lower_np.copy()); upper = torch.from_numpy(upper_np.copy())
        identity = f"{control['system']}:{control['case_id']}"
        cache: dict[bytes, dict] = {}
        def get_rollout(value):
            key_bytes = value.tobytes()
            if key_bytes not in cache: cache[key_bytes] = rollout_fn(value)
            return cache[key_bytes]
        for record in control["active_set"]["iterations"]:
            if not record["accepted"]: continue
            theta = np.asarray(record["theta"], np.float64)
            displacement = np.asarray(record["actual_displacement"], np.float64)
            next_theta = theta + displacement
            tensor = torch.from_numpy(theta.copy())
            _, gradient_t, hessian_t = derivatives(tensor, torch.empty(0))
            projected = active.projected_gradient(tensor, gradient_t, lower, upper).numpy()
            output.append(step_metric(
                cohort="normal_high_precision", identity=identity,
                step=int(record["accepted_step_before"]), theta=theta,
                next_theta=next_theta, loss=float(record["loss"]),
                next_loss=float(record["loss"] - record["loss_decrease"]),
                gradient=gradient_t.numpy(), projected_gradient=projected,
                hessian=hessian_t.numpy(), direction=np.asarray(record["full_direction"]),
                damping=float(record["damping"]), alpha=float(record["accepted_alpha"]),
                backtrack=int(record["accepted_backtrack"]), teacher=teacher,
                rollout_before=get_rollout(theta), rollout_after=get_rollout(next_theta),
                extra={"stop_reason": control["active_set"]["stop_reason"],
                       "recovered": control["active_set"]["recovered"]}))
    return output


def infer_alpha(theta: np.ndarray, next_theta: np.ndarray,
                direction: torch.Tensor, identity: str = "", step: int = -1) -> tuple[float, int]:
    tensor = torch.from_numpy(theta.copy())
    for backtrack in range(53):
        alpha = 2.0 ** (-backtrack)
        if np.array_equal((tensor + alpha * direction).numpy(), next_theta):
            return alpha, backtrack
    trials = [(float(np.max(np.abs((tensor + 2.0 ** (-k) * direction).numpy() -
                                   next_theta))), k) for k in range(53)]
    raise RuntimeError(f"could not recover saved accepted alpha {identity} step={step}; "
                       f"nearest={min(trials)}")


def successful_small_step_metrics() -> tuple[list[dict], list[dict]]:
    traces = [json.loads(line) for line in EARLY_TRACES.read_text().splitlines()]
    candidates = []
    for trace in traces:
        if not trace["label_PASS"]: continue
        alphas = [row["accepted_alpha"] for row in trace["detailed_iterations"]
                  if row.get("accepted")]
        if alphas: candidates.append((min(alphas), trace))
    # Four smallest-alpha successful Kepler paths plus the successful ball path
    # with the most backtracking in the same saved corpus.
    kepler = [row for row in sorted(candidates, key=lambda item: item[0])
              if row[1]["system"] == "kepler"][:4]
    ball = [row for row in sorted(candidates, key=lambda item: item[0])
            if row[1]["system"] == "bouncing_ball"][:1]
    selected = kepler + ball
    configs = {name: basin._system_config(name) for name in ("kepler", "bouncing_ball")}
    output = []
    selection = []
    for minimum, trace in selected:
        config = configs[trace["system"]]
        teacher = config["teacher"]
        theta_history = [np.asarray(value, np.float64) for value in trace["replay"]["theta_history"]]
        loss_history = trace["replay"]["loss_history"]
        if trace["system"] == "kepler":
            trajectory = staged.trajectories()[config["case"]]
            rollout_fn = lambda value, trajectory=trajectory: production._rollout(
                "kepler5", trajectory, value)
        else:
            trajectory = ball5._trajectories()[config["case"]]
            rollout_fn = lambda value, trajectory=trajectory: production._rollout(
                "reflection_ball", trajectory, value)
        cache = {}
        def get_rollout(value):
            key = value.tobytes()
            if key not in cache: cache[key] = rollout_fn(value)
            return cache[key]
        identity = f"{trace['system']}:sobol_{trace['sobol_index']}"
        selection.append({"identity": identity, "minimum_saved_alpha": minimum,
                          "accepted_steps": len(theta_history) - 1,
                          "final_loss": loss_history[-1]})
        # Use the exact saved first-three-step records.  They were generated by
        # an earlier frozen derivative assembly; replaying today's derivative
        # code changes the huge damped directions slightly, while the saved
        # states and B*d=-g relation retain the exact accepted-step evidence.
        for detailed in trace["detailed_iterations"]:
            if not detailed.get("accepted"): continue
            index = int(detailed["iteration_index"])
            theta = theta_history[index]; next_theta = theta_history[index + 1]
            output.append(saved_newton_relation_metric(
                cohort="successful_small_alpha", identity=identity, step=index,
                theta=theta, next_theta=next_theta, loss=float(loss_history[index]),
                next_loss=float(loss_history[index + 1]),
                gradient_norm=float(detailed["gradient_norm"]),
                direction=np.asarray(detailed["newton_direction"], np.float64),
                gradient_dot_direction=float(detailed["gradient_dot_direction"]),
                alpha=float(detailed["accepted_alpha"]),
                backtrack=int(detailed["accepted_backtrack_index"]), teacher=teacher,
                rollout_before=get_rollout(theta), rollout_after=get_rollout(next_theta),
                extra={"eventual_recovery": True,
                       "projected_equals_raw_because_strictly_interior":
                           bool(detailed["minimum_bound_margin"] > 0.0),
                       "saved_historical_derivative_state": True}))
    return output, selection


def old_alias_metrics() -> list[dict]:
    unresolved = {row["example_id"]: row for row in (
        json.loads(line) for line in UNRESOLVED.read_text().splitlines())}
    output = []
    for path in sorted(NONTERM_ROOT.glob("case_*.json")):
        saved = json.loads(path.read_text())
        source = unresolved[saved["example_id"]]
        _, derivatives, _, lower_np, upper_np, teacher, rollout_fn = active.alias_context(source)
        lower = torch.from_numpy(lower_np.copy()); upper = torch.from_numpy(upper_np.copy())
        identity = f"kepler_alias:{saved['case_id']}:{saved['example_id'][:12]}"
        cache = {}
        def get_rollout(value):
            key = value.tobytes()
            if key not in cache: cache[key] = rollout_fn(value)
            return cache[key]
        for phase in ("early_exact_window", "late_exact_window"):
            for number, record in enumerate(saved[phase]["steps"]):
                if not record["accepted"]: continue
                before = record["before"]
                theta = np.asarray(before["theta"], np.float64)
                displacement = np.asarray(record["parameter_delta"], np.float64)
                next_theta = theta + displacement
                tensor = torch.from_numpy(theta.copy())
                _, gradient_t, hessian_t = derivatives(tensor, torch.empty(0))
                projected = active.projected_gradient(tensor, gradient_t, lower, upper).numpy()
                output.append(step_metric(
                    cohort="old_kepler_boundary_alias_tail", identity=identity,
                    step=number, theta=theta, next_theta=next_theta,
                    loss=float(before["loss"]), next_loss=float(record["candidate_loss"]),
                    gradient=gradient_t.numpy(), projected_gradient=projected,
                    hessian=hessian_t.numpy(), direction=np.asarray(before["newton_direction"]),
                    damping=float(before["damping"]), alpha=float(record["accepted_alpha"]),
                    backtrack=int(record["accepted_backtrack"]), teacher=teacher,
                    rollout_before=get_rollout(theta), rollout_after=get_rollout(next_theta),
                    extra={"window": phase, "active_lower": before["active_lower_bounds"],
                           "active_upper": before["active_upper_bounds"]}))
    return output


def quantiles(rows: list[dict], key: str) -> dict:
    values = np.asarray([row[key] for row in rows
                         if row.get(key) is not None and math.isfinite(float(row[key]))],
                        np.float64)
    if not len(values): return {"count": 0}
    return {"count": len(values), "minimum": float(values.min()),
            "p10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "maximum": float(values.max())}


def summarize(rows: list[dict]) -> dict:
    keys = ["direction_norm", "accepted_displacement_norm", "accepted_alpha",
            "backtracking_count", "displacement_over_direction",
            "displacement_local_ulps_max", "actual_improvement_loss_ulps",
            "actual_loss_decrease", "predicted_decrease_accepted_displacement",
            "predicted_decrease_full_direction", "actual_over_predicted_accepted",
            "actual_over_predicted_full", "accepted_model_over_full_model",
            "gradient_norm", "projected_gradient_norm", "teacher_distance_improvement",
            "rollout_max_error_improvement"]
    return {key: quantiles(rows, key) for key in keys}


def main() -> None:
    before = hashes()
    case_rows, case_replay = case186_metrics()
    normal_rows = normal_control_metrics()
    successful_rows, successful_selection = successful_small_step_metrics()
    alias_rows = old_alias_metrics()
    rows = case_rows + normal_rows + successful_rows + alias_rows
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "accepted_steps.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(finite(row), separators=(",", ":"), allow_nan=False) + "\n")

    cohorts = {
        "case186_all": case_rows,
        "case186_late_64": case_rows[-64:],
        "normal_high_precision": normal_rows,
        "successful_small_alpha": successful_rows,
        "old_kepler_boundary_alias_tail": alias_rows,
    }
    # Explicit counterexamples to each tempting one-dimensional rule.
    smallest_success_alpha = min(successful_rows, key=lambda row: row["accepted_alpha"])
    smallest_success_model_fraction = min(
        (row for row in successful_rows if row["accepted_model_over_full_model"] is not None),
        key=lambda row: row["accepted_model_over_full_model"])
    low_ulp_success = [row for row in normal_rows
                       if row["actual_improvement_loss_ulps"] <= 32]
    report = {
        "diagnostic_only": True,
        "sources": {
            "case186": str(C186_REPORT),
            "normal_controls": str(ACTIVE_ROOT / "report.json"),
            "successful_small_alpha": str(EARLY_TRACES),
            "old_alias_tails": str(NONTERM_ROOT / "report.json"),
        },
        "counts": {name: len(value) for name, value in cohorts.items()},
        "case186_replay": case_replay,
        "successful_small_alpha_selection": successful_selection,
        "cohort_summaries": {name: summarize(value) for name, value in cohorts.items()},
        "counterexamples": {
            "smallest_alpha_that_still_recovered": smallest_success_alpha,
            "smallest_model_fraction_that_still_recovered": smallest_success_model_fraction,
            "normal_successful_steps_with_loss_improvement_at_most_32_ulps": len(low_ulp_success),
            "normal_low_ulp_examples": low_ulp_success[:8],
        },
        "checkpoint_hashes_unchanged": before == hashes(),
    }
    atomic_json(ROOT / "report.json", report)
    print(json.dumps({
        "counts": report["counts"], "case186_replay": case_replay,
        "smallest_success_alpha": smallest_success_alpha["accepted_alpha"],
        "normal_low_ulp_steps": len(low_ulp_success),
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
