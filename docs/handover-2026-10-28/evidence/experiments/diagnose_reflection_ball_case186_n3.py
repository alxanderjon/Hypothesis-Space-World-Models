#!/usr/bin/env python3
"""Targeted, bounded diagnosis of reflection-ball archive case 186.

The N1/N2 replay and each N3 iteration use the production implementations and
float64 order.  MAX_DIAGNOSTIC_STEPS only bounds trace collection; it is never
used as an optimizer stop or a GOOD/BAD label.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import expand_chronological_archive_handoff_dataset as expanded
from experiments import test_n3_active_set_repair as active
from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/chronological_archive_handoff_supervision_expanded")
OUT = ROOT / "case186_n3_diagnosis"
MAX_DIAGNOSTIC_STEPS = 128
MAX_DIAGNOSTIC_SECONDS = 720.0
SNAPSHOT_STEPS = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512}


def atomic_json(path: Path, value) -> None:
    def finite(item):
        if isinstance(item, dict):
            return {key: finite(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [finite(entry) for entry in item]
        if isinstance(item, tuple):
            return [finite(entry) for entry in item]
        if isinstance(item, float) and not math.isfinite(item):
            return "NaN" if math.isnan(item) else ("Infinity" if item > 0 else "-Infinity")
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bits_equal(a, b) -> bool:
    left = np.ascontiguousarray(np.asarray(a, np.float64)).view(np.uint64)
    right = np.ascontiguousarray(np.asarray(b, np.float64)).view(np.uint64)
    return left.shape == right.shape and bool(np.array_equal(left, right))


def loss_ulp_decrease(before: float, after: float) -> int:
    if not (before >= after >= 0.0):
        return 0
    return int(np.float64(before).view(np.uint64)) - int(np.float64(after).view(np.uint64))


def topology(trajectory: np.ndarray, theta: np.ndarray) -> dict:
    prefix = trajectory[:expanded.ball.ball.PREFIX_STATES - 1]
    signatures = [expanded.ball.event_signature(state, theta) for state in prefix]
    payload = json.dumps(signatures, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "impact_count": int(sum(len(value) for value in signatures)),
        "impacting_transition_count": int(sum(bool(value) for value in signatures)),
        "multiple_impact_transition_count": int(sum(len(value) > 1 for value in signatures)),
        "signature_histogram": {str(key): value for key, value in Counter(signatures).items()},
    }


def rollout(trajectory: np.ndarray, theta: np.ndarray) -> dict:
    result = production._rollout("reflection_ball", trajectory, theta)
    return {
        "continuation_max_error": result["continuation_max_error"],
        "continuation_mse": result["continuation_mse"],
        "collision_topology_regressions": result["collision_topology_regressions"],
        "rollout_finite": result["rollout_finite"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before_hashes = hashes()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    spec = next(row for row in manifest["trajectories"]
                if row["system"] == "reflection_ball" and row["case_id"] == 186)
    trajectory, initial = expanded.trajectory_for(spec)
    objective, torch_objective = production._objectives("reflection_ball", trajectory)
    lower_np, upper_np, teacher = production._config("reflection_ball")

    runner = expanded.ArchiveN1Runner(objective, lower_np, upper_np,
                                      7_100_000 + 186)
    runner.advance_one()
    # The interrupted case persisted n=1 lossless state and the n=2/site-0
    # query.  It correctly did not persist an incomplete n=2 archive state.
    saved_path = ROOT / "states/reflection_ball_new_case_00186_n00001.npz"
    saved = np.load(saved_path)
    replay = {
        "saved_n1_normalized_sites_bit_exact": bits_equal(
            runner.points, saved["normalized_sites"]),
        "saved_n1_site_losses_bit_exact": bits_equal(
            runner.losses, saved["site_losses"]),
        "evaluation_count_exact": len(runner.losses) == int(saved["evaluation_count"]),
        "incomplete_n2_state_was_not_persisted": not (
            ROOT / "states/reflection_ball_new_case_00186_n00002.npz").exists(),
    }
    runner.advance_one()
    saved_n2_site0 = json.loads((ROOT / "query_cache/reflection_ball/case_00186/"
                                 "n00002_i00001.json").read_text())
    replay["n2_site0_theta_bit_exact"] = bits_equal(
        lower_np + (upper_np - lower_np) * np.asarray(runner.points[0]),
        saved_n2_site0["theta"])
    replay["n2_site0_loss_bit_exact"] = bits_equal(
        [runner.losses[0]], [saved_n2_site0["archived_loss"]])
    replay["target_site_reconstructed_by_exact_deterministic_n1"] = True
    archived_theta = lower_np + (upper_np - lower_np) * np.asarray(runner.points[1])
    archived_loss = float(runner.losses[1])
    radius = (upper_np - lower_np) / 2.0 ** (1.0 / 5.0)
    box_lower = np.maximum(lower_np, archived_theta - radius)
    box_upper = np.minimum(upper_np, archived_theta + radius)
    n2 = staged._run_n2(objective, archived_theta, archived_loss, box_lower, box_upper)
    n2_record = {
        "archived_theta": archived_theta.tolist(), "archived_loss": archived_loss,
        "box_radius": radius.tolist(), "box_lower": box_lower.tolist(),
        "box_upper": box_upper.tolist(), "n2_endpoint": n2["theta"].tolist(),
        "n2_loss": float(n2["loss"]), "n2_evaluations": int(n2["evaluations"]),
        "n2_stop_reason": n2["message"], "n2_historical_gate": bool(n2["gate"]),
        "teacher_inside_box": bool(np.all(teacher >= box_lower) and np.all(teacher <= box_upper)),
    }
    atomic_json(OUT / "replay_and_n2.json", {"spec": spec, "initial": initial,
                                             "replay": replay, "n2": n2_record})
    if not all(replay.values()):
        raise RuntimeError(replay)

    derivatives, unconstrained_direction_fn, _ = staged._make_n3(
        torch_objective, lower_np, upper_np)
    lower = torch.from_numpy(lower_np.copy()); upper = torch.from_numpy(upper_np.copy())
    theta = torch.from_numpy(np.asarray(n2["theta"], np.float64).copy())
    with torch.no_grad():
        recomputed_start_loss = float(torch_objective(theta))
    # Production seeds N3 with N2's stored scalar loss.  Preserve that exact
    # behavior even when recomputation differs by one ULP.
    current_loss = float(n2["loss"])
    n2_record["n3_recomputed_start_loss"] = recomputed_start_loss
    n2_record["n3_seed_loss_ulp_difference"] = loss_ulp_decrease(
        max(current_loss, recomputed_start_loss),
        min(current_loss, recomputed_start_loss))

    compact = []
    snapshots = []
    seen_theta = {}
    topology_first_seen = {}
    topology_changes = 0
    previous_topology = None
    stop_reason = None
    started = time.monotonic()
    accepted_steps = 0
    total_backtracks = 0
    while accepted_steps < MAX_DIAGNOSTIC_STEPS and (
            time.monotonic() - started) < MAX_DIAGNOSTIC_SECONDS:
        theta_np = theta.detach().numpy().copy()
        loss_tensor, gradient, hessian = derivatives(theta, torch.empty(0))
        derivative_loss = float(loss_tensor)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        unconstrained, condition, damping, route = unconstrained_direction_fn(
            gradient, hessian)
        direction, qp = dimension_general_box_qp_direction(
            theta, gradient, hessian, lower, upper, unconstrained_direction_fn)
        projected = active.projected_gradient(theta, gradient, lower, upper)
        active_lower = torch.nonzero((theta == lower) & (gradient >= 0.0)).flatten().tolist()
        active_upper = torch.nonzero((theta == upper) & (gradient <= 0.0)).flatten().tolist()
        topo = topology(trajectory, theta_np)
        if previous_topology is not None and topo["sha256"] != previous_topology:
            topology_changes += 1
        previous_topology = topo["sha256"]
        topology_first_seen.setdefault(topo["sha256"], accepted_steps)
        theta_key = theta_np.tobytes()
        repeated_from = seen_theta.get(theta_key)
        seen_theta.setdefault(theta_key, accepted_steps)

        base = {
            "accepted_step_before": accepted_steps, "theta": theta_np.tolist(),
            "loss": current_loss, "teacher_distance": float(np.linalg.norm(theta_np - teacher)),
            "derivative_recomputed_loss": derivative_loss,
            "gradient": gradient.tolist(),
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "projected_gradient": projected.tolist(),
            "projected_gradient_norm": float(torch.linalg.vector_norm(projected)),
            "hessian_eigenvalues": eigenvalues.tolist(),
            "hessian_condition": float(condition), "damping": float(damping),
            "unconstrained_direction": None if unconstrained is None else unconstrained.tolist(),
            "unconstrained_direction_norm": None if unconstrained is None else float(
                torch.linalg.vector_norm(unconstrained)),
            "box_qp_direction": None if direction is None else direction.tolist(),
            "box_qp_direction_norm": None if direction is None else float(
                torch.linalg.vector_norm(direction)),
            "box_qp": qp, "active_lower": active_lower, "active_upper": active_upper,
            "topology": topo, "exact_theta_repeat_from": repeated_from,
            "accepted": False, "accepted_backtrack": None, "accepted_alpha": None,
            "loss_decrease": None, "loss_decrease_ulps": None,
            "actual_displacement": None, "actual_displacement_norm": None,
            "actual_displacement_local_ulps": None,
        }
        if not math.isfinite(base["gradient_norm"]):
            stop_reason = "nonfinite_gradient"; compact.append(base); break
        if base["gradient_norm"] == 0.0:
            stop_reason = "floating_point_zero_gradient"; compact.append(base); break
        if direction is None:
            stop_reason = qp["route_status"]; compact.append(base); break
        directional = float(torch.dot(gradient, direction))
        base["directional_derivative"] = directional
        accepted = False; unchanged = False
        for backtrack in range(polish.MAX_BACKTRACKS + 1):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            if bool(torch.equal(candidate, theta)):
                unchanged = True; break
            if not bool(torch.all(candidate >= lower) and torch.all(candidate <= upper)):
                continue
            with torch.no_grad():
                candidate_loss = float(torch_objective(candidate))
            if candidate_loss < current_loss:
                old = theta_np; new = candidate.detach().numpy().copy()
                delta = new - old
                spacing = np.abs(np.spacing(old))
                base.update(
                    accepted=True, accepted_backtrack=backtrack,
                    accepted_alpha=alpha, loss_decrease=current_loss - candidate_loss,
                    loss_decrease_ulps=loss_ulp_decrease(current_loss, candidate_loss),
                    actual_displacement=delta.tolist(),
                    actual_displacement_norm=float(np.linalg.norm(delta)),
                    actual_displacement_local_ulps=np.divide(
                        delta, spacing, out=np.zeros_like(delta), where=spacing != 0.0).tolist(),
                )
                theta = candidate.detach(); current_loss = candidate_loss
                accepted_steps += 1; total_backtracks += backtrack; accepted = True
                break
        compact.append({
            "accepted_step_before": base["accepted_step_before"],
            "loss": base["loss"], "teacher_distance": base["teacher_distance"],
            "gradient_norm": base["gradient_norm"],
            "projected_gradient_norm": base["projected_gradient_norm"],
            "minimum_hessian_eigenvalue": min(base["hessian_eigenvalues"]),
            "maximum_hessian_eigenvalue": max(base["hessian_eigenvalues"]),
            "hessian_condition": base["hessian_condition"], "damping": base["damping"],
            "unconstrained_direction_norm": base["unconstrained_direction_norm"],
            "box_qp_direction_norm": base["box_qp_direction_norm"],
            "active_lower": active_lower, "active_upper": active_upper,
            "face": qp["winning_face_status"], "topology_sha256": topo["sha256"],
            "exact_theta_repeat_from": repeated_from, "accepted": base["accepted"],
            "accepted_backtrack": base["accepted_backtrack"],
            "accepted_alpha": base["accepted_alpha"],
            "loss_decrease": base["loss_decrease"],
            "loss_decrease_ulps": base["loss_decrease_ulps"],
            "actual_displacement_norm": base["actual_displacement_norm"],
            "maximum_absolute_displacement_ulps": None if base["actual_displacement_local_ulps"] is None
                else float(max(abs(value) for value in base["actual_displacement_local_ulps"])),
        })
        if base["accepted_step_before"] in SNAPSHOT_STEPS:
            base["rollout"] = rollout(trajectory, theta_np)
            snapshots.append(base)
            atomic_json(OUT / "progress.json", {
                "accepted_steps": accepted_steps, "current_theta": theta.tolist(),
                "current_loss": current_loss, "snapshots": snapshots,
                "elapsed_seconds": time.monotonic() - started,
            })
        if not accepted:
            predicted = max(0.0, -directional)
            spacing = abs(float(np.spacing(np.float64(current_loss))))
            stop_reason = ("floating_point_no_further_improvement"
                           if unchanged or predicted <= 32.0 * spacing
                           else "backtracking_no_improvement")
            break

    if stop_reason is None:
        stop_reason = ("diagnostic_step_limit" if accepted_steps >= MAX_DIAGNOSTIC_STEPS
                       else "diagnostic_wall_limit")
    final_np = theta.numpy().copy()
    if not snapshots or snapshots[-1]["accepted_step_before"] != accepted_steps:
        snapshots.append({
            "accepted_step_before": accepted_steps, "theta": final_np.tolist(),
            "loss": current_loss, "teacher_distance": float(np.linalg.norm(final_np - teacher)),
            "topology": topology(trajectory, final_np), "rollout": rollout(trajectory, final_np),
        })
    report = {
        "diagnostic_only": True, "no_label_assigned": True,
        "collection_limits": {"accepted_steps": MAX_DIAGNOSTIC_STEPS,
                              "wall_seconds": MAX_DIAGNOSTIC_SECONDS},
        "replay": replay, "n2": n2_record,
        "n3": {
            "accepted_steps_observed": accepted_steps, "collection_stop": stop_reason,
            "elapsed_seconds": time.monotonic() - started,
            "final_theta": final_np.tolist(), "final_loss": current_loss,
            "final_teacher_distance": float(np.linalg.norm(final_np - teacher)),
            "final_rollout": rollout(trajectory, final_np),
            "exact_theta_repeats": sum(row["exact_theta_repeat_from"] is not None for row in compact),
            "unique_event_topologies": len(topology_first_seen),
            "event_topology_changes": topology_changes,
            "topology_first_seen": topology_first_seen,
            "total_backtracks_for_accepted_steps": total_backtracks,
            "compact_history": compact, "snapshots": snapshots,
        },
        "normal_control": json.loads((ROOT / "query_cache/reflection_ball/case_00187/"
                                       "n00001_i00001.json").read_text()),
        "checkpoint_hashes_unchanged": before_hashes == hashes(),
    }
    atomic_json(OUT / "report.json", report)
    print(json.dumps({
        "replay": replay, "n2": n2_record,
        "accepted_steps_observed": accepted_steps, "collection_stop": stop_reason,
        "final_loss": current_loss, "final_teacher_distance": report["n3"]["final_teacher_distance"],
        "event_topology_changes": topology_changes,
        "unique_event_topologies": len(topology_first_seen),
        "exact_theta_repeats": report["n3"]["exact_theta_repeats"],
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
