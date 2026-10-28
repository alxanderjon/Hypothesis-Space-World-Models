"""Isolated short continuations using canonical coordinate fallback.

This diagnostic does not alter the production or opt-in N3 implementations.
The primary stabilized-residual route is always attempted first.  Coordinate
directions are considered only when its unchanged backtracking fails.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho import run_stage1b_n3_gradient_polishing as production_n3
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import (
    STABILIZED_RESIDUAL,
    OptInResidualN3Runtime,
    residual_and_jacobian,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/generic_update_network_targeted_recovery/companion_6")
LIVE_RESULT = Path("outputs/sho/companion6_stabilized_residual_n3_live/result.json")
BRANCH_43 = ROOT / "n00004_i00003_resume_result.json"
BRANCH_41 = ROOT / "n00004_i00001_n3.json"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/coordinate_fallback_short_continuations")
HANDOFF = 1.0e-20
MAX_ACCEPTED = 200


def bits_equal(left, right):
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def clean(value):
    if isinstance(value, torch.Tensor):
        return clean(value.detach().numpy())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    return value


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_specs():
    live = json.loads(LIVE_RESULT.read_text())
    b43 = json.loads(BRANCH_43.read_text())
    b41_document = json.loads(BRANCH_41.read_text())
    b41 = b41_document["result"]
    return [
        {
            "name": "stabilized_5_4_step_25711",
            "source": str(LIVE_RESULT),
            "source_sha256": sha256(LIVE_RESULT),
            "theta": live["theta"],
            "loss": live["loss"],
            "saved_stop": live["stop_reason"],
            "saved_active_bounds": live["active_bounds"],
            "starting_stagnation_count": live["stagnation_count"],
        },
        {
            "name": "existing_4_3_terminal",
            "source": str(BRANCH_43),
            "source_sha256": sha256(BRANCH_43),
            "theta": b43["theta"],
            "loss": b43["loss"],
            "saved_stop": b43["stop_reason"],
            "saved_active_bounds": None,
            "starting_stagnation_count": b43["stagnation_count"],
        },
        {
            "name": "existing_4_1_terminal",
            "source": str(BRANCH_41),
            "source_sha256": sha256(BRANCH_41),
            "theta": b41["theta"],
            "loss": b41["loss"],
            "saved_stop": b41["stop_reason"],
            "saved_active_bounds": None,
            # This legacy artifact predates persistence of this counter.  The
            # isolated continuation begins with no consecutive accepted tail.
            "starting_stagnation_count": 0,
            "stagnation_counter_note": "legacy source did not persist the counter; isolated continuation starts at zero",
        },
    ]


def backtrack(runtime, theta, loss, direction):
    trials = 0
    rejections = 0
    unchanged = False
    for backtrack_index in range(production_n3.MAX_BACKTRACKS + 1):
        candidate = runtime.controls.candidate(theta, direction, backtrack_index)
        if torch.equal(candidate, theta):
            unchanged = True
            break
        if not runtime.controls.in_bounds(candidate, runtime.lower.numpy(), runtime.upper.numpy()):
            trials += 1
            rejections += 1
            continue
        candidate_loss = runtime.scalar_loss(candidate.numpy())
        trials += 1
        if runtime.controls.accepts(candidate_loss, loss):
            displacement = candidate - theta
            return {
                "accepted": True,
                "candidate": candidate.detach(),
                "loss": candidate_loss,
                "backtrack": backtrack_index,
                "trials": trials,
                "rejections": rejections,
                "displacement_norm": float(displacement.norm()),
                "actual_decrease": loss - candidate_loss,
                "unchanged": False,
            }
        rejections += 1
    return {
        "accepted": False,
        "trials": trials,
        "rejections": rejections,
        "unchanged": unchanged,
    }


def reconstructed_state(runtime, adapter, trajectory, theta, loss):
    replay_loss = runtime.scalar_loss(theta.numpy())
    values = runtime.derivatives_and_direction(theta)
    result, gradient, hessian, gradient_norm, pre, route, direction, qp = values
    residual, jacobian = residual_and_jacobian(adapter, trajectory, theta)
    if not bits_equal(loss, replay_loss) or not bits_equal(loss, result.loss):
        raise RuntimeError("saved state loss did not replay exactly")
    projected = gradient.clone()
    projected[((theta == runtime.lower) & (gradient >= 0)) | ((theta == runtime.upper) & (gradient <= 0))] = 0
    return {
        "loss": replay_loss,
        "residual": residual,
        "jacobian": jacobian,
        "gradient": gradient,
        "hessian": hessian,
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": float(projected.norm()),
        "pre": pre,
        "route": route,
        "direction": direction,
        "qp": qp,
        "active_bounds": torch.where((theta == runtime.lower) | (theta == runtime.upper))[0],
    }


def run_one(spec, trajectory):
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
    theta = torch.tensor(spec["theta"])
    loss = float(spec["loss"])
    start = reconstructed_state(runtime, adapter, trajectory, theta, loss)
    if spec["saved_active_bounds"] is not None and start["active_bounds"].tolist() != spec["saved_active_bounds"]:
        raise RuntimeError(f"saved active bounds mismatch for {spec['name']}")
    start_verification = {
        "theta_matches_source": bits_equal(theta, spec["theta"]),
        "loss_bits_exact": True,
        "residual_norm": float(start["residual"].norm()),
        "jacobian_shape": list(start["jacobian"].shape),
        "gradient": start["gradient"],
        "hessian": start["hessian"],
        "damping": float(start["qp"]["damping"]),
        "bounds": [runtime.lower, runtime.upper],
        "active_bounds": start["active_bounds"],
        "saved_active_bounds_match": spec["saved_active_bounds"] is None or start["active_bounds"].tolist() == spec["saved_active_bounds"],
    }
    # The reconstruction above is validation only; continuation counters begin
    # after it so costs describe the isolated route itself.
    runtime.objective_evaluations = 0
    accepted = 0
    primary_steps = 0
    fallback_steps = 0
    fallback_attempts = 0
    handbacks = 0
    previous_was_fallback = False
    path_length = 0.0
    backtracking_rejections = 0
    stagnation = int(spec["starting_stagnation_count"])
    routes = Counter()
    coordinate_counts = Counter()
    records = []
    stop_reason = None

    while accepted < MAX_ACCEPTED:
        if loss < HANDOFF:
            stop_reason = "HANDOFF_THRESHOLD_REACHED"
            break
        result, gradient, hessian, gradient_norm, pre, route, primary, qp = runtime.derivatives_and_direction(theta)
        if not bits_equal(result.loss, loss):
            raise RuntimeError(f"derivative loss mismatch for {spec['name']}")
        if pre != "CONTINUE":
            stop_reason = pre
            break
        if primary is None:
            stop_reason = qp["route_status"]
            break
        routes[route["route"]] += 1
        projected = gradient.clone()
        projected[((theta == runtime.lower) & (gradient >= 0)) | ((theta == runtime.upper) & (gradient <= 0))] = 0
        primary_predicted = -float(qp["quadratic_value"])
        primary_replay = backtrack(runtime, theta, loss, primary)
        backtracking_rejections += primary_replay["rejections"]
        chosen = primary_replay if primary_replay["accepted"] else None
        chosen_route = "primary" if chosen is not None else None
        coordinate = None
        predicted = primary_predicted
        residual = runtime.last_residual_state["residual"]
        jacobian = runtime.last_residual_state["jacobian"]
        regularization = float(runtime.last_residual_state["regularization"])

        if chosen is None:
            fallback_attempts += 1
            lower_step = runtime.lower - theta
            upper_step = runtime.upper - theta
            for j in range(theta.numel()):
                column = jacobian[:, j]
                linear = column @ residual
                curvature = column @ column + regularization
                step = torch.clamp(-linear / curvature, lower_step[j], upper_step[j])
                candidate_direction = torch.zeros_like(theta)
                candidate_direction[j] = step
                candidate_replay = backtrack(runtime, theta, loss, candidate_direction)
                backtracking_rejections += candidate_replay["rejections"]
                if candidate_replay["accepted"]:
                    chosen = candidate_replay
                    chosen_route = "fallback"
                    coordinate = j
                    model_delta = 2 * linear * step + curvature * step * step
                    predicted = -float(model_delta / residual.numel())
                    break

        if chosen is None:
            dot = float(gradient @ primary)
            numerical, _, _ = runtime.controls.no_further_improvement(
                bool(primary_replay["unchanged"]), dot, loss
            )
            stop_reason = "floating_point_no_further_improvement" if numerical else "backtracking_no_improvement"
            break

        old_loss = loss
        old_theta = theta
        theta = chosen["candidate"]
        loss = float(chosen["loss"])
        accepted += 1
        path_length += chosen["displacement_norm"]
        stagnation = production_n3._updated_float64_stagnation_count(
            stagnation, old_loss, loss, predicted
        )
        active_after = torch.where((theta == runtime.lower) | (theta == runtime.upper))[0]
        if chosen_route == "primary":
            primary_steps += 1
            if previous_was_fallback:
                handbacks += 1
            previous_was_fallback = False
        else:
            fallback_steps += 1
            coordinate_counts[coordinate] += 1
            previous_was_fallback = True
        records.append({
            "accepted_step": accepted,
            "route": chosen_route,
            "fallback_coordinate": coordinate,
            "primary_backtracking_failed": not primary_replay["accepted"],
            "backtrack": chosen["backtrack"],
            "loss_before": old_loss,
            "loss_after": loss,
            "actual_decrease": chosen["actual_decrease"],
            "predicted_decrease": predicted,
            "parameter_displacement": theta - old_theta,
            "parameter_displacement_norm": chosen["displacement_norm"],
            "gradient_norm": gradient_norm,
            "projected_gradient_norm": float(projected.norm()),
            "active_bounds_before": torch.where((old_theta == runtime.lower) | (old_theta == runtime.upper))[0],
            "active_bounds_after": active_after,
            "stagnation_count": stagnation,
            "damping": float(qp["damping"]),
            "primary_backtracking_trials": primary_replay["trials"],
        })
        if accepted <= 5 or accepted % 25 == 0:
            print(f"{spec['name']} step={accepted} route={chosen_route} coord={coordinate} loss={loss:.17g} stagnation={stagnation}", flush=True)
        if stagnation >= production_n3.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE:
            stop_reason = production_n3.FLOAT64_STAGNATION_STOP_REASON
            break

    if stop_reason is None:
        stop_reason = "DIAGNOSTIC_ACCEPTED_STEP_LIMIT"
    return {
        **spec,
        "start_verification": start_verification,
        "screen_limit": MAX_ACCEPTED,
        "accepted_steps": accepted,
        "primary_steps": primary_steps,
        "fallback_steps": fallback_steps,
        "fallback_attempts": fallback_attempts,
        "fallback_coordinates": {str(k): v for k, v in sorted(coordinate_counts.items())},
        "fallback_to_primary_handbacks": handbacks,
        "final_loss": loss,
        "total_loss_reduction": float(spec["loss"]) - loss,
        "parameter_path_length": path_length,
        "backtracking_rejections": backtracking_rejections,
        "objective_evaluations": runtime.objective_evaluations,
        "final_active_bounds": torch.where((theta == runtime.lower) | (theta == runtime.upper))[0],
        "final_stagnation_count": stagnation,
        "stop_reason": stop_reason,
        "handoff": bool(loss < HANDOFF),
        "route_counts": dict(routes),
        "records": records,
    }


def main():
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    results = []
    for spec in load_specs():
        result = run_one(spec, trajectory)
        results.append(result)
        atomic(OUT / f"{spec['name']}.json", result)
    material = []
    for result in results:
        relative = result["total_loss_reduction"] / result["loss"]
        material.append(relative)
    supported = all(result["fallback_to_primary_handbacks"] > 0 for result in results) and all(
        reduction > 0.01 for reduction in material
    )
    status = (
        "COORDINATE_FALLBACK_TRAJECTORY_SUPPORTED"
        if supported
        else "COORDINATE_FALLBACK_ESCAPES_STOP_BUT_DOES_NOT_SOLVE_RECOVERY"
    )
    report = {
        "status": status,
        "contract": "primary stabilized residual-QP first; only after no strict decrease test one-coordinate exact residual-QP restrictions in canonical order 0..21 and accept the first strict decrease",
        "results": results,
        "source_hashes": {
            "script": sha256(Path(__file__)),
            "trajectory_data": sha256(DATA),
        },
    }
    atomic(OUT / "report.json", report)
    print(json.dumps(clean({
        "status": status,
        "results": [{
            "name": r["name"],
            "accepted_steps": r["accepted_steps"],
            "primary_steps": r["primary_steps"],
            "fallback_steps": r["fallback_steps"],
            "fallback_coordinates": r["fallback_coordinates"],
            "final_loss": r["final_loss"],
            "total_loss_reduction": r["total_loss_reduction"],
            "parameter_path_length": r["parameter_path_length"],
            "fallback_to_primary_handbacks": r["fallback_to_primary_handbacks"],
            "stop_reason": r["stop_reason"],
            "handoff": r["handoff"],
        } for r in results],
    }), indent=2))


if __name__ == "__main__":
    main()
