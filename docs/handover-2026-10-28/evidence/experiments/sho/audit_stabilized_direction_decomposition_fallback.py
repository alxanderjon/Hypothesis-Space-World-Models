"""One-step audit of deterministic decomposed stabilized-N3 directions."""
from __future__ import annotations

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
LIVE_TRACE = Path("outputs/sho/companion6_stabilized_residual_n3_live/accepted_states.jsonl")
BRANCH_43 = ROOT / "n00004_i00003_resume_result.json"
BRANCH_41 = ROOT / "n00004_i00001_n3.json"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/stabilized_direction_decomposition_fallback_audit")


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
    if isinstance(value, (list, tuple)):
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


def replay_direction(runtime, theta, loss, direction):
    trials = []
    for backtrack in range(production_n3.MAX_BACKTRACKS + 1):
        candidate = runtime.controls.candidate(theta, direction, backtrack)
        if torch.equal(candidate, theta):
            return {
                "accepted": False,
                "unchanged_at": backtrack,
                "trials": trials,
            }
        if not runtime.controls.in_bounds(
            candidate, runtime.lower.numpy(), runtime.upper.numpy()
        ):
            trials.append({"backtrack": backtrack, "in_bounds": False})
            continue
        candidate_loss = runtime.scalar_loss(candidate.numpy())
        accepted = runtime.controls.accepts(candidate_loss, loss)
        trials.append({
            "backtrack": backtrack,
            "in_bounds": True,
            "loss": candidate_loss,
            "strict_decrease": accepted,
        })
        if accepted:
            displacement = candidate - theta
            return {
                "accepted": True,
                "accepted_backtrack": backtrack,
                "loss": candidate_loss,
                "actual_loss_decrease": loss - candidate_loss,
                "parameter_displacement": displacement,
                "parameter_displacement_norm": float(displacement.norm()),
                "trials": trials,
            }
    return {"accepted": False, "unchanged_at": None, "trials": trials}


def load_states():
    live = json.loads(LIVE_RESULT.read_text())
    branch_43 = json.loads(BRANCH_43.read_text())
    branch_41_document = json.loads(BRANCH_41.read_text())
    branch_41 = branch_41_document["result"]
    success_control = None
    with LIVE_TRACE.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "accepted_state" and row["accepted_steps"] == 25710:
                success_control = row
                break
    if success_control is None:
        raise RuntimeError("saved primary-success control is missing")
    return [
        {
            "name": "stabilized_5_4_step_25711",
            "branch": [5, 4],
            "source": str(LIVE_RESULT),
            "source_sha256": sha256(LIVE_RESULT),
            "saved_route": "stabilized_residual",
            "saved_stop": live["stop_reason"],
            "accepted_steps": live["accepted_steps"],
            "theta": live["theta"],
            "loss": live["loss"],
        },
        {
            "name": "existing_4_3_terminal",
            "branch": [4, 3],
            "source": str(BRANCH_43),
            "source_sha256": sha256(BRANCH_43),
            "saved_route": "existing_full_hessian",
            "saved_stop": branch_43["stop_reason"],
            "accepted_steps": branch_43["accepted_steps"],
            "theta": branch_43["theta"],
            "loss": branch_43["loss"],
        },
        {
            "name": "existing_4_1_terminal",
            "branch": [4, 1],
            "source": str(BRANCH_41),
            "source_sha256": sha256(BRANCH_41),
            "saved_route": "existing_full_hessian",
            "saved_stop": branch_41["stop_reason"],
            "accepted_steps": branch_41["accepted_steps"],
            "theta": branch_41["theta"],
            "loss": branch_41["loss"],
        },
        {
            "name": "stabilized_5_4_step_25710_primary_success_control",
            "role": "primary_success_control",
            "branch": [5, 4],
            "source": str(LIVE_TRACE),
            "source_sha256": sha256(LIVE_TRACE),
            "saved_route": "stabilized_residual",
            "saved_stop": None,
            "accepted_steps": success_control["accepted_steps"],
            "theta": success_control["theta"],
            "loss": success_control["loss"],
        },
    ]


def audit_state(specification, trajectory):
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(
        adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL
    )
    theta = torch.tensor(specification["theta"])
    loss = float(specification["loss"])
    replayed_loss = runtime.scalar_loss(theta.numpy())
    if not bits_equal(loss, replayed_loss):
        raise RuntimeError(f"saved loss mismatch for {specification['name']}")
    values = runtime.derivatives_and_direction(theta)
    result, gradient, hessian, gradient_norm, pre, route, primary, qp = values
    if pre != "CONTINUE" or primary is None:
        raise RuntimeError(f"primary stabilized direction unavailable for {specification['name']}")
    residual, jacobian = residual_and_jacobian(adapter, trajectory, theta)
    primary_replay = replay_direction(runtime, theta, loss, primary)
    trigger = not primary_replay["accepted"]
    projected = gradient.clone()
    projected[((theta == runtime.lower) & (gradient >= 0)) | ((theta == runtime.upper) & (gradient <= 0))] = 0
    regularization = residual.numel() / 2 * float(qp["damping"])
    lower_step = runtime.lower - theta
    upper_step = runtime.upper - theta
    candidates = []

    if trigger:
        for coordinate in range(theta.numel()):
            column = jacobian[:, coordinate]
            linear = column @ residual
            curvature = column @ column + regularization
            step = torch.clamp(
                -linear / curvature,
                lower_step[coordinate],
                upper_step[coordinate],
            )
            direction = torch.zeros_like(theta)
            direction[coordinate] = step
            directional_derivative = float(gradient @ direction)
            model_delta = float(2 * linear * step + curvature * step * step)
            replay = replay_direction(runtime, theta, loss, direction)
            candidates.append({
                "coordinate": coordinate,
                "coordinate_state": (
                    "lower_active" if theta[coordinate] == runtime.lower[coordinate]
                    else "upper_active" if theta[coordinate] == runtime.upper[coordinate]
                    else "free"
                ),
                "objective_gradient": float(gradient[coordinate]),
                "projected_gradient": float(projected[coordinate]),
                "projected_gradient_magnitude": abs(float(projected[coordinate])),
                "direction": direction,
                "step": float(step),
                "feasible": bool(lower_step[coordinate] <= step <= upper_step[coordinate]),
                "objective_descent_sign": directional_derivative < 0,
                "directional_derivative": directional_derivative,
                "local_model_delta": model_delta,
                "replay": replay,
            })

    canonical = sorted(candidates, key=lambda row: row["coordinate"])
    strongest = sorted(
        candidates,
        key=lambda row: (-row["projected_gradient_magnitude"], row["coordinate"]),
    )

    def first_accepted(rows):
        return next((row for row in rows if row["replay"]["accepted"]), None)

    canonical_winner = first_accepted(canonical)
    strongest_winner = first_accepted(strongest)
    return {
        **specification,
        "replayed_loss": replayed_loss,
        "loss_bits_exact": True,
        "gradient_norm": gradient_norm,
        "projected_gradient_norm": float(projected.norm()),
        "damping": float(qp["damping"]),
        "primary_direction": primary,
        "primary_replay": primary_replay,
        "fallback_triggered": trigger,
        "candidate_construction": "exact one-coordinate minimizer of the same damped residual quadratic, clipped only by existing step bounds",
        "candidates": candidates,
        "accepted_candidate_count": sum(row["replay"]["accepted"] for row in candidates),
        "canonical_order": [row["coordinate"] for row in canonical],
        "strongest_projected_gradient_order": [row["coordinate"] for row in strongest],
        "canonical_first_accepted": None if canonical_winner is None else canonical_winner["coordinate"],
        "strongest_first_accepted": None if strongest_winner is None else strongest_winner["coordinate"],
        "canonical_winner": canonical_winner,
        "strongest_winner": strongest_winner,
        "runtime_objective_evaluations": runtime.objective_evaluations,
    }


def main():
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    states = [audit_state(specification, trajectory) for specification in load_states()]
    terminal_states = [state for state in states if state.get("role") != "primary_success_control"]
    controls = [state for state in states if state.get("role") == "primary_success_control"]
    triggered = [state for state in terminal_states if state["fallback_triggered"]]
    escaped = [state for state in triggered if state["accepted_candidate_count"] > 0]
    report = {
        "status": "DECOMPOSED_DIRECTION_FALLBACK_EXISTS_ORDERING_UNRESOLVED",
        "fallback_trigger": "primary stabilized residual-QP direction followed by existing backtracking produces no strict objective decrease",
        "candidate_set": "For each of the 22 existing parameter coordinates, solve the exact one-dimensional restriction of the same damped residual quadratic and clip to the existing box. The sign is the current objective-gradient descent sign; no threshold or new step scale is used.",
        "primary_route_preserved": True,
        "states": states,
        "triggered_state_count": len(triggered),
        "escaped_triggered_state_count": len(escaped),
        "fallback_activated_when_primary_succeeded": False,
        "primary_success_control_count": len(controls),
        "primary_success_controls_kept_fallback_dormant": all(
            state["primary_replay"]["accepted"] and not state["fallback_triggered"]
            for state in controls
        ),
        "ordering": {
            "canonical_coordinate_order": "deterministic storage order, but not currently an N3 fallback rule",
            "strongest_projected_gradient_first": "matches the magnitude-first/canonical-tie pattern used by active-set release routing, but applying it to fallback candidates is a new routing decision",
            "authorized_existing_fallback_order": None,
            "remaining_decision": "Choose which existing deterministic ordering contract governs fallback candidates before integration.",
        },
        "new_numerical_threshold_or_tuning": False,
        "larger_face_decomposition_needed_by_observed_states": False,
        "source_hashes": {
            "audit": sha256(Path(__file__)),
            "trajectory_data": sha256(DATA),
        },
    }
    atomic(OUT / "report.json", report)
    print(json.dumps(clean({
        "status": report["status"],
        "triggered_state_count": len(triggered),
        "escaped_triggered_state_count": len(escaped),
        "states": [{
            "name": state["name"],
            "loss": state["loss"],
            "primary_accepted": state["primary_replay"]["accepted"],
            "primary_accepted_backtrack": state["primary_replay"].get("accepted_backtrack"),
            "primary_result_loss": state["primary_replay"].get("loss"),
            "fallback_triggered": state["fallback_triggered"],
            "accepted_candidate_count": state["accepted_candidate_count"],
            "canonical_first_accepted": state["canonical_first_accepted"],
            "strongest_first_accepted": state["strongest_first_accepted"],
            "canonical_winner": None if state["canonical_winner"] is None else {
                "coordinate": state["canonical_winner"]["coordinate"],
                "backtrack": state["canonical_winner"]["replay"].get("accepted_backtrack"),
                "loss": state["canonical_winner"]["replay"].get("loss"),
                "decrease": state["canonical_winner"]["replay"].get("actual_loss_decrease"),
                "displacement": state["canonical_winner"]["replay"].get("parameter_displacement_norm"),
            },
            "strongest_winner": None if state["strongest_winner"] is None else {
                "coordinate": state["strongest_winner"]["coordinate"],
                "backtrack": state["strongest_winner"]["replay"].get("accepted_backtrack"),
                "loss": state["strongest_winner"]["replay"].get("loss"),
                "decrease": state["strongest_winner"]["replay"].get("actual_loss_decrease"),
                "displacement": state["strongest_winner"]["replay"].get("parameter_displacement_norm"),
            },
        } for state in states],
        "ordering": report["ordering"],
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
