"""One-step audit of structurally implied multi-coordinate terminal escapes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho import run_stage1b_n3_gradient_polishing as production_n3
from experiments.sho.audit_companion6_stabilized_terminal_bounds import traced_bvls
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import (
    STABILIZED_RESIDUAL,
    OptInResidualN3Runtime,
    _bounded_residual_least_squares,
    residual_and_jacobian,
)


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SOURCE = Path("outputs/sho/coordinate_fallback_short_continuations")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/structured_multicoordinate_terminal_escape_audit")
FILES = (
    "stabilized_5_4_step_25711.json",
    "existing_4_3_terminal.json",
    "existing_4_1_terminal.json",
)


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


def reconstruct_terminal(document):
    theta = torch.tensor(document["theta"])
    for row in document["records"]:
        theta = theta + torch.tensor(row["parameter_displacement"])
    return theta, float(document["final_loss"])


def backtrack(runtime, theta, loss, direction):
    evaluated = []
    for backtrack_index in range(production_n3.MAX_BACKTRACKS + 1):
        candidate = runtime.controls.candidate(theta, direction, backtrack_index)
        if torch.equal(candidate, theta):
            return {
                "accepted": False,
                "unchanged_at": backtrack_index,
                "evaluated": evaluated,
            }
        if not runtime.controls.in_bounds(candidate, runtime.lower.numpy(), runtime.upper.numpy()):
            evaluated.append({"backtrack": backtrack_index, "in_bounds": False})
            continue
        candidate_loss = runtime.scalar_loss(candidate.numpy())
        accepted = runtime.controls.accepts(candidate_loss, loss)
        evaluated.append({
            "backtrack": backtrack_index,
            "in_bounds": True,
            "loss": candidate_loss,
            "strict_decrease": accepted,
        })
        if accepted:
            displacement = candidate - theta
            return {
                "accepted": True,
                "backtrack": backtrack_index,
                "loss": candidate_loss,
                "actual_decrease": loss - candidate_loss,
                "parameter_displacement": displacement,
                "parameter_displacement_norm": float(displacement.norm()),
                "evaluated": evaluated,
            }
    return {"accepted": False, "unchanged_at": None, "evaluated": evaluated}


def restricted_direction(matrix, residual, theta, lower, upper, support):
    support_tensor = torch.tensor(support, dtype=torch.int64)
    direction = torch.zeros_like(theta)
    if len(support):
        restricted, metadata = _bounded_residual_least_squares(
            matrix[:, support_tensor],
            residual,
            theta[support_tensor],
            lower[support_tensor],
            upper[support_tensor],
        )
        direction[support_tensor] = restricted
    else:
        metadata = {"iterations": 0, "active": [], "free_model_gradient_norm": 0.0}
    return direction, metadata


def candidate_result(runtime, theta, loss, matrix, augmented_residual, name, reason, support):
    direction, metadata = restricted_direction(
        matrix, augmented_residual, theta, runtime.lower, runtime.upper, support
    )
    replay = backtrack(runtime, theta, loss, direction)
    model_before = float(augmented_residual @ augmented_residual)
    modeled = augmented_residual + matrix @ direction
    model_after = float(modeled @ modeled)
    return {
        "name": name,
        "structural_reason": reason,
        "support": support,
        "support_size": len(support),
        "direction": direction,
        "direction_norm": float(direction.norm()),
        "restricted_solver": metadata,
        "model_predicted_decrease_unscaled": model_before - model_after,
        "replay": replay,
    }


def audit_one(path, trajectory):
    document = json.loads(path.read_text())
    theta, loss = reconstruct_terminal(document)
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
    replayed_loss = runtime.scalar_loss(theta.numpy())
    if not bits_equal(loss, replayed_loss):
        raise RuntimeError(f"terminal reconstruction mismatch for {document['name']}")
    result, gradient, hessian, gradient_norm, pre, route, primary, qp = runtime.derivatives_and_direction(theta)
    if not bits_equal(result.loss, loss) or pre != "CONTINUE" or primary is None:
        raise RuntimeError(f"primary reconstruction failed for {document['name']}")
    residual, jacobian = residual_and_jacobian(adapter, trajectory, theta)
    mu = float(qp["damping"])
    regularization = residual.numel() / 2 * mu
    matrix = torch.cat((
        jacobian,
        torch.sqrt(torch.tensor(regularization)) * torch.eye(theta.numel()),
    ))
    augmented_residual = torch.cat((residual, torch.zeros_like(theta)))
    traced_direction, status, initial_model_gradient, final_model_gradient, events = traced_bvls(
        matrix, augmented_residual, theta, runtime.lower, runtime.upper
    )
    if not bits_equal(traced_direction, primary):
        raise RuntimeError(f"traced primary mismatch for {document['name']}")
    active = torch.where((theta == runtime.lower) | (theta == runtime.upper))[0].tolist()
    free = [coordinate for coordinate in range(theta.numel()) if coordinate not in active]
    objective_violations = []
    for coordinate in active:
        at_lower = bool(theta[coordinate] == runtime.lower[coordinate])
        violates = bool(gradient[coordinate] < 0) if at_lower else bool(gradient[coordinate] > 0)
        if violates:
            objective_violations.append(coordinate)

    simultaneous_groups = []
    for event_index, event in enumerate(events):
        if event["event"] == "bound_hit" and len(event["activated"]) > 1:
            group = [row["coordinate"] for row in event["activated"]]
            simultaneous_groups.append({
                "event_index": event_index,
                "active_set_iteration": event["iteration"],
                "coordinates": group,
            })

    candidates = []
    if len(free) >= 2:
        candidates.append(candidate_result(
            runtime, theta, loss, matrix, augmented_residual,
            "current_free_face",
            "all and only coordinates not exactly on a current box face; current bound coordinates remain fixed",
            free,
        ))
    released_face = sorted(set(free + objective_violations))
    if objective_violations and len(released_face) >= 2:
        candidates.append(candidate_result(
            runtime, theta, loss, matrix, augmented_residual,
            "objective_kkt_released_face",
            "current free face plus all and only exact-bound coordinates whose actual-objective gradient has the KKT-violating inward-descent sign",
            released_face,
        ))
    for group in simultaneous_groups:
        candidates.append(candidate_result(
            runtime, theta, loss, matrix, augmented_residual,
            f"simultaneous_activation_event_{group['event_index']}",
            f"coordinates activated simultaneously by existing active-set iteration {group['active_set_iteration']}; event order supplies candidate order",
            group["coordinates"],
        ))

    primary_replay = backtrack(runtime, theta, loss, primary)
    single_replays = []
    for coordinate in range(theta.numel()):
        direction, _ = restricted_direction(
            matrix, augmented_residual, theta, runtime.lower, runtime.upper, [coordinate]
        )
        single_replays.append({
            "coordinate": coordinate,
            "direction_norm": float(direction.norm()),
            "replay": backtrack(runtime, theta, loss, direction),
        })
    accepted_multi = [candidate for candidate in candidates if candidate["replay"]["accepted"]]
    for candidate in candidates:
        candidate["direction_matches_primary_bits"] = bits_equal(candidate["direction"], primary)
        candidate["direction_max_absolute_difference_from_primary"] = float(
            torch.max(torch.abs(candidate["direction"] - primary))
        )
        attempted_losses = [
            row["loss"] for row in candidate["replay"]["evaluated"] if "loss" in row
        ]
        candidate["best_attempted_loss"] = min(attempted_losses) if attempted_losses else None
        candidate["best_attempted_decrease"] = (
            loss - candidate["best_attempted_loss"] if attempted_losses else None
        )
    return {
        "name": document["name"],
        "source": str(path),
        "source_sha256": sha256(path),
        "terminal_reconstruction": {
            "theta": theta,
            "saved_loss": loss,
            "replayed_loss": replayed_loss,
            "loss_bits_exact": True,
            "residual": residual,
            "jacobian": jacobian,
            "gradient": gradient,
            "gradient_norm": gradient_norm,
            "hessian": hessian,
            "damping": mu,
            "bounds": [runtime.lower, runtime.upper],
            "active_coordinates": active,
            "free_coordinates": free,
            "current_stabilized_direction": primary,
        },
        "objective_kkt_violating_active_coordinates": objective_violations,
        "active_set_final_status": status,
        "active_set_initial_model_gradient": initial_model_gradient,
        "active_set_final_model_gradient": final_model_gradient,
        "active_set_events": events,
        "simultaneous_activation_groups": simultaneous_groups,
        "original_coupled_qp": {
            "direction_norm": float(primary.norm()),
            "replay": primary_replay,
        },
        "canonical_single_coordinate_replay": {
            "accepted_count": sum(row["replay"]["accepted"] for row in single_replays),
            "first_accepted": next((row["coordinate"] for row in single_replays if row["replay"]["accepted"]), None),
            "directions": single_replays,
        },
        "structured_candidates": candidates,
        "structured_candidate_count": len(candidates),
        "accepted_structured_candidate_count": len(accepted_multi),
    }


def main():
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    states = [audit_one(SOURCE / filename, trajectory) for filename in FILES]
    accepted = [
        candidate
        for state in states
        for candidate in state["structured_candidates"]
        if candidate["replay"]["accepted"]
    ]
    status = "MULTI_COORDINATE_FACE_ESCAPE_SUPPORTED" if accepted else "NO_MULTI_COORDINATE_ESCAPE"
    report = {
        "status": status,
        "terminal_fallback_line_exhausted": not accepted,
        "materiality_note": "Classification is descriptive relative to the approximately five-to-six-order loss gap to the 1e-20 handoff; the audit does not add an acceptance threshold.",
        "candidate_ordering": "current-free-face first, then the unique all-objective-KKT-violations released face, then simultaneous activation groups in the existing active-set event order; candidates are all audited independently, so ordering does not select or suppress a result",
        "states": states,
        "source_hashes": {
            "script": sha256(Path(__file__)),
            "trajectory_data": sha256(DATA),
        },
    }
    atomic(OUT / "report.json", report)
    print(json.dumps(clean({
        "status": status,
        "terminal_fallback_line_exhausted": not accepted,
        "states": [{
            "name": state["name"],
            "loss": state["terminal_reconstruction"]["saved_loss"],
            "active": state["terminal_reconstruction"]["active_coordinates"],
            "free_count": len(state["terminal_reconstruction"]["free_coordinates"]),
            "objective_violations": state["objective_kkt_violating_active_coordinates"],
            "simultaneous_groups": state["simultaneous_activation_groups"],
            "primary_accepted": state["original_coupled_qp"]["replay"]["accepted"],
            "single_accepted_count": state["canonical_single_coordinate_replay"]["accepted_count"],
            "candidates": [{
                "name": candidate["name"],
                "support": candidate["support"],
                "direction_norm": candidate["direction_norm"],
                "accepted": candidate["replay"]["accepted"],
                "backtrack": candidate["replay"].get("backtrack"),
                "loss": candidate["replay"].get("loss"),
                "decrease": candidate["replay"].get("actual_decrease"),
                "displacement": candidate["replay"].get("parameter_displacement_norm"),
                "direction_matches_primary_bits": candidate["direction_matches_primary_bits"],
                "best_attempted_loss": candidate["best_attempted_loss"],
                "best_attempted_decrease": candidate["best_attempted_decrease"],
            } for candidate in state["structured_candidates"]],
        } for state in states],
    }), indent=2))


if __name__ == "__main__":
    main()
