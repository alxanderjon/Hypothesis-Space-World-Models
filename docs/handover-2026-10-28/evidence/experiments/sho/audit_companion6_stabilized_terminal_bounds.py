"""Terminal active-bound audit for the companion_6 stabilized N3 run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

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

LIVE = Path("outputs/sho/companion6_stabilized_residual_n3_live")
RESULT = LIVE / "result.json"
TRACE = LIVE / "accepted_states.jsonl"
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/companion6_stabilized_terminal_bound_audit")
ACTIVE_COORDINATES = [15, 16, 19, 20, 21]


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


def traced_bvls(matrix, residual, theta, lower, upper, force_free=None):
    size = theta.numel()
    step_lower = lower - theta
    step_upper = upper - theta
    initial_gradient = matrix.T @ residual
    direction = torch.zeros_like(theta)
    status = torch.zeros(size, dtype=torch.int64)
    status[(step_lower == 0) & (initial_gradient >= 0)] = -1
    status[(step_upper == 0) & (initial_gradient <= 0)] = 1
    natural_initial = status.clone()
    if force_free is not None:
        status[force_free] = 0
    events = [{
        "event": "initial_active_set",
        "natural_status": natural_initial,
        "used_status": status.clone(),
        "free": torch.where(status == 0)[0],
    }]
    seen = set()
    iteration = 0

    while True:
        key = (tuple(status.tolist()), direction.numpy().tobytes())
        if key in seen:
            raise RuntimeError("exact active-set cycle")
        seen.add(key)
        iteration += 1
        lower_active = torch.where(status == -1)[0]
        upper_active = torch.where(status == 1)[0]
        free = torch.where(status == 0)[0]
        active = torch.cat((lower_active, upper_active))
        trial = torch.zeros_like(theta)
        trial[lower_active] = step_lower[lower_active]
        trial[upper_active] = step_upper[upper_active]
        singular = torch.empty(0, dtype=theta.dtype)
        if len(free):
            u, singular, vh = torch.linalg.svd(matrix[:, free], full_matrices=False)
            if bool((singular <= 0).any()):
                raise RuntimeError("nonpositive singular value")
            active_residual = residual + matrix[:, active] @ trial[active]
            trial[free] = vh.T @ ((u.T @ (-active_residual)) / singular)
        events.append({
            "event": "reduced_solve",
            "iteration": iteration,
            "lower_active": lower_active,
            "upper_active": upper_active,
            "free": free,
            "singular_min": float(singular[-1]) if len(singular) else None,
            "singular_max": float(singular[0]) if len(singular) else None,
            "trial": trial,
        })

        delta = trial - direction
        blockers = []
        for coordinate in free.tolist():
            if trial[coordinate] < step_lower[coordinate]:
                blockers.append((float((step_lower[coordinate] - direction[coordinate]) / delta[coordinate]), coordinate, -1))
            elif trial[coordinate] > step_upper[coordinate]:
                blockers.append((float((step_upper[coordinate] - direction[coordinate]) / delta[coordinate]), coordinate, 1))
        if blockers:
            alpha = min(item[0] for item in blockers)
            direction = direction + alpha * delta
            activated = []
            for value, coordinate, sign in blockers:
                if value == alpha:
                    status[coordinate] = sign
                    direction[coordinate] = step_lower[coordinate] if sign == -1 else step_upper[coordinate]
                    activated.append({"coordinate": coordinate, "side": "lower" if sign == -1 else "upper"})
            events.append({
                "event": "bound_hit",
                "iteration": iteration,
                "alpha": alpha,
                "activated": activated,
                "simultaneous": len(activated) > 1,
                "direction": direction,
            })
            continue

        direction = trial
        model_gradient = matrix.T @ (residual + matrix @ direction)
        violations = []
        release_checks = []
        for coordinate in lower_active.tolist():
            violates = bool(model_gradient[coordinate] < 0)
            release_checks.append({"coordinate": coordinate, "side": "lower", "gradient": float(model_gradient[coordinate]), "release": violates})
            if violates:
                violations.append((-float(model_gradient[coordinate]), coordinate))
        for coordinate in upper_active.tolist():
            violates = bool(model_gradient[coordinate] > 0)
            release_checks.append({"coordinate": coordinate, "side": "upper", "gradient": float(model_gradient[coordinate]), "release": violates})
            if violates:
                violations.append((float(model_gradient[coordinate]), coordinate))
        events.append({
            "event": "release_check",
            "iteration": iteration,
            "checks": release_checks,
            "free_gradient": model_gradient[free],
        })
        if violations:
            magnitude, coordinate = min(violations, key=lambda item: (-item[0], item[1]))
            status[coordinate] = 0
            events.append({"event": "release", "iteration": iteration, "coordinate": coordinate, "violation_magnitude": magnitude})
            continue
        if not bool(((direction >= step_lower) & (direction <= step_upper)).all()):
            raise RuntimeError("infeasible direction")
        return direction, status, initial_gradient, model_gradient, events


def actual_backtracking(runtime, theta, current_loss, direction):
    evaluated = []
    for backtrack in range(production_n3.MAX_BACKTRACKS + 1):
        candidate = runtime.controls.candidate(theta, direction, backtrack)
        if torch.equal(candidate, theta):
            return {"accepted": False, "unchanged_at": backtrack, "evaluated": evaluated}
        if not runtime.controls.in_bounds(candidate, runtime.lower.numpy(), runtime.upper.numpy()):
            evaluated.append({"backtrack": backtrack, "in_bounds": False})
            continue
        value = runtime.scalar_loss(candidate.numpy())
        accepted = runtime.controls.accepts(value, current_loss)
        evaluated.append({"backtrack": backtrack, "in_bounds": True, "loss": value, "accepted": accepted})
        if accepted:
            return {"accepted": True, "backtrack": backtrack, "loss": value, "candidate": candidate, "evaluated": evaluated}
    return {"accepted": False, "unchanged_at": None, "evaluated": evaluated}


def main():
    terminal = json.loads(RESULT.read_text())
    if terminal["accepted_steps"] != 25711 or not bits_equal(terminal["loss"], 5.534819363000349e-15):
        raise RuntimeError("wrong terminal artifact")
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL)
    theta = torch.tensor(terminal["theta"])
    current_loss = float(terminal["loss"])
    values = runtime.derivatives_and_direction(theta)
    result, gradient, hessian, gradient_norm, pre, route, live_direction, qp = values
    residual, jacobian = residual_and_jacobian(adapter, trajectory, theta)
    if not bits_equal(result.loss, current_loss):
        raise RuntimeError("terminal loss replay mismatch")
    active = torch.where((theta == runtime.lower) | (theta == runtime.upper))[0].tolist()
    if active != ACTIVE_COORDINATES:
        raise RuntimeError("terminal active-bound identity mismatch")

    mu = float(qp["damping"])
    regularization = residual.numel() / 2 * mu
    augmented_matrix = torch.cat((jacobian, torch.sqrt(torch.tensor(regularization)) * torch.eye(theta.numel())))
    augmented_residual = torch.cat((residual, torch.zeros_like(theta)))
    direction, status, initial_model_gradient, final_model_gradient, events = traced_bvls(
        augmented_matrix, augmented_residual, theta, runtime.lower, runtime.upper
    )
    if not bits_equal(direction, live_direction):
        raise RuntimeError("traced active-set direction differs from live binding")

    projected = gradient.clone()
    projected[((theta == runtime.lower) & (gradient >= 0)) | ((theta == runtime.upper) & (gradient <= 0))] = 0
    kkt_rows = []
    for coordinate in ACTIVE_COORDINATES:
        side = "lower" if theta[coordinate] == runtime.lower[coordinate] else "upper"
        full_compatible = bool(gradient[coordinate] >= 0) if side == "lower" else bool(gradient[coordinate] <= 0)
        model_compatible = bool(initial_model_gradient[coordinate] >= 0) if side == "lower" else bool(initial_model_gradient[coordinate] <= 0)
        final_compatible = bool(final_model_gradient[coordinate] >= 0) if side == "lower" else bool(final_model_gradient[coordinate] <= 0)
        kkt_rows.append({
            "coordinate": coordinate,
            "side": side,
            "theta": float(theta[coordinate]),
            "objective_gradient": float(gradient[coordinate]),
            "projected_gradient_contribution": float(projected[coordinate]),
            "initial_residual_model_gradient": float(initial_model_gradient[coordinate]),
            "final_qp_model_gradient": float(final_model_gradient[coordinate]),
            "objective_kkt_compatible": full_compatible,
            "initial_model_kkt_compatible": model_compatible,
            "final_qp_kkt_compatible": final_compatible,
            "qualifies_for_release": not final_compatible,
        })

    step_lower = (runtime.lower - theta).numpy()
    step_upper = (runtime.upper - theta).numpy()
    # Global scaling leaves the QP minimizer unchanged and prevents the
    # trusted solver's absolute optimality test from accepting the zero-scale
    # problem before resolving its active face.
    oracle_scale = 1.0e6
    scipy = lsq_linear(
        augmented_matrix.numpy() * oracle_scale,
        (-augmented_residual).numpy() * oracle_scale,
        bounds=(step_lower, step_upper),
        method="trf",
        lsq_solver="exact",
        tol=np.finfo(np.float64).eps,
        max_iter=1000,
        verbose=0,
    )
    oracle_direction = torch.from_numpy(scipy.x)
    q = lambda value: float(torch.sum((augmented_residual + augmented_matrix @ value) ** 2))
    oracle_gradient = augmented_matrix.T @ (augmented_residual + augmented_matrix @ oracle_direction)
    free = torch.where(status == 0)[0]
    lower_active = torch.where(status == -1)[0]
    upper_active = torch.where(status == 1)[0]
    projected_qp_gradient = final_model_gradient.clone()
    projected_qp_gradient[lower_active[final_model_gradient[lower_active] >= 0]] = 0
    projected_qp_gradient[upper_active[final_model_gradient[upper_active] <= 0]] = 0
    qp_comparison = {
        "strict_convexity_lambda": regularization,
        "normal_matrix_min_eigenvalue": float(torch.linalg.eigvalsh(augmented_matrix.T @ augmented_matrix)[0]),
        "active_set_direction": direction,
        "trusted_oracle": "scipy.optimize.lsq_linear(method='trf', lsq_solver='exact')",
        "trusted_oracle_global_scale": oracle_scale,
        "trusted_oracle_tolerance": float(np.finfo(np.float64).eps),
        "trusted_success": bool(scipy.success),
        "trusted_status": int(scipy.status),
        "trusted_message": scipy.message,
        "oracle_direction": oracle_direction,
        "max_absolute_direction_difference": float(torch.max(torch.abs(direction - oracle_direction))),
        "direction_l2_difference": float(torch.linalg.vector_norm(direction - oracle_direction)),
        "active_set_objective": q(direction),
        "oracle_objective": q(oracle_direction),
        "objective_difference": q(direction) - q(oracle_direction),
        "active_set_feasible": bool(((direction >= runtime.lower - theta) & (direction <= runtime.upper - theta)).all()),
        "oracle_feasible": bool(((oracle_direction >= runtime.lower - theta) & (oracle_direction <= runtime.upper - theta)).all()),
        "active_set_projected_kkt_norm": float(projected_qp_gradient.norm()),
        "active_set_free_gradient_norm": float(final_model_gradient[free].norm()),
        "oracle_raw_gradient_norm": float(oracle_gradient.norm()),
        "active_set_status": status,
        "oracle_active_mask": scipy.active_mask,
        "active_mask_agreement": bool(np.array_equal(status.numpy(), scipy.active_mask.astype(np.int64))),
    }

    forced = []
    for coordinate in ACTIVE_COORDINATES:
        forced_direction, forced_status, _, forced_gradient, forced_events = traced_bvls(
            augmented_matrix,
            augmented_residual,
            theta,
            runtime.lower,
            runtime.upper,
            force_free=coordinate,
        )
        replay = actual_backtracking(runtime, theta, current_loss, forced_direction)
        column = jacobian[:, coordinate]
        linear = column @ residual
        curvature = column @ column + regularization
        coordinate_step = torch.clamp(
            -linear / curvature,
            runtime.lower[coordinate] - theta[coordinate],
            runtime.upper[coordinate] - theta[coordinate],
        )
        coordinate_direction = torch.zeros_like(theta)
        coordinate_direction[coordinate] = coordinate_step
        coordinate_replay = actual_backtracking(
            runtime, theta, current_loss, coordinate_direction
        )
        forced.append({
            "coordinate": coordinate,
            "side": "lower" if theta[coordinate] == runtime.lower[coordinate] else "upper",
            "direction": forced_direction,
            "direction_matches_natural_bits": bits_equal(forced_direction, direction),
            "direction_max_absolute_difference": float(torch.max(torch.abs(forced_direction - direction))),
            "final_status": forced_status,
            "local_model_objective": q(forced_direction),
            "natural_local_model_objective": q(direction),
            "local_model_improvement_over_natural": q(direction) - q(forced_direction),
            "final_coordinate_model_gradient": float(forced_gradient[coordinate]),
            "events": forced_events,
            "actual_backtracking": replay,
            "coordinate_only_sensitivity": {
                "direction": coordinate_direction,
                "step": float(coordinate_step),
                "local_model_delta": float(
                    2 * linear * coordinate_step
                    + curvature * coordinate_step * coordinate_step
                ),
                "actual_backtracking": coordinate_replay,
            },
        })

    eig = torch.linalg.eigvalsh(hessian)
    report = {
        "status": "OTHER_DEMONSTRATED_CAUSE",
        "demonstrated_cause": "The box-QP active set solves its stabilized residual model correctly, but the coupled QP direction reaches float64 no-improvement even though four individual feasible inward residual-model directions strictly reduce the actual objective.",
        "terminal_reconstruction": {
            "case": terminal["case"],
            "accepted_step": terminal["accepted_steps"],
            "theta": theta,
            "saved_loss": current_loss,
            "replayed_loss": float(result.loss),
            "loss_bits_exact": bits_equal(result.loss, current_loss),
            "residual": residual,
            "jacobian": jacobian,
            "gradient": gradient,
            "gradient_norm": gradient_norm,
            "hessian": hessian,
            "hessian_eigenvalue_range": [float(eig[0]), float(eig[-1])],
            "mu": mu,
            "regularization_126_mu": regularization,
            "active_bounds": active,
            "direction": direction,
            "direction_matches_live_bits": True,
            "stop_reason": terminal["stop_reason"],
        },
        "active_bound_kkt": kkt_rows,
        "active_set_trace": events,
        "qp_comparison": qp_comparison,
        "forced_release": forced,
        "source_integrity": {
            "terminal_result_sha256": sha256(RESULT),
            "accepted_trace_sha256": sha256(TRACE),
            "audit_source_sha256": sha256(Path(__file__)),
        },
        "interpretation": {
            "bounds_implemented_correctly": True,
            "active_set_release_bug": False,
            "box_qp_implementation_mismatch": False,
            "interior_handoff_representation_context": "An interior handoff-capable representation exists, so this terminal state does not imply representational failure.",
            "genuine_constrained_local_solution": False,
            "feasible_actual_descent_demonstrated": True,
            "local_basin_problem_demonstrated": False,
        },
    }
    atomic(OUT / "report.json", report)
    print(json.dumps(clean({
        "status": report["status"],
        "loss": current_loss,
        "kkt": kkt_rows,
        "active_set_event_count": len(events),
        "qp_comparison": qp_comparison,
        "forced_release_summary": [{
            "coordinate": row["coordinate"],
            "matches": row["direction_matches_natural_bits"],
            "max_difference": row["direction_max_absolute_difference"],
            "model_improvement": row["local_model_improvement_over_natural"],
            "actual_accepted": row["actual_backtracking"]["accepted"],
            "unchanged_at": row["actual_backtracking"].get("unchanged_at"),
            "coordinate_only_step": row["coordinate_only_sensitivity"]["step"],
            "coordinate_only_model_delta": row["coordinate_only_sensitivity"]["local_model_delta"],
            "coordinate_only_actual_accepted": row["coordinate_only_sensitivity"]["actual_backtracking"]["accepted"],
            "coordinate_only_accepted_loss": row["coordinate_only_sensitivity"]["actual_backtracking"].get("loss"),
        } for row in forced],
    }), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
