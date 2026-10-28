"""Fail-closed validation of the generic-SHO stabilized-residual opt-in path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.resume_hybrid_student_after_ckdtree import Log
from experiments.sho import run_stage1b_n3_gradient_polishing as production_n3
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.generic_update_network_adapter import GenericSHOUpdateNetworkAdapter
from experiments.sho.generic_update_residual_n3_opt_in import (
    DEFAULT_DIRECTION_MODE,
    EXISTING_FULL_HESSIAN,
    STABILIZED_RESIDUAL,
    OptInResidualN3Runtime,
    residual_and_jacobian,
)
from experiments.sho.validate_generic_update_network_adapter import protected


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path("outputs/sho/stabilized_residual_n3_from_n2")
REFERENCE = ROOT / "stabilized.jsonl"
N2_ENDPOINT = Path(
    "outputs/sho/generic_update_network_targeted_recovery/companion_6/"
    "n00005_i00004_n2.json"
)
EXISTING_REFERENCE = Path(
    "outputs/sho/generic_update_network_targeted_recovery/companion_6/"
    "n00005_i00004_n3.jsonl"
)
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
OUT = Path("outputs/sho/stabilized_residual_n3_opt_in_integration")
IMPLEMENTATION_SOURCE = Path("experiments/sho/generic_update_residual_n3_opt_in.py")
VALIDATION_SOURCE = Path("experiments/sho/validate_generic_update_residual_n3_opt_in.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bit_equal(left, right) -> bool:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_reference():
    rows = []
    with REFERENCE.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid reference JSONL line {line_number}") from exc
    if [row["accepted_steps"] for row in rows] != list(range(501)):
        raise RuntimeError("reference trace is not the exact 0..500 accepted-step trajectory")
    return rows


def fail(step: int, operation: str, teacher, student):
    def serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    raise RuntimeError(
        json.dumps(
            {
                "first_divergence_step": step,
                "operation": operation,
                "reference": serializable(teacher),
                "integrated": serializable(student),
            },
            allow_nan=False,
        )
    )


def require_bits(step, operation, teacher, student):
    if not bit_equal(teacher, student):
        fail(step, operation, teacher, student)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = load_reference()
    trajectory = torch.from_numpy(np.load(DATA)["prefix_6"].copy())
    n2 = json.loads(N2_ENDPOINT.read_text())["result"]
    theta0 = torch.tensor(n2["theta"])
    loss0 = float(n2["loss"])
    if not bit_equal(theta0, reference[0]["phi"]) or not bit_equal(loss0, reference[0]["loss"]):
        raise RuntimeError("saved N2 endpoint does not match reference start")

    before = {
        "protected": protected(),
        "checkpoints": hashes(),
        "reference": sha256(REFERENCE),
        "n2_endpoint": sha256(N2_ENDPOINT),
    }

    # Default-binding proof: absence of a mode selects the old direction and
    # never invokes the stabilized direction output.
    default_adapter = GenericSHOUpdateNetworkAdapter()
    default_runtime = OptInResidualN3Runtime(default_adapter, trajectory, Log())
    if default_runtime.direction_mode != DEFAULT_DIRECTION_MODE or DEFAULT_DIRECTION_MODE != EXISTING_FULL_HESSIAN:
        raise RuntimeError("default direction binding changed")
    default_values = default_runtime.derivatives_and_direction(theta0)
    with EXISTING_REFERENCE.open() as stream:
        existing_reference = json.loads(next(stream))
    require_bits(
        0,
        "default_direction",
        existing_reference["diagnostics"]["direction"],
        default_values[6],
    )
    if default_runtime.binding_counters.stabilized_direction_outputs != 0:
        raise RuntimeError("default path invoked stabilized direction")

    adapter = GenericSHOUpdateNetworkAdapter()
    runtime = OptInResidualN3Runtime(
        adapter, trajectory, Log(), direction_mode=STABILIZED_RESIDUAL
    )
    theta = theta0.clone()
    loss = loss0
    accepted = 0
    stagnation = 0
    objective_evaluations = 0
    backtracking_rejections = 0
    path_length = 0.0
    comparisons = {
        "states": 0,
        "phi_values": 0,
        "residual_values": 0,
        "gradient_values": 0,
        "hessian_values": 0,
        "direction_values": 0,
        "accepted_updates": 0,
    }
    trace_path = OUT / "integrated_stabilized.jsonl"
    start_time = time.monotonic()
    with trace_path.open("w") as trace:
        while True:
            ref = reference[accepted]
            values = runtime.derivatives_and_direction(theta)
            result, gradient, hessian, gradient_norm, pre, route, direction, qp = values
            residual, jacobian = residual_and_jacobian(adapter, trajectory, theta)
            ref_theta = torch.tensor(ref["phi"])
            ref_residual, _ = residual_and_jacobian(adapter, trajectory, ref_theta)

            require_bits(accepted, "phi", ref["phi"], theta)
            require_bits(accepted, "loss", ref["loss"], loss)
            require_bits(accepted, "objective", ref["loss"], result.loss)
            require_bits(accepted, "residual", ref_residual, residual)
            require_bits(accepted, "gradient", ref["gradient"], gradient)
            require_bits(accepted, "hessian", ref["hessian"], hessian)
            require_bits(accepted, "gradient_norm", ref["gradient_norm"], gradient_norm)
            require_bits(accepted, "mu", ref["mu"], qp["damping"])
            require_bits(accepted, "direction", ref["direction"], direction)
            require_bits(
                accepted,
                "full_predicted_decrease",
                ref["full_predicted_decrease"],
                -qp["quadratic_value"],
            )
            if route != ref["route"]:
                fail(accepted, "curvature_route", ref["route"], route)
            bounds = torch.where((theta == runtime.lower) | (theta == runtime.upper))[0].tolist()
            if bounds != ref["bound_contacts"]:
                fail(accepted, "bound_contacts", ref["bound_contacts"], bounds)
            if stagnation != ref["stagnation_count"]:
                fail(accepted, "stagnation_count", ref["stagnation_count"], stagnation)

            comparisons["states"] += 1
            comparisons["phi_values"] += theta.numel()
            comparisons["residual_values"] += residual.numel()
            comparisons["gradient_values"] += gradient.numel()
            comparisons["hessian_values"] += hessian.numel()
            comparisons["direction_values"] += direction.numel()

            row = {
                "accepted_steps": accepted,
                "phi": theta.tolist(),
                "loss": loss,
                "residual": residual.tolist(),
                "gradient": gradient.tolist(),
                "hessian": hessian.tolist(),
                "gradient_norm": gradient_norm,
                "mu": qp["damping"],
                "route": route,
                "direction": direction.tolist(),
                "bound_contacts": bounds,
                "stagnation_count": stagnation,
                "full_predicted_decrease": -qp["quadratic_value"],
            }
            if accepted >= 500:
                row["stop_reason"] = "VALIDATION_LIMIT_500_ACCEPTED_STEPS"
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                break
            if pre != "CONTINUE":
                fail(accepted, "unexpected_pre_step_stop", "CONTINUE", pre)

            runtime.matrix_rhs.construct(hessian, gradient, float(qp["damping"]), runtime.log)
            descent, dot, _ = runtime.controls.descent.check(gradient, direction)
            if not descent:
                fail(accepted, "descent_route", True, False)

            took_step = False
            unchanged = False
            for backtrack in range(production_n3.MAX_BACKTRACKS + 1):
                candidate = runtime.controls.candidate(theta, direction, backtrack)
                if torch.equal(candidate, theta):
                    unchanged = True
                    break
                if not runtime.controls.in_bounds(
                    candidate, runtime.lower.numpy(), runtime.upper.numpy()
                ):
                    backtracking_rejections += 1
                    continue
                new_loss = runtime.scalar_loss(candidate.numpy())
                objective_evaluations += 1
                if runtime.controls.accepts(new_loss, loss):
                    ref_update = ref["accepted_update"]
                    if backtrack != ref_update["backtrack"]:
                        fail(accepted, "accepted_backtrack", ref_update["backtrack"], backtrack)
                    require_bits(accepted, "accepted_phi", ref_update["new_phi"], candidate)
                    require_bits(accepted, "accepted_loss", ref_update["new_loss"], new_loss)
                    displacement = candidate - theta
                    displacement_norm = float(displacement.norm())
                    require_bits(
                        accepted,
                        "accepted_displacement_norm",
                        ref_update["displacement_norm"],
                        displacement_norm,
                    )
                    actual_decrease = loss - new_loss
                    require_bits(
                        accepted,
                        "actual_loss_decrease",
                        ref_update["actual_loss_decrease"],
                        actual_decrease,
                    )
                    row["accepted_update"] = {
                        "backtrack": backtrack,
                        "new_phi": candidate.tolist(),
                        "new_loss": new_loss,
                        "displacement_norm": displacement_norm,
                        "actual_loss_decrease": actual_decrease,
                    }
                    stagnation = production_n3._updated_float64_stagnation_count(
                        stagnation,
                        loss,
                        new_loss,
                        -float(qp["quadratic_value"]),
                    )
                    theta = candidate.detach()
                    loss = new_loss
                    accepted += 1
                    path_length += displacement_norm
                    comparisons["accepted_updates"] += 1
                    took_step = True
                    break
                backtracking_rejections += 1

            trace.write(json.dumps(row, allow_nan=False) + "\n")
            if not took_step:
                numerical, _, _ = runtime.controls.no_further_improvement(unchanged, dot, loss)
                fail(
                    accepted,
                    "unexpected_terminal_stop",
                    "accepted step",
                    "floating_point_no_further_improvement" if numerical else "backtracking_no_improvement",
                )

    require_bits(500, "final_loss", 8.903393153798347e-12, loss)
    if runtime.binding_counters.existing_direction_outputs != 0:
        raise RuntimeError("stabilized replay emitted an existing-direction output")
    if runtime.binding_counters.stabilized_direction_outputs != 501:
        raise RuntimeError("unexpected stabilized binding invocation count")

    after = {
        "protected": protected(),
        "checkpoints": hashes(),
        "reference": sha256(REFERENCE),
        "n2_endpoint": sha256(N2_ENDPOINT),
    }
    if before != after:
        raise RuntimeError("frozen/checkpoint/reference integrity changed")

    report = {
        "status": "STABILIZED_RESIDUAL_N3_OPT_IN_INTEGRATED",
        "case": "companion_6",
        "branch": [5, 4],
        "n2_evaluations": 1500,
        "direction_modes": {
            "default": DEFAULT_DIRECTION_MODE,
            "opt_in": STABILIZED_RESIDUAL,
        },
        "default_binding": {
            "existing_direction_outputs": default_runtime.binding_counters.existing_direction_outputs,
            "stabilized_direction_outputs": default_runtime.binding_counters.stabilized_direction_outputs,
            "direction_exact": True,
        },
        "opt_in_binding": vars(runtime.binding_counters),
        "accepted_steps": accepted,
        "start_loss": loss0,
        "final_loss": loss,
        "reference_final_loss": 8.903393153798347e-12,
        "comparisons": comparisons,
        "backtracking_rejections": backtracking_rejections,
        "objective_evaluations": objective_evaluations,
        "parameter_path_length": path_length,
        "elapsed_seconds": time.monotonic() - start_time,
        "trace_sha256": sha256(trace_path),
        "reference_sha256": sha256(REFERENCE),
        "source_hashes": {
            "implementation": sha256(IMPLEMENTATION_SOURCE),
            "validation": sha256(VALIDATION_SOURCE),
        },
        "checkpoint_hashes": after["checkpoints"],
        "validated_checkpoints": [1, 10, 50, 100, 250, 500],
        "integrity_unchanged": True,
        "production_modified": False,
    }
    write_json(OUT / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
