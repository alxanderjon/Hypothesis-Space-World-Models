#!/usr/bin/env python3
"""Recover exact unary negation and resume fail-closed hybrid N3 replay."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.float64_unary_negation_assembly import (
    Float64UnaryNegationAssembly,
)
from experiments.geometry.n3_eigenspectrum_condition_assembly import (
    N3EigenspectrumConditionAssembly,
)
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.recover_full_k_sho_forward_arithmetic import (
    CHECKPOINT_REFERENCE,
    comparison,
    specification,
    theta_corpus,
)
from experiments.recover_full_k_sho_mm_backward_dot import native_gradient
from experiments.resume_hybrid_student_at_curvature import (
    load_locked_router,
    student_direct,
    teacher_damping,
)
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho.assemble_n3_learned_polisher import ControlAssembly
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/float64_unary_negation_and_hybrid_resume")
IMPLEMENTATION = Path("experiments/geometry/float64_unary_negation_assembly.py")
XOR_MANIFEST = Path(
    "outputs/sho/stage1B/shared_one_bit_xor/shared_one_bit_xor_locked.json"
)
PRIOR = Path("outputs/hybrid_student_after_ckdtree/downstream_report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(left, right) -> dict:
    return comparison(np.asarray(left, np.float64), np.asarray(right, np.float64))


def edge_controls(assembly: Float64UnaryNegationAssembly) -> dict:
    patterns = np.asarray([
        0x0000_0000_0000_0000,  # +0
        0x8000_0000_0000_0000,  # -0
        0x0000_0000_0000_0001,  # minimum positive subnormal
        0x8000_0000_0000_0001,
        0x000F_FFFF_FFFF_FFFF,  # maximum positive subnormal
        0x800F_FFFF_FFFF_FFFF,
        0x0010_0000_0000_0000,  # minimum positive normal
        0x8010_0000_0000_0000,
        0x3FF0_0000_0000_0000,  # +/-1
        0xBFF0_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,  # maximum finite
        0xFFEF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0000,  # infinities
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,  # quiet NaNs with payload
        0xFFF8_0000_0000_1234,
        0x7FF0_0000_0000_0001,  # signaling NaNs with payload
        0xFFF0_0000_0000_1234,
    ], dtype=np.uint64)
    inputs = torch.from_numpy(patterns.view(np.float64).copy())
    teacher = torch.negative(inputs)
    student = assembly(inputs)
    twice = assembly(student)
    return {
        "values": len(patterns),
        "patterns_hex": [f"0x{int(value):016x}" for value in patterns],
        "negation": metric(teacher.numpy(), student.numpy()),
        "double_negation": metric(inputs.numpy(), twice.numpy()),
        "nan_payloads_preserved": bool(np.array_equal(
            teacher.numpy().view(np.uint64)[-4:] & np.uint64(0x7FFF_FFFF_FFFF_FFFF),
            patterns[-4:] & np.uint64(0x7FFF_FFFF_FFFF_FFFF),
        )),
    }


def sho_rhs_corpus(assembly: Float64UnaryNegationAssembly) -> dict:
    gradients = []
    by_source = Counter()
    states = 0
    for case_id in range(64):
        trajectory, _, _, _, _, _, teacher = readiness.case_data(specification(case_id))
        inputs = torch.from_numpy(trajectory[:-1].copy())
        targets = torch.from_numpy(trajectory[1:].copy())
        for source, theta in theta_corpus(case_id, teacher):
            _, gradient_value = native_gradient(inputs, targets, theta)
            gradients.append(gradient_value.reshape(-1))
            by_source[source] += 1
            states += 1
    values = torch.from_numpy(np.asarray(gradients, np.float64))
    teacher_rhs = torch.negative(values)
    student_rhs = assembly(values)
    row_exact = torch.all(
        teacher_rhs.view(torch.int64) == student_rhs.view(torch.int64), dim=1
    )
    return {
        "cases": 64,
        "states": states,
        "rhs_values": int(values.numel()),
        "exact_rhs_values": int(torch.count_nonzero(
            teacher_rhs.view(torch.int64) == student_rhs.view(torch.int64))),
        "exact_rhs_vectors": int(torch.count_nonzero(row_exact)),
        "metric": metric(teacher_rhs.numpy(), student_rhs.numpy()),
        "by_source_states": dict(by_source),
    }


def find_spec(system: str, case_id: int) -> dict:
    manifest = json.loads(readiness.MANIFEST.read_text())
    return next(row for row in manifest["cases"]
                if row["system"] == system and int(row["case_id"]) == case_id)


def saved_branch(system: str, case_id: int) -> dict:
    return json.loads((readiness.CASES / system / f"case_{case_id:05d}" /
                       "result.json").read_text())["successful_branch"]


def divergence(operation: str, system: str, iteration: int, **details) -> dict:
    return {
        "operation": operation,
        "system": system,
        "iteration": iteration,
        **details,
    }


def replay_system(row: dict, assembly: Float64UnaryNegationAssembly,
                  controls: ControlAssembly,
                  spectral: N3EigenspectrumConditionAssembly) -> dict:
    system, case_id = row["system"], int(row["case_id"])
    spec = find_spec(system, case_id)
    _, _, _, torch_objective, lower, upper, _ = readiness.case_data(spec)
    derivatives, direction_function, _ = staged._make_n3(
        torch_objective, lower, upper
    )
    theta = torch.tensor(row["n2_endpoint"], dtype=torch.float64)
    current_loss = float(torch_objective(theta))
    lower_tensor = torch.from_numpy(lower.copy())
    upper_tensor = torch.from_numpy(upper.copy())
    stagnation_count = 0
    counts = Counter()
    diagnostic_dot_mismatches = 0
    first_divergence = None
    stop_reason = None
    accepted_steps = 0

    while True:
        _, gradient_value, hessian_value = derivatives(theta, torch.empty(0))
        gradient_norm = float(torch.linalg.vector_norm(gradient_value))
        teacher_pre = (
            "nonfinite_gradient" if not math.isfinite(gradient_norm)
            else "floating_point_zero_gradient" if gradient_norm == 0.0
            else "CONTINUE"
        )
        student_pre = controls.pre_step_route(0, gradient_norm)
        counts["genuine_stop_routes"] += 1
        if teacher_pre != student_pre:
            first_divergence = divergence(
                "genuine numerical STOP/CONTINUE routes", system, accepted_steps,
                teacher=teacher_pre, student=student_pre, mismatch_type="routing",
                mismatch_count=1,
            )
            break
        if teacher_pre != "CONTINUE":
            stop_reason = teacher_pre
            break

        spectrum = spectral.analyze(gradient_value, hessian_value)
        teacher_direct = bool(
            spectrum.finite and spectrum.minimum_eigenvalue > 0.0
            and spectrum.condition <= production.CONDITION_LIMIT
        )
        learned_direct = student_direct(
            controls.curvature, spectrum.finite,
            spectrum.positive, spectrum.condition_ok,
        )
        counts["curvature_routes"] += 1
        if teacher_direct != learned_direct:
            first_divergence = divergence(
                "DIRECT/DAMPED curvature route", system, accepted_steps,
                teacher=teacher_direct, student=learned_direct,
                mismatch_type="routing", mismatch_count=1,
            )
            break

        # Validate every retry value before invoking the declared solve/box-QP
        # deterministic oracles.  This does not alter which trial is executed.
        learned_initial = controls.damping.initial(
            spectrum.scale, spectrum.minimum_eigenvalue, learned_direct
        )
        for attempt in range(production.MAX_DAMPING_TRIALS):
            expected_initial, expected_value = teacher_damping(
                spectrum.scale, spectrum.minimum_eigenvalue,
                teacher_direct, attempt,
            )
            learned_value = controls.damping.value(
                spectrum.scale, learned_initial, learned_direct, attempt
            )
            counts["damping_retry_values"] += 1
            if (np.float64(expected_initial).view(np.uint64) !=
                    np.float64(learned_initial).view(np.uint64) or
                    np.float64(expected_value).view(np.uint64) !=
                    np.float64(learned_value).view(np.uint64)):
                first_divergence = divergence(
                    "damping arithmetic/retry values", system, accepted_steps,
                    attempt=attempt, teacher_initial=expected_initial,
                    student_initial=learned_initial, teacher=expected_value,
                    student=learned_value, mismatch_type="float64 bits",
                    mismatch_count=1,
                )
                break
        if first_divergence:
            break

        # Retain dense solve and dimension-general box-QP as explicit D oracles.
        direction, qp = dimension_general_box_qp_direction(
            theta, gradient_value, hessian_value,
            lower_tensor, upper_tensor, direction_function,
        )
        counts["box_qp_oracle_calls"] += 1
        if direction is None:
            stop_reason = qp["route_status"]
            break

        damping = float(qp["damping"])
        identity = torch.eye(len(gradient_value), dtype=torch.float64)
        teacher_matrix = hessian_value + damping * identity
        student_matrix = controls.damping.matrix(hessian_value, damping)
        teacher_rhs = torch.negative(gradient_value)
        student_rhs = assembly(gradient_value)
        matrix_metric = metric(teacher_matrix.numpy(), student_matrix.numpy())
        rhs_metric = metric(teacher_rhs.numpy(), student_rhs.numpy())
        counts["damped_matrix_values"] += teacher_matrix.numel()
        counts["rhs_values"] += teacher_rhs.numel()
        if matrix_metric["bit_mismatches"] or rhs_metric["bit_mismatches"]:
            first_divergence = divergence(
                "damped matrix and RHS assembly", system, accepted_steps,
                teacher_matrix=teacher_matrix.tolist(),
                student_matrix=student_matrix.tolist(), matrix_metric=matrix_metric,
                teacher_rhs=teacher_rhs.tolist(), student_rhs=student_rhs.tolist(),
                rhs_metric=rhs_metric, mismatch_type="float64 bits",
                mismatch_count=(matrix_metric["bit_mismatches"] +
                                rhs_metric["bit_mismatches"]),
            )
            break

        teacher_dot = float(torch.dot(gradient_value, direction))
        learned_descent, learned_dot, _ = controls.descent.check(
            gradient_value, direction
        )
        teacher_descent = bool(torch.isfinite(direction).all()) and teacher_dot < 0.0
        counts["descent_routes"] += 1
        diagnostic_dot_mismatches += int(
            np.float64(teacher_dot).view(np.uint64) !=
            np.float64(learned_dot).view(np.uint64)
        )
        if teacher_descent != learned_descent:
            first_divergence = divergence(
                "descent validity", system, accepted_steps,
                teacher=teacher_descent, student=learned_descent,
                teacher_dot=teacher_dot, student_internal_dot=learned_dot,
                mismatch_type="routing", mismatch_count=1,
            )
            break

        full_predicted_decrease = -float(qp["quadratic_value"])
        accepted = False
        unchanged = False
        for backtrack in range(production.MAX_BACKTRACKS + 1):
            teacher_candidate = theta + (2.0 ** (-backtrack)) * direction
            student_candidate = controls.candidate(theta, direction, backtrack)
            candidate_metric = metric(
                teacher_candidate.numpy(), student_candidate.numpy()
            )
            counts["backtracking_candidates"] += 1
            if candidate_metric["bit_mismatches"]:
                first_divergence = divergence(
                    "power-of-two backtracking schedule", system, accepted_steps,
                    backtrack=backtrack, teacher=teacher_candidate.tolist(),
                    student=student_candidate.tolist(), metric=candidate_metric,
                    mismatch_type="float64 bits",
                    mismatch_count=candidate_metric["bit_mismatches"],
                )
                break
            if bool(torch.equal(teacher_candidate, theta)):
                unchanged = True
                break
            teacher_bounds = bool(
                torch.all(teacher_candidate >= lower_tensor)
                and torch.all(teacher_candidate <= upper_tensor)
            )
            student_bounds = controls.in_bounds(
                student_candidate, lower, upper
            )
            counts["bounds_routes"] += 1
            if teacher_bounds != student_bounds:
                first_divergence = divergence(
                    "candidate bounds route", system, accepted_steps,
                    backtrack=backtrack, teacher=teacher_bounds,
                    student=student_bounds, mismatch_type="routing",
                    mismatch_count=1,
                )
                break
            if not teacher_bounds:
                continue
            candidate_loss = float(torch_objective(teacher_candidate))
            teacher_accept = candidate_loss < current_loss
            student_accept = controls.accepts(candidate_loss, current_loss)
            counts["strict_acceptance_routes"] += 1
            if teacher_accept != student_accept:
                first_divergence = divergence(
                    "strict step acceptance", system, accepted_steps,
                    backtrack=backtrack, current_loss=current_loss,
                    candidate_loss=candidate_loss, teacher=teacher_accept,
                    student=student_accept, mismatch_type="routing",
                    mismatch_count=1,
                )
                break
            if teacher_accept:
                stagnation_count = production._updated_float64_stagnation_count(
                    stagnation_count, current_loss, candidate_loss,
                    full_predicted_decrease,
                )
                theta = teacher_candidate.detach()
                current_loss = candidate_loss
                accepted_steps += 1
                accepted = True
                break
        if first_divergence:
            break
        if accepted:
            if (stagnation_count >=
                    production.FLOAT64_STAGNATION_ACCEPTED_STEP_PERSISTENCE):
                stop_reason = production.FLOAT64_STAGNATION_STOP_REASON
                break
            continue

        learned_stop, _, _ = controls.no_further_improvement(
            unchanged, teacher_dot, current_loss
        )
        teacher_stop = bool(
            unchanged or max(0.0, -teacher_dot) <=
            32.0 * abs(float(np.spacing(np.float64(current_loss))))
        )
        counts["genuine_stop_routes"] += 1
        if learned_stop != teacher_stop:
            first_divergence = divergence(
                "genuine numerical STOP/CONTINUE routes", system, accepted_steps,
                teacher=teacher_stop, student=learned_stop,
                teacher_dot=teacher_dot, current_loss=current_loss,
                mismatch_type="routing", mismatch_count=1,
            )
            break
        stop_reason = (
            "floating_point_no_further_improvement" if teacher_stop
            else "backtracking_no_improvement"
        )
        break

    saved = saved_branch(system, case_id)
    final_metric = metric(saved["final_theta"], theta.numpy())
    loss_metric = metric([saved["final_loss"]], [current_loss])
    return {
        "system": system,
        "case_id": case_id,
        "accepted_steps": accepted_steps,
        "stop_reason": stop_reason,
        "final_theta": theta.tolist(),
        "final_loss": current_loss,
        "saved_teacher_accepted_steps": int(saved["n3_accepted_steps"]),
        "saved_teacher_stop_reason": saved["n3_stop_reason"],
        "final_theta_metric": final_metric,
        "final_loss_metric": loss_metric,
        "operation_counts": dict(counts),
        "descent_internal_dot_bit_mismatches_diagnostic_only": diagnostic_dot_mismatches,
        "first_divergence": first_divergence,
    }


def run() -> dict:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected:
        raise RuntimeError("checkpoint inventory changed before unary-negation replay")
    protected_before = {
        "shared_xor_manifest": sha256(XOR_MANIFEST),
        **{path: value for path, value in checkpoint_hashes().items()},
    }
    xor_manifest = json.loads(XOR_MANIFEST.read_text())
    if xor_manifest.get("status") != "LOCKED_ASSEMBLY":
        raise RuntimeError("locked one-bit XOR assembly is unavailable")

    negation = Float64UnaryNegationAssembly()
    edges = edge_controls(negation)
    corpus = sho_rhs_corpus(negation)
    unary_exact = bool(
        edges["negation"]["bit_mismatches"] == 0
        and edges["double_negation"]["bit_mismatches"] == 0
        and edges["nan_payloads_preserved"]
        and corpus["metric"]["bit_mismatches"] == 0
    )
    if not unary_exact:
        raise RuntimeError("unary-negation exactness gate failed")

    manifest = {
        "status": "LOCKED_ASSEMBLY",
        "name": "float64_unary_negation",
        "formula": "reinterpret binary64 -> XOR sign bit with 1 -> repack unchanged payload",
        "reused": "shared_one_bit_xor LOCKED_ASSEMBLY",
        "new_parameters": 0,
        "implementation": str(IMPLEMENTATION),
        "implementation_sha256": sha256(IMPLEMENTATION),
        "xor_manifest": str(XOR_MANIFEST),
        "xor_manifest_sha256": sha256(XOR_MANIFEST),
        "edge_validation": edges,
        "corpus_validation": corpus,
    }
    (OUT / "unary_negation_locked.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    gradient.OPS = gradient.LearnedArithmetic()
    controls = ControlAssembly()
    # Explicitly load the locked curvature checkpoint into the active router.
    locked_router = load_locked_router()
    controls.curvature.and_router.load_state_dict(
        locked_router.and_router.state_dict()
    )
    spectral = N3EigenspectrumConditionAssembly(production.CONDITION_LIMIT)
    prior = json.loads(PRIOR.read_text())
    rows = []
    first_divergence = None
    for row in prior["archive_n2_replay"]:
        replay = replay_system(row, negation, controls, spectral)
        rows.append(replay)
        if replay["first_divergence"] is not None:
            first_divergence = replay["first_divergence"]
            break

    operations = [
        "float64 unary negation",
        "damped matrix and RHS assembly",
    ]
    if first_divergence is None:
        operations.extend([
            "descent validity",
            "power-of-two backtracking schedule",
            "candidate bounds route",
            "strict step acceptance",
            "genuine numerical STOP/CONTINUE routes",
        ])
    else:
        ordered = [
            "descent validity", "power-of-two backtracking schedule",
            "candidate bounds route", "strict step acceptance",
            "genuine numerical STOP/CONTINUE routes",
        ]
        operations.extend(ordered[:ordered.index(first_divergence["operation"])])

    protected_after = {
        "shared_xor_manifest": sha256(XOR_MANIFEST),
        **{path: value for path, value in checkpoint_hashes().items()},
    }
    report = {
        "status": (
            "STOPPED_FIRST_A_B_DIVERGENCE" if first_divergence
            else "NO_NEW_A_B_DIVERGENCE_IN_EXISTING_HYBRID_REPLAY"
        ),
        "unary_negation": manifest,
        "hybrid_replay": rows,
        "successfully_integrated_operations": operations,
        "retained_D_oracles": [
            "Accelerate dgesdd/dsyevd",
            "torch.linalg.solve dense solve",
            "dimension-general box-QP active-set direction",
            "FLOAT64_STAGNATION_TAIL stop",
        ],
        "first_divergence": first_divergence,
        "updated_integration_boundary": (
            first_divergence["operation"] if first_divergence else
            "no unresolved A/B operation remains after RHS in the existing representative N3 replay"
        ),
        "checkpoint_hashes_unchanged": protected_before == protected_after,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "unary_negation": {
            "edge_values": edges["values"],
            "edge_bit_mismatches": edges["negation"]["bit_mismatches"],
            "rhs_values": corpus["rhs_values"],
            "rhs_bit_mismatches": corpus["metric"]["bit_mismatches"],
            "rhs_vectors_exact": corpus["exact_rhs_vectors"],
        },
        "systems": [{
            "system": row["system"],
            "accepted_steps": row["accepted_steps"],
            "stop_reason": row["stop_reason"],
            "theta_bit_mismatches": row["final_theta_metric"]["bit_mismatches"],
            "loss_bit_mismatches": row["final_loss_metric"]["bit_mismatches"],
            "first_divergence": row["first_divergence"],
        } for row in rows],
        "integrated": operations,
        "first_divergence": first_divergence,
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))
    return report


if __name__ == "__main__":
    run()

