#!/usr/bin/env python3
"""Resume fail-closed hybrid integration at the locked N3 curvature router."""
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
from experiments.geometry.full_k_sho_hessian_assembly import FullKSHOHessianAssembly
from experiments.geometry.n3_eigenspectrum_condition_assembly import (
    N3EigenspectrumConditionAssembly,
)
from experiments.recover_full_k_sho_forward_arithmetic import (
    CHECKPOINT_REFERENCE,
    comparison,
    specification,
    theta_corpus,
)
from experiments.recover_full_k_sho_hessian_replay import teacher_details
from experiments.recover_full_k_sho_mm_backward_dot import native_gradient
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho import train_n3_curvature_router as curvature
from experiments.sho.assemble_n3_newton_step_operator import (
    DampingAssembly,
    ScalarArithmetic,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_at_curvature")
CURVATURE_CHECKPOINT = Path(
    "outputs/sho/stage1B/n3_learned_polisher/curvature_router/n3_curvature_router.pt"
)
SPECTRAL_REPORT = Path("outputs/n3_eigenspectrum_condition_replay/report.json")
PRIOR_HYBRID_REPORT = Path("outputs/hybrid_student_after_ckdtree/downstream_report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left: float, right: float) -> bool:
    return bool(np.float64(left).view(np.uint64) == np.float64(right).view(np.uint64))


def metric(left, right) -> dict:
    return comparison(np.asarray(left, np.float64), np.asarray(right, np.float64))


def load_locked_router() -> curvature.CurvatureRouter:
    saved = torch.load(CURVATURE_CHECKPOINT, map_location="cpu", weights_only=False)
    if saved.get("status") != "LOCKED":
        raise RuntimeError("locked N3 curvature router is unavailable")
    router = curvature.CurvatureRouter()
    router.and_router.load_state_dict(saved["state_dict"])
    router.and_router.eval().requires_grad_(False)
    return router


def student_direct(router: curvature.CurvatureRouter, finite: bool,
                   positive: bool, condition_ok: bool) -> bool:
    # Production exits before the Boolean router on nonfinite derivatives.
    return bool(finite and router.conjunction(positive, condition_ok))


def teacher_direct(finite: bool, minimum: float | None,
                   condition: float) -> bool:
    return bool(
        finite
        and minimum is not None
        and minimum > 0.0
        and condition <= production.CONDITION_LIMIT
    )


def curvature_corpus(router: curvature.CurvatureRouter,
                     spectral: N3EigenspectrumConditionAssembly) -> dict:
    hessian_assembly = FullKSHOHessianAssembly()
    teacher_counts = Counter()
    student_counts = Counter()
    route_mismatches = 0
    states = 0
    unique_hessians = 0
    finite_gradient_checks = 0
    nonfinite_gradient_checks = 0
    cases = []
    for case_id in range(64):
        trajectory, _, _, _, _, _, teacher_theta = readiness.case_data(specification(case_id))
        inputs_np, targets_np = trajectory[:-1], trajectory[1:]
        _, hessian_np = hessian_assembly(inputs_np)
        hessian = torch.from_numpy(hessian_np.copy())
        state_rows = theta_corpus(case_id, teacher_theta)

        # Full-K SHO is quadratic, so H and its spectral state are constant
        # over theta for a trajectory.  Check every gradient's finite state,
        # evaluate LAPACK once per unique H, then account for every saved state.
        all_finite = True
        for _, theta in state_rows:
            _, gradient_np = native_gradient(
                torch.from_numpy(inputs_np.copy()),
                torch.from_numpy(targets_np.copy()),
                theta,
            )
            finite = bool(np.isfinite(gradient_np).all() and np.isfinite(hessian_np).all())
            finite_gradient_checks += int(finite)
            nonfinite_gradient_checks += int(not finite)
            all_finite &= finite
        if not all_finite:
            raise RuntimeError(f"unexpected nonfinite derivative in SHO case {case_id}")

        result = spectral.analyze(torch.zeros(4, dtype=torch.float64), hessian)
        teacher = teacher_direct(result.finite, result.minimum_eigenvalue, result.condition)
        student = student_direct(router, result.finite, result.positive, result.condition_ok)
        count = len(state_rows)
        states += count
        unique_hessians += 1
        teacher_counts["DIRECT" if teacher else "DAMPED"] += count
        student_counts["DIRECT" if student else "DAMPED"] += count
        route_mismatches += int(teacher != student) * count
        cases.append({
            "case_id": case_id,
            "states": count,
            "teacher_route": "DIRECT" if teacher else "DAMPED",
            "student_route": "DIRECT" if student else "DAMPED",
            "minimum_eigenvalue": result.minimum_eigenvalue,
            "condition": result.condition,
        })
    return {
        "states": states,
        "unique_hessians": unique_hessians,
        "finite_gradient_checks": finite_gradient_checks,
        "nonfinite_gradient_checks": nonfinite_gradient_checks,
        "teacher_counts": dict(teacher_counts),
        "student_counts": dict(student_counts),
        "route_mismatches": route_mismatches,
        "exact_states": states - route_mismatches,
        "cases": cases,
    }


def curvature_controls(router: curvature.CurvatureRouter,
                       spectral: N3EigenspectrumConditionAssembly) -> dict:
    zero_gradient = torch.zeros(4, dtype=torch.float64)
    controls = [
        ("condition_exactly_1e12", zero_gradient.clone(),
         torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0e-12]))),
        ("condition_above_1e12", zero_gradient.clone(),
         torch.diag(torch.tensor([1.0, 1.0, 1.0, np.nextafter(1.0e-12, 0.0)]))),
        ("zero_eigenvalue", zero_gradient.clone(),
         torch.diag(torch.tensor([1.0, 0.5, 0.25, 0.0]))),
        ("indefinite", zero_gradient.clone(),
         torch.diag(torch.tensor([1.0, 0.5, 0.25, -0.125]))),
        ("nan_gradient", zero_gradient.clone(), torch.eye(4)),
        ("inf_gradient", zero_gradient.clone(), torch.eye(4)),
        ("nan_hessian", zero_gradient.clone(), torch.eye(4)),
        ("inf_hessian", zero_gradient.clone(), torch.eye(4)),
    ]
    controls[4][1][0] = math.nan
    controls[5][1][0] = math.inf
    controls[6][2][0, 0] = math.nan
    controls[7][2][0, 0] = math.inf
    rows = []
    for name, gradient_value, hessian_value in controls:
        before = (spectral.svd_calls, spectral.eigh_calls)
        result = spectral.analyze(gradient_value, hessian_value)
        after = (spectral.svd_calls, spectral.eigh_calls)
        teacher = teacher_direct(result.finite, result.minimum_eigenvalue, result.condition)
        student = student_direct(router, result.finite, result.positive, result.condition_ok)
        rows.append({
            "name": name,
            "finite": result.finite,
            "minimum_eigenvalue": result.minimum_eigenvalue,
            "condition": result.condition,
            "positive": result.positive,
            "condition_ok": result.condition_ok,
            "teacher_route": "DIRECT" if teacher else "DAMPED",
            "student_route": "DIRECT" if student else "DAMPED",
            "exact": teacher == student,
            "solver_calls": [after[0] - before[0], after[1] - before[1]],
        })
    return {
        "cases": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "mismatches": sum(not row["exact"] for row in rows),
        "boundary_equality": next(row for row in rows if row["name"] == "condition_exactly_1e12"),
        "rows": rows,
    }


def teacher_damping(scale: float, minimum: float, direct: bool,
                    attempt: int) -> tuple[float, float]:
    initial = 0.0 if direct else max(
        scale * production.DAMPING_RELATIVE_FLOOR,
        -minimum + scale * production.DAMPING_RELATIVE_FLOOR,
    )
    if attempt == 0 and direct:
        return initial, 0.0
    base = initial if initial > 0.0 else scale * production.DAMPING_RELATIVE_FLOOR
    exponent = attempt if not direct else attempt - 1
    return initial, base * (10.0 ** exponent)


def damping_replay(router: curvature.CurvatureRouter,
                   spectral: N3EigenspectrumConditionAssembly,
                   controls: dict) -> dict:
    gradient.OPS = gradient.LearnedArithmetic()
    damping = DampingAssembly(ScalarArithmetic())
    rows = []
    initial_mismatches = value_mismatches = 0
    # The control matrices cover both DIRECT and DAMPED routes and the exact
    # condition equality.  Nonfinite rows exit before damping and are omitted.
    for control in controls["rows"]:
        if not control["finite"]:
            continue
        scale_source = next(
            matrix for name, matrix in (
                ("condition_exactly_1e12", torch.diag(torch.tensor([1., 1., 1., 1e-12]))),
                ("condition_above_1e12", torch.diag(torch.tensor(
                    [1., 1., 1., np.nextafter(1e-12, 0.)]))),
                ("zero_eigenvalue", torch.diag(torch.tensor([1., .5, .25, 0.]))),
                ("indefinite", torch.diag(torch.tensor([1., .5, .25, -.125]))),
            ) if name == control["name"]
        )
        result = spectral.analyze(torch.zeros(4), scale_source)
        direct = student_direct(router, result.finite, result.positive, result.condition_ok)
        learned_initial = damping.initial(result.scale, result.minimum_eigenvalue, direct)
        values = []
        for attempt in range(production.MAX_DAMPING_TRIALS):
            expected_initial, expected_value = teacher_damping(
                result.scale, result.minimum_eigenvalue, direct, attempt)
            learned_value = damping.value(result.scale, learned_initial, direct, attempt)
            initial_mismatches += int(not bits_equal(expected_initial, learned_initial))
            value_mismatches += int(not bits_equal(expected_value, learned_value))
            values.append({
                "attempt": attempt,
                "teacher": expected_value,
                "student": learned_value,
                "exact": bits_equal(expected_value, learned_value),
            })
        rows.append({
            "name": control["name"],
            "route": "DIRECT" if direct else "DAMPED",
            "teacher_initial": teacher_damping(
                result.scale, result.minimum_eigenvalue, direct, 0)[0],
            "student_initial": learned_initial,
            "values": values,
        })
    return {
        "controls": len(rows),
        "attempt_values": sum(len(row["values"]) for row in rows),
        "initial_bit_mismatches": initial_mismatches,
        "retry_value_bit_mismatches": value_mismatches,
        "rows": rows,
        "assembly": damping,
    }


def first_post_curvature_divergence(router: curvature.CurvatureRouter,
                                    spectral: N3EigenspectrumConditionAssembly,
                                    damping: DampingAssembly) -> dict:
    prior = json.loads(PRIOR_HYBRID_REPORT.read_text())
    row = next(item for item in prior["archive_n2_replay"]
               if item["system"] == "full_k_sho")
    case_id = int(row["case_id"])
    trajectory, _, _, _, _, _, _ = readiness.case_data(specification(case_id))
    theta = np.asarray(row["n2_endpoint"], np.float64)
    derivative = teacher_details(trajectory[:-1], trajectory[1:], theta)
    gradient_value = torch.from_numpy(derivative["gradient"].copy())
    hessian_value = torch.from_numpy(derivative["symmetric"].copy())
    result = spectral.analyze(gradient_value, hessian_value)
    direct = student_direct(router, result.finite, result.positive, result.condition_ok)

    expected_initial, expected_damping = teacher_damping(
        result.scale, result.minimum_eigenvalue, direct, 0)
    student_initial = damping.initial(result.scale, result.minimum_eigenvalue, direct)
    student_damping = damping.value(result.scale, student_initial, direct, 0)
    damping_initial_metric = metric([expected_initial], [student_initial])
    damping_value_metric = metric([expected_damping], [student_damping])
    if damping_initial_metric["bit_mismatches"] or damping_value_metric["bit_mismatches"]:
        return {
            "operation": "damping arithmetic/retry values",
            "category": "B",
            "teacher_initial": expected_initial,
            "student_initial": student_initial,
            "teacher_value": expected_damping,
            "student_value": student_damping,
            "initial_metric": damping_initial_metric,
            "value_metric": damping_value_metric,
        }

    identity = torch.eye(len(gradient_value), dtype=torch.float64)
    teacher_matrix = hessian_value + expected_damping * identity
    teacher_rhs = -gradient_value
    student_matrix = damping.matrix(hessian_value, student_damping)
    student_rhs = damping.rhs(gradient_value)
    matrix_metric = metric(teacher_matrix.numpy(), student_matrix.detach().numpy())
    rhs_metric = metric(teacher_rhs.numpy(), student_rhs.detach().numpy())
    if matrix_metric["bit_mismatches"] or rhs_metric["bit_mismatches"]:
        return {
            "operation": "damped matrix and RHS assembly",
            "suboperation": "RHS unary negation",
            "category": "B",
            "system": "full_k_sho",
            "case_id": case_id,
            "theta": theta.tolist(),
            "route": "DIRECT" if direct else "DAMPED",
            "damping": student_damping,
            "teacher_matrix": teacher_matrix.tolist(),
            "student_matrix": student_matrix.tolist(),
            "matrix_metric": matrix_metric,
            "teacher_rhs": teacher_rhs.tolist(),
            "student_rhs": student_rhs.tolist(),
            "rhs_metric": rhs_metric,
            "type": "signed-zero/implementation-order",
            "diagnosis": (
                "production unary negation maps each +0 gradient entry to -0; the locked "
                "subtraction assembly computes +0 - +0 as +0. Nonzero RHS entries and the "
                "entire damped matrix remain bit-exact"
            ),
            "later_operations_invoked": False,
        }
    return {"operation": None}


def run() -> dict:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected:
        raise RuntimeError("checkpoint inventory changed before replay")
    curvature_hash_before = sha256(CURVATURE_CHECKPOINT)
    spectral_report = json.loads(SPECTRAL_REPORT.read_text())
    if spectral_report["status"] != "N3_EIGENSPECTRUM_CONDITION_TEACHER_EXACT":
        raise RuntimeError("exact spectral gate has not passed")

    router = load_locked_router()
    spectral = N3EigenspectrumConditionAssembly(production.CONDITION_LIMIT)
    corpus = curvature_corpus(router, spectral)
    controls = curvature_controls(router, spectral)
    curvature_exact = corpus["route_mismatches"] == 0 and controls["mismatches"] == 0
    if not curvature_exact:
        divergence = {
            "operation": "DIRECT/DAMPED curvature route",
            "category": "A",
            "corpus_route_mismatches": corpus["route_mismatches"],
            "control_route_mismatches": controls["mismatches"],
        }
        damping = None
    else:
        damping_replay_result = damping_replay(router, spectral, controls)
        damping = damping_replay_result.pop("assembly")
        if (damping_replay_result["initial_bit_mismatches"] or
                damping_replay_result["retry_value_bit_mismatches"]):
            divergence = {
                "operation": "damping arithmetic/retry values",
                "category": "B",
                "mismatch_count": (
                    damping_replay_result["initial_bit_mismatches"]
                    + damping_replay_result["retry_value_bit_mismatches"]
                ),
            }
        else:
            divergence = first_post_curvature_divergence(router, spectral, damping)

    integrated = []
    if curvature_exact:
        integrated.append({
            "operation": "DIRECT/DAMPED curvature route",
            "category": "A",
            "implementation": "locked 6-parameter Boolean AND router plus nonfinite early return",
            "status": "EXACT",
        })
    if curvature_exact and divergence.get("operation") != "damping arithmetic/retry values":
        integrated.append({
            "operation": "damping arithmetic/retry values",
            "category": "B",
            "implementation": "locked arithmetic damping construction and powers-of-ten retry",
            "status": "EXACT",
        })

    report = {
        "status": "STOPPED_FIRST_A_B_DIVERGENCE" if divergence.get("operation") else "REPLAY_PREFIX_EXACT",
        "scope": (
            "resume hybrid integration at curvature; fail closed before dense solve, box-QP, "
            "descent, backtracking, or later N3 operations"
        ),
        "curvature_router": {
            "checkpoint": str(CURVATURE_CHECKPOINT),
            "checkpoint_sha256": curvature_hash_before,
            "architecture": "float64 Linear(2,2), 6 reused parameters",
            "new_parameters": 0,
            "teacher_semantics": (
                "DIRECT iff finite AND minimum_eigenvalue>0 AND condition<=1e12"
            ),
            "corpus": corpus,
            "controls": controls,
            "exact": curvature_exact,
        },
        "damping_replay": damping_replay_result if curvature_exact else None,
        "integrated_A_B_operations_after_spectrum": integrated,
        "first_divergence": divergence,
        "updated_integration_boundary": {
            "last_exact_operation": integrated[-1]["operation"] if integrated else (
                "Hessian eigenspectrum and condition"
            ),
            "first_unintegrated_operation": divergence.get("operation"),
            "later_operations_not_invoked": [
                "dense linear solve",
                "dimension-general box-QP active-set direction",
                "descent validity",
                "power-of-two backtracking schedule",
                "candidate bounds route",
                "strict step acceptance",
                "genuine numerical STOP/CONTINUE routes",
                "FLOAT64_STAGNATION_TAIL stop",
            ],
        },
        "checkpoint_hashes_unchanged": (
            checkpoint_hashes() == expected
            and sha256(CURVATURE_CHECKPOINT) == curvature_hash_before
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "curvature": {
            "teacher_counts": corpus["teacher_counts"],
            "student_counts": corpus["student_counts"],
            "route_mismatches": corpus["route_mismatches"],
            "controls": controls["cases"],
            "control_mismatches": controls["mismatches"],
        },
        "integrated": integrated,
        "first_divergence": divergence,
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))
    return report


if __name__ == "__main__":
    run()

