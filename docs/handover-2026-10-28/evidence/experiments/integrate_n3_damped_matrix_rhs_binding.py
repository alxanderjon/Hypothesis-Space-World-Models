#!/usr/bin/env python3
"""Integrate and validate only N3 damped-matrix and RHS construction."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3DampedMatrixRHSBinding,
    HybridN3GradientBinding,
    HybridN3HessianBinding,
    Log,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_damped_matrix_rhs_binding")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
SAVED_CASES = {
    "full_k_sho": (0,),
    "reflection_ball": (215,),
    "kepler5": (228,),
}


def metric(expected, actual) -> dict:
    left = np.asarray(expected, np.float64)
    right = np.asarray(actual, np.float64)
    if left.shape != right.shape:
        return {
            "values": int(left.size), "shape_match": False,
            "bit_mismatches": int(max(left.size, right.size)),
            "signed_zero_mismatches": 0,
        }
    both_zero = (left == 0.0) & (right == 0.0)
    return {
        "values": int(left.size),
        "shape_match": True,
        "bit_mismatches": int(np.count_nonzero(
            left.view(np.uint64) != right.view(np.uint64)
        )),
        "signed_zero_mismatches": int(np.count_nonzero(
            both_zero & (np.signbit(left) != np.signbit(right))
        )),
    }


def case_spec(system: str, case_id: int) -> dict:
    rows = json.loads(readiness.MANIFEST.read_text())["cases"]
    return next(
        row for row in rows
        if row["system"] == system and int(row["case_id"]) == case_id
    )


def saved_rows(system: str, case_id: int):
    result = json.loads(
        (readiness.CASES / system / f"case_{case_id:05d}" / "result.json").read_text()
    )
    trace = np.load(result["successful_branch"]["trace_npz"])
    theta = trace["n3_theta"]
    damping = trace["n3_damping"]
    indexes = sorted(set((0, len(theta) - 1)))
    return [(index, theta[index].copy(), float(damping[index])) for index in indexes]


def compare_construction(binding, hessian, gradient, damping, log):
    identity = torch.eye(len(gradient), dtype=torch.float64)
    teacher_scaled = torch.tensor(damping, dtype=torch.float64) * identity
    teacher_matrix = hessian + teacher_scaled
    teacher_rhs = -gradient
    student = binding.construct(hessian, gradient, damping, log)
    return {
        "dimension": len(gradient),
        "damping": float(damping),
        "route": "DIRECT" if float(damping) == 0.0 else "DAMPED",
        "hessian_input": metric(hessian.numpy(), hessian.numpy()),
        "lambda_input": metric([damping], [student["damping"]]),
        "identity": metric(identity.numpy(), student["identity"].numpy()),
        "damped_identity": metric(
            teacher_scaled.numpy(), student["damped_identity"].numpy()
        ),
        "damped_matrix": metric(
            teacher_matrix.numpy(), student["damped_matrix"].numpy()
        ),
        "gradient_input": metric(gradient.numpy(), gradient.numpy()),
        "rhs": metric(teacher_rhs.numpy(), student["rhs"].numpy()),
    }


def row_mismatches(row):
    stages = (
        "hessian_input", "lambda_input", "identity", "damped_identity",
        "damped_matrix", "gradient_input", "rhs",
    )
    return sum(row[name]["bit_mismatches"] for name in stages)


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    gradient_binding = HybridN3GradientBinding()
    hessian_binding = HybridN3HessianBinding(gradient_binding)
    system_binding = HybridN3DampedMatrixRHSBinding()
    log = Log()
    saved = []
    by_system = {}

    for system, case_ids in SAVED_CASES.items():
        local = Counter()
        for case_id in case_ids:
            trajectory, _, _, _, _, _, _ = readiness.case_data(
                case_spec(system, case_id)
            )
            for index, theta, damping in saved_rows(system, case_id):
                derivatives = hessian_binding.evaluate(
                    system, trajectory, theta, log
                )
                row = compare_construction(
                    system_binding, derivatives.symmetric_hessian,
                    derivatives.gradient, damping, log,
                )
                row.update({
                    "system": system,
                    "case_id": case_id,
                    "n3_trace_index": index,
                })
                mismatch = row_mismatches(row)
                row["mismatches"] = mismatch
                local["states"] += 1
                local[row["route"]] += 1
                local["mismatches"] += mismatch
                saved.append(row)
        by_system[system] = dict(local)

    # Explicit shape/zero controls exercise direct zero damping, positive
    # damping, zero off-diagonals, and both signs of floating-point zero.
    h4 = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    h4[0, 1] = -0.0
    h4[1, 0] = -0.0
    g4 = torch.tensor([0.0, -0.0, 1.0, -1.0])
    h5 = torch.zeros((5, 5), dtype=torch.float64)
    h5.diagonal().copy_(torch.tensor([1.0, 0.5, 0.25, 0.125, 0.0625]))
    h5[0, 4] = -0.0
    h5[4, 0] = -0.0
    g5 = torch.tensor([-0.0, 0.0, -2.0, 3.0, -4.0])
    controls = [
        {"name": "direct_lambda_positive_zero", **compare_construction(
            system_binding, h4, g4, 0.0, log
        )},
        {"name": "positive_damping_zero_offdiagonals", **compare_construction(
            system_binding, h5, g5, 1.0e-12, log
        )},
    ]
    for row in controls:
        row["mismatches"] = row_mismatches(row)

    all_rows = saved + controls
    matrix_entries = sum(row["damped_matrix"]["values"] for row in all_rows)
    rhs_entries = sum(row["rhs"]["values"] for row in all_rows)
    matrix_mismatches = sum(
        row["damped_identity"]["bit_mismatches"]
        + row["damped_matrix"]["bit_mismatches"]
        for row in all_rows
    )
    rhs_mismatches = sum(row["rhs"]["bit_mismatches"] for row in all_rows)
    signed_zero_mismatches = sum(
        row[stage]["signed_zero_mismatches"]
        for row in all_rows
        for stage in ("damped_identity", "damped_matrix", "rhs")
    )
    shape_mismatches = sum(
        not row[stage]["shape_match"]
        for row in all_rows
        for stage in ("damped_identity", "damped_matrix", "rhs")
    )
    after = checkpoint_hashes()
    fallback_calls = system_binding.native_construction_fallback_calls
    exact = bool(
        matrix_mismatches == 0
        and rhs_mismatches == 0
        and signed_zero_mismatches == 0
        and shape_mismatches == 0
        and fallback_calls == 0
        and before == after == expected_hashes
    )
    report = {
        "operation": "N3 damped matrix and RHS construction",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "damped_matrix": "locked add(H, locked multiply(lambda, I))",
            "rhs": "locked sign-bit unary negation(gradient)",
            "identity_and_storage": "deterministic dimension-general scaffolding",
            "new_parameter_count": 0,
            "dense_solve_invocations": 0,
        },
        "binding_invocation_count": system_binding.invocations,
        "logged_binding_invocation_count": int(
            log.counts["N3 damped matrix and RHS construction"]
        ),
        "matrix_entries_checked": matrix_entries,
        "rhs_entries_checked": rhs_entries,
        "native_construction_fallback_calls": fallback_calls,
        "matrix_mismatches": matrix_mismatches,
        "rhs_mismatches": rhs_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "shape_or_ordering_mismatches": shape_mismatches,
        "by_system": by_system,
        "saved_states": saved,
        "controls": controls,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_hashes,
        },
        "next_unresolved_ab_binding": "N3 dense linear solve",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_damped_matrix_rhs_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["binding_invocation_count"],
        "matrix_entries_checked": matrix_entries,
        "rhs_entries_checked": rhs_entries,
        "matrix_mismatches": matrix_mismatches,
        "rhs_mismatches": rhs_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "shape_mismatches": shape_mismatches,
        "native_fallbacks": fallback_calls,
        "by_system": by_system,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
