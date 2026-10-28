#!/usr/bin/env python3
"""Integrate and validate only the live N3 DIRECT/DAMPED curvature route."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3CurvatureRouteBinding,
    HybridN3GradientBinding,
    HybridN3HessianBinding,
    Log,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_curvature_route_binding")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
CONDITION_LIMIT = 1.0e12
CASES = {
    "full_k_sho": (0, 17, 51),
    "reflection_ball": (215, 253, 265),
    "kepler5": (192, 193, 231),
}


def case_spec(system: str, case_id: int) -> dict:
    rows = json.loads(readiness.MANIFEST.read_text())["cases"]
    return next(
        row for row in rows
        if row["system"] == system and int(row["case_id"]) == case_id
    )


def sampled_thetas(system: str, case_id: int, teacher: np.ndarray):
    result = json.loads(
        (readiness.CASES / system / f"case_{case_id:05d}" / "result.json").read_text()
    )
    values = np.load(result["successful_branch"]["trace_npz"])["n3_theta"]
    indexes = sorted(set((0, len(values) // 2, len(values) - 1)))
    candidates = [(f"saved_n3_{index}", values[index].copy()) for index in indexes]
    candidates.append(("teacher", np.asarray(teacher, np.float64).copy()))
    answer, seen = [], set()
    for source, theta in candidates:
        key = theta.tobytes()
        if key not in seen:
            answer.append((source, theta))
            seen.add(key)
    return answer


def teacher_route(spectrum) -> str:
    direct = bool(
        spectrum.finite
        and spectrum.minimum_eigenvalue is not None
        and spectrum.minimum_eigenvalue > 0.0
        and spectrum.condition <= CONDITION_LIMIT
    )
    return "DIRECT" if direct else "DAMPED"


def controls(binding: HybridN3CurvatureRouteBinding, log: Log):
    zero = torch.zeros(4, dtype=torch.float64)
    next_condition = np.nextafter(np.float64(CONDITION_LIMIT), np.float64(np.inf))
    controls = [
        ("ordinary_direct", zero.clone(), torch.diag(torch.tensor([1.0, 0.5, 0.25, 0.125])), "DIRECT"),
        ("indefinite_hessian", zero.clone(), torch.diag(torch.tensor([1.0, 0.5, 0.25, -0.125])), "DAMPED"),
        ("zero_minimum_eigenvalue", zero.clone(), torch.diag(torch.tensor([1.0, 0.5, 0.25, 0.0])), "DAMPED"),
        ("singular_condition", zero.clone(), torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0])), "DAMPED"),
        ("condition_exactly_1e12", zero.clone(), torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0e-12])), "DIRECT"),
        # 1 / nextafter(1e12,+inf) produces the requested adjacent condition.
        ("condition_nextafter_above_1e12", zero.clone(), torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0 / next_condition])), "DAMPED"),
    ]
    nan_gradient = zero.clone(); nan_gradient[0] = math.nan
    inf_hessian = torch.eye(4, dtype=torch.float64); inf_hessian[0, 0] = math.inf
    controls.extend([
        ("nonfinite_gradient", nan_gradient, torch.eye(4, dtype=torch.float64), "DAMPED"),
        ("nonfinite_hessian", zero.clone(), inf_hessian, "DAMPED"),
    ])
    rows = []
    for name, gradient, hessian, expected in controls:
        spectrum, student = binding.evaluate(gradient, hessian, log)
        reference = teacher_route(spectrum)
        rows.append({
            "name": name,
            "expected_route": expected,
            "teacher_route": reference,
            "student_route": student["route"],
            "finite": spectrum.finite,
            "minimum_eigenvalue": spectrum.minimum_eigenvalue,
            "condition": spectrum.condition,
            "positive_predicate": spectrum.positive,
            "condition_ok_predicate": spectrum.condition_ok,
            "exact": reference == student["route"] == expected,
        })
    return rows


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    gradient_binding = HybridN3GradientBinding()
    hessian_binding = HybridN3HessianBinding(gradient_binding)
    curvature_binding = HybridN3CurvatureRouteBinding(CONDITION_LIMIT)
    log = Log()
    by_system = {}
    states = []
    route_mismatches = 0
    teacher_counts = Counter()

    for system, case_ids in CASES.items():
        local_counts = Counter()
        local_rows = []
        for case_id in case_ids:
            trajectory, _, _, _, _, _, teacher = readiness.case_data(
                case_spec(system, case_id)
            )
            for source, theta in sampled_thetas(system, case_id, teacher):
                derivatives = hessian_binding.evaluate(system, trajectory, theta, log)
                spectrum, student = curvature_binding.evaluate(
                    derivatives.gradient, derivatives.symmetric_hessian, log
                )
                reference = teacher_route(spectrum)
                mismatch = reference != student["route"]
                route_mismatches += int(mismatch)
                teacher_counts[reference] += 1
                local_counts[reference] += 1
                local_counts["mismatches"] += int(mismatch)
                row = {
                    "system": system,
                    "case_id": case_id,
                    "source": source,
                    "teacher_route": reference,
                    "student_route": student["route"],
                    "finite": spectrum.finite,
                    "minimum_eigenvalue": spectrum.minimum_eigenvalue,
                    "condition": spectrum.condition,
                    "positive_predicate": spectrum.positive,
                    "condition_ok_predicate": spectrum.condition_ok,
                    "exact": not mismatch,
                }
                local_rows.append(row)
                states.append(row)
        by_system[system] = {
            "states": len(local_rows),
            "teacher_DIRECT": int(local_counts["DIRECT"]),
            "teacher_DAMPED": int(local_counts["DAMPED"]),
            "routing_mismatches": int(local_counts["mismatches"]),
        }

    boundary = controls(curvature_binding, log)
    boundary_mismatches = sum(not row["exact"] for row in boundary)
    for row in boundary:
        teacher_counts[row["teacher_route"]] += 1

    after = checkpoint_hashes()
    fallback_calls = curvature_binding.native_routing_fallback_calls
    exact = bool(
        route_mismatches == 0
        and boundary_mismatches == 0
        and fallback_calls == 0
        and before == after == expected_hashes
    )
    report = {
        "operation": "N3 DIRECT / DAMPED curvature route",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "teacher_semantics": "DIRECT iff finite AND min_eigenvalue > 0 AND condition <= 1e12",
            "student_semantics": "two calls to locked Boolean conjunction/router assembly",
            "retained_d_oracle": "Torch/Accelerate eigenspectrum and condition calculation",
            "new_parameter_count": 0,
            "damping_construction_integrated": False,
        },
        "binding_invocation_count": curvature_binding.invocations,
        "logged_binding_invocation_count": int(log.counts["N3 DIRECT/DAMPED curvature route"]),
        "native_routing_fallback_calls": fallback_calls,
        "teacher_counts_including_controls": dict(teacher_counts),
        "saved_state_route_mismatches": route_mismatches,
        "equality_boundary_mismatches": boundary_mismatches,
        "by_system": by_system,
        "boundary_controls": boundary,
        "saved_states": states,
        "retained_spectral_oracle_calls": {
            "svd": curvature_binding.spectral_oracle.svd_calls,
            "eigh": curvature_binding.spectral_oracle.eigh_calls,
        },
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_hashes,
        },
        "next_unresolved_ab_binding": "N3 damping construction",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_curvature_route_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["binding_invocation_count"],
        "teacher_counts": report["teacher_counts_including_controls"],
        "by_system": by_system,
        "boundary_controls": boundary,
        "fallback_calls": fallback_calls,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
