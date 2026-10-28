#!/usr/bin/env python3
"""Integrate and validate only N3 damping construction and retry scalars."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3CurvatureRouteBinding,
    HybridN3DampingBinding,
    HybridN3GradientBinding,
    HybridN3HessianBinding,
    Log,
)
from experiments.resume_hybrid_student_at_curvature import teacher_damping
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_damping_binding")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
SAVED_CASES = {
    "full_k_sho": (0,),
    "reflection_ball": (215,),
    "kepler5": (228,),
}


def bits_equal(left: float, right: float) -> bool:
    return bool(
        np.float64(left).view(np.uint64) == np.float64(right).view(np.uint64)
    )


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


def replay_values(binding, scale, minimum, direct, log):
    initial = binding.initial(scale, minimum, direct, log)
    initial_teacher, _ = teacher_damping(scale, minimum, direct, 0)
    rows = []
    for attempt in range(production.MAX_DAMPING_TRIALS):
        expected_initial, expected = teacher_damping(
            scale, minimum, direct, attempt
        )
        actual = binding.value(scale, initial, direct, attempt, log)
        rows.append({
            "attempt": attempt,
            "teacher_initial": expected_initial,
            "teacher_value": expected,
            "student_value": actual,
            "exact": bits_equal(expected, actual),
            "student_bits": f"0x{int(np.float64(actual).view(np.uint64)):016x}",
        })
    return {
        "scale": scale,
        "minimum_eigenvalue": minimum,
        "route": "DIRECT" if direct else "DAMPED",
        "teacher_initial": initial_teacher,
        "student_initial": initial,
        "initial_exact": bits_equal(initial_teacher, initial),
        "initial_bits": f"0x{int(np.float64(initial).view(np.uint64)):016x}",
        "retry_values": rows,
        "retry_mismatches": sum(not row["exact"] for row in rows),
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    gradient = HybridN3GradientBinding()
    hessian = HybridN3HessianBinding(gradient)
    curvature = HybridN3CurvatureRouteBinding()
    damping = HybridN3DampingBinding(curvature)
    log = Log()
    by_system = {}
    saved = []
    initial_mismatches = retry_mismatches = trace_mismatches = 0

    for system, case_ids in SAVED_CASES.items():
        local = Counter()
        for case_id in case_ids:
            trajectory, _, _, _, _, _, _ = readiness.case_data(
                case_spec(system, case_id)
            )
            for index, theta, trace_damping in saved_rows(system, case_id):
                derivatives = hessian.evaluate(system, trajectory, theta, log)
                spectrum, route = curvature.evaluate(
                    derivatives.gradient, derivatives.symmetric_hessian, log
                )
                row = replay_values(
                    damping, spectrum.scale, spectrum.minimum_eigenvalue,
                    route["direct"], log,
                )
                schedule_values = [entry["student_value"] for entry in row["retry_values"]]
                trace_matches = [
                    attempt for attempt, value in enumerate(schedule_values)
                    if bits_equal(value, trace_damping)
                ]
                row.update({
                    "system": system,
                    "case_id": case_id,
                    "n3_trace_index": index,
                    "trace_damping": trace_damping,
                    "trace_matching_attempts": trace_matches,
                })
                initial_mismatches += int(not row["initial_exact"])
                retry_mismatches += row["retry_mismatches"]
                trace_mismatches += int(not trace_matches)
                local[row["route"]] += 1
                local["states"] += 1
                local["initial_mismatches"] += int(not row["initial_exact"])
                local["retry_mismatches"] += row["retry_mismatches"]
                local["trace_mismatches"] += int(not trace_matches)
                saved.append(row)
        by_system[system] = dict(local)

    tiny = float(torch.finfo(torch.float64).tiny)
    controls_spec = [
        ("direct_positive", 2.0, 0.5, True),
        ("damped_indefinite", 1.0, -0.125, False),
        ("damped_zero_minimum", 1.0, 0.0, False),
        ("damped_negative_zero_minimum", 1.0, -0.0, False),
        ("damped_positive_minimum_bad_condition", 1.0, 0.25, False),
        ("direct_tiny_scale", tiny, tiny, True),
    ]
    controls = []
    for name, scale, minimum, direct in controls_spec:
        row = replay_values(damping, scale, minimum, direct, log)
        row["name"] = name
        controls.append(row)
        initial_mismatches += int(not row["initial_exact"])
        retry_mismatches += row["retry_mismatches"]

    direct_zero_controls = [
        row for row in controls
        if row["route"] == "DIRECT"
    ]
    signed_zero_mismatches = sum(
        row["initial_bits"] != "0x0000000000000000"
        or row["retry_values"][0]["student_bits"] != "0x0000000000000000"
        for row in direct_zero_controls
    )

    after = checkpoint_hashes()
    native_fallbacks = damping.native_damping_fallback_calls
    exact = bool(
        initial_mismatches == 0
        and retry_mismatches == 0
        and trace_mismatches == 0
        and signed_zero_mismatches == 0
        and native_fallbacks == 0
        and before == after == expected_hashes
    )
    report = {
        "operation": "N3 damping construction",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "direct_initial": "positive +0.0",
            "damped_initial": "max(scale*1e-12, -minimum_eigenvalue + scale*1e-12)",
            "retry_schedule": "production powers of ten, attempts 0..13",
            "assemblies": [
                "locked multiply/add/subtract", "locked unary sign-bit negation",
                "locked strict comparison/select", "locked Boolean routing",
            ],
            "new_parameter_count": 0,
            "damped_matrix_or_rhs_integrated": False,
        },
        "binding_invocation_count": (
            damping.initial_invocations + damping.retry_invocations
        ),
        "initial_binding_invocations": damping.initial_invocations,
        "retry_binding_invocations": damping.retry_invocations,
        "logged_initial_invocations": int(log.counts["N3 damping construction"]),
        "logged_retry_invocations": int(log.counts["N3 damping retry value"]),
        "native_damping_fallback_calls": native_fallbacks,
        "initial_value_mismatches": initial_mismatches,
        "retry_value_mismatches": retry_mismatches,
        "saved_trace_value_mismatches": trace_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "by_system": by_system,
        "saved_states": saved,
        "controls": controls,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_hashes,
        },
        "next_unresolved_ab_binding": "N3 damped matrix and RHS construction",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_damping_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "binding_invocations": report["binding_invocation_count"],
        "initial_mismatches": initial_mismatches,
        "retry_mismatches": retry_mismatches,
        "trace_mismatches": trace_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "native_fallbacks": native_fallbacks,
        "by_system": by_system,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
