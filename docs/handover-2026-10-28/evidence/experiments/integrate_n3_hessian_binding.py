#!/usr/bin/env python3
"""Force and validate only the live symmetric N3 Hessian binding."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3GradientBinding, HybridN3HessianBinding, Log,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_hessian_binding")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
CASES = {
    "full_k_sho": (0, 17, 51),
    "reflection_ball": (215, 253, 265),
    "kepler5": (192, 193, 231),
}


def metric(expected, actual) -> dict:
    left = np.asarray(expected, np.float64).reshape(-1)
    right = np.asarray(actual, np.float64).reshape(-1)
    unequal = left.view(np.uint64) != right.view(np.uint64)
    return {
        "values": int(left.size),
        "bit_mismatches": int(np.count_nonzero(unequal)),
        "sign_mismatches": int(np.count_nonzero(
            np.signbit(left) != np.signbit(right)
        )),
        "zero_nonzero_mismatches": int(np.count_nonzero(
            (left == 0.0) != (right == 0.0)
        )),
        "signed_zero_mismatches": int(np.count_nonzero(
            (left == 0.0) & (right == 0.0)
            & (np.signbit(left) != np.signbit(right))
        )),
        "max_absolute_error": (
            float(np.max(np.abs(left - right))) if left.size else 0.0
        ),
    }


def add(total: Counter, prefix: str, row: dict):
    for key in ("values", "bit_mismatches", "sign_mismatches",
                "zero_nonzero_mismatches", "signed_zero_mismatches"):
        total[f"{prefix}_{key}"] += int(row[key])
    total[f"{prefix}_max_absolute_error"] = max(
        float(total[f"{prefix}_max_absolute_error"]), row["max_absolute_error"],
    )


def spec(system: str, case_id: int) -> dict:
    rows = json.loads(readiness.MANIFEST.read_text())["cases"]
    return next(row for row in rows
                if row["system"] == system and int(row["case_id"]) == case_id)


def sampled_thetas(system: str, case_id: int, teacher: np.ndarray):
    result = json.loads((readiness.CASES / system / f"case_{case_id:05d}" /
                         "result.json").read_text())
    values = np.load(result["successful_branch"]["trace_npz"])["n3_theta"]
    indexes = sorted(set((0, len(values) // 2, len(values) - 1)))
    rows = [(f"saved_n3_{index}", values[index].copy()) for index in indexes]
    rows.append(("teacher", np.asarray(teacher, np.float64).copy()))
    unique, seen = [], set()
    for source, theta in rows:
        if theta.tobytes() not in seen:
            unique.append((source, theta)); seen.add(theta.tobytes())
    return unique


def teacher_derivatives(torch_objective, theta_np: np.ndarray):
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
    loss = torch_objective(theta)
    gradient = torch.autograd.grad(
        loss, theta, create_graph=True, retain_graph=True,
    )[0]
    rows = tuple(
        torch.autograd.grad(
            gradient[index], theta, retain_graph=True,
        )[0]
        for index in range(len(theta))
    )
    raw = torch.stack(rows)
    transpose = raw.T
    summed = raw + transpose
    symmetric = 0.5 * summed
    return {
        "loss": loss.detach(), "gradient": gradient.detach(),
        "rows": tuple(row.detach() for row in rows),
        "raw": raw.detach(), "transpose": transpose.detach(),
        "summed": summed.detach(), "symmetric": symmetric.detach(),
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    gradient_binding = HybridN3GradientBinding()
    hessian_binding = HybridN3HessianBinding(gradient_binding)
    log = Log()
    totals = Counter()
    by_system = {}
    rows = []
    first_mismatch = None
    for system, case_ids in CASES.items():
        local = Counter()
        evaluations = 0
        power_first = power_second = event_calls = 0
        for case_id in case_ids:
            trajectory, _, _, torch_objective, _, _, teacher = readiness.case_data(
                spec(system, case_id)
            )
            for source, theta in sampled_thetas(system, case_id, teacher):
                reference = teacher_derivatives(torch_objective, theta)
                result = hessian_binding.evaluate(
                    system, trajectory, theta, log,
                )
                row_metrics = {
                    "gradient_input": metric(
                        reference["gradient"].numpy(), result.gradient.numpy()),
                    "raw_rows": metric(
                        torch.stack(reference["rows"]).numpy(),
                        torch.stack(result.raw_rows).numpy()),
                    "raw_hessian": metric(
                        reference["raw"].numpy(), result.raw_hessian.numpy()),
                    "transpose": metric(
                        reference["transpose"].numpy(), result.transpose.numpy()),
                    "add_intermediate": metric(
                        reference["summed"].numpy(), result.summed.numpy()),
                    "symmetric_hessian": metric(
                        reference["symmetric"].numpy(),
                        result.symmetric_hessian.numpy()),
                }
                for name, value in row_metrics.items():
                    add(local, name, value); add(totals, name, value)
                power_first += int(result.retained_oracles.get(
                    "production_variable_power_first_adjoint_calls", 0
                ))
                power_second += int(result.retained_oracles.get(
                    "production_variable_power_second_adjoint_calls", 0
                ))
                event_calls += int(result.retained_oracles.get(
                    "deterministic_event_branch_step_calls", 0
                ))
                evaluations += 1
                mismatch = any(value["bit_mismatches"]
                               for value in row_metrics.values())
                row = {
                    "system": system, "case_id": case_id, "source": source,
                    "metrics": row_metrics,
                    "retained_oracles": result.retained_oracles,
                    "exact": not mismatch,
                }
                rows.append(row)
                if mismatch and first_mismatch is None:
                    first_mismatch = row
        by_system[system] = {
            "cases": len(case_ids), "hessian_evaluations": evaluations,
            "raw_hessian_entries_checked": int(local["raw_hessian_values"]),
            "symmetric_hessian_entries_checked": int(
                local["symmetric_hessian_values"]
            ),
            "raw_hessian_bit_mismatches": int(
                local["raw_hessian_bit_mismatches"]
            ),
            "symmetric_hessian_bit_mismatches": int(
                local["symmetric_hessian_bit_mismatches"]
            ),
            "raw_hessian_sign_mismatches": int(
                local["raw_hessian_sign_mismatches"]
            ),
            "raw_hessian_zero_nonzero_mismatches": int(
                local["raw_hessian_zero_nonzero_mismatches"]
            ),
            "symmetric_hessian_sign_mismatches": int(
                local["symmetric_hessian_sign_mismatches"]
            ),
            "symmetric_hessian_zero_nonzero_mismatches": int(
                local["symmetric_hessian_zero_nonzero_mismatches"]
            ),
            "retained_variable_power_first_adjoint_calls": power_first,
            "retained_variable_power_second_adjoint_calls": power_second,
            "retained_event_branch_step_calls": event_calls,
        }
        print(system, json.dumps(by_system[system]), flush=True)

    after = checkpoint_hashes()
    mismatch_total = sum(
        value for key, value in totals.items() if key.endswith("bit_mismatches")
    )
    raw_fallbacks = (
        hessian_binding.assembly.raw_autograd_final_hessian_fallback_calls
    )
    exact = (
        mismatch_total == 0 and raw_fallbacks == 0
        and before == after == expected
    )
    report = {
        "binding": "n3_hessian_binding",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "row_order": "flattened gradient coordinates 0..d-1",
            "symmetrization": "RN(0.5 * RN(raw_H + raw_H.T))",
            "full_k_sho": (
                "recovered FMA forward and ascending-FMA MmBackward "
                "second-adjoint implementation order"
            ),
            "reflection_ball": (
                "locked second-adjoint arithmetic on retained deterministic "
                "collision/event branches"
            ),
            "kepler5": (
                "locked second-adjoint arithmetic with production variable-power "
                "first/second adjoints retained internally"
            ),
            "new_parameter_count": 0,
        },
        "hessian_binding_invocation_count": int(
            log.counts["N3 symmetric Hessian operator"]
        ),
        "raw_autograd_final_hessian_fallback_calls": int(raw_fallbacks),
        "by_system": by_system,
        "aggregate": dict(totals),
        "retained_internal_oracles": {
            "kepler_variable_power_first_adjoint_calls": int(
                by_system["kepler5"][
                    "retained_variable_power_first_adjoint_calls"
                ]
            ),
            "kepler_variable_power_second_adjoint_calls": int(
                by_system["kepler5"][
                    "retained_variable_power_second_adjoint_calls"
                ]
            ),
            "reflection_event_branch_step_calls": int(
                by_system["reflection_ball"]["retained_event_branch_step_calls"]
            ),
        },
        "first_mismatch": first_mismatch,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected,
        },
        "learned_variable_power_adjoint_binding_integrated": False,
        "eigenspectrum_condition_executed": False,
        "next_unresolved_ab_binding": "N3 positive-base variable-power adjoints",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_hessian_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["hessian_binding_invocation_count"],
        "raw_final_hessian_fallbacks": raw_fallbacks,
        "by_system": by_system,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
