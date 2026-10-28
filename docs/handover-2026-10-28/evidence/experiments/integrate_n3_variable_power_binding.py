#!/usr/bin/env python3
"""Integrate and validate only the positive-base variable-power adjoints."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import audit_known_structure as kepler5
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.kepler.assemble_n3_variable_power_adjoint import (
    scalar_derivatives, variable_power,
)
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3GradientBinding, HybridN3HessianBinding, Log,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_variable_power_binding")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
CASE_IDS = (192, 193, 231)


def metric(expected, actual) -> dict:
    left = np.asarray(expected, np.float64).reshape(-1)
    right = np.asarray(actual, np.float64).reshape(-1)
    return {
        "values": int(left.size),
        "bit_mismatches": int(np.count_nonzero(
            left.view(np.uint64) != right.view(np.uint64)
        )),
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


def add(total: Counter, prefix: str, value: dict):
    for key in ("values", "bit_mismatches", "sign_mismatches",
                "zero_nonzero_mismatches", "signed_zero_mismatches"):
        total[f"{prefix}_{key}"] += int(value[key])
    total[f"{prefix}_max_absolute_error"] = max(
        float(total[f"{prefix}_max_absolute_error"]),
        value["max_absolute_error"],
    )


def case_spec(case_id: int) -> dict:
    rows = json.loads(readiness.MANIFEST.read_text())["cases"]
    return next(row for row in rows
                if row["system"] == "kepler5"
                and int(row["case_id"]) == case_id)


def sampled_thetas(case_id: int, teacher: np.ndarray):
    result = json.loads((readiness.CASES / "kepler5" /
                         f"case_{case_id:05d}" / "result.json").read_text())
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
    rows = tuple(torch.autograd.grad(
        gradient[index], theta, retain_graph=True,
    )[0] for index in range(len(theta)))
    raw = torch.stack(rows)
    return gradient.detach(), raw.detach(), (0.5 * (raw + raw.T)).detach()


def active_bases(trajectory: np.ndarray, theta_np: np.ndarray):
    states = torch.from_numpy(
        trajectory[:readiness.production.kepler_teacher.PREFIX_STATES - 1].copy()
    )
    theta = torch.from_numpy(theta_np.copy())
    dx = states[..., 4] - states[..., 0]
    dy = states[..., 5] - states[..., 1]
    first = torch.sqrt(dx * dx + dy * dy)
    prediction = kepler5.step(states, theta, torch)
    dx2 = prediction[..., 4] - prediction[..., 0]
    dy2 = prediction[..., 5] - prediction[..., 1]
    second = torch.sqrt(dx2 * dx2 + dy2 * dy2)
    return torch.cat((first, second)).detach()


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    gradient_binding = HybridN3GradientBinding()
    hessian_binding = HybridN3HessianBinding(gradient_binding)
    log = Log()
    totals = Counter()
    rows = []
    base_min = np.inf
    base_max = 0.0
    exponent_min = np.inf
    exponent_max = -np.inf
    binding_invocations = 0
    pow_fallbacks = 0
    first_mismatch = None

    for case_id in CASE_IDS:
        trajectory, _, _, torch_objective, _, _, teacher = readiness.case_data(
            case_spec(case_id)
        )
        for source, theta in sampled_thetas(case_id, teacher):
            bases = active_bases(trajectory, theta)
            exponents = torch.full_like(bases, float(theta[4]))
            # Deterministic nonzero upstream values exercise both signs while
            # remaining entirely inside the observed production base domain.
            upstream = torch.linspace(
                -1.0, 1.0, len(bases), dtype=torch.float64,
            )
            production = scalar_derivatives(
                torch.pow, bases, exponents, upstream,
            )
            assembled = scalar_derivatives(
                variable_power, bases, exponents, upstream,
            )
            names = (
                "forward_y", "first_base", "first_exponent",
                "second_base_base", "second_base_exponent",
                "second_exponent_base", "second_exponent_exponent",
            )
            isolated = {}
            for name, teacher_value, student_value in zip(
                    names, production, assembled):
                value = metric(
                    teacher_value.detach().numpy(),
                    student_value.detach().numpy(),
                )
                isolated[name] = value
                add(totals, name, value)
            binding_invocations += 1

            teacher_gradient, teacher_raw, teacher_symmetric = (
                teacher_derivatives(torch_objective, theta)
            )
            result = hessian_binding.evaluate(
                "kepler5", trajectory, theta, log,
            )
            full = {
                "full_gradient": metric(
                    teacher_gradient.numpy(), result.gradient.numpy()),
                "full_raw_hessian": metric(
                    teacher_raw.numpy(), result.raw_hessian.numpy()),
                "full_symmetric_hessian": metric(
                    teacher_symmetric.numpy(),
                    result.symmetric_hessian.numpy()),
            }
            for name, value in full.items():
                add(totals, name, value)
            pow_fallbacks += int(result.retained_oracles.get(
                "production_pow_backward_fallback_calls", 0
            ))
            binding_invocations += int(result.retained_oracles.get(
                "assembled_variable_power_first_adjoint_calls", 0
            ))
            binding_invocations += int(result.retained_oracles.get(
                "assembled_variable_power_second_adjoint_calls", 0
            ))
            base_min = min(base_min, float(bases.min()))
            base_max = max(base_max, float(bases.max()))
            exponent_min = min(exponent_min, float(exponents.min()))
            exponent_max = max(exponent_max, float(exponents.max()))
            mismatch = any(value["bit_mismatches"]
                           for value in (*isolated.values(), *full.values()))
            row = {
                "case_id": case_id, "source": source,
                "base_min": float(bases.min()), "base_max": float(bases.max()),
                "exponent": float(theta[4]), "isolated": isolated,
                "full": full, "retained": result.retained_oracles,
                "exact": not mismatch,
            }
            rows.append(row)
            if mismatch and first_mismatch is None:
                first_mismatch = row

    after = checkpoint_hashes()
    mismatch_total = sum(
        value for key, value in totals.items() if key.endswith("bit_mismatches")
    )
    exact = (
        mismatch_total == 0 and pow_fallbacks == 0
        and before == after == expected_checkpoints
    )
    first_values = int(
        totals["first_base_values"] + totals["first_exponent_values"]
    )
    second_values = int(sum(
        totals[f"{name}_values"] for name in (
            "second_base_base", "second_base_exponent",
            "second_exponent_base", "second_exponent_exponent",
        )
    ))
    report = {
        "binding": "n3_variable_power_adjoint_binding",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "forward": "unchanged deterministic torch.pow",
            "first_base": "incoming * (p * x**(p-1))",
            "first_exponent": "incoming * (x**p * log(x))",
            "second_adjoints": (
                "frozen positive-base learned assembly with locked "
                "arithmetic and production operation order"
            ),
            "new_parameter_count": 0,
        },
        "active_domain": {
            "base_min": base_min, "base_max": base_max,
            "exponent_min": exponent_min, "exponent_max": exponent_max,
            "all_bases_positive": base_min > 0.0,
        },
        "saved_states": len(rows),
        "variable_power_binding_invocations": int(binding_invocations),
        "first_adjoint_values_checked": first_values,
        "second_adjoint_values_checked": second_values,
        "production_pow_backward_fallback_calls": int(pow_fallbacks),
        "aggregate": dict(totals),
        "full_gradient_values_checked": int(totals["full_gradient_values"]),
        "full_gradient_bit_mismatches": int(
            totals["full_gradient_bit_mismatches"]
        ),
        "full_raw_hessian_values_checked": int(
            totals["full_raw_hessian_values"]
        ),
        "full_raw_hessian_bit_mismatches": int(
            totals["full_raw_hessian_bit_mismatches"]
        ),
        "full_symmetric_hessian_values_checked": int(
            totals["full_symmetric_hessian_values"]
        ),
        "full_symmetric_hessian_bit_mismatches": int(
            totals["full_symmetric_hessian_bit_mismatches"]
        ),
        "first_mismatch": first_mismatch,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_checkpoints,
        },
        "eigenspectrum_condition_executed": False,
        "next_unresolved_ab_binding": (
            "N3 DIRECT/DAMPED curvature route "
            "(after retained eigenspectrum/condition D oracle)"
        ),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_variable_power_adjoint_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "saved_states": len(rows),
        "binding_invocations": binding_invocations,
        "first_values": first_values,
        "second_values": second_values,
        "pow_fallbacks": pow_fallbacks,
        "gradient_mismatches": report["full_gradient_bit_mismatches"],
        "raw_hessian_mismatches": report["full_raw_hessian_bit_mismatches"],
        "symmetric_hessian_mismatches": (
            report["full_symmetric_hessian_bit_mismatches"]
        ),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
