#!/usr/bin/env python3
"""Force and validate only the live hybrid N3 gradient binding."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import audit_known_structure as kepler5
from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.bouncing_ball import run_reflection_peelback as reflection
from experiments.resume_hybrid_student_after_ckdtree import HybridN3GradientBinding, Log
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_n3_gradient_binding")
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
    manifest = json.loads(readiness.MANIFEST.read_text())["cases"]
    return next(row for row in manifest
                if row["system"] == system and int(row["case_id"]) == case_id)


def sampled_thetas(system: str, case_id: int, teacher: np.ndarray):
    result = json.loads((readiness.CASES / system / f"case_{case_id:05d}" /
                         "result.json").read_text())
    trace = np.load(result["successful_branch"]["trace_npz"])
    values = trace["n3_theta"]
    indexes = sorted(set((
        0, len(values) // 4, len(values) // 2,
        max(0, len(values) - 2), len(values) - 1,
    )))
    rows = [(f"saved_n3_{index}", values[index].copy()) for index in indexes]
    rows.append(("teacher", np.asarray(teacher, np.float64).copy()))
    unique, seen = [], set()
    for source, theta in rows:
        key = theta.tobytes()
        if key not in seen:
            unique.append((source, theta)); seen.add(key)
    return unique


def teacher_details(system: str, trajectory: np.ndarray,
                    theta_np: np.ndarray, torch_objective):
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
    prefix = torch.from_numpy(trajectory.copy())
    if system == "full_k_sho":
        prediction = prefix[:-1] @ theta.reshape(2, 2).T
        loss = (prediction - prefix[1:]).square().mean()
        incoming = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    elif system == "reflection_ball":
        observed = prefix[:reflection.ball.PREFIX_STATES]
        predictions = [
            reflection.torch_step(observed[index], theta)
            for index in range(reflection.ball.PREFIX_STATES - 1)
        ]
        errors = [prediction - observed[index + 1]
                  for index, prediction in enumerate(predictions)]
        loss = torch.stack(errors).square().mean()
        incoming = torch.stack(torch.autograd.grad(
            loss, predictions, retain_graph=True,
        ))
        prediction = torch.stack(predictions)
    else:
        observed = prefix[:readiness.production.kepler_teacher.PREFIX_STATES]
        prediction = kepler5.step(observed[:-1], theta, torch)
        loss = (prediction - observed[1:]).square().mean()
        incoming = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    # This raw gradient exists only as the frozen teacher reference. It is not
    # returned to or used by the live hybrid binding.
    gradient = torch.autograd.grad(loss, theta)[0]
    reference_loss = torch_objective(torch.from_numpy(theta_np.copy()))
    return {
        "loss": loss.detach(), "objective_loss": reference_loss.detach(),
        "prediction": prediction.detach(), "incoming": incoming.detach(),
        "gradient": gradient.detach(),
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    binding = HybridN3GradientBinding()
    log = Log()
    totals = Counter()
    by_system = {}
    rows = []
    first_mismatch = None
    for system, case_ids in CASES.items():
        local = Counter()
        local_rows = 0
        oracle_calls = 0
        for case_id in case_ids:
            trajectory, _, _, torch_objective, _, _, teacher = readiness.case_data(
                spec(system, case_id)
            )
            for source, theta in sampled_thetas(system, case_id, teacher):
                reference = teacher_details(
                    system, trajectory, theta, torch_objective,
                )
                result = binding.evaluate(system, trajectory, theta, log)
                input_theta = metric(theta, theta.copy())
                input_trajectory = metric(trajectory, trajectory.copy())
                loss = metric(reference["loss"].numpy(), result.loss.numpy())
                objective_loss = metric(
                    reference["objective_loss"].numpy(), result.loss.numpy(),
                )
                prediction = metric(
                    reference["prediction"].numpy(), result.prediction.numpy(),
                )
                incoming = metric(
                    reference["incoming"].numpy(),
                    result.prediction_adjoint.numpy(),
                )
                gradient = metric(
                    reference["gradient"].numpy(), result.gradient.numpy(),
                )
                for name, value in (
                    ("input_theta", input_theta),
                    ("input_trajectory", input_trajectory),
                    ("loss", loss), ("objective_loss", objective_loss),
                    ("prediction", prediction),
                    ("intermediate_adjoint", incoming),
                    ("gradient", gradient),
                ):
                    add(local, name, value); add(totals, name, value)
                oracle_calls += int(result.retained_oracles.get(
                    "production_variable_power_backward_graphs", 0
                ))
                local_rows += 1
                mismatch = any(value["bit_mismatches"] for value in (
                    input_theta, input_trajectory, loss, objective_loss,
                    prediction, incoming, gradient,
                ))
                row = {
                    "system": system, "case_id": case_id, "source": source,
                    "gradient": gradient, "loss": loss,
                    "prediction": prediction,
                    "intermediate_adjoint": incoming,
                    "retained_oracles": result.retained_oracles,
                    "raw_autograd_final_gradient_fallback_calls": (
                        result.raw_autograd_final_gradient_fallback_calls
                    ),
                    "exact": not mismatch,
                }
                rows.append(row)
                if mismatch and first_mismatch is None:
                    first_mismatch = row
        by_system[system] = {
            "cases": len(case_ids), "gradient_evaluations": local_rows,
            "gradient_elements_checked": int(local["gradient_values"]),
            "gradient_bit_mismatches": int(local["gradient_bit_mismatches"]),
            "gradient_sign_mismatches": int(local["gradient_sign_mismatches"]),
            "gradient_zero_nonzero_mismatches": int(
                local["gradient_zero_nonzero_mismatches"]
            ),
            "gradient_signed_zero_mismatches": int(
                local["gradient_signed_zero_mismatches"]
            ),
            "intermediate_adjoint_values_checked": int(
                local["intermediate_adjoint_values"]
            ),
            "intermediate_adjoint_bit_mismatches": int(
                local["intermediate_adjoint_bit_mismatches"]
            ),
            "retained_variable_power_oracle_calls": oracle_calls,
        }
        print(system, json.dumps(by_system[system]), flush=True)

    after = checkpoint_hashes()
    mismatch_total = sum(
        value for key, value in totals.items() if key.endswith("bit_mismatches")
    )
    raw_fallbacks = binding.assembly.raw_autograd_final_gradient_fallback_calls
    exact = (
        mismatch_total == 0 and raw_fallbacks == 0
        and before == after == expected_checkpoints
    )
    report = {
        "binding": "n3_gradient_binding",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "implementation": {
            "full_k_sho": (
                "exact FMA matmul forward + locked residual/square/mean adjoints + "
                "ascending-FMA MmBackward dot assembly"
            ),
            "reflection_ball": (
                "exact 2x2 FMA matvec plus locked arithmetic reverse-adjoint "
                "graph on the frozen event branch"
            ),
            "kepler5": (
                "locked arithmetic reverse-adjoint graph with production "
                "positive-base variable-power/PowBackward retained internally"
            ),
            "graph_traversal": "deterministic scaffolding",
            "new_parameter_count": 0,
        },
        "gradient_binding_invocation_count": int(log.counts["N3 gradient operator"]),
        "raw_autograd_final_gradient_fallback_calls": int(raw_fallbacks),
        "by_system": by_system,
        "aggregate": dict(totals),
        "retained_internal_oracles": {
            "kepler_variable_power_backward_graphs": int(
                by_system["kepler5"]["retained_variable_power_oracle_calls"]
            ),
            "other_retained_gradient_primitives": 0,
        },
        "first_mismatch": first_mismatch,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_checkpoints,
        },
        "hessian_binding_integrated": False,
        "variable_power_adjoint_binding_integrated": False,
        "next_unresolved_ab_binding": "N3 symmetric Hessian operator",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n3_gradient_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["gradient_binding_invocation_count"],
        "raw_final_gradient_fallbacks": raw_fallbacks,
        "by_system": by_system,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
