#!/usr/bin/env python3
"""Validate the runtime-length SHO derivative binding and legacy case-0 repair."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.geometry.n3_gradient_binding_assembly import N3GradientBindingAssembly
from experiments.geometry.n3_hessian_binding_assembly import N3SymmetricHessianBindingAssembly
from experiments.geometry.torch_matmul2_fma_assembly import TorchMatmul2FMAAssembly
from experiments.geometry.torch_mm_backward_dot_assembly import TorchMMBackwardDotAssembly
from experiments.resume_hybrid_student_after_ckdtree import Log, run_n2_with_best_ever_assembly
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.validate_complete_hybrid_student_stage2 import records, load_case, normalize_expected
from experiments.validate_complete_hybrid_student_v2 import HybridN3, HybridObjective, bits_equal, metric


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho_derivative_runtime_length_binding")
CHECKPOINT_REFERENCE = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
RESULTS = Path("outputs/sho/stage1B/checkpointed_resource_edirect/results_partial.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher(trajectory, theta):
    inputs = torch.from_numpy(np.asarray(trajectory[:-1], np.float64).copy())
    targets = torch.from_numpy(np.asarray(trajectory[1:], np.float64).copy())
    value = torch.from_numpy(np.asarray(theta, np.float64).copy()).requires_grad_(True)
    prediction = inputs @ value.reshape(2, 2).T
    loss = (prediction - targets).square().mean()
    gradient = torch.autograd.grad(loss, value, create_graph=True)[0]
    raw = torch.stack([
        torch.autograd.grad(gradient[index], value, retain_graph=True)[0]
        for index in range(4)
    ])
    symmetric = 0.5 * (raw + raw.T)
    return prediction.detach(), loss.detach(), gradient.detach(), raw.detach(), symmetric.detach()


def signed_zero_mismatches(left, right):
    a = np.asarray(left, np.float64).reshape(-1)
    b = np.asarray(right, np.float64).reshape(-1)
    zero = (a == 0.0) & (b == 0.0)
    return int(np.count_nonzero(a[zero].view(np.uint64) != b[zero].view(np.uint64)))


def length_kernel_controls():
    rng = np.random.default_rng(20_261_009)
    forward = TorchMatmul2FMAAssembly()
    dot = TorchMMBackwardDotAssembly()
    rows = []
    for length in (126, 127):
        forward_values = dot_values = 0
        forward_mismatch = dot_mismatch = 0
        for _ in range(32):
            inputs = rng.normal(size=(length, 2)) * np.exp2(
                rng.integers(-20, 21, size=(length, 2)))
            matrix = rng.normal(size=(2, 2))
            expected_forward = (
                torch.from_numpy(inputs) @ torch.from_numpy(matrix).T
            ).numpy()
            actual_forward = forward(inputs, matrix)
            forward_values += expected_forward.size
            forward_mismatch += metric(expected_forward, actual_forward)["bit_mismatches"]

            left = rng.normal(size=(2, length)) * np.exp2(
                rng.integers(-20, 21, size=(2, length)))
            right = rng.normal(size=(length, 2)) * np.exp2(
                rng.integers(-20, 21, size=(length, 2)))
            expected_dot = (torch.from_numpy(left) @ torch.from_numpy(right)).numpy()
            actual_dot = dot(left, right)
            dot_values += expected_dot.size
            dot_mismatch += metric(expected_dot, actual_dot)["bit_mismatches"]
        rows.append({
            "transition_count": length,
            "forward_path": "two-row vector body" + (" + scalar odd tail" if length % 2 else ""),
            "forward_values": forward_values,
            "forward_bit_mismatches": forward_mismatch,
            "dot_path": "ascending FMA reduction indices 0..N-1",
            "dot_values": dot_values,
            "dot_bit_mismatches": dot_mismatch,
        })
    return rows


def mean_scale_control(length):
    prediction = torch.zeros((length, 2), dtype=torch.float64, requires_grad=True)
    target = torch.ones_like(prediction)
    loss = (prediction - target).square().mean()
    incoming = torch.autograd.grad(loss, prediction, create_graph=True)[0]
    second = torch.autograd.grad(incoming[0, 0], prediction)[0][0, 0]
    assembly = N3SymmetricHessianBindingAssembly().sho
    actual = assembly.second_adjoint_scale(length)
    return {
        "transition_count": length,
        "residual_elements": 2 * length,
        "teacher_scale": float(second), "student_scale": float(actual),
        "teacher_bits": int(np.float64(second).view(np.uint64)),
        "student_bits": int(np.float64(actual).view(np.uint64)),
        "bit_exact": int(np.float64(second).view(np.uint64)) == int(np.float64(actual).view(np.uint64)),
        "student_order": "RN(1/residual_elements), then locked RN(multiply by 2)",
    }


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes:
        raise RuntimeError("checkpoint inventory changed before validation")

    data = np.load(DATA)
    teacher_theta = np.asarray(data["teachers"], np.float64).reshape(-1, 4)
    prior = {int(row["case_id"]): row for row in json.loads(RESULTS.read_text())}
    cases = (0, 1, 7, 15, 31, 32, 33, 64, 128, 255)
    gradient_binding = N3GradientBindingAssembly()
    hessian_binding = N3SymmetricHessianBindingAssembly(gradient_binding)
    rows = []
    totals = Counter()
    for case_id in cases:
        trajectory = np.asarray(data[f"prefix_{case_id}"], np.float64)
        states = [
            ("saved_n2_endpoint", np.asarray(prior[case_id]["n2_theta"], np.float64)),
            ("teacher", teacher_theta[case_id].copy()),
            ("teacher_nextafter", np.nextafter(teacher_theta[case_id], np.inf)),
        ]
        for source, theta in states:
            prediction, loss, gradient, raw, symmetric = teacher(trajectory, theta)
            recovered_gradient = gradient_binding.evaluate("full_k_sho", trajectory, theta)
            recovered_hessian = hessian_binding.evaluate("full_k_sho", trajectory, theta)
            comparisons = {
                "forward_prediction": metric(prediction.numpy(), recovered_gradient.prediction.numpy()),
                "loss": metric([float(loss)], [float(recovered_gradient.loss)]),
                "gradient": metric(gradient.numpy(), recovered_gradient.gradient.numpy()),
                "raw_hessian": metric(raw.numpy(), recovered_hessian.raw_hessian.numpy()),
                "symmetric_hessian": metric(symmetric.numpy(), recovered_hessian.symmetric_hessian.numpy()),
            }
            teacher_direction = torch.linalg.solve(symmetric, -gradient)
            student_direction = torch.linalg.solve(
                recovered_hessian.symmetric_hessian, -recovered_gradient.gradient)
            comparisons["newton_direction"] = metric(
                teacher_direction.numpy(), student_direction.numpy())
            signed_zero = sum(signed_zero_mismatches(a, b) for a, b in (
                (prediction.numpy(), recovered_gradient.prediction.numpy()),
                (gradient.numpy(), recovered_gradient.gradient.numpy()),
                (raw.numpy(), recovered_hessian.raw_hessian.numpy()),
                (symmetric.numpy(), recovered_hessian.symmetric_hessian.numpy()),
            ))
            for name, value in comparisons.items():
                totals[f"{name}_values"] += int(np.asarray(
                    prediction if name == "forward_prediction" else
                    gradient if name in ("gradient", "newton_direction") else
                    raw if name == "raw_hessian" else
                    symmetric if name == "symmetric_hessian" else loss
                ).size)
                totals[f"{name}_bit_mismatches"] += value["bit_mismatches"]
            totals["signed_zero_mismatches"] += signed_zero
            rows.append({
                "case_id": case_id, "source": source,
                "state_count": int(len(trajectory)),
                "transition_count": int(len(trajectory) - 1),
                "comparisons": comparisons,
                "signed_zero_mismatches": signed_zero,
            })

    kernel_controls = length_kernel_controls()
    scales = [mean_scale_control(length) for length in (126, 127)]
    exact = bool(
        all(value == 0 for key, value in totals.items() if key.endswith("bit_mismatches"))
        and totals["signed_zero_mismatches"] == 0
        and all(row["forward_bit_mismatches"] == 0 and row["dot_bit_mismatches"] == 0
                for row in kernel_controls)
        and all(row["bit_exact"] for row in scales)
    )
    if not exact:
        raise RuntimeError("runtime-length derivative exactness gate failed")

    # Re-run only the originally failing Stage-2 old SHO case 0.
    record = next(row for row in records()
                  if row["cohort"] == "old_sho" and row["case_id"] == 0)
    trajectory, lower, upper, _, start, start_loss, box_lower, box_upper = load_case(record)
    expected = normalize_expected(record)
    log = Log()
    objective = HybridObjective("full_k_sho", trajectory, log)
    n2 = run_n2_with_best_ever_assembly(
        objective, start, start_loss, box_lower, box_upper, log)
    n2_exact = bool(
        int(n2["evaluations"]) == int(expected["n2_evaluations"])
        and n2["message"] == expected["n2_stop_reason"]
        and bits_equal(n2["theta"], expected["n2_endpoint"])
        and bits_equal([n2["loss"]], [expected["n2_loss"]])
    )
    if not n2_exact:
        raise RuntimeError("legacy case-0 N2 precondition changed")
    n3 = HybridN3("full_k_sho", trajectory, lower, upper, log).run(
        n2["theta"], expected, crosscheck_gradient=True)
    case0_frozen_artifact_exact = bool(
        n3["first_divergence"] is None
        and n3["accepted_steps"] == n3["expected_steps"]
        and n3["stop_reason"] == n3["expected_stop"]
        and n3["theta_metric"]["bit_mismatches"] == 0
        and n3["loss_metric"]["bit_mismatches"] == 0
    )
    original_hessian_divergence_repaired = n3["first_divergence"] is None
    if not original_hessian_divergence_repaired:
        raise RuntimeError("legacy case-0 still fails in the repaired derivative path")

    # Distinguish the requested derivative repair from the next integration
    # boundary.  The current deterministic N3 reference is run unchanged.
    torch_inputs = torch.from_numpy(trajectory[:-1].copy())
    torch_targets = torch.from_numpy(trajectory[1:].copy())
    torch_objective = lambda theta: (
        torch_inputs @ theta.reshape(2, 2).T - torch_targets
    ).square().mean()
    _, _, current_run = staged._make_n3(torch_objective, lower, upper)
    endpoint = polish.N2Endpoint(
        0, n2["theta"].copy(), float(n2["loss"]),
        "runtime_length_repair", "legacy_case0", False,
    )
    current_teacher_n3 = current_run(endpoint, torch.empty(0))
    current_teacher_theta = np.asarray(current_teacher_n3.theta_history[-1], np.float64)
    current_teacher_loss = float(current_teacher_n3.loss_history[-1])
    current_teacher_exact = bool(
        int(current_teacher_n3.accepted_steps) == n3["accepted_steps"]
        and current_teacher_n3.stop_reason == n3["stop_reason"]
        and bits_equal(current_teacher_theta, n3["final_theta"])
        and bits_equal([current_teacher_loss], [n3["final_loss"]])
    )
    if not current_teacher_exact:
        raise RuntimeError("repaired hybrid diverges from the current deterministic N3 reference")

    report = {
        "status": "INTEGRATED_EXACT",
        "binding": "sho_derivative_runtime_length_binding",
        "scope": "SHO forward/gradient/Hessian runtime transition length only",
        "static_contract_audit": {
            "repaired_live_assumptions": [
                "FullKSHOHessianAssembly constructor stored transition_count=127",
                "raw Hessian required inputs.shape == (127,2)",
                "square/mean second-adjoint scale was precomputed from 254 residuals",
            ],
            "already_runtime_general": [
                "TorchMatmul2FMAAssembly reads N from inputs and selects its odd scalar tail from N % 2",
                "TorchMMBackwardDotAssembly reduces in ascending order over lhs.shape[-1]",
                "N3GradientBindingAssembly obtains loss scaling from the runtime Torch mean",
                "production-order residual/MSE assembly uses runtime residual element count",
                "Hessian row seeds, parameter flatten order, and symmetrization depend on 2x2 K, not N",
            ],
            "historical_only_not_live": [
                "127/254 descriptions in recovery reports and audit scripts",
            ],
            "new_runtime_scale_order": "RN(1/(2*N)) followed by locked RN(multiply by 2)",
        },
        "saved_prefix_lengths": {
            "available_state_lengths": {"127": 32, "128": 224},
            "available_transition_lengths": {"126": 32, "127": 224},
            "validated_cases": list(cases),
            "validated_states": len(rows),
        },
        "exact_replay_totals": dict(totals),
        "per_state": rows,
        "runtime_mean_scale": scales,
        "even_odd_kernel_controls": kernel_controls,
        "legacy_stage2_case0": {
            "n2_exact": n2_exact,
            "n2_evaluations": int(n2["evaluations"]),
            "original_hessian_shape_failure_repaired": original_hessian_divergence_repaired,
            "current_deterministic_n3_exact": current_teacher_exact,
            "frozen_legacy_result_exact": case0_frozen_artifact_exact,
            "accepted_steps_teacher": n3["expected_steps"],
            "accepted_steps_student": n3["accepted_steps"],
            "stop_reason_teacher": n3["expected_stop"],
            "stop_reason_student": n3["stop_reason"],
            "final_theta_metric": n3["theta_metric"],
            "final_loss_metric": n3["loss_metric"],
            "current_teacher": {
                "accepted_steps": int(current_teacher_n3.accepted_steps),
                "stop_reason": current_teacher_n3.stop_reason,
                "final_theta": current_teacher_theta.tolist(),
                "final_loss": current_teacher_loss,
            },
            "frozen_legacy_artifact": {
                "accepted_steps": n3["expected_steps"],
                "stop_reason": n3["expected_stop"],
                "final_theta": expected["final_theta"],
                "final_loss": expected["final_loss"],
            },
            "full_822_regression_resumed": False,
        },
        "new_parameters": 0,
        "checkpoint_hashes_unchanged": checkpoint_hashes() == before == expected_hashes,
        "implementation": "experiments/geometry/full_k_sho_hessian_assembly.py",
        "implementation_sha256": sha256(Path("experiments/geometry/full_k_sho_hessian_assembly.py")),
        "next_integration_boundary": (
            "the repaired student and current deterministic N3 reference agree, but the "
            "frozen legacy case-0 artifact records a historical three-step result while "
            "the current reference takes two steps; full 822-case regression remains "
            "paused and no later case was run"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "sho_derivative_runtime_length_binding": "INTEGRATED_EXACT",
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
        "full_822_regression_resumed": False,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "validated_states": len(rows),
        "transition_lengths": [126, 127],
        "bit_mismatches": {
            key: value for key, value in totals.items() if key.endswith("bit_mismatches")
        },
        "kernel_controls": kernel_controls,
        "legacy_stage2_case0": report["legacy_stage2_case0"],
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
