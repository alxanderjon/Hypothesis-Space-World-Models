#!/usr/bin/env python3
"""Continue the hybrid composition gate after exact N1 replay.

Archive and exact-Powell N2 are replayed on the frozen representative
branches.  The run then stops at the first same-input A/B divergence in N3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_after_ckdtree")
FREEZE = Path("outputs/deterministic_teacher_frozen")
CHECKPOINT_REFERENCE = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")

CASES = (
    ("full_k_sho", 0),
    ("reflection_ball", 209),
    ("kepler5", 192),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def metric(left, right):
    a, b = np.asarray(left, np.float64).reshape(-1), np.asarray(right, np.float64).reshape(-1)
    return {
        "values": len(a), "bit_mismatches": int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64))),
        "numeric_mismatches": int(np.count_nonzero(a != b)),
        "sign_mismatches": int(np.count_nonzero(np.signbit(a) != np.signbit(b))),
        "zero_nonzero_mismatches": int(np.count_nonzero((a == 0.0) != (b == 0.0))),
        "max_absolute_error": float(np.max(np.abs(a - b))) if len(a) else 0.0,
    }


def spec_for(system, case_id):
    manifest = json.loads(readiness.MANIFEST.read_text())
    return next(row for row in manifest["cases"]
                if row["system"] == system and int(row["case_id"]) == case_id)


def saved_result(system, case_id):
    path = readiness.CASES / system / f"case_{case_id:05d}" / "result.json"
    return json.loads(path.read_text())


def replay_n2():
    rows = []
    for system, case_id in CASES:
        spec = spec_for(system, case_id)
        trajectory, _, objective, _, global_lower, global_upper, _ = readiness.case_data(spec)
        saved = saved_result(system, case_id)["successful_branch"]
        archive_theta = np.asarray(saved["archive_theta"], np.float64)
        archive_count = int(saved["current_evaluation_count"])
        radius = (global_upper - global_lower) / float(archive_count) ** (
            1.0 / len(global_lower))
        oracle_lower = np.maximum(global_lower, archive_theta - radius)
        oracle_upper = np.minimum(global_upper, archive_theta + radius)
        result = staged._run_n2(
            objective,
            archive_theta,
            float(saved["archive_loss"]),
            np.asarray(saved["n2_box_lower"], np.float64),
            np.asarray(saved["n2_box_upper"], np.float64),
        )
        rows.append({
            "system": system, "case_id": case_id,
            "archive_count": int(saved["current_evaluation_count"]),
            "archive_index": int(saved["archive_index"]),
            "archive_order": "newest-first, then canonical older-site order",
            "absorbing_success": True,
            "box_lower_exact": bits_equal(saved["n2_box_lower"], oracle_lower),
            "box_upper_exact": bits_equal(saved["n2_box_upper"], oracle_upper),
            "n2_evaluations_teacher": int(saved["n2_evaluations"]),
            "n2_evaluations_student": int(result["evaluations"]),
            "n2_endpoint_metric": metric(saved["n2_endpoint"], result["theta"]),
            "n2_loss_metric": metric([saved["n2_loss"]], [result["loss"]]),
            "n2_stop_teacher": saved["n2_stop_reason"],
            "n2_stop_student": result["message"],
            "n2_endpoint": result["theta"].tolist(),
            "trajectory_shape": list(trajectory.shape),
        })
    return rows


def sho_learned_objective(theta, inputs, targets):
    predictions = []
    for value in inputs:
        predictions.append(torch.stack((
            gradient.add(gradient.multiply(theta[0], value[0]),
                         gradient.multiply(theta[1], value[1])),
            gradient.add(gradient.multiply(theta[2], value[0]),
                         gradient.multiply(theta[3], value[1])),
        )))
    prediction = torch.stack(predictions)
    return gradient.mean(gradient.square(gradient.subtract(prediction, targets)))


def first_n3_divergence(n2_rows):
    # Execution order reaches full-K SHO first.  Stop after its gradient output;
    # Hessian/control operations are intentionally not invoked after mismatch.
    row = n2_rows[0]
    spec = spec_for("full_k_sho", 0)
    trajectory, _, _, torch_objective, _, _, _ = readiness.case_data(spec)
    inputs = torch.from_numpy(trajectory[:-1].copy())
    targets = torch.from_numpy(trajectory[1:].copy())
    theta_value = np.asarray(row["n2_endpoint"], np.float64)

    teacher_theta = torch.from_numpy(theta_value.copy()).requires_grad_(True)
    teacher_loss = torch_objective(teacher_theta)
    teacher_gradient = torch.autograd.grad(teacher_loss, teacher_theta)[0].detach()

    gradient.OPS = gradient.LearnedArithmetic()
    student_theta = torch.from_numpy(theta_value.copy()).requires_grad_(True)
    student_loss = sho_learned_objective(student_theta, inputs, targets)
    student_gradient = torch.autograd.grad(student_loss, student_theta)[0].detach()

    loss_metric = metric([float(teacher_loss.detach())], [float(student_loss.detach())])
    gradient_metric = metric(teacher_gradient.numpy(), student_gradient.numpy())
    if not loss_metric["bit_mismatches"] and not gradient_metric["bit_mismatches"]:
        return None
    return {
        "operation": "gradient operator",
        "section": "N3", "category": "A", "system": "full_k_sho", "dimension": 4,
        "input_theta": theta_value.tolist(),
        "teacher_forward_loss": float(teacher_loss.detach()),
        "student_forward_loss": float(student_loss.detach()),
        "forward_loss_metric": loss_metric,
        "teacher_output": teacher_gradient.tolist(),
        "student_output": student_gradient.tolist(),
        "mismatch": gradient_metric,
        "type": "numerical/implementation-order",
        "localization": (
            "the frozen scalar learned arithmetic graph evaluates the free-K matrix-vector "
            "products as explicit multiply-then-add operations, while the production full-K "
            "Torch objective uses its matrix-multiplication/reduction kernel; their forward and "
            "backward float64 evaluation orders differ at the N2 endpoint"
        ),
        "hessian_or_later_n3_invoked": False,
    }


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    n1_report = json.loads((OUT / "report.json").read_text())
    if n1_report["status"] != "N1_REPLAY_EXACT":
        raise RuntimeError("N1 gate did not pass")
    expected = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected:
        raise RuntimeError("checkpoint inventory changed")
    n2_rows = replay_n2()
    n2_errors = sum(
        row["n2_endpoint_metric"]["bit_mismatches"] +
        row["n2_loss_metric"]["bit_mismatches"] +
        int(row["n2_evaluations_teacher"] != row["n2_evaluations_student"]) +
        int(row["n2_stop_teacher"] != row["n2_stop_student"])
        for row in n2_rows
    )
    if n2_errors:
        divergence = {
            "operation": "bounded Powell mathematical state machine",
            "section": "N2", "category": "B", "mismatch_count": n2_errors,
            "type": "state-machine or implementation-order", "rows": n2_rows,
        }
    else:
        divergence = first_n3_divergence(n2_rows)

    # Carry forward the pre-distance operations from the original fail-closed
    # integration.  This run resumes that gate; it does not re-audit them.
    prior_invocations = json.loads(Path(
        "outputs/hybrid_student_baseline/invocations.json").read_text())
    inherited = [
        {"section": row["section"], "operation": row["operation"],
         "category": row["category"], "invocations": row["invocations"],
         "implementation": row["implementation"], "inherited_from_prior_gate": True}
        for row in prior_invocations
        if row["reached_before_gate_stop"] and row["operation"] != "Euclidean distance"
    ]
    active_before_stop = inherited + list(n1_report["active_A_B_operations"])
    active_before_stop.extend([
        {"section": "Archive", "operation": "absorbing success latch", "category": "B",
         "invocations": len(n2_rows), "implementation": "locked Boolean latch assembly"},
        {"section": "N2", "operation": "bounded Powell mathematical state machine", "category": "B",
         "invocations": len(n2_rows), "implementation": "exact SciPy-1.13 Powell assembly"},
        {"section": "N2", "operation": "best-ever strict-loss selection", "category": "B",
         "invocations": len(n2_rows), "implementation": "locked strict comparison/routing assembly"},
    ])
    if divergence and divergence["operation"] == "gradient operator":
        active_before_stop.append({
            "section": "N3", "operation": "gradient operator", "category": "A",
            "invocations": 1,
            "implementation": "frozen learned scalar-arithmetic/autograd gradient graph",
            "status": "FIRST_DIVERGENCE",
        })
    report = {
        "status": "STOPPED_FIRST_A_B_DIVERGENCE" if divergence else "HYBRID_STUDENT_BASELINE",
        "teacher_status": "DETERMINISTIC_TEACHER_VALIDATED",
        "n1_resume_status": n1_report["status"],
        "strict_nearest_owner": n1_report["strict_nearest_owner"],
        "archive_n2_replay": n2_rows,
        "archive_n2_exact": n2_errors == 0,
        "first_divergence": divergence,
        "active_A_B_operations_before_stop": active_before_stop,
        "remaining_D_oracles": n1_report["remaining_D_oracles"],
        "checkpoint_hashes_unchanged": checkpoint_hashes() == expected,
        "frozen_inventory_sha256": sha256(FREEZE / "operation_inventory.json"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "downstream_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "archive_n2_exact": report["archive_n2_exact"],
        "n2": [{"system": row["system"], "evaluations": row["n2_evaluations_student"],
                "endpoint_bit_mismatches": row["n2_endpoint_metric"]["bit_mismatches"],
                "loss_bit_mismatches": row["n2_loss_metric"]["bit_mismatches"]}
               for row in n2_rows],
        "first_divergence": divergence,
        "checkpoints_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
