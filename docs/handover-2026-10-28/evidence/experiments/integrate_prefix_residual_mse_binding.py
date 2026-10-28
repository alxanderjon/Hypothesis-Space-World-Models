#!/usr/bin/env python3
"""Integrate and validate only prefix residual/square/MSE arithmetic."""
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
from experiments.resume_hybrid_student_after_ckdtree import (
    HybridN3GradientBinding,
    HybridPrefixResidualMSEBinding,
    Log,
)
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/hybrid_student_prefix_residual_mse_binding")
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
        "values": int(left.size), "shape_match": True,
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


def saved_thetas(system: str, case_id: int):
    result = json.loads(
        (readiness.CASES / system / f"case_{case_id:05d}" / "result.json").read_text()
    )
    theta = np.load(result["successful_branch"]["trace_npz"])["n3_theta"]
    indexes = sorted(set((0, len(theta) - 1)))
    return [(index, theta[index].copy()) for index in indexes]


def numpy_prediction(system, trajectory, theta):
    if system == "full_k_sho":
        return np.einsum("ij,nj->ni", theta.reshape(2, 2), trajectory[:-1]), trajectory[1:]
    if system == "reflection_ball":
        prefix = trajectory[:reflection.ball.PREFIX_STATES]
        prediction = np.stack([
            reflection.numpy_step(prefix[index], theta)
            for index in range(reflection.ball.PREFIX_STATES - 1)
        ])
        return prediction, prefix[1:]
    prefix = trajectory[:128]
    return kepler5.step(prefix[:-1], theta, np), prefix[1:]


def torch_prediction(system, trajectory, theta):
    prefix = torch.from_numpy(np.asarray(trajectory, np.float64).copy())
    parameter = torch.from_numpy(np.asarray(theta, np.float64).copy())
    if system == "full_k_sho":
        return prefix[:-1] @ parameter.reshape(2, 2).T, prefix[1:]
    if system == "reflection_ball":
        prefix = prefix[:reflection.ball.PREFIX_STATES]
        return torch.stack([
            reflection.torch_step(prefix[index], parameter)
            for index in range(reflection.ball.PREFIX_STATES - 1)
        ]), prefix[1:]
    prefix = prefix[:128]
    return kepler5.step(prefix[:-1], parameter, torch), prefix[1:]


def teacher_numpy_reduction(system, residual, square):
    if system == "reflection_ball":
        row_terms = []
        accumulators = []
        total = np.float64(0.0)
        for row in residual:
            term = np.float64(np.dot(row, row))
            row_terms.append(term)
            total = np.float64(total + term)
            accumulators.append(total)
        loss = np.float64(total / np.float64(square.size))
        return np.asarray(row_terms), np.asarray(accumulators), total, loss
    total = np.float64(np.sum(square))
    loss = np.float64(np.mean(square))
    return np.asarray([total]), np.asarray([total]), total, loss


def compare_path(binding, system, predicted, target, path, log):
    if path == "numpy":
        prediction_np = np.asarray(predicted, np.float64)
        target_np = np.asarray(target, np.float64)
        teacher_residual = prediction_np - target_np
        teacher_square = teacher_residual * teacher_residual
        terms, accumulators, total, loss = teacher_numpy_reduction(
            system, teacher_residual, teacher_square
        )
    else:
        prediction_tensor = predicted
        target_tensor = target
        teacher_residual_tensor = prediction_tensor - target_tensor
        teacher_square_tensor = teacher_residual_tensor.square()
        teacher_residual = teacher_residual_tensor.detach().numpy()
        teacher_square = teacher_square_tensor.detach().numpy()
        total = float(teacher_square_tensor.sum())
        loss = float(teacher_square_tensor.mean())
        terms = np.asarray([total], np.float64)
        accumulators = terms.copy()
    result = binding.evaluate(
        predicted, target, system, reduction_path=path, log=log
    )
    return {
        "path": path,
        "prediction": metric(np.asarray(predicted), result.prediction.detach().numpy()),
        "residual": metric(teacher_residual, result.residual.detach().numpy()),
        "square": metric(teacher_square, result.square.detach().numpy()),
        "reduction_terms": metric(terms, result.reduction_terms.detach().numpy()),
        "reduction_accumulators": metric(
            accumulators, result.reduction_accumulators.detach().numpy()
        ),
        "total": metric([total], [float(result.total)]),
        "loss": metric([loss], [float(result.loss)]),
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINT_REFERENCE.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes:
        raise RuntimeError("checkpoint inventory differs from frozen reference")

    binding = HybridPrefixResidualMSEBinding()
    gradient = HybridN3GradientBinding()
    log = Log()
    rows = []
    by_system = {}
    stage_names = (
        "prediction", "residual", "square", "reduction_terms",
        "reduction_accumulators", "total", "loss",
    )

    for system, case_ids in SAVED_CASES.items():
        local = Counter()
        for case_id in case_ids:
            trajectory, _, _, torch_objective, _, _, _ = readiness.case_data(
                case_spec(system, case_id)
            )
            for index, theta in saved_thetas(system, case_id):
                np_prediction, np_target = numpy_prediction(system, trajectory, theta)
                numpy_row = compare_path(
                    binding, system, np_prediction, np_target, "numpy", log
                )

                native_prediction, torch_target = torch_prediction(system, trajectory, theta)
                live = gradient.evaluate(system, trajectory, theta, log)
                prediction_input = metric(
                    native_prediction.detach().numpy(), live.prediction.numpy()
                )
                torch_row = compare_path(
                    binding, system, live.prediction, torch_target, "torch", log
                )
                native_loss = torch_objective(
                    torch.from_numpy(theta.copy())
                ).detach()
                live_loss = metric([float(native_loss)], [float(live.loss)])
                row = {
                    "system": system, "case_id": case_id,
                    "n3_trace_index": index,
                    "numpy": numpy_row, "torch": torch_row,
                    "live_prediction_input": prediction_input,
                    "live_gradient_objective_loss": live_loss,
                }
                mismatch = sum(
                    path_row[name]["bit_mismatches"]
                    for path_row in (numpy_row, torch_row)
                    for name in stage_names
                ) + prediction_input["bit_mismatches"] + live_loss["bit_mismatches"]
                row["mismatches"] = mismatch
                rows.append(row)
                local["saved_states"] += 1
                local["binding_invocations"] += 2
                local["mismatches"] += mismatch
        by_system[system] = dict(local)

    # Signed-zero control for both active generic mean paths.  Reflection's
    # special NumPy reduction is already exercised by its saved states.
    control_prediction = np.asarray([[0.0, -0.0], [-0.0, 0.0]], np.float64)
    control_target = np.asarray([[-0.0, 0.0], [0.0, -0.0]], np.float64)
    controls = [
        compare_path(binding, "full_k_sho", control_prediction,
                     control_target, "numpy", log),
        compare_path(binding, "full_k_sho",
                     torch.from_numpy(control_prediction.copy()),
                     torch.from_numpy(control_target.copy()), "torch", log),
    ]

    all_path_rows = [row[path] for row in rows for path in ("numpy", "torch")]
    all_path_rows.extend(controls)
    residual_values = sum(row["residual"]["values"] for row in all_path_rows)
    square_values = sum(row["square"]["values"] for row in all_path_rows)
    residual_mismatches = sum(row["residual"]["bit_mismatches"] for row in all_path_rows)
    square_mismatches = sum(row["square"]["bit_mismatches"] for row in all_path_rows)
    reduction_mismatches = sum(
        row[name]["bit_mismatches"]
        for row in all_path_rows
        for name in ("reduction_terms", "reduction_accumulators", "total")
    )
    loss_mismatches = sum(row["loss"]["bit_mismatches"] for row in all_path_rows)
    loss_mismatches += sum(
        row["live_gradient_objective_loss"]["bit_mismatches"] for row in rows
    )
    prediction_mismatches = sum(
        row["live_prediction_input"]["bit_mismatches"] for row in rows
    )
    signed_zero_mismatches = sum(
        row[name]["signed_zero_mismatches"]
        for row in all_path_rows
        for name in ("residual", "square", "total", "loss")
    )
    after = checkpoint_hashes()
    fallback_calls = binding.native_objective_arithmetic_fallback_calls
    exact = bool(
        prediction_mismatches == residual_mismatches == square_mismatches == 0
        and reduction_mismatches == loss_mismatches == signed_zero_mismatches == 0
        and fallback_calls == 0
        and before == after == expected_hashes
    )
    report = {
        "operation": "one-step prefix residual/MSE reduction",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "production_order": {
            "full_k_sho_numpy": "subtract -> square -> NumPy pairwise sum -> divide by 254",
            "full_k_sho_torch": "subtract -> square -> recovered aten::sum order -> locked divide by 254",
            "reflection_ball_numpy": (
                "per-step subtract -> rounded squares -> (q0+q2)+(q1+q3) "
                "dot -> sequential 127-step sum -> divide by 508"
            ),
            "reflection_ball_torch": "subtract -> square -> recovered aten::sum order -> locked divide by 508",
            "kepler5_numpy": "subtract -> square -> NumPy pairwise sum -> divide by 1016",
            "kepler5_torch": "subtract -> square -> recovered aten::sum order -> locked divide by 1016",
        },
        "deterministic_reduction_scaffolding": (
            "active Torch aten::sum implementation order retained exactly as established "
            "by the teacher-exact forward-objective audit"
        ),
        "binding_invocation_count": binding.invocations,
        "binding_invocation_count_by_system": binding.by_system,
        "logged_binding_invocation_count": int(
            log.counts["one-step prefix residual/MSE reduction"]
        ),
        "scalar_residual_values_checked": residual_values,
        "scalar_square_values_checked": square_values,
        "native_objective_arithmetic_fallback_calls": fallback_calls,
        "prediction_input_mismatches": prediction_mismatches,
        "residual_mismatches": residual_mismatches,
        "square_mismatches": square_mismatches,
        "reduction_mismatches": reduction_mismatches,
        "final_loss_mismatches": loss_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "by_system": by_system,
        "saved_states": rows,
        "controls": controls,
        "checkpoint_integrity": {
            "checkpoint_count": len(before),
            "unchanged": before == after == expected_hashes,
        },
        "next_action": "binding inventory audit only",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "prefix_residual_mse_binding": report["status"],
        "next_action": report["next_action"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["binding_invocation_count"],
        "by_system": report["binding_invocation_count_by_system"],
        "residual_values": residual_values,
        "square_values": square_values,
        "prediction_mismatches": prediction_mismatches,
        "residual_mismatches": residual_mismatches,
        "square_mismatches": square_mismatches,
        "reduction_mismatches": reduction_mismatches,
        "loss_mismatches": loss_mismatches,
        "signed_zero_mismatches": signed_zero_mismatches,
        "native_fallbacks": fallback_calls,
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
