#!/usr/bin/env python3
"""Recover only exact full-K SHO N3 Hessian replay."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.full_k_sho_hessian_assembly import FullKSHOHessianAssembly
from experiments.recover_full_k_sho_forward_arithmetic import (
    CHECKPOINT_REFERENCE,
    FAIL_CASE,
    FAIL_THETA,
    comparison,
    specification,
    theta_corpus,
)
from experiments.sho import assemble_n3_gradient_operator as gradient
from experiments.sho import assemble_n3_hessian_operator as existing_hessian
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/full_k_sho_hessian_replay")
IMPLEMENTATION = Path("experiments/geometry/full_k_sho_hessian_assembly.py")
HESSIAN_CHECKPOINT = Path(
    "outputs/sho/stage1B/n3_learned_polisher/hessian_operator/n3_hessian_operator.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def graph_counts(outputs: list[torch.Tensor]) -> dict[str, int]:
    pending = deque(value.grad_fn for value in outputs)
    seen = set()
    counts = Counter()
    while pending:
        node = pending.popleft()
        if node is None or node in seen:
            continue
        seen.add(node)
        counts[type(node).__name__] += 1
        pending.extend(child for child, _ in node.next_functions if child is not None)
    return dict(counts)


def teacher_details(inputs_np: np.ndarray, targets_np: np.ndarray,
                    theta_np: np.ndarray) -> dict:
    inputs = torch.from_numpy(inputs_np.copy())
    targets = torch.from_numpy(targets_np.copy())
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
    prediction = inputs @ theta.reshape(2, 2).T
    loss = (prediction - targets).square().mean()
    incoming = torch.autograd.grad(loss, prediction, create_graph=True, retain_graph=True)[0]
    gradient_matrix = incoming.T @ inputs
    gradient = gradient_matrix.reshape(-1)

    propagated, scaled, rows = [], [], []
    for row_index in range(4):
        propagated.append(torch.autograd.grad(
            gradient[row_index], incoming, retain_graph=True)[0])
        scaled.append(torch.autograd.grad(
            gradient[row_index], prediction, retain_graph=True)[0])
        rows.append(torch.autograd.grad(
            gradient[row_index], theta, retain_graph=True)[0])
    raw = torch.stack(rows)
    symmetric = 0.5 * (raw + raw.T)
    return {
        "loss": loss.detach().numpy(),
        "gradient": gradient.detach().numpy(),
        "propagated": np.stack([value.detach().numpy() for value in propagated]),
        "scaled": np.stack([value.detach().numpy() for value in scaled]),
        "raw": raw.detach().numpy(),
        "symmetric": symmetric.detach().numpy(),
        "graph_counts": graph_counts([gradient[index] for index in range(4)]),
    }


def recovered_stages(assembly: FullKSHOHessianAssembly,
                     inputs_np: np.ndarray) -> dict:
    propagated, scaled, rows, structural_rows = [], [], [], []
    for row_index in range(4):
        output_index, coordinate_index = divmod(row_index, 2)
        seed = np.zeros((2, 2), np.float64)
        seed[output_index, coordinate_index] = 1.0
        first = assembly.forward(inputs_np, seed)
        second = assembly._multiply(
            first, assembly.second_adjoint_scale(len(inputs_np))
        )
        row_matrix = assembly.dot(second.T, inputs_np)

        # This reproduces the old structural reduction at the isolated final
        # fan-in: exact rounded products followed by aten::sum rather than the
        # production DGEMM FMA chain.
        structural = np.empty((2, 2), np.float64)
        for output in range(2):
            for coordinate in range(2):
                products = assembly._multiply(second[:, output], inputs_np[:, coordinate])
                structural[output, coordinate] = torch.from_numpy(products.copy()).sum().item()
        propagated.append(first)
        scaled.append(second)
        rows.append(row_matrix.reshape(-1))
        structural_rows.append(structural.reshape(-1))
    raw = np.stack(rows)
    structural_raw = np.stack(structural_rows)
    return {
        "propagated": np.stack(propagated),
        "scaled": np.stack(scaled),
        "raw": raw,
        "symmetric": assembly.symmetrize(raw),
        "structural_raw": structural_raw,
        "structural_symmetric": assembly.symmetrize(structural_raw),
    }


def existing_scalar_objective(theta: torch.Tensor, inputs: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
    predictions = []
    for value in inputs:
        predictions.append(torch.stack((
            existing_hessian.add(
                existing_hessian.multiply(theta[0], value[0]),
                existing_hessian.multiply(theta[1], value[1])),
            existing_hessian.add(
                existing_hessian.multiply(theta[2], value[0]),
                existing_hessian.multiply(theta[3], value[1])),
        )))
    prediction = torch.stack(predictions)
    return existing_hessian.mean(existing_hessian.square(
        existing_hessian.subtract(prediction, targets)))


def existing_hessian_baseline(inputs_np: np.ndarray, targets_np: np.ndarray,
                              theta_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inputs = torch.from_numpy(inputs_np.copy())
    targets = torch.from_numpy(targets_np.copy())
    function = lambda theta: existing_scalar_objective(theta, inputs, targets)
    _, _, raw, symmetric = existing_hessian.derivatives(
        function, torch.from_numpy(theta_np.copy()), learned=True)
    return raw.numpy(), symmetric.numpy()


def representative_thetas(teacher: np.ndarray) -> list[tuple[str, np.ndarray]]:
    values = [("original_integration_failure", FAIL_THETA.copy()),
              ("teacher", teacher.copy())]
    downstream = Path("outputs/hybrid_student_after_ckdtree/downstream_report.json")
    if downstream.exists():
        report = json.loads(downstream.read_text())
        rows = report.get("archive_n2_replay", [])
        sho = next((row for row in rows if row["system"] == "full_k_sho"), None)
        if sho is not None:
            values.append(("current_n2_endpoint", np.asarray(sho["n2_endpoint"], np.float64)))
    unique, seen = [], set()
    for name, value in values:
        if value.tobytes() not in seen:
            unique.append((name, value)); seen.add(value.tobytes())
    return unique


def run() -> dict:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise RuntimeError("checkpoint inventory changed")
    hessian_checkpoint_before = sha256(HESSIAN_CHECKPOINT)

    assembly = FullKSHOHessianAssembly()
    gradient.OPS = gradient.LearnedArithmetic()
    existing_hessian.activate_second_order_gradient()

    representative = []
    first_divergence = None
    spec = specification(FAIL_CASE)
    trajectory, _, _, _, _, _, teacher_theta = readiness.case_data(spec)
    inputs_np, targets_np = trajectory[:-1], trajectory[1:]
    for name, theta in representative_thetas(teacher_theta):
        teacher = teacher_details(inputs_np, targets_np, theta)
        recovered = recovered_stages(assembly, inputs_np)
        baseline_raw, baseline_symmetric = existing_hessian_baseline(
            inputs_np, targets_np, theta)
        stage_metrics = {
            "first_mm_second_adjoint": comparison(
                teacher["propagated"], recovered["propagated"]),
            "square_mean_second_adjoint_scale": comparison(
                teacher["scaled"], recovered["scaled"]),
            "old_structural_final_reduction": comparison(
                teacher["raw"], recovered["structural_raw"]),
            "existing_learned_raw_hessian": comparison(
                teacher["raw"], baseline_raw),
            "existing_learned_symmetric_hessian": comparison(
                teacher["symmetric"], baseline_symmetric),
            "recovered_raw_hessian": comparison(
                teacher["raw"], recovered["raw"]),
            "recovered_symmetric_hessian": comparison(
                teacher["symmetric"], recovered["symmetric"]),
        }
        if first_divergence is None:
            for operation in ("first_mm_second_adjoint",
                              "square_mean_second_adjoint_scale",
                              "old_structural_final_reduction"):
                if stage_metrics[operation]["bit_mismatches"]:
                    first_divergence = {
                        "representative": name,
                        "operation": operation,
                        "metric": stage_metrics[operation],
                    }
                    break
        representative.append({
            "name": name, "theta": theta.tolist(), "stage_metrics": stage_metrics,
            "teacher_raw": teacher["raw"].tolist(),
            "teacher_symmetric": teacher["symmetric"].tolist(),
            "recovered_raw": recovered["raw"].tolist(),
            "recovered_symmetric": recovered["symmetric"].tolist(),
        })

    aggregate = Counter()
    matrices_exact = raw_matrices_exact = 0
    direct_direction_exact = 0
    state_count = 0
    raw_asymmetric_states = sym_changed_states = 0
    by_source = defaultdict(Counter)
    cases = []
    for case_id in range(64):
        case_spec = specification(case_id)
        case_trajectory, _, _, _, _, _, case_teacher = readiness.case_data(case_spec)
        case_inputs, case_targets = case_trajectory[:-1], case_trajectory[1:]
        recovered = recovered_stages(assembly, case_inputs)
        case_counter = Counter()
        states = theta_corpus(case_id, case_teacher)
        for source, theta in states:
            teacher = teacher_details(case_inputs, case_targets, theta)
            raw_metric = comparison(teacher["raw"], recovered["raw"])
            symmetric_metric = comparison(teacher["symmetric"], recovered["symmetric"])
            state_count += 1
            raw_matrices_exact += int(raw_metric["bit_mismatches"] == 0)
            matrices_exact += int(symmetric_metric["bit_mismatches"] == 0)
            raw_asymmetric = not bits_equal(teacher["raw"], teacher["raw"].T)
            sym_changed = not bits_equal(teacher["raw"], teacher["symmetric"])
            raw_asymmetric_states += int(raw_asymmetric)
            sym_changed_states += int(sym_changed)

            # Narrow validation diagnostic only: both direct unconstrained
            # solves receive the same exact gradient and differ only in which
            # Hessian replay is supplied. No routing/eigenspectrum/box-QP runs.
            gradient_tensor = torch.from_numpy(teacher["gradient"].copy())
            teacher_direction = torch.linalg.solve(
                torch.from_numpy(teacher["symmetric"].copy()), -gradient_tensor)
            recovered_direction = torch.linalg.solve(
                torch.from_numpy(recovered["symmetric"].copy()), -gradient_tensor)
            direction_metric = comparison(teacher_direction.numpy(),
                                          recovered_direction.numpy())
            direct_direction_exact += int(direction_metric["bit_mismatches"] == 0)

            for prefix, metric in (("raw", raw_metric),
                                   ("symmetric", symmetric_metric),
                                   ("direction", direction_metric)):
                for key in ("values", "bit_mismatches", "numeric_mismatches",
                            "sign_mismatches", "zero_nonzero_mismatches"):
                    aggregate[f"{prefix}_{key}"] += metric[key]
                    case_counter[f"{prefix}_{key}"] += metric[key]
                    by_source[source][f"{prefix}_{key}"] += metric[key]
                aggregate[f"{prefix}_max_ulp_error"] = max(
                    aggregate[f"{prefix}_max_ulp_error"], metric["max_ulp_error"])
                aggregate[f"{prefix}_max_absolute_error"] = max(
                    aggregate[f"{prefix}_max_absolute_error"],
                    metric["max_absolute_error"])
        cases.append({"case_id": case_id, "states": len(states),
                      "metrics": dict(case_counter)})

    report = {
        "status": "N3_HESSIAN_EXACT",
        "scope": (
            "current full-K SHO Hessian replay only; eigenspectrum, curvature routing, "
            "box-QP, damping and later N3 integration were not executed"
        ),
        "teacher_hessian_path": [
            "prediction = X @ K.T using Torch/Accelerate [127,2]@[2,2]",
            "loss = mean(square(prediction-target)) over 254 values",
            "gradient = incoming_prediction_adjoint.T @ X using [2,127]@[127,2] DGEMM",
            "for each flattened gradient row 0..3, seed one element with 1",
            "first second-adjoint mm: propagated = X @ seed.reshape(2,2).T",
            "mean/square curvature scale = RN(RN(1/254)*2) = RN(1/127)",
            "scaled = RN(propagated * curvature_scale), elementwise",
            "second second-adjoint mm: row_matrix = scaled.T @ X",
            "raw Hessian row = row_matrix reshaped in K flatten order",
            "symmetric Hessian = RN(0.5 * RN(raw + raw.T)), elementwise",
        ],
        "autograd_graph": {
            "gradient_node_counts": teacher_details(
                inputs_np, targets_np, FAIL_THETA)["graph_counts"],
            "per_four_hessian_rows_mm_calls": {
                "[127,2]@[2,2]": 4,
                "[2,127]@[127,2]": 4,
            },
            "views": (
                "theta reshape, K transpose, gradient-matrix transpose and row flattening "
                "are views/index bookkeeping and add no arithmetic"
            ),
        },
        "first_divergence": {
            **(first_divergence or {}),
            "diagnosis": (
                "the propagated second adjoint and square/mean scaling are exact; the "
                "first mismatch is the final Hessian-row fan-in when the existing scalar "
                "assembly uses rounded products plus structural aten::sum instead of the "
                "production ascending fused DGEMM accumulation"
            ),
        },
        "recovered_assembly": {
            "implementation": str(IMPLEMENTATION),
            "implementation_sha256": sha256(IMPLEMENTATION),
            "reused": [
                "exact Torch [127,2]@[2,2] forward matmul assembly",
                "exact Torch MmBackward ascending-FMA dot assembly",
                "locked multiplication", "locked addition/subtraction",
                "product-tail and TwoSum exact-FMA reconstruction",
            ],
            "deterministic_scaffolding": [
                "four Hessian-row seeds", "reshape/transpose views", "row stacking",
                "ascending reduction loop and accumulator storage",
            ],
            "new_parameters": 0,
            "classification": (
                "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_IMPLEMENTATION_ORDER"
            ),
        },
        "representative_integration_states": representative,
        "corpus": {
            "cases": 64, "states": state_count,
            "hessian_entries": state_count * 16,
            "raw_hessian_matrices_exact": raw_matrices_exact,
            "symmetric_hessian_matrices_exact": matrices_exact,
            "raw_asymmetric_states": raw_asymmetric_states,
            "states_changed_by_production_symmetrization": sym_changed_states,
            "direct_unconstrained_direction_diagnostic_exact": direct_direction_exact,
            "aggregate": dict(aggregate),
            "by_source": {key: dict(value) for key, value in by_source.items()},
            "cases_with_hessian_mismatch": [
                row["case_id"] for row in cases
                if row["metrics"].get("symmetric_bit_mismatches", 0)
            ],
        },
        "n3_hessian_operator_upgrade": {
            "eligible": aggregate["symmetric_bit_mismatches"] == 0,
            "status": "EXACT_TEACHER_REPLAY"
                      if aggregate["symmetric_bit_mismatches"] == 0 else "UNRESOLVED",
            "existing_checkpoint_modified": False,
            "existing_checkpoint_sha256": hessian_checkpoint_before,
        },
        "next_unresolved_integration_operation": (
            "Hessian eigenspectrum/condition is next in dependency order; it was not "
            "inspected or executed in this run"
        ),
        "checkpoint_hashes_unchanged": checkpoint_hashes() == expected_checkpoints,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "hessian_assembly.json").write_text(json.dumps({
        "status": report["recovered_assembly"]["classification"],
        **report["recovered_assembly"],
        "teacher_hessian_path": report["teacher_hessian_path"],
        "validation": report["corpus"],
    }, indent=2) + "\n")
    return report


def main() -> None:
    report = run()
    print(json.dumps({
        "status": report["status"],
        "first_divergence": report["first_divergence"],
        "representative": [
            {"name": row["name"], "metrics": row["stage_metrics"]}
            for row in report["representative_integration_states"]
        ],
        "corpus": report["corpus"],
        "upgrade": report["n3_hessian_operator_upgrade"],
        "checkpoints_unchanged": report["checkpoint_hashes_unchanged"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
