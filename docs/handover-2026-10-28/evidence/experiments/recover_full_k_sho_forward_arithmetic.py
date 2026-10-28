#!/usr/bin/env python3
"""Recover only the full-K SHO forward objective's float64 arithmetic order."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.torch_matmul2_fma_assembly import TorchMatmul2FMAAssembly
from experiments.sho import assemble_n3_gradient_operator as learned
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/full_k_sho_forward_arithmetic")
CHECKPOINT_REFERENCE = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FAIL_CASE = 0
FAIL_THETA = np.asarray([
    2.784315259745031e-17, 0.9999999999999999,
    -1.0, 1.4352654443122448,
], np.float64)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, np.float64).view(np.uint64)
    return np.where(raw >> np.uint64(63), ~raw,
                    raw ^ (np.uint64(1) << np.uint64(63)))


def comparison(expected, actual) -> dict:
    left = np.asarray(expected, np.float64).reshape(-1)
    right = np.asarray(actual, np.float64).reshape(-1)
    different = left.view(np.uint64) != right.view(np.uint64)
    ulp = np.maximum(ordered(left), ordered(right)) - np.minimum(ordered(left), ordered(right))
    return {
        "values": int(len(left)), "bit_mismatches": int(np.count_nonzero(different)),
        "numeric_mismatches": int(np.count_nonzero(left != right)),
        "sign_mismatches": int(np.count_nonzero(np.signbit(left) != np.signbit(right))),
        "zero_nonzero_mismatches": int(np.count_nonzero((left == 0.0) != (right == 0.0))),
        "max_absolute_error": float(np.max(np.abs(left - right))) if len(left) else 0.0,
        "max_ulp_error": int(np.max(ulp)) if len(left) else 0,
    }


def add_metric(total: Counter, name: str, expected, actual) -> dict:
    row = comparison(expected, actual)
    for key in ("values", "bit_mismatches", "numeric_mismatches",
                "sign_mismatches", "zero_nonzero_mismatches"):
        total[f"{name}_{key}"] += row[key]
    total[f"{name}_max_ulp_error"] = max(total[f"{name}_max_ulp_error"], row["max_ulp_error"])
    total[f"{name}_max_absolute_error"] = max(
        float(total[f"{name}_max_absolute_error"]), row["max_absolute_error"])
    return row


def specification(case_id: int) -> dict:
    manifest = json.loads(readiness.MANIFEST.read_text())
    return next(row for row in manifest["cases"]
                if row["system"] == "full_k_sho" and int(row["case_id"]) == case_id)


def saved_case(case_id: int) -> dict:
    return json.loads((readiness.CASES / "full_k_sho" /
                       f"case_{case_id:05d}" / "result.json").read_text())


def theta_corpus(case_id: int, teacher: np.ndarray) -> list[tuple[str, np.ndarray]]:
    branch = saved_case(case_id)["successful_branch"]
    trace = np.load(branch["trace_npz"])
    rows = [("saved_n3", np.asarray(value, np.float64).copy())
            for value in trace["n3_theta"]]
    rows.append(("teacher", teacher.copy()))
    for axis in range(4):
        lower = teacher.copy(); lower[axis] = np.nextafter(lower[axis], -np.inf)
        upper = teacher.copy(); upper[axis] = np.nextafter(upper[axis], np.inf)
        rows.extend((("teacher_nextafter", lower), ("teacher_nextafter", upper)))
    unique = []
    seen = set()
    for source, value in rows:
        key = value.tobytes()
        if key not in seen:
            unique.append((source, value)); seen.add(key)
    return unique


def batch_length_controls(assembly: TorchMatmul2FMAAssembly) -> dict:
    """Exercise the two-row vector body and odd scalar tail independently."""
    rng = np.random.default_rng(20_261_008)
    lengths = [*range(1, 34), 63, 64, 65, 126, 127, 128, 129, 255, 256, 257]
    values = mismatches = 0
    per_length = []
    for length in lengths:
        local = 0
        for _ in range(8):
            inputs = rng.normal(size=(length, 2))
            matrix = rng.normal(size=(2, 2))
            expected = (torch.from_numpy(inputs) @ torch.from_numpy(matrix).T).numpy()
            actual = assembly(inputs, matrix)
            local += int(np.count_nonzero(
                expected.view(np.uint64) != actual.view(np.uint64)))
            values += expected.size
        mismatches += local
        per_length.append({"rows": length, "bit_mismatches": local})
    return {"lengths": len(lengths), "random_matrices_per_length": 8,
            "values": values, "bit_mismatches": mismatches,
            "failing_lengths": [row for row in per_length if row["bit_mismatches"]]}


ASSEMBLY: TorchMatmul2FMAAssembly


class ExactForwardMatmul2(torch.autograd.Function):
    """Correct forward with the existing scalar learned adjoint reduction."""

    @staticmethod
    def forward(ctx, inputs, theta):
        ctx.save_for_backward(inputs, theta)
        result = ASSEMBLY(inputs.detach().numpy(), theta.detach().numpy().reshape(2, 2))
        return torch.from_numpy(result.copy())

    @staticmethod
    def backward(ctx, incoming):
        inputs, theta = ctx.saved_tensors
        # This is the existing scalar learned assembly semantics: exact locked
        # products followed by Torch's structural sum bookkeeping.  It is kept
        # deliberately distinct from production MmBackward's BLAS dot kernel.
        terms = []
        for output in range(2):
            for coordinate in range(2):
                products = learned.OPS.multiply(incoming[:, output], inputs[:, coordinate])
                terms.append(products.sum())
        return None, torch.stack(terms).reshape_as(theta)


def learned_objective(inputs: torch.Tensor, targets: torch.Tensor,
                      theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = ExactForwardMatmul2.apply(inputs, theta)
    residual = learned.subtract(prediction, targets)
    squares = learned.square(residual)
    return learned.mean(squares), prediction


def run() -> dict:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise RuntimeError("checkpoint inventory changed")
    global ASSEMBLY
    ASSEMBLY = TorchMatmul2FMAAssembly()
    learned.OPS = learned.LearnedArithmetic()
    length_controls = batch_length_controls(ASSEMBLY)

    total = Counter()
    by_source = defaultdict(Counter)
    cases = []
    failing_detail = None
    first_gradient_mismatch = None
    state_count = forward_exact_states = gradient_exact_states = 0
    for case_id in range(64):
        spec = specification(case_id)
        trajectory, _, _, _, _, _, teacher = readiness.case_data(spec)
        inputs_np, targets_np = trajectory[:-1], trajectory[1:]
        inputs = torch.from_numpy(inputs_np.copy())
        targets = torch.from_numpy(targets_np.copy())
        case_counter = Counter()
        states = theta_corpus(case_id, teacher)
        for source, theta_np in states:
            state_count += 1
            matrix = theta_np.reshape(2, 2)
            teacher_theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
            teacher_prediction = inputs @ teacher_theta.reshape(2, 2).T
            teacher_residual = teacher_prediction - targets
            teacher_squares = teacher_residual.square()
            teacher_loss = teacher_squares.mean()
            teacher_prediction_adjoint = torch.autograd.grad(
                teacher_loss, teacher_prediction, retain_graph=True)[0].detach()
            teacher_gradient = torch.autograd.grad(teacher_loss, teacher_theta)[0].detach()

            # Previous scalar assembly: both products round before addition.
            first = learned.OPS.multiply(inputs[:, 0, None],
                                         torch.from_numpy(matrix[:, 0].copy()))
            second = learned.OPS.multiply(inputs[:, 1, None],
                                          torch.from_numpy(matrix[:, 1].copy()))
            separate_prediction = learned.OPS.add(first, second).detach().numpy()

            corrected_prediction_np = ASSEMBLY(inputs_np, matrix)
            corrected_prediction = torch.from_numpy(corrected_prediction_np.copy())
            corrected_residual = learned.OPS.subtract(corrected_prediction, targets)
            corrected_squares = learned.OPS.multiply(corrected_residual, corrected_residual)
            corrected_loss = corrected_squares.mean()

            local = Counter()
            baseline = add_metric(local, "separate_prediction",
                                  teacher_prediction.detach().numpy(), separate_prediction)
            prediction = add_metric(local, "prediction",
                                    teacher_prediction.detach().numpy(), corrected_prediction_np)
            residual = add_metric(local, "residual",
                                  teacher_residual.detach().numpy(), corrected_residual.detach().numpy())
            squares = add_metric(local, "squares",
                                 teacher_squares.detach().numpy(), corrected_squares.detach().numpy())
            loss = add_metric(local, "loss", [float(teacher_loss.detach())],
                              [float(corrected_loss.detach())])
            # Mean/reduction characterization: production mean on the corrected
            # square tensor versus the explicit production sum then division.
            sum_div = corrected_squares.sum() / corrected_squares.numel()
            add_metric(local, "mean_vs_sum_div", [float(teacher_loss.detach())], [float(sum_div)])

            student_theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)
            student_loss, student_prediction = learned_objective(inputs, targets, student_theta)
            student_prediction_adjoint = torch.autograd.grad(
                student_loss, student_prediction, retain_graph=True)[0].detach()
            student_gradient = torch.autograd.grad(student_loss, student_theta)[0].detach()
            add_metric(local, "prediction_adjoint", teacher_prediction_adjoint.numpy(),
                       student_prediction_adjoint.numpy())
            # This is the isolated production adjoint target.  It is retained
            # only as a diagnostic identity, not substituted into the student.
            production_dot = teacher_prediction_adjoint.T @ inputs
            add_metric(local, "production_mm_backward_identity", teacher_gradient.numpy(),
                       production_dot.detach().numpy().reshape(-1))
            gradient_row = add_metric(local, "gradient", teacher_gradient.numpy(),
                                      student_gradient.numpy())

            for key, value in local.items():
                if key.endswith("max_ulp_error") or key.endswith("max_absolute_error"):
                    total[key] = max(total[key], value)
                    by_source[source][key] = max(by_source[source][key], value)
                    case_counter[key] = max(case_counter[key], value)
                else:
                    total[key] += value; by_source[source][key] += value; case_counter[key] += value
            exact_forward = not any((prediction["bit_mismatches"], residual["bit_mismatches"],
                                     squares["bit_mismatches"], loss["bit_mismatches"]))
            forward_exact_states += int(exact_forward)
            gradient_exact_states += int(gradient_row["bit_mismatches"] == 0)
            if gradient_row["bit_mismatches"] and first_gradient_mismatch is None:
                first_gradient_mismatch = {
                    "case_id": case_id, "source": source, "theta": theta_np.tolist(),
                    "teacher_gradient": teacher_gradient.tolist(),
                    "student_gradient": student_gradient.tolist(),
                    "metric": gradient_row,
                    "smallest_remaining_map": (
                        "for each K entry: (incoming residual adjoints[127], observed input "
                        "coordinate[127]) -> production MmBackward float64 dot accumulation"
                    ),
                }
            if case_id == FAIL_CASE and np.array_equal(theta_np.view(np.uint64),
                                                       FAIL_THETA.view(np.uint64)):
                failing_detail = {
                    "theta": theta_np.tolist(),
                    "teacher_prediction_vs_separate": baseline,
                    "teacher_prediction_vs_recovered": prediction,
                    "teacher_loss": float(teacher_loss.detach()),
                    "separate_loss": float(torch.from_numpy(
                        ((separate_prediction - targets_np) ** 2)).mean()),
                    "recovered_loss": float(corrected_loss.detach()),
                    "teacher_gradient": teacher_gradient.tolist(),
                    "corrected_forward_student_gradient": student_gradient.tolist(),
                    "gradient_metric": gradient_row,
                }
        cases.append({"case_id": case_id, "states": len(states),
                      "metrics": dict(case_counter)})

    report = {
        "status": "FORWARD_OBJECTIVE_EXACT_GRADIENT_ADJOINT_REDUCTION_UNRESOLVED",
        "scope": "full-K SHO float64 forward objective only; no Hessian or later N3 integration",
        "environment": {
            "torch": torch.__version__, "numpy": np.__version__, "machine": "arm64",
            "blas": "Accelerate (from torch.__config__)",
        },
        "production_order": [
            "K = theta.reshape(2,2); K.T is a transpose view",
            "prediction = torch_inputs @ K.T, shapes [127,2] @ [2,2]",
            "two-row vector body: p0=RN(x0*k0); prediction=RN_fma(x1*k1+p0)",
            "odd final-row scalar tail: prediction=RN(RN(x0*k0)+RN(x1*k1))",
            "residual = RN(prediction-target), elementwise row-major [127,2]",
            "squares = RN(residual*residual), elementwise",
            "loss = aten::mean(squares) over all 254 contiguous values",
        ],
        "first_divergence": (
            "the prior scalar path used RN(RN(x0*k0)+RN(x1*k1)); production fuses "
            "the second product with the add"
        ),
        "recovered_assembly": {
            "implementation": "experiments/geometry/torch_matmul2_fma_assembly.py",
            "implementation_sha256": sha256(Path(
                "experiments/geometry/torch_matmul2_fma_assembly.py")),
            "formula": ("two-row body RN_fma(x1*k1 + RN(x0*k0)) reconstructed by "
                        "product-tail + TwoSum; odd scalar tail uses two products + add"),
            "reused": ["locked multiplication", "locked addition/subtraction",
                       "validated exact product-tail", "TwoSum assembly",
                       "production aten::mean reduction/state scaffold"],
            "new_parameters": 0,
        },
        "corpus": {
            "cases": 64, "states": state_count,
            "sources": {key: int(value["loss_values"]) for key, value in by_source.items()},
            "forward_exact_states": forward_exact_states,
            "gradient_exact_states_after_forward_repair": gradient_exact_states,
        },
        "batch_length_controls": length_controls,
        "aggregate": dict(total),
        "by_source": {key: dict(value) for key, value in by_source.items()},
        "failing_state": failing_detail,
        "gradient_diagnosis": {
            "forward_loss_exact_before_gradient": True,
            "prediction_adjoint_bit_mismatches": total["prediction_adjoint_bit_mismatches"],
            "production_mm_backward_identity_bit_mismatches": total[
                "production_mm_backward_identity_bit_mismatches"],
            "gradient_not_automatically_exact": total["gradient_bit_mismatches"] > 0,
            "first_mismatch": first_gradient_mismatch,
            "earliest_remaining_operation": (
                "production MmBackward dot-product accumulation/reduction order for the four "
                "K gradients; locked per-product arithmetic plus structural torch.sum is not "
                "bit-identical"
            ),
        },
        "cases": cases,
        "checkpoint_hashes_unchanged": checkpoint_hashes() == expected_checkpoints,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "forward_assembly.json").write_text(json.dumps({
        "status": "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_IMPLEMENTATION_ORDER",
        **report["recovered_assembly"],
        "validation": {"states": state_count, "exact_states": forward_exact_states,
                       "loss_bit_mismatches": total["loss_bit_mismatches"]},
    }, indent=2) + "\n")
    return report


def main():
    report = run()
    print(json.dumps({
        "status": report["status"], "corpus": report["corpus"],
        "forward": {key: report["aggregate"][key] for key in (
            "separate_prediction_bit_mismatches", "prediction_bit_mismatches",
            "residual_bit_mismatches", "squares_bit_mismatches", "loss_bit_mismatches")},
        "gradient": {key: report["aggregate"][key] for key in (
            "gradient_values", "gradient_bit_mismatches", "gradient_sign_mismatches",
            "gradient_zero_nonzero_mismatches", "gradient_max_ulp_error")},
        "failing_state": report["failing_state"],
        "checkpoints_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
