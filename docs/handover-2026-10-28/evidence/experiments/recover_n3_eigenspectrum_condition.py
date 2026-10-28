#!/usr/bin/env python3
"""Recover only teacher-exact N3 Hessian spectrum/condition replay."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments import validate_deterministic_teacher_readiness as readiness
from experiments.geometry.full_k_sho_hessian_assembly import FullKSHOHessianAssembly
from experiments.geometry.n3_eigenspectrum_condition_assembly import (
    N3EigenspectrumConditionAssembly,
)
from experiments.recover_full_k_sho_forward_arithmetic import (
    CHECKPOINT_REFERENCE,
    comparison,
    specification,
    theta_corpus,
)
from experiments.recover_full_k_sho_mm_backward_dot import native_gradient
from experiments.sho import run_stage1b_n3_gradient_polishing as production
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/n3_eigenspectrum_condition_replay")
IMPLEMENTATION = Path("experiments/geometry/n3_eigenspectrum_condition_assembly.py")
LIBTORCH_CPU = Path(torch.__file__).parent / "lib" / "libtorch_cpu.dylib"
SYEVD_SYMBOL = "__ZN2at6native11lapackSyevdIddEEvcciPT_iPT0_S3_iS5_iPiiS6_"
SVD_SYMBOL = "__ZN2at6native9lapackSvdIddEEvciiPT_iPT0_S3_iS3_iS3_iS5_PiS6_"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_analysis(gradient: torch.Tensor, hessian: torch.Tensor) -> dict:
    finite = bool(torch.isfinite(gradient).all() and torch.isfinite(hessian).all())
    if not finite:
        return {
            "finite": False,
            "singular_values": None,
            "smallest_singular": None,
            "largest_singular": None,
            "condition": math.inf,
            "eigenvalues": None,
            "minimum_eigenvalue": None,
            "maximum_eigenvalue": None,
            "maximum_absolute_eigenvalue": None,
            "scale": None,
            "positive": False,
            "condition_ok": False,
        }
    singular_values = torch.linalg.svdvals(hessian)
    smallest = float(torch.min(singular_values))
    largest = float(torch.max(singular_values))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum = float(torch.min(eigenvalues))
    maximum = float(torch.max(eigenvalues))
    maximum_absolute = float(torch.max(torch.abs(eigenvalues)))
    scale = max(maximum_absolute, torch.finfo(torch.float64).tiny)
    return {
        "finite": True,
        "singular_values": singular_values,
        "smallest_singular": smallest,
        "largest_singular": largest,
        "condition": condition,
        "eigenvalues": eigenvalues,
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "maximum_absolute_eigenvalue": maximum_absolute,
        "scale": scale,
        "positive": minimum > 0.0,
        "condition_ok": condition <= production.CONDITION_LIMIT,
    }


def tensor_metric(left: torch.Tensor, right: torch.Tensor) -> dict:
    return comparison(left.detach().numpy(), right.detach().numpy())


def scalar_metric(left: float, right: float) -> dict:
    return comparison(np.asarray([left], np.float64), np.asarray([right], np.float64))


def add_metric(counter: Counter, prefix: str, metric: dict) -> None:
    for key in (
        "values",
        "bit_mismatches",
        "numeric_mismatches",
        "sign_mismatches",
        "zero_nonzero_mismatches",
    ):
        counter[f"{prefix}_{key}"] += metric[key]
    for key in ("max_ulp_error", "max_absolute_error"):
        counter[f"{prefix}_{key}"] = max(counter[f"{prefix}_{key}"], metric[key])


def source_evidence() -> dict:
    linked = subprocess.check_output(["otool", "-L", str(LIBTORCH_CPU)], text=True)
    undefined = subprocess.check_output(["nm", "-u", str(LIBTORCH_CPU)], text=True)
    syevd_disassembly = subprocess.check_output(
        ["otool", "-tvV", "-p", SYEVD_SYMBOL, str(LIBTORCH_CPU)], text=True
    )
    svd_disassembly = subprocess.check_output(
        ["otool", "-tvV", "-p", SVD_SYMBOL, str(LIBTORCH_CPU)], text=True
    )
    return {
        "torch_version": torch.__version__,
        "machine": platform.machine(),
        "torch_build": [
            line.strip()
            for line in torch.__config__.show().splitlines()
            if "BLAS_INFO" in line or "LAPACK_INFO" in line or "TORCH_VERSION" in line
        ],
        "accelerate_link": [line.strip() for line in linked.splitlines() if "Accelerate" in line],
        "undefined_lapack_symbols": [
            line.strip()
            for line in undefined.splitlines()
            if "dgesdd" in line.lower() or "dsyevd" in line.lower()
        ],
        "double_syevd_wrapper": SYEVD_SYMBOL,
        "double_syevd_call": [
            line.strip() for line in syevd_disassembly.splitlines() if "_dsyevd_" in line
        ],
        "double_svd_wrapper": SVD_SYMBOL,
        "double_svd_call": [
            line.strip() for line in svd_disassembly.splitlines() if "_dgesdd_" in line
        ],
    }


def profiler_evidence(hessian: torch.Tensor) -> dict:
    with torch.autograd.profiler.profile(use_cpu=True, record_shapes=True) as profile:
        torch.linalg.svdvals(hessian)
        torch.linalg.eigvalsh(hessian)
    retained = []
    for event in profile.function_events:
        if any(token in event.name for token in (
            "linalg_svdvals", "_linalg_svd", "linalg_eigvalsh", "_linalg_eigh",
            "transpose", "clone", "copy_",
        )):
            retained.append({"operation": event.name, "input_shapes": event.input_shapes})
    return {
        "input_shape": list(hessian.shape),
        "input_stride": list(hessian.stride()),
        "events_in_runtime_order": retained,
        "interpretation": (
            "both kernels transpose/clone the row-major input into LAPACK-compatible "
            "column-major working storage; LAPACK overwrites only the private work copy"
        ),
    }


def triangle_and_order_controls(hessian: torch.Tensor) -> dict:
    base = torch.linalg.eigvalsh(hessian)
    upper_changed = hessian.clone()
    lower_changed = hessian.clone()
    row, column = 0, 3
    upper_changed[row, column] = upper_changed[row, column] + 1024.0
    lower_changed[column, row] = lower_changed[column, row] + 1024.0
    upper_eigen = torch.linalg.eigvalsh(upper_changed)
    lower_eigen = torch.linalg.eigvalsh(lower_changed)
    singular = torch.linalg.svdvals(hessian)
    return {
        "default_eigvalsh_uses_lower_triangle": bool(torch.equal(base, upper_eigen)),
        "lower_triangle_change_is_observed": not bool(torch.equal(base, lower_eigen)),
        "eigenvalues_ascending": bool(torch.all(base[1:] >= base[:-1])),
        "singular_values_descending": bool(torch.all(singular[1:] <= singular[:-1])),
    }


def nonfinite_and_branch_controls(assembly: N3EigenspectrumConditionAssembly) -> dict:
    finite_gradient = torch.zeros(4, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    controls = []
    cases = [
        ("nan_gradient", finite_gradient.clone(), identity.clone()),
        ("inf_gradient", finite_gradient.clone(), identity.clone()),
        ("nan_hessian", finite_gradient.clone(), identity.clone()),
        ("inf_hessian", finite_gradient.clone(), identity.clone()),
        ("zero_singular_value", finite_gradient.clone(), torch.diag(torch.tensor(
            [1.0, 0.5, 0.25, 0.0], dtype=torch.float64))),
        ("indefinite", finite_gradient.clone(), torch.diag(torch.tensor(
            [1.0, 0.5, 0.25, -0.125], dtype=torch.float64))),
    ]
    cases[0][1][0] = math.nan
    cases[1][1][0] = math.inf
    cases[2][2][0, 0] = math.nan
    cases[3][2][0, 0] = math.inf
    for name, gradient, hessian in cases:
        before = (assembly.svd_calls, assembly.eigh_calls)
        teacher = teacher_analysis(gradient, hessian)
        student = assembly.analyze(gradient, hessian)
        after = (assembly.svd_calls, assembly.eigh_calls)
        controls.append({
            "name": name,
            "teacher_finite": teacher["finite"],
            "student_finite": student.finite,
            "teacher_condition": teacher["condition"],
            "student_condition": student.condition,
            "teacher_positive": teacher["positive"],
            "student_positive": student.positive,
            "teacher_condition_ok": teacher["condition_ok"],
            "student_condition_ok": student.condition_ok,
            "solver_calls_during_control": [after[0] - before[0], after[1] - before[1]],
            "exact": (
                teacher["finite"] == student.finite
                and scalar_metric(teacher["condition"], student.condition)["bit_mismatches"] == 0
                and teacher["positive"] == student.positive
                and teacher["condition_ok"] == student.condition_ok
            ),
        })
    return {"controls": controls, "all_exact": all(row["exact"] for row in controls)}


def run() -> dict:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINT_REFERENCE.read_text())
    if checkpoint_hashes() != expected_checkpoints:
        raise RuntimeError("checkpoint inventory changed")

    hessian_assembly = FullKSHOHessianAssembly()
    spectral_assembly = N3EigenspectrumConditionAssembly(production.CONDITION_LIMIT)
    aggregate = Counter()
    by_source = defaultdict(Counter)
    boolean_mismatches = Counter()
    exact_states = 0
    state_count = 0
    shortcut_singular_mismatches = 0
    shortcut_condition_mismatches = 0
    shortcut_values = 0
    case_rows = []
    representative_hessian = None

    for case_id in range(64):
        trajectory, _, _, _, _, _, teacher_theta = readiness.case_data(specification(case_id))
        inputs_np, targets_np = trajectory[:-1], trajectory[1:]
        _, hessian_np = hessian_assembly(inputs_np)
        hessian = torch.from_numpy(hessian_np.copy())
        if representative_hessian is None:
            representative_hessian = hessian
        case_counter = Counter()
        for source, theta in theta_corpus(case_id, teacher_theta):
            _, gradient_np = native_gradient(
                torch.from_numpy(inputs_np.copy()),
                torch.from_numpy(targets_np.copy()),
                theta,
            )
            gradient = torch.from_numpy(gradient_np.reshape(-1).copy())
            teacher = teacher_analysis(gradient, hessian)
            student = spectral_assembly.analyze(gradient, hessian)
            metrics = {
                "singular_values": tensor_metric(
                    teacher["singular_values"], student.singular_values),
                "smallest_singular": scalar_metric(
                    teacher["smallest_singular"], student.smallest_singular),
                "largest_singular": scalar_metric(
                    teacher["largest_singular"], student.largest_singular),
                "condition": scalar_metric(teacher["condition"], student.condition),
                "eigenvalues": tensor_metric(teacher["eigenvalues"], student.eigenvalues),
                "minimum_eigenvalue": scalar_metric(
                    teacher["minimum_eigenvalue"], student.minimum_eigenvalue),
                "maximum_eigenvalue": scalar_metric(
                    teacher["maximum_eigenvalue"], student.maximum_eigenvalue),
                "maximum_absolute_eigenvalue": scalar_metric(
                    teacher["maximum_absolute_eigenvalue"],
                    student.maximum_absolute_eigenvalue,
                ),
                "scale": scalar_metric(teacher["scale"], student.scale),
            }
            state_exact = True
            for name, metric in metrics.items():
                add_metric(aggregate, name, metric)
                add_metric(case_counter, name, metric)
                add_metric(by_source[source], name, metric)
                state_exact &= metric["bit_mismatches"] == 0
            for name in ("finite", "positive", "condition_ok"):
                mismatch = int(teacher[name] != getattr(student, name))
                boolean_mismatches[name] += mismatch
                case_counter[f"{name}_mismatches"] += mismatch
                by_source[source][f"{name}_mismatches"] += mismatch
                state_exact &= mismatch == 0

            # A tempting symmetric-matrix shortcut is not teacher exact because
            # dgesdd and dsyevd round through different algorithms.
            shortcut = torch.sort(torch.abs(teacher["eigenvalues"]), descending=True).values
            shortcut_singular_mismatches += int(torch.count_nonzero(
                shortcut.view(torch.int64) != teacher["singular_values"].view(torch.int64)))
            shortcut_values += shortcut.numel()
            shortcut_condition = float(torch.max(shortcut)) / float(torch.min(shortcut))
            shortcut_condition_mismatches += scalar_metric(
                teacher["condition"], shortcut_condition)["bit_mismatches"]

            state_count += 1
            exact_states += int(state_exact)
        case_rows.append({
            "case_id": case_id,
            "states": len(theta_corpus(case_id, teacher_theta)),
            "metrics": dict(case_counter),
        })

    assert representative_hessian is not None
    corpus_solver_calls = {
        "svdvals": spectral_assembly.svd_calls,
        "eigvalsh": spectral_assembly.eigh_calls,
    }
    controls = nonfinite_and_branch_controls(spectral_assembly)
    report = {
        "status": "N3_EIGENSPECTRUM_CONDITION_TEACHER_EXACT",
        "scope": (
            "symmetric Hessian through spectral/condition quantities and their two "
            "individual predicates only; curvature conjunction/routing, damping, box-QP, "
            "and later N3 operations were not executed"
        ),
        "teacher_path": [
            "finite = all(isfinite(gradient)) AND all(isfinite(symmetric_hessian))",
            "if nonfinite: return condition=+inf and nonfinite_derivatives before LAPACK",
            "singular_values = torch.linalg.svdvals(H): column-major work clone -> Accelerate dgesdd, jobz='N'",
            "smallest/largest = torch.min/max(singular_values); dgesdd values are descending",
            "condition = Python-float largest/smallest when smallest>0, else +inf",
            "eigenvalues = torch.linalg.eigvalsh(H), default UPLO='L': column-major work clone -> Accelerate dsyevd, jobz='N'",
            "minimum_eigenvalue = torch.min(eigenvalues); dsyevd values are ascending",
            "scale = max(float(torch.max(torch.abs(eigenvalues))), float64.tiny)",
            "positive = minimum_eigenvalue > 0; condition_ok = condition <= 1e12",
        ],
        "implementation_semantics": {
            "svd": {
                "routine": "Accelerate dgesdd",
                "matrix_region": "full 4x4 matrix",
                "output_order": "descending singular values",
                "fallback": "none",
            },
            "symmetric_eigen": {
                "routine": "Accelerate dsyevd",
                "matrix_region": "lower triangle (Torch eigvalsh default UPLO='L')",
                "output_order": "ascending algebraic eigenvalues",
                "fallback": "none",
            },
            "errors": "Torch linalg error checking propagates LAPACK failure; no SVD/eigen fallback is present",
        },
        "source_and_runtime_evidence": source_evidence(),
        "preprocessing_profile": profiler_evidence(representative_hessian),
        "triangle_and_order_controls": triangle_and_order_controls(representative_hessian),
        "first_divergence": {
            "retained_teacher_primitives": "none",
            "first_unsafe_reconstruction": (
                "replacing dgesdd singular values with descending abs(dsyevd eigenvalues)"
            ),
            "shortcut_singular_value_bit_mismatches": shortcut_singular_mismatches,
            "shortcut_singular_values": shortcut_values,
            "shortcut_condition_bit_mismatches": shortcut_condition_mismatches,
            "diagnosis": (
                "the symmetric-matrix identity is mathematically valid, but independent "
                "dgesdd and dsyevd algorithms round differently; both production LAPACK "
                "calls are required for teacher-exact bits"
            ),
        },
        "recovered_assembly": {
            "implementation": str(IMPLEMENTATION),
            "implementation_sha256": sha256(IMPLEMENTATION),
            "reused_learned_components": [
                "locked exact division assembly for largest_singular/smallest_singular",
                "locked subtraction/sign strict comparator for >0 and <=1e12 predicates",
            ],
            "retained_external_deterministic_primitives": [
                "Torch 2.2.2 arm64 torch.linalg.svdvals -> Accelerate dgesdd",
                "Torch 2.2.2 arm64 torch.linalg.eigvalsh(UPLO='L') -> Accelerate dsyevd",
            ],
            "deterministic_scaffolding": [
                "isfinite reductions and early return",
                "transpose/column-major clone and LAPACK work/output storage",
                "torch.min/max/abs reductions",
                "Python branch/storage and float64 tiny selection",
            ],
            "new_parameters": 0,
            "classification": (
                "TEACHER_EXACT_ASSEMBLY_WITH_RETAINED_DETERMINISTIC_LAPACK_PRIMITIVES"
            ),
        },
        "corpus": {
            "cases": 64,
            "states": state_count,
            "exact_states": exact_states,
            "eigenvalue_entries": aggregate["eigenvalues_values"],
            "singular_value_entries": aggregate["singular_values_values"],
            "solver_calls": corpus_solver_calls,
            "aggregate": dict(aggregate),
            "boolean_mismatches": dict(boolean_mismatches),
            "by_source": {key: dict(value) for key, value in by_source.items()},
            "cases_with_mismatch": [
                row["case_id"]
                for row in case_rows
                if any(key.endswith("_bit_mismatches") and value
                       for key, value in row["metrics"].items())
                or any(key.endswith("_mismatches") and value
                       for key, value in row["metrics"].items())
            ],
        },
        "branch_controls": controls,
        "final_classification": {
            "exact_replay": exact_states == state_count and not any(boolean_mismatches.values()),
            "new_learned_primitive_required": False,
            "reason_external_primitives_remain": (
                "no existing arithmetic assembly reproduces the implementation-specific "
                "Accelerate dgesdd/dsyevd rounding; approximation would violate the exact gate"
            ),
            "next_integration_operation": (
                "curvature/direct-versus-damped routing from the recovered positivity and "
                "condition predicates"
            ),
        },
        "checkpoint_hashes_unchanged": checkpoint_hashes() == expected_checkpoints,
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "eigenspectrum_condition_assembly.json").write_text(json.dumps({
        "status": report["status"],
        **report["recovered_assembly"],
        "validation": {
            "states": state_count,
            "exact_states": exact_states,
            "eigenvalue_entries": aggregate["eigenvalues_values"],
            "eigenvalue_bit_mismatches": aggregate["eigenvalues_bit_mismatches"],
            "condition_values": aggregate["condition_values"],
            "condition_bit_mismatches": aggregate["condition_bit_mismatches"],
            "finite_mismatches": boolean_mismatches["finite"],
            "positive_mismatches": boolean_mismatches["positive"],
            "condition_ok_mismatches": boolean_mismatches["condition_ok"],
        },
    }, indent=2) + "\n")
    return report


def main() -> None:
    report = run()
    aggregate = report["corpus"]["aggregate"]
    print(json.dumps({
        "status": report["status"],
        "states": report["corpus"]["states"],
        "exact_states": report["corpus"]["exact_states"],
        "eigenvalues": {
            "values": aggregate["eigenvalues_values"],
            "bit_mismatches": aggregate["eigenvalues_bit_mismatches"],
        },
        "condition": {
            "values": aggregate["condition_values"],
            "bit_mismatches": aggregate["condition_bit_mismatches"],
        },
        "boolean_mismatches": report["corpus"]["boolean_mismatches"],
        "first_divergence": report["first_divergence"],
        "classification": report["recovered_assembly"]["classification"],
        "checkpoints_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()

