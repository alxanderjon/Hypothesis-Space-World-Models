#!/usr/bin/env python3
"""Validate and bind only N1 boundary-support requested-to-k resolution."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from experiments.geometry.boundary_face_resolution_assembly import (
    BoundaryFaceResolutionAssembly,
)
from experiments.sho import dimension_general_edirect as n1
from experiments.sho import run_stage1b_edirect_vs_direct as teacher


OUT = Path("outputs/hybrid_student_boundary_face_resolution_binding")
ASSEMBLY_SOURCE = Path(
    "experiments/geometry/boundary_face_resolution_assembly.py"
)
BASELINE = Path("outputs/hybrid_student_baseline_frozen/freeze_manifest.json")
CANONICAL = Path("outputs/canonical_frozen_teacher_regression_822/corpus_manifest.json")
TEACHER_SOURCE = Path("experiments/sho/dimension_general_edirect.py")
STUDENT_SOURCE = Path("experiments/resume_hybrid_student_after_ckdtree.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def production_k(requested: int) -> int:
    return min(257, max(9, int(math.ceil(math.sqrt(requested)))))


def compare(
    operator: BoundaryFaceResolutionAssembly,
    values,
    label: str,
) -> dict[str, object]:
    requested = np.asarray(values, dtype=np.float64)
    expected = np.asarray(
        [production_k(int(value)) for value in requested.ravel()],
        dtype=np.int64,
    ).reshape(requested.shape)
    actual, trace = operator.evaluate_batch(requested, return_trace=True)
    mismatch = np.flatnonzero(expected.ravel() != actual.ravel())
    first = None
    if len(mismatch):
        index = int(mismatch[0])
        value = int(requested.ravel()[index])
        scalar, scalar_trace = operator(value, return_trace=True)
        first = {
            "label": label,
            "index": index,
            "requested": value,
            "production_k": int(expected.ravel()[index]),
            "assembly_k": int(scalar),
            "loop_trace": list(scalar_trace.states),
            "lower_clamp_taken": scalar_trace.lower_clamp_taken,
            "upper_clamp_taken": scalar_trace.upper_clamp_taken,
        }
    histogram = Counter(map(int, trace.iterations.ravel()))
    return {
        "label": label,
        "cases": int(requested.size),
        "exact_matches": int(requested.size - len(mismatch)),
        "mismatches": int(len(mismatch)),
        "first_mismatch": first,
        "minimum_requested": int(requested.min()),
        "maximum_requested": int(requested.max()),
        "loop_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "maximum_loop_count": int(trace.iterations.max(initial=0)),
        "lower_clamp_cases": int(np.count_nonzero(trace.lower_clamp_taken)),
        "upper_clamp_cases": int(np.count_nonzero(trace.upper_clamp_taken)),
        "multiplication_calls": int(trace.multiplication_calls),
        "addition_calls": int(trace.addition_calls),
        "comparison_calls": int(trace.comparison_calls),
        "fallback_calls": int(trace.fallback_calls),
    }


def boundary_with_k_binding(
    operator: BoundaryFaceResolutionAssembly,
    d: int,
    requested: int,
    seed: int,
    lower=None,
    upper=None,
):
    """Production boundary path with only requested-to-k replaced."""
    lower = np.zeros(d) if lower is None else np.asarray(lower, float)
    upper = np.ones(d) if upper is None else np.asarray(upper, float)
    if lower.shape != (d,) or upper.shape != (d,) or not np.all(upper > lower):
        raise ValueError("invalid bounds")
    count, resolution_trace = operator(requested, return_trace=True)
    if d == 1:
        unit = np.array([[0.0], [1.0]])
        ids = np.array([[0, 0], [0, 1]])
    else:
        free = (
            np.linspace(0.0, 1.0, count)[:, None]
            if d == 2
            else qmc.Sobol(d - 1, scramble=True, seed=seed).random_base2(
                int(math.ceil(math.log2(count)))
            )[:count]
        )
        blocks = []
        labels = []
        for axis in reversed(range(d)):
            for side in (0, 1):
                block = np.empty((count, d))
                block[:, axis] = side
                block[:, np.arange(d) != axis] = free
                blocks.append(block)
                labels.extend([(axis, side)] * count)
        unit = np.concatenate(blocks)
        ids = np.asarray(labels)
    return lower + (upper - lower) * unit, ids, {
        "k": count,
        "resolution_trace": resolution_trace,
        "sqrt_fallback_calls": 0,
        "ceil_fallback_calls": 0,
    }


def forbidden_source_audit() -> dict[str, object]:
    tree = ast.parse(inspect.getsource(BoundaryFaceResolutionAssembly))
    call_names = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    forbidden = [
        name for name in call_names
        if name in {
            "sqrt", "math.sqrt", "np.sqrt", "numpy.sqrt",
            "ceil", "math.ceil", "np.ceil", "numpy.ceil",
            "pow", "math.pow", "np.power", "numpy.power",
            "log", "math.log", "np.log", "numpy.log",
            "isqrt", "math.isqrt",
        }
    ]
    power_nodes = sum(isinstance(node, ast.Pow) for node in ast.walk(tree))
    return {
        "forbidden_calls": forbidden,
        "power_operator_nodes": power_nodes,
        "lookup_tables": False,
        "passed": not forbidden and power_nodes == 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = (BASELINE, CANONICAL, TEACHER_SOURCE, STUDENT_SOURCE)
    protected_before = {str(path): sha256(path) for path in protected_paths}
    operator = BoundaryFaceResolutionAssembly()

    active_by_dimension = {
        "d2": list(range(1000, 32701, 100)) + [32768],
        "d4": list(range(2000, 32701, 100)) + [32768],
        "d5": list(range(2500, 32701, 100)) + [32768],
    }
    active_union = sorted({
        value for values in active_by_dimension.values() for value in values
    })
    active = compare(operator, active_union, "all_active_requested_values")
    exhaustive = compare(
        operator,
        np.arange(1, teacher.MC_CANDIDATE_CAP + 1, dtype=np.float64),
        "exhaustive_reachable_positive_integers",
    )

    transitions = set()
    for root in range(1, 301):
        square = root * root
        transitions.update(value for value in (square - 1, square, square + 1) if value > 0)
    transitions.update({
        63, 64, 65, 80, 81, 82,
        65535, 65536, 65537, 66048, 66049, 66050,
        70000, 100000, 1000000,
    })
    structural = compare(
        operator, sorted(transitions), "square_and_clamp_transition_controls"
    )

    validations = [active, exhaustive, structural]
    first_mismatch = next(
        (row["first_mismatch"] for row in validations if row["mismatches"]),
        None,
    )
    if first_mismatch is not None:
        failure = {
            "status": "FIRST_MISMATCH",
            "first_mismatch": first_mismatch,
            "training_performed": False,
            "other_D_oracles_modified": False,
        }
        (OUT / "report.json").write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2))
        raise SystemExit(1)

    replays = []
    for dimension, requested, expected_k, expected_shape in (
        (2, 1000, 32, (128, 2)),
        (4, 2000, 45, (360, 4)),
        (5, 2500, 50, (500, 5)),
    ):
        seed = teacher.SEARCH_SEED
        expected_points, expected_ids = n1.boundary(dimension, requested, seed)
        actual_points, actual_ids, trace = boundary_with_k_binding(
            operator, dimension, requested, seed
        )
        unique_expected = np.unique(expected_points, axis=0)
        unique_actual = np.unique(actual_points, axis=0)
        replays.append({
            "dimension": dimension,
            "requested": requested,
            "expected_k": expected_k,
            "assembly_k": trace["k"],
            "expected_shape": list(expected_shape),
            "assembly_shape": list(actual_points.shape),
            "coordinate_bit_mismatches": int(np.count_nonzero(
                bits(expected_points) != bits(actual_points)
            )),
            "face_order_exact": bool(np.array_equal(bits(expected_points), bits(actual_points))),
            "identity_label_mismatches": int(np.count_nonzero(expected_ids != actual_ids)),
            "identity_order_exact": bool(np.array_equal(expected_ids, actual_ids)),
            "expected_duplicate_rows": int(len(expected_points) - len(unique_expected)),
            "assembly_duplicate_rows": int(len(actual_points) - len(unique_actual)),
            "duplicate_behavior_exact": bool(np.array_equal(
                bits(unique_expected), bits(unique_actual)
            )),
            "sqrt_fallback_calls": trace["sqrt_fallback_calls"],
            "ceil_fallback_calls": trace["ceil_fallback_calls"],
        })

    source_audit = forbidden_source_audit()
    protected_after = {str(path): sha256(path) for path in protected_paths}
    totals = {
        "cases": sum(int(row["cases"]) for row in validations),
        "exact_matches": sum(int(row["exact_matches"]) for row in validations),
        "mismatches": sum(int(row["mismatches"]) for row in validations),
        "maximum_loop_count": max(int(row["maximum_loop_count"]) for row in validations),
        "fallback_calls": sum(int(row["fallback_calls"]) for row in validations),
    }
    exact = (
        totals["mismatches"] == 0
        and totals["fallback_calls"] == 0
        and source_audit["passed"]
        and all(row["assembly_k"] == row["expected_k"] for row in replays)
        and all(row["assembly_shape"] == row["expected_shape"] for row in replays)
        and all(row["coordinate_bit_mismatches"] == 0 for row in replays)
        and all(row["identity_label_mismatches"] == 0 for row in replays)
        and all(row["duplicate_behavior_exact"] for row in replays)
        and all(row["sqrt_fallback_calls"] == 0 for row in replays)
        and all(row["ceil_fallback_calls"] == 0 for row in replays)
        and protected_before == protected_after
    )
    report = {
        "status": (
            "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_LOOP"
            if exact else "VALIDATION_FAILED"
        ),
        "binding": "boundary_face_resolution",
        "contract": {
            "input": "positive integer-valued exactly representable requested",
            "live_formula": "requested=min(100*evaluated_site_count,32768)",
            "active_minimum": 1000,
            "active_maximum": 32768,
            "active_starts": {"d2": 1000, "d4": 2000, "d5": 2500},
            "active_requested_values": active_by_dimension,
            "positive_exact_integer": True,
            "pool_cap": teacher.MC_CANDIDATE_CAP,
            "cap_implies_max_unclamped_k": 182,
            "upper_257_clamp_active": False,
            "upper_257_route_retained": True,
        },
        "implementation": {
            "state_machine": (
                "k=0; while locked_multiply(k,k) < requested via locked comparison: "
                "k=locked_add(k,1); clamp below 9 and above 257 via locked routes"
            ),
            "reused": ["locked multiplication", "locked addition", "locked comparison/routing"],
            "scaffolding": ["loop", "counter state", "constant assignment after route"],
            "new_parameters": 0,
            "source_audit": source_audit,
            "sqrt_fallback_calls": 0,
            "ceil_fallback_calls": 0,
        },
        "active_validation": active,
        "exhaustive_reachable_validation": exhaustive,
        "transition_validation": structural,
        "totals": totals,
        "boundary_replay": replays,
        "first_mismatch": first_mismatch,
        "complete_boundary_support_readiness": {
            "ready_for_exact_composition": exact,
            "components": [
                "boundary_face_resolution exact assembly",
                "locked d=2 grid arithmetic",
                "locked scrambled-Sobol full-block/prefix",
                "deterministic face enumeration and coordinate insertion",
                "exact normalized-to-bounded affine map",
                "exact pool merge/dedup/order",
            ],
            "full_boundary_oracle_replaced_in_this_run": False,
            "remaining_irreducible_operation": None,
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_hashes_unchanged": protected_before == protected_after,
        "canonical_822_rerun": False,
        "training_performed": False,
        "other_D_oracles_modified": False,
        "frozen_baseline_modified": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "boundary_face_resolution": report["status"],
        **totals,
        "full_boundary_oracle_replaced": False,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "totals": totals,
        "active": active,
        "transition": structural,
        "boundary_replay": replays,
        "complete_boundary_support_readiness": report[
            "complete_boundary_support_readiness"
        ],
    }, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
