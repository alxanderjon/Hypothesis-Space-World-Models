#!/usr/bin/env python3
"""Peel and validate only the modular-lattice Euclidean GCD assembly."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.geometry.initial_design_remainder_assembly import (
    InitialDesignGCDAssembly,
    InitialDesignRemainderAssembly,
    initial_design_with_remainder_assembly,
    multipliers_with_gcd_assembly,
)
from experiments.sho.run_full_k_transfer import initial_design as teacher_design


OUT = Path("outputs/hybrid_student_initial_design_gcd_binding")
BASELINE = Path("outputs/hybrid_student_baseline_frozen/freeze_manifest.json")
REMAINDER_REPORT = Path(
    "outputs/hybrid_student_initial_design_remainder_binding/report.json"
)
ASSEMBLY_SOURCE = Path(
    "experiments/geometry/initial_design_remainder_assembly.py"
)
TEACHER_SOURCES = (
    Path("experiments/sho/run_full_k_transfer.py"),
    Path("experiments/sho/dimension_general_edirect.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float_bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def production_scan(dimension: int) -> tuple[list[int], list[dict[str, int]]]:
    n = 5 * dimension
    schedule: list[int] = []
    calls: list[dict[str, int]] = []
    candidate = 1
    while len(schedule) < dimension:
        value = math.gcd(candidate, n)
        calls.append({"candidate": candidate, "n": n, "gcd": value})
        if value == 1:
            schedule.append(candidate)
        candidate += 1
    return schedule, calls


def first_failure_trace(
    operator: InitialDesignGCDAssembly, a: float, b: float
) -> list[dict[str, object]]:
    _, trace = operator(
        np.float64(a), np.float64(b), return_trace=True, record_states=True
    )
    assert trace.state_traces is not None
    return list(trace.state_traces[0])


def compare_pairs(
    operator: InitialDesignGCDAssembly,
    a,
    b,
    label: str,
) -> dict[str, object]:
    left, right = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    expected = np.fromiter(
        (math.gcd(int(x), int(y)) for x, y in zip(left.ravel(), right.ravel())),
        dtype=np.float64,
        count=left.size,
    ).reshape(left.shape)
    actual, trace = operator(left, right, return_trace=True)
    mismatch = np.flatnonzero(
        float_bits(expected).ravel() != float_bits(actual).ravel()
    )
    first = None
    if len(mismatch):
        index = int(mismatch[0])
        first = {
            "label": label,
            "index": index,
            "a": float(left.ravel()[index]),
            "b": float(right.ravel()[index]),
            "production": float(expected.ravel()[index]),
            "assembly": float(actual.ravel()[index]),
            "production_bits": int(float_bits(expected).ravel()[index]),
            "assembly_bits": int(float_bits(actual).ravel()[index]),
            "euclidean_state_trace": first_failure_trace(
                operator, left.ravel()[index], right.ravel()[index]
            ),
        }
    euclidean_histogram = Counter(map(int, trace.iterations.ravel()))
    nested_histogram = Counter(
        map(int, trace.nested_remainder_iterations.ravel())
    )
    return {
        "label": label,
        "pairs": int(expected.size),
        "exact_matches": int(expected.size - len(mismatch)),
        "mismatches": int(len(mismatch)),
        "first_mismatch": first,
        "euclidean_iteration_histogram": {
            str(key): value for key, value in sorted(euclidean_histogram.items())
        },
        "maximum_euclidean_iterations": int(trace.iterations.max(initial=0)),
        "nested_remainder_iteration_histogram": {
            str(key): value for key, value in sorted(nested_histogram.items())
        },
        "maximum_nested_remainder_iterations": int(
            trace.nested_remainder_iterations.max(initial=0)
        ),
        "total_nested_remainder_iterations": int(
            trace.nested_remainder_iterations.sum()
        ),
        "remainder_calls": int(trace.remainder_calls),
        "equality_calls": int(trace.equality_calls),
        "fallback_calls": int(trace.fallback_calls),
    }


def target_ast_audit() -> dict[str, object]:
    snippets = (
        inspect.getsource(InitialDesignGCDAssembly),
        inspect.getsource(multipliers_with_gcd_assembly),
    )
    tree = ast.parse("\n".join(snippets))
    # PEP 604 type unions use the same AST operator node as executable ``|``.
    # Remove annotations before checking whether the GCD algorithm performs
    # any runtime bitwise operation (as a binary/Stein implementation would).
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        elif isinstance(node, ast.AnnAssign):
            node.annotation = None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    ]
    forbidden_gcd = [
        name for name in calls
        if name in {"math.gcd", "np.gcd", "np.gcd.reduce", "numpy.gcd"}
    ]
    forbidden_remainder = [
        name for name in calls
        if name in {"np.remainder", "np.mod", "numpy.remainder"}
    ]
    forbidden_division = sum(
        isinstance(node, (ast.Div, ast.FloorDiv)) for node in ast.walk(tree)
    )
    forbidden_mod = sum(isinstance(node, ast.Mod) for node in ast.walk(tree))
    forbidden_bitwise = sum(
        isinstance(node, (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift))
        for node in ast.walk(tree)
    )
    return {
        "math_or_numpy_gcd_calls": forbidden_gcd,
        "production_remainder_calls": forbidden_remainder,
        "division_nodes": forbidden_division,
        "modulo_nodes": forbidden_mod,
        "bitwise_nodes": forbidden_bitwise,
        "passed": not (
            forbidden_gcd or forbidden_remainder or forbidden_division
            or forbidden_mod or forbidden_bitwise
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remainder_report = json.loads(REMAINDER_REPORT.read_text())
    if remainder_report["status"] != "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_LOOP":
        raise RuntimeError("locked remainder prerequisite is not exact")

    protected_paths = (BASELINE, REMAINDER_REPORT, *TEACHER_SOURCES)
    protected_before = {str(path): sha256(path) for path in protected_paths}
    remainder = InitialDesignRemainderAssembly()
    gcd = InitialDesignGCDAssembly(remainder)

    active_rows = []
    active_pair_rows = []
    expected_schedules = {2: [1, 3], 4: [1, 3, 7, 9], 5: [1, 2, 3, 4, 6]}
    for dimension in (2, 4, 5):
        schedule, calls = production_scan(dimension)
        left = np.asarray([row["candidate"] for row in calls], dtype=np.float64)
        right = np.full_like(left, np.float64(5 * dimension))
        replay = compare_pairs(gcd, left, right, f"active_scan_d{dimension}")
        actual_schedule, assembly_trace = multipliers_with_gcd_assembly(
            dimension, gcd, return_trace=True
        )
        pair_rows = []
        for oracle_row, assembly_row in zip(calls, assembly_trace["calls"]):
            pair_rows.append({
                **oracle_row,
                "assembly_gcd": assembly_row["gcd"],
                "euclidean_iterations": assembly_row["euclidean_iterations"],
                "nested_remainder_iterations": assembly_row[
                    "nested_remainder_iterations"
                ],
            })
        active_pair_rows.extend(
            [{"dimension": dimension, **row} for row in pair_rows]
        )
        active_rows.append({
            "dimension": dimension,
            "n": 5 * dimension,
            "candidate_range": [1, calls[-1]["candidate"]],
            "pairs": pair_rows,
            "production_schedule": schedule,
            "assembly_schedule": actual_schedule,
            "required_schedule": expected_schedules[dimension],
            "schedule_exact": (
                schedule == actual_schedule == expected_schedules[dimension]
            ),
            "replay": replay,
            "production_gcd_fallback_calls": assembly_trace[
                "production_gcd_fallback_calls"
            ],
            "production_remainder_fallback_calls": assembly_trace[
                "production_remainder_fallback_calls"
            ],
        })

    # Exhaustive positive controls stay within the caller's actual sign/domain.
    bound = 128
    grid_a, grid_b = np.meshgrid(
        np.arange(1, bound + 1, dtype=np.float64),
        np.arange(1, bound + 1, dtype=np.float64),
        indexing="ij",
    )
    small_controls = compare_pairs(
        gcd, grid_a.ravel(), grid_b.ravel(), "positive_integer_grid_1_to_128"
    )

    structural_left = []
    structural_right = []
    structural_dimensions = []
    for dimension in range(1, 65):
        schedule, calls = production_scan(dimension)
        structural_left.extend(row["candidate"] for row in calls)
        structural_right.extend(row["n"] for row in calls)
        structural_dimensions.append({
            "dimension": dimension,
            "n": 5 * dimension,
            "candidate_range": [1, calls[-1]["candidate"]],
            "pairs": len(calls),
            "last_multiplier": schedule[-1],
            "candidate_exceeds_n": calls[-1]["candidate"] > 5 * dimension,
        })
    structural_controls = compare_pairs(
        gcd,
        np.asarray(structural_left, dtype=np.float64),
        np.asarray(structural_right, dtype=np.float64),
        "structural_design_scans_d1_to_d64",
    )

    # d=6006 is the first structural scan whose accepted schedule crosses n.
    cross_dimension = 6006
    cross_schedule, cross_calls = production_scan(cross_dimension)
    cross_left = np.asarray(
        [row["candidate"] for row in cross_calls], dtype=np.float64
    )
    cross_right = np.full_like(cross_left, np.float64(5 * cross_dimension))
    cross_controls = compare_pairs(
        gcd, cross_left, cross_right, "cross_modulus_scan_d6006"
    )
    cross_summary = {
        "dimension": cross_dimension,
        "n": 5 * cross_dimension,
        "candidate_range": [1, cross_calls[-1]["candidate"]],
        "pairs": len(cross_calls),
        "accepted_multipliers": len(cross_schedule),
        "accepted_above_n": sum(value > 5 * cross_dimension for value in cross_schedule),
        "last_multiplier": cross_schedule[-1],
        "replay": cross_controls,
    }

    design_rows = []
    for dimension in (2, 4, 5):
        expected = teacher_design(dimension)
        actual, trace = initial_design_with_remainder_assembly(
            dimension, remainder, gcd, return_trace=True
        )
        mismatch = np.flatnonzero(
            float_bits(expected).ravel() != float_bits(actual).ravel()
        )
        design_rows.append({
            "dimension": dimension,
            "site_count_teacher": int(expected.shape[0]),
            "site_count_assembly": int(actual.shape[0]),
            "coordinate_values": int(expected.size),
            "coordinate_bit_mismatches": int(len(mismatch)),
            "row_column_order_exact": bool(
                np.array_equal(float_bits(expected), float_bits(actual))
            ),
            "multipliers": trace["multipliers"],
            "multiplier_order_exact": (
                trace["multipliers"] == expected_schedules[dimension]
            ),
            "production_gcd_fallback_calls": trace[
                "production_gcd_fallback_calls"
            ],
            "production_remainder_fallback_calls": trace[
                "production_remainder_fallback_calls"
            ],
        })

    ast_audit = target_ast_audit()
    validations = [
        *(row["replay"] for row in active_rows),
        small_controls,
        structural_controls,
        cross_controls,
    ]
    first_mismatch = next(
        (row["first_mismatch"] for row in validations if row["mismatches"]),
        None,
    )
    protected_after = {str(path): sha256(path) for path in protected_paths}
    exact = (
        first_mismatch is None
        and all(row["schedule_exact"] for row in active_rows)
        and all(row["production_gcd_fallback_calls"] == 0 for row in active_rows)
        and all(
            row["production_remainder_fallback_calls"] == 0
            for row in active_rows
        )
        and all(row["coordinate_bit_mismatches"] == 0 for row in design_rows)
        and all(row["production_gcd_fallback_calls"] == 0 for row in design_rows)
        and all(
            row["production_remainder_fallback_calls"] == 0
            for row in design_rows
        )
        and ast_audit["passed"]
        and protected_before == protected_after
    )
    totals = {
        "tested_pairs": sum(int(row["pairs"]) for row in validations),
        "exact_matches": sum(int(row["exact_matches"]) for row in validations),
        "mismatches": sum(int(row["mismatches"]) for row in validations),
        "maximum_euclidean_iterations": max(
            int(row["maximum_euclidean_iterations"]) for row in validations
        ),
        "maximum_nested_remainder_iterations": max(
            int(row["maximum_nested_remainder_iterations"])
            for row in validations
        ),
        "total_nested_remainder_iterations": sum(
            int(row["total_nested_remainder_iterations"])
            for row in validations
        ),
        "fallback_calls": sum(int(row["fallback_calls"]) for row in validations),
    }
    report = {
        "status": (
            "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_LOOPS"
            if exact else "FIRST_MISMATCH"
        ),
        "target": "initial-design arbitrary-positive-integer GCD",
        "contract": {
            "input_a": "positive integer-valued exactly representable float64 candidate",
            "input_b": "positive integer-valued exactly representable float64 n=5d",
            "intermediates": (
                "nonnegative exact integer residues; every nonzero divisor satisfies "
                "the locked remainder contract"
            ),
            "excluded": [
                "negative inputs", "fractional inputs", "NaN", "infinity",
                "negative zero",
            ],
        },
        "implementation": {
            "state_machine": (
                "while locked_equal(b,0) is false: "
                "r=locked_exact_remainder(a,b); a,b=b,r; return a"
            ),
            "reused_components": [
                "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_LOOP remainder",
                "locked equality/comparison",
            ],
            "deterministic_scaffolding": [
                "Euclidean loop", "state assignment", "active masks", "counters",
            ],
            "new_parameters": 0,
            "ast_audit": ast_audit,
        },
        "active_contract_audit": active_rows,
        "active_pair_table": active_pair_rows,
        "small_positive_integer_controls": small_controls,
        "structural_controls": {
            "dimensions": [1, 64],
            "dimension_summaries": structural_dimensions,
            "replay": structural_controls,
            "cross_modulus": cross_summary,
        },
        "totals": totals,
        "initial_design_replay": design_rows,
        "first_mismatch": first_mismatch,
        "initial_design_decomposition_after_peel": {
            "A_B_exact": [
                "n=5d arithmetic", "candidate increment", "equality/comparison",
                "coprime predicate through assembled Euclidean GCD",
                "multiplication", "arbitrary-modulus remainder", "+0.5",
                "division by n",
            ],
            "C_deterministic_scaffolding": [
                "candidate scan loop", "Euclidean/remainder loops",
                "row/column loops", "counters/list storage",
                "arange/column_stack storage",
            ],
            "D_unresolved_arithmetic": [],
            "former_initial_design_D_oracle_eliminated": exact,
            "initial_design_status": (
                "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE"
                if exact else "D_ORACLE_REMAINS"
            ),
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "canonical_822_rerun": False,
        "training_performed": False,
        "other_D_oracles_modified": False,
        "frozen_baseline_manifest_modified": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "initial_design_gcd_binding": report["status"],
        "initial_design_status": report[
            "initial_design_decomposition_after_peel"
        ]["initial_design_status"],
        **totals,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "totals": totals,
        "active_schedules": {
            str(row["dimension"]): row["assembly_schedule"] for row in active_rows
        },
        "cross_modulus": cross_summary,
        "initial_design_replay": design_rows,
        "initial_design_status": report[
            "initial_design_decomposition_after_peel"
        ]["initial_design_status"],
        "first_mismatch": first_mismatch,
    }, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
