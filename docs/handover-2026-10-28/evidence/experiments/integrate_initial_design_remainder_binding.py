#!/usr/bin/env python3
"""Validate only the initial-design arbitrary-modulus remainder assembly."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.geometry.initial_design_remainder_assembly import (
    InitialDesignRemainderAssembly,
    initial_design_with_remainder_assembly,
)
from experiments.sho.run_full_k_transfer import initial_design as teacher_design


OUT = Path("outputs/hybrid_student_initial_design_remainder_binding")
ASSEMBLY_SOURCE = Path(
    "experiments/geometry/initial_design_remainder_assembly.py"
)
BASELINE = Path("outputs/hybrid_student_baseline_frozen/freeze_manifest.json")
TEACHER_SOURCES = (
    Path("experiments/sho/run_full_k_transfer.py"),
    Path("experiments/sho/dimension_general_edirect.py"),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def multipliers(dimension: int) -> list[int]:
    count = 5 * dimension
    result = []
    candidate = 1
    while len(result) < dimension:
        if math.gcd(candidate, count) == 1:
            result.append(candidate)
        candidate += 1
    return result


def lattice_operands(dimension: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    count = 5 * dimension
    base = np.arange(count, dtype=np.float64)
    schedule = multipliers(dimension)
    x = np.concatenate([
        np.multiply(value, base, dtype=np.float64) for value in schedule
    ])
    n = np.full_like(x, np.float64(count))
    return x, n, schedule


def compare(operator, x, n, label: str) -> dict:
    expected = np.remainder(
        np.asarray(x, dtype=np.float64), np.asarray(n, dtype=np.float64)
    )
    actual, trace = operator(x, n, return_trace=True)
    mismatch = np.flatnonzero(bits(expected).ravel() != bits(actual).ravel())
    first = None
    if len(mismatch):
        index = int(mismatch[0])
        first = {
            "label": label,
            "index": index,
            "x": float(np.asarray(x, np.float64).ravel()[index]),
            "n": float(np.asarray(n, np.float64).ravel()[index]),
            "production": float(expected.ravel()[index]),
            "assembly": float(actual.ravel()[index]),
            "production_bits": int(bits(expected).ravel()[index]),
            "assembly_bits": int(bits(actual).ravel()[index]),
            "iterations": int(trace.iterations.ravel()[index]),
        }
    histogram = Counter(map(int, trace.iterations.ravel()))
    return {
        "label": label,
        "calls": int(expected.size),
        "exact_matches": int(expected.size - len(mismatch)),
        "mismatches": int(len(mismatch)),
        "first_mismatch": first,
        "iteration_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "maximum_iterations": int(trace.iterations.max(initial=0)),
        "comparison_calls": int(trace.comparison_calls),
        "subtraction_calls": int(trace.subtraction_calls),
        "fallback_calls": int(trace.fallback_calls),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected_before = {
        str(BASELINE): sha256(BASELINE),
        **{str(path): sha256(path) for path in TEACHER_SOURCES},
    }
    operator = InitialDesignRemainderAssembly()

    active = []
    active_domain = []
    for dimension in (2, 4, 5):
        x, n, schedule = lattice_operands(dimension)
        row = compare(operator, x, n, f"active_d{dimension}")
        row.update({
            "dimension": dimension,
            "n": 5 * dimension,
            "multipliers": schedule,
            "minimum_x": float(x.min(initial=0.0)),
            "maximum_x": float(x.max(initial=0.0)),
            "integer_valued_x": bool(np.all(x == np.asarray(x, np.int64))),
            "integer_valued_n": bool(np.all(n == np.asarray(n, np.int64))),
        })
        active.append(row)
        active_domain.append({
            "dimension": dimension, "n": 5 * dimension,
            "multipliers": schedule, "calls": len(x),
            "x_range": [float(x.min(initial=0.0)), float(x.max(initial=0.0))],
        })

    control_x = []
    control_n = []
    for modulus in range(1, 65):
        # Exhaustive coverage includes x<n, x=n, every k*n, and k*n +/- 1.
        values = np.arange(0, 16 * modulus + 2, dtype=np.float64)
        control_x.append(values)
        control_n.append(np.full_like(values, np.float64(modulus)))
    controls = compare(
        operator, np.concatenate(control_x), np.concatenate(control_n),
        "exhaustive_small_integer_controls",
    )

    structural_rows = []
    for dimension in range(1, 33):
        x, n, schedule = lattice_operands(dimension)
        row = compare(operator, x, n, f"structural_d{dimension}")
        row.update({
            "dimension": dimension, "n": 5 * dimension,
            "multipliers": schedule,
            "maximum_x": float(x.max(initial=0.0)),
        })
        structural_rows.append(row)

    design_rows = []
    for dimension in (2, 4, 5):
        expected = teacher_design(dimension)
        actual, trace = initial_design_with_remainder_assembly(
            dimension, operator, return_trace=True
        )
        mismatch = np.flatnonzero(bits(expected).ravel() != bits(actual).ravel())
        design_rows.append({
            "dimension": dimension,
            "values": int(expected.size),
            "coordinate_bit_mismatches": int(len(mismatch)),
            "row_order_exact": bool(np.array_equal(bits(expected), bits(actual))),
            "column_order_exact": trace["multipliers"] == multipliers(dimension),
            "multipliers": trace["multipliers"],
            "production_remainder_fallback_calls": trace[
                "production_remainder_fallback_calls"
            ],
        })

    tree = ast.parse(ASSEMBLY_SOURCE.read_text())
    forbidden_mod_nodes = sum(isinstance(node, ast.Mod) for node in ast.walk(tree))
    forbidden_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in {"np.remainder", "np.mod", "np.floor", "np.trunc"}:
                forbidden_calls.append(name)

    all_replays = active + [controls] + structural_rows
    first_mismatch = next(
        (row["first_mismatch"] for row in all_replays if row["mismatches"]),
        None,
    )
    protected_after = {
        str(BASELINE): sha256(BASELINE),
        **{str(path): sha256(path) for path in TEACHER_SOURCES},
    }
    exact = (
        first_mismatch is None
        and not forbidden_mod_nodes
        and not forbidden_calls
        and all(row["coordinate_bit_mismatches"] == 0 for row in design_rows)
        and all(row["production_remainder_fallback_calls"] == 0 for row in design_rows)
        and protected_before == protected_after
    )
    totals = {
        "calls": sum(row["calls"] for row in all_replays),
        "exact_matches": sum(row["exact_matches"] for row in all_replays),
        "mismatches": sum(row["mismatches"] for row in all_replays),
        "maximum_iterations": max(row["maximum_iterations"] for row in all_replays),
        "fallback_calls": sum(row["fallback_calls"] for row in all_replays),
    }
    report = {
        "status": (
            "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_LOOP"
            if exact else "FIRST_MISMATCH"
        ),
        "target": "nonnegative integer-valued float64 arbitrary-modulus remainder",
        "contract": {
            "x": "nonnegative integer-valued, exactly representable float64",
            "n": "positive integer-valued, exactly representable float64",
            "excluded": ["negative", "fractional", "NaN", "infinity", "negative zero"],
        },
        "active_domain": active_domain,
        "exact_integer_subtraction_reason": (
            "Every active and structural state is a nonnegative exactly representable "
            "integer; r-n remains an exactly representable integer, so the locked "
            "float64 subtraction rounds exactly."
        ),
        "implementation": {
            "state_machine": "r=x; while not locked_compare(r<n): r=locked_subtract(r,n)",
            "comparison": "StrictLossSelector locked subtraction + frozen zero-sign threshold",
            "subtraction": "locked Subtraction checkpoint with weights [1,-1]",
            "deterministic_scaffolding": ["loop", "active mask", "state storage", "counters"],
            "new_parameters": 0,
            "forbidden_mod_ast_nodes": forbidden_mod_nodes,
            "forbidden_calls": forbidden_calls,
        },
        "active_replay": active,
        "small_integer_controls": controls,
        "structural_dimensions": {
            "dimensions": [1, 32],
            "rows": structural_rows,
        },
        "totals": totals,
        "initial_design_replay": design_rows,
        "first_mismatch": first_mismatch,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "gcd_next_target": {
            "implemented": False,
            "reducible": exact,
            "assembly": "Euclidean repeated locked remainder until exact zero",
            "other_requirements": [
                "existing equality/comparison/select", "deterministic loop/state"
            ],
        },
        "canonical_822_rerun": False,
        "training_performed": False,
        "other_D_oracles_modified": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "initial_design_remainder_binding": report["status"],
        "calls": totals["calls"],
        "mismatches": totals["mismatches"],
        "fallback_calls": totals["fallback_calls"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
