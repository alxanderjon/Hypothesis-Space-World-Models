#!/usr/bin/env python3
"""Compose and validate only the post-baseline N1 boundary-support path."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.geometry.n1_boundary_support_assembly import (
    N1BoundarySupportAssembly,
    scheduled_pool_with_boundary_assembly,
)
from experiments.sho import dimension_general_edirect as teacher_n1
from experiments.sho.validate_n1_boundary_merge_dedup import independent_unique


OUT = Path("outputs/hybrid_student_boundary_support_composition")
BASELINE = Path("outputs/hybrid_student_baseline_frozen/freeze_manifest.json")
CANONICAL = Path("outputs/canonical_frozen_teacher_regression_822/corpus_manifest.json")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
ASSEMBLY_SOURCE = Path("experiments/geometry/n1_boundary_support_assembly.py")
RESOLUTION_SOURCE = Path("experiments/geometry/boundary_face_resolution_assembly.py")
HYBRID_SOURCE = Path("experiments/resume_hybrid_student_after_ckdtree.py")
MERGE_SOURCE = Path("experiments/sho/validate_n1_boundary_merge_dedup.py")
TEACHER_SOURCE = Path("experiments/sho/dimension_general_edirect.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bits(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).view(np.uint64)


def mismatch_bits(left, right) -> tuple[int, dict[str, object] | None]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size), {
            "teacher_shape": list(a.shape), "student_shape": list(b.shape)
        }
    locations = np.argwhere(bits(a) != bits(b))
    if not len(locations):
        return 0, None
    index = tuple(map(int, locations[0]))
    return len(locations), {
        "index": index,
        "teacher": float(a[index]),
        "student": float(b[index]),
        "teacher_bits": int(bits(a)[index]),
        "student_bits": int(bits(b)[index]),
    }


def direct_case(
    operator: N1BoundarySupportAssembly,
    name: str,
    dimension: int,
    requested: int,
    seed: int,
    lower=None,
    upper=None,
) -> dict[str, object]:
    log = hybrid.Log()
    teacher_points, teacher_ids = teacher_n1.boundary(
        dimension, requested, seed, lower, upper
    )
    student_points, student_ids, state = operator(
        dimension, requested, seed, lower, upper, log=log
    )
    coordinate_errors, first_coordinate = mismatch_bits(
        teacher_points, student_points
    )
    identity_locations = np.argwhere(teacher_ids != student_ids)
    first_identity = None
    if len(identity_locations):
        index = tuple(map(int, identity_locations[0]))
        first_identity = {
            "index": index,
            "teacher": int(teacher_ids[index]),
            "student": int(student_ids[index]),
        }
    teacher_unique = np.unique(teacher_points, axis=0)
    student_unique = np.unique(student_points, axis=0)
    unique_errors, first_unique = mismatch_bits(teacher_unique, student_unique)

    k = int(state["k"])
    block_errors = []
    if dimension > 1:
        for block_index in range(2 * dimension):
            start = block_index * k
            stop = start + k
            axis = dimension - 1 - block_index // 2
            side = block_index % 2
            expected_value = (
                float(side) if lower is None
                else float(np.asarray(lower if side == 0 else upper)[axis])
            )
            block = student_points[start:stop]
            ids = student_ids[start:stop]
            if not np.all(block[:, axis] == expected_value):
                block_errors.append({"block": block_index, "kind": "face_value"})
            if not np.all(ids == np.asarray([axis, side])):
                block_errors.append({"block": block_index, "kind": "identity"})

    mismatch = bool(
        coordinate_errors or len(identity_locations) or unique_errors
        or block_errors or teacher_points.dtype != student_points.dtype
        or teacher_ids.dtype != student_ids.dtype
    )
    return {
        "name": name,
        "dimension": dimension,
        "requested": requested,
        "seed": seed,
        "bounds": None if lower is None else {
            "lower": np.asarray(lower, np.float64).tolist(),
            "upper": np.asarray(upper, np.float64).tolist(),
        },
        "k": k,
        "teacher_shape": list(teacher_points.shape),
        "student_shape": list(student_points.shape),
        "teacher_dtype": str(teacher_points.dtype),
        "student_dtype": str(student_points.dtype),
        "identity_dtype": str(student_ids.dtype),
        "coordinate_bit_mismatches": coordinate_errors,
        "first_coordinate_mismatch": first_coordinate,
        "identity_mismatches": int(len(identity_locations)),
        "first_identity_mismatch": first_identity,
        "face_boundary_or_order_errors": block_errors,
        "teacher_duplicate_rows": int(len(teacher_points) - len(teacher_unique)),
        "student_duplicate_rows": int(len(student_points) - len(student_unique)),
        "unique_coordinate_bit_mismatches": unique_errors,
        "first_unique_mismatch": first_unique,
        "route": state["route"],
        "binding_trace": {
            "face_resolution": state["face_resolution_binding_invocations"],
            "grid": state["grid_binding_invocations"],
            "sobol": state["sobol_binding_invocations"],
            "affine": state["affine_binding_invocations"],
            "old_boundary_oracle_calls": state["old_boundary_oracle_calls"],
            "forbidden_fallback_calls": state["forbidden_fallback_calls"],
            "log_counts": dict(log.counts),
            "log_implementations": {
                key: list(value) for key, value in log.implementation.items()
            },
        },
        "mismatch": mismatch,
    }


def pool_case(
    boundary: N1BoundarySupportAssembly,
    dimension: int,
    sample_count: int,
    case_id: int,
    iteration: int,
) -> dict[str, object]:
    log = hybrid.Log()
    student_pool, state = scheduled_pool_with_boundary_assembly(
        dimension, sample_count, case_id, iteration, log, boundary
    )
    teacher_pool = teacher_n1.pool(
        dimension, sample_count, case_id, iteration
    )
    pool_errors, first_pool = mismatch_bits(teacher_pool, student_pool)

    requested = state["requested"]
    seed = state["seed"]
    teacher_interior = qmc.Sobol(
        dimension, scramble=True, seed=seed
    ).random_base2(state["exponent"])[:requested]
    teacher_faces, teacher_ids = teacher_n1.boundary(
        dimension, requested, seed
    )
    teacher_raw = np.concatenate((teacher_interior, teacher_faces), axis=0)
    teacher_source_ids = np.concatenate((
        np.full((len(teacher_interior), 2), -1, dtype=np.int64), teacher_ids
    ))
    _, teacher_first = np.unique(teacher_raw, axis=0, return_index=True)
    raw_errors, first_raw = mismatch_bits(teacher_raw, state["raw_pool"])
    index_errors = int(np.count_nonzero(
        teacher_first != state["merge_retained_indices"]
    ))
    source_errors = int(np.count_nonzero(
        teacher_source_ids[teacher_first] != state["merged_source_identities"]
    ))
    boundary_unique = np.unique(teacher_faces, axis=0)
    return {
        "dimension": dimension,
        "sample_count": sample_count,
        "case_id": case_id,
        "iteration": iteration,
        "requested": requested,
        "seed": seed,
        "raw_rows": len(teacher_raw),
        "merged_rows": len(teacher_pool),
        "boundary_raw_rows": len(teacher_faces),
        "boundary_duplicate_rows": int(len(teacher_faces) - len(boundary_unique)),
        "raw_input_bit_mismatches": raw_errors,
        "first_raw_mismatch": first_raw,
        "merged_pool_bit_mismatches": pool_errors,
        "first_pool_mismatch": first_pool,
        "retained_index_mismatches": index_errors,
        "retained_identity_mismatches": source_errors,
        "face_resolution_binding_invocations": state["boundary_state"][
            "face_resolution_binding_invocations"
        ],
        "grid_binding_invocations": state["boundary_state"][
            "grid_binding_invocations"
        ],
        "sobol_binding_invocations": state["boundary_state"][
            "sobol_binding_invocations"
        ],
        "affine_binding_invocations": state["boundary_state"][
            "affine_binding_invocations"
        ],
        "pool_merge_binding_invocations": state[
            "pool_merge_binding_invocations"
        ],
        "old_boundary_oracle_calls": state["old_boundary_oracle_calls"],
        "forbidden_fallback_calls": state["forbidden_fallback_calls"],
        "log_counts": dict(log.counts),
        "mismatch": bool(
            raw_errors or pool_errors or index_errors or source_errors
        ),
    }


def binding_audit() -> dict[str, object]:
    old_source = inspect.getsource(hybrid.scheduled_pool)
    new_pool_source = inspect.getsource(scheduled_pool_with_boundary_assembly)
    new_boundary_source = inspect.getsource(N1BoundarySupportAssembly)
    dependencies = {
        "face_resolution": {
            "implementation": "BoundaryFaceResolutionAssembly",
            "source": str(RESOLUTION_SOURCE),
            "sha256": sha256(RESOLUTION_SOURCE),
        },
        "d2_grid_arithmetic": {
            "implementation": "LearnedArithmetic.divide/multiply",
            "source": "experiments/sho/assemble_n3_gradient_operator.py",
            "sha256": sha256(Path("experiments/sho/assemble_n3_gradient_operator.py")),
        },
        "d_gt_2_sobol": {
            "implementation": "scrambled_sobol_full_block + locked schedule/prefix",
            "source": str(HYBRID_SOURCE),
            "sha256": sha256(HYBRID_SOURCE),
        },
        "affine": {
            "implementation": "normalized_to_physical_assembly",
            "source": str(HYBRID_SOURCE),
            "sha256": sha256(HYBRID_SOURCE),
        },
        "pool_merge": {
            "implementation": "independent_unique",
            "source": str(MERGE_SOURCE),
            "sha256": sha256(MERGE_SOURCE),
        },
    }
    return {
        "old_oracle_call_site": (
            "experiments/resume_hybrid_student_after_ckdtree.py:scheduled_pool"
        ),
        "old_source_contains_boundary_oracle_call": "n1.boundary" in old_source,
        "post_baseline_entry_point": (
            "experiments.geometry.n1_boundary_support_assembly."
            "scheduled_pool_with_boundary_assembly"
        ),
        "replacement_function": (
            "experiments.geometry.n1_boundary_support_assembly."
            "N1BoundarySupportAssembly"
        ),
        "post_baseline_source_contains_old_oracle_call": bool(
            "n1.boundary" in new_pool_source
            or "n1.boundary" in new_boundary_source
        ),
        "post_baseline_source_contains_production_sobol": "qmc.Sobol" in new_boundary_source,
        "post_baseline_source_contains_np_linspace": "np.linspace" in new_boundary_source,
        "dependencies": dependencies,
        "stale_reachability": (
            "The frozen baseline entry point remains immutable and contains its historical "
            "oracle. The post-baseline entry point neither calls nor delegates to it."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = (BASELINE, CANONICAL, CHECKPOINTS, HYBRID_SOURCE, TEACHER_SOURCE)
    protected_before = {str(path): sha256(path) for path in protected_paths}
    baseline = json.loads(BASELINE.read_text())
    audit = binding_audit()
    if (
        audit["post_baseline_source_contains_old_oracle_call"]
        or audit["post_baseline_source_contains_production_sobol"]
        or audit["post_baseline_source_contains_np_linspace"]
    ):
        raise RuntimeError("post-baseline boundary path contains a forbidden fallback")

    boundary = N1BoundarySupportAssembly()
    direct_specs = [
        ("active_d2", 2, 1000, 8730000, None, None),
        ("active_d4", 4, 2000, 8730000, None, None),
        ("active_d5", 5, 2500, 8730000, None, None),
        ("d1_lower_clamp", 1, 1, 910001, None, None),
        ("d2_lower_clamp", 2, 64, 910002, None, None),
        ("d3_first_unclamped", 3, 82, 910003, None, None),
        ("d6_mid_k", 6, 257, 910004, None, None),
        ("d8_k32", 8, 1024, 910005, None, None),
        (
            "d5_physical_bounds", 5, 100, 910006,
            np.asarray([-0.25, -1.0, -2.0, -0.5, 1.5], np.float64),
            np.asarray([0.5, 2.0, 3.0, 2.5, 4.5], np.float64),
        ),
    ]
    direct_rows = []
    for spec in direct_specs:
        row = direct_case(boundary, *spec)
        direct_rows.append(row)
        if row["mismatch"]:
            failure = {
                "status": "FIRST_MISMATCH",
                "stage": "raw_boundary",
                "case": row,
                "binding_audit": audit,
            }
            (OUT / "report.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(json.dumps(failure, indent=2))
            raise SystemExit(1)

    pool_rows = []
    for dimension, sample_count, case_id, iteration in (
        (2, 10, 0, 0),
        (4, 20, 1, 0),
        (5, 25, 2, 0),
        (3, 11, 3, 2),
        (7, 13, 4, 5),
    ):
        row = pool_case(boundary, dimension, sample_count, case_id, iteration)
        pool_rows.append(row)
        if row["mismatch"]:
            failure = {
                "status": "FIRST_MISMATCH",
                "stage": "pool_interface",
                "case": row,
                "binding_audit": audit,
            }
            (OUT / "report.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(json.dumps(failure, indent=2))
            raise SystemExit(1)

    expected_remaining_n1 = [
        "production cKDTree exact-tie ownership",
        "production scalar np.log10",
        "ordered top-min(24,n) neighbor retrieval",
        "production scalar np.exp in Gaussian kernel",
    ]
    removed = {
        "dimension-general initial evaluated design",
        "dimension-general boundary-face support",
    }
    retained_all = [
        row for row in baseline["retained_D_oracles"]
        if row["operation"] not in removed
    ]
    remaining_n1 = [
        row["operation"] for row in retained_all if row["section"] == "N1"
    ]
    inventory = {
        "status": "POST_BASELINE_PEELBACK_INVENTORY",
        "frozen_baseline_manifest": str(BASELINE),
        "frozen_baseline_manifest_sha256": sha256(BASELINE),
        "frozen_baseline_modified": False,
        "eliminated_after_baseline": [
            {
                "operation": "dimension-general initial evaluated design",
                "status": "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE",
            },
            {
                "operation": "dimension-general boundary-face support",
                "status": "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE",
            },
        ],
        "remaining_N1_D_oracles": remaining_n1,
        "retained_D_oracles_all_sections": retained_all,
    }
    (OUT / "post_baseline_D_oracle_inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n"
    )

    protected_after = {str(path): sha256(path) for path in protected_paths}
    instrumentation = Counter()
    forbidden_fallbacks = 0
    for row in direct_rows:
        trace = row["binding_trace"]
        instrumentation["face_resolution"] += trace["face_resolution"]
        instrumentation["grid"] += trace["grid"]
        instrumentation["sobol"] += trace["sobol"]
        instrumentation["affine_coordinates"] += trace["affine"]
        instrumentation["old_boundary_oracle"] += trace["old_boundary_oracle_calls"]
        forbidden_fallbacks += trace["forbidden_fallback_calls"]
    for row in pool_rows:
        instrumentation["face_resolution"] += row["face_resolution_binding_invocations"]
        instrumentation["grid"] += row["grid_binding_invocations"]
        instrumentation["sobol"] += row["sobol_binding_invocations"]
        instrumentation["affine_coordinates"] += row["affine_binding_invocations"]
        instrumentation["pool_merge"] += row["pool_merge_binding_invocations"]
        instrumentation["old_boundary_oracle"] += row["old_boundary_oracle_calls"]
        forbidden_fallbacks += row["forbidden_fallback_calls"]

    source_compatible = (
        baseline["student_source_hashes"][str(HYBRID_SOURCE)] == sha256(HYBRID_SOURCE)
    )
    checkpoint_compatible = (
        baseline["checkpoint_manifest_sha256"] == sha256(CHECKPOINTS)
    )
    passed = bool(
        not any(row["mismatch"] for row in direct_rows + pool_rows)
        and instrumentation["face_resolution"] > 0
        and instrumentation["grid"] > 0
        and instrumentation["sobol"] > 0
        and instrumentation["affine_coordinates"] > 0
        and instrumentation["pool_merge"] > 0
        and instrumentation["old_boundary_oracle"] == 0
        and forbidden_fallbacks == 0
        and remaining_n1 == expected_remaining_n1
        and source_compatible and checkpoint_compatible
        and protected_before == protected_after
    )
    report = {
        "status": (
            "EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_CONTROL_STORAGE"
            if passed else "VALIDATION_FAILED"
        ),
        "binding": "N1_boundary_support",
        "binding_audit": audit,
        "direct_raw_boundary_replay": direct_rows,
        "pool_level_replay": pool_rows,
        "dimension_general_structural_dimensions": [1, 2, 3, 4, 5, 6, 7, 8],
        "instrumentation": dict(instrumentation),
        "forbidden_fallback_count": forbidden_fallbacks,
        "old_boundary_oracle_calls": int(instrumentation["old_boundary_oracle"]),
        "new_parameters": 0,
        "remaining_N1_D_oracles": remaining_n1,
        "expected_remaining_N1_D_oracles": expected_remaining_n1,
        "post_baseline_inventory": str(
            OUT / "post_baseline_D_oracle_inventory.json"
        ),
        "source_compatibility": {
            "hybrid_source_matches_frozen_baseline": source_compatible,
            "checkpoint_manifest_matches_frozen_baseline": checkpoint_compatible,
            "dependency_hashes": audit["dependencies"],
        },
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_hashes_unchanged": protected_before == protected_after,
        "frozen_baseline_modified": False,
        "canonical_822_rerun": False,
        "training_performed": False,
        "other_D_oracles_modified": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "N1_boundary_support": report["status"],
        "direct_cases": len(direct_rows),
        "pool_cases": len(pool_rows),
        "old_boundary_oracle_calls": report["old_boundary_oracle_calls"],
        "forbidden_fallback_count": forbidden_fallbacks,
        "remaining_N1_D_oracles": remaining_n1,
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "binding_audit": audit,
        "direct_cases": direct_rows,
        "pool_cases": pool_rows,
        "instrumentation": dict(instrumentation),
        "remaining_N1_D_oracles": remaining_n1,
        "source_compatibility": report["source_compatibility"],
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
