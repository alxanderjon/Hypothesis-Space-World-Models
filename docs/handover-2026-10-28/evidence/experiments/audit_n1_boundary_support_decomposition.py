#!/usr/bin/env python3
"""Read-only decomposition of the live dimension-general N1 boundary support."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import scipy

from experiments.sho import dimension_general_edirect as n1
from experiments.sho import run_stage1b_edirect_vs_direct as teacher


OUT = Path("outputs/hybrid_student_boundary_support_decomposition")
BASELINE = Path("outputs/hybrid_student_baseline_frozen/freeze_manifest.json")
CANONICAL = Path("outputs/canonical_frozen_teacher_regression_822/corpus_manifest.json")
LIVE_STUDENT = Path("experiments/resume_hybrid_student_after_ckdtree.py")
LIVE_TEACHER = Path("experiments/sho/dimension_general_edirect.py")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def array_sha(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def bits(value: float) -> str:
    return f"0x{np.float64(value).view(np.uint64):016x}"


def replay(dimension: int) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    sample_count = 5 * dimension
    requested = min(
        teacher.MC_FACTOR_PER_SITE * sample_count,
        teacher.MC_CANDIDATE_CAP,
    )
    case_id = 0
    iteration = 0
    seed = teacher.SEARCH_SEED + 104729 * case_id + 1009 * iteration
    points, identities = n1.boundary(dimension, requested, seed)
    count = min(257, max(9, int(math.ceil(math.sqrt(requested)))))
    expected_blocks = 2 * dimension
    block_rows = []
    for block_index in range(expected_blocks):
        start = block_index * count
        stop = start + count
        axis = dimension - 1 - block_index // 2
        side = block_index % 2
        block = points[start:stop]
        block_ids = identities[start:stop]
        block_rows.append({
            "block_index": block_index,
            "axis": axis,
            "side": side,
            "row_range": [start, stop - 1],
            "rows": count,
            "identity_exact": bool(np.all(block_ids == np.array([axis, side]))),
            "fixed_coordinate_exact": bool(np.all(block[:, axis] == side)),
            "first_point": block[0].tolist(),
            "last_point": block[-1].tolist(),
            "coordinate_sha256": array_sha(block),
        })
    unique = np.unique(points, axis=0)
    corner_mask = np.all((points == 0.0) | (points == 1.0), axis=1)
    face_counts = np.sum((points == 0.0) | (points == 1.0), axis=1)
    return {
        "dimension": dimension,
        "sample_count": sample_count,
        "requested": requested,
        "case_id": case_id,
        "iteration": iteration,
        "seed": seed,
        "count_per_face": count,
        "raw_rows": len(points),
        "shape": list(points.shape),
        "dtype": str(points.dtype),
        "identity_shape": list(identities.shape),
        "identity_dtype": str(identities.dtype),
        "unique_rows": len(unique),
        "duplicate_rows": len(points) - len(unique),
        "corner_rows": int(np.count_nonzero(corner_mask)),
        "rows_on_exactly_one_face": int(np.count_nonzero(face_counts == 1)),
        "rows_on_multiple_faces": int(np.count_nonzero(face_counts > 1)),
        "coordinate_sha256": array_sha(points),
        "identity_sha256": array_sha(identities),
        "first_row": points[0].tolist(),
        "last_row": points[-1].tolist(),
        "first_row_bits": [bits(value) for value in points[0]],
        "last_row_bits": [bits(value) for value in points[-1]],
        "blocks": block_rows,
    }, points, identities


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protected_paths = [BASELINE, LIVE_STUDENT, LIVE_TEACHER]
    if CANONICAL.exists():
        protected_paths.append(CANONICAL)
    protected_before = {str(path): sha256(path) for path in protected_paths}

    rows = []
    arrays = {}
    for dimension in (2, 4, 5):
        row, points, identities = replay(dimension)
        rows.append(row)
        arrays[f"d{dimension}_points"] = points
        arrays[f"d{dimension}_identities"] = identities
    np.savez(OUT / "active_boundary_outputs.npz", **arrays)

    source = inspect.getsource(n1.boundary)
    report = {
        "status": "DECOMPOSED_NO_TRAINING",
        "target": "N1 dimension-general boundary-face support",
        "live_path": {
            "teacher_function": "experiments/sho/dimension_general_edirect.py:boundary",
            "teacher_call_site": "experiments/sho/dimension_general_edirect.py:pool",
            "student_call_site": "experiments/resume_hybrid_student_after_ckdtree.py:scheduled_pool",
            "student_status": "explicit temporary production boundary D oracle",
            "source_sha256": sha256(LIVE_TEACHER),
            "function_source_sha256": sha256_bytes(source.encode()),
            "inputs": {
                "d": "positive runtime Python integer",
                "requested": "integer Sobol-prefix request derived from evaluated-site count",
                "seed": "SEARCH_SEED + 104729*case_id + 1009*iteration",
                "lower_upper": "optional float64 vectors; omitted by the live pool call",
            },
            "outputs": {
                "points": "float64 array [2*d*k,d] for d>1; [2,1] for d=1",
                "identities": "int64 array [number_of_points,2] containing (axis,side)",
            },
            "invocation": (
                "once per raw pool construction used by cell proposals and by the "
                "greedy pool (the latter uses iteration+1,000,000)"
            ),
            "state_dependence": {
                "dimension": True,
                "pool_request_or_evaluated_site_count": True,
                "case_and_iteration_via_seed": "d>2 only",
                "evaluated_site_coordinates": False,
                "losses_or_incumbent": False,
                "system_identity": False,
                "physical_bounds": False,
                "live_space": "normalized [0,1]^d",
            },
        },
        "mechanism": {
            "count": "k=min(257,max(9,int(ceil(sqrt(requested)))))",
            "d1": (
                "two explicit unit points [[0],[1]] with identities "
                "[(0,0),(0,1)]"
            ),
            "d2": (
                "free=linspace(0,1,k) as [k,1]; enumerate axes in reverse "
                "order and sides lower then upper"
            ),
            "d_greater_2": (
                "free=first k rows from one freshly seeded scrambled Sobol block "
                "of dimension d-1 and size 2^ceil(log2(k)); the same free matrix "
                "is reused on every face"
            ),
            "block_order": (
                "axis d-1 lower, axis d-1 upper, axis d-2 lower, axis d-2 "
                "upper, ..., axis 0 lower, axis 0 upper"
            ),
            "coordinate_insertion": (
                "for each block, set the selected axis to exact side 0 or 1; "
                "assign free columns in ascending order to the remaining axes"
            ),
            "concatenation": "concatenate all 2*d blocks in block order",
            "mapping": "lower + (upper-lower)*unit in NumPy float64 source order",
            "live_mapping": "lower=0 and upper=1, so output is normalized unit coordinates",
            "randomness": (
                "no nondeterministic randomness; d>2 uses deterministic SciPy "
                "scrambled Sobol coordinates fixed by seed; d<=2 ignores seed"
            ),
            "duplicates": (
                "boundary() performs no deduplication. d=2 emits each corner twice "
                "across adjacent face blocks. Later pool merge performs exact row "
                "deduplication and lexicographic ordering through an existing B assembly."
            ),
            "clipping": False,
            "branches": ["d==1", "d==2", "d>2"],
            "ties": "none inside generation; duplicate identities survive until pool merge",
        },
        "classification": [
            {
                "suboperation": "integer square/comparison and min/max routing for k",
                "class": "B",
                "reason": (
                    "locked multiplication/comparison can test k*k>=requested and "
                    "locked routing supplies the 9/257 clamps; this exactly expresses "
                    "ceil(sqrt(integer)) without a new scalar primitive"
                ),
            },
            {
                "suboperation": "least-k search loop and counter",
                "class": "C",
                "reason": "deterministic variable-length integer iteration/state",
            },
            {
                "suboperation": "d=1 endpoint constants",
                "class": "C",
                "reason": "constant construction and storage",
            },
            {
                "suboperation": "d=2 free-coordinate values",
                "class": "B",
                "reason": (
                    "the current d=2 boundary recipe was already validated in the "
                    "locked N1 boundary/merge work; shared add/divide arithmetic "
                    "expresses its linspace coordinates"
                ),
            },
            {
                "suboperation": "d=2 grid index iteration/order",
                "class": "C",
                "reason": "deterministic index and storage traversal",
            },
            {
                "suboperation": "d>2 free-coordinate block",
                "class": "B",
                "reason": (
                    "reuse the locked dimension-general scrambled-Sobol coordinate "
                    "assembly and locked full-block sizing/prefix assembly at dimension d-1"
                ),
            },
            {
                "suboperation": "reverse-axis/lower-upper enumeration and identity labels",
                "class": "C",
                "reason": "loops, integer indices, constants, and storage order",
            },
            {
                "suboperation": "insert side coordinate and route free columns",
                "class": "C",
                "reason": "constant assignment, masks, indexing, and storage only",
            },
            {
                "suboperation": "normalized-to-bounded affine map",
                "class": "B",
                "reason": "existing exact normalized-to-physical affine assembly",
            },
            {
                "suboperation": "concatenate blocks",
                "class": "C",
                "reason": "variable-length storage",
            },
            {
                "suboperation": "cross-source merge/dedup/order after boundary generation",
                "class": "B",
                "reason": "already integrated exact pool merge/dedup/order assembly; outside this target",
            },
        ],
        "dimension_generality": {
            "runtime_dimension_general": True,
            "fixed_d4_d5_slots": False,
            "point_count": "2*d*k for d>1, linear in d rather than 2^d",
            "fixed_corner_count": False,
            "corners": (
                "explicitly arise from d=2 linspace endpoints; d>2 construction "
                "samples face interiors and does not enumerate 2^d corners"
            ),
            "bounds_assumption": "axis-aligned finite vectors with elementwise upper>lower",
            "special_cases": ["d=1 endpoints", "d=2 deterministic grid", "d>2 Sobol faces"],
            "library_limit": (
                "d>2 inherits the installed Sobol direction-table maximum for free "
                "dimension d-1; this is far beyond active d=4/d=5 but is a structural library bound"
            ),
            "active_requested_cap_effect": (
                "with MC_CANDIDATE_CAP=32768, active k<=182, so the nominal k=257 "
                "cap is not reached by the live pool schedule"
            ),
        },
        "active_replay": rows,
        "active_output_artifact": {
            "path": str(OUT / "active_boundary_outputs.npz"),
            "sha256": sha256(OUT / "active_boundary_outputs.npz"),
        },
        "smallest_unresolved_target": {
            "genuinely_unresolved_D_operation": None,
            "boundary_support_reducible": True,
            "training_required": False,
            "next_peel": (
                "assemble and validate only the requested->k integer face-resolution "
                "route, then compose the already locked Sobol/grid, coordinate insertion, "
                "ordering, and affine components"
            ),
            "why_this_is_next": (
                "It is the earliest live arithmetic not yet directly bound in the boundary "
                "path; it is an exact assembly target, not a learned-model target."
            ),
        },
        "changes": {
            "teacher": False,
            "hybrid_baseline": False,
            "canonical_822": False,
            "checkpoints": False,
            "training": False,
            "search_behavior": False,
            "other_D_oracles": False,
        },
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "protected_hashes_before": protected_before,
    }
    protected_after = {str(path): sha256(path) for path in protected_paths}
    report["protected_hashes_after"] = protected_after
    report["protected_hashes_unchanged"] = protected_before == protected_after
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "live_path": report["live_path"],
        "active_replay": rows,
        "smallest_unresolved_target": report["smallest_unresolved_target"],
        "protected_hashes_unchanged": report["protected_hashes_unchanged"],
    }, indent=2))


if __name__ == "__main__":
    main()
