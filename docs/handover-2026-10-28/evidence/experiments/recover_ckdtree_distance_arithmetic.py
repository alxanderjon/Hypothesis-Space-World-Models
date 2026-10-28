#!/usr/bin/env python3
"""Recover and validate only SciPy 1.13.0 cKDTree distance arithmetic."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import scipy
import torch

from experiments.build_hybrid_student_baseline import metric, representative_records
from experiments.geometry.ckdtree_distance_assembly import CKDTreeEuclideanDistance
from experiments.sho import dimension_general_edirect as production_n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = Path("outputs/ckdtree_distance_recovery")
FREEZE = Path("outputs/deterministic_teacher_frozen")
CHECKPOINT_REFERENCE = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
SCIPY_SOURCE = Path("/tmp/scipy_source_1130/scipy/spatial/ckdtree/src/distance.h")
SCIPY_BINARY = Path(scipy.__file__).parent / "spatial" / "_ckdtree.cpython-311-darwin.so"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_group(path: Path, dimension: int) -> str:
    text = str(path)
    if "checkpointed_resource_edirect" in text:
        return "full_k_sho_d4"
    if "full_k_50_transfer" in text:
        return "full_k_ball_d4"
    if "reflection" in text:
        return "reflection_ball_d5"
    if "kepler" in text:
        return "kepler_d5"
    return f"other_d{dimension}"


def paired_returned_distances(points: np.ndarray, query_rows: int = 64):
    """Use cKDTree only as the frozen identity/distance authority.

    Returned identities are held fixed before arithmetic replay.  This script
    neither reconstructs nor assesses tie ordering or top-k ordering.
    """
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.empty((0, points.shape[1])), np.empty((0, points.shape[1])), np.empty(0)
    row_count = min(query_rows, len(points))
    rows = np.linspace(0, len(points) - 1, row_count, dtype=np.int64)
    queries = points[rows]
    k = min(25, len(points))
    distances, indices = cKDTree(points).query(queries, k=k)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    paired_queries = np.repeat(queries, k, axis=0)
    paired_sites = points[indices.reshape(-1)]
    return paired_queries, paired_sites, distances.reshape(-1)


def validate_integration(assembly: CKDTreeEuclideanDistance):
    rows = []
    grouped_query, grouped_site, grouped_reference = defaultdict(list), defaultdict(list), defaultdict(list)
    spans = defaultdict(list)
    for item in representative_records():
        record = item["record"]
        site_count = min(20, len(record["points"]))
        sites = np.asarray(record["points"][:site_count], dtype=np.float64)
        pool = production_n1.pool(item["dimension"], site_count, item["case_id"], 0)
        reference, indices = cKDTree(sites).query(pool, k=2)
        queries = np.repeat(pool, 2, axis=0)
        paired_sites = sites[indices.reshape(-1)]
        reference_flat = reference.reshape(-1)
        dimension = item["dimension"]
        begin = sum(len(values) for values in grouped_reference[dimension])
        end = begin + len(reference_flat)
        grouped_query[dimension].append(queries)
        grouped_site[dimension].append(paired_sites)
        grouped_reference[dimension].append(reference_flat)
        spans[dimension].append((begin, end, item))

    for dimension in sorted(grouped_query):
        result = assembly(np.concatenate(grouped_query[dimension]),
                          np.concatenate(grouped_site[dimension]))
        reference = np.concatenate(grouped_reference[dimension])
        for begin, end, item in spans[dimension]:
            exact = metric(reference[begin:end], result[begin:end])
            local_query = np.concatenate(grouped_query[dimension])[begin:end]
            local_site = np.concatenate(grouped_site[dimension])[begin:end]
            separate_rounding = np.sqrt(np.sum(
                (local_query - local_site) * (local_query - local_site), axis=1
            ))
            baseline = metric(reference[begin:end], separate_rounding)
            rows.append({
                "system": item["system"], "dimension": item["dimension"],
                "state_path": str(item["record"]["state_path"]), **exact,
                "separate_multiply_add_baseline": baseline,
            })
    return rows


def choose_broader_records():
    groups = defaultdict(list)
    for record in corpus():
        dimension = int(record["points"].shape[1])
        group = path_group(record["state_path"], dimension)
        if group in {"full_k_sho_d4", "full_k_ball_d4",
                     "reflection_ball_d5", "kepler_d5"}:
            groups[group].append(record)
    chosen = []
    for group in ("full_k_sho_d4", "full_k_ball_d4",
                  "reflection_ball_d5", "kepler_d5"):
        records = groups[group]
        if not records:
            continue
        ids = np.linspace(0, len(records) - 1, min(8, len(records)), dtype=np.int64)
        chosen.extend((group, records[int(index)]) for index in ids)
    return chosen


def validate_broader(assembly: CKDTreeEuclideanDistance):
    batches = []
    for group, record in choose_broader_records():
        query, site, reference = paired_returned_distances(record["points"])
        batches.append((group, record, query, site, reference))
    grouped_query, grouped_site = defaultdict(list), defaultdict(list)
    for group, _, query, site, _ in batches:
        dimension = query.shape[1]
        grouped_query[dimension].append(query)
        grouped_site[dimension].append(site)
    predictions, offsets = {}, defaultdict(int)
    for dimension in sorted(grouped_query):
        predictions[dimension] = assembly(
            np.concatenate(grouped_query[dimension]),
            np.concatenate(grouped_site[dimension]),
        )
    rows = []
    for group, record, _, _, reference in batches:
        dimension = int(record["points"].shape[1])
        begin, end = offsets[dimension], offsets[dimension] + len(reference)
        exact = metric(reference, predictions[dimension][begin:end])
        offsets[dimension] = end
        rows.append({
            "group": group, "dimension": dimension,
            "state_path": str(record["state_path"]), **exact,
        })
    summaries = []
    for group in sorted({row["group"] for row in rows}):
        local = [row for row in rows if row["group"] == group]
        summaries.append({
            "group": group, "records": len(local),
            "values": sum(row["values"] for row in local),
            "bit_mismatches": sum(row["bit_mismatches"] for row in local),
            "max_ulp_error": max(row["max_ulp_error"] for row in local),
        })
    return rows, summaries


def main() -> None:
    started = time.monotonic()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    if scipy.__version__ != "1.13.0":
        raise RuntimeError(f"SciPy 1.13.0 required, got {scipy.__version__}")
    if not SCIPY_SOURCE.exists() or not SCIPY_BINARY.exists():
        raise RuntimeError("official source extraction and production wheel are required")

    checkpoint_reference = json.loads(CHECKPOINT_REFERENCE.read_text())
    checkpoint_before = checkpoint_hashes()
    if checkpoint_before != checkpoint_reference:
        raise RuntimeError("checkpoint inventory differs from frozen teacher")
    freeze = json.loads((FREEZE / "freeze_manifest.json").read_text())
    frozen_sources_before = {
        path: sha256(Path(path)) for path in freeze["sources"]
    }

    assembly = CKDTreeEuclideanDistance()
    integration = validate_integration(assembly)
    if any(row["bit_mismatches"] for row in integration):
        raise RuntimeError("integration-corpus distance replay failed")
    broader_rows, broader_summary = validate_broader(assembly)

    checkpoint_after = checkpoint_hashes()
    frozen_sources_after = {
        path: sha256(Path(path)) for path in freeze["sources"]
    }
    intact = (checkpoint_before == checkpoint_after == checkpoint_reference and
              frozen_sources_before == frozen_sources_after)
    passed = (not any(row["bit_mismatches"] for row in integration) and
              not any(row["bit_mismatches"] for row in broader_rows) and intact)

    report = {
        "status": ("EXACT_LEARNED_ASSEMBLY_WITH_DETERMINISTIC_IMPLEMENTATION_ORDER"
                   if passed else "FAILED_EXACT_REPLAY"),
        "scope": "SciPy 1.13.0 cKDTree p=2 point-point distance arithmetic only",
        "training": False, "new_parameters": 0,
        "scipy_version": scipy.__version__,
        "source_authority": {
            "release": "SciPy 1.13.0",
            "file": "scipy/spatial/ckdtree/src/distance.h",
            "source_sha256": sha256(SCIPY_SOURCE),
            "production_binary": str(SCIPY_BINARY),
            "production_binary_sha256": sha256(SCIPY_BINARY),
            "compiler": "clang 15.0.0, macOS arm64 wheel",
        },
        "source_operation_order": [
            "delta = u[i] - v[i]",
            "four independent accumulators acc[0:4]",
            "acc[lane] = RN_fma(delta*delta + acc[lane]) for each full four-coordinate block",
            "s = ((acc[0] + acc[1]) + acc[2]) + acc[3]",
            "s = RN_fma(delta*delta + s) for each tail coordinate",
            "distance = correctly-rounded sqrt(s)",
        ],
        "d4_order": "four fused squares from zero; left-associated four-lane reduction; sqrt",
        "d5_order": "d4 order; then fused square-add of coordinate 5 into the reduced sum; sqrt",
        "cause_of_previous_d5_one_ulp_errors": (
            "The prior generic assembly materialized RN(delta[4]*delta[4]) and then added it. "
            "The compiled SciPy wheel emits one fused multiply-add for that tail coordinate, "
            "rounding delta[4]^2 + the four-lane sum only once. The source also permits fused "
            "lane accumulation for dimensions with multiple four-coordinate blocks."
        ),
        "binary_evidence": {
            "main_lane_fma_addresses": ["0x55520", "0x55524", "0x55528", "0x5552c"],
            "lane_reduction_addresses": ["0x5553c", "0x55540", "0x55544"],
            "tail_fma_address": "0x55604",
            "sqrt_address_examples": ["0x56b2c", "0x5a0f4"],
        },
        "reused_components": [
            "locked subtraction", "locked addition", "locked multiplication",
            "validated exact product-tail assembly", "locked two-sum arithmetic assembly",
            "N1CorrectlyRoundedSqrt",
        ],
        "deterministic_scaffolding": [
            "four accumulator storage slots", "dimension loop and lane indexing",
            "fixed source/binary parenthesization",
        ],
        "integration_corpus": integration,
        "broader_corpus": broader_summary,
        "broader_record_details": broader_rows,
        "checkpoint_and_teacher_integrity": intact,
        "implementation": "experiments/geometry/ckdtree_distance_assembly.py",
        "implementation_sha256": sha256(Path("experiments/geometry/ckdtree_distance_assembly.py")),
        "validator_sha256": sha256(Path(__file__)),
        "dimension_general_arithmetic": True,
        "bit_exact_wheel_scope": "active teacher dimensions d=4 and d=5 (same scalar branch for d<12)",
        "requires_dimension_dependent_implementation_order_scaffold": True,
        "dimension_note": (
            "The arithmetic primitives and four-lane source algorithm are dimension-general, "
            "but the clang arm64 wheel selects a separately vectorized implementation at d>=12. "
            "No current teacher trace uses that branch, so the locked exact contract fails closed "
            "outside d<12 rather than claiming unvalidated compiler-order equivalence."
        ),
        "excluded": ["cKDTree tie ownership", "ordered top-24 retrieval", "hybrid integration past distance"],
        "next_integration_operation": (
            "nearest-owner comparison/routing away from exact ties; exact-tie ownership remains "
            "the following unresolved operation"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "integration_replay.json").write_text(json.dumps(integration, indent=2) + "\n")
    (OUT / "broader_replay.json").write_text(json.dumps(broader_rows, indent=2) + "\n")
    (OUT / "ckdtree_distance_locked_assembly.json").write_text(json.dumps({
        key: report[key] for key in (
            "status", "scope", "training", "new_parameters", "source_authority",
            "source_operation_order", "reused_components", "deterministic_scaffolding",
            "implementation", "implementation_sha256", "dimension_general_arithmetic",
            "bit_exact_wheel_scope", "requires_dimension_dependent_implementation_order_scaffold",
        )
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"], "integration": integration,
        "broader": broader_summary, "integrity": intact,
        "report": str(OUT / "report.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
