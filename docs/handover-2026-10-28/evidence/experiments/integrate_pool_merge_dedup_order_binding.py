#!/usr/bin/env python3
"""Focused integration replay for exact N1 pool merge/dedup/order."""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.validate_n1_boundary_merge_dedup import independent_unique


OUT = Path("outputs/hybrid_student_pool_merge_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")


def bits_equal(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def frozen_sources_unchanged():
    manifest = json.loads(FREEZE.read_text())
    return all(Path(path).exists() and
               hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in manifest["sources"].items())


def saved_cases(per_dimension=6):
    result = {4: [], 5: []}
    seen = {4: set(), 5: set()}
    for item in corpus():
        d = int(item["points"].shape[1])
        if d not in result or len(result[d]) >= per_dimension:
            continue
        if not item["continuous_path"].exists():
            continue
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        case_id = int(traces[0]["case_id"])
        if case_id in seen[d]:
            continue
        seen[d].add(case_id)
        result[d].append((item, case_id))
        if all(len(result[k]) >= per_dimension for k in result):
            break
    if any(len(result[d]) < per_dimension for d in result):
        raise RuntimeError("insufficient saved d=4/d=5 states")
    return result


def control_cases():
    pz = np.float64(0.0)
    nz = np.float64(-0.0)
    return {
        "no_duplicates": np.asarray([
            [0.2, 0.3, 0.4, 0.5],
            [0.1, 0.3, 0.4, 0.5],
            [0.2, 0.1, 0.4, 0.5],
        ], np.float64),
        "duplicate_interior_rows": np.asarray([
            [0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5],
            [0.1, 0.3, 0.4, 0.5],
        ], np.float64),
        "duplicate_boundary_interior_rows": np.asarray([
            [pz, 0.25, 0.5, 0.75, 1.0],
            [0.4, 0.2, 0.3, 0.4, 0.5],
            [pz, 0.25, 0.5, 0.75, 1.0],
            [1.0, 0.25, 0.5, 0.75, 1.0],
        ], np.float64),
        "multiple_identical_and_signed_zero": np.asarray([
            [nz, 0.25, 0.5, 0.75, 1.0],
            [pz, 0.25, 0.5, 0.75, 1.0],
            [nz, 0.25, 0.5, 0.75, 1.0],
            [0.3, 0.2, 0.1, 0.4, 0.5],
            [0.3, 0.2, 0.1, 0.4, 0.5],
        ], np.float64),
    }


def compare_merge(raw):
    teacher, teacher_first = np.unique(raw, axis=0, return_index=True)
    assembled, assembled_first = independent_unique(raw)
    row_count_error = int(len(teacher) != len(assembled))
    coordinate_errors = (int(np.count_nonzero(
        teacher.view(np.uint64) != assembled.view(np.uint64)
    )) if teacher.shape == assembled.shape else int(max(teacher.size, assembled.size)))
    dedup_error = int(not np.array_equal(teacher_first, assembled_first))
    ordering_error = int(not bits_equal(teacher, assembled))
    return teacher, assembled, teacher_first, assembled_first, dict(
        raw_rows=len(raw), unique_rows=len(teacher),
        duplicates_removed=len(raw) - len(teacher),
        row_count_error=row_count_error,
        deduplication_error=dedup_error,
        ordering_error=ordering_error,
        coordinate_bit_mismatches=coordinate_errors,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    live_source = inspect.getsource(hybrid.scheduled_pool)
    production_fallback_calls = live_source.count("np.unique")
    if production_fallback_calls:
        raise RuntimeError("live pool still calls production np.unique")

    invocation_count = 0
    by_dimension = Counter()
    row_count_mismatches = dedup_mismatches = ordering_mismatches = 0
    coordinate_bit_mismatches = raw_input_mismatches = 0
    complete_pool_mismatches = 0
    raw_rows = unique_rows = duplicate_rows = 0
    first_mismatch = None
    saved_rows = []

    for d in (4, 5):
        for item, case_id in saved_cases()[d]:
            log = hybrid.Log()
            pool, meta = hybrid.scheduled_pool(d, 20, case_id, 0, log)
            invocation_count += int(log.counts[
                "pool merge, exact deduplication, lexicographic order"
            ] > 0)
            production_fallback_calls += int(
                meta["pool_merge_np_unique_fallback_invocations"]
            )
            requested = meta["requested"]
            seed = meta["seed"]
            teacher_interior = n1.qmc.Sobol(
                d, scramble=True, seed=seed,
            ).random_base2(meta["exponent"])[:requested]
            teacher_faces, _ = n1.boundary(d, requested, seed)
            teacher_raw = np.concatenate((teacher_interior, teacher_faces))
            local_raw = int(not bits_equal(meta["raw_pool"], teacher_raw))
            teacher, _, teacher_first, _, metrics = compare_merge(teacher_raw)
            local_pool = int(not bits_equal(pool, teacher))
            local_first = int(not np.array_equal(
                meta["merge_retained_indices"], teacher_first
            ))
            metrics["deduplication_error"] += local_first
            raw_input_mismatches += local_raw
            complete_pool_mismatches += local_pool
            row_count_mismatches += metrics["row_count_error"]
            dedup_mismatches += metrics["deduplication_error"]
            ordering_mismatches += metrics["ordering_error"]
            coordinate_bit_mismatches += metrics["coordinate_bit_mismatches"]
            raw_rows += metrics["raw_rows"]
            unique_rows += metrics["unique_rows"]
            duplicate_rows += metrics["duplicates_removed"]
            by_dimension[d] += 1
            row = dict(dimension=d, state=str(item["state_path"]),
                       case_id=case_id, raw_input_mismatch=local_raw,
                       complete_pool_mismatch=local_pool, **metrics)
            saved_rows.append(row)
            if any((local_raw, local_pool, *metrics.values())) and first_mismatch is None:
                if (local_raw or local_pool or metrics["row_count_error"] or
                        metrics["deduplication_error"] or metrics["ordering_error"] or
                        metrics["coordinate_bit_mismatches"]):
                    first_mismatch = row

    controls = []
    for name, raw in control_cases().items():
        _, assembled, teacher_first, assembled_first, metrics = compare_merge(raw)
        invocation_count += 1
        raw_rows += metrics["raw_rows"]
        unique_rows += metrics["unique_rows"]
        duplicate_rows += metrics["duplicates_removed"]
        row_count_mismatches += metrics["row_count_error"]
        dedup_mismatches += metrics["deduplication_error"]
        ordering_mismatches += metrics["ordering_error"]
        coordinate_bit_mismatches += metrics["coordinate_bit_mismatches"]
        controls.append(dict(
            name=name, dimension=raw.shape[1], **metrics,
            retained_indices=assembled_first.tolist(),
            retained_coordinate_bits=assembled.view(np.uint64).tolist(),
            signed_zero_first_occurrence_preserved=bool(
                name != "multiple_identical_and_signed_zero" or
                np.signbit(assembled[0, 0])
            ),
        ))

    after = checkpoint_hashes()
    passed = bool(
        by_dimension[4] and by_dimension[5] and production_fallback_calls == 0 and
        row_count_mismatches == dedup_mismatches == ordering_mismatches == 0 and
        coordinate_bit_mismatches == raw_input_mismatches == 0 and
        complete_pool_mismatches == 0 and before == after == expected_checkpoints and
        frozen_sources_unchanged()
    )
    report = dict(
        status="INTEGRATED_EXACT" if passed else "FAILED",
        binding="pool_merge_dedup_order_binding",
        expected_implementation="n1_boundary_merge_dedup independent lexsort assembly",
        actual_invoked_implementation="independent_unique(raw_pool)",
        binding_invocation_count=invocation_count,
        live_saved_states_by_dimension={str(k): v for k, v in sorted(by_dimension.items())},
        production_np_unique_fallback_calls=production_fallback_calls,
        raw_rows_compared=raw_rows, unique_rows_compared=unique_rows,
        duplicate_rows_removed=duplicate_rows,
        raw_input_mismatches=raw_input_mismatches,
        row_count_mismatches=row_count_mismatches,
        deduplication_mismatches=dedup_mismatches,
        ordering_mismatches=ordering_mismatches,
        coordinate_bit_mismatches=coordinate_bit_mismatches,
        complete_candidate_pool_mismatches=complete_pool_mismatches,
        controls=controls, first_mismatch=first_mismatch,
        checkpoint_hashes_unchanged=(before == after == expected_checkpoints),
        frozen_teacher_sources_unchanged=frozen_sources_unchanged(),
        next_unresolved_binding="loss floor clamp",
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "saved_state_replay.json").write_text(json.dumps(saved_rows, indent=2))
    (OUT / "binding_status.json").write_text(json.dumps({
        "pool_merge_dedup_order_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "binding_invocation_count": invocation_count,
        "production_np_unique_fallback_calls": production_fallback_calls,
    }, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("pool merge/dedup/order binding replay failed")


if __name__ == "__main__":
    main()
