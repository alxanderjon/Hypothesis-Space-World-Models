#!/usr/bin/env python3
"""Focused replay for the live Sobol block-sizing/prefix binding."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from experiments.resume_hybrid_student_after_ckdtree import Log, scheduled_pool
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = Path("outputs/hybrid_student_sobol_block_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")


def bits_equal(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def source_integrity():
    manifest = json.loads(FREEZE.read_text())
    bad = []
    for name, expected in manifest["sources"].items():
        path = Path(name)
        if (not path.exists() or
                hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            bad.append(name)
    return bad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints:
        raise RuntimeError("checkpoint inventory differs from frozen reference")
    frozen_source_mismatches = source_integrity()
    if frozen_source_mismatches:
        raise RuntimeError(f"frozen teacher changed: {frozen_source_mismatches}")

    states = Counter()
    requested_values = Counter()
    invocation_count = fallback_invocations = 0
    count_errors = prefix_errors = state_errors = pool_errors = 0
    returned_coordinates = generated_coordinates = 0
    first_mismatch = None
    records = []

    for item in corpus():
        dimension = int(item["points"].shape[1])
        if dimension not in (4, 5) or len(item["points"]) < 20:
            continue
        if not item["continuous_path"].exists():
            continue
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        case_id = int(traces[0]["case_id"])
        sample_count = 20
        iteration = 0

        # Prove the live binding cannot reach production n1.pool.
        original_pool = n1.pool
        live_pool_calls = 0
        def forbidden_pool(*args, **kwargs):
            nonlocal live_pool_calls
            live_pool_calls += 1
            raise RuntimeError("live sizing/prefix binding called production n1.pool")
        n1.pool = forbidden_pool
        try:
            log = Log()
            student_pool, meta = scheduled_pool(
                dimension, sample_count, case_id, iteration, log,
            )
        finally:
            n1.pool = original_pool

        fallback_invocations += live_pool_calls + int(
            meta["sizing_prefix_fallback_invocations"]
        )
        invocation_count += int(log.counts["Sobol full-block sizing and prefix"])

        requested = min(
            n1.teacher.MC_FACTOR_PER_SITE * sample_count,
            n1.teacher.MC_CANDIDATE_CAP,
        )
        exponent = int(math.ceil(math.log2(max(2, requested))))
        generated = 2 ** exponent
        seed = n1.teacher.SEARCH_SEED + 104729 * case_id + 1009 * iteration
        teacher_engine = qmc.Sobol(dimension, scramble=True, seed=seed)
        teacher_full = teacher_engine.random_base2(exponent)
        teacher_prefix = teacher_full[:requested]
        teacher_pool = original_pool(dimension, sample_count, case_id, iteration)

        local_count = int(any((
            meta["requested"] != requested,
            meta["exponent"] != exponent,
            meta["generated"] != generated,
            meta["generated_length"] != generated,
            meta["returned_prefix_length"] != requested,
            meta["engine_num_generated"] != generated,
        )))
        local_prefix = int(
            not bits_equal(meta["full_block"], teacher_full) or
            not bits_equal(meta["prefix"], teacher_prefix)
        )
        local_state = int(not np.array_equal(
            np.asarray(meta["engine_final_state"]), teacher_engine._quasi
        ))
        local_pool = int(not bits_equal(student_pool, teacher_pool))
        count_errors += local_count
        prefix_errors += local_prefix
        state_errors += local_state
        pool_errors += local_pool
        states[dimension] += 1
        requested_values[requested] += 1
        returned_coordinates += requested * dimension
        generated_coordinates += generated * dimension
        row = dict(
            state=str(item["state_path"]), dimension=dimension,
            case_id=case_id, requested=requested, exponent=exponent,
            generated=generated, returned=requested,
            engine_num_generated=meta["engine_num_generated"],
            count_error=local_count, prefix_error=local_prefix,
            state_error=local_state, final_pool_error=local_pool,
        )
        records.append(row)
        if any((local_count, local_prefix, local_state, local_pool)) and first_mismatch is None:
            first_mismatch = row

    after = checkpoint_hashes()
    passed = bool(
        states[4] and states[5] and invocation_count == sum(states.values()) and
        fallback_invocations == 0 and count_errors == 0 and
        prefix_errors == 0 and state_errors == 0 and pool_errors == 0 and
        before == after == expected_checkpoints
    )
    report = dict(
        status=("INTEGRATED_EXACT" if passed else "FAILED"),
        binding="sobol_block_sizing_prefix_binding",
        expected_implementation=(
            "validate_sobol_batching_truncation.schedule + exact first-N prefix"
        ),
        actual_invoked_implementation=(
            "scheduled_pool -> validate_sobol_batching_truncation.schedule + full_block[:N]"
        ),
        teacher_inputs=dict(
            sample_count="current evaluated-site count",
            requested="min(MC_FACTOR_PER_SITE*sample_count, MC_CANDIDATE_CAP)",
            dimension="hypothesis dimension (coordinate generation only)",
            seed="SEARCH_SEED + 104729*case_id + 1009*iteration",
        ),
        teacher_outputs=dict(
            exponent="ceil(log2(max(2, requested)))",
            generated="2**exponent",
            returned="full_block[:requested]",
            state="fresh engine advances by generated; prefix slicing does not advance it",
        ),
        invocation_count=invocation_count,
        production_n1_pool_fallback_invocation_count=fallback_invocations,
        states_by_dimension={str(k): v for k, v in sorted(states.items())},
        requested_counts={str(k): v for k, v in sorted(requested_values.items())},
        count_errors=count_errors, prefix_coordinate_errors=prefix_errors,
        state_advancement_errors=state_errors,
        final_returned_pool_coordinate_errors=pool_errors,
        returned_coordinates=returned_coordinates,
        generated_coordinates=generated_coordinates,
        first_mismatch=first_mismatch,
        scrambled_generation_binding="UNCHANGED_PRODUCTION_FALLBACK",
        pool_merge_binding="UNCHANGED_PRODUCTION_FALLBACK",
        checkpoint_hashes_unchanged=(before == after == expected_checkpoints),
        frozen_teacher_sources_unchanged=(not frozen_source_mismatches),
        next_unresolved_binding="scrambled Sobol coordinate generation",
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "state_replay.json").write_text(json.dumps(records, indent=2))
    (OUT / "binding_status.json").write_text(json.dumps({
        "sobol_block_sizing_prefix_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "invocation_count": invocation_count,
        "fallback_invocation_count": fallback_invocations,
    }, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Sobol block sizing/prefix binding replay failed")


if __name__ == "__main__":
    main()
