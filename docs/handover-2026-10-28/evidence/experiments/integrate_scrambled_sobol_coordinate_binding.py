#!/usr/bin/env python3
"""Focused integration replay for the locked scrambled-Sobol coordinates."""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy._lib._util import rng_integers
from scipy.stats import qmc, _sobol

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus


OUT = Path("outputs/hybrid_student_scrambled_sobol_binding")
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


def saved_cases(per_dimension=8):
    result = {4: [], 5: []}
    seen = {4: set(), 5: set()}
    for item in corpus():
        d = int(item["points"].shape[1])
        if d not in result or not item["continuous_path"].exists():
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    # Fail if the coordinate assembly itself contains a production Sobol engine.
    source = inspect.getsource(hybrid.scrambled_sobol_full_block)
    coordinate_generator_qmc_calls = source.count("qmc.Sobol")
    if coordinate_generator_qmc_calls:
        raise RuntimeError("coordinate assembly still calls production qmc.Sobol")

    rows = []
    by_dimension = Counter()
    invocation_count = fallback_count = 0
    coordinate_mismatches = state_mismatches = initialization_mismatches = 0
    prefix_mismatches = live_pool_mismatches = 0
    coordinate_values_compared = 0
    first_mismatch = None
    saved = saved_cases()

    for d in (4, 5):
        for item, case_id in saved[d]:
            sample_count = 20
            iteration = 0
            requested = min(
                n1.teacher.MC_FACTOR_PER_SITE * sample_count,
                n1.teacher.MC_CANDIDATE_CAP,
            )
            exponent, generated = hybrid.sobol_schedule(requested)
            seed = n1.teacher.SEARCH_SEED + 104729 * case_id + 1009 * iteration

            log = hybrid.Log()
            live_pool, live = hybrid.scheduled_pool(
                d, sample_count, case_id, iteration, log,
            )
            invocation_count += int(log.counts["scrambled Sobol coordinate generation"] > 0)
            fallback_count += int(
                live["sobol_state"]["production_scipy_sobol_fallback_invocations"]
            )

            teacher = qmc.Sobol(d, scramble=True, seed=seed)
            teacher_full = teacher.random_base2(exponent)
            teacher_prefix = teacher_full[:requested]
            teacher_pool = n1.pool(d, sample_count, case_id, iteration)

            # Reconstruct teacher initialization inputs independently.
            dtype = np.uint32
            directions = np.zeros((d, 30), dtype=dtype)
            _sobol._initialize_v(directions, dim=d, bits=30)
            rng = np.random.default_rng(seed)
            raw_shift = rng_integers(rng, 2, size=(d, 30), dtype=dtype)
            raw_matrix = rng_integers(rng, 2, size=(d, 30, 30), dtype=dtype)
            state = live["sobol_state"]

            local_coordinates = int(np.count_nonzero(
                live["full_block"].view(np.uint64) != teacher_full.view(np.uint64)
            ))
            local_state = int(np.count_nonzero(
                np.asarray(state["final_state"], dtype=teacher._quasi.dtype) !=
                teacher._quasi
            ))
            local_initialization = int(
                not np.array_equal(state["directions"], directions) or
                not np.array_equal(state["raw_shift"], raw_shift) or
                not np.array_equal(state["raw_matrix"], raw_matrix) or
                not np.array_equal(
                    np.asarray(state["initial_state"], dtype=teacher._shift.dtype),
                    teacher._shift,
                ) or
                not np.array_equal(state["scrambled_directions"], teacher._sv)
            )
            local_prefix = int(not bits_equal(live["prefix"], teacher_prefix))
            local_pool = int(not bits_equal(live_pool, teacher_pool))
            coordinate_mismatches += local_coordinates
            coordinate_values_compared += teacher_full.size
            state_mismatches += local_state
            initialization_mismatches += local_initialization
            prefix_mismatches += local_prefix
            live_pool_mismatches += local_pool
            by_dimension[d] += 1
            row = dict(
                dimension=d, state=str(item["state_path"]), case_id=case_id,
                requested=requested, exponent=exponent, generated=generated,
                coordinate_bit_mismatches=local_coordinates,
                initialization_mismatch=local_initialization,
                final_state_word_mismatches=local_state,
                prefix_mismatch=local_prefix, final_pool_mismatch=local_pool,
                actual_implementation=log.implementation[
                    "scrambled Sobol coordinate generation"
                ][1],
            )
            rows.append(row)
            if any((local_coordinates, local_state, local_initialization,
                    local_prefix, local_pool)) and first_mismatch is None:
                first_mismatch = row

    # Consecutive calls through the same cached assembly must be replay-stable.
    repeated = []
    repeat_errors = 0
    for d, seed in ((4, 9026111), (5, 9026222)):
        first, first_state = hybrid.scrambled_sobol_full_block(d, 11, seed)
        second, second_state = hybrid.scrambled_sobol_full_block(d, 11, seed)
        teacher_a = qmc.Sobol(d, scramble=True, seed=seed).random_base2(11)
        teacher_b = qmc.Sobol(d, scramble=True, seed=seed).random_base2(11)
        error = int(not (bits_equal(first, second) and bits_equal(first, teacher_a)
                            and bits_equal(second, teacher_b) and
                            np.array_equal(first_state["final_state"],
                                           second_state["final_state"])))
        repeat_errors += error
        coordinate_values_compared += teacher_a.size + teacher_b.size
        fallback_count += (
            first_state["production_scipy_sobol_fallback_invocations"] +
            second_state["production_scipy_sobol_fallback_invocations"]
        )
        invocation_count += 2
        repeated.append(dict(dimension=d, seed=seed, consecutive_calls=2,
                             coordinate_or_state_error=error))

    after = checkpoint_hashes()
    passed = bool(
        by_dimension[4] and by_dimension[5] and fallback_count == 0 and
        coordinate_mismatches == state_mismatches == initialization_mismatches == 0 and
        prefix_mismatches == live_pool_mismatches == repeat_errors == 0 and
        before == after == expected_checkpoints and frozen_sources_unchanged()
    )
    report = dict(
        status="INTEGRATED_EXACT" if passed else "FAILED",
        binding="scrambled_sobol_coordinate_binding",
        expected_implementation=(
            "locked scramble/parity/XOR recurrence/float64 scaling assembly"
        ),
        actual_invoked_implementation=(
            "scrambled_sobol_full_block with deterministic direction/RNG scaffolding"
        ),
        invocation_count=invocation_count,
        production_scipy_sobol_fallback_count=fallback_count,
        live_saved_states_by_dimension={str(k): v for k, v in sorted(by_dimension.items())},
        coordinate_values_compared=coordinate_values_compared,
        coordinate_bit_mismatches=coordinate_mismatches,
        initialization_mismatches=initialization_mismatches,
        state_mismatches=state_mismatches,
        prefix_mismatches=prefix_mismatches,
        final_pool_mismatches=live_pool_mismatches,
        repeated_consecutive_calls=repeated,
        repeated_call_errors=repeat_errors,
        first_mismatch=first_mismatch,
        direction_initialization="scipy._sobol._initialize_v deterministic scaffold",
        seed_rng="active SciPy-1.13.0 integer-seed semantics: fresh Generator(PCG64)",
        pool_merge_binding="UNCHANGED_PRODUCTION_FALLBACK",
        checkpoint_hashes_unchanged=(before == after == expected_checkpoints),
        frozen_teacher_sources_unchanged=frozen_sources_unchanged(),
        next_unresolved_binding="pool merge, exact deduplication, lexicographic order",
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "state_replay.json").write_text(json.dumps(rows, indent=2))
    (OUT / "binding_status.json").write_text(json.dumps({
        "scrambled_sobol_coordinate_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "invocation_count": invocation_count,
        "production_scipy_sobol_fallback_count": fallback_count,
    }, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("scrambled Sobol coordinate binding replay failed")


if __name__ == "__main__":
    main()
