#!/usr/bin/env python3
"""Integrate and replay only continuous-Powell Sobol multistart generation."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model
from experiments.sho.validate_n1_continuous_powell_assembly import SEARCH_SEED


OUT = Path("outputs/hybrid_student_continuous_sobol_starts_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")


def frozen_sources_unchanged() -> bool:
    manifest = json.loads(FREEZE.read_text())
    return all(
        Path(path).exists()
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in manifest["sources"].items()
    )


def bit_errors(left, right) -> int:
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def teacher_starts(dimension, incumbent):
    requested = 2 * dimension
    exponent = int(math.ceil(math.log2(requested)))
    engine = qmc.Sobol(dimension, scramble=True, seed=SEARCH_SEED)
    full = engine.random_base2(exponent)
    prefix = full[:requested]
    starts = np.vstack((np.asarray(incumbent, np.float64), prefix))
    return starts, full, prefix, engine, requested, exponent


def compare(dimension, incumbent, log=None):
    student, state = hybrid.continuous_sobol_multistarts(
        dimension, incumbent, log,
    )
    teacher, full, prefix, engine, requested, exponent = teacher_starts(
        dimension, incumbent,
    )
    return student, state, Counter(
        calls=1,
        coordinate_values=int(full.size),
        complete_start_values=int(teacher.size),
        requested_count_mismatches=int(state["requested"] != requested),
        exponent_mismatches=int(state["exponent"] != exponent),
        block_size_mismatches=int(state["generated"] != 2 ** exponent),
        generated_length_mismatches=int(len(state["full_block"]) != len(full)),
        full_block_coordinate_bit_mismatches=bit_errors(state["full_block"], full),
        prefix_coordinate_bit_mismatches=bit_errors(state["prefix"], prefix),
        incumbent_coordinate_bit_mismatches=bit_errors(state["incumbent"], incumbent),
        complete_start_coordinate_bit_mismatches=bit_errors(student, teacher),
        ordering_mismatches=int(not np.array_equal(
            np.asarray(student).view(np.uint64), teacher.view(np.uint64),
        )),
        initial_state_word_mismatches=int(np.count_nonzero(
            np.asarray(state["engine_initial_state"], dtype=engine._shift.dtype)
            != engine._shift
        )),
        final_state_word_mismatches=int(np.count_nonzero(
            np.asarray(state["engine_final_state"], dtype=engine._quasi.dtype)
            != engine._quasi
        )),
        engine_num_generated_mismatches=int(
            state["engine_num_generated"] != engine.num_generated
        ),
        state_semantics_mismatches=int(state["persistent_engine_state"] is not False),
        production_scipy_sobol_fallback_invocations=int(
            state["production_scipy_sobol_fallback_invocations"]
        ),
    )


def saved_calls(per_dimension=8):
    result = {4: [], 5: []}
    seen = {4: set(), 5: set()}
    for record in corpus():
        dimension = int(record["points"].shape[1])
        if dimension not in result or len(result[dimension]) >= per_dimension:
            continue
        if not record["continuous_path"].exists():
            continue
        traces = json.loads(record["continuous_path"].read_text())
        if not traces:
            continue
        identity = (str(record["state_path"]), int(traces[0]["case_id"]))
        if identity in seen[dimension]:
            continue
        seen[dimension].add(identity)
        incumbent = record["points"][int(np.argmin(record["losses"]))]
        result[dimension].append((record, int(traces[0]["case_id"]), incumbent))
    if any(not result[d] for d in result):
        raise RuntimeError("missing saved d=4/d=5 continuous calls")
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    helper_source = inspect.getsource(hybrid.continuous_sobol_multistarts)
    live_source = inspect.getsource(hybrid.continuous_replay)
    forbidden = {
        "qmc_sobol_in_binding": helper_source.count("qmc.Sobol"),
        "qmc_sobol_in_continuous_replay": live_source.count("qmc.Sobol"),
        "random_base2_in_binding": helper_source.count("random_base2"),
        "random_base2_in_continuous_replay": live_source.count("random_base2"),
    }
    mismatch_keys = (
        "requested_count_mismatches", "exponent_mismatches",
        "block_size_mismatches", "generated_length_mismatches",
        "full_block_coordinate_bit_mismatches",
        "prefix_coordinate_bit_mismatches",
        "incumbent_coordinate_bit_mismatches",
        "complete_start_coordinate_bit_mismatches", "ordering_mismatches",
        "initial_state_word_mismatches", "final_state_word_mismatches",
        "engine_num_generated_mismatches", "state_semantics_mismatches",
        "production_scipy_sobol_fallback_invocations",
    )
    total = Counter()
    by_dimension = defaultdict(Counter)
    rows = []
    first_mismatch = None
    saved = saved_calls()
    for dimension in (4, 5):
        for record, case_id, incumbent in saved[dimension]:
            _, state, local = compare(dimension, incumbent)
            total.update(local); by_dimension[dimension].update(local)
            row = {
                "dimension": dimension, "state": str(record["state_path"]),
                "case_id": case_id, "requested": state["requested"],
                "exponent": state["exponent"], "block_size": state["generated"],
                **{key: int(local[key]) for key in mismatch_keys},
            }
            rows.append(row)
            if first_mismatch is None and any(local[key] for key in mismatch_keys):
                first_mismatch = row

    # Production constructs a fresh Sobol engine for every continuous call.
    # Repeated calls must therefore replay identically, not advance shared state.
    repeated = []
    for dimension in (4, 5):
        incumbent = np.linspace(0.125, 0.875, dimension, dtype=np.float64)
        first, first_state, first_stats = compare(dimension, incumbent)
        second, second_state, second_stats = compare(dimension, incumbent)
        total.update(first_stats); total.update(second_stats)
        local_error = int(
            bit_errors(first, second) != 0
            or not np.array_equal(
                first_state["engine_initial_state"],
                second_state["engine_initial_state"],
            )
            or not np.array_equal(
                first_state["engine_final_state"],
                second_state["engine_final_state"],
            )
        )
        total["repeated_call_state_advancement_mismatches"] += local_error
        repeated.append({
            "dimension": dimension, "calls": 2,
            "fresh_engine_replay_mismatch": local_error,
            "persistent_engine_state": first_state["persistent_engine_state"],
        })

    frozen = FrozenProposals()
    largest = load_model()
    live = []
    for dimension in (4, 5):
        record, case_id, _ = saved[dimension][0]
        item = {"record": record, "dimension": dimension, "case_id": case_id}
        log = hybrid.Log()
        state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, state, log)
        starts, start_state, local = compare(
            dimension, continuous["start_state"]["incumbent"],
        )
        total.update(local)
        teacher_complete, _, _, _, _, _ = teacher_starts(
            dimension, continuous["start_state"]["incumbent"],
        )
        live.append({
            "dimension": dimension, "state": str(record["state_path"]),
            "binding_log_invocations": int(
                log.counts["continuous-stage Sobol multistarts"]
            ),
            "requested": int(start_state["requested"]),
            "exponent": int(start_state["exponent"]),
            "block_size": int(start_state["generated"]),
            "complete_multistart_bit_mismatches": bit_errors(
                starts, teacher_complete,
            ),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "multistart_winner_mismatches": int(not continuous["winner_exact"]),
            "final_proposal_bit_mismatches": bit_errors(
                continuous["student_winner_point"],
                continuous["teacher_winner_point"],
            ),
            "scipy_sobol_fallback_invocations": int(
                continuous["start_state"][
                    "production_scipy_sobol_fallback_invocations"
                ]
            ),
        })

    after = checkpoint_hashes()
    live_exact = all(
        row["binding_log_invocations"] == 1
        and row["complete_multistart_bit_mismatches"] == 0
        and row["continuous_internal_state_errors"] == 0
        and row["continuous_controller_errors"] == 0
        and row["multistart_winner_mismatches"] == 0
        and row["final_proposal_bit_mismatches"] == 0
        and row["scipy_sobol_fallback_invocations"] == 0
        for row in live
    )
    passed = bool(
        rows and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and total["repeated_call_state_advancement_mismatches"] == 0
        and live_exact
        and before == after == expected
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "continuous_sobol_starts_binding",
        "teacher_semantics": [
            "requested = 2 * dimension",
            "block exponent = ceil(log2(requested))",
            "fresh scrambled Sobol engine with fixed SEARCH_SEED per call",
            "generate complete random_base2 block and retain exact first requested rows",
            "prepend current lowest-loss incumbent without changing Sobol order",
        ],
        "required_implementation": (
            "locked Sobol scheduling, scramble, recurrence, scaling, and prefix assemblies"
        ),
        "actual_invoked_implementation": (
            "continuous_sobol_multistarts -> scrambled_sobol_full_block"
        ),
        "binding_invocation_count": int(total["calls"]),
        "production_scipy_sobol_fallback_count": int(
            total["production_scipy_sobol_fallback_invocations"]
        ),
        "coordinate_values_checked": int(total["coordinate_values"]),
        "complete_start_values_checked": int(total["complete_start_values"]),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "production_scipy_sobol_fallback_invocations"},
        "repeated_call_state_advancement_mismatches": int(
            total["repeated_call_state_advancement_mismatches"]
        ),
        "saved_call_replays": rows,
        "saved_calls_by_dimension": {
            str(key): int(value["calls"])
            for key, value in sorted(by_dimension.items())
        },
        "repeated_call_controls": repeated,
        "live_continuous_replay": live,
        "multistart_winner_mismatches": int(sum(
            row["multistart_winner_mismatches"] for row in live
        )),
        "final_proposal_bit_mismatches": int(sum(
            row["final_proposal_bit_mismatches"] for row in live
        )),
        "forbidden_production_paths": forbidden,
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "continuous_powell_binding": "UNCHANGED_INTEGRATED_EXACT",
        "continuous_result_guards_binding": "UNCHANGED",
        "next_unresolved_A_B_binding": "continuous result guards/winner/duplicate route",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "continuous_sobol_starts_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "production_scipy_sobol_fallback_count": report[
            "production_scipy_sobol_fallback_count"
        ],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("continuous Sobol starts binding replay failed")


if __name__ == "__main__":
    main()
