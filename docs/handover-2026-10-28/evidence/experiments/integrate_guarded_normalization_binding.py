#!/usr/bin/env python3
"""Integrate and replay only guarded min/max normalization."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho.audit_n1_guarded_normalization import Audit, reconstruct_inputs
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_edirect_vs_direct import _normalize
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_guarded_normalization_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")


def frozen_sources_unchanged():
    manifest = json.loads(FREEZE.read_text())
    return all(
        Path(path).exists()
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in manifest["sources"].items()
    )


def bit_errors(left, right):
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def compare(values):
    values = np.asarray(values, np.float64)
    output, meta = hybrid.guarded_normalization_assembly(values)
    minimum = np.float64(np.min(values))
    maximum = np.float64(np.max(values))
    spread = np.float64(maximum - minimum)
    scale = np.float64(max(1.0, abs(minimum), abs(maximum)))
    tolerance = np.float64(1.0e-15 * scale)
    guarded = bool(spread <= tolerance)
    teacher = _normalize(values)
    teacher_numerator = teacher_denominator = None
    if not guarded:
        teacher_numerator = values - minimum
        teacher_denominator = np.float64(maximum - minimum)
    return Counter(
        calls=1, values=values.size,
        minimum_bit_mismatches=bit_errors(
            np.asarray([meta["minimum"]]), np.asarray([minimum])
        ),
        maximum_bit_mismatches=bit_errors(
            np.asarray([meta["maximum"]]), np.asarray([maximum])
        ),
        spread_bit_mismatches=bit_errors(
            np.asarray([meta["spread"]]), np.asarray([spread])
        ),
        scale_bit_mismatches=bit_errors(
            np.asarray([meta["scale"]]), np.asarray([scale])
        ),
        tolerance_bit_mismatches=bit_errors(
            np.asarray([meta["tolerance"]]), np.asarray([tolerance])
        ),
        guard_routing_mismatches=int(meta["guarded"] != guarded),
        numerator_bit_mismatches=(
            0 if guarded else bit_errors(meta["numerator"], teacher_numerator)
        ),
        denominator_bit_mismatches=(
            0 if guarded else bit_errors(
                np.asarray([meta["denominator"]]),
                np.asarray([teacher_denominator]),
            )
        ),
        final_output_bit_mismatches=bit_errors(output, teacher),
        signed_zero_mismatches=int(np.count_nonzero(
            (output == 0.0) & (teacher == 0.0)
            & (np.signbit(output) != np.signbit(teacher))
        )),
        production_normalization_fallback_invocations=int(
            meta["production_normalization_fallback_invocations"]
        ),
        guarded_calls=int(guarded), ordinary_calls=int(not guarded),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    cell_source = inspect.getsource(hybrid.student_cell_state)
    helper_source = inspect.getsource(hybrid.guarded_normalization_assembly)
    forbidden = {
        "production_normalize_calls_in_live_cell_path": cell_source.count(
            "n1.teacher._normalize"
        ),
        "production_normalize_calls_in_assembly": helper_source.count("_normalize("),
        "native_vector_division_in_assembly": helper_source.count(" / "),
    }

    audit = Audit()
    reconstructed_states = reconstruct_inputs(audit)
    total = Counter()
    by_kind = defaultdict(Counter)
    by_dimension = defaultdict(Counter)
    first_mismatch = None
    mismatch_keys = (
        "minimum_bit_mismatches", "maximum_bit_mismatches",
        "spread_bit_mismatches", "scale_bit_mismatches",
        "tolerance_bit_mismatches", "guard_routing_mismatches",
        "numerator_bit_mismatches", "denominator_bit_mismatches",
        "final_output_bit_mismatches", "signed_zero_mismatches",
        "production_normalization_fallback_invocations",
    )
    for values, kind, dimension in zip(
        audit.arrays, audit.kinds, audit.dimensions,
    ):
        local = compare(values)
        total.update(local)
        by_kind[kind].update(local)
        by_dimension[int(dimension)].update(local)
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "kind": kind, "dimension": int(dimension),
                **{key: int(local[key]) for key in mismatch_keys},
            }

    below = np.nextafter(np.float64(1.0e-15), np.float64(0.0))
    above = np.nextafter(np.float64(1.0e-15), np.float64(np.inf))
    controls = [
        ("positive_zero_range", np.asarray([0.0, 0.0])),
        ("signed_zero_range", np.asarray([-0.0, +0.0])),
        ("constant_negative", np.asarray([-3.0, -3.0, -3.0])),
        ("guard_below", np.asarray([0.0, below])),
        ("guard_equal", np.asarray([0.0, np.float64(1.0e-15)])),
        ("guard_above", np.asarray([0.0, above])),
        ("ordinary_unit", np.asarray([0.0, 1.0])),
        ("ordinary_signed", np.asarray([-3.0, 0.0, 5.0])),
    ]
    control_rows = []
    for name, values in controls:
        local = compare(values)
        total.update(local)
        control_rows.append({
            "name": name, "values": values.tolist(),
            "input_bits": values.view(np.uint64).tolist(),
            **{key: int(value) for key, value in local.items()},
        })
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "kind": f"control:{name}",
                **{key: int(local[key]) for key in mismatch_keys},
            }

    # Force all four call kinds through the actual live cell path, then retain
    # the established continuous Powell replay as a downstream exactness check.
    records = corpus()
    frozen = FrozenProposals()
    largest = load_model()
    live = []
    for dimension in (4, 5):
        record = next(
            row for row in records
            if row["points"].shape[1] == dimension
            and row["continuous_path"].exists()
            and json.loads(row["continuous_path"].read_text())
        )
        case_id = int(json.loads(record["continuous_path"].read_text())[0]["case_id"])
        item = {"record": record, "dimension": dimension, "case_id": case_id}
        log = hybrid.Log()
        state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, state, log)
        live.append({
            "dimension": dimension, "state": str(record["state_path"]),
            "implementation": log.implementation[
                "guarded min-max normalization"
            ][1],
            "binding_invocations": int(log.counts[
                "guarded min-max normalization"
            ]),
            "expected_minimum_call_kinds": [
                "cell_size", "LOO_error", "candidate_distance",
                "candidate_prediction",
            ],
            "production_normalization_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    after = checkpoint_hashes()
    passed = bool(
        reconstructed_states
        and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and all(
            row["binding_invocations"] >= 4
            and row["production_normalization_fallback_invocations"] == 0
            and row["proposal_bit_mismatches"] == 0
            and row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"]
            for row in live
        )
        and before == after == expected_checkpoints
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "guarded_normalization_binding",
        "actual_invoked_implementation": (
            "deterministic min/max + locked subtract/abs/max/multiply/guard/"
            "repeated-denominator-subtract/divide assembly"
        ),
        "operation_order_preserved": True,
        "established_cell_states": reconstructed_states,
        "binding_invocation_count": int(total["calls"]),
        "normalized_values_checked": int(total["values"]),
        "production_normalization_fallback_count": int(
            total["production_normalization_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "production_normalization_fallback_invocations"},
        "guarded_calls": int(total["guarded_calls"]),
        "ordinary_calls": int(total["ordinary_calls"]),
        "by_kind": {
            key: {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(by_kind.items())
        },
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(by_dimension.items())
        },
        "controls": control_rows,
        "forbidden_production_paths": forbidden,
        "live_proposal_replay": live,
        "full_live_proposal_replay_exact": all(
            row["proposal_bit_mismatches"] == 0
            and row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"]
            for row in live
        ),
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected_checkpoints,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "retained_D_oracles": [
            "np.log10", "ordered top-k retrieval", "np.exp",
        ],
        "next_unresolved_A_B_binding": "selected-cell weight",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "guarded_normalization_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "normalized_values_checked": report["normalized_values_checked"],
        "production_normalization_fallback_count": (
            report["production_normalization_fallback_count"]
        ),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("guarded normalization binding replay failed")


if __name__ == "__main__":
    main()
