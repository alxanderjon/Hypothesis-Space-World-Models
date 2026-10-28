#!/usr/bin/env python3
"""Integrate and replay only the normalized-to-physical affine binding."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_normalized_to_physical_binding")
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


def production_map(unit, lower, upper):
    """Frozen teacher order: lower + ((upper - lower) * unit)."""
    u = np.asarray(unit, np.float64)
    lo = np.asarray(lower, np.float64)
    hi = np.asarray(upper, np.float64)
    width = np.subtract(hi, lo)
    scaled = np.multiply(width, u)
    theta = np.add(lo, scaled)
    return theta, width, scaled


def compare(unit, lower, upper, log=None):
    student, meta = hybrid.normalized_to_physical_assembly(
        unit, lower, upper, log,
    )
    teacher, width, scaled = production_map(unit, lower, upper)
    count = int(np.asarray(unit).size)
    return student, Counter(
        calls=1,
        scalar_coordinates=count,
        normalized_input_bit_mismatches=bit_errors(meta["normalized"], unit),
        lower_input_bit_mismatches=bit_errors(meta["lower"], lower),
        upper_input_bit_mismatches=bit_errors(meta["upper"], upper),
        width_bit_mismatches=bit_errors(meta["width"], width),
        scaled_coordinate_bit_mismatches=bit_errors(meta["scaled"], scaled),
        final_coordinate_bit_mismatches=bit_errors(student, teacher),
        signed_zero_mismatches=int(np.count_nonzero(
            (np.asarray(student) == 0.0)
            & (np.signbit(np.asarray(student)) != np.signbit(teacher))
        )),
        production_affine_fallback_invocations=int(
            meta["production_affine_fallback_invocations"]
        ),
    )


def bounds(dimension: int):
    if dimension == 4:
        return (
            np.full(4, -1.99, np.float64),
            np.full(4, 1.99, np.float64),
        )
    # The active five-parameter Kepler domain exercises asymmetric widths.
    return (
        np.asarray([-0.25, -1.0, -2.0, -0.5, 1.5], np.float64),
        np.asarray([0.5, 2.0, 3.0, 2.5, 4.5], np.float64),
    )


def saved_inputs(limit_per_dimension=3):
    used = Counter()
    for record in corpus():
        dimension = int(record["points"].shape[1])
        if dimension not in (4, 5) or used[dimension] >= limit_per_dimension:
            continue
        points = np.asarray(record["points"], np.float64)
        if not len(points):
            continue
        # This binding is pointwise. A bounded prefix provides saved search
        # endpoints, exact boundaries, and ordinary interior coordinates.
        take = min(len(points), 2048)
        used[dimension] += 1
        yield dimension, str(record["state_path"]), points[:take]
        if all(used[d] >= limit_per_dimension for d in (4, 5)):
            break


def controls():
    result = []
    for dimension in (4, 5):
        lo, hi = bounds(dimension)
        zero = np.zeros(dimension, np.float64)
        negative_zero = zero.copy(); negative_zero[0] = -0.0
        one = np.ones(dimension, np.float64)
        interior = np.linspace(0.125, 0.875, dimension, dtype=np.float64)
        near_zero = np.full(
            dimension, np.nextafter(np.float64(0.0), np.float64(1.0)),
        )
        near_one = np.full(
            dimension, np.nextafter(np.float64(1.0), np.float64(0.0)),
        )
        for name, unit in (
            ("u_zero", zero), ("u_one", one), ("interior", interior),
            ("signed_zero", negative_zero),
            ("nextafter_zero_toward_one", near_zero),
            ("nextafter_one_toward_zero", near_one),
        ):
            yield name, dimension, unit, lo, hi
    # A separate mixed-sign asymmetric control makes signed-zero propagation
    # observable at a zero lower bound without changing production semantics.
    lo = np.asarray([0.0, -3.0, 2.0, -0.0, 0.25], np.float64)
    hi = np.asarray([2.0, 5.0, 4.5, 7.0, 1.25], np.float64)
    yield "asymmetric_signed_zero", 5, np.asarray(
        [-0.0, 0.25, 0.5, +0.0, 1.0], np.float64,
    ), lo, hi


def live_replays(total: Counter):
    records = corpus()
    frozen = FrozenProposals()
    largest = load_model()
    rows = []
    for dimension in (4, 5):
        record = next(
            row for row in records
            if row["points"].shape[1] == dimension
            and row["continuous_path"].exists()
            and json.loads(row["continuous_path"].read_text())
        )
        trace = json.loads(record["continuous_path"].read_text())
        item = {
            "record": record, "dimension": dimension,
            "case_id": int(trace[0]["case_id"]),
        }
        log = hybrid.Log()
        state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        lo, hi = bounds(dimension)
        _, proposal_stats = compare(
            state["student_proposals"], lo, hi, log,
        )
        total.update(proposal_stats)
        continuous = hybrid.continuous_replay(item, state, log)
        _, winner_stats = compare(
            continuous["student_winner_point"], lo, hi, log,
        )
        total.update(winner_stats)
        teacher_proposals, _, _ = production_map(
            state["teacher_proposals"], lo, hi,
        )
        student_proposals, _ = hybrid.normalized_to_physical_assembly(
            state["student_proposals"], lo, hi,
        )
        teacher_winner, _, _ = production_map(
            continuous["teacher_winner_point"], lo, hi,
        )
        student_winner, _ = hybrid.normalized_to_physical_assembly(
            continuous["student_winner_point"], lo, hi,
        )
        rows.append({
            "dimension": dimension,
            "state": str(record["state_path"]),
            "proposal_coordinates_mapped": int(state["student_proposals"].size),
            "proposal_normalized_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "proposal_physical_bit_mismatches": bit_errors(
                student_proposals, teacher_proposals,
            ),
            "continuous_starts": int(continuous["starts"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
            "continuous_winner_physical_bit_mismatches": bit_errors(
                student_winner, teacher_winner,
            ),
            "binding_log_invocations": int(
                log.counts["normalized-to-physical affine map"]
            ),
        })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    source = inspect.getsource(hybrid.normalized_to_physical_assembly)
    forbidden = {
        "native_np_subtract": source.count("np.subtract"),
        "native_np_multiply": source.count("np.multiply"),
        "native_np_add": source.count("np.add"),
        "native_infix_upper_minus_lower": source.count("upper_bound - lower_bound"),
        "native_infix_unit_times_width": source.count("unit * width"),
        "native_infix_lower_plus_scaled": source.count("lower_bound + scaled"),
    }
    keys = (
        "normalized_input_bit_mismatches", "lower_input_bit_mismatches",
        "upper_input_bit_mismatches", "width_bit_mismatches",
        "scaled_coordinate_bit_mismatches", "final_coordinate_bit_mismatches",
        "signed_zero_mismatches", "production_affine_fallback_invocations",
    )
    total = Counter()
    by_dimension = defaultdict(Counter)
    saved = []
    first_mismatch = None
    for dimension, state, unit in saved_inputs():
        lo, hi = bounds(dimension)
        _, local = compare(unit, lo, hi)
        total.update(local); by_dimension[dimension].update(local)
        row = {
            "dimension": dimension, "state": state,
            "vectors_checked": len(unit),
            **{key: int(local[key]) for key in keys},
        }
        saved.append(row)
        if first_mismatch is None and any(local[key] for key in keys):
            first_mismatch = row

    control_rows = []
    for name, dimension, unit, lo, hi in controls():
        student, local = compare(unit, lo, hi)
        total.update(local); by_dimension[dimension].update(local)
        row = {
            "name": name, "dimension": dimension,
            "normalized_bits": np.asarray(unit).view(np.uint64).tolist(),
            "physical_bits": np.asarray(student).view(np.uint64).tolist(),
            **{key: int(local[key]) for key in keys},
        }
        control_rows.append(row)
        if first_mismatch is None and any(local[key] for key in keys):
            first_mismatch = row

    live = live_replays(total)
    after = checkpoint_hashes()
    live_exact = all(
        row["proposal_normalized_bit_mismatches"] == 0
        and row["proposal_physical_bit_mismatches"] == 0
        and row["continuous_internal_state_errors"] == 0
        and row["continuous_controller_errors"] == 0
        and row["continuous_winner_exact"]
        and row["continuous_winner_physical_bit_mismatches"] == 0
        and row["binding_log_invocations"]
        == row["proposal_coordinates_mapped"] + row["dimension"]
        for row in live
    )
    passed = bool(
        saved and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in keys)
        and live_exact
        and before == after == expected
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "normalized_to_physical_binding",
        "teacher_source_order": [
            "width = upper - lower",
            "scaled = width * normalized",
            "physical = lower + scaled",
        ],
        "required_implementation": (
            "locked subtraction -> locked multiplication -> locked addition"
        ),
        "actual_invoked_implementation": (
            "normalized_to_physical_assembly using locked arithmetic in source order"
        ),
        "binding_invocation_count": int(total["calls"]),
        "scalar_coordinates_checked": int(total["scalar_coordinates"]),
        "production_affine_fallback_count": int(
            total["production_affine_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in keys
           if key != "production_affine_fallback_invocations"},
        "saved_state_replays": saved,
        "by_dimension": {
            str(key): {name: int(value) for name, value in values.items()}
            for key, values in sorted(by_dimension.items())
        },
        "controls": control_rows,
        "forbidden_native_student_paths": forbidden,
        "live_proposal_replay": live,
        "full_live_d4_d5_proposal_replay_exact": bool(live_exact),
        "continuous_powell_replay_exact": bool(all(
            row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"]
            for row in live
        )),
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "retained_D_oracles": ["np.log10", "ordered top-k retrieval", "np.exp"],
        "next_unresolved_A_B_binding": "continuous result guards/winner/duplicate route",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "normalized_to_physical_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "scalar_coordinates_checked": report["scalar_coordinates_checked"],
        "production_affine_fallback_count": report[
            "production_affine_fallback_count"
        ],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("normalized-to-physical binding replay failed")


if __name__ == "__main__":
    main()
