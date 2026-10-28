#!/usr/bin/env python3
"""Focused integration replay for N1 loss floor clamp only."""
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
from experiments.sho.recover_n1_loss_encoding import FLOOR, corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_loss_floor_clamp_binding")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")


def frozen_sources_unchanged():
    manifest = json.loads(FREEZE.read_text())
    return all(Path(path).exists() and
               hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in manifest["sources"].items())


def compare(values):
    raw = np.asarray(values, np.float64)
    log = hybrid.Log()
    points = np.arange(len(raw) * 2, dtype=np.float64).reshape(len(raw), 2)
    surface, meta = hybrid.gaussian_surface_with_clamp_assembly(points, raw, log)
    teacher_clamped = np.maximum(raw, FLOOR)
    teacher_route = raw < FLOOR
    assembled_route = np.subtract(raw, FLOOR) < 0.0
    return dict(
        raw=raw, teacher_clamped=teacher_clamped,
        assembled_clamped=meta["clamped_losses"],
        teacher_route=teacher_route, assembled_route=assembled_route,
        surface_values=surface.values,
        scalar_invocations=int(log.counts["loss floor clamp"]),
        fallback_invocations=int(meta["production_clamp_fallback_invocations"]),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    source = inspect.getsource(hybrid.gaussian_surface_with_clamp_assembly)
    constructor_fallbacks = source.count("GaussianLocalResponse(")
    maximum_fallbacks = source.count("np.maximum")
    if constructor_fallbacks or maximum_fallbacks:
        raise RuntimeError("live clamp helper retains a production clamp path")

    states = Counter()
    state_calls = scalar_invocations = fallback_invocations = 0
    raw_input_mismatches = routing_mismatches = output_bit_mismatches = 0
    encoded_value_mismatches = 0
    first_mismatch = None
    rows = []
    records = corpus()
    for item in records:
        d = int(item["points"].shape[1])
        if d not in (4, 5) or len(item["losses"]) < 20:
            continue
        raw = np.asarray(item["losses"][:20], np.float64)
        result = compare(raw)
        teacher_surface = n1.teacher.GaussianLocalResponse(
            np.asarray(item["points"][:20], np.float64), raw,
        )
        local_raw = int(np.count_nonzero(
            result["raw"].view(np.uint64) != raw.view(np.uint64)
        ))
        local_route = int(np.count_nonzero(
            result["teacher_route"] != result["assembled_route"]
        ))
        local_output = int(np.count_nonzero(
            result["teacher_clamped"].view(np.uint64) !=
            result["assembled_clamped"].view(np.uint64)
        ))
        local_encoded = int(np.count_nonzero(
            result["surface_values"].view(np.uint64) !=
            teacher_surface.values.view(np.uint64)
        ))
        raw_input_mismatches += local_raw
        routing_mismatches += local_route
        output_bit_mismatches += local_output
        encoded_value_mismatches += local_encoded
        scalar_invocations += result["scalar_invocations"]
        fallback_invocations += result["fallback_invocations"]
        state_calls += 1
        states[d] += 1
        row = dict(dimension=d, state=str(item["state_path"]), values=len(raw),
                   raw_input_mismatches=local_raw,
                   routing_mismatches=local_route,
                   output_bit_mismatches=local_output,
                   encoded_value_mismatches=local_encoded)
        rows.append(row)
        if any((local_raw, local_route, local_output, local_encoded)) and first_mismatch is None:
            first_mismatch = row

    controls = np.asarray([
        np.float64(-0.0), np.float64(0.0),
        np.nextafter(FLOOR, np.float64(0.0)), FLOOR,
        np.nextafter(FLOOR, np.float64(np.inf)),
        np.nextafter(np.float64(0.0), np.float64(np.inf)),
        np.float64(1.0),
    ], np.float64)
    control = compare(controls)
    control_route_errors = int(np.count_nonzero(
        control["teacher_route"] != control["assembled_route"]
    ))
    control_output_errors = int(np.count_nonzero(
        control["teacher_clamped"].view(np.uint64) !=
        control["assembled_clamped"].view(np.uint64)
    ))
    scalar_invocations += control["scalar_invocations"]
    fallback_invocations += control["fallback_invocations"]
    state_calls += 1

    # Force the repaired binding through the actual student call site once per d.
    frozen = FrozenProposals()
    largest = load_model()
    live_calls = []
    for d in (4, 5):
        item = next(row for row in records
                    if row["points"].shape[1] == d and
                    row["continuous_path"].exists() and
                    json.loads(row["continuous_path"].read_text()))
        case_id = int(json.loads(item["continuous_path"].read_text())[0]["case_id"])
        log = hybrid.Log()
        state = hybrid.student_cell_state(
            {"record": item, "dimension": d, "case_id": case_id},
            frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        live_calls.append(dict(
            dimension=d, state=str(item["state_path"]),
            implementation=log.implementation["loss floor clamp"][1],
            scalar_invocations=int(log.counts["loss floor clamp"]),
            fallback_invocations=int(
                state["loss_encoding"]["production_clamp_fallback_invocations"]
            ),
            proposal_bit_mismatches=int(state["proposal_bit_mismatches"]),
        ))

    after = checkpoint_hashes()
    passed = bool(
        states[4] and states[5] and fallback_invocations == 0 and
        constructor_fallbacks == maximum_fallbacks == 0 and
        raw_input_mismatches == routing_mismatches == output_bit_mismatches == 0 and
        encoded_value_mismatches == control_route_errors == control_output_errors == 0 and
        all(row["fallback_invocations"] == 0 and
            row["proposal_bit_mismatches"] == 0 for row in live_calls) and
        before == after == expected_checkpoints and frozen_sources_unchanged()
    )
    report = dict(
        status="INTEGRATED_EXACT" if passed else "FAILED",
        binding="loss_floor_clamp_binding",
        formula="clamped_loss = max(loss, 1e-300)",
        expected_implementation=(
            "assembled_clamp locked subtraction/zero-sign comparison/select"
        ),
        actual_invoked_implementation=(
            "gaussian_surface_with_clamp_assembly -> assembled_clamp"
        ),
        saved_state_calls=state_calls - 1,
        saved_states_by_dimension={str(k): v for k, v in sorted(states.items())},
        scalar_binding_invocation_count=scalar_invocations,
        production_clamp_fallback_count=fallback_invocations,
        production_constructor_clamp_references=constructor_fallbacks,
        raw_input_mismatches=raw_input_mismatches,
        routing_mismatches=routing_mismatches + control_route_errors,
        output_bit_mismatches=output_bit_mismatches + control_output_errors,
        downstream_encoded_value_mismatches=encoded_value_mismatches,
        controls=dict(
            inputs=controls.tolist(), input_bits=controls.view(np.uint64).tolist(),
            routes=control["assembled_route"].tolist(),
            outputs=control["assembled_clamped"].tolist(),
            output_bits=control["assembled_clamped"].view(np.uint64).tolist(),
            routing_mismatches=control_route_errors,
            output_bit_mismatches=control_output_errors,
            signed_negative_zero_clamped_to_floor=bool(
                control["assembled_clamped"][0].view(np.uint64) == FLOOR.view(np.uint64)
            ),
        ),
        live_student_cell_state_calls=live_calls,
        first_mismatch=first_mismatch,
        gaussian_bandwidth_ratio_binding="UNCHANGED_PRODUCTION_FALLBACK",
        checkpoint_hashes_unchanged=(before == after == expected_checkpoints),
        frozen_teacher_sources_unchanged=frozen_sources_unchanged(),
        next_unresolved_binding="Gaussian bandwidth floor and ratio",
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    (OUT / "state_replay.json").write_text(json.dumps(rows, indent=2))
    (OUT / "binding_status.json").write_text(json.dumps({
        "loss_floor_clamp_binding": report["status"],
        "expected_implementation": report["expected_implementation"],
        "actual_invoked_implementation": report["actual_invoked_implementation"],
        "scalar_binding_invocation_count": scalar_invocations,
        "production_clamp_fallback_count": fallback_invocations,
    }, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("loss-floor clamp binding replay failed")


if __name__ == "__main__":
    main()
