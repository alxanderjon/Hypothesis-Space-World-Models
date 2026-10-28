#!/usr/bin/env python3
"""Integrate and replay only selected-cell weight arithmetic."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho import dimension_general_edirect as dimension_general
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_selected_cell_weight_binding")
CELL_CORPUS = Path("outputs/sho/stage1B/n1_scalar_exp_downstream/cell_states.json")
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


def compare(size_normalized, loo_normalized, dimension=None):
    size = np.asarray(size_normalized, np.float64)
    loo = np.asarray(loo_normalized, np.float64)
    output, meta = hybrid.selected_cell_weight_assembly(size, loo)
    teacher_size_term = np.float64(0.5) * size
    teacher_loo_term = np.float64(0.5) * loo
    teacher = teacher_size_term + teacher_loo_term
    return Counter(
        calls=1,
        values=teacher.size,
        size_input_bit_mismatches=bit_errors(meta["size_normalized"], size),
        loo_input_bit_mismatches=bit_errors(meta["loo_normalized"], loo),
        size_term_bit_mismatches=bit_errors(meta["size_term"], teacher_size_term),
        loo_term_bit_mismatches=bit_errors(meta["loo_term"], teacher_loo_term),
        final_weight_bit_mismatches=bit_errors(output, teacher),
        signed_zero_mismatches=int(np.count_nonzero(
            (output == 0.0) & (teacher == 0.0)
            & (np.signbit(output) != np.signbit(teacher))
        )),
        production_weight_arithmetic_fallback_invocations=int(
            meta["production_weight_arithmetic_fallback_invocations"]
        ),
    )


def reconstructed_inputs():
    records = {str(item["state_path"]): item for item in corpus()}
    for row in json.loads(CELL_CORPUS.read_text()):
        item = records.get(row["state"])
        if item is None or not item["events_path"].exists() or not item["continuous_path"].exists():
            continue
        events = json.loads(item["events_path"].read_text())
        traces = json.loads(item["continuous_path"].read_text())
        if not traces:
            continue
        iteration = int(row["iteration"])
        first = next(i for i, event in enumerate(events)
                     if event["source"] == "cell" and int(event["iteration"]) == iteration)
        points = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension = points.shape[1]
        api = dimension_general.composition(dimension)
        data = api["_cell_data"](
            points, losses, int(traces[0]["case_id"]), iteration,
        )
        surface = GaussianLocalResponse(points, losses)
        size_normalized = _normalize(data.sizes[data.selected])
        loo_normalized = _normalize(surface.loo_errors(data.selected))
        yield size_normalized, loo_normalized, dimension, row["state"], iteration


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    live_source = inspect.getsource(hybrid.student_cell_state)
    helper_source = inspect.getsource(hybrid.selected_cell_weight_assembly)
    forbidden = {
        "native_expression_in_live_path": live_source.count(
            "0.5 * size_normalized + 0.5 * loo_normalized"
        ),
        "native_np_multiply_in_assembly": helper_source.count("np.multiply"),
        "native_np_add_in_assembly": helper_source.count("np.add"),
    }

    total = Counter()
    by_dimension = defaultdict(Counter)
    states = 0
    first_mismatch = None
    mismatch_keys = (
        "size_input_bit_mismatches", "loo_input_bit_mismatches",
        "size_term_bit_mismatches", "loo_term_bit_mismatches",
        "final_weight_bit_mismatches", "signed_zero_mismatches",
        "production_weight_arithmetic_fallback_invocations",
    )
    for size, loo, dimension, state, iteration in reconstructed_inputs():
        local = compare(size, loo, dimension)
        total.update(local)
        by_dimension[dimension].update(local)
        states += 1
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "state": state, "iteration": iteration, "dimension": dimension,
                **{key: int(local[key]) for key in mismatch_keys},
            }

    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    controls = [
        ("positive_zero", [0.0], [0.0]),
        ("negative_zero", [-0.0], [-0.0]),
        ("mixed_zero_left", [-0.0], [+0.0]),
        ("mixed_zero_right", [+0.0], [-0.0]),
        ("size_only", [1.0], [0.0]),
        ("loo_only", [0.0], [1.0]),
        ("both_one", [1.0], [1.0]),
        ("half", [0.5], [0.5]),
        ("smallest_subnormal", [tiny], [tiny]),
        ("vector_boundary", [0.0, 1.0, 0.5], [1.0, 0.0, 0.5]),
    ]
    control_rows = []
    for name, size, loo in controls:
        size = np.asarray(size, np.float64)
        loo = np.asarray(loo, np.float64)
        local = compare(size, loo)
        total.update(local)
        control_rows.append({
            "name": name,
            "size_bits": size.view(np.uint64).tolist(),
            "loo_bits": loo.view(np.uint64).tolist(),
            **{key: int(value) for key, value in local.items()},
        })
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "control": name,
                **{key: int(local[key]) for key in mismatch_keys},
            }

    # Force the binding in the live d=4/d=5 path and replay its downstream
    # proposal plus continuous Powell call without replacing later bindings.
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
            "dimension": dimension,
            "state": str(record["state_path"]),
            "implementation": log.implementation["selected-cell weight"][1],
            "binding_invocations": int(log.counts["selected-cell weight"]),
            "production_arithmetic_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(continuous["internal_state_errors"]),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    after = checkpoint_hashes()
    passed = bool(
        states and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and all(
            row["binding_invocations"] > 0
            and row["production_arithmetic_fallback_invocations"] == 0
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
        "binding": "selected_cell_weight_binding",
        "teacher_formula": "weight = (0.5 * size_normalized) + (0.5 * loo_normalized)",
        "operation_order": [
            "size_term = multiply(0.5, size_normalized)",
            "loo_term = multiply(0.5, loo_normalized)",
            "weight = add(size_term, loo_term)",
            "largest-cell override remains a later separate operation",
        ],
        "constants": [0.5],
        "clipping_or_guards": "none",
        "binding_invocation_count": int(total["calls"]),
        "selected_cell_weights_checked": int(total["values"]),
        "production_arithmetic_fallback_count": int(
            total["production_weight_arithmetic_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "production_weight_arithmetic_fallback_invocations"},
        "established_cell_states": states,
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(by_dimension.items())
        },
        "controls": control_rows,
        "forbidden_production_paths": forbidden,
        "live_proposal_replay": live,
        "full_live_proposal_replay_exact": all(
            row["proposal_bit_mismatches"] == 0 for row in live
        ),
        "continuous_powell_replay_exact": all(
            row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"] for row in live
        ),
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected_checkpoints,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "retained_D_oracles": ["np.log10", "ordered top-k retrieval", "np.exp"],
        "next_unresolved_A_B_binding": "ordinary-cell winner",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "selected_cell_weight_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "selected_cell_weights_checked": report["selected_cell_weights_checked"],
        "production_arithmetic_fallback_count": report[
            "production_arithmetic_fallback_count"
        ],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("selected-cell weight binding replay failed")


if __name__ == "__main__":
    main()
