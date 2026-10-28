#!/usr/bin/env python3
"""Integrate and replay only the empty-proposal global fallback."""
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
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_empty_proposal_fallback_binding")
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


def compare(pool, nearest):
    points = np.asarray(pool, np.float64)
    values = np.asarray(nearest, np.float64)
    selected, meta = hybrid.empty_proposal_fallback_assembly(points, values)
    teacher_order = np.argsort(values)[::-1]
    teacher_selected = next(
        (int(index) for index in teacher_order if values[index] > 1.0e-12), None,
    )
    comparison_errors = tie_errors = 0
    for left, right, result in meta["comparisons"]:
        expected = bool(np.float64(left) < np.float64(right))
        comparison_errors += int(result != expected)
        tie_errors += int(np.float64(left) == np.float64(right) and result)
    assembled_list = (
        np.empty((0, points.shape[1]), np.float64)
        if selected is None else points[[selected]]
    )
    teacher_list = (
        np.empty((0, points.shape[1]), np.float64)
        if teacher_selected is None else points[[teacher_selected]]
    )
    return selected, Counter(
        calls=1,
        candidates=len(values),
        ranking_comparisons=meta["comparison_count"],
        threshold_decisions=len(meta["threshold_decisions"]),
        input_candidate_bit_mismatches=0,
        input_distance_bit_mismatches=0,
        ranking_comparison_mismatches=comparison_errors,
        tie_routing_mismatches=tie_errors,
        full_order_mismatches=int(
            not np.array_equal(meta["ascending_order"], np.argsort(values))
        ),
        selected_candidate_mismatches=int(selected != teacher_selected),
        selected_coordinate_bit_mismatches=bit_errors(assembled_list, teacher_list),
        final_proposal_list_bit_mismatches=bit_errors(assembled_list, teacher_list),
        reverse_np_argsort_fallback_invocations=int(
            meta["reverse_np_argsort_fallback_invocations"]
        ),
        no_eligible_candidate=int(teacher_selected is None),
    )


def saved_states(limit_per_dimension=3):
    records = {str(item["state_path"]): item for item in corpus()}
    used = Counter()
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
        sites = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension = sites.shape[1]
        if dimension not in (4, 5) or used[dimension] >= limit_per_dimension:
            continue
        api = dimension_general.composition(dimension)
        data = api["_cell_data"](
            sites, losses, int(traces[0]["case_id"]), iteration,
        )
        used[dimension] += 1
        yield {
            "pool": data.points, "nearest": data.distances,
            "dimension": dimension, "state": row["state"],
            "iteration": iteration,
        }
        if all(used[d] >= limit_per_dimension for d in (4, 5)):
            break


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    live_source = inspect.getsource(hybrid.student_cell_state)
    helper_source = inspect.getsource(hybrid.empty_proposal_fallback_assembly)
    forbidden = {
        "reverse_argsort_in_live_path": live_source.count("np.argsort(distances)[::-1]"),
        "np_argsort_in_fallback_assembly": helper_source.count("np.argsort"),
    }
    mismatch_keys = (
        "input_candidate_bit_mismatches", "input_distance_bit_mismatches",
        "ranking_comparison_mismatches", "tie_routing_mismatches",
        "full_order_mismatches", "selected_candidate_mismatches",
        "selected_coordinate_bit_mismatches", "final_proposal_list_bit_mismatches",
        "reverse_np_argsort_fallback_invocations",
    )
    total = Counter()
    by_dimension = defaultdict(Counter)
    first_mismatch = None
    saved_rows = []
    for row in saved_states():
        selected, local = compare(row["pool"], row["nearest"])
        total.update(local)
        by_dimension[row["dimension"]].update(local)
        saved_rows.append({
            "dimension": row["dimension"], "state": row["state"],
            "iteration": row["iteration"], "candidate_count": len(row["nearest"]),
            "selected": selected,
            **{key: int(local[key]) for key in mismatch_keys},
        })
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = saved_rows[-1]

    controls = []
    for dimension in (4, 5):
        def pool(count):
            return np.arange(count * dimension, dtype=np.float64).reshape(count, dimension)

        control_values = [
            ("unique_winner", np.asarray([0.2, 0.9, 0.4, 0.1])),
            ("exact_tie_small", np.asarray([1.0, 1.0, 0.5, 0.2])),
            # Length > SMALL_QUICKSORT exercises NumPy's non-stable partition
            # tie order rather than merely a reverse stable ordering.
            ("exact_tie_quicksort", np.asarray([
                1.0, 1.0, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08,
                0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0,
            ])),
            ("signed_zero_tie", np.asarray([-0.0, +0.0, 0.5, 0.5])),
            ("all_candidates_filtered", np.asarray([
                0.0, np.nextafter(np.float64(1.0e-12), np.float64(0.0)),
                np.float64(1.0e-12),
            ])),
            ("threshold_above", np.asarray([
                np.float64(1.0e-12),
                np.nextafter(np.float64(1.0e-12), np.float64(np.inf)),
            ])),
        ]
        for name, values in control_values:
            selected, local = compare(pool(len(values)), values)
            total.update(local)
            controls.append({
                "name": name, "dimension": dimension,
                "selected": selected, "distance_bits": values.view(np.uint64).tolist(),
                "no_eligible_candidate": int(local["no_eligible_candidate"]),
                **{key: int(value) for key, value in local.items()},
            })
            if first_mismatch is None and any(local[key] for key in mismatch_keys):
                first_mismatch = controls[-1]

    # Preserve the ordinary live path and continuous replay. The fallback itself
    # was forced above on actual saved pool/distance inputs even when production's
    # selected-cell proposals made the branch unnecessary in these two states.
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
        cell_state = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, cell_state, log)
        live.append({
            "dimension": dimension,
            "state": str(record["state_path"]),
            "fallback_needed_in_live_state": bool(
                log.counts["empty-proposal global fallback"]
            ),
            "production_argsort_fallback_invocations": 0,
            "proposal_bit_mismatches": int(cell_state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(continuous["internal_state_errors"]),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    after = checkpoint_hashes()
    passed = bool(
        saved_rows and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and any(row["no_eligible_candidate"] for row in controls)
        and all(
            row["production_argsort_fallback_invocations"] == 0
            and row["proposal_bit_mismatches"] == 0
            and row["continuous_internal_state_errors"] == 0
            and row["continuous_controller_errors"] == 0
            and row["continuous_winner_exact"] for row in live
        )
        and before == after == expected_checkpoints
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "empty_proposal_fallback_binding",
        "teacher_semantics": [
            "branch iff no selected-cell proposal survived duplicate filtering",
            "candidate pool is the complete canonical reservoir data.points",
            "scores are each reservoir point's production nearest-site distance",
            "rank with NumPy-1.26 default indirect quicksort ascending, traverse reverse",
            "select first reverse-order candidate with distance > 1e-12",
            "return no proposal if every candidate is at or below the threshold",
        ],
        "binding_invocation_count": int(total["calls"]),
        "fallback_cases_exercised": int(total["calls"]),
        "saved_fallback_inputs_exercised": len(saved_rows),
        "candidate_scores_checked": int(total["candidates"]),
        "ranking_comparisons_checked": int(total["ranking_comparisons"]),
        "threshold_decisions_checked": int(total["threshold_decisions"]),
        "production_argsort_fallback_count": int(
            total["reverse_np_argsort_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "reverse_np_argsort_fallback_invocations"},
        "saved_state_forced_replays": saved_rows,
        "by_dimension": {
            str(key): {name: int(value) for name, value in counts.items()}
            for key, counts in sorted(by_dimension.items())
        },
        "controls": controls,
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
        "next_unresolved_A_B_binding": "normalized-to-physical affine map",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "empty_proposal_fallback_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "fallback_cases_exercised": report["fallback_cases_exercised"],
        "production_argsort_fallback_count": report[
            "production_argsort_fallback_count"
        ],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("empty-proposal fallback binding replay failed")


if __name__ == "__main__":
    main()
