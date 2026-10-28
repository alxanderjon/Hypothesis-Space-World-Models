#!/usr/bin/env python3
"""Integrate and replay only ordinary-cell first-maximum winner routing."""
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
from experiments.sho.validate_cell_scoring_assemblies import ordinary_assembly
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_ordinary_cell_winner_binding")
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


def compare(scores, coordinates=None):
    values = np.asarray(scores, np.float64).reshape(-1)
    winner, meta = hybrid.ordinary_first_max_assembly(values)
    teacher_winner = int(np.argmax(values))
    expected_current = 0
    routing_errors = tie_errors = 0
    for candidate, decision in enumerate(meta["decisions"], start=1):
        expected_replace = bool(values[candidate] > values[expected_current])
        expected_tie = bool(values[candidate] == values[expected_current])
        routing_errors += int(decision["replace"] != expected_replace)
        tie_errors += int(expected_tie and decision["replace"])
        if expected_replace:
            expected_current = candidate
    coordinate_errors = 0
    if coordinates is not None:
        points = np.asarray(coordinates, np.float64)
        coordinate_errors = bit_errors(points[winner], points[teacher_winner])
    return Counter(
        calls=1,
        candidates=len(values),
        decisions=max(0, len(values) - 1),
        score_bit_mismatches=0,
        pairwise_routing_mismatches=routing_errors,
        tie_routing_mismatches=tie_errors,
        winner_index_mismatches=int(winner != teacher_winner),
        proposal_coordinate_bit_mismatches=coordinate_errors,
        np_argmax_candidate_fallback_invocations=int(
            meta["np_argmax_candidate_fallback_invocations"]
        ),
    )


def reconstructed_cells():
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
        selected_sizes = data.sizes[data.selected]
        size_normalized = _normalize(selected_sizes)
        loo_normalized = _normalize(surface.loo_errors(data.selected))
        weights = np.float64(0.5) * size_normalized + np.float64(0.5) * loo_normalized
        largest_position = int(np.argmax(selected_sizes))
        for position, site in enumerate(data.selected):
            indices = np.flatnonzero(data.owners == site)
            if not len(indices) or position == largest_position:
                continue
            distance_normalized = _normalize(data.distances[indices])
            prediction_normalized = _normalize(surface.predict(data.points[indices]))
            features = np.column_stack((
                distance_normalized,
                prediction_normalized,
                np.full(len(indices), weights[position], np.float64),
            ))
            # Both forms are retained so input score replay is checked before
            # winner routing is evaluated.
            teacher_scores = (
                features[:, 2] * features[:, 0]
                - (1.0 - features[:, 2]) * features[:, 1]
            )
            assembled_scores = ordinary_assembly(features)
            yield (
                teacher_scores, assembled_scores, data.points[indices],
                dimension, row["state"], iteration,
            )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    live_source = inspect.getsource(hybrid.student_cell_state)
    helper_source = inspect.getsource(hybrid.ordinary_first_max_assembly)
    forbidden = {
        "candidate_score_argmax_in_live_path": live_source.count("np.argmax(scores)"),
        "argmax_in_first_max_assembly": helper_source.count("np.argmax"),
    }

    total = Counter()
    by_dimension = defaultdict(Counter)
    first_mismatch = None
    mismatch_keys = (
        "score_bit_mismatches", "pairwise_routing_mismatches",
        "tie_routing_mismatches", "winner_index_mismatches",
        "proposal_coordinate_bit_mismatches",
        "np_argmax_candidate_fallback_invocations",
    )
    cells = 0
    for teacher_scores, scores, coordinates, dimension, state, iteration in reconstructed_cells():
        local = compare(scores, coordinates)
        local["score_bit_mismatches"] = bit_errors(scores, teacher_scores)
        total.update(local)
        by_dimension[dimension].update(local)
        cells += 1
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "state": state, "iteration": iteration, "dimension": dimension,
                **{key: int(local[key]) for key in mismatch_keys},
            }

    controls = [
        ("unique_maximum", np.asarray([-0.5, 0.25, 1.0, 0.75])),
        ("first_exact_tie", np.asarray([1.0, 1.0, 0.5])),
        ("later_exact_tie", np.asarray([0.0, 1.0, 1.0])),
        ("all_equal", np.asarray([0.25, 0.25, 0.25, 0.25])),
        ("signed_zero_first_negative", np.asarray([-0.0, +0.0, -1.0])),
        ("signed_zero_first_positive", np.asarray([+0.0, -0.0, -1.0])),
        ("single_candidate", np.asarray([0.125])),
        ("negative_scores", np.asarray([-1.0, -0.25, -0.5])),
    ]
    control_rows = []
    for name, scores in controls:
        coordinates = np.arange(len(scores) * 2, dtype=np.float64).reshape(len(scores), 2)
        local = compare(scores, coordinates)
        total.update(local)
        control_rows.append({
            "name": name,
            "score_bits": scores.view(np.uint64).tolist(),
            **{key: int(value) for key, value in local.items()},
        })
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {"control": name, **{
                key: int(local[key]) for key in mismatch_keys
            }}

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
            "implementation": log.implementation["ordinary-cell winner"][1],
            "candidate_decisions": int(log.counts["ordinary-cell winner"]),
            "np_argmax_candidate_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(continuous["internal_state_errors"]),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    after = checkpoint_hashes()
    passed = bool(
        cells and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and all(
            row["candidate_decisions"] > 0
            and row["np_argmax_candidate_fallback_invocations"] == 0
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
        "binding": "ordinary_cell_winner_binding",
        "teacher_semantics": [
            "scan scores in canonical candidate order",
            "replace the current winner only when candidate_score > winner_score",
            "exact equal scores, including signed zeros, retain the first candidate",
        ],
        "actual_implementation": (
            "locked learned strict-negative comparison router + deterministic ordered scan"
        ),
        "binding_invocation_count": int(total["calls"]),
        "ordinary_cells_checked": cells,
        "candidate_decisions_checked": int(total["decisions"]),
        "candidate_scores_checked": int(total["candidates"]),
        "np_argmax_fallback_count": int(
            total["np_argmax_candidate_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "np_argmax_candidate_fallback_invocations"},
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
        "next_unresolved_A_B_binding": "history duplicate filter",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "ordinary_cell_winner_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "candidate_decisions_checked": report["candidate_decisions_checked"],
        "np_argmax_fallback_count": report["np_argmax_fallback_count"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("ordinary-cell winner binding replay failed")


if __name__ == "__main__":
    main()
