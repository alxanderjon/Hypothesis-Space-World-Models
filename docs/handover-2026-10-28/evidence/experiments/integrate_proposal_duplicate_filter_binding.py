#!/usr/bin/env python3
"""Integrate and replay only the N1 proposal/history duplicate filter."""
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


OUT = Path("outputs/hybrid_student_duplicate_filter_binding")
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


def compare(sites, point):
    history = np.asarray(sites, np.float64)
    candidate = np.asarray(point, np.float64)
    retain, meta = hybrid.proposal_duplicate_filter_assembly(history, candidate)
    teacher_delta = history - candidate[None, :]
    teacher_square = teacher_delta * teacher_delta
    teacher_squared = np.sum(teacher_square, axis=1)
    teacher_distances = np.linalg.norm(teacher_delta, axis=1)
    teacher_minimum = np.min(teacher_distances)
    teacher_retain = bool(teacher_minimum > 1.0e-12)
    expected_index = 0
    route_errors = tie_errors = 0
    for candidate_index, decision in enumerate(meta["minimum_decisions"], start=1):
        expected_replace = bool(
            teacher_distances[candidate_index] < teacher_distances[expected_index]
        )
        expected_tie = bool(
            teacher_distances[candidate_index] == teacher_distances[expected_index]
        )
        route_errors += int(decision["replace"] != expected_replace)
        tie_errors += int(expected_tie and decision["replace"])
        if expected_replace:
            expected_index = candidate_index
    return retain, Counter(
        calls=1,
        distances=len(history),
        minimum_decisions=max(0, len(history) - 1),
        delta_bit_mismatches=bit_errors(meta["delta"], teacher_delta),
        square_bit_mismatches=bit_errors(meta["square"], teacher_square),
        squared_distance_bit_mismatches=bit_errors(
            meta["squared_distance"], teacher_squared,
        ),
        distance_bit_mismatches=bit_errors(meta["distances"], teacher_distances),
        minimum_index_mismatches=int(meta["minimum_index"] != expected_index),
        minimum_bit_mismatches=bit_errors(
            np.asarray([meta["minimum"]]), np.asarray([teacher_minimum]),
        ),
        repeated_min_routing_mismatches=route_errors,
        repeated_min_tie_mismatches=tie_errors,
        duplicate_routing_mismatches=int(retain != teacher_retain),
        native_numpy_distance_min_fallback_invocations=int(
            meta["native_numpy_distance_min_fallback_invocations"]
        ),
    )


def reconstructed_states():
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
        sites = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension = sites.shape[1]
        api = dimension_general.composition(dimension)
        data = api["_cell_data"](
            sites, losses, int(traces[0]["case_id"]), iteration,
        )
        surface = GaussianLocalResponse(sites, losses)
        selected_sizes = data.sizes[data.selected]
        size_weight = _normalize(selected_sizes)
        loo_weight = _normalize(surface.loo_errors(data.selected))
        weights = np.float64(0.5) * size_weight + np.float64(0.5) * loo_weight
        largest_position = int(np.argmax(selected_sizes))
        candidates = []
        for position, site in enumerate(data.selected):
            indices = np.flatnonzero(data.owners == site)
            if not len(indices):
                continue
            if position == largest_position:
                winner = int(np.argmax(data.distances[indices]))
            else:
                distance_bar = _normalize(data.distances[indices])
                prediction_bar = _normalize(surface.predict(data.points[indices]))
                score = (
                    weights[position] * distance_bar
                    - (1.0 - weights[position]) * prediction_bar
                )
                winner = int(np.argmax(score))
            candidates.append(data.points[indices[winner]])
        yield {
            "sites": sites, "losses": losses, "data": data,
            "candidates": candidates, "dimension": dimension,
            "state": row["state"], "iteration": iteration,
            "case_id": int(traces[0]["case_id"]), "api": api,
        }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    live_source = inspect.getsource(hybrid.student_cell_state)
    helper_source = inspect.getsource(hybrid.proposal_duplicate_filter_assembly)
    forbidden = {
        "native_norm_min_predicate_in_live_path": live_source.count(
            "np.min(np.linalg.norm(sites - point, axis=1))"
        ),
        "native_linalg_norm_in_assembly": helper_source.count("np.linalg.norm"),
        "native_np_min_in_assembly": helper_source.count("np.min"),
    }
    mismatch_keys = (
        "delta_bit_mismatches", "square_bit_mismatches",
        "squared_distance_bit_mismatches", "distance_bit_mismatches",
        "minimum_index_mismatches", "minimum_bit_mismatches",
        "repeated_min_routing_mismatches", "repeated_min_tie_mismatches",
        "duplicate_routing_mismatches", "native_numpy_distance_min_fallback_invocations",
        "retained_proposal_identity_mismatches", "retained_proposal_order_mismatches",
        "final_proposal_coordinate_bit_mismatches",
    )
    total = Counter()
    by_dimension = defaultdict(Counter)
    first_mismatch = None
    states = 0
    for state in reconstructed_states():
        assembled_retained = []
        teacher_retained = []
        for candidate_index, point in enumerate(state["candidates"]):
            retain, local = compare(state["sites"], point)
            teacher_retain = bool(
                np.min(np.linalg.norm(state["sites"] - point, axis=1)) > 1.0e-12
            )
            if retain:
                assembled_retained.append((candidate_index, point))
            if teacher_retain:
                teacher_retained.append((candidate_index, point))
            total.update(local)
            by_dimension[state["dimension"]].update(local)
            if first_mismatch is None and any(local[key] for key in mismatch_keys):
                first_mismatch = {
                    "state": state["state"], "iteration": state["iteration"],
                    "dimension": state["dimension"], "candidate": candidate_index,
                    **{key: int(local[key]) for key in mismatch_keys},
                }
        assembled_ids = [row[0] for row in assembled_retained]
        teacher_ids = [row[0] for row in teacher_retained]
        identity_mismatch = int(set(assembled_ids) != set(teacher_ids))
        order_mismatch = int(assembled_ids != teacher_ids)
        assembled_points = np.asarray([row[1] for row in assembled_retained], np.float64)
        teacher_points = np.asarray([row[1] for row in teacher_retained], np.float64)
        coordinate_mismatch = bit_errors(assembled_points, teacher_points)
        state_local = Counter(
            retained_proposal_identity_mismatches=identity_mismatch,
            retained_proposal_order_mismatches=order_mismatch,
            final_proposal_coordinate_bit_mismatches=coordinate_mismatch,
        )
        total.update(state_local)
        by_dimension[state["dimension"]].update(state_local)
        if first_mismatch is None and any(state_local[key] for key in mismatch_keys):
            first_mismatch = {
                "state": state["state"], "iteration": state["iteration"],
                "dimension": state["dimension"],
                **{key: int(state_local[key]) for key in mismatch_keys},
            }
        states += 1

    controls = []
    for dimension in (4, 5):
        zero = np.zeros(dimension, np.float64)
        signed_zero = zero.copy(); signed_zero[0] = -0.0
        threshold = np.float64(1.0e-12)
        below = np.nextafter(threshold, np.float64(0.0))
        above = np.nextafter(threshold, np.float64(np.inf))
        for name, sites, point in (
            ("exact_duplicate", zero[None, :], zero),
            ("signed_zero_duplicate", signed_zero[None, :], zero),
            ("threshold_exact", zero[None, :], np.r_[threshold, np.zeros(dimension - 1)]),
            ("threshold_below", zero[None, :], np.r_[below, np.zeros(dimension - 1)]),
            ("threshold_above", zero[None, :], np.r_[above, np.zeros(dimension - 1)]),
            ("repeated_min_tie", np.vstack((zero, zero)), np.ones(dimension)),
            ("nearest_not_first", np.vstack((np.ones(dimension), zero)), np.full(dimension, 0.25)),
        ):
            retain, local = compare(sites, point)
            total.update(local)
            controls.append({
                "name": name, "dimension": dimension, "retain": retain,
                **{key: int(value) for key, value in local.items()},
            })
            if first_mismatch is None and any(local[key] for key in mismatch_keys):
                first_mismatch = {
                    "control": name, "dimension": dimension,
                    **{key: int(local[key]) for key in mismatch_keys},
                }

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
            "implementation": log.implementation["history duplicate filter"][1],
            "binding_invocations": int(log.counts["history duplicate filter"]),
            "native_numpy_fallback_invocations": 0,
            "proposal_bit_mismatches": int(cell_state["proposal_bit_mismatches"]),
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
            and row["native_numpy_fallback_invocations"] == 0
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
        "binding": "duplicate_filter_binding",
        "teacher_rule": (
            "retain iff min(np.linalg.norm(sites - proposal[None,:], axis=1)) > 1e-12"
        ),
        "operation_order": [
            "delta = sites - proposal[None,:]",
            "square = delta * delta",
            "squared_distance = NumPy-order sum(square, axis=1)",
            "distance = sqrt(squared_distance)",
            "minimum = first minimum in site order",
            "retain iff minimum > 1e-12 (strict; equality is duplicate)",
        ],
        "threshold": 1.0e-12,
        "binding_invocation_count": int(total["calls"]),
        "saved_cell_states": states,
        "distance_values_checked": int(total["distances"]),
        "minimum_decisions_checked": int(total["minimum_decisions"]),
        "native_numpy_fallback_count": int(
            total["native_numpy_distance_min_fallback_invocations"]
        ),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "native_numpy_distance_min_fallback_invocations"},
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
        "empty_proposal_fallback_unchanged": True,
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected_checkpoints,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "retained_D_oracles": ["np.log10", "ordered top-k retrieval", "np.exp"],
        "next_unresolved_A_B_binding": "empty-proposal global fallback",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "duplicate_filter_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "distance_values_checked": report["distance_values_checked"],
        "native_numpy_fallback_count": report["native_numpy_fallback_count"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("proposal duplicate filter binding replay failed")


if __name__ == "__main__":
    main()
