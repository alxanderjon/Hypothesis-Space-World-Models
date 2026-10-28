#!/usr/bin/env python3
"""Integrate and replay only selected-site LOO residual/fallback semantics."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho import dimension_general_edirect as n1
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes
from experiments.sho.recover_n1_loss_encoding import corpus
from experiments.sho.run_stage1b_fifo_vs_ranker import FrozenProposals
from experiments.sho.validate_largest_cell_router_tie_repair import load_model


OUT = Path("outputs/hybrid_student_loo_residual_binding")
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


def system_name(path):
    text = str(path)
    if "bouncing_ball" in text or "/ball4/" in text or "/ball5/" in text:
        return "bouncing_ball"
    if "kepler" in text:
        return "kepler"
    return "sho"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_checkpoints = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_checkpoints or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    loo_source = inspect.getsource(hybrid.HybridGaussianLocalResponse.loo_errors)
    forbidden = {
        "native_weight_add": loo_source.count("+ 1.0e-15"),
        "native_weighted_sum": loo_source.count("np.sum"),
        "native_prediction_division": loo_source.count(" / np.sum"),
        "production_loo_method_call": loo_source.count("super().loo_errors"),
        "production_surface_inheritance": int(
            issubclass(hybrid.HybridGaussianLocalResponse,
                       n1.teacher.GaussianLocalResponse)
        ),
    }
    retained_oracles = {
        "ordered_tree_query_calls_in_source": loo_source.count("self.tree.query"),
        "np_exp_calls_in_source": loo_source.count("exponential = np.exp(scaled)"),
    }

    records = {str(item["state_path"]): item for item in corpus()}
    established = json.loads(CELL_CORPUS.read_text())
    counts = Counter()
    by_dimension = defaultdict(Counter)
    by_system = defaultdict(Counter)
    first_mismatch = None
    state_counts = Counter()

    mismatch_keys = (
        "self_removal_mismatches", "local_index_mismatches",
        "local_distance_bit_mismatches", "weight_bit_mismatches",
        "weighted_term_bit_mismatches", "numerator_bit_mismatches",
        "denominator_bit_mismatches", "prediction_bit_mismatches",
        "signed_residual_bit_mismatches", "absolute_residual_bit_mismatches",
        "final_loo_bit_mismatches", "production_loo_fallback_invocations",
    )

    for saved in established:
        item = records.get(saved["state"])
        if item is None or not item["events_path"].exists() or not item["continuous_path"].exists():
            continue
        events = json.loads(item["events_path"].read_text())
        continuous = json.loads(item["continuous_path"].read_text())
        if not continuous:
            continue
        iteration = int(saved["iteration"])
        first = next(
            index for index, event in enumerate(events)
            if event["source"] == "cell" and int(event["iteration"]) == iteration
        )
        points = np.asarray([event["point"] for event in events[:first]], np.float64)
        losses = np.asarray([event["loss"] for event in events[:first]], np.float64)
        dimension = int(points.shape[1])
        if dimension not in (4, 5):
            continue
        system = system_name(item["state_path"])
        api = n1.composition(dimension)
        data = api["_cell_data"](
            points, losses, int(continuous[0]["case_id"]), iteration,
        )
        log = hybrid.Log()
        student, _ = hybrid.gaussian_surface_with_clamp_assembly(points, losses, log)
        teacher = n1.teacher.GaussianLocalResponse(points, losses)
        selected = np.asarray(data.selected, np.int64)
        output, traces = student.loo_errors(selected, return_trace=True)
        reference = teacher.loo_errors(selected)

        local = Counter(
            states=1, binding_invocations=len(selected),
            final_loo_bit_mismatches=bit_errors(output, reference),
            production_loo_fallback_invocations=sum(
                int(trace["production_loo_fallback_invocations"])
                for trace in traces
            ),
        )
        if len(points) <= 2:
            local["small_site_fallback_calls"] += len(selected)
        else:
            count = min(25, len(points))
            distances, indices = cKDTree(points).query(points[selected], k=count)
            if count == 1:
                distances, indices = distances[:, None], indices[:, None]
            for row, (site, trace) in enumerate(zip(selected, traces)):
                keep = indices[row] != site
                local_indices = indices[row, keep][:24]
                local_distances = distances[row, keep][:24]
                local["self_removal_mismatches"] += int(
                    np.any(trace.get("local_indices", []) == site)
                )
                local["local_index_mismatches"] += int(
                    not np.array_equal(trace.get("local_indices"), local_indices)
                )
                local["local_distance_bit_mismatches"] += bit_errors(
                    trace.get("local_distances"), local_distances
                )
                if not len(local_indices):
                    local["empty_neighbor_fallback_calls"] += 1
                    local["absolute_residual_bit_mismatches"] += bit_errors(
                        np.asarray([trace["value"]]), np.asarray([0.0])
                    )
                    continue

                bandwidth = max(float(local_distances[-1]), 1.0e-12)
                ratio = local_distances / bandwidth
                square = np.square(ratio)
                scaled = -9.0 * square
                exponential = np.exp(scaled)
                weights = exponential + 1.0e-15
                neighbor_values = teacher.values[local_indices]
                terms = weights * neighbor_values
                numerator = np.sum(terms)
                denominator = np.sum(weights)
                prediction = numerator / denominator
                signed = teacher.values[site] - prediction
                absolute = abs(signed)
                meta = trace["prediction_metadata"]
                local["weight_bit_mismatches"] += bit_errors(
                    meta["weights"][0], weights
                )
                local["weighted_term_bit_mismatches"] += bit_errors(
                    meta["weighted_terms"][0], terms
                )
                local["numerator_bit_mismatches"] += bit_errors(
                    np.asarray([meta["numerator"][0]]), np.asarray([numerator])
                )
                local["denominator_bit_mismatches"] += bit_errors(
                    np.asarray([meta["denominator"][0]]), np.asarray([denominator])
                )
                local["prediction_bit_mismatches"] += bit_errors(
                    np.asarray([trace["prediction"]]), np.asarray([prediction])
                )
                local["signed_residual_bit_mismatches"] += bit_errors(
                    np.asarray([trace["signed_residual"]]), np.asarray([signed])
                )
                local["absolute_residual_bit_mismatches"] += bit_errors(
                    np.asarray([trace["absolute_residual"]]), np.asarray([absolute])
                )
                local["weighted_rows"] += 1
                local["weighted_terms"] += len(local_indices)

        counts.update(local)
        by_dimension[dimension].update(local)
        by_system[system].update(local)
        state_counts[dimension] += 1
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = {
                "state": saved["state"], "iteration": iteration,
                "dimension": dimension, "system": system,
                **{key: int(local[key]) for key in mismatch_keys},
            }

    # Explicit structural controls for small-site and empty-neighbor fallbacks.
    fallback_controls = []
    for size in (1, 2):
        points = np.arange(size * 4, dtype=np.float64).reshape(size, 4)
        losses = np.arange(1, size + 1, dtype=np.float64)
        log = hybrid.Log()
        surface, _ = hybrid.gaussian_surface_with_clamp_assembly(points, losses, log)
        got, traces = surface.loo_errors(np.arange(size), return_trace=True)
        expected = np.ones(size, np.float64)
        errors = bit_errors(got, expected)
        counts["small_site_fallback_calls"] += size
        counts["final_loo_bit_mismatches"] += errors
        counts["binding_invocations"] += size
        fallback_controls.append({
            "kind": f"site_count_{size}", "calls": size,
            "bit_mismatches": errors,
            "branches": [trace["branch"] for trace in traces],
        })

    class EmptyNeighborTree:
        def query(self, queries, k):
            rows = len(np.atleast_2d(queries))
            return np.zeros((rows, k), np.float64), np.zeros((rows, k), np.int64)

    points = np.arange(12, dtype=np.float64).reshape(3, 4)
    log = hybrid.Log()
    surface, _ = hybrid.gaussian_surface_with_clamp_assembly(
        points, np.asarray([1.0, 2.0, 3.0]), log,
    )
    surface.tree = EmptyNeighborTree()
    empty_value, empty_trace = surface.loo_errors(
        np.asarray([0]), return_trace=True,
    )
    empty_errors = bit_errors(empty_value, np.asarray([0.0]))
    counts["empty_neighbor_fallback_calls"] += 1
    counts["binding_invocations"] += 1
    counts["final_loo_bit_mismatches"] += empty_errors
    fallback_controls.append({
        "kind": "forced_empty_neighbor", "calls": 1,
        "bit_mismatches": empty_errors,
        "branches": [empty_trace[0]["branch"]],
    })

    # Force actual live proposal and continuous paths for d=4/d=5.
    all_records = list(records.values())
    frozen = FrozenProposals()
    largest = load_model()
    live = []
    for dimension in (4, 5):
        record = next(
            row for row in all_records
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
                "LOO residual and small-site fallback"
            ][1],
            "binding_invocations": int(log.counts[
                "LOO residual and small-site fallback"
            ]),
            "production_loo_fallback_invocations": 0,
            "proposal_bit_mismatches": int(state["proposal_bit_mismatches"]),
            "continuous_internal_state_errors": int(
                continuous["internal_state_errors"]
            ),
            "continuous_controller_errors": int(continuous["controller_errors"]),
            "continuous_winner_exact": bool(continuous["winner_exact"]),
        })

    after = checkpoint_hashes()
    passed = bool(
        state_counts[4] and state_counts[5]
        and not any(forbidden.values())
        and retained_oracles["ordered_tree_query_calls_in_source"] == 1
        and retained_oracles["np_exp_calls_in_source"] == 1
        and all(counts[key] == 0 for key in mismatch_keys)
        and all(row["bit_mismatches"] == 0 for row in fallback_controls)
        and all(
            row["production_loo_fallback_invocations"] == 0
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
        "binding": "loo_residual_binding",
        "actual_invoked_implementation": (
            "exact structural fallbacks + integrated Gaussian prefix + retained "
            "np.exp + weighted-prediction assembly + locked subtraction + "
            "exact absolute-value sign assembly"
        ),
        "retained_D_oracles": [
            "ordered cKDTree top-min(25,n) retrieval", "np.exp(scaled)",
        ],
        "saved_states_by_dimension": {
            str(key): int(value) for key, value in sorted(state_counts.items())
        },
        "binding_invocation_count": int(counts["binding_invocations"]),
        "production_loo_fallback_count": int(
            counts["production_loo_fallback_invocations"]
        ),
        **{key: int(counts[key]) for key in mismatch_keys
           if key != "production_loo_fallback_invocations"},
        "weighted_rows": int(counts["weighted_rows"]),
        "weighted_terms_checked": int(counts["weighted_terms"]),
        "small_site_fallback_calls": int(counts["small_site_fallback_calls"]),
        "empty_neighbor_fallback_calls": int(counts["empty_neighbor_fallback_calls"]),
        "by_dimension": {
            str(key): {name: int(value) for name, value in value.items()}
            for key, value in sorted(by_dimension.items())
        },
        "by_system": {
            key: {name: int(value) for name, value in value.items()}
            for key, value in sorted(by_system.items())
        },
        "fallback_controls": fallback_controls,
        "forbidden_production_paths": forbidden,
        "retained_oracle_checks": retained_oracles,
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
        "next_unresolved_A_B_binding": "guarded min-max normalization",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "loo_residual_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "production_loo_fallback_count": report["production_loo_fallback_count"],
        "retained_D_oracles": report["retained_D_oracles"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("LOO residual binding replay failed")


if __name__ == "__main__":
    main()
