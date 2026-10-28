#!/usr/bin/env python3
"""Integrate and replay only the post-Powell continuous-result guards."""
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


OUT = Path("outputs/hybrid_student_continuous_result_guards_binding")
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
    if left is None or right is None:
        return int(left is not right)
    a, b = np.asarray(left, np.float64), np.asarray(right, np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))


def normalize(result, index):
    return {
        "start_id": int(result.get("start_id", index)),
        "point": np.asarray(
            result["x"] if "x" in result else result["point"], np.float64,
        ),
        "prediction": float(
            result["fun"] if "fun" in result else result["prediction"]
        ),
    }


def teacher_guard(results, sites):
    rows = [normalize(result, index) for index, result in enumerate(results)]
    statuses = []
    for row in rows:
        point = row["point"]
        prediction_finite = bool(np.isfinite(row["prediction"]))
        if not prediction_finite:
            raise RuntimeError("Invalid continuous surrogate minimizer")
        finite = bool(np.isfinite(point).all())
        if not finite:
            raise RuntimeError("Invalid continuous surrogate minimizer")
        in_bounds = bool(not np.any((point < 0.0) | (point > 1.0)))
        statuses.append((True, in_bounds, in_bounds))
        if not in_bounds:
            raise RuntimeError("Invalid continuous surrogate minimizer")
    winner = min(
        range(len(rows)),
        key=lambda index: (rows[index]["prediction"], rows[index]["start_id"]),
    )
    point = rows[winner]["point"]
    duplicate = bool(np.any(np.all(np.asarray(sites, np.float64) == point, axis=1)))
    return None if duplicate else point.copy(), {
        "statuses": statuses, "winner": winner,
        "winner_start_id": rows[winner]["start_id"], "duplicate": duplicate,
    }


def compare(results, sites, log=None):
    invocation = Counter(calls=1, start_results=len(results))
    try:
        teacher_proposal, teacher = teacher_guard(results, sites)
        teacher_invalid = False
    except RuntimeError:
        teacher_proposal = None; teacher = None; teacher_invalid = True
    try:
        student_proposal, student = hybrid.continuous_result_guards_assembly(
            results, sites, log,
        )
        student_invalid = False
    except hybrid.ContinuousResultGuardError as error:
        student_proposal = None; student = error.metadata; student_invalid = True

    invocation["invalid_route_mismatches"] = int(teacher_invalid != student_invalid)
    invocation["native_guard_fallback_invocations"] = int(
        student["native_guard_fallback_invocations"]
    )
    teacher_rows = [] if teacher is None else teacher["statuses"]
    for index, status in enumerate(student["statuses"]):
        if teacher_invalid:
            row = normalize(results[index], index)
            point = row["point"]
            prediction_finite = bool(np.isfinite(row["prediction"]))
            coordinates_finite = (
                bool(np.isfinite(point).all()) if prediction_finite else None
            )
            in_bounds = (
                bool(not np.any((point < 0.0) | (point > 1.0)))
                if coordinates_finite else None
            )
            expected = (
                bool(prediction_finite and coordinates_finite),
                in_bounds,
                bool(in_bounds) if in_bounds is not None else False,
            )
        else:
            expected = teacher_rows[index]
        got = (status["finite_result"], status["in_bounds"], status["valid"])
        invocation["status_routing_mismatches"] += int(got != expected)

    if not teacher_invalid and not student_invalid:
        invocation["winner_mismatches"] = int(
            teacher["winner"] != student["winner_index"]
        )
        invocation["start_id_tie_route_mismatches"] = int(
            teacher["winner_start_id"] != student["winner_start_id"]
        )
        invocation["duplicate_route_mismatches"] = int(
            teacher["duplicate"] != student["duplicate"]
        )
        invocation["final_proposal_bit_mismatches"] = bit_errors(
            teacher_proposal, student_proposal,
        )
        invocation["prediction_ties"] = sum(
            int(decision["exact_prediction_tie"])
            for decision in student["winner_decisions"]
        )
    return student_proposal, student, invocation


def saved_calls(per_dimension=8):
    result = {4: [], 5: []}
    for record in corpus():
        dimension = int(record["points"].shape[1])
        if dimension not in result or len(result[dimension]) >= per_dimension:
            continue
        if not record["continuous_path"].exists():
            continue
        traces = json.loads(record["continuous_path"].read_text())
        if not traces:
            continue
        saved = traces[0]
        with np.load(record["state_path"], allow_pickle=False) as data:
            sites = np.asarray(data["points"], np.float64)[:int(saved["site_count"])]
        result[dimension].append((record, saved, sites))
    if any(not result[d] for d in result):
        raise RuntimeError("missing saved d=4/d=5 continuous calls")
    return result


def targeted_controls():
    for dimension in (4, 5):
        base = np.linspace(0.2, 0.8, dimension, dtype=np.float64)
        other = np.linspace(0.25, 0.75, dimension, dtype=np.float64)
        sites = np.vstack((base, np.linspace(0.1, 0.9, dimension)))
        yield "equal_predictions_lowest_start_id", dimension, [
            {"start_id": 0, "point": other, "prediction": 0.25},
            {"start_id": 1, "point": base + 0.01, "prediction": 0.25},
        ], sites
        yield "signed_zero_prediction_tie", dimension, [
            {"start_id": 0, "point": other, "prediction": -0.0},
            {"start_id": 1, "point": base + 0.01, "prediction": +0.0},
        ], sites
        yield "exact_duplicate", dimension, [
            {"start_id": 0, "point": base.copy(), "prediction": 0.1},
            {"start_id": 1, "point": other, "prediction": 0.2},
        ], sites
        yield "nonfinite_prediction", dimension, [
            {"start_id": 0, "point": other, "prediction": np.nan},
        ], sites
        nonfinite_point = other.copy(); nonfinite_point[-1] = np.inf
        yield "nonfinite_coordinate", dimension, [
            {"start_id": 0, "point": nonfinite_point, "prediction": 0.1},
        ], sites
        below = other.copy(); below[0] = np.nextafter(0.0, -np.inf)
        yield "out_of_bounds_below", dimension, [
            {"start_id": 0, "point": below, "prediction": 0.1},
        ], sites
        above = other.copy(); above[-1] = np.nextafter(1.0, np.inf)
        yield "out_of_bounds_above", dimension, [
            {"start_id": 0, "point": above, "prediction": 0.1},
        ], sites


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    source = inspect.getsource(hybrid.continuous_result_guards_assembly)
    live_source = inspect.getsource(hybrid.continuous_replay)
    forbidden = {
        "native_min_in_guard_assembly": source.count("min("),
        "np_isfinite_in_guard_assembly": source.count("np.isfinite"),
        "np_any_in_guard_assembly": source.count("np.any"),
        "np_all_in_guard_assembly": source.count("np.all"),
        "native_student_min_in_live_path": live_source.count("student_winner = min"),
        "native_student_duplicate_in_live_path": live_source.count(
            "student_duplicate = bool(np.any"
        ),
    }
    mismatch_keys = (
        "invalid_route_mismatches", "status_routing_mismatches",
        "winner_mismatches", "start_id_tie_route_mismatches",
        "duplicate_route_mismatches", "final_proposal_bit_mismatches",
        "native_guard_fallback_invocations",
    )
    total = Counter()
    by_dimension = defaultdict(Counter)
    saved_rows = []
    first_mismatch = None
    saved = saved_calls()
    for dimension in (4, 5):
        for record, trace, sites in saved[dimension]:
            _, state, local = compare(trace["starts"], sites)
            total.update(local); by_dimension[dimension].update(local)
            row = {
                "dimension": dimension, "state": str(record["state_path"]),
                "start_results": len(trace["starts"]),
                "teacher_chosen_start": int(trace["chosen_start"]),
                "student_chosen_start": state["winner_index"],
                "teacher_duplicate": bool(trace["exact_duplicate_rejected"]),
                "student_duplicate": state["duplicate"],
                **{key: int(local[key]) for key in mismatch_keys},
            }
            saved_rows.append(row)
            if first_mismatch is None and any(local[key] for key in mismatch_keys):
                first_mismatch = row

    controls = []
    for name, dimension, results, sites in targeted_controls():
        _, state, local = compare(results, sites)
        total.update(local); by_dimension[dimension].update(local)
        invalid = state["winner_index"] is None
        row = {
            "name": name, "dimension": dimension,
            "invalid_result_exit": invalid,
            "winner_start_id": state["winner_start_id"],
            "duplicate": state["duplicate"],
            "prediction_ties": int(local["prediction_ties"]),
            **{key: int(local[key]) for key in mismatch_keys},
        }
        controls.append(row)
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = row

    frozen = FrozenProposals(); largest = load_model()
    live = []
    for dimension in (4, 5):
        record, trace, _ = saved[dimension][0]
        item = {
            "record": record, "dimension": dimension,
            "case_id": int(trace["case_id"]),
        }
        log = hybrid.Log()
        cell = hybrid.student_cell_state(
            item, frozen, hybrid.StrictOwnerAssembly(frozen, log), largest, log,
        )
        continuous = hybrid.continuous_replay(item, cell, log)
        proposal_mismatches = bit_errors(
            continuous["teacher_proposal"], continuous["student_proposal"],
        )
        row = {
            "dimension": dimension, "state": str(record["state_path"]),
            "binding_log_invocations": int(log.counts[
                "continuous result guards/winner/duplicate route"
            ]),
            "powell_internal_state_errors": int(continuous["internal_state_errors"]),
            "powell_controller_errors": int(continuous["controller_errors"]),
            "winner_mismatch": int(
                continuous["teacher_winner"] != continuous["student_winner"]
            ),
            "duplicate_route_mismatch": int(
                continuous["teacher_duplicate"] != continuous["student_duplicate"]
            ),
            "final_proposal_bit_mismatches": proposal_mismatches,
            "winner_exact": bool(continuous["winner_exact"]),
        }
        live.append(row)

    after = checkpoint_hashes()
    live_exact = all(
        row["binding_log_invocations"] == 1
        and row["powell_internal_state_errors"] == 0
        and row["powell_controller_errors"] == 0
        and row["winner_mismatch"] == 0
        and row["duplicate_route_mismatch"] == 0
        and row["final_proposal_bit_mismatches"] == 0
        and row["winner_exact"] for row in live
    )
    passed = bool(
        saved_rows and by_dimension[4]["calls"] and by_dimension[5]["calls"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and live_exact
        and before == after == expected
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "continuous_result_guards_binding",
        "teacher_semantics": [
            "reject the complete call if any prediction/endpoint is nonfinite",
            "reject the complete call if any endpoint coordinate is outside [0,1]",
            "select lowest prediction; exact ties choose lowest canonical start_id",
            "duplicate iff every coordinate exactly equals one evaluated-site row",
            "return none for duplicate, otherwise return the selected endpoint",
        ],
        "actual_invoked_implementation": (
            "locked IEEE finite, subtraction/comparison, equality, Boolean/select assembly"
        ),
        "binding_invocation_count": int(total["calls"]),
        "start_results_checked": int(total["start_results"]),
        "native_guard_fallback_count": int(total[
            "native_guard_fallback_invocations"
        ]),
        **{key: int(total[key]) for key in mismatch_keys
           if key != "native_guard_fallback_invocations"},
        "prediction_ties_checked": int(total["prediction_ties"]),
        "saved_call_replays": saved_rows,
        "saved_calls_by_dimension": {
            str(key): int(value["calls"])
            for key, value in sorted(by_dimension.items())
        },
        "targeted_controls": controls,
        "live_d4_d5_proposal_replay": live,
        "live_replay_exact": bool(live_exact),
        "forbidden_native_student_paths": forbidden,
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "retained_D_oracles": ["np.log10", "ordered top-k retrieval", "np.exp"],
        "next_unresolved_A_B_binding": "archive absorbing-success latch",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "continuous_result_guards_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "native_guard_fallback_count": report["native_guard_fallback_count"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("continuous-result guards binding replay failed")


if __name__ == "__main__":
    main()
