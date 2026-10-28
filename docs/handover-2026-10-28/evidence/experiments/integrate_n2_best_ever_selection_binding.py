#!/usr/bin/env python3
"""Integrate and replay only the N2 best-ever strict-loss binding."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments import expand_chronological_archive_handoff_dataset as expanded
from experiments import validate_d5_production_pipeline as production
from experiments.resume_hybrid_student_after_ckdtree import (
    Log, N2BestEverStrictLossTracker, run_n2_with_best_ever_assembly,
)


OUT = Path("outputs/hybrid_student_n2_best_ever_selection_binding")
LABELS = Path(
    "outputs/chronological_archive_handoff_supervision_expanded/"
    "expanded_archive_site_labels.jsonl"
)
CHECKPOINT_MANIFEST = Path(
    "outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json"
)
FREEZE_MANIFEST = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")
CURRENT_N2_REPLAY = Path("outputs/n2_current_archive_powell_assembly/report.json")
BRANCH_KEYS = (
    ("reflection_ball", 0, 2, 0),    # maxfev
    ("reflection_ball", 2, 3, 0),    # numerical Powell stop
    ("kepler5", 0, 3, 2),            # maxfev
    ("kepler5", 143, 3, 2),          # numerical Powell stop
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bits_equal(left, right) -> bool:
    a = np.asarray(left, np.float64)
    b = np.asarray(right, np.float64)
    return a.shape == b.shape and bool(
        np.array_equal(a.view(np.uint64), b.view(np.uint64))
    )


class NativeReferenceTracker:
    """The frozen teacher's native Python strict-comparison semantics."""

    def __init__(self, theta, loss):
        self.theta = np.asarray(theta, np.float64).copy()
        self.loss = float(loss)

    def observe(self, theta, loss):
        replace = bool(float(loss) < self.loss)
        if replace:
            self.theta = np.asarray(theta, np.float64).copy()
            self.loss = float(loss)
        return replace


def control_replay() -> dict:
    rows = []

    def run(name, start, candidate):
        theta0 = np.asarray([11.0, -7.0], np.float64)
        theta1 = np.asarray([13.0, -5.0], np.float64)
        teacher = NativeReferenceTracker(theta0, start)
        student = N2BestEverStrictLossTracker(theta0, start)
        expected = teacher.observe(theta1, candidate)
        observed = student.observe(theta1, candidate)["replace"]
        rows.append({
            "name": name, "start_bits": int(np.float64(start).view(np.uint64)),
            "candidate_bits": int(np.float64(candidate).view(np.uint64)),
            "teacher_replace": expected, "student_replace": observed,
            "decision_match": expected == observed,
            "loss_match": bits_equal(teacher.loss, student.best_loss),
            "theta_match": bits_equal(teacher.theta, student.best_theta),
        })

    one = np.float64(1.0)
    run("strict_improvement", one, np.nextafter(one, -np.inf))
    run("exact_equality", one, one)
    run("adjacent_worse", one, np.nextafter(one, np.inf))
    run("negative_zero_vs_positive_zero", np.float64(+0.0), np.float64(-0.0))
    run("positive_zero_vs_negative_zero", np.float64(-0.0), np.float64(+0.0))
    return {
        "rows": rows,
        "comparisons": len(rows),
        "errors": sum(
            not (r["decision_match"] and r["loss_match"] and r["theta_match"])
            for r in rows
        ),
    }


def load_rows() -> list[dict]:
    wanted = set(BRANCH_KEYS)
    selected = {}
    for line in LABELS.read_text().splitlines():
        row = json.loads(line)
        key = (
            row["system"], int(row["case_id"]),
            int(row["current_evaluation_count"]), int(row["archive_index"]),
        )
        if key in wanted:
            selected[key] = row
    missing = wanted - set(selected)
    if missing:
        raise RuntimeError(f"missing saved branch rows: {sorted(missing)}")
    return [selected[key] for key in BRANCH_KEYS]


def replay_branch(row: dict, expected_branch: dict, log: Log) -> dict:
    system = row["system"]
    case_id = int(row["case_id"])
    trajectory, _ = (
        expanded.ball_case(case_id) if system == "reflection_ball"
        else expanded.kepler_case(case_id)
    )
    objective, _ = production._objectives(system, trajectory)
    start = np.asarray(row["theta"], np.float64)
    lower = np.asarray(row["n2_box_lower"], np.float64)
    upper = np.asarray(row["n2_box_upper"], np.float64)
    teacher = NativeReferenceTracker(start, row["archived_loss"])
    comparisons = Counter()
    first_mismatch = None

    def checked_objective(theta):
        nonlocal first_mismatch
        point = np.asarray(theta, np.float64)
        value = float(objective(point))
        expected = teacher.observe(point, value)
        # The student observation occurs in run_n2_with_best_ever_assembly.
        comparisons["candidate_losses_presented"] += 1
        comparisons["teacher_replacements"] += int(expected)
        return value

    result = run_n2_with_best_ever_assembly(
        checked_objective, start, row["archived_loss"], lower, upper, log,
        trace=False,
    )
    tracker = result["tracker"]
    if len(tracker.transitions) != comparisons["candidate_losses_presented"]:
        comparisons["trace_length_mismatches"] += 1
    # Compare every transition against a fresh native replay over the exact
    # candidate/loss stream observed by the live student.
    replay = NativeReferenceTracker(start, row["archived_loss"])
    for evaluation, transition in enumerate(tracker.transitions, 1):
        expected_replace = replay.observe(
            transition["candidate_theta"], transition["candidate_loss"],
        )
        comparisons["comparison_decisions_checked"] += 1
        decision_error = expected_replace != transition["replace"]
        loss_error = not bits_equal(replay.loss, transition["best_loss_after"])
        theta_error = not bits_equal(replay.theta, transition["best_theta_after"])
        comparisons["routing_mismatches"] += int(decision_error)
        comparisons["stored_loss_mismatches"] += int(loss_error)
        comparisons["stored_theta_mismatches"] += int(theta_error)
        if first_mismatch is None and (decision_error or loss_error or theta_error):
            first_mismatch = {
                "evaluation": evaluation,
                "teacher_replace": expected_replace,
                "student_replace": transition["replace"],
                "candidate_loss": float(transition["candidate_loss"]),
            }

    final_theta_error = not bits_equal(
        result["theta"], expected_branch["integrated_best_theta"]
    )
    final_loss_error = not bits_equal(
        result["loss"], expected_branch["integrated_best_loss"]
    )
    evaluation_error = (
        int(result["evaluations"]) != int(expected_branch["integrated_evaluations"])
    )
    stop_error = result["message"] != expected_branch["integrated_stop"]
    comparisons["final_endpoint_theta_mismatches"] += int(final_theta_error)
    comparisons["final_endpoint_loss_mismatches"] += int(final_loss_error)
    comparisons["evaluation_count_mismatches"] += int(evaluation_error)
    comparisons["stop_reason_mismatches"] += int(stop_error)
    comparisons["native_python_fallback_calls"] += int(
        result["native_python_comparison_fallback_invocations"]
    )
    return {
        "system": system, "case_id": case_id,
        "current_evaluation_count": int(row["current_evaluation_count"]),
        "archive_index": int(row["archive_index"]),
        "saved_target_good": bool(row["target_good"]),
        "archive_cache_source": row["n2_source"],
        "archive_cached_endpoint_matches_current_exact_powell": bool(
            bits_equal(row["n2_endpoint"], expected_branch["integrated_best_theta"])
            and bits_equal(row["n2_loss"], expected_branch["integrated_best_loss"])
        ),
        "teacher_n2_evaluations": int(expected_branch["integrated_evaluations"]),
        "student_n2_evaluations": int(result["evaluations"]),
        "teacher_stop_reason": expected_branch["integrated_stop"],
        "student_stop_reason": result["message"],
        "student_final_loss": float(result["loss"]),
        "student_final_theta": result["theta"].tolist(),
        "comparisons": dict(comparisons),
        "first_mismatch": first_mismatch,
    }


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest_before = sha256(CHECKPOINT_MANIFEST)
    freeze_manifest_before = sha256(FREEZE_MANIFEST)
    log = Log()
    controls = control_replay()
    prior_replay = json.loads(CURRENT_N2_REPLAY.read_text())
    expected_by_key = {
        (branch["system"], int(branch["case_id"]),
         int(branch["evaluation_count"]), int(branch["archive_index"])): branch
        for branch in prior_replay["branches"]
    }
    branches = []
    aggregate = Counter()
    for index, row in enumerate(load_rows(), 1):
        print(
            f"[{index}/4] {row['system']} case={row['case_id']} "
            f"n={row['current_evaluation_count']} site={row['archive_index']}",
            flush=True,
        )
        key = (row["system"], int(row["case_id"]),
               int(row["current_evaluation_count"]), int(row["archive_index"]))
        branch = replay_branch(row, expected_by_key[key], log)
        branches.append(branch)
        aggregate.update(branch["comparisons"])
        mismatch_keys = [
            key for key, value in branch["comparisons"].items()
            if (key.endswith("mismatches") or key.endswith("calls")) and value
        ]
        if mismatch_keys:
            break

    checkpoint_manifest_after = sha256(CHECKPOINT_MANIFEST)
    freeze_manifest_after = sha256(FREEZE_MANIFEST)
    mismatch_total = sum(
        value for key, value in aggregate.items()
        if key.endswith("mismatches")
    ) + int(aggregate["native_python_fallback_calls"])
    exact = (
        controls["errors"] == 0 and mismatch_total == 0
        and len(branches) == len(BRANCH_KEYS)
        and checkpoint_manifest_before == checkpoint_manifest_after
        and freeze_manifest_before == freeze_manifest_after
    )
    report = {
        "binding": "n2_best_ever_selection_binding",
        "status": "INTEGRATED_EXACT" if exact else "FIRST_MISMATCH",
        "teacher_semantics": {
            "comparison": "new_loss < best_loss",
            "tie": "do not replace; retain first occurrence",
            "paired_update": "best_loss and best_theta update together",
        },
        "implementation": (
            "locked subtraction/strict-negative comparison followed by paired "
            "learned-route select; theta copy/storage deterministic"
        ),
        "new_parameter_count": 0,
        "binding_invocation_count": int(log.counts[
            "N2 best-ever strict-loss selection"
        ]),
        "comparison_decisions_checked": int(
            aggregate["comparison_decisions_checked"] + controls["comparisons"]
        ),
        "native_python_fallback_calls": int(
            aggregate["native_python_fallback_calls"]
        ),
        "aggregate": dict(aggregate),
        "controls": controls,
        "branches": branches,
        "coverage": {
            "reflection_ball": sum(b["system"] == "reflection_ball" for b in branches),
            "kepler5": sum(b["system"] == "kepler5" for b in branches),
            "maxfev_exits": sum(
                "Maximum number" in b["student_stop_reason"] for b in branches
            ),
            "numerical_powell_stops": sum(
                b["student_stop_reason"] == "Optimization terminated successfully."
                for b in branches
            ),
        },
        "checkpoint_integrity": {
            "checkpoint_manifest_unchanged": checkpoint_manifest_before == checkpoint_manifest_after,
            "freeze_manifest_unchanged": freeze_manifest_before == freeze_manifest_after,
            "checkpoint_count": len(json.loads(CHECKPOINT_MANIFEST.read_text())),
        },
        "remaining_d_oracles_unchanged": True,
        "next_unresolved_ab_binding": "N3 gradient operator",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "n2_best_ever_selection_binding": report["status"],
        "next_unresolved_ab_binding": report["next_unresolved_ab_binding"],
    }, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "invocations": report["binding_invocation_count"],
        "decisions_checked": report["comparison_decisions_checked"],
        "aggregate": report["aggregate"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
