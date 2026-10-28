#!/usr/bin/env python3
"""Integrate and replay only the archive absorbing-success latch."""
from __future__ import annotations

from collections import Counter
import glob
import hashlib
import inspect
import json
from pathlib import Path
import re

import experiments.resume_hybrid_student_after_ckdtree as hybrid
from experiments.sho.deterministic_full_k_framework import hashes as checkpoint_hashes


OUT = Path("outputs/hybrid_student_archive_absorbing_success_binding")
CASES = Path("outputs/deterministic_teacher_readiness/cases")
CHECKPOINTS = Path("outputs/deterministic_teacher_readiness/checkpoint_hashes_after.json")
FREEZE = Path("outputs/deterministic_teacher_frozen/freeze_manifest.json")
KEPLER_CHECKS = Path("outputs/kepler_archive_absorbing_handoff/checks")


def frozen_sources_unchanged() -> bool:
    manifest = json.loads(FREEZE.read_text())
    return all(
        Path(path).exists()
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in manifest["sources"].items()
    )


def expected_replay(events, stop_at_first_success):
    success = False; site = None; rows = []
    for event in events:
        before_success, before_site = success, site
        if not success and bool(event["oracle_good"]):
            success = True; site = event["site_id"]
        rows.append({
            "success_before": before_success,
            "successful_site_before": before_site,
            "success_after": success,
            "successful_site_after": site,
            "stop": success, "continue_n1": not success,
        })
        if success and stop_at_first_success:
            break
    return {"success": success, "successful_site": site, "processed": rows}


def compare(events, stop_at_first_success=True):
    log = hybrid.Log()
    student = hybrid.replay_archive_absorbing_latch(
        events, log, stop_at_first_success,
    )
    teacher = expected_replay(events, stop_at_first_success)
    local = Counter(
        calls=1,
        probe_updates=len(student["processed"]),
        binding_log_invocations=int(log.counts["archive absorbing-success latch"]),
        final_success_mismatches=int(student["success"] != teacher["success"]),
        successful_site_identity_mismatches=int(
            student["successful_site"] != teacher["successful_site"]
        ),
        stop_continue_mismatches=int(
            student["stop"] != teacher["success"]
            or student["continue_n1"] == teacher["success"]
        ),
        metadata_latch_fallback_invocations=int(
            student["metadata_latch_fallback_invocations"]
        ),
    )
    for got, wanted in zip(student["processed"], teacher["processed"]):
        for key in (
            "success_before", "successful_site_before", "success_after",
            "successful_site_after", "stop", "continue_n1",
        ):
            local["latch_routing_mismatches"] += int(got[key] != wanted[key])
        if got["success_before"]:
            local["absorbing_state_violations"] += int(
                not got["success_after"]
                or got["successful_site_after"] != got["successful_site_before"]
                or not got["stop"] or got["continue_n1"]
            )
        local["metadata_latch_fallback_invocations"] += int(
            got["metadata_latch_fallback_invocations"]
        )
    local["archive_replay_mismatches"] = sum(local[key] for key in (
        "final_success_mismatches", "successful_site_identity_mismatches",
        "stop_continue_mismatches", "latch_routing_mismatches",
        "absorbing_state_violations",
    ))
    return student, local


def readiness_histories():
    rows = []
    for path in sorted(CASES.glob("*/*/result.json")):
        payload = json.loads(path.read_text())
        events = [{
            "oracle_good": bool(branch["success"]),
            "site_id": int(branch["archive_index"]),
            "evaluation_count": int(branch["evaluation_count"]),
            "source": str(path),
        } for branch in payload["branch_sequence"]]
        rows.append((path, payload, events))
    if not rows:
        raise RuntimeError("missing deterministic teacher-readiness histories")
    return rows


def read_kepler(case_id, n, one_based_index):
    path = KEPLER_CHECKS / f"case_{case_id:03d}" / (
        f"n{n:05d}_i{one_based_index:05d}.json"
    )
    row = json.loads(path.read_text())
    return {
        "oracle_good": bool(row["full_pipeline_pass"]),
        "site_id": int(row["archive_index"]),
        "evaluation_count": int(row["current_evaluation"]),
        "archived_loss": float(row["archived_loss"]),
        "source": str(path),
    }


def saved_post_success_controls():
    # Same site: GOOD with the n=4 box, BAD with the contracted n=20 box.
    box_good = read_kepler(43, 4, 3)
    box_bad = read_kepler(43, 20, 3)
    # At n=6, newest site 5 is GOOD; site 3 has a lower archived loss but BAD.
    low_loss_good = read_kepler(0, 6, 6)
    low_loss_bad = read_kepler(0, 6, 4)
    # The same successful site remains a repeated GOOD oracle signal at n=20.
    repeated_good = read_kepler(0, 20, 6)
    return [
        ("same_site_good_then_bad_after_box_contraction",
         [box_good, box_bad], {
             "same_site": box_good["site_id"] == box_bad["site_id"],
             "first_good": box_good["oracle_good"],
             "later_bad": not box_bad["oracle_good"],
         }),
        ("later_lower_loss_bad_cannot_overwrite",
         [low_loss_good, low_loss_bad], {
             "later_loss_is_lower": (
                 low_loss_bad["archived_loss"] < low_loss_good["archived_loss"]
             ),
             "first_good": low_loss_good["oracle_good"],
             "later_bad": not low_loss_bad["oracle_good"],
         }),
        ("repeated_success_signal_preserves_first_site",
         [low_loss_good, repeated_good], {
             "both_good": bool(
                 low_loss_good["oracle_good"] and repeated_good["oracle_good"]
             ),
         }),
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    expected_hashes = json.loads(CHECKPOINTS.read_text())
    before = checkpoint_hashes()
    if before != expected_hashes or not frozen_sources_unchanged():
        raise RuntimeError("frozen teacher/checkpoint integrity failure")

    latch_source = inspect.getsource(hybrid.ArchiveAbsorbingSuccessLatch)
    path_source = inspect.getsource(hybrid.replay_archive_absorbing_latch)
    forbidden = {
        "saved_absorbing_metadata_in_latch": latch_source.count("absorbing_success"),
        "saved_state_target_in_latch": latch_source.count("target_any_good"),
        "saved_successful_site_metadata_in_latch": latch_source.count(
            "successful_handoff_site"
        ),
        "teacher_archive_function_in_live_path": path_source.count(
            "validate_deterministic_teacher_readiness"
        ),
    }
    mismatch_keys = (
        "latch_routing_mismatches", "successful_site_identity_mismatches",
        "absorbing_state_violations", "archive_replay_mismatches",
        "metadata_latch_fallback_invocations",
    )
    total = Counter(); replay_rows = []
    categories = Counter()
    first_mismatch = None
    for path, payload, events in readiness_histories():
        student, local = compare(events, True)
        total.update(local)
        immediate = bool(events and events[0]["oracle_good"])
        failed_before = sum(not event["oracle_good"] for event in events)
        older = bool(payload.get("successful_site_newest") is False)
        nonincumbent = bool(payload.get("successful_site_nonincumbent"))
        categories["immediate_success"] += int(immediate)
        categories["success_after_failed_probes"] += int(failed_before > 0)
        categories["older_successful_site"] += int(older)
        categories["nonincumbent_successful_site"] += int(nonincumbent)
        row = {
            "system": payload["system"], "case_id": int(payload["case_id"]),
            "source": str(path), "probes": len(events),
            "failed_probes_before_success": failed_before,
            "immediate_success": immediate,
            "older_successful_site": older,
            "nonincumbent_successful_site": nonincumbent,
            "teacher_successful_site": payload["successful_handoff_site"],
            "student_successful_site": student["successful_site"],
            **{key: int(local[key]) for key in mismatch_keys},
        }
        replay_rows.append(row)
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = row

    controls = []
    for name, events, evidence in saved_post_success_controls():
        student, local = compare(events, False)
        total.update(local)
        row = {
            "name": name, "sources": [event["source"] for event in events],
            "oracle_sequence": [event["oracle_good"] for event in events],
            "site_sequence": [event["site_id"] for event in events],
            "stored_successful_site": student["successful_site"],
            "final_stop": student["stop"],
            "final_continue_n1": student["continue_n1"],
            "evidence": evidence,
            **{key: int(local[key]) for key in mismatch_keys},
        }
        controls.append(row)
        if first_mismatch is None and any(local[key] for key in mismatch_keys):
            first_mismatch = row

    after = checkpoint_hashes()
    passed = bool(
        replay_rows
        and categories["immediate_success"]
        and categories["success_after_failed_probes"]
        and categories["older_successful_site"]
        and categories["nonincumbent_successful_site"]
        and not any(forbidden.values())
        and all(total[key] == 0 for key in mismatch_keys)
        and all(all(control["evidence"].values()) for control in controls)
        and before == after == expected_hashes
        and frozen_sources_unchanged()
    )
    report = {
        "status": "INTEGRATED_EXACT" if passed else "FAILED",
        "binding": "archive_absorbing_success_binding",
        "teacher_semantics": [
            "prior success OR current valid-handoff oracle result",
            "capture current archived site only on the first false-to-true transition",
            "retain the captured site for all later GOOD or BAD oracle signals",
            "STOP whenever latched success is true; CONTINUE_N1 only while false",
        ],
        "retained_D_oracles": [
            "archive retesting schedule", "newest-first/older-site branch ordering",
            "valid-handoff/recovered-law decision",
        ],
        "binding_invocation_count": int(total["calls"]),
        "probe_updates_checked": int(total["probe_updates"]),
        "metadata_fallback_latch_calls": int(
            total["metadata_latch_fallback_invocations"]
        ),
        "latch_routing_mismatches": int(total["latch_routing_mismatches"]),
        "successful_site_identity_mismatches": int(
            total["successful_site_identity_mismatches"]
        ),
        "absorbing_state_violations": int(total["absorbing_state_violations"]),
        "archive_replay_mismatches": int(total["archive_replay_mismatches"]),
        "teacher_histories_replayed": len(replay_rows),
        "coverage": dict(categories),
        "saved_post_success_controls": controls,
        "forbidden_metadata_or_teacher_paths": forbidden,
        "first_mismatch": first_mismatch,
        "checkpoint_hashes_unchanged": before == after == expected_hashes,
        "frozen_teacher_sources_unchanged": frozen_sources_unchanged(),
        "next_unresolved_A_B_binding": "N2 best-ever strict-loss selection",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "binding_status.json").write_text(json.dumps({
        "archive_absorbing_success_binding": report["status"],
        "binding_invocation_count": report["binding_invocation_count"],
        "metadata_fallback_latch_calls": report["metadata_fallback_latch_calls"],
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("archive absorbing-success binding replay failed")


if __name__ == "__main__":
    main()
