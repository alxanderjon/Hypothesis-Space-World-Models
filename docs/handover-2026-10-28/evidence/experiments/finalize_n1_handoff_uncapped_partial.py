#!/usr/bin/env python3
"""Finalize the uncapped-N3 handoff relabel audit without inventing labels.

Rows whose N3 continuation has not reached a genuine numerical stop remain
explicitly unresolved and are excluded from the corrected training splits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments import validate_d5_production_pipeline as production
from experiments.sho.deterministic_full_k_framework import hashes


SOURCE = Path("outputs/n1_candidate_handoff_supervision")
OUT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
CACHE = OUT / "label_cache"
PROGRESS = OUT / "uncapped_progress"
PERSISTENT_KEPLER = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}
SPLITS = ("train", "held_out", "fresh_test")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


def distribution(values):
    values = np.asarray(list(values), np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9, method="higher")),
        "p99": float(np.quantile(values, .99, method="higher")),
        "maximum": float(values.max()),
    }


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_disagreements(rows, label_key):
    total = 0
    by_system = Counter()
    for system in ("reflection_ball", "kepler5"):
        current = [row for row in rows if row["system"] == system]
        lower, upper, _ = production._config(system)
        theta = np.asarray([row["theta"] for row in current], np.float64)
        theta = (theta - lower) / (upper - lower)
        labels = np.asarray([bool(row[label_key]) for row in current])
        for index in range(len(current)):
            distance = np.sum((theta - theta[index]) ** 2, axis=1)
            distance[index] = math.inf
            nearest = int(np.argmin(distance))
            if labels[index] != labels[nearest]:
                total += 1
                by_system[system] += 1
    return {"total": total, "by_system": dict(by_system)}


def main() -> None:
    source_path = SOURCE / "dataset.json"
    source = json.loads(source_path.read_text())
    handoff = [row for row in source if row["task"] == "handoff_promotion"]
    candidate = [row for row in source if row["task"] == "candidate_selection"]

    corrections = {}
    for path in CACHE.rglob("*.json"):
        row = json.loads(path.read_text())
        corrections[row["example_id"]] = row
    progress = {}
    for path in PROGRESS.rglob("*.json"):
        row = json.loads(path.read_text())
        progress[path.stem] = row

    audit_rows = []
    corrected_rows = []
    unresolved_rows = []
    for row in handoff:
        updated = dict(row)
        updated["old_target_good"] = bool(row["target_good"])
        updated["old_label_fields"] = {
            key: value for key, value in row.items() if key.startswith("label_")
        }
        correction = corrections.get(row["example_id"])
        if correction is not None:
            updated["corrected_label_status"] = "COMPLETED_GENUINE_N3_STOP"
            updated["corrected_target_good"] = bool(correction["corrected_target_good"])
            updated["target_good"] = bool(correction["corrected_target_good"])
            for key, value in correction.items():
                if key not in ("example_id", "system", "case_id", "state_id",
                               "entity_index", "entity_source", "split"):
                    updated[f"corrected_{key}"] = value
            corrected_rows.append(updated)
        else:
            state = progress.get(row["example_id"])
            updated["corrected_label_status"] = "UNRESOLVED_NO_GENUINE_N3_STOP"
            updated["corrected_target_good"] = None
            updated["target_good"] = None
            updated["uncapped_progress"] = state
            updated["accepted_steps_so_far"] = (
                int(row["label_n3_accepted_steps"]) +
                int((state or {}).get("additional_accepted_steps", 0)))
            unresolved_rows.append(updated)
        audit_rows.append(updated)

    atomic_jsonl(OUT / "handoff_relabel_audit_all_rows.jsonl", audit_rows)
    atomic_json(OUT / "handoff_promotion_dataset_completed.json", corrected_rows)
    atomic_jsonl(OUT / "handoff_promotion_dataset_completed.jsonl", corrected_rows)
    atomic_json(OUT / "handoff_promotion_unresolved.json", unresolved_rows)
    atomic_jsonl(OUT / "handoff_promotion_unresolved.jsonl", unresolved_rows)
    for split in SPLITS:
        atomic_jsonl(OUT / f"handoff_promotion_{split}_completed.jsonl",
                     [row for row in corrected_rows if row["split"] == split])

    # Absorbing state target over the evaluated-site examples represented in
    # the existing handoff corpus. A positive is certified immediately. A
    # negative is emitted only when every represented site completed.
    groups = defaultdict(list)
    for row in audit_rows:
        groups[row["state_id"]].append(row)
    archive = []
    for state_id, rows in sorted(groups.items()):
        successful = [row for row in rows if row["corrected_target_good"] is True]
        unresolved = [row for row in rows if row["corrected_target_good"] is None]
        if successful:
            target = True
            status = "CERTIFIED_ABSORBING_SUCCESS"
        elif unresolved:
            target = None
            status = "UNRESOLVED"
        else:
            target = False
            status = "NO_SUCCESS_AMONG_REPRESENTED_COMPLETED_SITES"
        archive.append({
            "state_id": state_id,
            "system": rows[0]["system"],
            "case_id": rows[0]["case_id"],
            "split": rows[0]["split"],
            "evaluation_count": rows[0]["evaluation_count"],
            "absorbing_any_successful_handoff": target,
            "target_status": status,
            "successful_site_indices": [int(row["entity_index"]) for row in successful],
            "successful_example_ids": [row["example_id"] for row in successful],
            "successful_site_sources": [row["entity_source"] for row in successful],
            "unresolved_example_ids": [row["example_id"] for row in unresolved],
            "represented_handoff_sites": len(rows),
            "total_evaluated_sites": int(np.load(rows[0]["state_npz"])["sites"].shape[0]),
            "target_scope": "existing handoff-promotion examples in this saved state",
            "success_is_absorbing": True,
        })
    atomic_json(OUT / "archive_state_targets_partial.json", archive)
    atomic_jsonl(OUT / "archive_state_targets_partial.jsonl", archive)

    changed = [row for row in corrected_rows
               if row["old_target_good"] != row["target_good"]]
    bad_good = [row for row in changed
                if not row["old_target_good"] and row["target_good"]]
    good_bad = [row for row in changed
                if row["old_target_good"] and not row["target_good"]]
    by_system = {}
    for system in ("reflection_ball", "kepler5"):
        complete = [row for row in corrected_rows if row["system"] == system]
        pending = [row for row in unresolved_rows if row["system"] == system]
        by_system[system] = {
            "source_total": sum(row["system"] == system for row in handoff),
            "completed": len(complete),
            "unresolved": len(pending),
            "old_good_completed": sum(row["old_target_good"] for row in complete),
            "old_bad_completed": sum(not row["old_target_good"] for row in complete),
            "corrected_good": sum(row["target_good"] for row in complete),
            "corrected_bad": sum(not row["target_good"] for row in complete),
            "bad_to_good": sum((not row["old_target_good"]) and row["target_good"]
                               for row in complete),
            "good_to_bad": sum(row["old_target_good"] and not row["target_good"]
                               for row in complete),
        }

    complete_groups = defaultdict(list)
    old_groups = defaultdict(list)
    for row in corrected_rows:
        complete_groups[row["state_id"]].append(row)
        old_groups[row["state_id"]].append(row)
    old_mixed = sum(any(r["old_target_good"] for r in rows) and
                    any(not r["old_target_good"] for r in rows)
                    for rows in old_groups.values())
    new_mixed = sum(any(r["target_good"] for r in rows) and
                    any(not r["target_good"] for r in rows)
                    for rows in complete_groups.values())

    nonincumbent_changes = 0
    for row in changed:
        losses = np.load(row["state_npz"])["site_losses"]
        nonincumbent_changes += int(row["entity_index"] != int(np.argmin(losses)))

    before_path = OUT / "checkpoint_hashes_before.json"
    before = json.loads(before_path.read_text()) if before_path.exists() else hashes()
    after = hashes()
    atomic_json(OUT / "checkpoint_hashes_after.json", after)
    atomic_json(OUT / "checkpoint_hash_status.json", {
        "unchanged": before == after, "before": before, "after": after,
    })

    steps = [row["corrected_n3_accepted_steps"] for row in corrected_rows]
    beyond = [value for value in steps if value > 20]
    report = {
        "completion_status": "INCOMPLETE_UNCAPPED_TRAJECTORIES_HAVE_NO_GENUINE_STOP_YET"
            if unresolved_rows else "COMPLETE",
        "label_definition": (
            "unchanged saved production N2 endpoint -> N3 with no semantic accepted-step cap; "
            "label only after an existing genuine non-resource stop; GOOD uses the existing "
            "final_loss < 1e-20 and finite-rollout rule"
        ),
        "source_handoff_examples": len(handoff),
        "completed_handoff_examples": len(corrected_rows),
        "unresolved_handoff_examples": len(unresolved_rows),
        "old_good_all_source": sum(row["target_good"] for row in handoff),
        "old_bad_all_source": sum(not row["target_good"] for row in handoff),
        "corrected_good_completed": sum(row["target_good"] for row in corrected_rows),
        "corrected_bad_completed": sum(not row["target_good"] for row in corrected_rows),
        "bad_to_good_completed": len(bad_good),
        "good_to_bad_completed": len(good_bad),
        "by_system": by_system,
        "changes": {
            "nonincumbent": nonincumbent_changes,
            "low_loss_old_bad_alias": sum(
                (not row["old_target_good"]) and float(row["observed_loss"]) <= 1e-4
                for row in changed),
            "persistent_kepler": sum(
                row["system"] == "kepler5" and int(row["case_id"]) in PERSISTENT_KEPLER
                for row in changed),
            "by_split": dict(Counter(row["split"] for row in changed)),
        },
        "accepted_steps_completed": {
            "all": distribution(steps),
            "beyond_20": distribution(beyond),
            "examples_beyond_20": len(beyond),
        },
        "stop_reasons_completed": dict(Counter(
            row["corrected_n3_stop_reason"] for row in corrected_rows)),
        "unresolved_progress": {
            "accepted_steps_so_far": distribution(
                row["accepted_steps_so_far"] for row in unresolved_rows),
            "rows": [{
                "example_id": row["example_id"], "system": row["system"],
                "case_id": row["case_id"], "state_id": row["state_id"],
                "entity_index": row["entity_index"],
                "old_target_good": row["old_target_good"],
                "accepted_steps_so_far": row["accepted_steps_so_far"],
                "last_loss": (row.get("uncapped_progress") or {}).get("loss"),
                "last_stop_reason": (row.get("uncapped_progress") or {}).get("last_stop_reason"),
            } for row in unresolved_rows],
        },
        "neighborhood_diagnostics_completed_rows": {
            "old_mixed_label_states": old_mixed,
            "corrected_mixed_label_states": new_mixed,
            "old_nearest_theta_label_disagreements":
                nearest_disagreements(corrected_rows, "old_target_good"),
            "corrected_nearest_theta_label_disagreements":
                nearest_disagreements(corrected_rows, "target_good"),
            "scope_note": "diagnostic only; no learned full-representation neighborhood was refit",
        },
        "archive_targets": {
            "states": len(archive),
            "certified_absorbing_success": sum(
                row["target_status"] == "CERTIFIED_ABSORBING_SUCCESS" for row in archive),
            "completed_negative": sum(
                row["target_status"] == "NO_SUCCESS_AMONG_REPRESENTED_COMPLETED_SITES"
                for row in archive),
            "unresolved": sum(row["target_status"] == "UNRESOLVED" for row in archive),
            "scope_limitation": (
                "the source promotion corpus stores canonical evaluated-site examples rather "
                "than every raw archived site; positives are certified, negatives are limited "
                "to the represented sites"
            ),
        },
        "candidate_selection": {
            "source_rows": len(candidate),
            "modified": False,
        },
        "source_dataset_sha256": source_hash(source_path),
        "checkpoint_hashes_unchanged": before == after,
    }
    atomic_json(OUT / "report_partial.json", report)
    atomic_json(OUT / "schema_partial.json", {
        "version": 1,
        "source_dataset": str(source_path),
        "corrected_task": "handoff_promotion only",
        "candidate_selection_labels": "untouched in source; not copied into relabel output",
        "n2": "saved exact unchanged production N2 output reused",
        "n3": "production N3 continued in exact 1000-accepted-step checkpoints; checkpoint boundaries never assign a label",
        "corrected_label": "assigned only after an existing genuine non-resource N3 stop",
        "unresolved_label": "null; never coerced to BAD",
        "splits": "original whole-trajectory train/held_out/fresh_test values preserved",
        "archive_target": "absorbing ANY corrected GOOD among represented site examples",
        "files": {
            "all_rows_audit": "handoff_relabel_audit_all_rows.jsonl",
            "completed_training_rows": "handoff_promotion_dataset_completed.jsonl",
            "unresolved_rows": "handoff_promotion_unresolved.jsonl",
            "archive_targets": "archive_state_targets_partial.jsonl",
        },
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
