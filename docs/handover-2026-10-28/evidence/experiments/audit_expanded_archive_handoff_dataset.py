#!/usr/bin/env python3
"""Integrity and coverage audit for expanded archive-handoff supervision."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.validate_lossless_edirect_representation import bits_equal, decode


ROOT = Path("outputs/chronological_archive_handoff_supervision_expanded")
BASE = Path("outputs/chronological_archive_handoff_supervision")
SPLITS = ("train", "held_out", "fresh_test")
SYSTEMS = ("reflection_ball", "kepler5")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    states = rows(ROOT / "new_archive_states.jsonl")
    sites = rows(ROOT / "new_archive_site_labels.jsonl")
    hard = rows(ROOT / "new_hard_first_positive_states.jsonl")
    by_state = defaultdict(list)
    for site in sites:
        by_state[site["state_id"]].append(site)

    chronology_errors = []
    binary_errors = []
    trajectory_splits = defaultdict(set)
    for state in states:
        key = (state["system"], int(state["case_id"]))
        trajectory_splits[key].add(state["split"])
        labels = sorted(by_state[state["state_id"]], key=lambda row: row["archive_index"])
        n = int(state["evaluation_count"])
        if len(labels) != n or [row["archive_index"] for row in labels] != list(range(n)):
            chronology_errors.append([state["state_id"], "site_order"])
        good = [row["archive_index"] for row in labels if row["target_good"]]
        if good != state["good_site_indices"] or bool(good) != state["target_any_good"]:
            chronology_errors.append([state["state_id"], "target_mismatch"])
        saved = np.load(state["state_npz"])
        decoded = decode(Path(state["lossless_state_binary"]).read_bytes())
        exact = (
            decoded.dimension == int(saved["dimension"])
            and decoded.iteration == int(saved["iteration"])
            and bits_equal(decoded.lower, saved["lower"])
            and bits_equal(decoded.upper, saved["upper"])
            and bits_equal(decoded.points, saved["normalized_sites"])
            and bits_equal(decoded.losses, saved["site_losses"])
            and hashlib.sha256(Path(state["lossless_state_binary"]).read_bytes()).hexdigest()
            == state["lossless_state_sha256"]
        )
        if not exact:
            binary_errors.append(state["state_id"])

    case_rows = []
    for path in sorted((ROOT / "cases").glob("*/*.json")):
        case_rows.append(json.loads(path.read_text()))
    absorbing_errors = []
    for case in case_rows:
        sequence = [state for state in states
                    if state["system"] == case["system"]
                    and int(state["case_id"]) == int(case["case_id"])]
        sequence.sort(key=lambda row: row["evaluation_count"])
        targets = [row["target_any_good"] for row in sequence]
        if (len(sequence) != case["first_any_good_evaluation"]
                or any(targets[:-1]) or not targets[-1]):
            absorbing_errors.append([case["system"], case["case_id"]])

    accepted_ball = [row for row in manifest["trajectories"]
                     if row["system"] == "reflection_ball"
                     and (ROOT / "cases" / "reflection_ball" /
                          f"case_{row['case_id']:05d}.json").exists()]
    ball_valid = all(row["identifiability"]["prefix_impacts"] >= 1
                     and row["identifiability"]["jacobian_rank"] == 5
                     for row in accepted_ball)
    before = json.loads((ROOT / "checkpoint_hashes_before.json").read_text())
    after = hashes()
    report = {
        "completed_new_trajectories": len(case_rows),
        "completed_by_system_split": {
            f"{system}:{split}": sum(row["system"] == system and row["split"] == split
                                     for row in case_rows)
            for system in SYSTEMS for split in SPLITS
        },
        "chronological_states": len(states),
        "site_at_count_labels": len(sites),
        "hard_first_positive_states": len(hard),
        "hard_by_system_split": {
            f"{system}:{split}": sum(row["system"] == system and row["split"] == split
                                     for row in hard)
            for system in SYSTEMS for split in SPLITS
        },
        "chronology_errors": chronology_errors,
        "absorbing_stop_errors": absorbing_errors,
        "lossless_binary_errors": binary_errors,
        "trajectory_split_leakage": {
            f"{system}:{case}": sorted(values) for (system, case), values in trajectory_splits.items()
            if len(values) != 1
        },
        "reflection_ball_completed": len(accepted_ball),
        "reflection_ball_all_impacting_rank5": ball_valid,
        "reflection_ball_minimum_prefix_impacts": min(
            row["identifiability"]["prefix_impacts"] for row in accepted_ball),
        "reflection_ball_rejections": dict(Counter(
            row["classification"] for row in manifest["ball_rejections"])),
        "query_cache_unique_files": len(list((ROOT / "query_cache").glob("*/*/*.json"))),
        "persisted_query_elapsed_seconds": float(sum(row["elapsed_seconds"] for row in sites)),
        "persisted_n2_evaluations": int(sum(row["n2_evaluations"] for row in sites)),
        "persisted_n3_accepted_steps": int(sum(row["n3_accepted_steps"] for row in sites)),
        "maximum_completed_n3_accepted_steps": int(max(row["n3_accepted_steps"] for row in sites)),
        "checkpoint_hashes_unchanged": before == after,
        "unresolved": json.loads((ROOT / "unresolved_cases" /
                                   "reflection_ball_case_00186.json").read_text()),
        "audit_pass_for_persisted_supervised_rows": not (
            chronology_errors or absorbing_errors or binary_errors
            or any(len(values) != 1 for values in trajectory_splits.values())
            or not ball_valid or before != after),
    }
    (ROOT / "integrity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
