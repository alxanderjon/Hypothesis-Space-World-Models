#!/usr/bin/env python3
"""Measure direct N2->N3 archive probing cost from finalized supervision."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.train_archive_handoff_screen import ArchiveSiteSelector, ArchiveState


DATA = Path("outputs/chronological_archive_handoff_supervision_expanded")
OUT = Path("outputs/archive_direct_n2_probe_cost")
POLICIES = ("newest_first", "incumbent_first", "canonical", "learned_selector")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def distribution(values) -> dict:
    values = np.asarray(list(values), np.float64)
    return {
        "count": int(len(values)), "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9, method="higher")),
        "maximum": float(values.max()), "total": float(values.sum()),
    }


def model_scores(states: dict[str, ArchiveState]) -> tuple[dict[str, list[float]], str]:
    path = Path("outputs/archive_handoff_relational_screen/archive_site_selector_screen_only.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ArchiveSiteSelector(); model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad():
        scores = {state_id: [float(value) for value in model(state)]
                  for state_id, state in states.items()}
    return scores, sha(path)


def order_for(policy: str, meta: dict, scores: dict[str, list[float]]) -> list[int]:
    n = int(meta["site_count"]); canonical = list(range(n))
    if policy == "canonical":
        return canonical
    if policy == "newest_first":
        return [n - 1] + list(range(n - 1))
    if policy == "incumbent_first":
        incumbent = int(meta["incumbent_site_index"])
        return [incumbent] + [index for index in canonical if index != incumbent]
    values = np.asarray(scores[meta["state_id"]], np.float64)
    return [int(index) for index in np.lexsort((np.arange(n), -values))]


def simulate(policy: str, chronological: list[dict], labels_by_state: dict[str, list[dict]],
             scores: dict[str, list[float]]) -> dict:
    branches = []; seen_sites: set[int] = set(); success = None
    for meta in chronological:
        rows = {int(row["archive_index"]): row for row in labels_by_state[meta["state_id"]]}
        for index in order_for(policy, meta, scores):
            row = rows[index]
            repeated = index in seen_sites
            branches.append({
                "state_id": meta["state_id"],
                "evaluation_count": int(meta["evaluation_count"]),
                "archive_index": index,
                "insertion_evaluation": int(row["insertion_evaluation"]),
                "retest_of_older_site": bool(repeated),
                "target_good": bool(row["target_good"]),
                "n2_evaluations": int(row["n2_evaluations"]),
                "n3_accepted_steps": int(row["n3_accepted_steps"]),
                "elapsed_seconds_saved_query": float(row["elapsed_seconds"]),
            })
            seen_sites.add(index)
            if row["target_good"]:
                success = (meta, row)
                break
        if success is not None:
            break
    if success is None:
        raise RuntimeError((policy, chronological[-1]["state_id"]))
    meta, row = success
    index = int(row["archive_index"]); n = int(meta["evaluation_count"])
    return {
        "policy": policy,
        "first_successful_n1_evaluation": n,
        "archive_sites_at_success": int(meta["site_count"]),
        "branches_tested": len(branches),
        "repeated_older_site_tests": sum(branch["retest_of_older_site"] for branch in branches),
        "successful_archive_index": index,
        "successful_site_insertion_evaluation": int(row["insertion_evaluation"]),
        "successful_site_newest": bool(index == n - 1),
        "successful_site_older": bool(index < n - 1),
        "successful_site_incumbent": bool(index == int(meta["incumbent_site_index"])),
        "successful_site_nonincumbent": bool(index != int(meta["incumbent_site_index"])),
        "total_n2_evaluations": sum(branch["n2_evaluations"] for branch in branches),
        "total_n3_accepted_steps": sum(branch["n3_accepted_steps"] for branch in branches),
        "total_saved_query_seconds": sum(branch["elapsed_seconds_saved_query"] for branch in branches),
        "branches": branches,
    }


def fractions(values) -> dict:
    total = len(values)
    return {
        "one_branch": sum(value == 1 for value in values) / total,
        "two_branches": sum(value == 2 for value in values) / total,
        "three_branches": sum(value == 3 for value in values) / total,
        "four_or_more_branches": sum(value >= 4 for value in values) / total,
        "counts": {
            "one_branch": sum(value == 1 for value in values),
            "two_branches": sum(value == 2 for value in values),
            "three_branches": sum(value == 3 for value in values),
            "four_or_more_branches": sum(value >= 4 for value in values),
        },
    }


def summarize(results: list[dict], trajectories: list[dict]) -> dict:
    output = {}
    for system in ("reflection_ball", "kepler5"):
        system_rows = [row for row in results if row["system"] == system]
        system_trajectories = [row for row in trajectories if row["system"] == system]
        first_positive = {row["trajectory_key"]: row for row in system_trajectories}
        system_result = {
            "trajectories": len(system_trajectories),
            "strict_older_retest_necessary": sum(not row["newest_good_at_first_positive"]
                                                  for row in system_trajectories),
            "newest_first_immediate_at_positive_state": sum(row["newest_good_at_first_positive"]
                                                              for row in system_trajectories),
            "incumbent_first_immediate_at_positive_state": sum(row["incumbent_good_at_first_positive"]
                                                                 for row in system_trajectories),
            "policies": {},
        }
        for policy in POLICIES:
            selected = [row for row in system_rows if row["policy"] == policy]
            branches = [row["branches_tested"] for row in selected]
            system_result["policies"][policy] = {
                "recovered": sum(row["recovered"] for row in selected),
                "branches_before_success": distribution(branches),
                "n2_objective_evaluations": distribution(
                    row["total_n2_evaluations"] for row in selected),
                "n3_accepted_steps": distribution(
                    row["total_n3_accepted_steps"] for row in selected),
                "saved_serial_query_seconds": distribution(
                    row["total_saved_query_seconds"] for row in selected),
                "branch_count_fractions": fractions(branches),
                "successful_site_older": sum(row["successful_site_older"] for row in selected),
                "successful_site_nonincumbent": sum(row["successful_site_nonincumbent"] for row in selected),
                "repeated_older_site_tests": distribution(
                    row["repeated_older_site_tests"] for row in selected),
            }
        output[system] = system_result
    return output


def main() -> None:
    state_path = DATA / "expanded_archive_states.jsonl"
    label_path = DATA / "expanded_archive_site_labels.jsonl"
    states_meta = load_jsonl(state_path); site_labels = load_jsonl(label_path)
    labels_by_state = defaultdict(list)
    for row in site_labels: labels_by_state[row["state_id"]].append(row)
    archive_states = {meta["state_id"]: ArchiveState(meta, labels_by_state[meta["state_id"]])
                      for meta in states_meta}
    scores, selector_hash = model_scores(archive_states)

    by_trajectory = defaultdict(list)
    for meta in states_meta:
        by_trajectory[(meta["system"], int(meta["case_id"]))].append(meta)
    results = []; trajectory_rows = []
    for (system, case_id), chronological in sorted(by_trajectory.items()):
        chronological.sort(key=lambda row: int(row["evaluation_count"]))
        positive = chronological[-1]
        if not positive["target_any_good"] or any(row["target_any_good"] for row in chronological[:-1]):
            raise RuntimeError((system, case_id, "non-absorbing chronology"))
        final_labels = labels_by_state[positive["state_id"]]
        good = {int(row["archive_index"]) for row in final_labels if row["target_good"]}
        newest = int(positive["site_count"]) - 1
        incumbent = int(positive["incumbent_site_index"])
        trajectory_key = f"{system}:{case_id}"
        trajectory_rows.append({
            "trajectory_key": trajectory_key, "system": system, "case_id": case_id,
            "split": positive["split"], "first_successful_n1_evaluation": int(positive["evaluation_count"]),
            "good_site_indices": sorted(good),
            "newest_good_at_first_positive": newest in good,
            "incumbent_good_at_first_positive": incumbent in good,
        })
        for policy in POLICIES:
            value = simulate(policy, chronological, labels_by_state, scores)
            value.update({"trajectory_key": trajectory_key, "system": system,
                          "case_id": case_id, "split": positive["split"],
                          "recovered": True})
            results.append(value)

    report = {
        "status": "READ_ONLY_COST_AUDIT_NO_TRAINING",
        "dataset": {"trajectories": len(by_trajectory), "archive_states": len(states_meta),
                    "site_at_count_labels": len(site_labels)},
        "branch_semantics": (
            "Every (archived site, current evaluation count) query is charged once. "
            "A site is retested when the count-derived box changes. Testing stops at "
            "the first GOOD query and success is absorbing."),
        "policy_remaining_order": {
            "newest_first": "newest, then canonical archive order",
            "incumbent_first": "incumbent, then canonical archive order",
            "canonical": "archive insertion order",
            "learned_selector": "frozen selector descending score, archive index tie-break",
        },
        "summary": summarize(results, trajectory_rows),
        "source_hashes": {str(state_path): sha(state_path), str(label_path): sha(label_path),
                          "selector_checkpoint": selector_hash},
        "all_policies_preserve_recovery": all(row["recovered"] for row in results),
        "artifacts": {"per_trajectory_policy": str(OUT / "per_trajectory_policy.jsonl"),
                      "trajectory_first_positive": str(OUT / "trajectory_first_positive.jsonl"),
                      "compact_table": str(OUT / "compact_policy_table.csv")},
    }
    compact = [{key: value for key, value in row.items() if key != "branches"} for row in results]
    atomic_jsonl(OUT / "per_trajectory_policy.jsonl", results)
    atomic_jsonl(OUT / "trajectory_first_positive.jsonl", trajectory_rows)
    atomic_csv(OUT / "compact_policy_table.csv", compact)
    atomic_json(OUT / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
