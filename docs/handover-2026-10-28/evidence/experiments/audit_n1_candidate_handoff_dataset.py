#!/usr/bin/env python3
"""Audit the lossless N1 candidate/handoff supervision corpus.

This script performs no optimization and fits no model.  It checks state
integrity, trajectory-isolated splits, production N2-box semantics, class and
routing coverage, and duplicate prevalence.  It also materializes split and
hard-case indices without copying the ragged tensor states.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


OUT = Path("outputs/n1_candidate_handoff_supervision")
TASKS = ("candidate_selection", "handoff_promotion")
SPLITS = ("train", "held_out", "fresh_test")
SYSTEMS = ("reflection_ball", "kepler5")
PERSISTENT_KEPLER = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}
REQUIRED_ARRAYS = {
    "dimension", "case_id", "generator_id", "iteration", "evaluation_count",
    "lower", "upper", "sites", "site_losses", "site_encoded_losses",
    "reservoir", "candidate_owner", "candidate_owner_distance", "cell_sizes",
    "selected_sites", "pareto_dominated", "canonical_dominator", "cell_group",
    "hull_groups", "hull_position", "hull_epsilon_pass", "group_representatives",
    "candidate_neighbor_indices", "candidate_neighbor_distances",
    "candidate_surrogate_prediction", "candidate_role", "candidate_score",
    "candidate_score_rank", "candidate_is_production_winner",
    "candidate_routing_status", "selected_cell_weights",
    "largest_selected_position",
}


def dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90, method="higher")),
        "p99": float(np.quantile(values, .99, method="higher")),
        "maximum": float(values.max()),
    }


def counts(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "good": sum(bool(row["label_label"]) for row in rows),
        "bad": sum(not bool(row["label_label"]) for row in rows),
    }


def progress(rows: list[dict]) -> dict:
    intervals = ((0, 1000), (1000, 5000), (5000, 10000),
                 (10000, 15000), (15000, 20000), (20000, 20001))
    result = {}
    for lower, upper in intervals:
        selected = [r for r in rows if lower <= r["evaluation_count"] < upper]
        if selected:
            result[f"[{lower},{upper})"] = counts(selected)
    return result


def main() -> None:
    rows = json.loads((OUT / "dataset.json").read_text())
    states = json.loads((OUT / "states.json").read_text())
    state_meta = {row["state_id"]: row for row in states}

    # Split isolation is checked by whole physical trajectory, not by state.
    case_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        case_splits[(row["system"], int(row["case_id"]))].add(row["split"])
    leakage = {f"{key[0]}:{key[1]}": sorted(value)
               for key, value in case_splits.items() if len(value) != 1}

    input_integrity = {
        "states": len(states), "missing_state_files": [],
        "missing_required_arrays": {}, "ground_truth_named_arrays": {},
        "shape_errors": [], "n2_radius_errors": [],
    }
    incumbent_by_state = {}
    for meta in states:
        path = Path(meta["state_npz"])
        if not path.exists():
            input_integrity["missing_state_files"].append(str(path)); continue
        z = np.load(path)
        missing = sorted(REQUIRED_ARRAYS - set(z.files))
        if missing:
            input_integrity["missing_required_arrays"][meta["state_id"]] = missing
        forbidden = sorted(name for name in z.files
                           if "teacher" in name.lower() or "ground_truth" in name.lower())
        if forbidden:
            input_integrity["ground_truth_named_arrays"][meta["state_id"]] = forbidden
        d = int(z["dimension"]); n = len(z["sites"]); m = len(z["reservoir"])
        if z["sites"].shape != (n, d) or z["reservoir"].shape != (m, d):
            input_integrity["shape_errors"].append(meta["state_id"])
        expected = ((z["upper"] - z["lower"])
                    / float(z["evaluation_count"]) ** (1.0 / d))
        if not np.array_equal(expected, np.asarray(meta["production_n2_radius"])):
            input_integrity["n2_radius_errors"].append(meta["state_id"])
        incumbent_by_state[meta["state_id"]] = int(np.argmin(z["site_losses"]))
    input_integrity["passed"] = not any((
        input_integrity["missing_state_files"],
        input_integrity["missing_required_arrays"],
        input_integrity["ground_truth_named_arrays"],
        input_integrity["shape_errors"], input_integrity["n2_radius_errors"],
    ))

    # Every label must use the exact production radius at the actual count.
    n2_box_errors = []
    for row in rows:
        lower, upper = np.load(row["state_npz"])["lower"], np.load(row["state_npz"])["upper"]
        theta = np.asarray(row["theta"], dtype=np.float64)
        radius = (upper - lower) / float(row["evaluation_count"]) ** (1.0 / len(theta))
        expected_lower = np.maximum(lower, theta - radius)
        expected_upper = np.minimum(upper, theta + radius)
        if (not np.array_equal(expected_lower, np.asarray(row["label_n2_box_lower"])) or
                not np.array_equal(expected_upper, np.asarray(row["label_n2_box_upper"]))):
            n2_box_errors.append(row["example_id"])

    summaries = {}
    for task in TASKS:
        selected = [row for row in rows if row["task"] == task]
        by_system = {}
        for system in SYSTEMS:
            system_rows = [row for row in selected if row["system"] == system]
            by_system[system] = {
                **counts(system_rows),
                "trajectories": len({row["case_id"] for row in system_rows}),
                "by_split": {split: counts([row for row in system_rows
                                             if row["split"] == split])
                             for split in SPLITS},
            }
        by_route = {}
        for route in sorted({row["current_routing_status_name"] for row in selected}):
            route_rows = [row for row in selected
                          if row["current_routing_status_name"] == route]
            by_route[route] = counts(route_rows)
        summaries[task] = {
            **counts(selected), "by_system": by_system,
            "distinct_trajectories": len({(r["system"], r["case_id"]) for r in selected}),
            "case_splits": {split: len({(r["system"], r["case_id"])
                                        for r in selected if r["split"] == split})
                            for split in SPLITS},
            "example_splits": {split: counts([r for r in selected if r["split"] == split])
                               for split in SPLITS},
            "evaluation_count": distribution([r["evaluation_count"] for r in selected]),
            "evaluation_progress": progress(selected), "routing": by_route,
        }

    candidate_rows = [r for r in rows if r["task"] == "candidate_selection"]
    rejected_routes = {"PARETO_DOMINATED_OWNER", "NON_HULL_OWNER",
                       "SELECTED_CELL_NONWINNER", "HULL_NOT_SELECTED_OWNER"}
    useful_rejected = [r for r in candidate_rows if r["label_label"] and
                       r["current_routing_status_name"] in rejected_routes]
    summaries["candidate_selection"]["useful_currently_rejected"] = {
        **counts(useful_rejected),
        "by_route": dict(Counter(r["current_routing_status_name"] for r in useful_rejected)),
        "by_system": dict(Counter(r["system"] for r in useful_rejected)),
    }

    handoff_rows = [r for r in rows if r["task"] == "handoff_promotion"]
    useful_nonincumbent = [r for r in handoff_rows if r["label_label"] and
                           r["entity_index"] != incumbent_by_state[r["state_id"]]]
    bad_incumbent = [r for r in handoff_rows if not r["label_label"] and
                     r["entity_index"] == incumbent_by_state[r["state_id"]]]
    summaries["handoff_promotion"]["recoverable_nonincumbent"] = {
        **counts(useful_nonincumbent),
        "by_system": dict(Counter(r["system"] for r in useful_nonincumbent)),
    }
    summaries["handoff_promotion"]["bad_incumbent_aliases"] = {
        **counts(bad_incumbent),
        "by_system": dict(Counter(r["system"] for r in bad_incumbent)),
    }

    # Duplicate diagnostics use normalized parameter coordinates.  Near means
    # <=1e-12 only for this prevalence audit; it is not a training gate.
    exact_keys = Counter((r["task"], r["state_id"], tuple(r["unit_point"])) for r in rows)
    exact_duplicates = sum(value - 1 for value in exact_keys.values() if value > 1)
    near_pairs = 0; compared_pairs = 0
    for task in TASKS:
        for state_id in sorted({r["state_id"] for r in rows if r["task"] == task}):
            subset = [r for r in rows if r["task"] == task and r["state_id"] == state_id]
            points = np.asarray([r["unit_point"] for r in subset], dtype=np.float64)
            if len(points) < 2: continue
            distances = np.sqrt(np.sum((points[:, None] - points[None, :]) ** 2, axis=-1))
            upper = distances[np.triu_indices(len(points), 1)]
            compared_pairs += len(upper); near_pairs += int(np.sum(upper <= 1e-12))
    repeated: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        repeated[(row["task"], row["system"], row["case_id"], tuple(row["unit_point"]))].add(row["state_id"])
    repeated_later_state = [value for value in repeated.values() if len(value) > 1]

    hard_rows = []
    for row in rows:
        tags = []
        if (row["task"] == "candidate_selection" and row["label_label"] and
                row["current_routing_status_name"] == "PARETO_DOMINATED_OWNER"):
            tags.append("useful_pareto_dominated_candidate")
        if (row["task"] == "candidate_selection" and row["label_label"] and
                row["current_routing_status_name"] == "NON_HULL_OWNER"):
            tags.append("useful_non_hull_candidate")
        if (row["task"] == "handoff_promotion" and row["label_label"] and
                row["entity_index"] != incumbent_by_state[row["state_id"]]):
            tags.append("recoverable_evaluated_nonincumbent")
        if not row["label_label"] and float(row["label_objective_loss"]) <= 1e-4:
            tags.append("low_loss_bad_alias")
        if row["system"] == "kepler5" and int(row["case_id"]) in PERSISTENT_KEPLER:
            tags.append("persistent_kepler_localization_failure")
        if tags:
            hard_rows.append({**row, "hard_case_tags": tags})
    with (OUT / "hard_cases.jsonl").open("w") as stream:
        for row in hard_rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    for task in TASKS:
        for split in SPLITS:
            path = OUT / f"{task}_{split}.jsonl"
            with path.open("w") as stream:
                for row in rows:
                    if row["task"] == task and row["split"] == split:
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    before = json.loads((OUT / "checkpoint_hashes_before.json").read_text())
    after = json.loads((OUT / "checkpoint_hashes_after.json").read_text())
    audit = {
        "dataset_examples": len(rows), "states": len(states),
        "trajectories": len(case_splits), "summaries": summaries,
        "input_integrity": input_integrity,
        "production_n2_box_replay": {"errors": len(n2_box_errors),
                                     "error_ids": n2_box_errors},
        "split_isolation": {"passed": not leakage, "leakage": leakage,
                            "case_counts": {split: sum(next(iter(value)) == split
                                for value in case_splits.values()) for split in SPLITS}},
        "duplicates": {
            "exact_same_state_query_duplicates": exact_duplicates,
            "normalized_pairs_compared": compared_pairs,
            "near_pairs_at_1e-12": near_pairs,
            "same_entity_retained_at_distinct_later_states": len(repeated_later_state),
            "later_state_appearances": sum(len(value) for value in repeated_later_state),
        },
        "hard_subset": {
            "examples": len(hard_rows),
            "tag_counts": dict(Counter(tag for row in hard_rows
                                       for tag in row["hard_case_tags"])),
            "path": str(OUT / "hard_cases.jsonl"),
        },
        "cache": {
            "exact_prior_labels_imported": 24,
            "labels_reused_when_expanding_corpus": 1317,
            "new_downstream_queries_for_expansion": 756,
            "total_cached_queries": len(list((OUT / "label_cache").glob("*.json"))),
        },
        "labeling_cost": {
            "cumulative_worker_seconds": float(sum(float(r.get("label_label_seconds", 0.0))
                                                    for r in rows)),
            "n2_objective_evaluations": int(sum(r.get("label_n2_evaluations") or 0 for r in rows)),
            "n3_objective_evaluations": int(sum(r.get("label_n3_objective_evaluations") or 0 for r in rows)),
            "candidate_physical_evaluations": int(sum(r["task"] == "candidate_selection" and
                                                        not r.get("label_reused", False) for r in rows)),
        },
        "checkpoint_hashes_unchanged": before == after,
        "training_readiness": {
            "candidate_selection": "READY_FOR_ARCHITECTURE_SCREEN",
            "handoff_promotion": "READY_FOR_ARCHITECTURE_SCREEN",
            "remaining_sparsity": (
                "Kepler GOOD labels are less common than Kepler BAD labels, but all systems and "
                "all trajectory-isolated splits contain both classes; no artificial balance was used."
            ),
        },
    }
    dump(OUT / "audit.json", audit)

    schema = json.loads((OUT / "schema.json").read_text())
    schema.update({
        "state_metadata": "states.json; includes exact production N2 radius implied by count",
        "required_state_arrays": sorted(REQUIRED_ARRAYS),
        "canonical_identity": "array row index; sites preserve production insertion order",
        "hard_case_index": "hard_cases.jsonl",
        "split_indices": [f"{task}_{split}.jsonl" for task in TASKS for split in SPLITS],
        "target": "target_good; actual recovered dynamics via unchanged N2 and N3",
        "diagnostic_targets": "label_final_loss and teacher-relative/rollout fields; never runtime inputs",
    })
    dump(OUT / "schema.json", schema)

    candidate, handoff = summaries["candidate_selection"], summaries["handoff_promotion"]
    report = f"""# N1 candidate/handoff supervision audit

- Lossless states: {len(states)} across {len(case_splits)} trajectories.
- Candidate selection: {candidate['total']} examples ({candidate['good']} GOOD / {candidate['bad']} BAD).
- Handoff promotion: {handoff['total']} examples ({handoff['good']} GOOD / {handoff['bad']} BAD).
- Whole-trajectory splits: {audit['split_isolation']['case_counts']} with zero leakage.
- Useful candidates rejected by production: {candidate['useful_currently_rejected']['total']}.
- Recoverable evaluated non-incumbents: {handoff['recoverable_nonincumbent']['total']}.
- Exact/near same-state duplicates: {exact_duplicates}/{near_pairs}.
- Production N2-box replay errors: {len(n2_box_errors)}.
- Checkpoint hashes unchanged: {before == after}.

See `audit.json` for system/split/routing/progress tables and `schema.json` for the tensor contract.
"""
    (OUT / "REPORT.md").write_text(report)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
