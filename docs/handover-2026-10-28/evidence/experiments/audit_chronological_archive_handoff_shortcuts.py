"""Audit bookkeeping-only shortcuts in chronological archive handoff data."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("outputs/chronological_archive_handoff_supervision")
SPLITS = ("train", "held_out", "fresh_test")


def confusion(rows: list[dict], predict) -> dict:
    counts = Counter()
    for row in rows:
        target = bool(row["target_any_good"])
        prediction = bool(predict(row))
        key = "tp" if prediction and target else "fp" if prediction else "fn" if target else "tn"
        counts[key] += 1
    total = len(rows)
    return {
        "count": total,
        "accuracy": (counts["tp"] + counts["tn"]) / total,
        **{key: counts[key] for key in ("tp", "fp", "tn", "fn")},
    }


def split_confusions(rows: list[dict], predict) -> dict:
    return {
        split: confusion([row for row in rows if row["split"] == split], predict)
        for split in SPLITS
    } | {"all": confusion(rows, predict)}


def majority_lookup(rows: list[dict], fields: tuple[str, ...]):
    votes: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        votes[tuple(row[field] for field in fields)][bool(row["target_any_good"])] += 1
    lookup = {key: count[True] > count[False] for key, count in votes.items()}
    global_default = sum(row["target_any_good"] for row in rows) > len(rows) / 2
    return lambda row: lookup.get(tuple(row[field] for field in fields), global_default), {
        str(key): value for key, value in lookup.items()
    }


def site_policy(states: list[dict], site_by_state: dict[str, dict[int, dict]], policy: str) -> dict:
    results = {}
    for system in ("reflection_ball", "kepler5", "all"):
        results[system] = {}
        for split in (*SPLITS, "all"):
            rows = [row for row in states
                    if row["target_any_good"]
                    and (system == "all" or row["system"] == system)
                    and (split == "all" or row["split"] == split)]
            successes = 0
            for row in rows:
                index = ({
                    "newest": row["site_count"] - 1,
                    "incumbent": row["incumbent_site_index"],
                    "oldest": 0,
                })[policy]
                successes += bool(site_by_state[row["state_id"]][index]["target_good"])
            results[system][split] = {
                "successes": successes,
                "failures": len(rows) - successes,
                "total": len(rows),
                "accuracy": successes / len(rows) if rows else None,
            }
    return results


def main() -> None:
    states = [json.loads(line) for line in (ROOT / "archive_states.jsonl").read_text().splitlines()]
    sites = [json.loads(line) for line in (ROOT / "archive_site_labels.jsonl").read_text().splitlines()]
    train = [row for row in states if row["split"] == "train"]
    count_lookup, count_map = majority_lookup(train, ("evaluation_count",))
    system_count_lookup, system_count_map = majority_lookup(train, ("system", "evaluation_count"))

    site_by_state: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in sites:
        site_by_state[row["state_id"]][int(row["archive_index"])] = row
    positive = [row for row in states if row["target_any_good"]]
    hard = []
    for row in positive:
        newest = row["site_count"] - 1
        incumbent = row["incumbent_site_index"]
        newest_good = bool(site_by_state[row["state_id"]][newest]["target_good"])
        incumbent_good = bool(site_by_state[row["state_id"]][incumbent]["target_good"])
        if not newest_good or not incumbent_good:
            hard.append({
                "system": row["system"], "case_id": row["case_id"],
                "split": row["split"], "evaluation_count": row["evaluation_count"],
                "good_site_indices": row["good_site_indices"],
                "newest_site_index": newest, "newest_is_good": newest_good,
                "incumbent_site_index": incumbent, "incumbent_is_good": incumbent_good,
                "requires_older_site": not newest_good,
                "requires_nonincumbent_site": not incumbent_good,
            })

    hard_composition = Counter((row["system"], row["split"]) for row in hard)
    report = {
        "dataset": {"archive_states": len(states), "positive_states": len(positive),
                    "site_labels": len(sites)},
        "state_gate_baselines": {
            "always_continue": split_confusions(states, lambda _: False),
            "always_handoff": split_confusions(states, lambda _: True),
            "evaluation_count_ge_3": split_confusions(
                states, lambda row: row["evaluation_count"] >= 3),
            "evaluation_count_ge_3_by_system": {
                system: split_confusions(
                    [row for row in states if row["system"] == system],
                    lambda row: row["evaluation_count"] >= 3)
                for system in ("reflection_ball", "kepler5")
            },
            "train_majority_by_evaluation_count": {
                "lookup": count_map,
                "metrics": split_confusions(states, count_lookup),
            },
            "train_majority_by_system_and_evaluation_count_diagnostic_only": {
                "lookup": system_count_map,
                "metrics": split_confusions(states, system_count_lookup),
            },
            "site_count": "exactly equal to evaluation_count in every state; no independent signal",
            "dimension": "constant d=5; no signal",
            "canonical_ordering": "constant; no signal",
            "kepler_eval_1_2_continue_eval_3_handoff": split_confusions(
                [row for row in states if row["system"] == "kepler5"],
                lambda row: row["evaluation_count"] == 3),
            "posthoc_last_saved_state": (
                "would be 100%, but is forbidden future/serialization leakage and not runtime bookkeeping"),
        },
        "site_selection_baselines": {
            "choose_newest": site_policy(positive, site_by_state, "newest"),
            "choose_lowest_loss_incumbent": site_policy(positive, site_by_state, "incumbent"),
            "choose_lowest_loss_site": "identical to incumbent by dataset construction",
            "choose_oldest": site_policy(positive, site_by_state, "oldest"),
        },
        "hard_subset": {
            "definition": "newest site is BAD and/or incumbent site is BAD",
            "count": len(hard),
            "composition": {f"{system}:{split}": count
                            for (system, split), count in sorted(hard_composition.items())},
            "requires_older_site_count": sum(row["requires_older_site"] for row in hard),
            "requires_nonincumbent_site_count": sum(
                row["requires_nonincumbent_site"] for row in hard),
            "rows": hard,
        },
        "minimum_baselines_to_beat": {
            "state_gate": "evaluation_count >= 3",
            "site_selector": "choose newest archived site",
            "required_nontrivial_evidence": (
                "performance on the six hard states, especially the five older-site ball states; "
                "Kepler gate accuracy alone is not evidence because count solves it perfectly"),
        },
    }
    path = ROOT / "shortcut_baseline_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
