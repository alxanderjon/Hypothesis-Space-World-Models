#!/usr/bin/env python3
"""State-balanced hardest-BAD ranking screen for the N1 candidate ranker.

This is an offline architecture/supervision screen.  It does not run N1 and
does not modify production checkpoints.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch
from torch import nn

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import (
    DATA, SEED, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_candidate_state_relative_screen import native
from experiments.train_n1_relational_architecture_screen import atomic_json, parameters


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_top_of_list_screen")
PREVIOUS_BINARY = Path("outputs/n1_candidate_statewise_ranking_audit/report.json")
PREVIOUS_RELATIVE = Path("outputs/n1_candidate_state_relative_screen/report.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hardest_bad_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
    good = target.bool()
    if not bool(good.any()) or not bool((~good).any()):
        return None
    best_good = logits[good].max()
    best_bad = logits[~good].max()
    return nn.functional.softplus(best_bad - best_good)


@torch.no_grad()
def mean_loss(model, states, grouped, state_ids):
    model.eval(); values = []
    for state_id in state_ids:
        examples = grouped[state_id]
        target = torch.tensor([bool(row["target_good"]) for row in examples])
        loss = hardest_bad_loss(model(states[state_id], examples), target)
        if loss is not None:
            values.append(float(loss))
    return float(np.mean(values)) if values else None


def train(model, states, grouped, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    eligible = {}
    for state_id, examples in grouped.items():
        target = np.asarray([bool(row["target_good"]) for row in examples])
        if target.any() and (~target).any():
            eligible[state_id] = examples
    train_ids = sorted(s for s, v in eligible.items() if v[0]["split"] == "train")
    held_ids = sorted(s for s, v in eligible.items() if v[0]["split"] == "held_out")
    history = []; best = None; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); random.Random(SEED + epoch).shuffle(train_ids)
        optimizer.zero_grad(); accumulation = 0; losses = []
        for state_id in train_ids:
            examples = grouped[state_id]
            target = torch.tensor([bool(row["target_good"]) for row in examples])
            loss = hardest_bad_loss(model(states[state_id], examples), target)
            if loss is None:
                continue
            (loss / 4.0).backward(); losses.append(float(loss.detach())); accumulation += 1
            if accumulation == 4:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(); accumulation = 0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad()
        train_value = mean_loss(model, states, grouped, train_ids)
        held_value = mean_loss(model, states, grouped, held_ids)
        record = {"epoch": epoch, "train_hardest_bad_loss": train_value,
                  "held_hardest_bad_loss": held_value}
        history.append(record); print(json.dumps(record), flush=True)
        if best is None or held_value < best[0]:
            best = (held_value, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    return {"history": history, "best_epoch": best[2], "seconds": time.monotonic() - started,
            "eligible_train_states": len(train_ids), "eligible_held_states": len(held_ids)}


@torch.no_grad()
def score_states(model, states, grouped):
    model.eval(); output = {}
    for state_id, examples in grouped.items():
        output[state_id] = [float(v) for v in model(states[state_id], examples).cpu().numpy()]
    return output


def state_records(grouped, scores):
    records = []
    for state_id, examples in grouped.items():
        logits = np.asarray(scores[state_id], np.float64)
        good = np.asarray([bool(row["target_good"]) for row in examples])
        order = np.lexsort((np.arange(len(examples)), -logits))
        ranks = np.empty(len(examples), np.int64); ranks[order] = np.arange(1, len(examples) + 1)
        record = {"state_id": state_id, "system": examples[0]["system"],
                  "case_id": examples[0]["case_id"], "split": examples[0]["split"],
                  "evaluation_count": examples[0]["evaluation_count"],
                  "candidate_count": len(examples), "good_count": int(good.sum()),
                  "any_good": bool(good.any())}
        if good.any() and (~good).any():
            gi, bi = np.flatnonzero(good), np.flatnonzero(~good)
            margin = float(logits[gi].max() - logits[bi].max())
            record.update(best_good_rank=int(ranks[gi].min()),
                          best_good_minus_best_bad=margin,
                          best_good_score=float(logits[gi].max()),
                          best_bad_score=float(logits[bi].max()),
                          top_is_good=bool(good[order[0]]),
                          top_candidate_example_id=examples[int(order[0])]["example_id"],
                          pairwise_wins=int(sum(logits[g] > logits[b] for g in gi for b in bi)),
                          pairwise_pairs=int(len(gi) * len(bi)))
        records.append(record)
    return records


def ranking_metrics(records):
    positive = [r for r in records if r.get("best_good_rank") is not None]
    ranks = [r["best_good_rank"] for r in positive]
    margins = np.asarray([r["best_good_minus_best_bad"] for r in positive], np.float64)
    wins = sum(r["pairwise_wins"] for r in positive)
    pairs = sum(r["pairwise_pairs"] for r in positive)
    return {"positive_states": len(positive), "top_good_states": int(sum(r == 1 for r in ranks)),
            "top_good_rate": float(sum(r == 1 for r in ranks) / max(1, len(ranks))),
            "best_good_rank_histogram": dict(sorted(Counter(ranks).items())),
            "best_good_rank_median": float(np.median(ranks)) if ranks else None,
            "best_good_rank_p90": float(np.quantile(ranks, .9)) if ranks else None,
            "margin_min": float(margins.min()) if len(margins) else None,
            "margin_p10": float(np.quantile(margins, .1)) if len(margins) else None,
            "margin_median": float(np.median(margins)) if len(margins) else None,
            "margin_mean": float(margins.mean()) if len(margins) else None,
            "margin_max": float(margins.max()) if len(margins) else None,
            "pairwise_good_above_bad": int(wins), "pairwise_pairs": int(pairs),
            "pairwise_accuracy": float(wins / max(1, pairs))}


def hard_good_metrics(grouped, scores, hard_tags, tag):
    examples_count = above_all_bad = wins = pairs = 0
    for state_id, examples in grouped.items():
        logits = np.asarray(scores[state_id]); good = np.asarray([bool(r["target_good"]) for r in examples])
        if not good.any() or not (~good).any():
            continue
        bad = np.flatnonzero(~good)
        for index, row in enumerate(examples):
            if tag not in hard_tags.get(row["example_id"], ()):
                continue
            examples_count += 1
            above_all_bad += int(logits[index] > logits[bad].max())
            wins += int(np.sum(logits[index] > logits[bad])); pairs += len(bad)
    return {"examples": examples_count, "outranks_every_bad": above_all_bad,
            "outranks_every_bad_rate": float(above_all_bad / max(1, examples_count)),
            "pairwise_wins": wins, "pairwise_pairs": pairs,
            "pairwise_accuracy": float(wins / max(1, pairs))}


def alias_metrics(grouped, scores, hard_tags):
    examples_count = positive_state_examples = alias_wins = alias_above_best_good = 0
    no_good_aliases = 0
    for state_id, examples in grouped.items():
        logits = np.asarray(scores[state_id]); good = np.asarray([bool(r["target_good"]) for r in examples])
        top = int(np.argmax(logits))
        for index, row in enumerate(examples):
            if "low_loss_bad_alias" not in hard_tags.get(row["example_id"], ()):
                continue
            examples_count += 1
            if good.any():
                positive_state_examples += 1
                alias_wins += int(index == top)
                alias_above_best_good += int(logits[index] > logits[good].max())
            else:
                no_good_aliases += 1
    return {"examples": examples_count, "positive_state_examples": positive_state_examples,
            "wins_top_choice_in_positive_state": alias_wins,
            "outranks_best_good": alias_above_best_good,
            "no_good_state_examples_excluded_from_ranking": no_good_aliases}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    before = hashes(); paths = [DATA / "dataset.json", DATA / "states.json", DATA / "hard_cases.jsonl"]
    data_before = {p.name: sha256(p) for p in paths}
    rows = [r for r in json.loads((DATA / "dataset.json").read_text()) if r["task"] == "candidate_selection"]
    grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata = {r["state_id"]: r for r in json.loads((DATA / "states.json").read_text())}
    states = {s: GraphState(metadata[s], e) for s, e in grouped.items()}
    hard_tags = defaultdict(set)
    with (DATA / "hard_cases.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); hard_tags[row["example_id"]].update(row["hard_case_tags"])

    torch.manual_seed(SEED + 320); np.random.seed(SEED); random.seed(SEED)
    model = DeepCandidateModel(32, 3); fit = train(model, states, grouped, args.epochs)
    scores = score_states(model, states, grouped); records = state_records(grouped, scores)
    evaluation = {}
    for split in ("train", "held_out", "fresh_test", "all"):
        selected = records if split == "all" else [r for r in records if r["split"] == split]
        evaluation[split] = {"overall": ranking_metrics(selected),
                             "by_system": {system: ranking_metrics([r for r in selected if r["system"] == system])
                                           for system in ("reflection_ball", "kepler5")}}
    persistent = [r for r in records if r["system"] == "kepler5" and int(r["case_id"]) in PERSISTENT]
    persistent_metrics = ranking_metrics(persistent)
    persistent_failures = [{k: r[k] for k in ("state_id", "case_id", "split", "evaluation_count",
                                               "best_good_rank", "best_good_minus_best_bad",
                                               "best_good_score", "best_bad_score", "candidate_count", "good_count")}
                           for r in persistent if r.get("best_good_rank", 1) != 1]
    hard = {"useful_pareto_dominated_candidate": hard_good_metrics(grouped, scores, hard_tags, "useful_pareto_dominated_candidate"),
            "useful_non_hull_candidate": hard_good_metrics(grouped, scores, hard_tags, "useful_non_hull_candidate"),
            "low_loss_bad_alias": alias_metrics(grouped, scores, hard_tags)}

    checkpoint = OUT / "candidate_selection_hardest_bad_screen_only.pt"
    torch.save({"status": "ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED", "task": "candidate_selection",
                "supervision": "STATE_BALANCED_HARDEST_BAD_ZERO_MARGIN", "width": 32, "depth": 3,
                "parameters": parameters(model), "best_epoch": fit["best_epoch"],
                "model_state": model.state_dict(), "schema_version": 1}, checkpoint)
    after = hashes(); changed = [p for p in set(before) & set(after)
                                 if before[p] != after[p] and not p.startswith(str(OUT))]
    prior_binary = json.loads(PREVIOUS_BINARY.read_text())["models"]["depth_3"]
    prior_relative_report = json.loads(PREVIOUS_RELATIVE.read_text())
    report = native({"status": "TOP_OF_LIST_SCREEN_ONLY_NOT_PROMOTED",
        "architecture": {"width": 32, "depth": 3, "parameters": parameters(model),
                         "global_pooling": False, "state_gate": False},
        "objective": "one softplus(max_BAD_score - max_GOOD_score) contribution per eligible state",
        "fit": fit, "evaluation": evaluation,
        "persistent_kepler": {"metrics": persistent_metrics, "remaining_bad_wins": persistent_failures},
        "hard_subsets": hard,
        "previous_depth3_binary": prior_binary,
        "previous_state_relative": {"overall": prior_relative_report["evaluation"]["all"]["ranking"],
                                    "held_out": prior_relative_report["evaluation"]["held_out"]["ranking"],
                                    "fresh_test": prior_relative_report["evaluation"]["fresh_test"]["ranking"],
                                    "persistent_kepler": prior_relative_report["persistent_kepler"]["ranking"],
                                    "hard_subsets": prior_relative_report["hard_subsets"]},
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "dataset_hashes_before": data_before, "dataset_hashes_after": {p.name: sha256(p) for p in paths},
        "dataset_unchanged": data_before == {p.name: sha256(p) for p in paths},
        "protected_checkpoint_changes": changed, "checkpoint_hashes_unchanged": not changed})
    atomic_json(OUT / "report.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
