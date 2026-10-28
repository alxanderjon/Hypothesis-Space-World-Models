#!/usr/bin/env python3
"""Screen an explicit ordered-site encoder for N1 candidate ranking only."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from experiments.train_n1_candidate_top_of_list_screen import (
    alias_metrics, hard_good_metrics, hardest_bad_loss, ranking_metrics, state_records,
)
from experiments.train_n1_relational_architecture_screen import QueryAttention, atomic_json, parameters


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_temporal_screen")
BASE_CHECKPOINT = Path("outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
BASE_REPORT = Path("outputs/n1_candidate_top_of_list_screen/report.json")
TEMPORAL_AUDIT = Path("outputs/n1_candidate_temporal_diagnosis/report.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TemporalCandidateModel(DeepCandidateModel):
    """Frozen snapshot ranker plus one shared causal history encoder."""
    def __init__(self):
        super().__init__(32, 3)
        saved = torch.load(BASE_CHECKPOINT, weights_only=False)["model_state"]
        self.load_state_dict(saved)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.temporal_gru = nn.GRU(32, 32, num_layers=1, batch_first=True)
        self.temporal_attention = QueryAttention(32)
        self.temporal_fusion = nn.Linear(32, 32, bias=False)
        nn.init.zeros_(self.temporal_fusion.weight)

    def temporal_parameters(self):
        return [p for name, p in self.named_parameters() if name.startswith("temporal_")]

    def forward(self, state: GraphState, examples: list[dict]):
        site, candidate = self.encode_state(state)
        indices = torch.tensor([int(row["entity_index"]) for row in examples], dtype=torch.long)
        for _ in range(self.depth):
            site, candidate = self.relational(state, site, candidate)

        # The tensor order is the exact evaluated-site insertion order.  Every
        # site is retained; there is no temporal pooling or hand summary.
        temporal_site, _ = self.temporal_gru(site[None])
        temporal_site = temporal_site[0]

        query = candidate[indices]
        relation_messages = []
        for position, index in enumerate(indices.tolist()):
            neighbor_index, distance = state.query_neighbors[index]
            rank = torch.arange(len(neighbor_index), dtype=torch.float32) / max(1, len(neighbor_index)-1)
            relation = torch.stack((torch.log1p(distance), rank), dim=-1)
            bias = self.neighbor_bias(relation).squeeze(-1)[None]
            relation_messages.append(self.neighbor_attention(
                query[position:position+1], site[neighbor_index], bias)[0])
        query = self.update_norms[0](query + torch.stack(relation_messages))
        query = self.update_norms[1](query + self.site_attention(query, site))
        query = self.update_norms[2](query + self.candidate_attention(query, candidate))
        state_values = torch.tensor([
            math.log1p(state.eval_count), math.log1p(state.site_count),
            math.log1p(state.candidate_count), math.log1p(state.dimension),
        ], dtype=torch.float32)
        query = self.update_norms[3](query + self.state_features(state_values)[None])

        temporal = self.temporal_attention(query, temporal_site)
        fused = query + self.temporal_fusion(temporal)
        return self.output(fused).squeeze(-1)


@torch.no_grad()
def mean_loss(model, states, grouped, ids):
    model.eval(); result = []
    for state_id in ids:
        examples = grouped[state_id]
        target = torch.tensor([bool(row["target_good"]) for row in examples])
        loss = hardest_bad_loss(model(states[state_id], examples), target)
        if loss is not None: result.append(float(loss))
    return float(np.mean(result))


def train(model, states, grouped, epochs):
    trainable = model.temporal_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=2e-3, weight_decay=1e-5)
    eligible = {s: e for s, e in grouped.items()
                if any(r["target_good"] for r in e) and any(not r["target_good"] for r in e)}
    train_ids = sorted(s for s, e in eligible.items() if e[0]["split"] == "train")
    held_ids = sorted(s for s, e in eligible.items() if e[0]["split"] == "held_out")
    history = []; best = None; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); random.Random(SEED + epoch).shuffle(train_ids)
        losses = []
        for state_id in train_ids:
            optimizer.zero_grad()
            examples = grouped[state_id]
            target = torch.tensor([bool(row["target_good"]) for row in examples])
            loss = hardest_bad_loss(model(states[state_id], examples), target)
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step(); losses.append(float(loss.detach()))
        train_value = mean_loss(model, states, grouped, train_ids)
        held_value = mean_loss(model, states, grouped, held_ids)
        record = {"epoch": epoch, "train_hardest_bad_loss": train_value,
                  "held_hardest_bad_loss": held_value}
        history.append(record); print(json.dumps(record), flush=True)
        if best is None or held_value < best[0]:
            best = (held_value, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    return {"history": history, "best_epoch": best[2], "seconds": time.monotonic()-started,
            "eligible_train_states": len(train_ids), "eligible_held_states": len(held_ids)}


@torch.no_grad()
def score_states(model, states, grouped):
    model.eval(); return {s: [float(v) for v in model(states[s], e).cpu().numpy()]
                          for s, e in grouped.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    before = hashes(); paths = [DATA/"dataset.json", DATA/"states.json", DATA/"hard_cases.jsonl"]
    data_before = {p.name: sha256(p) for p in paths}
    rows = [r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"] == "candidate_selection"]
    grouped = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata = {r["state_id"]: r for r in json.loads((DATA/"states.json").read_text())}
    states = {s: GraphState(metadata[s], e) for s, e in grouped.items()}
    hard_tags = defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); hard_tags[row["example_id"]].update(row["hard_case_tags"])

    torch.manual_seed(SEED + 640); np.random.seed(SEED); random.seed(SEED)
    model = TemporalCandidateModel()
    # Zero residual must reproduce the base ranker exactly before fitting.
    base = DeepCandidateModel(32, 3)
    base.load_state_dict(torch.load(BASE_CHECKPOINT, weights_only=False)["model_state"]); base.eval(); model.eval()
    initial_max_difference = 0.0
    with torch.no_grad():
        for state_id in sorted(grouped)[:8]:
            examples = grouped[state_id]
            initial_max_difference = max(initial_max_difference,
                float(torch.max(torch.abs(model(states[state_id], examples)-base(states[state_id], examples)))))
    if initial_max_difference != 0.0:
        raise RuntimeError(f"temporal residual does not preserve baseline: {initial_max_difference}")

    fit = train(model, states, grouped, args.epochs)
    scores = score_states(model, states, grouped); records = state_records(grouped, scores)
    evaluation = {}
    for split in ("train", "held_out", "fresh_test", "all"):
        selected = records if split == "all" else [r for r in records if r["split"] == split]
        evaluation[split] = {"overall": ranking_metrics(selected),
                             "by_system": {system: ranking_metrics([r for r in selected if r["system"] == system])
                                           for system in ("reflection_ball", "kepler5")}}
    persistent = [r for r in records if r["system"] == "kepler5" and int(r["case_id"]) in PERSISTENT]
    persistent_by_state = {r["state_id"]: r for r in persistent}
    base_report = json.loads(BASE_REPORT.read_text())
    base_failures = {r["state_id"]: r for r in base_report["persistent_kepler"]["remaining_bad_wins"]}
    audit = json.loads(TEMPORAL_AUDIT.read_text())
    cue_count = {}
    directions = audit["descriptive_temporal_cue_count"]["features"]
    for pair in audit["pairs"]:
        cue_count[pair["state_id"]] = sum(
            pair["good"][f] < pair["bad"][f] if direction == "GOOD_LOW"
            else pair["good"][f] > pair["bad"][f] for f, direction in directions.items())
    base_failure_results = []
    for state_id, prior in base_failures.items():
        current = persistent_by_state[state_id]
        base_failure_results.append(dict(
            state_id=state_id, case_id=int(current["case_id"]), split=current["split"],
            evaluation_count=int(current["evaluation_count"]), temporal_cues_favoring_good=int(cue_count[state_id]),
            prior_margin=float(prior["best_good_minus_best_bad"]),
            temporal_margin=float(current["best_good_minus_best_bad"]),
            repaired=bool(current["best_good_rank"] == 1),
            temporal_best_good_rank=int(current["best_good_rank"])))
    base_failure_results.sort(key=lambda r: (r["case_id"], r["evaluation_count"]))
    hard = {"useful_pareto_dominated_candidate": hard_good_metrics(grouped, scores, hard_tags, "useful_pareto_dominated_candidate"),
            "useful_non_hull_candidate": hard_good_metrics(grouped, scores, hard_tags, "useful_non_hull_candidate"),
            "low_loss_bad_alias": alias_metrics(grouped, scores, hard_tags)}

    checkpoint_path = OUT/"candidate_selection_temporal_screen_only.pt"
    torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED", "task":"candidate_selection",
                "supervision":"STATE_BALANCED_HARDEST_BAD_ZERO_MARGIN", "width":32, "depth":3,
                "temporal_encoder":"one_layer_unidirectional_gru_over_all_ordered_sites_plus_candidate_attention",
                "parameters":parameters(model), "trainable_temporal_parameters":sum(p.numel() for p in model.temporal_parameters()),
                "best_epoch":fit["best_epoch"], "model_state":model.state_dict(), "schema_version":1}, checkpoint_path)
    after = hashes(); changed = [p for p in set(before)&set(after) if before[p] != after[p] and not p.startswith(str(OUT))]
    report = native(dict(
        status="TEMPORAL_ENCODER_SCREEN_ONLY_NOT_PROMOTED",
        architecture=dict(width=32, depth=3, base_parameters=36066, total_parameters=parameters(model),
            trainable_temporal_parameters=sum(p.numel() for p in model.temporal_parameters()),
            frozen_base=True, temporal_encoder="one shared width-32 unidirectional GRU over every site in insertion order",
            candidate_relative_fusion="candidate query attention over all temporal site states; zero-initialized residual before frozen output head",
            global_pooling=False, prior_reservoirs_used=False, hand_designed_temporal_summaries=False),
        initialization=dict(max_logit_difference_from_hardest_bad_baseline=initial_max_difference),
        fit=fit, evaluation=evaluation,
        persistent_kepler=dict(metrics=ranking_metrics(persistent), baseline_failure_results=base_failure_results,
            repaired=sum(r["repaired"] for r in base_failure_results),
            remaining=sum(not r["repaired"] for r in base_failure_results)),
        hard_subsets=hard,
        baseline=dict(checkpoint=str(BASE_CHECKPOINT), checkpoint_sha256=sha256(BASE_CHECKPOINT),
            report=base_report), temporal_audit_reference=dict(report=str(TEMPORAL_AUDIT), features=directions),
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path),
        dataset_hashes_before=data_before, dataset_hashes_after={p.name:sha256(p) for p in paths},
        dataset_unchanged=data_before == {p.name:sha256(p) for p in paths},
        protected_checkpoint_changes=changed, checkpoint_hashes_unchanged=not changed))
    atomic_json(OUT/"report.json", report); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
