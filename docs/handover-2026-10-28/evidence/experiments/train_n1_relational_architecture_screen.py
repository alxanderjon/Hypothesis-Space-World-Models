#!/usr/bin/env python3
"""First relational architecture screen for N1 candidate and handoff labels.

The production search is never called.  Models consume the saved lossless
ragged states, use shared coordinate/entity weights, and emit a score from the
queried candidate/site token.  No global mean/max pooling is used.
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


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

DATA = Path("outputs/n1_candidate_handoff_supervision")
OUT = Path("outputs/n1_relational_architecture_screen")
SEED = 20261004


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def signed_log(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    finite = np.isfinite(value)
    result = np.zeros_like(value)
    result[finite] = np.sign(value[finite]) * np.log1p(np.abs(value[finite]))
    return result.astype(np.float32)


class RaggedState:
    def __init__(self, meta: dict, examples: list[dict]):
        self.meta = meta
        self.examples = examples
        z = np.load(meta["state_npz"])
        self.dimension = int(z["dimension"])
        self.site_coords = torch.from_numpy(np.asarray(z["sites"], np.float32))
        self.candidate_coords = torch.from_numpy(np.asarray(z["reservoir"], np.float32))
        n, m = len(self.site_coords), len(self.candidate_coords)

        selected = np.zeros(n, dtype=np.float32)
        selected_weight = np.zeros(n, dtype=np.float32)
        selected_defined = np.zeros(n, dtype=np.float32)
        selected_ids = np.asarray(z["selected_sites"], dtype=np.int64)
        selected[selected_ids] = 1.0
        selected_weight[selected_ids] = np.asarray(z["selected_cell_weights"], np.float32)
        selected_defined[selected_ids] = 1.0
        representatives = np.zeros(n, dtype=np.float32)
        representatives[np.asarray(z["group_representatives"], dtype=np.int64)] = 1.0
        hull_position = np.asarray(z["hull_position"], dtype=np.int64)
        group = np.asarray(z["cell_group"], dtype=np.int64)

        site_features = np.column_stack((
            signed_log(z["site_losses"]), signed_log(z["site_encoded_losses"]),
            signed_log(z["cell_sizes"]), np.asarray(z["pareto_dominated"], np.float32),
            (np.asarray(z["canonical_dominator"]) >= 0).astype(np.float32),
            (hull_position >= 0).astype(np.float32),
            signed_log(np.maximum(hull_position + 1, 0)),
            np.asarray(z["hull_epsilon_pass"], np.float32), selected,
            selected_weight, selected_defined,
            np.arange(n, dtype=np.float32) / max(1, n - 1),
            group.astype(np.float32) / max(1, int(group.max(initial=0))),
            representatives,
        )).astype(np.float32)
        self.site_features = torch.from_numpy(site_features)
        self.dominator = torch.from_numpy(np.asarray(z["canonical_dominator"], np.int64))

        score = np.asarray(z["candidate_score"], np.float64)
        role = np.asarray(z["candidate_role"], np.float32)
        status = np.asarray(z["candidate_routing_status"], np.int64)
        status_one_hot = np.zeros((m, 5), dtype=np.float32)
        status_one_hot[np.arange(m), np.clip(status - 1, 0, 4)] = 1.0
        candidate_features = np.column_stack((
            signed_log(z["candidate_owner_distance"]),
            signed_log(z["candidate_surrogate_prediction"]), role / 2.0,
            signed_log(score), np.isfinite(score).astype(np.float32),
            signed_log(z["candidate_score_rank"]),
            np.asarray(z["candidate_is_production_winner"], np.float32),
            status_one_hot,
            np.arange(m, dtype=np.float32) / max(1, m - 1),
        )).astype(np.float32)
        self.candidate_features = torch.from_numpy(candidate_features)
        self.owner = torch.from_numpy(np.asarray(z["candidate_owner"], np.int64))

        # Only queried candidate rows need their explicit ordered top-24 edges.
        self.query_neighbors = {}
        neighbor_indices = z["candidate_neighbor_indices"]
        neighbor_distances = z["candidate_neighbor_distances"]
        for row in examples:
            if row["task"] == "candidate_selection":
                index = int(row["entity_index"])
                self.query_neighbors[index] = (
                    torch.from_numpy(np.asarray(neighbor_indices[index], np.int64)),
                    torch.from_numpy(np.asarray(neighbor_distances[index], np.float32)),
                )
        self.site_count = n
        self.candidate_count = m
        self.eval_count = int(meta["evaluation_count"])


class QueryAttention(nn.Module):
    """Query-token cross attention; variable tokens are never pooled globally."""
    def __init__(self, width: int):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.scale = width ** -0.5

    def forward(self, query: torch.Tensor, tokens: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.q(query) @ self.k(tokens).T * self.scale
        if bias is not None:
            logits = logits + bias
        weights = torch.softmax(logits, dim=-1)
        return self.o(weights @ self.v(tokens))


class RelationalQueryModel(nn.Module):
    def __init__(self, width: int, task: str):
        super().__init__()
        self.width = width
        self.task = task
        # The same recurrent coordinate encoder is used for every coordinate,
        # entity, dimension and physical system. Coordinate position is a scalar
        # token feature rather than a fixed coordinate slot.
        self.coordinate = nn.GRU(3, width, batch_first=True)
        self.site_features = nn.Linear(14, width)
        self.candidate_features = nn.Linear(13, width)
        self.dominator_message = nn.Linear(width, width, bias=False)
        self.owner_message = nn.Linear(width, width, bias=False)
        self.site_norm = nn.LayerNorm(width)
        self.candidate_norm = nn.LayerNorm(width)
        self.neighbor_attention = QueryAttention(width)
        self.owned_candidate_attention = QueryAttention(width)
        self.site_attention = QueryAttention(width)
        self.candidate_attention = QueryAttention(width)
        self.neighbor_bias = nn.Sequential(nn.Linear(2, width), nn.Tanh(), nn.Linear(width, 1))
        self.state_features = nn.Linear(4, width)
        self.update_norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.output = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def encode_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        count, dimension = coordinates.shape
        position = torch.arange(dimension, dtype=coordinates.dtype, device=coordinates.device)
        position = position / max(1, dimension - 1)
        inverse_dimension = torch.full_like(position, 1.0 / dimension)
        extra = torch.stack((position, inverse_dimension), dim=-1)
        inputs = torch.cat((coordinates[..., None], extra[None].expand(count, -1, -1)), dim=-1)
        _, hidden = self.coordinate(inputs)
        return hidden[-1]

    def encode_state(self, state: RaggedState) -> tuple[torch.Tensor, torch.Tensor]:
        site = self.encode_coordinates(state.site_coords) + self.site_features(state.site_features)
        valid = state.dominator >= 0
        if bool(valid.any()):
            source = site[state.dominator[valid]]
            message = torch.zeros_like(site)
            message[valid] = self.dominator_message(source)
            site = site + message
        site = self.site_norm(site)
        candidate = (self.encode_coordinates(state.candidate_coords)
                     + self.candidate_features(state.candidate_features)
                     + self.owner_message(site[state.owner]))
        return site, self.candidate_norm(candidate)

    def forward(self, state: RaggedState, examples: list[dict]) -> torch.Tensor:
        site, candidate = self.encode_state(state)
        indices = torch.tensor([int(row["entity_index"]) for row in examples], dtype=torch.long)
        query = candidate[indices] if self.task == "candidate_selection" else site[indices]

        if self.task == "candidate_selection":
            relation_messages = []
            for position, index in enumerate(indices.tolist()):
                neighbor_index, distance = state.query_neighbors[index]
                rank = torch.arange(len(neighbor_index), dtype=torch.float32) / max(1, len(neighbor_index)-1)
                relation = torch.stack((torch.log1p(distance), rank), dim=-1)
                bias = self.neighbor_bias(relation).squeeze(-1)[None]
                relation_messages.append(
                    self.neighbor_attention(query[position:position+1], site[neighbor_index], bias)[0]
                )
            query = self.update_norms[0](query + torch.stack(relation_messages))
        else:
            messages = []
            for position, index in enumerate(indices.tolist()):
                owned = torch.nonzero(state.owner == index, as_tuple=False).flatten()
                if len(owned):
                    messages.append(self.owned_candidate_attention(
                        query[position:position+1], candidate[owned])[0])
                else:
                    messages.append(torch.zeros_like(query[position]))
            query = self.update_norms[0](query + torch.stack(messages))

        query = self.update_norms[1](query + self.site_attention(query, site))
        query = self.update_norms[2](query + self.candidate_attention(query, candidate))
        state_values = torch.tensor([
            math.log1p(state.eval_count), math.log1p(state.site_count),
            math.log1p(state.candidate_count), math.log1p(state.dimension),
        ], dtype=torch.float32)
        query = self.update_norms[3](query + self.state_features(state_values)[None])
        return self.output(query).squeeze(-1)


def parameters(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters())


def binary_metrics(rows: list[dict], scores: list[float]) -> dict:
    truth = np.asarray([bool(row["target_good"]) for row in rows], dtype=bool)
    predicted = np.asarray(scores) >= 0.5
    tp = int(np.sum(predicted & truth)); fp = int(np.sum(predicted & ~truth))
    tn = int(np.sum(~predicted & ~truth)); fn = int(np.sum(~predicted & truth))
    return {
        "examples": len(rows), "accuracy": float((tp + tn) / max(1, len(rows))),
        "good_precision": float(tp / max(1, tp + fp)),
        "good_recall": float(tp / max(1, tp + fn)),
        "true_positive": tp, "false_positive": fp, "true_negative": tn,
        "false_negative": fn,
    }


def ranking_metrics(rows: list[dict], scores: list[float]) -> dict:
    groups: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        groups[row["state_id"]].append((bool(row["target_good"]), float(score)))
    eligible = [values for values in groups.values()
                if any(v[0] for v in values) and any(not v[0] for v in values)]
    top_good = 0; wins = 0; pairs = 0; ties = 0
    for values in eligible:
        best = max(range(len(values)), key=lambda i: (values[i][1], -i))
        top_good += int(values[best][0])
        good = [v[1] for v in values if v[0]]; bad = [v[1] for v in values if not v[0]]
        for left in good:
            for right in bad:
                pairs += 1
                wins += int(left > right); ties += int(left == right)
    return {
        "eligible_states": len(eligible), "top_scored_is_good": top_good,
        "top_scored_good_rate": float(top_good / max(1, len(eligible))),
        "good_bad_pairs": pairs, "strict_pairwise_wins": wins,
        "ties": ties, "pairwise_ranking_accuracy": float(wins / max(1, pairs)),
    }


@torch.no_grad()
def score_rows(model: nn.Module, states: dict[str, RaggedState], rows: list[dict]) -> list[float]:
    model.eval(); output = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    for state_id, examples in grouped.items():
        probability = torch.sigmoid(model(states[state_id], examples)).cpu().numpy()
        for row, value in zip(examples, probability): output[row["example_id"]] = float(value)
    return [output[row["example_id"]] for row in rows]


def evaluate(model, states, task_rows, hard_tags) -> dict:
    result = {}
    all_scores = {}
    for split in ("train", "held_out", "fresh_test"):
        rows = [r for r in task_rows if r["split"] == split]
        scores = score_rows(model, states, rows); all_scores.update(zip((r["example_id"] for r in rows), scores))
        result[split] = {**binary_metrics(rows, scores), "ranking": ranking_metrics(rows, scores),
                         "by_system": {system: binary_metrics(
                             [r for r in rows if r["system"] == system],
                             [s for r, s in zip(rows, scores) if r["system"] == system])
                             for system in ("reflection_ball", "kepler5")}}
    rows = task_rows; scores = [all_scores[r["example_id"]] for r in rows]
    result["all"] = {**binary_metrics(rows, scores), "ranking": ranking_metrics(rows, scores)}

    tags = ("useful_pareto_dominated_candidate", "useful_non_hull_candidate",
            "recoverable_evaluated_nonincumbent", "low_loss_bad_alias",
            "persistent_kepler_localization_failure")
    result["hard_subsets"] = {}
    for tag in tags:
        selected = [(r, s) for r, s in zip(rows, scores) if tag in hard_tags.get(r["example_id"], ())]
        if selected:
            result["hard_subsets"][tag] = binary_metrics(
                [v[0] for v in selected], [v[1] for v in selected])

    truth = np.asarray([r["target_good"] for r in rows], bool); prediction = np.asarray(scores) >= .5
    error_rows = [(r, s, "FP" if not r["target_good"] else "FN")
                  for r, s, wrong in zip(rows, scores, prediction != truth) if wrong]
    false_positive = sorted((v for v in error_rows if v[2] == "FP"), key=lambda x: -x[1])[:20]
    false_negative = sorted((v for v in error_rows if v[2] == "FN"), key=lambda x: x[1])[:20]
    result["worst_errors"] = [dict(
        example_id=r["example_id"], kind=kind, probability=s, system=r["system"],
        split=r["split"], case_id=r["case_id"], state_id=r["state_id"],
        evaluation_count=r["evaluation_count"], routing=r["current_routing_status_name"],
        state_sites=states[r["state_id"]].site_count,
        state_candidates=states[r["state_id"]].candidate_count,
    ) for r, s, kind in false_positive + false_negative]

    def progress_bin(value):
        return ("<1k" if value < 1000 else "1k-5k" if value < 5000 else
                "5k-10k" if value < 10000 else "10k-15k" if value < 15000 else ">=15k")
    result["error_clusters"] = {
        "system": dict(Counter(r["system"] for r, _, _ in error_rows)),
        "search_progress": dict(Counter(progress_bin(r["evaluation_count"]) for r, _, _ in error_rows)),
        "routing": dict(Counter(r["current_routing_status_name"] for r, _, _ in error_rows)),
        "trajectory": [{"system": key[0], "case_id": key[1], "errors": count}
                       for key, count in Counter((r["system"], r["case_id"])
                                                 for r, _, _ in error_rows).most_common(15)],
        "state_size": distribution([states[r["state_id"]].site_count for r, _, _ in error_rows]),
    }
    return result


def distribution(values) -> dict:
    values = np.asarray(values, np.float64)
    if not len(values): return {"count": 0}
    return {"count": len(values), "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9, method="higher")),
            "maximum": float(values.max())}


def train(task: str, width: int, states: dict[str, RaggedState], rows: list[dict], epochs: int) -> tuple[nn.Module, dict]:
    torch.manual_seed(SEED + width + (0 if task == "candidate_selection" else 1000))
    model = RelationalQueryModel(width, task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_rows = [r for r in rows if r["split"] == "train"]
    held_rows = [r for r in rows if r["split"] == "held_out"]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in train_rows: grouped[row["state_id"]].append(row)
    state_ids = sorted(grouped)
    best = None; history = []; started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); random.Random(SEED + epoch).shuffle(state_ids)
        losses=[]; optimizer.zero_grad(); accumulation=0
        for state_id in state_ids:
            examples=grouped[state_id]
            target=torch.tensor([float(r["target_good"]) for r in examples], dtype=torch.float32)
            loss=criterion(model(states[state_id], examples), target)
            (loss/4.0).backward();accumulation+=1;losses.append(float(loss.detach()))
            if accumulation == 4:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step();optimizer.zero_grad();accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0);optimizer.step();optimizer.zero_grad()
        train_scores=score_rows(model,states,train_rows);held_scores=score_rows(model,states,held_rows)
        train_metrics=binary_metrics(train_rows,train_scores);held_metrics=binary_metrics(held_rows,held_scores)
        held_loss=float(nn.functional.binary_cross_entropy(
            torch.tensor(held_scores).clamp(1e-7,1-1e-7),
            torch.tensor([float(r["target_good"]) for r in held_rows])))
        record={"epoch":epoch,"loss":float(np.mean(losses)),"held_loss":held_loss,
                "train_accuracy":train_metrics["accuracy"],"held_accuracy":held_metrics["accuracy"],
                "held_precision":held_metrics["good_precision"],"held_recall":held_metrics["good_recall"]}
        history.append(record)
        if best is None or held_loss < best[0]:
            best=(held_loss,{key:value.detach().clone() for key,value in model.state_dict().items()},epoch)
        print(task,width,json.dumps(record),flush=True)
    model.load_state_dict(best[1])
    return model,{"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=24)
    parser.add_argument("--candidate-width32-only",action="store_true");args=parser.parse_args()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);OUT.mkdir(parents=True,exist_ok=True)
    before=hashes();atomic_json(OUT/"checkpoint_hashes_before.json",before)
    rows=json.loads((DATA/"dataset.json").read_text());state_rows=json.loads((DATA/"states.json").read_text())
    examples_by_state=defaultdict(list)
    for row in rows:examples_by_state[row["state_id"]].append(row)
    states={}
    for number,meta in enumerate(state_rows,1):
        print(f"load {number}/{len(state_rows)} {meta['state_id']}",flush=True)
        states[meta["state_id"]]=RaggedState(meta,examples_by_state[meta["state_id"]])
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);hard_tags[row["example_id"]].update(row["hard_case_tags"])

    if args.candidate_width32_only:
        report=json.loads((OUT/"report.json").read_text())
        task="candidate_selection";task_rows=[r for r in rows if r["task"]==task]
        model,fit=train(task,32,states,task_rows,args.epochs);metrics=evaluate(model,states,task_rows,hard_tags)
        screen={"width":32,"parameters":parameters(model),"fit":fit,"metrics":metrics}
        report["results"][task]["screens"].append(screen)
        prior=report["results"][task]
        prior_screen=min(prior["screens"][:-1],key=lambda value:value["fit"]["history"][value["fit"]["best_epoch"]-1]["held_loss"])
        prior_loss=prior_screen["fit"]["history"][prior_screen["fit"]["best_epoch"]-1]["held_loss"]
        new_loss=fit["history"][fit["best_epoch"]-1]["held_loss"]
        if new_loss < prior_loss:
            prior.update(selected_width=32,parameters=parameters(model),metrics=metrics)
            checkpoint=OUT/f"{task}_screen_only.pt"
            torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":task,
                        "model_state":model.state_dict(),"width":32,
                        "parameters":parameters(model),"schema_version":1},checkpoint)
        torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":task,
                    "model_state":model.state_dict(),"width":32,
                    "parameters":parameters(model),"schema_version":1},
                   OUT/"candidate_selection_width32_screen_only.pt")
        after=hashes();atomic_json(OUT/"checkpoint_hashes_after.json",after)
        changed=[path for path in set(before)&set(after)
                 if before[path]!=after[path] and not path.startswith(str(OUT))]
        report["checkpoint_hashes_unchanged"]=not changed
        report["protected_checkpoint_changes"]=changed
        atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2));return

    results={}
    for task in ("candidate_selection","handoff_promotion"):
        task_rows=[r for r in rows if r["task"]==task]
        screens=[]
        model,fit=train(task,8,states,task_rows,args.epochs)
        metrics=evaluate(model,states,task_rows,hard_tags)
        screens.append({"width":8,"parameters":parameters(model),"fit":fit,"metrics":metrics})
        # Increase only when the smallest model has clear training underfit.
        # This is a screen diagnostic, not an acceptance gate or production rule.
        underfit=metrics["train"]["accuracy"] < .90
        if underfit:
            larger,larger_fit=train(task,16,states,task_rows,args.epochs)
            larger_metrics=evaluate(larger,states,task_rows,hard_tags)
            screens.append({"width":16,"parameters":parameters(larger),"fit":larger_fit,"metrics":larger_metrics})
            # Choose only for reporting by held-out BCE at the saved best epoch.
            if larger_fit["history"][larger_fit["best_epoch"]-1]["held_loss"] < fit["history"][fit["best_epoch"]-1]["held_loss"]:
                model,fit,metrics=larger,larger_fit,larger_metrics
        checkpoint=OUT/f"{task}_screen_only.pt"
        torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":task,
                    "model_state":model.state_dict(),"width":model.width,
                    "parameters":parameters(model),"schema_version":1},checkpoint)
        results[task]={"selected_width":model.width,"parameters":parameters(model),
                       "checkpoint":str(checkpoint),"status":"SCREEN_ONLY_NOT_PROMOTED",
                       "screens":screens,"metrics":metrics}
    after=hashes();atomic_json(OUT/"checkpoint_hashes_after.json",after)
    changed=[path for path in set(before)&set(after)
             if before[path]!=after[path] and not path.startswith(str(OUT))]
    report={"status":"COMPLETE_ARCHITECTURE_SCREEN","dataset":str(DATA/"dataset.jsonl"),
            "architecture":{
                "name":"query-token sparse relational attention",
                "coordinate_handling":"shared GRU over scalar coordinate tokens with scalar ordinal/dimension features",
                "site_relations":["canonical dominator edge","selected/hull/history token fields"],
                "candidate_relations":["candidate-owner edge","ordered top-24 query-neighbor edges"],
                "global_state":"query cross-attention to every site token and every reservoir token",
                "pooling":"none: no global mean/max; output is read from the updated query token",
            },"capacity_policy":"width 8 first; width 16 only when width-8 train accuracy <0.90",
            "class_balance":"natural corpus; no weights, duplication, or resampling",
            "ground_truth_runtime_input":False,"results":results,
            "checkpoint_hashes_unchanged":not changed,
            "protected_checkpoint_changes":changed,
            "new_screen_artifacts":sorted(path for path in set(after)-set(before)
                                          if path.startswith(str(OUT)))}
    atomic_json(OUT/"report.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
