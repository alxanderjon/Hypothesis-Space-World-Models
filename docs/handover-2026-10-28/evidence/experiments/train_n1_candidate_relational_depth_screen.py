#!/usr/bin/env python3
"""Screen recurrent relational depth for the saved N1 candidate corpus only.

No search is run and no production checkpoint is modified.  The candidate
score is emitted from the contextualized candidate token; no global mean/max
pooling or fixed coordinate slots are used.
"""
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
from experiments.train_n1_relational_architecture_screen import (
    DATA, SEED, QueryAttention, RaggedState, RelationalQueryModel,
    atomic_json, binary_metrics, evaluate, parameters, ranking_metrics,
)


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_relational_depth_screen")
BASELINE = Path("outputs/n1_relational_architecture_screen/candidate_selection_screen_only.pt")
PERSISTENT = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraphState(RaggedState):
    """Original lossless state plus deterministic indices for stored relations."""
    def __init__(self, meta: dict, examples: list[dict]):
        super().__init__(meta, examples)
        z = np.load(meta["state_npz"])
        n, m = self.site_count, self.candidate_count
        group = np.asarray(z["cell_group"], np.int64)
        representatives = np.asarray(z["group_representatives"], np.int64)
        if len(representatives) <= int(group.max(initial=-1)):
            raise ValueError("group representative relation is incomplete")
        self.group_representative = torch.from_numpy(representatives[group])

        self.site_previous = torch.arange(n, dtype=torch.long) - 1
        self.site_next = torch.arange(n, dtype=torch.long) + 1
        if n:
            self.site_previous[0] = -1; self.site_next[-1] = -1
        self.candidate_previous = torch.arange(m, dtype=torch.long) - 1
        self.candidate_next = torch.arange(m, dtype=torch.long) + 1
        if m:
            self.candidate_previous[0] = -1; self.candidate_next[-1] = -1

        hull_position = np.asarray(z["hull_position"], np.int64)
        ordered = np.flatnonzero(hull_position >= 0)
        ordered = ordered[np.argsort(hull_position[ordered], kind="stable")]
        hull_previous = np.full(n, -1, np.int64)
        hull_next = np.full(n, -1, np.int64)
        if len(ordered) > 1:
            hull_previous[ordered[1:]] = ordered[:-1]
            hull_next[ordered[:-1]] = ordered[1:]
        self.hull_previous = torch.from_numpy(hull_previous)
        self.hull_next = torch.from_numpy(hull_next)


def relation_message(tokens: torch.Tensor, index: torch.Tensor,
                     transform: nn.Module) -> torch.Tensor:
    valid = index >= 0
    output = torch.zeros_like(tokens)
    if bool(valid.any()):
        output[valid] = transform(tokens[index[valid]])
    return output


class SparseRelationalLayer(nn.Module):
    """One shared graph-message round over relations already in the corpus."""
    def __init__(self, width: int):
        super().__init__()
        self.site_self = nn.Linear(width, width, bias=False)
        self.site_dominator = nn.Linear(width, width, bias=False)
        self.site_group = nn.Linear(width, width, bias=False)
        self.site_previous = nn.Linear(width, width, bias=False)
        self.site_next = nn.Linear(width, width, bias=False)
        self.site_hull_previous = nn.Linear(width, width, bias=False)
        self.site_hull_next = nn.Linear(width, width, bias=False)
        self.candidate_self = nn.Linear(width, width, bias=False)
        self.candidate_owner = nn.Linear(width, width, bias=False)
        self.candidate_previous = nn.Linear(width, width, bias=False)
        self.candidate_next = nn.Linear(width, width, bias=False)
        self.site_norm = nn.LayerNorm(width)
        self.candidate_norm = nn.LayerNorm(width)
        self.activation = nn.Tanh()

    def forward(self, state: GraphState, site: torch.Tensor,
                candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        site_message = self.site_self(site)
        site_message = site_message + relation_message(site, state.dominator, self.site_dominator)
        site_message = site_message + relation_message(site, state.group_representative, self.site_group)
        site_message = site_message + relation_message(site, state.site_previous, self.site_previous)
        site_message = site_message + relation_message(site, state.site_next, self.site_next)
        site_message = site_message + relation_message(site, state.hull_previous, self.site_hull_previous)
        site_message = site_message + relation_message(site, state.hull_next, self.site_hull_next)
        updated_site = self.site_norm(site + self.activation(site_message))

        candidate_message = self.candidate_self(candidate)
        candidate_message = candidate_message + self.candidate_owner(updated_site[state.owner])
        candidate_message = candidate_message + relation_message(
            candidate, state.candidate_previous, self.candidate_previous)
        candidate_message = candidate_message + relation_message(
            candidate, state.candidate_next, self.candidate_next)
        updated_candidate = self.candidate_norm(candidate + self.activation(candidate_message))
        return updated_site, updated_candidate


class DeepCandidateModel(RelationalQueryModel):
    """Baseline encoders/query decoder plus a recurrent state graph layer."""
    def __init__(self, width: int, depth: int):
        super().__init__(width, "candidate_selection")
        self.depth = depth
        # One parameter-shared block is recurrently applied. Depth 1 and 2 have
        # identical parameter counts; only relational computation depth changes.
        self.relational = SparseRelationalLayer(width)

    def forward(self, state: GraphState, examples: list[dict],
                return_stages: bool = False):
        site, candidate = self.encode_state(state)
        indices = torch.tensor([int(row["entity_index"]) for row in examples], dtype=torch.long)
        stages = {"encoded": candidate[indices]}
        for layer in range(self.depth):
            site, candidate = self.relational(state, site, candidate)
            stages[f"relational_{layer + 1}"] = candidate[indices]

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
        stages["ordered_neighbor"] = query
        query = self.update_norms[1](query + self.site_attention(query, site))
        stages["site_attention"] = query
        query = self.update_norms[2](query + self.candidate_attention(query, candidate))
        stages["candidate_attention"] = query
        state_values = torch.tensor([
            math.log1p(state.eval_count), math.log1p(state.site_count),
            math.log1p(state.candidate_count), math.log1p(state.dimension),
        ], dtype=torch.float32)
        query = self.update_norms[3](query + self.state_features(state_values)[None])
        stages["prehead"] = query
        logits = self.output(query).squeeze(-1)
        return (logits, stages) if return_stages else logits


def train(depth: int, states: dict[str, GraphState], rows: list[dict], epochs: int):
    torch.manual_seed(SEED + 320)
    model = DeepCandidateModel(32, depth)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    train_rows = [row for row in rows if row["split"] == "train"]
    held_rows = [row for row in rows if row["split"] == "held_out"]
    grouped = defaultdict(list)
    for row in train_rows: grouped[row["state_id"]].append(row)
    state_ids = sorted(grouped); history=[]; best=None; started=time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train(); random.Random(SEED + epoch).shuffle(state_ids)
        optimizer.zero_grad(); accumulation=0; losses=[]
        for state_id in state_ids:
            examples=grouped[state_id]
            target=torch.tensor([float(row["target_good"]) for row in examples])
            loss=criterion(model(states[state_id], examples), target)
            (loss / 4.0).backward(); accumulation += 1; losses.append(float(loss.detach()))
            if accumulation == 4:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); optimizer.zero_grad(); accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad()
        model.eval()
        train_scores=score(model,states,train_rows); held_scores=score(model,states,held_rows)
        train_metric=binary_metrics(train_rows,train_scores); held_metric=binary_metrics(held_rows,held_scores)
        held_loss=float(nn.functional.binary_cross_entropy(
            torch.tensor(held_scores).clamp(1e-7,1-1e-7),
            torch.tensor([float(row["target_good"]) for row in held_rows])))
        record={"epoch":epoch,"loss":float(np.mean(losses)),"held_loss":held_loss,
                "train_accuracy":train_metric["accuracy"],"held_accuracy":held_metric["accuracy"],
                "held_precision":held_metric["good_precision"],"held_recall":held_metric["good_recall"]}
        history.append(record); print("depth",depth,json.dumps(record),flush=True)
        if best is None or held_loss < best[0]:
            best=(held_loss,{key:value.detach().clone() for key,value in model.state_dict().items()},epoch)
    model.load_state_dict(best[1])
    return model,{"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started}


@torch.no_grad()
def score(model, states, rows):
    model.eval(); grouped=defaultdict(list); result={}
    for row in rows: grouped[row["state_id"]].append(row)
    for state_id, examples in grouped.items():
        values=torch.sigmoid(model(states[state_id],examples)).cpu().numpy()
        for row,value in zip(examples,values): result[row["example_id"]]=float(value)
    return [result[row["example_id"]] for row in rows]


def subset_metrics(rows, scores, predicate):
    selected=[(row,value) for row,value in zip(rows,scores) if predicate(row)]
    return binary_metrics([row for row,_ in selected],[value for _,value in selected])


def full_metrics(model, states, rows, hard_tags):
    result=evaluate(model,states,rows,hard_tags)
    all_scores=score(model,states,rows)
    result["kepler_all"]=subset_metrics(rows,all_scores,lambda row:row["system"]=="kepler5")
    result["persistent_kepler_good"]=subset_metrics(
        rows,all_scores,lambda row:row["system"]=="kepler5" and int(row["case_id"]) in PERSISTENT and row["target_good"])
    return result


def cosine(left, right):
    denominator=np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.dot(left,right)/denominator) if denominator else 1.0


@torch.no_grad()
def representation_geometry(model, states, rows):
    model.eval(); grouped=defaultdict(list)
    for row in rows:
        if row["system"]=="kepler5": grouped[row["state_id"]].append(row)
    grouped={key:value for key,value in grouped.items()
             if any(row["target_good"] for row in value) and any(not row["target_good"] for row in value)}
    stage_values=defaultdict(list)
    for state_id, examples in grouped.items():
        _,stages=model(states[state_id],examples,return_stages=True)
        good=np.asarray([bool(row["target_good"]) for row in examples])
        for name,tensor in stages.items():
            value=tensor.cpu().numpy()
            pairwise=np.linalg.norm(value[:,None]-value[None,:],axis=-1)
            positive=pairwise[np.triu_indices(len(value),1)]
            scale=float(np.median(positive[positive>0])) if np.any(positive>0) else 1.0
            for index in np.flatnonzero(good):
                bad=np.flatnonzero(~good); nearest=bad[int(np.argmin(pairwise[index,bad]))]
                stage_values[name].append((pairwise[index,nearest]/scale,cosine(value[index],value[nearest])))
    return {name:{"good_count":len(values),
                  "nearest_bad_distance_over_state_median":float(np.median([v[0] for v in values])),
                  "nearest_bad_cosine":float(np.median([v[1] for v in values]))}
            for name,values in stage_values.items()}


def baseline_summary(rows, hard_tags):
    report=json.loads(Path("outputs/n1_relational_architecture_screen/report.json").read_text())
    selected=report["results"]["candidate_selection"]
    return {"width":selected["selected_width"],"parameters":selected["parameters"],
            "status":selected["status"],"metrics":selected["metrics"],
            "hard_kepler_prehead_geometry":json.loads(
                Path("outputs/n1_candidate_kepler_failure_diagnosis/report.json").read_text()
            )["representation_geometry"]["prehead"]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=8)
    parser.add_argument("--depths",type=int,nargs="+",default=[1,2]);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    before=hashes(); dataset_hashes={path.name:sha256(path) for path in
        [DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]}
    rows=[row for row in json.loads((DATA/"dataset.json").read_text()) if row["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata=json.loads((DATA/"states.json").read_text()); states={}
    for number,meta in enumerate(metadata,1):
        if meta["state_id"] in grouped:
            print(f"load {number}/{len(metadata)} {meta['state_id']}",flush=True)
            states[meta["state_id"]]=GraphState(meta,grouped[meta["state_id"]])
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);hard_tags[row["example_id"]].update(row["hard_case_tags"])

    screens=[]
    for depth in args.depths:
        model,fit=train(depth,states,rows,args.epochs)
        metrics=full_metrics(model,states,rows,hard_tags)
        geometry=representation_geometry(model,states,rows)
        checkpoint=OUT/f"candidate_selection_depth{depth}_screen_only.pt"
        torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection",
                    "depth":depth,"width":32,"parameters":parameters(model),
                    "model_state":model.state_dict(),"schema_version":1},checkpoint)
        screens.append({"depth":depth,"width":32,"parameters":parameters(model),
                        "checkpoint":str(checkpoint),"fit":fit,"metrics":metrics,
                        "representation_geometry":geometry})
        atomic_json(OUT/f"depth{depth}_result.json",screens[-1])
    after=hashes(); changed=[path for path in set(before)&set(after)
                             if before[path]!=after[path] and not path.startswith(str(OUT))]
    after_dataset={path.name:sha256(path) for path in
        [DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]}
    report={"status":"RELATIONAL_DEPTH_SCREEN_ONLY_NOT_PROMOTED",
            "dataset_hashes_before":dataset_hashes,"dataset_hashes_after":after_dataset,
            "dataset_unchanged":dataset_hashes==after_dataset,
            "baseline":baseline_summary(rows,hard_tags),"screens":screens,
            "protected_checkpoint_changes":changed,"checkpoint_hashes_unchanged":not changed}
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
