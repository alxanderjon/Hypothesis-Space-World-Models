#!/usr/bin/env python3
"""Screen a same-state pairwise comparator on the frozen N1 candidate corpus.

The saved width-32/depth-3 hardest-BAD relational encoder is frozen.  Only a
small joint pair head is trained.  No search or production checkpoint is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
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
OUT = Path("outputs/n1_candidate_pairwise_comparator_screen")
ENCODER_CHECKPOINT = Path(
    "outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
BASELINE_REPORT = Path("outputs/n1_candidate_top_of_list_screen/report.json")
PERSISTENT_FAILED = {
    "kepler5_case_009_n14999_it1404",
    "kepler5_case_022_n5004_it491", "kepler5_case_022_n15000_it1367",
    "kepler5_case_033_n9998_it933", "kepler5_case_033_n19995_it1735",
    "kepler5_case_041_n5002_it504", "kepler5_case_041_n15000_it1364",
    "kepler5_case_041_n19984_it1764", "kepler5_case_053_n19998_it1717",
    "kepler5_case_054_n14993_it1426", "kepler5_case_058_n5002_it487",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class JointPairComparator(nn.Module):
    """Antisymmetric pair decision with a genuinely joint ordered-pair map."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.joint = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, 1))

    def ordered_value(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.joint(torch.cat((left, right), dim=-1)).squeeze(-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.ordered_value(left, right) - self.ordered_value(right, left)


@torch.no_grad()
def encode_states(encoder, states, grouped):
    encoder.eval(); output = {}
    for number, state_id in enumerate(sorted(grouped), 1):
        examples = grouped[state_id]
        _, stages = encoder(states[state_id], examples, return_stages=True)
        output[state_id] = stages["prehead"].detach().clone()
        if number % 25 == 0:
            print(json.dumps({"encoded_states": number, "total": len(grouped)}), flush=True)
    return output


def eligible_pairs(examples):
    good = [i for i, row in enumerate(examples) if row["target_good"]]
    bad = [i for i, row in enumerate(examples) if not row["target_good"]]
    return [(g, b) for g in good for b in bad]


@torch.no_grad()
def state_pair_loss(model, representation, examples):
    pairs = eligible_pairs(examples)
    if not pairs: return None
    g = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    forward = model(representation[g], representation[b])
    reverse = model(representation[b], representation[g])
    # Both orientations are supervised explicitly.
    return 0.5 * (nn.functional.softplus(-forward).mean() +
                  nn.functional.softplus(reverse).mean())


@torch.no_grad()
def mean_loss(model, representations, grouped, state_ids):
    model.eval(); values=[]
    for state_id in state_ids:
        value=state_pair_loss(model, representations[state_id], grouped[state_id])
        if value is not None: values.append(float(value))
    return float(np.mean(values)) if values else None


def train(model, representations, grouped, epochs):
    optimizer=torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    eligible={s:e for s,e in grouped.items() if eligible_pairs(e)}
    train_ids=sorted(s for s,e in eligible.items() if e[0]["split"]=="train")
    held_ids=sorted(s for s,e in eligible.items() if e[0]["split"]=="held_out")
    best=None; history=[]; started=time.monotonic()
    for epoch in range(1, epochs+1):
        model.train(); order=train_ids.copy(); random.Random(SEED+epoch).shuffle(order)
        optimizer.zero_grad(); accumulation=0; losses=[]
        for state_id in order:
            rep=representations[state_id]; examples=grouped[state_id]
            pairs=eligible_pairs(examples)
            g=torch.tensor([p[0] for p in pairs]); b=torch.tensor([p[1] for p in pairs])
            forward=model(rep[g],rep[b]); reverse=model(rep[b],rep[g])
            loss=0.5*(nn.functional.softplus(-forward).mean()+
                      nn.functional.softplus(reverse).mean())
            (loss/8.0).backward(); losses.append(float(loss.detach())); accumulation+=1
            if accumulation==8:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step(); optimizer.zero_grad(); accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step(); optimizer.zero_grad()
        train_value=mean_loss(model,representations,grouped,train_ids)
        held_value=mean_loss(model,representations,grouped,held_ids)
        record={"epoch":epoch,"train_pair_loss":train_value,"held_pair_loss":held_value}
        history.append(record); print(json.dumps(record),flush=True)
        if best is None or held_value<best[0]:
            best=(held_value,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch)
    model.load_state_dict(best[1])
    return {"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started,
            "eligible_train_states":len(train_ids),"eligible_held_states":len(held_ids)}


@torch.no_grad()
def pair_matrix(model, representation):
    count=len(representation)
    left=representation[:,None,:].expand(count,count,-1).reshape(count*count,-1)
    right=representation[None,:,:].expand(count,count,-1).reshape(count*count,-1)
    return model(left,right).reshape(count,count).cpu().numpy().astype(np.float64)


def directed_cycle_count(matrix):
    count=0; n=len(matrix)
    for a in range(n):
        for b in range(a+1,n):
            for c in range(b+1,n):
                ab=matrix[a,b]>0; bc=matrix[b,c]>0; ca=matrix[c,a]>0
                if (ab and bc and ca) or ((not ab) and (not bc) and (not ca)): count+=1
    return count


def rank_from_matrix(matrix):
    # Copeland wins, then summed pair margin, then canonical candidate order.
    wins=(matrix>0).sum(axis=1)
    margin=matrix.sum(axis=1)
    return np.lexsort((np.arange(len(matrix)), -margin, -wins)), wins, margin


def state_records(model, representations, grouped):
    records=[]
    for state_id,examples in grouped.items():
        matrix=pair_matrix(model,representations[state_id]); order,wins,margin=rank_from_matrix(matrix)
        good=np.asarray([bool(row["target_good"]) for row in examples]); pairs=eligible_pairs(examples)
        correct=sum(matrix[g,b]>0 for g,b in pairs)
        reverse_errors=sum(
            np.sign(matrix[a,b]) != -np.sign(matrix[b,a])
            for a in range(len(matrix)) for b in range(a+1,len(matrix)))
        antisym=float(np.max(np.abs(matrix+matrix.T))) if len(matrix) else 0.0
        record={"state_id":state_id,"system":examples[0]["system"],"case_id":int(examples[0]["case_id"]),
                "split":examples[0]["split"],"evaluation_count":int(examples[0]["evaluation_count"]),
                "candidate_count":len(examples),"good_count":int(good.sum()),
                "pair_correct":int(correct),"pair_total":len(pairs),
                "swap_disagreements":int(reverse_errors),"max_antisymmetry_residual":antisym,
                "directed_three_cycles":directed_cycle_count(matrix),
                "top_example_id":examples[int(order[0])]["example_id"],
                "top_is_good":bool(good[order[0]])}
        if good.any() and (~good).any():
            ranks=np.empty(len(examples),np.int64); ranks[order]=np.arange(1,len(examples)+1)
            gi=np.flatnonzero(good); bi=np.flatnonzero(~good)
            best_good=int(gi[np.argmin(ranks[gi])]); winning_bad=int(bi[np.argmin(ranks[bi])])
            record.update(best_good_rank=int(ranks[gi].min()),
                best_good_example_id=examples[best_good]["example_id"],
                winning_bad_example_id=examples[winning_bad]["example_id"],
                best_good_pair_wins=int(wins[best_good]),winning_bad_pair_wins=int(wins[winning_bad]),
                best_good_margin_sum=float(margin[best_good]),winning_bad_margin_sum=float(margin[winning_bad]),
                direct_good_over_winning_bad=float(matrix[best_good,winning_bad]))
        records.append(record)
    return records


def aggregate(records):
    pair_correct=sum(r["pair_correct"] for r in records); pair_total=sum(r["pair_total"] for r in records)
    positive=[r for r in records if "best_good_rank" in r]
    return {"states":len(records),"positive_states":len(positive),
            "pair_correct":int(pair_correct),"pair_total":int(pair_total),
            "pairwise_accuracy":float(pair_correct/max(1,pair_total)),
            "top_good_states":int(sum(r["top_is_good"] for r in positive)),
            "top_good_rate":float(sum(r["top_is_good"] for r in positive)/max(1,len(positive))),
            "best_good_rank_histogram":dict(sorted(Counter(r["best_good_rank"] for r in positive).items())),
            "states_with_cycles":int(sum(r["directed_three_cycles"]>0 for r in records)),
            "directed_three_cycles":int(sum(r["directed_three_cycles"] for r in records)),
            "swap_disagreements":int(sum(r["swap_disagreements"] for r in records)),
            "max_antisymmetry_residual":float(max((r["max_antisymmetry_residual"] for r in records),default=0.0))}


def alias_metrics(records, grouped, hard_tags):
    wins=examples_count=positive_aliases=0
    by_state={r["state_id"]:r for r in records}
    for state_id,examples in grouped.items():
        if not any(row["target_good"] for row in examples): continue
        top=by_state[state_id]["top_example_id"]
        for row in examples:
            if "low_loss_bad_alias" in hard_tags.get(row["example_id"],()):
                examples_count+=1; positive_aliases+=1; wins+=int(row["example_id"]==top)
    return {"examples_in_positive_states":examples_count,"top_choice_wins":wins,
            "rejected":positive_aliases-wins}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--epochs",type=int,default=80)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    before=hashes(); paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={p.name:sha256(p) for p in paths}
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    states={s:GraphState(metadata[s],e) for s,e in grouped.items()}
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line); hard_tags[row["example_id"]].update(row["hard_case_tags"])

    saved=torch.load(ENCODER_CHECKPOINT,weights_only=False)
    encoder=DeepCandidateModel(32,3); encoder.load_state_dict(saved["model_state"]); encoder.eval()
    for parameter in encoder.parameters(): parameter.requires_grad_(False)
    representations=encode_states(encoder,states,grouped)

    torch.manual_seed(SEED+911); np.random.seed(SEED+911); random.seed(SEED+911)
    model=JointPairComparator(32); fit=train(model,representations,grouped,args.epochs)
    records=state_records(model,representations,grouped)
    evaluation={}
    for split in ("train","held_out","fresh_test","all"):
        selected=records if split=="all" else [r for r in records if r["split"]==split]
        evaluation[split]={"overall":aggregate(selected),
            "kepler":aggregate([r for r in selected if r["system"]=="kepler5"]),
            "reflection_ball":aggregate([r for r in selected if r["system"]=="reflection_ball"])}
    persistent=[r for r in records if r["system"]=="kepler5" and r["case_id"] in PERSISTENT and "best_good_rank" in r]
    failed=[r for r in records if r["state_id"] in PERSISTENT_FAILED]
    baseline=json.loads(BASELINE_REPORT.read_text())
    checkpoint=OUT/"same_state_pairwise_candidate_comparator_screen_only.pt"
    torch.save({"status":"PAIRWISE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection_pairwise",
        "frozen_encoder_checkpoint":str(ENCODER_CHECKPOINT),"frozen_encoder_sha256":sha256(ENCODER_CHECKPOINT),
        "pair_head_parameters":parameters(model),"total_parameters_including_frozen_encoder":parameters(model)+parameters(encoder),
        "architecture":"antisymmetric joint([A,B])-joint([B,A]) width-32 MLP",
        "best_epoch":fit["best_epoch"],"model_state":model.state_dict(),"schema_version":1},checkpoint)
    after=hashes(); changed=[p for p in set(before)&set(after) if before[p]!=after[p] and not p.startswith(str(OUT))]
    report=native({"status":"PAIRWISE_SCREEN_ONLY_NOT_PROMOTED",
        "architecture":{"frozen_encoder":"width-32 depth-3 hardest-BAD relational encoder",
            "frozen_encoder_parameters":parameters(encoder),"trainable_pair_head_parameters":parameters(model),
            "total_parameters":parameters(encoder)+parameters(model),"global_pooling":False,
            "pair_operation":"joint(concat(A,B))-joint(concat(B,A))","antisymmetry_by_construction":True},
        "supervision":"all same-state GOOD/BAD pairs in both orientations; mean per state",
        "fit":fit,"evaluation":evaluation,"persistent_kepler":aggregate(persistent),
        "persistent_kepler_records":persistent,"previous_11_failed_states":failed,
        "cases_22_41":[r for r in failed if r["case_id"] in (22,41)],
        "low_loss_bad_alias":alias_metrics(records,grouped,hard_tags),
        "baseline":{"overall_top_good":"104/129","persistent_kepler_top_good":"17/28",
            "fresh_kepler_top_good":"7/10","low_loss_bad_alias_wins":20,
            "checkpoint":baseline["checkpoint"],"checkpoint_sha256":baseline["checkpoint_sha256"]},
        "checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
        "dataset_hashes_before":data_before,"dataset_hashes_after":{p.name:sha256(p) for p in paths},
        "dataset_unchanged":data_before=={p.name:sha256(p) for p in paths},
        "protected_checkpoint_changes":changed,"checkpoint_hashes_unchanged":not changed})
    atomic_json(OUT/"report.json",report)
    print(json.dumps({"headline":{"train":evaluation["train"]["overall"],
        "held_out":evaluation["held_out"]["overall"],"fresh":evaluation["fresh_test"]["overall"],
        "persistent":report["persistent_kepler"],"failed_11_top_good":sum(r["top_is_good"] for r in failed),
        "alias_wins":report["low_loss_bad_alias"]["top_choice_wins"]},
        "checkpoint_hashes_unchanged":not changed},indent=2))


if __name__=="__main__": main()
