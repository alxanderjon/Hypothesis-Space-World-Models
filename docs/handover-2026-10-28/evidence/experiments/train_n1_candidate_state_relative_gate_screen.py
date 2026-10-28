#!/usr/bin/env python3
"""Explicit state-query gate plus listwise candidate ranking, screen only."""
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

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[name]="1"

import numpy as np
import torch
from torch import nn

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import DATA,SEED,DeepCandidateModel,GraphState,PERSISTENT
from experiments.train_n1_candidate_state_relative_screen import (
    hard_metrics, native, rank_metrics, gate_metrics, sha256,
)
from experiments.train_n1_relational_architecture_screen import atomic_json,parameters


torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
OUT=Path("outputs/n1_candidate_state_relative_screen")
OLD_AUDIT=Path("outputs/n1_candidate_statewise_ranking_audit/report.json")
DEPTH_REPORT=Path("outputs/n1_candidate_relational_depth_screen/report.json")


class StateQueryCandidateModel(DeepCandidateModel):
    def __init__(self):
        super().__init__(32,3)
        self.state_query=nn.Parameter(torch.zeros(32))

    def forward_state(self,state:GraphState,examples:list[dict]):
        site,candidate=self.encode_state(state)
        for _ in range(self.depth):site,candidate=self.relational(state,site,candidate)
        indices=torch.tensor([int(row["entity_index"]) for row in examples],dtype=torch.long)
        query=candidate[indices]
        messages=[]
        for position,index in enumerate(indices.tolist()):
            neighbor_index,distance=state.query_neighbors[index]
            rank=torch.arange(len(neighbor_index),dtype=torch.float32)/max(1,len(neighbor_index)-1)
            relation=torch.stack((torch.log1p(distance),rank),dim=-1)
            bias=self.neighbor_bias(relation).squeeze(-1)[None]
            messages.append(self.neighbor_attention(query[position:position+1],site[neighbor_index],bias)[0])
        query=self.update_norms[0](query+torch.stack(messages))
        query=self.update_norms[1](query+self.site_attention(query,site))
        query=self.update_norms[2](query+self.candidate_attention(query,candidate))
        state_values=torch.tensor([math.log1p(state.eval_count),math.log1p(state.site_count),
                                   math.log1p(state.candidate_count),math.log1p(state.dimension)])
        state_message=self.state_features(state_values)[None]
        query=self.update_norms[3](query+state_message)
        candidate_logits=self.output(query).squeeze(-1)

        gate=self.state_query[None]
        gate=self.update_norms[1](gate+self.site_attention(gate,site))
        gate=self.update_norms[2](gate+self.candidate_attention(gate,candidate))
        gate=self.update_norms[3](gate+state_message)
        gate_logit=self.output(gate).squeeze()
        return candidate_logits,gate_logit

    def forward(self,state,examples):return self.forward_state(state,examples)[0]


def state_loss(candidate_logits,gate_logit,target):
    any_good=target.bool().any()
    gate=nn.functional.binary_cross_entropy_with_logits(
        gate_logit,torch.as_tensor(float(any_good)))
    if not bool(any_good):return gate
    good=target.bool()
    rank=torch.logsumexp(candidate_logits,dim=0)-torch.logsumexp(candidate_logits[good],dim=0)
    return gate+rank


@torch.no_grad()
def mean_loss(model,states,grouped,ids):
    model.eval();values=[]
    for state_id in ids:
        examples=grouped[state_id];target=torch.tensor([bool(row["target_good"]) for row in examples])
        candidate,gate=model.forward_state(states[state_id],examples)
        values.append(float(state_loss(candidate,gate,target)))
    return float(np.mean(values))


def train(model,states,grouped,epochs):
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    train_ids=sorted(s for s,v in grouped.items() if v[0]["split"]=="train")
    held_ids=sorted(s for s,v in grouped.items() if v[0]["split"]=="held_out")
    history=[];best=None;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();random.Random(SEED+epoch).shuffle(train_ids)
        optimizer.zero_grad();accumulation=0;losses=[]
        for state_id in train_ids:
            examples=grouped[state_id];target=torch.tensor([bool(row["target_good"]) for row in examples])
            candidate,gate=model.forward_state(states[state_id],examples)
            loss=state_loss(candidate,gate,target);(loss/4.).backward();losses.append(float(loss.detach()));accumulation+=1
            if accumulation==4:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();optimizer.zero_grad();accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();optimizer.zero_grad()
        train_value=mean_loss(model,states,grouped,train_ids);held_value=mean_loss(model,states,grouped,held_ids)
        record={"epoch":epoch,"train_state_loss":train_value,"held_state_loss":held_value}
        history.append(record);print(json.dumps(record),flush=True)
        if best is None or held_value<best[0]:
            best=(held_value,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch)
    model.load_state_dict(best[1]);return {"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started}


@torch.no_grad()
def scores(model,states,grouped):
    model.eval();output={}
    for state_id,examples in grouped.items():
        candidate,gate=model.forward_state(states[state_id],examples)
        output[state_id]={"logits":[float(v) for v in candidate.cpu().numpy()],"gate_logit":float(gate)}
    return output


def records(grouped,values):
    output=[]
    for state_id,examples in grouped.items():
        logits=np.asarray(values[state_id]["logits"]);gate=float(values[state_id]["gate_logit"])
        good=np.asarray([bool(r["target_good"]) for r in examples]);order=np.lexsort((np.arange(len(examples)),-logits))
        rank=np.empty(len(examples),int);rank[order]=np.arange(1,len(examples)+1)
        row={"state_id":state_id,"system":examples[0]["system"],"case_id":examples[0]["case_id"],
             "split":examples[0]["split"],"evaluation_count":examples[0]["evaluation_count"],
             "any_good":bool(good.any()),"predict_any":bool(gate>0),"gate_logit":gate,
             "candidate_count":len(examples),"good_count":int(good.sum())}
        if good.any():
            gi=np.flatnonzero(good);bi=np.flatnonzero(~good)
            row.update(best_good_rank=int(rank[gi].min()),
                       best_good_minus_best_bad=float(logits[gi].max()-logits[bi].max()),
                       pairwise_wins=int(sum(logits[g]>logits[b] for g in gi for b in bi)),
                       pairwise_pairs=int(len(gi)*len(bi)))
        output.append(row)
    return output


def adapted_hard_metrics(grouped,values,hard_tags,tag,good_tag):
    selected=[]
    for state_id,examples in grouped.items():
        logits=np.asarray(values[state_id]["logits"]);gate=float(values[state_id]["gate_logit"])
        good=np.asarray([bool(r["target_good"]) for r in examples])
        for index,row in enumerate(examples):
            if tag in hard_tags.get(row["example_id"],()):selected.append((index,logits,gate,good))
    if good_tag:
        above=sum(logits[i]>0 for i,logits,_,_ in selected)
        allbad=sum(logits[i]>logits[~good].max() for i,logits,_,good in selected)
        wins=sum(int(logits[i]>logits[b]) for i,logits,_,good in selected for b in np.flatnonzero(~good))
        pairs=sum(int((~good).sum()) for *_,good in selected)
        return {"examples":len(selected),"above_candidate_zero":above,
                "outranks_every_bad":allbad,"outranks_every_bad_rate":allbad/max(1,len(selected)),
                "pairwise_wins":wins,"pairwise_pairs":pairs,"pairwise_accuracy":wins/max(1,pairs)}
    rejected=0;top=0
    for i,logits,gate,good in selected:
        if good.any():rejected+=int(logits[i]<logits[good].max());top+=int(i==int(np.argmax(logits)))
        else:rejected+=int(gate<=0)
    return {"examples":len(selected),"rejected_by_state_decision":rejected,
            "rejected_by_state_decision_rate":rejected/max(1,len(selected)),
            "selected_as_top_in_positive_state":top}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=12);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);before=hashes();paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={p.name:sha256(p) for p in paths}
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    states={s:GraphState(metadata[s],e) for s,e in grouped.items()}
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);hard_tags[row["example_id"]].update(row["hard_case_tags"])
    torch.manual_seed(SEED+320);np.random.seed(SEED);random.seed(SEED)
    model=StateQueryCandidateModel();fit=train(model,states,grouped,args.epochs)
    value=scores(model,states,grouped);record=records(grouped,value)
    evaluation={}
    for split in ("train","held_out","fresh_test","all"):
        selected=record if split=="all" else [r for r in record if r["split"]==split]
        evaluation[split]={"gate":gate_metrics(selected),"ranking":rank_metrics(selected),
            "by_system":{system:{"gate":gate_metrics([r for r in selected if r["system"]==system]),
                                 "ranking":rank_metrics([r for r in selected if r["system"]==system])}
                         for system in ("reflection_ball","kepler5")}}
    persistent=[r for r in record if r["system"]=="kepler5" and int(r["case_id"]) in PERSISTENT]
    hard={tag:adapted_hard_metrics(grouped,value,hard_tags,tag,tag!="low_loss_bad_alias") for tag in
          ("useful_pareto_dominated_candidate","useful_non_hull_candidate","low_loss_bad_alias")}
    checkpoint=OUT/"candidate_selection_state_query_listwise_screen_only.pt"
    torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection",
                "supervision":"EXPLICIT_STATE_GATE_PLUS_MULTI_POSITIVE_LISTWISE","depth":3,"width":32,
                "parameters":parameters(model),"best_epoch":fit["best_epoch"],"model_state":model.state_dict(),
                "schema_version":1},checkpoint)
    after=hashes();changed=[p for p in set(before)&set(after) if before[p]!=after[p] and not p.startswith(str(OUT))]
    data_after={p.name:sha256(p) for p in paths}
    prior_depth=json.loads(DEPTH_REPORT.read_text())["screens"][-1]
    report=native({"status":"STATE_QUERY_LISTWISE_SCREEN_ONLY_NOT_PROMOTED",
        "architecture":{"width":32,"depth":3,"parameters":parameters(model),"new_parameters_for_state_query":32,
                        "candidate_architecture_unchanged":True,"global_pooling":False,
                        "loss":"unweighted state gate BCE plus multi-positive listwise rank loss; one scalar per state"},
        "fit":fit,"evaluation":evaluation,"persistent_kepler":{"gate":gate_metrics(persistent),"ranking":rank_metrics(persistent)},
        "hard_subsets":hard,"previous_depth3_binary":{"parameters":prior_depth["parameters"],
            "metrics":prior_depth["metrics"],"statewise":json.loads(OLD_AUDIT.read_text())["models"]["depth_3"]},
        "fixed_null_preliminary":json.loads((OUT/"fixed_null_preliminary_report.json").read_text()),
        "checkpoint":str(checkpoint),"dataset_hashes_before":data_before,"dataset_hashes_after":data_after,
        "dataset_unchanged":data_before==data_after,"protected_checkpoint_changes":changed,
        "checkpoint_hashes_unchanged":not changed})
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
