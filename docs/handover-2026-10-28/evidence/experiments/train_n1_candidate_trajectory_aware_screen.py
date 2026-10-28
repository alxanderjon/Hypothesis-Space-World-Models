#!/usr/bin/env python3
"""Trajectory-aware candidate ranker screen on the unchanged N1 corpus."""
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

from experiments.train_n1_candidate_relational_depth_screen import (
    DATA, SEED, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_candidate_state_relative_screen import native
from experiments.train_n1_candidate_top_of_list_screen import (
    alias_metrics, hard_good_metrics, hardest_bad_loss, ranking_metrics, state_records,
)
from experiments.train_n1_relational_architecture_screen import QueryAttention, atomic_json, parameters
from experiments.validate_d5_production_pipeline import _ball_case, _kepler_case


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_trajectory_aware_screen")
BASE_CHECKPOINT = Path(
    "outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
BASE_REPORT = Path("outputs/n1_candidate_top_of_list_screen/report.json")
PRIOR_FAILURES = {
    "kepler5_case_009_n14999_it1404",
    "kepler5_case_022_n5004_it491", "kepler5_case_022_n15000_it1367",
    "kepler5_case_033_n9998_it933", "kepler5_case_033_n19995_it1735",
    "kepler5_case_041_n5002_it504", "kepler5_case_041_n15000_it1364",
    "kepler5_case_041_n19984_it1764", "kepler5_case_053_n19998_it1717",
    "kepler5_case_054_n14993_it1426", "kepler5_case_058_n5002_it487",
}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TransitionTokenEncoder(nn.Module):
    """Variable T/D encoder retaining every time-component transition token."""
    def __init__(self, width=32):
        super().__init__(); self.width=width
        self.scalar=nn.Sequential(nn.Linear(6,width),nn.Tanh(),nn.Linear(width,width))
        self.component=nn.GRU(width,width,batch_first=True)
        self.temporal=nn.GRU(width,width,batch_first=True)
        self.norm=nn.LayerNorm(width)

    def forward(self, prefix):
        # One token for each (transition time, state component).  Inputs are
        # observed values only: x_t, x_(t+1), and generic positional identity.
        current=prefix[:-1]; following=prefix[1:]; t,d=current.shape
        time=torch.arange(t,dtype=current.dtype,device=current.device)/max(1,t-1)
        component=torch.arange(d,dtype=current.dtype,device=current.device)/max(1,d-1)
        inv_t=torch.full((t,d),1.0/max(1,t),dtype=current.dtype,device=current.device)
        inv_d=torch.full((t,d),1.0/max(1,d),dtype=current.dtype,device=current.device)
        raw=torch.stack((current,following,
            time[:,None].expand(-1,d),component[None,:].expand(t,-1),inv_t,inv_d),dim=-1)
        base=self.scalar(raw)
        component_context,_=self.component(base)
        temporal_context,_=self.temporal(base.transpose(0,1))
        return self.norm(base+component_context+temporal_context.transpose(0,1)).reshape(t*d,self.width)


class TrajectoryAwareHead(nn.Module):
    """Candidate-conditioned prefix reader plus mandatory tri-source fusion."""
    def __init__(self,width=32):
        super().__init__(); self.width=width
        self.trajectory=TransitionTokenEncoder(width)
        self.reader=QueryAttention(width)
        self.parameter_projection=nn.Linear(width,width)
        self.search_projection=nn.Linear(width,width)
        self.trajectory_projection=nn.Linear(width,width)
        self.output=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))

    def forward(self,prefix,parameter,search,base_logits):
        tokens=self.trajectory(prefix)
        message=self.reader(search,tokens)
        candidate=torch.tanh(self.parameter_projection(parameter)+self.search_projection(search))
        # Multiplicative fusion requires observed trajectory information; the
        # learned correction cannot reduce to a new search-only scalar head.
        joint=candidate*torch.tanh(self.trajectory_projection(message))
        return base_logits+self.output(joint).squeeze(-1)


@torch.no_grad()
def encode_search(encoder,states,grouped):
    output={}; encoder.eval()
    for number,state_id in enumerate(sorted(grouped),1):
        examples=grouped[state_id]; state=states[state_id]
        logits,stages=encoder(state,examples,return_stages=True)
        indices=torch.tensor([int(row["entity_index"]) for row in examples],dtype=torch.long)
        parameter=encoder.encode_coordinates(state.candidate_coords[indices])
        output[state_id]=dict(base=logits.detach().clone(),parameter=parameter.detach().clone(),
                              search=stages["prehead"].detach().clone())
        if number%25==0: print(json.dumps({"encoded_states":number,"total":len(grouped)}),flush=True)
    return output


def trajectories(grouped):
    cache={}; output={}
    for state_id,examples in grouped.items():
        key=(examples[0]["system"],int(examples[0]["case_id"]))
        if key not in cache:
            trajectory=(_ball_case(key[1])[0] if key[0]=="reflection_ball" else _kepler_case(key[1])[0])
            cache[key]=torch.from_numpy(np.asarray(trajectory[:128],np.float32))
        output[state_id]=cache[key]
    return output


def state_logits(model,state_id,encoded,prefixes):
    value=encoded[state_id]
    return model(prefixes[state_id],value["parameter"],value["search"],value["base"])


@torch.no_grad()
def mean_loss(model,encoded,prefixes,grouped,state_ids):
    model.eval(); values=[]
    for state_id in state_ids:
        examples=grouped[state_id]; target=torch.tensor([bool(r["target_good"]) for r in examples])
        loss=hardest_bad_loss(state_logits(model,state_id,encoded,prefixes),target)
        if loss is not None: values.append(float(loss))
    return float(np.mean(values)) if values else None


def train(model,encoded,prefixes,grouped,epochs):
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    eligible={s:e for s,e in grouped.items() if any(r["target_good"] for r in e) and any(not r["target_good"] for r in e)}
    train_ids=sorted(s for s,e in eligible.items() if e[0]["split"]=="train")
    held_ids=sorted(s for s,e in eligible.items() if e[0]["split"]=="held_out")
    best=None;history=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train(); order=train_ids.copy();random.Random(SEED+epoch).shuffle(order)
        optimizer.zero_grad();accumulation=0;losses=[]
        for state_id in order:
            examples=grouped[state_id];target=torch.tensor([bool(r["target_good"]) for r in examples])
            loss=hardest_bad_loss(state_logits(model,state_id,encoded,prefixes),target)
            (loss/4.0).backward();losses.append(float(loss.detach()));accumulation+=1
            if accumulation==4:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step();optimizer.zero_grad();accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step();optimizer.zero_grad()
        train_value=mean_loss(model,encoded,prefixes,grouped,train_ids)
        held_value=mean_loss(model,encoded,prefixes,grouped,held_ids)
        record={"epoch":epoch,"train_hardest_bad_loss":train_value,"held_hardest_bad_loss":held_value}
        history.append(record);print(json.dumps(record),flush=True)
        if best is None or held_value<best[0]:
            best=(held_value,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch)
    model.load_state_dict(best[1])
    return {"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started,
            "eligible_train_states":len(train_ids),"eligible_held_states":len(held_ids)}


@torch.no_grad()
def score_states(model,encoded,prefixes,grouped):
    model.eval();return {s:[float(v) for v in state_logits(model,s,encoded,prefixes).cpu().numpy()] for s in grouped}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=24);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={p.name:sha256(p) for p in paths};checkpoint_before=sha256(BASE_CHECKPOINT)
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    states={s:GraphState(metadata[s],e) for s,e in grouped.items()}
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);hard_tags[row["example_id"]].update(row["hard_case_tags"])

    saved=torch.load(BASE_CHECKPOINT,weights_only=False)
    encoder=DeepCandidateModel(32,3);encoder.load_state_dict(saved["model_state"]);encoder.eval()
    for p in encoder.parameters():p.requires_grad_(False)
    encoded=encode_search(encoder,states,grouped);prefixes=trajectories(grouped)
    torch.manual_seed(SEED+1501);np.random.seed(SEED+1501);random.seed(SEED+1501)
    model=TrajectoryAwareHead(32);fit=train(model,encoded,prefixes,grouped,args.epochs)
    scores=score_states(model,encoded,prefixes,grouped);records=state_records(grouped,scores)
    evaluation={}
    for split in ("train","held_out","fresh_test","all"):
        selected=records if split=="all" else [r for r in records if r["split"]==split]
        evaluation[split]={"overall":ranking_metrics(selected),
            "by_system":{system:ranking_metrics([r for r in selected if r["system"]==system])
                         for system in ("reflection_ball","kepler5")}}
    persistent=[r for r in records if r["system"]=="kepler5" and int(r["case_id"]) in PERSISTENT]
    prior_fail=[r for r in records if r["state_id"] in PRIOR_FAILURES]
    hard={"useful_pareto_dominated_candidate":hard_good_metrics(grouped,scores,hard_tags,"useful_pareto_dominated_candidate"),
          "useful_non_hull_candidate":hard_good_metrics(grouped,scores,hard_tags,"useful_non_hull_candidate"),
          "low_loss_bad_alias":alias_metrics(grouped,scores,hard_tags)}
    checkpoint=OUT/"trajectory_aware_candidate_ranker_screen_only.pt"
    torch.save({"status":"TRAJECTORY_AWARE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection",
        "frozen_n1_encoder":str(BASE_CHECKPOINT),"frozen_n1_encoder_sha256":checkpoint_before,
        "trajectory_and_fusion_parameters":parameters(model),"total_parameters":parameters(model)+parameters(encoder),
        "architecture":"time-component transition tokens; shared component/time GRUs; candidate attention; multiplicative fusion",
        "best_epoch":fit["best_epoch"],"model_state":model.state_dict(),"schema_version":1},checkpoint)
    baseline=json.loads(BASE_REPORT.read_text());checkpoint_after=sha256(BASE_CHECKPOINT)
    report=native({"status":"TRAJECTORY_AWARE_SCREEN_ONLY_NOT_PROMOTED",
        "architecture":{"frozen_n1_encoder_parameters":parameters(encoder),
            "trainable_trajectory_and_fusion_parameters":parameters(model),
            "total_parameters":parameters(encoder)+parameters(model),"width":32,"relational_depth":3,
            "trajectory_pooling":"none; candidate-conditioned attention reads retained time-component tokens",
            "system_identity_input":False,"ground_truth_input":False,"candidate_residual_input":False,
            "candidate_predicted_transition_input":False,"aggregate_trajectory_loss_input":False},
        "objective":"unchanged one softplus(max_BAD_score-max_GOOD_score) per eligible state",
        "fit":fit,"evaluation":evaluation,"persistent_kepler":ranking_metrics(persistent),
        "persistent_kepler_records":persistent,"prior_11_failure_results":prior_fail,
        "cases_22_41":[r for r in prior_fail if int(r["case_id"]) in (22,41)],"hard_subsets":hard,
        "baseline":{"overall":baseline["evaluation"]["all"]["overall"],
            "held_out":baseline["evaluation"]["held_out"]["overall"],
            "fresh_test":baseline["evaluation"]["fresh_test"]["overall"],
            "kepler_all":baseline["evaluation"]["all"]["by_system"]["kepler5"],
            "fresh_kepler":baseline["evaluation"]["fresh_test"]["by_system"]["kepler5"],
            "persistent_kepler":baseline["persistent_kepler"]["metrics"],
            "low_loss_bad_alias":baseline["hard_subsets"]["low_loss_bad_alias"]},
        "checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
        "dataset_hashes_before":data_before,"dataset_hashes_after":{p.name:sha256(p) for p in paths},
        "dataset_unchanged":data_before=={p.name:sha256(p) for p in paths},
        "protected_checkpoint_changes":[],"checkpoint_hashes_unchanged":checkpoint_before==checkpoint_after})
    atomic_json(OUT/"report.json",report)
    print(json.dumps({"architecture":report["architecture"],"best_epoch":fit["best_epoch"],
        "evaluation":evaluation,"persistent":report["persistent_kepler"],
        "prior_failures":[{"state_id":r["state_id"],"top_is_good":r.get("top_is_good"),
            "rank":r.get("best_good_rank"),"margin":r.get("best_good_minus_best_bad")} for r in prior_fail],
        "aliases":hard["low_loss_bad_alias"],"checkpoint_hashes_unchanged":report["checkpoint_hashes_unchanged"]},indent=2))


if __name__=="__main__":main()
