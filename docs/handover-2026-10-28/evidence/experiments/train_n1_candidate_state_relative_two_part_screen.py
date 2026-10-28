#!/usr/bin/env python3
"""Final two-part screen: frozen listwise ranker plus learned state gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib,json,math,os,random,time
from pathlib import Path

for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[name]="1"
import numpy as np
import torch
from torch import nn

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import DATA,SEED,DeepCandidateModel,GraphState,PERSISTENT
from experiments.train_n1_candidate_state_relative_screen import gate_metrics,rank_metrics,native,sha256
from experiments.train_n1_candidate_state_relative_gate_screen import adapted_hard_metrics
from experiments.train_n1_relational_architecture_screen import atomic_json,parameters

torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
OUT=Path("outputs/n1_candidate_state_relative_screen")
RANK_CHECKPOINT=OUT/"candidate_selection_fixed_null_preliminary_screen_only.pt"
OLD_AUDIT=Path("outputs/n1_candidate_statewise_ranking_audit/report.json")
OLD_HARD=OUT/"previous_depth3_state_choice_hard_metrics.json"


class TwoPartModel(nn.Module):
    def __init__(self):
        super().__init__();self.ranker=DeepCandidateModel(32,3)
        saved=torch.load(RANK_CHECKPOINT,weights_only=False)["model_state"]
        self.ranker.load_state_dict({k:v for k,v in saved.items() if k!="none_logit"})
        for parameter in self.ranker.parameters():parameter.requires_grad_(False)
        self.state_query=nn.Parameter(torch.zeros(32));self.gate_head=nn.Linear(32,1)

    @torch.no_grad()
    def context(self,state):
        site,candidate=self.ranker.encode_state(state)
        for _ in range(self.ranker.depth):site,candidate=self.ranker.relational(state,site,candidate)
        return site,candidate

    def gate(self,state):
        site,candidate=self.context(state);query=self.state_query[None]
        query=self.ranker.update_norms[1](query+self.ranker.site_attention(query,site))
        query=self.ranker.update_norms[2](query+self.ranker.candidate_attention(query,candidate))
        values=torch.tensor([math.log1p(state.eval_count),math.log1p(state.site_count),
                             math.log1p(state.candidate_count),math.log1p(state.dimension)])
        query=self.ranker.update_norms[3](query+self.ranker.state_features(values)[None])
        return self.gate_head(query).squeeze()

    @torch.no_grad()
    def candidate_logits(self,state,examples):return self.ranker(state,examples)


@torch.no_grad()
def loss_on(model,states,grouped,ids):
    model.eval();loss=[]
    for state_id in ids:
        target=float(any(row["target_good"] for row in grouped[state_id]))
        loss.append(float(nn.functional.binary_cross_entropy_with_logits(model.gate(states[state_id]),torch.tensor(target))))
    return float(np.mean(loss))


def train(model,states,grouped,epochs):
    optimizer=torch.optim.AdamW([model.state_query,*model.gate_head.parameters()],lr=2e-3,weight_decay=1e-5)
    train_ids=sorted(s for s,v in grouped.items() if v[0]["split"]=="train")
    held_ids=sorted(s for s,v in grouped.items() if v[0]["split"]=="held_out")
    best=None;history=[];started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();random.Random(SEED+epoch).shuffle(train_ids);optimizer.zero_grad();acc=0;losses=[]
        for state_id in train_ids:
            target=torch.tensor(float(any(row["target_good"] for row in grouped[state_id])))
            loss=nn.functional.binary_cross_entropy_with_logits(model.gate(states[state_id]),target)
            (loss/4.).backward();losses.append(float(loss.detach()));acc+=1
            if acc==4:optimizer.step();optimizer.zero_grad();acc=0
        if acc:optimizer.step();optimizer.zero_grad()
        train_loss=loss_on(model,states,grouped,train_ids);held_loss=loss_on(model,states,grouped,held_ids)
        record={"epoch":epoch,"train_gate_loss":train_loss,"held_gate_loss":held_loss}
        history.append(record);print(json.dumps(record),flush=True)
        if best is None or held_loss<best[0]:best=(held_loss,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch)
    model.load_state_dict(best[1]);return {"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started}


@torch.no_grad()
def evaluate_values(model,states,grouped):
    model.eval();values={}
    for state_id,examples in grouped.items():
        values[state_id]={"logits":[float(v) for v in model.candidate_logits(states[state_id],examples).numpy()],
                          "gate_logit":float(model.gate(states[state_id]))}
    return values


def records(grouped,values):
    output=[]
    for state_id,examples in grouped.items():
        logits=np.asarray(values[state_id]["logits"]);good=np.asarray([bool(r["target_good"]) for r in examples])
        order=np.lexsort((np.arange(len(examples)),-logits));rank=np.empty(len(examples),int);rank[order]=np.arange(1,len(examples)+1)
        row={"state_id":state_id,"system":examples[0]["system"],"case_id":examples[0]["case_id"],
             "split":examples[0]["split"],"any_good":bool(good.any()),
             "predict_any":bool(values[state_id]["gate_logit"]>0),"gate_logit":values[state_id]["gate_logit"]}
        if good.any():
            gi=np.flatnonzero(good);bi=np.flatnonzero(~good)
            row.update(best_good_rank=int(rank[gi].min()),
                       pairwise_wins=int(sum(logits[g]>logits[b] for g in gi for b in bi)),
                       pairwise_pairs=int(len(gi)*len(bi)))
        output.append(row)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=16);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);before=hashes();paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={p.name:sha256(p) for p in paths}
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    states={s:GraphState(metadata[s],e) for s,e in grouped.items()};tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);tags[row["example_id"]].update(row["hard_case_tags"])
    torch.manual_seed(SEED+320);np.random.seed(SEED);random.seed(SEED)
    model=TwoPartModel();fit=train(model,states,grouped,args.epochs);values=evaluate_values(model,states,grouped);record=records(grouped,values)
    evaluation={}
    for split in ("train","held_out","fresh_test","all"):
        selected=record if split=="all" else [r for r in record if r["split"]==split]
        evaluation[split]={"gate":gate_metrics(selected),"ranking":rank_metrics(selected),
            "by_system":{system:{"gate":gate_metrics([r for r in selected if r["system"]==system]),
                                 "ranking":rank_metrics([r for r in selected if r["system"]==system])}
                         for system in ("reflection_ball","kepler5")}}
    persistent=[r for r in record if r["system"]=="kepler5" and int(r["case_id"]) in PERSISTENT]
    hard={tag:adapted_hard_metrics(grouped,values,tags,tag,tag!="low_loss_bad_alias") for tag in
          ("useful_pareto_dominated_candidate","useful_non_hull_candidate","low_loss_bad_alias")}
    checkpoint=OUT/"candidate_selection_two_part_state_relative_screen_only.pt"
    torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection",
                "supervision":"FROZEN_LISTWISE_RANKER_PLUS_STATE_GATE","width":32,"depth":3,
                "parameters":parameters(model),"trainable_gate_parameters":65,"best_epoch":fit["best_epoch"],
                "model_state":model.state_dict(),"schema_version":1},checkpoint)
    after=hashes();changed=[p for p in set(before)&set(after) if before[p]!=after[p] and not p.startswith(str(OUT))]
    report=native({"status":"TWO_PART_STATE_RELATIVE_SCREEN_ONLY_NOT_PROMOTED",
        "architecture":{"width":32,"depth":3,"total_parameters":parameters(model),"trainable_gate_parameters":65,
                        "ranker_parameters_frozen":36066,"global_pooling":False},"fit":fit,"evaluation":evaluation,
        "persistent_kepler":{"gate":gate_metrics(persistent),"ranking":rank_metrics(persistent)},"hard_subsets":hard,
        "previous_depth3_binary_statewise":json.loads(OLD_AUDIT.read_text())["models"]["depth_3"],
        "previous_depth3_hard":json.loads(OLD_HARD.read_text()),"fixed_null_preliminary":json.loads((OUT/"fixed_null_preliminary_report.json").read_text()),
        "explicit_joint_preliminary":json.loads((OUT/"report.json").read_text()),"checkpoint":str(checkpoint),
        "dataset_hashes_before":data_before,"dataset_hashes_after":{p.name:sha256(p) for p in paths},
        "dataset_unchanged":data_before=={p.name:sha256(p) for p in paths},
        "protected_checkpoint_changes":changed,"checkpoint_hashes_unchanged":not changed})
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
