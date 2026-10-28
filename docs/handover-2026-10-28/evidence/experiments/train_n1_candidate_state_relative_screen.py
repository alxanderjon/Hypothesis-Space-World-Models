#!/usr/bin/env python3
"""State-balanced listwise screen for the existing N1 candidate model."""
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
    os.environ[name]="1"

import numpy as np
import torch
from torch import nn

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import (
    DATA, SEED, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_relational_architecture_screen import atomic_json, parameters


torch.set_default_dtype(torch.float32);torch.set_num_threads(1)
OUT=Path("outputs/n1_candidate_state_relative_screen")
OLD_AUDIT=Path("outputs/n1_candidate_statewise_ranking_audit/report.json")


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {native(key):native(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [native(item) for item in value]
    return value


class StateRelativeCandidateModel(DeepCandidateModel):
    def __init__(self):
        super().__init__(32,3)
        # Null is an action, not a pooled state representation. Candidate logits
        # already condition on the complete ragged state through the decoder.
        self.none_logit=nn.Parameter(torch.zeros(()))


def listwise_loss(logits,target,none_logit):
    denominator=torch.logsumexp(torch.cat((logits,none_logit[None])),dim=0)
    good=target.bool()
    numerator=torch.logsumexp(logits[good],dim=0) if bool(good.any()) else none_logit
    return denominator-numerator


@torch.no_grad()
def state_scores(model,states,grouped):
    model.eval();output={}
    for state_id,examples in grouped.items():
        logits=model(states[state_id],examples).cpu().numpy()
        output[state_id]={"logits":[float(value) for value in logits],
                          "none_logit":float(model.none_logit.detach())}
    return output


@torch.no_grad()
def validation_loss(model,states,grouped,state_ids):
    model.eval();loss=[]
    for state_id in state_ids:
        examples=grouped[state_id]
        target=torch.tensor([bool(row["target_good"]) for row in examples])
        loss.append(float(listwise_loss(model(states[state_id],examples),target,model.none_logit)))
    return float(np.mean(loss))


def train(model,states,grouped,epochs):
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5)
    train_ids=sorted(state_id for state_id,examples in grouped.items() if examples[0]["split"]=="train")
    held_ids=sorted(state_id for state_id,examples in grouped.items() if examples[0]["split"]=="held_out")
    history=[];best=None;started=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();random.Random(SEED+epoch).shuffle(train_ids)
        optimizer.zero_grad();accumulation=0;losses=[]
        for state_id in train_ids:
            examples=grouped[state_id]
            target=torch.tensor([bool(row["target_good"]) for row in examples])
            loss=listwise_loss(model(states[state_id],examples),target,model.none_logit)
            (loss/4.0).backward();accumulation+=1;losses.append(float(loss.detach()))
            if accumulation==4:
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step();optimizer.zero_grad();accumulation=0
        if accumulation:
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();optimizer.zero_grad()
        held=validation_loss(model,states,grouped,held_ids)
        train_value=validation_loss(model,states,grouped,train_ids)
        record={"epoch":epoch,"train_state_listwise_loss":train_value,
                "held_state_listwise_loss":held,"none_logit":float(model.none_logit.detach())}
        history.append(record);print(json.dumps(record),flush=True)
        if best is None or held<best[0]:
            best=(held,{key:value.detach().clone() for key,value in model.state_dict().items()},epoch)
    model.load_state_dict(best[1])
    return {"history":history,"best_epoch":best[2],"seconds":time.monotonic()-started}


def gate_metrics(records):
    truth=np.asarray([record["any_good"] for record in records],bool)
    prediction=np.asarray([record["predict_any"] for record in records],bool)
    tp=int(np.sum(truth&prediction));tn=int(np.sum(~truth&~prediction))
    fp=int(np.sum(~truth&prediction));fn=int(np.sum(truth&~prediction))
    return {"states":len(records),"accuracy":float((tp+tn)/max(1,len(records))),
            "any_good_recall":float(tp/max(1,int(truth.sum()))),
            "none_recall":float(tn/max(1,int((~truth).sum()))),
            "any_good_true_positive":tp,"any_good_false_negative":fn,
            "none_true_negative":tn,"none_false_positive":fp,
            "any_good_states":int(truth.sum()),"none_states":int((~truth).sum())}


def rank_metrics(records):
    positive=[record for record in records if record["any_good"]]
    ranks=[record["best_good_rank"] for record in positive]
    wins=sum(record["pairwise_wins"] for record in positive)
    pairs=sum(record["pairwise_pairs"] for record in positive)
    return {"positive_states":len(positive),"top_good_states":sum(rank==1 for rank in ranks),
            "top_good_rate":float(sum(rank==1 for rank in ranks)/max(1,len(ranks))),
            "best_good_rank_histogram":dict(sorted(Counter(ranks).items())),
            "best_good_rank_median":float(np.median(ranks)) if ranks else None,
            "best_good_rank_p90":float(np.quantile(ranks,.9)) if ranks else None,
            "pairwise_good_above_bad":wins,"pairwise_pairs":pairs,
            "pairwise_accuracy":float(wins/max(1,pairs))}


def build_records(grouped,scores):
    records=[]
    for state_id,examples in grouped.items():
        logits=np.asarray(scores[state_id]["logits"]);none=float(scores[state_id]["none_logit"])
        good=np.asarray([bool(row["target_good"]) for row in examples])
        order=np.lexsort((np.arange(len(examples)),-logits));rank=np.empty(len(examples),int);rank[order]=np.arange(1,len(examples)+1)
        any_score=float(np.logaddexp.reduce(logits));predict_any=any_score>none
        record={"state_id":state_id,"system":examples[0]["system"],"case_id":examples[0]["case_id"],
                "split":examples[0]["split"],"evaluation_count":examples[0]["evaluation_count"],
                "any_good":bool(good.any()),"predict_any":bool(predict_any),
                "any_logit":any_score,"none_logit":none,"gate_margin":any_score-none,
                "candidate_count":len(examples),"good_count":int(good.sum())}
        if good.any():
            good_ids=np.flatnonzero(good);bad_ids=np.flatnonzero(~good)
            record["best_good_rank"]=int(rank[good_ids].min())
            record["best_good_minus_best_bad"]=float(logits[good_ids].max()-logits[bad_ids].max())
            record["pairwise_wins"]=int(sum(logits[g]>logits[b] for g in good_ids for b in bad_ids))
            record["pairwise_pairs"]=int(len(good_ids)*len(bad_ids))
        records.append(record)
    return records


def hard_metrics(rows,grouped,scores,hard_tags,tag,good_tag):
    selected=[]
    for state_id,examples in grouped.items():
        logits=np.asarray(scores[state_id]["logits"]);none=float(scores[state_id]["none_logit"])
        good=np.asarray([bool(row["target_good"]) for row in examples])
        for index,row in enumerate(examples):
            if tag not in hard_tags.get(row["example_id"],()):continue
            selected.append((state_id,index,row,logits,none,good))
    if good_tag:
        above_null=sum(logits[index]>none for _,index,_,logits,none,_ in selected)
        above_all_bad=sum(logits[index]>logits[~good].max() for _,index,_,logits,_,good in selected)
        wins=sum(int(logits[index]>logits[bad]) for _,index,_,logits,_,good in selected for bad in np.flatnonzero(~good))
        pairs=sum(int((~good).sum()) for *_,good in selected)
        return {"examples":len(selected),"above_null":above_null,"above_null_rate":above_null/max(1,len(selected)),
                "outranks_every_bad":above_all_bad,"outranks_every_bad_rate":above_all_bad/max(1,len(selected)),
                "pairwise_wins":wins,"pairwise_pairs":pairs,"pairwise_accuracy":wins/max(1,pairs)}
    rejected=0;individual=0;selected_top=0
    for _,index,_,logits,none,good in selected:
        individual+=int(logits[index]<=none)
        if good.any():
            rejected+=int(logits[index]<logits[good].max())
            selected_top+=int(index==int(np.argmax(logits)))
        else:
            rejected+=int(float(np.logaddexp.reduce(logits))<=none)
    return {"examples":len(selected),"individual_below_null":individual,
            "individual_below_null_rate":individual/max(1,len(selected)),
            "rejected_by_state_decision":rejected,
            "rejected_by_state_decision_rate":rejected/max(1,len(selected)),
            "selected_as_top_in_positive_state":selected_top}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--evaluate-only",action="store_true");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);before=hashes()
    paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={path.name:sha256(path) for path in paths}
    rows=[row for row in json.loads((DATA/"dataset.json").read_text()) if row["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    metadata={row["state_id"]:row for row in json.loads((DATA/"states.json").read_text())}
    states={state_id:GraphState(metadata[state_id],examples) for state_id,examples in grouped.items()}
    hard_tags=defaultdict(set)
    with (DATA/"hard_cases.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);hard_tags[row["example_id"]].update(row["hard_case_tags"])
    torch.manual_seed(SEED+320);np.random.seed(SEED);random.seed(SEED)
    model=StateRelativeCandidateModel()
    checkpoint=OUT/"candidate_selection_state_relative_screen_only.pt"
    if args.evaluate_only:
        saved=torch.load(checkpoint,weights_only=False);model.load_state_dict(saved["model_state"])
        history=[]
        log=OUT/"run.log"
        if log.exists():
            for line in log.read_text().splitlines():
                if line.startswith('{"epoch"'):history.append(json.loads(line))
        fit={"history":history,"best_epoch":int(saved["best_epoch"]),"seconds":None,
             "evaluation_only_reused_saved_best_checkpoint":True}
    else:
        fit=train(model,states,grouped,args.epochs)
    scores=state_scores(model,states,grouped);records=build_records(grouped,scores)
    evaluation={}
    for split in ("train","held_out","fresh_test","all"):
        selected=records if split=="all" else [record for record in records if record["split"]==split]
        evaluation[split]={"gate":gate_metrics(selected),"ranking":rank_metrics(selected),
                           "by_system":{system:{"gate":gate_metrics([r for r in selected if r["system"]==system]),
                                                "ranking":rank_metrics([r for r in selected if r["system"]==system])}
                                        for system in ("reflection_ball","kepler5")}}
    persistent=[record for record in records if record["system"]=="kepler5" and int(record["case_id"]) in PERSISTENT]
    hard={
        "useful_pareto_dominated_candidate":hard_metrics(rows,grouped,scores,hard_tags,"useful_pareto_dominated_candidate",True),
        "useful_non_hull_candidate":hard_metrics(rows,grouped,scores,hard_tags,"useful_non_hull_candidate",True),
        "low_loss_bad_alias":hard_metrics(rows,grouped,scores,hard_tags,"low_loss_bad_alias",False),
    }
    if not args.evaluate_only:
        torch.save({"status":"ARCHITECTURE_SCREEN_ONLY_NOT_PROMOTED","task":"candidate_selection",
                    "supervision":"NULL_AUGMENTED_MULTI_POSITIVE_LISTWISE","depth":3,"width":32,
                    "parameters":parameters(model),"best_epoch":fit["best_epoch"],
                    "model_state":model.state_dict(),"schema_version":1},checkpoint)
    after=hashes();changed=[path for path in set(before)&set(after)
                             if before[path]!=after[path] and not path.startswith(str(OUT))]
    data_after={path.name:sha256(path) for path in paths}
    report={"status":"STATE_RELATIVE_SCREEN_ONLY_NOT_PROMOTED","architecture":{
                "width":32,"depth":3,"parameters":parameters(model),"new_parameters_for_null_action":1,
                "loss":"one state-balanced null-augmented multi-positive listwise loss",
                "global_pooling":False},"fit":fit,"evaluation":evaluation,
            "persistent_kepler":{"gate":gate_metrics(persistent),"ranking":rank_metrics(persistent)},
            "hard_subsets":hard,
            "previous_depth3_binary":json.loads(OLD_AUDIT.read_text())["models"]["depth_3"],
            "checkpoint":str(checkpoint),"dataset_hashes_before":data_before,
            "dataset_hashes_after":data_after,"dataset_unchanged":data_before==data_after,
            "protected_checkpoint_changes":changed,"checkpoint_hashes_unchanged":not changed}
    report=native(report)
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
