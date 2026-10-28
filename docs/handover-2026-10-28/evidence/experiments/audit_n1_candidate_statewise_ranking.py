#!/usr/bin/env python3
"""Inference-only audit of state-wise candidate ranking supervision."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import (
    BASELINE, DATA, OUT as DEPTH_OUT, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_relational_architecture_screen import RelationalQueryModel, atomic_json


OUT=Path("outputs/n1_candidate_statewise_ranking_audit")


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def model_logits(model,states,grouped):
    model.eval();result={}
    for state_id,examples in grouped.items():
        values=model(states[state_id],examples).cpu().numpy()
        result[state_id]=[float(value) for value in values]
    return result


def distribution(values):
    values=np.asarray(values,float)
    if not len(values): return {"count":0}
    return {"count":len(values),"minimum":float(values.min()),
            "median":float(np.median(values)),"p90":float(np.quantile(values,.9)),
            "maximum":float(values.max())}


def model_state_metrics(grouped,logits):
    result=[]
    for state_id,rows in grouped.items():
        score=np.asarray(logits[state_id]); good=np.asarray([bool(row["target_good"]) for row in rows])
        order=np.lexsort((np.arange(len(rows)),-score)); rank=np.empty(len(rows),int);rank[order]=np.arange(1,len(rows)+1)
        positive=score>=0.0
        item={"state_id":state_id,"top_candidate_good":bool(good[order[0]]),
              "any_good_classified_positive":bool(np.any(positive&good)),
              "any_candidate_classified_positive":bool(np.any(positive)),
              "top_logit":float(score[order[0]])}
        if good.any():
            best_good=int(np.flatnonzero(good)[np.argmax(score[good])])
            best_bad=int(np.flatnonzero(~good)[np.argmax(score[~good])])
            item.update(best_good_rank=int(rank[best_good]),best_good_logit=float(score[best_good]),
                        best_bad_logit=float(score[best_bad]),
                        best_good_minus_best_bad=float(score[best_good]-score[best_bad]),
                        top_good_despite_no_positive=bool(good[order[0]] and not np.any(positive&good)))
        result.append(item)
    return result


def summarize_model(rows):
    positive=[row for row in rows if row.get("best_good_rank") is not None]
    no_good=[row for row in rows if row.get("best_good_rank") is None]
    ranks=Counter(row["best_good_rank"] for row in positive)
    return {"positive_states":len(positive),"top_good_states":sum(row["top_candidate_good"] for row in positive),
            "top_good_rate":sum(row["top_candidate_good"] for row in positive)/len(positive),
            "pairwise_or_listwise_harmless_binary_states":sum(row["top_good_despite_no_positive"] for row in positive),
            "states_with_any_good_classified_positive":sum(row["any_good_classified_positive"] for row in positive),
            "best_good_rank_distribution":dict(sorted(ranks.items())),
            "no_good_states":len(no_good),
            "no_good_states_with_any_false_positive":sum(row["any_candidate_classified_positive"] for row in no_good)}


def main():
    OUT.mkdir(parents=True,exist_ok=True);before=hashes()
    dataset_paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_hash_before={path.name:sha256(path) for path in dataset_paths}
    rows=[row for row in json.loads((DATA/"dataset.json").read_text()) if row["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    metadata={row["state_id"]:row for row in json.loads((DATA/"states.json").read_text())}
    states={state_id:GraphState(metadata[state_id],examples) for state_id,examples in grouped.items()}

    checkpoint=torch.load(BASELINE,weights_only=False)
    baseline=RelationalQueryModel(int(checkpoint["width"]),"candidate_selection")
    baseline.load_state_dict(checkpoint["model_state"])
    models={"baseline":baseline}
    for depth in (2,3):
        checkpoint=torch.load(DEPTH_OUT/f"candidate_selection_depth{depth}_screen_only.pt",weights_only=False)
        model=DeepCandidateModel(int(checkpoint["width"]),depth);model.load_state_dict(checkpoint["model_state"])
        models[f"depth_{depth}"]=model
    logits={name:model_logits(model,states,grouped) for name,model in models.items()}
    model_rows={name:model_state_metrics(grouped,value) for name,value in logits.items()}
    by_model_state={name:{row["state_id"]:row for row in values} for name,values in model_rows.items()}

    state_rows=[]
    for state_id,examples in grouped.items():
        good=sum(bool(row["target_good"]) for row in examples); bad=len(examples)-good
        production=[row for row in examples if row["entity_source"]=="production_selected"]
        item={"state_id":state_id,"system":examples[0]["system"],"case_id":examples[0]["case_id"],
              "split":examples[0]["split"],"evaluation_count":examples[0]["evaluation_count"],
              "candidates":len(examples),"good":good,"bad":bad,
              "production_selected_count":len(production),
              "production_selected_good_count":sum(bool(row["target_good"]) for row in production),
              "production_any_good":any(row["target_good"] for row in production)}
        for name in models:
            values=by_model_state[name][state_id]
            for key,value in values.items():
                if key!="state_id":item[f"{name}_{key}"]=value
        state_rows.append(item)

    positive=[row for row in state_rows if row["good"]>0]; no_good=[row for row in state_rows if row["good"]==0]
    persistent=[row for row in state_rows if row["system"]=="kepler5" and int(row["case_id"]) in PERSISTENT]
    persistent_positive=[row for row in persistent if row["good"]>0]
    production_candidates=[row for row in rows if row["entity_source"]=="production_selected"]
    production_by_state=[[row for row in examples if row["entity_source"]=="production_selected"]
                         for examples in grouped.values()]
    report={
        "status":"STATEWISE_RANKING_AUDIT_ONLY",
        "dataset_hashes_before":data_hash_before,
        "state_counts":{"total":len(state_rows),"at_least_one_good":len(positive),
                        "exactly_one_good":sum(row["good"]==1 for row in state_rows),
                        "multiple_good":sum(row["good"]>1 for row in state_rows),
                        "no_good":len(no_good),"good_count_histogram":dict(sorted(Counter(row["good"] for row in state_rows).items()))},
        "positive_state_alternatives":{"good_per_state":distribution([row["good"] for row in positive]),
                                       "bad_per_state":distribution([row["bad"] for row in positive]),
                                       "joint_histogram":{f"good_{g}_bad_{b}":count for (g,b),count in sorted(Counter((row["good"],row["bad"]) for row in positive).items())}},
        "production_selection":{"retained_candidates_per_state":dict(Counter(row["production_selected_count"] for row in state_rows)),
                                "good_candidates_per_state":dict(Counter(row["production_selected_good_count"] for row in state_rows)),
                                "candidate_level_good":sum(row["target_good"] for row in production_candidates),
                                "candidate_level_total":len(production_candidates),
                                "candidate_level_good_rate":sum(row["target_good"] for row in production_candidates)/len(production_candidates),
                                "first_retained_winner_good":sum(bool(values[0]["target_good"]) for values in production_by_state),
                                "last_retained_winner_good":sum(bool(values[-1]["target_good"]) for values in production_by_state),
                                "positive_states_with_any_production_good":sum(row["production_any_good"] for row in positive),
                                "positive_state_any_good_rate":sum(row["production_any_good"] for row in positive)/len(positive)},
        "models":{name:summarize_model(values) for name,values in model_rows.items()},
        "persistent_kepler":{"total_states":len(persistent),
                             "at_least_one_good":len(persistent_positive),
                             "exactly_one_good":sum(row["good"]==1 for row in persistent),
                             "multiple_good":sum(row["good"]>1 for row in persistent),
                             "no_good":sum(row["good"]==0 for row in persistent),
                             "models":{}},
        "no_good_states":{"total":len(no_good),"by_system":dict(Counter(row["system"] for row in no_good)),
                          "by_split":dict(Counter(row["split"] for row in no_good))},
    }
    for name in models:
        report["persistent_kepler"]["models"][name]={
            "best_good_rank_histogram":dict(sorted(Counter(row[f"{name}_best_good_rank"] for row in persistent_positive).items())),
            "best_good_rank":distribution([row[f"{name}_best_good_rank"] for row in persistent_positive]),
            "best_good_minus_best_bad_logit":distribution([row[f"{name}_best_good_minus_best_bad"] for row in persistent_positive]),
            "top_good_states":sum(row[f"{name}_top_candidate_good"] for row in persistent_positive),
            "top_good_rate":sum(row[f"{name}_top_candidate_good"] for row in persistent_positive)/len(persistent_positive),
        }

    fieldnames=list(state_rows[0])
    with (OUT/"statewise_table.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(state_rows)
    with (OUT/"persistent_kepler_table.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader();writer.writerows(persistent)
    after=hashes();changed=[path for path in set(before)&set(after) if before[path]!=after[path]]
    data_hash_after={path.name:sha256(path) for path in dataset_paths}
    report.update(dataset_hashes_after=data_hash_after,dataset_unchanged=data_hash_before==data_hash_after,
                  protected_checkpoint_changes=changed,checkpoint_hashes_unchanged=not changed)
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
