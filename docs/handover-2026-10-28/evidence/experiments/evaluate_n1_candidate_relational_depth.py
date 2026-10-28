#!/usr/bin/env python3
"""Inference-only, apples-to-apples diagnostics for the relational-depth screen."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from experiments.diagnose_n1_candidate_kepler_failures import capture as baseline_capture
from experiments.train_n1_candidate_relational_depth_screen import (
    BASELINE, DATA, OUT, DeepCandidateModel, GraphState, PERSISTENT,
    binary_metrics, ranking_metrics,
)
from experiments.train_n1_relational_architecture_screen import RelationalQueryModel, atomic_json


def cosine(left, right):
    denominator=np.linalg.norm(left)*np.linalg.norm(right)
    return float(np.dot(left,right)/denominator) if denominator else 1.0


def geometry(captures, grouped, stage):
    distances=[];cosines=[];centroids=[]
    for state_id,examples in grouped.items():
        value=captures[state_id][stage]
        good=np.asarray([bool(row["target_good"]) for row in examples])
        pairwise=np.linalg.norm(value[:,None]-value[None,:],axis=-1)
        positive=pairwise[np.triu_indices(len(value),1)]
        scale=float(np.median(positive[positive>0])) if np.any(positive>0) else 1.0
        for index in np.flatnonzero(good):
            bad=np.flatnonzero(~good); nearest=bad[int(np.argmin(pairwise[index,bad]))]
            distances.append(float(pairwise[index,nearest]/scale))
            cosines.append(cosine(value[index],value[nearest]))
        centroids.append(float(np.linalg.norm(value[good].mean(0)-value[~good].mean(0))/scale))
    return {"good_count":len(distances),
            "nearest_bad_distance_over_state_median":float(np.median(distances)),
            "nearest_bad_cosine":float(np.median(cosines)),
            "centroid_gap_over_state_median":float(np.median(centroids))}


@torch.no_grad()
def evaluate(name,model,states,rows,hard_grouped,is_baseline):
    model.eval(); grouped=defaultdict(list); score_by_id={}; captures={}
    for row in rows: grouped[row["state_id"]].append(row)
    for state_id,examples in grouped.items():
        state=states[state_id]
        if is_baseline:
            logits=model(state,examples)
        else:
            logits,_=model(state,examples,return_stages=True)
        probability=torch.sigmoid(logits).cpu().numpy()
        for row,value in zip(examples,probability): score_by_id[row["example_id"]]=float(value)
        if state_id in hard_grouped:
            hard=hard_grouped[state_id]
            if is_baseline:
                captures[state_id]=baseline_capture(model,state,hard)
            else:
                _,stage=model(state,hard,return_stages=True)
                captures[state_id]={key:value.cpu().numpy() for key,value in stage.items()}
    scores=[score_by_id[row["example_id"]] for row in rows]
    result={"kepler_all":binary_metrics(rows,scores),"kepler_ranking":ranking_metrics(rows,scores)}
    for split in ("train","held_out","fresh_test"):
        selected=[(row,value) for row,value in zip(rows,scores) if row["split"]==split]
        result[split]={"classification":binary_metrics([r for r,_ in selected],[v for _,v in selected]),
                       "ranking":ranking_metrics([r for r,_ in selected],[v for _,v in selected])}
    persistent=[(row,value) for row,value in zip(rows,scores) if int(row["case_id"]) in PERSISTENT]
    persistent_good=[(row,value) for row,value in persistent if row["target_good"]]
    result["persistent"]={
        "good_classification":binary_metrics([r for r,_ in persistent_good],[v for _,v in persistent_good]),
        "mixed_state_ranking":ranking_metrics([r for r,_ in persistent],[v for _,v in persistent]),
        "good_probability":{"minimum":min(v for _,v in persistent_good),
                            "median":float(np.median([v for _,v in persistent_good])),
                            "maximum":max(v for _,v in persistent_good)},
    }
    stage="prehead"
    result["hard_kepler_prehead_geometry"]=geometry(captures,hard_grouped,stage)
    return result


def main():
    all_rows=json.loads((DATA/"dataset.json").read_text())
    rows=[row for row in all_rows if row["task"]=="candidate_selection" and row["system"]=="kepler5"]
    grouped=defaultdict(list)
    for row in rows:grouped[row["state_id"]].append(row)
    hard_grouped={state_id:values for state_id,values in grouped.items()
                  if any(r["target_good"] for r in values) and any(not r["target_good"] for r in values)
                  and (any(r["split"] in ("held_out","fresh_test") and r["target_good"] for r in values)
                       or any(int(r["case_id"]) in PERSISTENT and r["target_good"] for r in values))}
    metadata={row["state_id"]:row for row in json.loads((DATA/"states.json").read_text())}
    states={state_id:GraphState(metadata[state_id],grouped[state_id]) for state_id in grouped}
    checkpoint=torch.load(BASELINE,weights_only=False)
    baseline=RelationalQueryModel(int(checkpoint["width"]),"candidate_selection")
    baseline.load_state_dict(checkpoint["model_state"])
    report={"baseline":evaluate("baseline",baseline,states,rows,hard_grouped,True),"depths":{}}
    for depth in (1,2,3):
        checkpoint=torch.load(OUT/f"candidate_selection_depth{depth}_screen_only.pt",weights_only=False)
        model=DeepCandidateModel(int(checkpoint["width"]),depth)
        model.load_state_dict(checkpoint["model_state"])
        report["depths"][str(depth)]=evaluate(f"depth{depth}",model,states,rows,hard_grouped,False)
    atomic_json(OUT/"kepler_comparable_diagnostics.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
