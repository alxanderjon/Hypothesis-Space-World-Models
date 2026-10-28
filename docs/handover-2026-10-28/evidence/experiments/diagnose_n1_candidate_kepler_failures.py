#!/usr/bin/env python3
"""Read-only diagnosis of hard Kepler candidate-selection failures."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.stats import spearmanr
import torch

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_relational_architecture_screen import (
    DATA, RaggedState, RelationalQueryModel,
)


torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_kepler_failure_diagnosis")
SCREEN = Path("outputs/n1_relational_architecture_screen/candidate_selection_screen_only.pt")
PERSISTENT = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / scale) if scale else 0.0


@torch.no_grad()
def capture(model: RelationalQueryModel, state: RaggedState, examples: list[dict],
            ablate: set[str] | None = None) -> dict:
    ablate = ablate or set()
    site_coordinates = model.encode_coordinates(state.site_coords)
    candidate_coordinates = model.encode_coordinates(state.candidate_coords)
    site_feature_values = state.site_features.clone()
    candidate_feature_values = state.candidate_features.clone()
    if "owner_loss" in ablate:
        site_feature_values[:, 0:2] = 0
    if "pareto" in ablate:
        site_feature_values[:, 3:5] = 0
    if "hull" in ablate:
        site_feature_values[:, 5:8] = 0
    if "surrogate" in ablate:
        candidate_feature_values[:, 1] = 0
    if "production_score" in ablate:
        candidate_feature_values[:, 2:12] = 0

    site_pre = site_coordinates + model.site_features(site_feature_values)
    site = site_pre.clone()
    valid = state.dominator >= 0
    if bool(valid.any()) and "dominator" not in ablate:
        message = torch.zeros_like(site)
        message[valid] = model.dominator_message(site[state.dominator[valid]])
        site = site + message
    site = model.site_norm(site)

    candidate_pre_owner = (candidate_coordinates
                           + model.candidate_features(candidate_feature_values))
    candidate = candidate_pre_owner.clone()
    if "owner" not in ablate:
        candidate = candidate + model.owner_message(site[state.owner])
    candidate = model.candidate_norm(candidate)

    indices = torch.tensor([int(row["entity_index"]) for row in examples], dtype=torch.long)
    raw = []
    for index in indices.tolist():
        owner = int(state.owner[index])
        raw.append(torch.cat((state.candidate_coords[index], state.candidate_features[index],
                              state.site_coords[owner], state.site_features[owner])))
    raw = torch.stack(raw)
    coordinate = model.candidate_norm(
        candidate_coordinates[indices] + model.owner_message(site_coordinates[state.owner[indices]])
    )
    base = model.candidate_norm(candidate_pre_owner[indices])
    relation = candidate[indices]
    query = relation

    messages = []
    for position, index in enumerate(indices.tolist()):
        neighbor_index, distance = state.query_neighbors[index]
        if "neighbor" in ablate:
            messages.append(torch.zeros_like(query[position])); continue
        rank = torch.arange(len(neighbor_index), dtype=torch.float32) / max(1, len(neighbor_index)-1)
        features = torch.stack((torch.log1p(distance), rank), dim=-1)
        bias = model.neighbor_bias(features).squeeze(-1)[None]
        messages.append(model.neighbor_attention(
            query[position:position+1], site[neighbor_index], bias)[0])
    neighbor = model.update_norms[0](query + torch.stack(messages))
    query = neighbor
    site_message = model.site_attention(query, site)
    if "site_attention" in ablate:
        site_message = torch.zeros_like(site_message)
    site_context = model.update_norms[1](query + site_message)
    query = site_context
    candidate_message = model.candidate_attention(query, candidate)
    if "candidate_attention" in ablate:
        candidate_message = torch.zeros_like(candidate_message)
    candidate_context = model.update_norms[2](query + candidate_message)
    query = candidate_context
    state_values = torch.tensor([
        math.log1p(state.eval_count), math.log1p(state.site_count),
        math.log1p(state.candidate_count), math.log1p(state.dimension),
    ], dtype=torch.float32)
    state_message = model.state_features(state_values)[None]
    if "state_features" in ablate:
        state_message = torch.zeros_like(state_message)
    prehead = model.update_norms[3](query + state_message)
    logits = model.output(prehead).squeeze(-1)
    return {name: value.cpu().numpy() for name, value in {
        "raw": raw, "coordinate": coordinate, "candidate_base": base,
        "relation": relation, "neighbor": neighbor,
        "site_attention": site_context, "candidate_attention": candidate_context,
        "prehead": prehead, "logit": logits,
        "probability": torch.sigmoid(logits),
    }.items()}


def state_margin(examples: list[dict], logits: np.ndarray) -> dict:
    good = logits[[bool(row["target_good"]) for row in examples]]
    bad = logits[[not bool(row["target_good"]) for row in examples]]
    return {
        "best_good_minus_best_bad": float(np.max(good)-np.max(bad)),
        "mean_good_minus_mean_bad": float(np.mean(good)-np.mean(bad)),
    }


def stage_geometry(examples_by_state, captures, stages) -> dict:
    result = {}
    for stage in stages:
        nearest_bad_cos=[]; nearest_bad_distance=[]; normalized_nearest=[]
        centroid=[]; nearest_label_bad=0; good_count=0
        for state_id, examples in examples_by_state.items():
            representation=captures[state_id][stage]
            good=np.asarray([bool(row["target_good"]) for row in examples])
            if not good.any() or good.all(): continue
            pairwise=np.linalg.norm(representation[:,None,:]-representation[None,:,:],axis=-1)
            positive=pairwise[np.triu_indices(len(examples),1)]
            scale=float(np.median(positive[positive>0])) if np.any(positive>0) else 1.0
            for index in np.flatnonzero(good):
                bad_ids=np.flatnonzero(~good); distances=pairwise[index,bad_ids]
                nearest=bad_ids[int(np.argmin(distances))]
                nearest_bad_distance.append(float(pairwise[index,nearest]))
                normalized_nearest.append(float(pairwise[index,nearest]/scale))
                nearest_bad_cos.append(cosine(representation[index],representation[nearest]))
                other=np.delete(np.arange(len(examples)),index)
                nearest_any=other[int(np.argmin(pairwise[index,other]))]
                nearest_label_bad += int(not good[nearest_any]);good_count+=1
            centroid.append(float(np.linalg.norm(representation[good].mean(0)-representation[~good].mean(0))/scale))
        def dist(values):
            a=np.asarray(values,float)
            return {"count":len(a),"median":float(np.median(a)),"p10":float(np.quantile(a,.1)),
                    "p90":float(np.quantile(a,.9)),"minimum":float(a.min()),"maximum":float(a.max())}
        result[stage]={"nearest_bad_cosine":dist(nearest_bad_cos),
                       "nearest_bad_distance":dist(nearest_bad_distance),
                       "nearest_bad_distance_over_state_median":dist(normalized_nearest),
                       "centroid_gap_over_state_median":dist(centroid),
                       "good_whose_nearest_alternative_is_bad":nearest_label_bad,
                       "good_count":good_count}
    return result


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);before=hashes()
    rows=json.loads((DATA/"dataset.json").read_text())
    rows=[row for row in rows if row["task"]=="candidate_selection" and row["system"]=="kepler5"]
    grouped_all=defaultdict(list)
    for row in rows:grouped_all[row["state_id"]].append(row)
    relevant={state_id:values for state_id,values in grouped_all.items()
              if any(r["target_good"] for r in values) and any(not r["target_good"] for r in values)
              and (any(r["split"] in ("held_out","fresh_test") and r["target_good"] for r in values)
                   or any(int(r["case_id"]) in PERSISTENT and r["target_good"] for r in values))}
    metadata={row["state_id"]:row for row in json.loads((DATA/"states.json").read_text())}
    states={state_id:RaggedState(metadata[state_id],grouped_all[state_id]) for state_id in relevant}
    checkpoint=torch.load(SCREEN,weights_only=False)
    model=RelationalQueryModel(int(checkpoint["width"]),"candidate_selection")
    model.load_state_dict(checkpoint["model_state"]);model.eval()

    captures={};detail=[];state_rows=[]
    for state_id,examples in relevant.items():
        state=states[state_id];values=capture(model,state,examples);captures[state_id]=values
        full_margin=state_margin(examples,values["logit"])
        state_rows.append({"state_id":state_id,"system":"kepler5","case_id":examples[0]["case_id"],
                           "split":examples[0]["split"],"evaluation_count":examples[0]["evaluation_count"],
                           "candidates":len(examples),"good":sum(r["target_good"] for r in examples),
                           "bad":sum(not r["target_good"] for r in examples),**full_margin})
        for position,row in enumerate(examples):
            index=int(row["entity_index"]);owner=int(state.owner[index])
            item={"state_id":state_id,"case_id":row["case_id"],"split":row["split"],
                  "evaluation_count":row["evaluation_count"],"example_id":row["example_id"],
                  "candidate_index":index,"owner_index":owner,
                  "candidate_coordinates_normalized":json.dumps(
                      [float(value) for value in state.candidate_coords[index].tolist()],
                      separators=(",", ":")),
                  "target_good":bool(row["target_good"]),
                  "routing":row["current_routing_status_name"],
                  "logit":float(values["logit"][position]),"probability":float(values["probability"][position]),
                  "owner_loss":float(np.expm1(abs(float(state.site_features[owner,0]))) * np.sign(float(state.site_features[owner,0]))),
                  "owner_encoded_loss_feature":float(state.site_features[owner,1]),
                  "owner_pareto_dominated":bool(state.site_features[owner,3]),
                  "owner_hull_member":bool(state.site_features[owner,5]),
                  "owner_selected":bool(state.site_features[owner,8]),
                  "owner_distance_feature":float(state.candidate_features[index,0]),
                  "surrogate_feature":float(state.candidate_features[index,1]),
                  "production_score_feature":float(state.candidate_features[index,3]),
                  "production_score_defined":bool(state.candidate_features[index,4]),
                  "production_rank_feature":float(state.candidate_features[index,5])}
            for stage in ("raw","coordinate","candidate_base","relation","neighbor",
                          "site_attention","candidate_attention","prehead"):
                item[f"{stage}_norm"]=float(np.linalg.norm(values[stage][position]))
            detail.append(item)

    # Frozen inference-only ablations. Positive contribution means the component
    # improved the GOOD-vs-BAD logit margin in the complete model.
    ablations=("owner_loss","pareto","hull","surrogate","production_score",
               "dominator","owner","neighbor","site_attention",
               "candidate_attention","state_features")
    attribution={name:[] for name in ablations}
    for state_id,examples in relevant.items():
        full=state_margin(examples,captures[state_id]["logit"])
        for name in ablations:
            ablated=capture(model,states[state_id],examples,{name})
            margin=state_margin(examples,ablated["logit"])
            attribution[name].append({key:full[key]-margin[key] for key in full})
    attribution_summary={}
    for name,values in attribution.items():
        attribution_summary[name]={key:{
            "mean":float(np.mean([v[key] for v in values])),
            "median":float(np.median([v[key] for v in values])),
            "helps_states":int(sum(v[key]>0 for v in values)),
            "hurts_states":int(sum(v[key]<0 for v in values)),
        } for key in ("best_good_minus_best_bad","mean_good_minus_mean_bad")}

    stages=("raw","coordinate","candidate_base","relation","neighbor",
            "site_attention","candidate_attention","prehead")
    geometry=stage_geometry(relevant,captures,stages)
    good=[row for row in detail if row["target_good"]];bad=[row for row in detail if not row["target_good"]]
    def score_summary(selected):
        p=np.asarray([row["probability"] for row in selected]);l=np.asarray([row["logit"] for row in selected])
        return {"count":len(selected),"logit":{"minimum":float(l.min()),"median":float(np.median(l)),"maximum":float(l.max())},
                "probability":{"minimum":float(p.min()),"median":float(np.median(p)),"maximum":float(p.max())},
                "classified_good":int(np.sum(p>=.5)),"classified_bad":int(np.sum(p<.5)),
                "confident_bad_probability_below_0_25":int(np.sum(p<.25))}
    score_distributions={"good":score_summary(good),"bad":score_summary(bad),
                         "held_out_good":score_summary([r for r in good if r["split"]=="held_out"]),
                         "fresh_good":score_summary([r for r in good if r["split"]=="fresh_test"]),
                         "persistent_good":score_summary([r for r in good if int(r["case_id"]) in PERSISTENT])}

    # N1-visible feature associations and matched pair directionality.
    fields=("owner_encoded_loss_feature","owner_pareto_dominated","owner_hull_member",
            "owner_selected","owner_distance_feature","surrogate_feature",
            "production_score_feature","production_rank_feature")
    visible={}
    label=np.asarray([row["target_good"] for row in detail],float)
    logits=np.asarray([row["logit"] for row in detail],float)
    for field in fields:
        values=np.asarray([float(row[field]) for row in detail])
        corr_logit=float(spearmanr(values,logits).statistic)
        corr_label=float(spearmanr(values,label).statistic)
        visible[field]={"good_median":float(np.median(values[label==1])),
                        "bad_median":float(np.median(values[label==0])),
                        "spearman_with_model_logit":corr_logit if math.isfinite(corr_logit) else None,
                        "spearman_with_label":corr_label if math.isfinite(corr_label) else None}
    pair_direction=defaultdict(lambda:dict(good_less=0,good_equal=0,good_greater=0,pairs=0))
    for state_id,examples in relevant.items():
        rows_here=[r for r in detail if r["state_id"]==state_id]
        for left in [r for r in rows_here if r["target_good"]]:
            for right in [r for r in rows_here if not r["target_good"]]:
                for field in fields:
                    pair_direction[field]["pairs"]+=1
                    key="good_less" if left[field]<right[field] else "good_greater" if left[field]>right[field] else "good_equal"
                    pair_direction[field][key]+=1

    # Descriptive, non-gating misranking characterization.
    for row in state_rows:
        gap=row["best_good_minus_best_bad"]
        row["ranking"]="GOOD_ABOVE_BAD" if gap>0 else "NEAR_TIE" if gap>=-0.05 else "BAD_ABOVE_GOOD"
        state_detail=[r for r in detail if r["state_id"]==row["state_id"]]
        best_good=max(r["probability"] for r in state_detail if r["target_good"])
        row["good_classification"]=("AT_LEAST_ONE_GOOD_CLASSIFIED_GOOD" if best_good>=.5 else
                                    "ALL_GOOD_CONFIDENT_BAD" if best_good<.25 else "ALL_GOOD_MODERATE_BAD")

    with (OUT/"candidate_scores.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    atomic_json(OUT/"state_misranking.json",state_rows)
    after=hashes();changed=[path for path in set(before)&set(after) if before[path]!=after[path]]
    report={"status":"DIAGNOSIS_COMPLETE","checkpoint":str(SCREEN),
            "states":len(relevant),"candidates":len(detail),"good":len(good),"bad":len(bad),
            "score_distributions":score_distributions,"state_misranking":{
                "good_above_bad":sum(r["ranking"]=="GOOD_ABOVE_BAD" for r in state_rows),
                "near_tie":sum(r["ranking"]=="NEAR_TIE" for r in state_rows),
                "bad_above_good":sum(r["ranking"]=="BAD_ABOVE_GOOD" for r in state_rows),
                "all_good_confident_bad":sum(r["good_classification"]=="ALL_GOOD_CONFIDENT_BAD" for r in state_rows),
                "all_good_moderate_bad":sum(r["good_classification"]=="ALL_GOOD_MODERATE_BAD" for r in state_rows)},
            "representation_geometry":geometry,"ablation_margin_contributions":attribution_summary,
            "visible_feature_associations":visible,"matched_pair_directionality":pair_direction,
            "state_rows":state_rows,"protected_checkpoint_changes":changed,
            "checkpoint_hashes_unchanged":not changed}
    atomic_json(OUT/"report.json",report);print(json.dumps(report,indent=2))


if __name__=="__main__":main()
