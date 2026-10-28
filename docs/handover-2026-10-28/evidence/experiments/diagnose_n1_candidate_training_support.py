#!/usr/bin/env python3
"""Read-only training-support audit for hardest persistent Kepler candidates."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import DATA, DeepCandidateModel, GraphState
from experiments.train_n1_relational_architecture_screen import atomic_json


OUT = Path("outputs/n1_candidate_training_support_diagnosis")
CHECKPOINT = Path("outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
BASE_REPORT = Path("outputs/n1_candidate_top_of_list_screen/report.json")
STAGES = ("encoded", "relational_1", "relational_3", "ordered_neighbor", "prehead")


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [native(v) for v in value]
    return value


def cosine(left, right):
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 1.0


@torch.no_grad()
def encode_corpus(model, grouped, metadata):
    records=[]; stage_values={name:[] for name in STAGES}
    for number,state_id in enumerate(sorted(grouped),1):
        examples=grouped[state_id];state=GraphState(metadata[state_id],examples)
        logits,stages=model(state,examples,return_stages=True)
        for index,row in enumerate(examples):
            records.append(dict(example_id=row["example_id"],state_id=state_id,
                system=row["system"],case_id=int(row["case_id"]),split=row["split"],
                evaluation_count=int(row["evaluation_count"]),entity_index=int(row["entity_index"]),
                entity_source=row["entity_source"],routing=row["current_routing_status_name"],
                unit_point=[float(value) for value in row["unit_point"]],
                theta=[float(value) for value in row["theta"]],owner_site=int(row["owner_site"]),
                target_good=bool(row["target_good"]),model_score=float(logits[index])))
            for name in STAGES: stage_values[name].append(stages[name][index].cpu().numpy())
        if number % 25 == 0: print(json.dumps({"encoded_states":number,"total":len(grouped)}),flush=True)
    arrays={name:np.asarray(values,np.float64) for name,values in stage_values.items()}
    # Concatenation retains every audited stage rather than pooling them.
    arrays["multilevel"]=np.concatenate([arrays[name] for name in STAGES],axis=1)
    return records,arrays


def support_reference(records, arrays):
    labels=np.asarray([r["target_good"] for r in records],bool)
    states=np.asarray([r["state_id"] for r in records],object)
    output={}
    for stage,values in arrays.items():
        distance=cdist(values,values)
        same_state=states[:,None]==states[None,:]
        distance[same_state]=np.inf
        any_nearest=np.min(distance,axis=1)
        class_nearest=np.empty(len(records))
        for index,label in enumerate(labels):
            valid=(labels==label)&(~same_state[index])
            class_nearest[index]=np.min(distance[index,valid])
        def q(values):
            return {str(key):float(np.quantile(values,key)) for key in (.5,.9,.95,.99)}|{"max":float(np.max(values))}
        output[stage]={"cross_state_nearest_any":q(any_nearest),
                       "cross_state_nearest_same_label":q(class_nearest),
                       "good_same_label":q(class_nearest[labels]),
                       "bad_same_label":q(class_nearest[~labels])}
    return output


def neighbor_record(query_index, train_index, records, arrays, stage):
    query=arrays[stage][query_index];neighbor=arrays[stage][train_index]
    row=records[train_index]
    return dict(example_id=row["example_id"],state_id=row["state_id"],system=row["system"],
        case_id=row["case_id"],split=row["split"],evaluation_count=row["evaluation_count"],
        target_good=row["target_good"],entity_source=row["entity_source"],routing=row["routing"],
        euclidean_distance=float(np.linalg.norm(query-neighbor)),cosine_similarity=cosine(query,neighbor))


def query_support(query_index, records, arrays, train_indices, references):
    labels=np.asarray([r["target_good"] for r in records],bool)
    query=records[query_index]; result={}
    for stage,values in arrays.items():
        eligible=np.asarray([i for i in train_indices if records[i]["state_id"]!=query["state_id"]],int)
        distances=np.linalg.norm(values[eligible]-values[query_index][None],axis=1)
        order=eligible[np.argsort(distances,kind="stable")]
        good=order[labels[order]];bad=order[~labels[order]]
        nearest12=order[:12]; good_count=int(labels[nearest12].sum())
        any_distance=float(np.linalg.norm(values[query_index]-values[order[0]]))
        same=good if query["target_good"] else bad
        same_distance=float(np.linalg.norm(values[query_index]-values[same[0]]))
        ref=references[stage]["good_same_label" if query["target_good"] else "bad_same_label"]
        if any_distance>references[stage]["cross_state_nearest_any"]["max"]: density="OUTSIDE_TRAIN_ENVELOPE"
        elif same_distance>ref["0.95"]: density="SPARSE_SAME_LABEL"
        elif good_count>=8: density="GOOD_DOMINATED"
        elif good_count<=4: density="BAD_DOMINATED"
        else: density="MIXED"
        result[stage]=dict(
            nearest_training_good=[neighbor_record(query_index,int(i),records,arrays,stage) for i in good[:5]],
            nearest_training_bad=[neighbor_record(query_index,int(i),records,arrays,stage) for i in bad[:5]],
            nearest12_good=good_count,nearest12_bad=12-good_count,
            nearest_any_distance=any_distance,nearest_same_label_distance=same_distance,
            same_label_distance_over_p95=float(same_distance/ref["0.95"]),support=density,
            nearest_good_from_other_system=next((neighbor_record(query_index,int(i),records,arrays,stage)
                for i in good if records[i]["system"]!=query["system"]),None),
            nearest_good_from_other_case=next((neighbor_record(query_index,int(i),records,arrays,stage)
                for i in good if records[i]["case_id"]!=query["case_id"] or records[i]["system"]!=query["system"]),None))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);before=hashes()
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    model=DeepCandidateModel(32,3);model.load_state_dict(torch.load(CHECKPOINT,weights_only=False)["model_state"]);model.eval()
    records,arrays=encode_corpus(model,grouped,metadata)
    index={r["example_id"]:i for i,r in enumerate(records)}
    train_indices=[i for i,r in enumerate(records) if r["split"]=="train"]
    train_records=[records[i] for i in train_indices]
    train_arrays={stage:values[train_indices] for stage,values in arrays.items()}
    # References are calculated in a pure training index space.
    references=support_reference(train_records,train_arrays)
    baseline=json.loads(BASE_REPORT.read_text())
    failures=baseline["persistent_kepler"]["remaining_bad_wins"]
    states=[]
    for failure in failures:
        state_id=failure["state_id"];examples=grouped[state_id]
        candidates=[index[row["example_id"]] for row in examples]
        good=[i for i in candidates if records[i]["target_good"]]
        bad=[i for i in candidates if not records[i]["target_good"]]
        best_good=max(good,key=lambda i:records[i]["model_score"])
        winning_bad=max(bad,key=lambda i:records[i]["model_score"])
        states.append(dict(state_id=state_id,case_id=int(failure["case_id"]),split=failure["split"],
            evaluation_count=int(failure["evaluation_count"]),
            good_minus_bad_margin=float(records[best_good]["model_score"]-records[winning_bad]["model_score"]),
            best_good={**records[best_good],"exact_member_of_training_corpus":records[best_good]["split"]=="train",
                       "support":query_support(best_good,records,arrays,train_indices,references)},
            winning_bad={**records[winning_bad],"exact_member_of_training_corpus":records[winning_bad]["split"]=="train",
                         "support":query_support(winning_bad,records,arrays,train_indices,references)}))
    after=hashes()
    report=native(dict(status="READ_ONLY_TRAINING_SUPPORT_DIAGNOSIS",
        representation_levels={"encoded":"candidate-local encoding plus owner/dominator message",
            "relational_1":"one owner/cell/site relation pass","relational_3":"full depth-3 hull/Pareto/dominator/history relational context",
            "ordered_neighbor":"candidate plus explicit ordered top-24 neighborhood",
            "prehead":"complete frozen snapshot-model candidate representation",
            "multilevel":"lossless concatenation of all five audited 32-dimensional representations"},
        scope=dict(failed_states=len(states),training_examples=len(train_indices),training_states=len(set(r["state_id"] for r in train_records))),
        empirical_training_support=references,states=states,
        methodology=dict(neighbor_search="exact Euclidean nearest neighbors in each frozen representation; cosine also reported",
            exclusions="the query state is excluded, including for train-split failures; exact membership is reported separately",
            no_state_pooling=True,no_training=True),
        dataset_hashes={p.name:sha256(p) for p in (DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl")},
        checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),
        protected_checkpoint_changes=[p for p in set(before)&set(after) if before[p]!=after[p]],checkpoint_hashes_unchanged=before==after))
    atomic_json(OUT/"report.json",report)
    np.savez_compressed(OUT/"embedding_cache.npz",**{stage:values for stage,values in arrays.items()})
    print(json.dumps({"scope":report["scope"],"states":[{"state_id":s["state_id"],"margin":s["good_minus_bad_margin"],
        "good_multilevel":{k:s["best_good"]["support"]["multilevel"][k] for k in ("nearest12_good","nearest12_bad","same_label_distance_over_p95","support")},
        "bad_multilevel":{k:s["winning_bad"]["support"]["multilevel"][k] for k in ("nearest12_good","nearest12_bad","same_label_distance_over_p95","support")}}
        for s in states],"checkpoint_hashes_unchanged":report["checkpoint_hashes_unchanged"]},indent=2))


if __name__=="__main__":main()
