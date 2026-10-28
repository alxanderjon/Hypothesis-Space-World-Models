#!/usr/bin/env python3
"""Read-only audit of prefix information absent from the N1 candidate state."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import audit_known_structure as known
from experiments.train_n1_candidate_relational_depth_screen import (
    DATA, DeepCandidateModel, GraphState, PERSISTENT,
)
from experiments.train_n1_relational_architecture_screen import atomic_json
from experiments.validate_d5_production_pipeline import _kepler_case


torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
OUT = Path("outputs/n1_candidate_prefix_information_diagnosis")
CHECKPOINT = Path(
    "outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
COMPONENTS = ("x1", "y1", "vx1", "vy1", "x2", "y2", "vx2", "vy2")
POSITION = np.asarray((0, 1, 4, 5)); VELOCITY = np.asarray((2, 3, 6, 7))
BODY1 = np.arange(4); BODY2 = np.arange(4, 8)
M1 = np.float64(1.0); M2 = np.float64(.25); MT = M1 + M2


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_ratio(left, right):
    return float(left / right) if right > 0 else (math.inf if left > 0 else 1.0)


def lag_correlation(values):
    left=values[:-1].reshape(-1); right=values[1:].reshape(-1)
    if np.std(left)==0 or np.std(right)==0: return 1.0
    return float(np.corrcoef(left,right)[0,1])


def adjacent_cosine(values):
    left=values[:-1]; right=values[1:]
    den=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
    valid=den>0
    return float(np.mean(np.sum(left[valid]*right[valid],axis=1)/den[valid])) if valid.any() else 1.0


def cross2(position, velocity):
    return position[...,0]*velocity[...,1]-position[...,1]*velocity[...,0]


def residual_diagnostics(theta, trajectory):
    inputs=np.asarray(trajectory[:127],np.float64); targets=np.asarray(trajectory[1:128],np.float64)
    predicted=known.step(inputs,np.asarray(theta,np.float64),np)
    error=predicted-targets; squared=error*error; time_mse=np.mean(squared,axis=1)
    total=float(np.mean(squared)); rms=math.sqrt(total)
    thirds=np.array_split(np.arange(len(error)),3)
    bias=np.mean(error,axis=0)

    # Relative-body radial/tangential residuals use only observed input geometry.
    relative=inputs[:,4:6]-inputs[:,0:2]
    radius=np.linalg.norm(relative,axis=1); radial=relative/radius[:,None]
    tangent=np.column_stack((-radial[:,1],radial[:,0]))
    rel_pos=error[:,4:6]-error[:,0:2]
    rel_vel=error[:,6:8]-error[:,2:4]
    rpos=np.sum(rel_pos*radial,axis=1); tpos=np.sum(rel_pos*tangent,axis=1)
    rvel=np.sum(rel_vel*radial,axis=1); tvel=np.sum(rel_vel*tangent,axis=1)

    com_pos=(M1*error[:,0:2]+M2*error[:,4:6])/MT
    com_vel=(M1*error[:,2:4]+M2*error[:,6:8])/MT
    predicted_l=M1*cross2(predicted[:,0:2],predicted[:,2:4])+M2*cross2(predicted[:,4:6],predicted[:,6:8])
    target_l=M1*cross2(targets[:,0:2],targets[:,2:4])+M2*cross2(targets[:,4:6],targets[:,6:8])
    angular_error=predicted_l-target_l

    values=dict(
        mse=total,rmse=rms,mean_absolute=float(np.mean(np.abs(error))),
        bias_norm=float(np.linalg.norm(bias)),bias_over_rms=safe_ratio(np.linalg.norm(bias),rms),
        position_mse=float(np.mean(squared[:,POSITION])),velocity_mse=float(np.mean(squared[:,VELOCITY])),
        body1_mse=float(np.mean(squared[:,BODY1])),body2_mse=float(np.mean(squared[:,BODY2])),
        position_velocity_ratio=safe_ratio(np.mean(squared[:,POSITION]),np.mean(squared[:,VELOCITY])),
        body1_body2_ratio=safe_ratio(np.mean(squared[:,BODY1]),np.mean(squared[:,BODY2])),
        early_mse=float(np.mean(time_mse[thirds[0]])),middle_mse=float(np.mean(time_mse[thirds[1]])),
        late_mse=float(np.mean(time_mse[thirds[2]])),late_early_ratio=safe_ratio(np.mean(time_mse[thirds[2]]),np.mean(time_mse[thirds[0]])),
        error_time_centroid=float(np.sum(np.arange(len(error))*time_mse)/max(np.sum(time_mse),np.finfo(float).tiny)/(len(error)-1)),
        peak_step=int(np.argmax(time_mse)),peak_mse=float(np.max(time_mse)),
        lag1_correlation=lag_correlation(error),adjacent_residual_cosine=adjacent_cosine(error),
        temporal_difference_ratio=safe_ratio(np.mean(np.diff(error,axis=0)**2),total),
        com_position_mse=float(np.mean(com_pos**2)),com_velocity_mse=float(np.mean(com_vel**2)),
        relative_position_mse=float(np.mean(rel_pos**2)),relative_velocity_mse=float(np.mean(rel_vel**2)),
        radial_position_mse=float(np.mean(rpos**2)),tangential_position_mse=float(np.mean(tpos**2)),
        radial_velocity_mse=float(np.mean(rvel**2)),tangential_velocity_mse=float(np.mean(tvel**2)),
        angular_momentum_error_mse=float(np.mean(angular_error**2)),
        angular_momentum_error_bias=float(np.mean(angular_error)),
    )
    for index,name in enumerate(COMPONENTS):
        values[f"{name}_mse"]=float(np.mean(squared[:,index]))
        values[f"{name}_bias"]=float(bias[index])
    profile=dict(time_mse=time_mse,component_mse=np.mean(squared,axis=0),
        component_bias=bias,error=error,predicted=predicted)
    return values,profile


def pair_diagnostics(good, bad, good_profile, bad_profile):
    ge=good_profile["error"]; be=bad_profile["error"]
    gn=np.linalg.norm(ge); bn=np.linalg.norm(be)
    cosine=float(np.sum(ge*be)/(gn*bn)) if gn and bn else 1.0
    gtime=good_profile["time_mse"]; btime=bad_profile["time_mse"]
    return dict(
        good_lower_time_steps=int(np.sum(gtime<btime)),time_steps=len(gtime),
        good_lower_time_fraction=float(np.mean(gtime<btime)),
        good_lower_components=[COMPONENTS[i] for i in np.flatnonzero(good_profile["component_mse"]<bad_profile["component_mse"])],
        residual_cosine=cosine,
        normalized_residual_shape_distance=float(np.linalg.norm(ge/max(gn,np.finfo(float).tiny)-be/max(bn,np.finfo(float).tiny))),
        largest_good_advantage_step=int(np.argmax(btime-gtime)),largest_bad_advantage_step=int(np.argmax(gtime-btime)),
        largest_good_advantage=float(np.max(btime-gtime)),largest_bad_advantage=float(np.max(gtime-btime)),
    )


def consistency(states, keys):
    output=[]
    for key in keys:
        deltas=np.asarray([math.log10(max(s["good"][key],1e-300))-math.log10(max(s["bad"][key],1e-300)) for s in states])
        output.append(dict(metric=key,good_lower=int(np.sum(deltas<0)),states=len(deltas),
                           median_log10_good_over_bad=float(np.median(deltas)),
                           minimum=float(np.min(deltas)),maximum=float(np.max(deltas))))
    return sorted(output,key=lambda x:(-x["good_lower"],x["median_log10_good_over_bad"]))


@torch.no_grad()
def main():
    OUT.mkdir(parents=True,exist_ok=True); checkpoint_before=sha256(CHECKPOINT)
    paths=[DATA/"dataset.json",DATA/"states.json",DATA/"hard_cases.jsonl"]
    data_before={p.name:sha256(p) for p in paths}
    rows=[r for r in json.loads((DATA/"dataset.json").read_text()) if r["task"]=="candidate_selection"]
    grouped=defaultdict(list)
    for row in rows: grouped[row["state_id"]].append(row)
    metadata={r["state_id"]:r for r in json.loads((DATA/"states.json").read_text())}
    saved=torch.load(CHECKPOINT,weights_only=False)
    model=DeepCandidateModel(32,3); model.load_state_dict(saved["model_state"]); model.eval()
    trajectory_cache={}; profile_arrays={}; records=[]
    for state_id in sorted(grouped):
        examples=grouped[state_id]
        if examples[0]["system"]!="kepler5" or int(examples[0]["case_id"]) not in PERSISTENT: continue
        truth=np.asarray([bool(r["target_good"]) for r in examples])
        if not truth.any() or truth.all(): continue
        state=GraphState(metadata[state_id],examples); scores=model(state,examples).cpu().numpy()
        gi=np.flatnonzero(truth)[np.argmax(scores[truth])]; bi=np.flatnonzero(~truth)[np.argmax(scores[~truth])]
        case_id=int(examples[0]["case_id"])
        if case_id not in trajectory_cache: trajectory_cache[case_id]=_kepler_case(case_id)[0]
        gv,gp=residual_diagnostics(examples[gi]["theta"],trajectory_cache[case_id])
        bv,bp=residual_diagnostics(examples[bi]["theta"],trajectory_cache[case_id])
        profile_arrays[f"{state_id}__good_error"]=gp["error"]
        profile_arrays[f"{state_id}__bad_error"]=bp["error"]
        profile_arrays[f"{state_id}__good_time_mse"]=gp["time_mse"]
        profile_arrays[f"{state_id}__bad_time_mse"]=bp["time_mse"]
        records.append(dict(state_id=state_id,case_id=case_id,split=examples[0]["split"],
            evaluation_count=int(examples[0]["evaluation_count"]),model_ranks_good_first=bool(scores[gi]>scores[bi]),
            model_good_minus_bad=float(scores[gi]-scores[bi]),
            good=dict(example_id=examples[gi]["example_id"],entity_index=int(examples[gi]["entity_index"]),
                      source=examples[gi]["entity_source"],theta=examples[gi]["theta"],model_score=float(scores[gi]),**gv),
            bad=dict(example_id=examples[bi]["example_id"],entity_index=int(examples[bi]["entity_index"]),
                     source=examples[bi]["entity_source"],theta=examples[bi]["theta"],model_score=float(scores[bi]),**bv),
            pair=pair_diagnostics(gv,bv,gp,bp),
            objective_replay_good_abs_error=abs(gv["mse"]-float(examples[gi]["label_objective_loss"])),
            objective_replay_bad_abs_error=abs(bv["mse"]-float(examples[bi]["label_objective_loss"]))))
    failed=[r for r in records if not r["model_ranks_good_first"]]; solved=[r for r in records if r["model_ranks_good_first"]]
    lower_better=("mse","position_mse","velocity_mse","body1_mse","body2_mse","early_mse","middle_mse","late_mse",
                  "peak_mse","com_position_mse","com_velocity_mse","relative_position_mse","relative_velocity_mse",
                  "radial_position_mse","tangential_position_mse","radial_velocity_mse","tangential_velocity_mse",
                  "angular_momentum_error_mse")+tuple(f"{name}_mse" for name in COMPONENTS)
    checkpoint_after=sha256(CHECKPOINT)
    report=dict(status="READ_ONLY_PREFIX_INFORMATION_DIAGNOSIS",scope=dict(states=len(records),failed=len(failed),solved=len(solved)),
        current_n1_representation=dict(includes_observed_prefix=False,includes_candidate_true_residual=False,
            includes_candidate_coordinates=True,includes_owner_scalar_loss=True,includes_surrogate_prediction=True,
            includes_search_geometry_and_relations=True),
        residual_definition="known five-parameter Kepler step(candidate, observed_prefix[t]) - observed_prefix[t+1]",
        records=records,
        consistency=dict(failed=consistency(failed,lower_better),solved=consistency(solved,lower_better),all=consistency(records,lower_better)),
        dataset_hashes_before=data_before,dataset_hashes_after={p.name:sha256(p) for p in paths},
        dataset_unchanged=data_before=={p.name:sha256(p) for p in paths},
        checkpoint=str(CHECKPOINT),checkpoint_sha256=checkpoint_after,
        protected_checkpoint_changes=[],checkpoint_hashes_unchanged=checkpoint_before==checkpoint_after)
    atomic_json(OUT/"report.json",report); np.savez_compressed(OUT/"residual_profiles.npz",**profile_arrays)
    print(json.dumps({"scope":report["scope"],
        "failed_top_metrics":report["consistency"]["failed"][:12],
        "solved_top_metrics":report["consistency"]["solved"][:12],
        "case22_41":[{"state_id":r["state_id"],"margin":r["model_good_minus_bad"],
            "good_bad_mse":[r["good"]["mse"],r["bad"]["mse"]],"pair":r["pair"]}
            for r in records if r["case_id"] in (22,41)],
        "objective_replay_max":max(max(r["objective_replay_good_abs_error"],r["objective_replay_bad_abs_error"]) for r in records),
        "checkpoint_hashes_unchanged":report["checkpoint_hashes_unchanged"]},indent=2))


if __name__=="__main__": main()
