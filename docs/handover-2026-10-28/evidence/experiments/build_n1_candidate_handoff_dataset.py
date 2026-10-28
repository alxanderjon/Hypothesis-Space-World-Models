#!/usr/bin/env python3
"""Build lossless ragged candidate-selection and handoff datasets.

No model is fitted.  Ground truth is used only inside the unchanged downstream
recovery check and never written into an input state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.spatial import cKDTree
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/n1_candidate_handoff_supervision")
STATE_DIR = OUT / "states"
CACHE_DIR = OUT / "label_cache"
BALL_ROOT = Path("outputs/d5_production_pipeline_validation/cases/reflection_ball")
KEPLER_ROOT = Path("outputs/d5_handoff_continuation_20000/cases")
KEPLER_HANDOFF_ROOT = Path("outputs/d5_handoff_characterization/kepler_cases")
KEPLER_DIAG = Path("outputs/d5_kepler_persistent_localization_diagnosis/report.json")
IDENT = Path("outputs/d5_handoff_characterization/reflection_identifiability.json")
PRIOR_AUDIT = Path("outputs/n1_kepler_good_bad_information_audit/candidate_states.json")
SCHEMA_VERSION = 1


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,)): return float(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    return value


def digest_arrays(*arrays) -> str:
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def round_starts(events: list[dict]) -> list[tuple[int,int]]:
    starts={}
    for row in events:
        if row["source"] != "initial":
            starts.setdefault(int(row["iteration"]),int(row["evaluation"])-1)
    return sorted((count,it) for it,count in starts.items())


def efficient_dominators(sizes: np.ndarray, losses: np.ndarray):
    """Exact dominance flag plus one canonical witness per dominated site."""
    n=len(sizes); witness=np.full(n,-1,dtype=np.int32); dominated=np.zeros(n,bool)
    order=np.lexsort((np.arange(n),losses,-sizes))
    pos=0; best_prev=-1
    while pos<n:
        end=pos+1; size=sizes[order[pos]]
        while end<n and sizes[order[end]] == size: end+=1
        group=order[pos:end]
        same_best=int(group[np.lexsort((group,losses[group]))[0]])
        for i0 in group:
            i=int(i0); options=[]
            if best_prev>=0 and losses[best_prev] <= losses[i]: options.append(best_prev)
            if losses[same_best] < losses[i]: options.append(same_best)
            if options:
                options.sort(key=lambda j:(losses[j],-sizes[j],j)); witness[i]=options[0]; dominated[i]=True
        candidates=list(group)+( [best_prev] if best_prev>=0 else [] )
        best_prev=min(candidates,key=lambda j:(losses[j],j))
        pos=end
    return dominated,witness


def hull_metadata(sizes: np.ndarray, losses: np.ndarray):
    n=len(sizes); order=np.lexsort((np.arange(n),losses,sizes))
    groups=[]; cell_group=np.full(n,-1,np.int32)
    for idx0 in order:
        idx=int(idx0); size=float(sizes[idx]); loss=float(losses[idx])
        if not groups or abs(size-groups[-1]["size"])>1e-14:
            groups.append(dict(size=size,loss=loss,sample_index=idx))
        elif loss < groups[-1]["loss"]:
            groups[-1].update(loss=loss,sample_index=idx)
        cell_group[idx]=len(groups)-1
    hull=[]
    for gi,g in enumerate(groups):
        while len(hull)>=2:
            a,b=groups[hull[-2]],groups[hull[-1]]
            cross=(b["size"]-a["size"])*(g["loss"]-a["loss"])-(b["loss"]-a["loss"])*(g["size"]-a["size"])
            scale=max(1.,abs(a["loss"]),abs(b["loss"]),abs(g["loss"]))
            if cross < -5e-14*scale: hull.pop()
            else: break
        hull.append(gi)
    minimum=float(np.min(losses)); target=minimum-edirect.teacher.EDIRECT_EPSILON*abs(minimum)
    hull_pos=np.full(n,-1,np.int32); epsilon=np.zeros(n,bool)
    for position,gi in enumerate(hull):
        g=groups[gi]; size=g["size"]; loss=g["loss"]
        lower=-math.inf if position==0 else (loss-groups[hull[position-1]]["loss"])/(size-groups[hull[position-1]]["size"])
        upper=math.inf if position==len(hull)-1 else (groups[hull[position+1]]["loss"]-loss)/(groups[hull[position+1]]["size"]-size)
        required=(loss-target)/size if size>0 else math.inf
        passed=upper>0 and max(lower,required,0.) <= upper+5e-14
        rep=int(g["sample_index"]); hull_pos[rep]=position; epsilon[rep]=passed
        g.update(hull_position=position,lower_slope=lower,upper_slope=upper,required_slope=required,epsilon_pass=passed)
    representatives=np.asarray([g["sample_index"] for g in groups],np.int32)
    return groups,np.asarray(hull,np.int32),cell_group,hull_pos,epsilon,representatives


def state_specs():
    specs=[]
    ident=json.loads(IDENT.read_text())
    impact_ids=[int(r["case_id"]) for r in ident["rows"] if r["classification"]=="IDENTIFIABILITY_COMPLETE"]
    targets=(1000,3000,4900)
    for case_id in impact_ids:
        root=BALL_ROOT/f"case_{case_id:03d}"
        events=json.loads((root/"events.json").read_text()); starts=round_starts(events)
        target=targets[case_id%len(targets)]
        count,iteration=min(starts,key=lambda x:abs(x[0]-target))
        specs.append(dict(system="reflection_ball",case_id=case_id,count=count,iteration=iteration,
                          generator_id=6_100_000+case_id,source=str(root/"n1_state.npz")))
    diag=json.loads(KEPLER_DIAG.read_text())
    persistent=set()
    for case in diag["cases_detail"]:
        cid=int(case["case_id"])
        persistent.add(cid)
        for row in case["representative_states"]:
            specs.append(dict(system="kepler5",case_id=cid,
                              count=int(row["round_start_evaluations"]),iteration=int(row["iteration"]),
                              generator_id=6_300_000+cid,
                              source=str(KEPLER_ROOT/f"case_{cid:03d}"/"n1_state.npz")))
    # The characterization run contains the other 54 real Kepler histories.
    # Each stopped at an actual endpoint (usually the first successful handoff),
    # so ``n1_iterations`` and ``actual_n1_evaluations`` identify a complete
    # production state without fabricating search progress.  These histories
    # are needed because GOOD Kepler handoffs are scarce in the ten persistent
    # localization failures above.
    for result_path in sorted(KEPLER_HANDOFF_ROOT.glob("case_*/result.json")):
        result=json.loads(result_path.read_text()); cid=int(result["case_id"])
        if cid in persistent:
            continue
        specs.append(dict(system="kepler5",case_id=cid,
                          count=int(result["actual_n1_evaluations"]),
                          iteration=int(result["n1_iterations"]),
                          generator_id=6_300_000+cid,
                          source=str(result_path.parent/"n1_state.npz")))
    return specs


def split_map(specs):
    result={}
    for system in ("reflection_ball","kepler5"):
        cases=sorted({s["case_id"] for s in specs if s["system"]==system},
                     key=lambda x:hashlib.sha256(f"{system}:{x}".encode()).hexdigest())
        n=len(cases); train=max(1,int(.70*n)); held=max(1,int(.15*n))
        if train+held>=n: held=max(1,n-train-1)
        for i,c in enumerate(cases): result[(system,c)]="train" if i<train else "held_out" if i<train+held else "fresh_test"
    return result


def routing_name(selected_owner, hull_position, dominated, score_defined, winner):
    if selected_owner: return "SELECTED_CELL_WINNER" if winner else "SELECTED_CELL_NONWINNER"
    if dominated: return "PARETO_DOMINATED_OWNER"
    if hull_position < 0: return "NON_HULL_OWNER"
    return "HULL_NOT_SELECTED_OWNER"


def build_state(spec, split, prior_lookup):
    sid=f"{spec['system']}_case_{spec['case_id']:03d}_n{spec['count']}_it{spec['iteration']}"
    path=STATE_DIR/f"{sid}.npz"; meta_path=STATE_DIR/f"{sid}.json"
    if path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        # Splits are assigned over the complete trajectory set, and therefore
        # can change when additional pre-existing histories are admitted.
        # The tensor state and its digest remain untouched.
        meta["split"]=split
        z=np.load(path)
        meta["production_n2_radius"]=(
            (z["upper"]-z["lower"])/float(spec["count"])**(1.0/int(z["dimension"]))
        ).tolist()
        atomic_json(meta_path,meta)
        return meta
    saved=np.load(spec["source"]); points=np.asarray(saved["points"][:spec["count"]],np.float64); losses=np.asarray(saved["losses"][:spec["count"]],np.float64)
    lower,upper,_=production._config(spec["system"]); api=edirect.composition(5)
    data=api["_cell_data"](points,losses,spec["generator_id"],spec["iteration"])
    surface=GaussianLocalResponse(points,losses)
    predictions=surface.predict(data.points)
    k=min(24,len(points)); neighbor_dist,neighbor_idx=cKDTree(points).query(data.points,k=k)
    if k==1: neighbor_dist=neighbor_dist[:,None];neighbor_idx=neighbor_idx[:,None]
    groups,hull,cell_group,hull_position,epsilon,reps=hull_metadata(data.sizes,losses)
    dominated,witness=efficient_dominators(data.sizes,losses)
    selected=np.asarray(data.selected,np.int32); selected_set=set(map(int,selected))
    selected_sizes=data.sizes[selected]; weights=.5*_normalize(selected_sizes)+.5*_normalize(surface.loo_errors(selected)); largest_pos=int(np.argmax(selected_sizes));weights[largest_pos]=1.
    role=np.zeros(len(data.points),np.int8); score=np.full(len(data.points),np.nan); score_rank=np.full(len(data.points),-1,np.int32); winner=np.zeros(len(data.points),bool)
    for pos,owner0 in enumerate(selected):
        owner=int(owner0); ids=np.flatnonzero(data.owners==owner)
        if not len(ids):
            continue
        role[ids]=2 if pos==largest_pos else 1
        if pos==largest_pos: values=data.distances[ids]
        else: values=weights[pos]*_normalize(data.distances[ids])-(1.-weights[pos])*_normalize(predictions[ids])
        order=np.lexsort((ids,-values)); score[ids]=values; score_rank[ids[order]]=np.arange(1,len(ids)+1); winner[ids[order[0]]]=data.distances[ids[order[0]]]>1e-12
    candidate_status=np.empty(len(data.points),np.int8)
    for i in range(len(data.points)):
        o=int(data.owners[i]); candidate_status[i]=(1 if winner[i] else 2 if o in selected_set else 3 if dominated[o] else 4 if hull_position[o]<0 else 5)

    np.savez_compressed(path,
        schema_version=np.int64(SCHEMA_VERSION),dimension=np.int64(5),case_id=np.int64(spec["case_id"]),generator_id=np.int64(spec["generator_id"]),iteration=np.int64(spec["iteration"]),evaluation_count=np.int64(spec["count"]),
        lower=lower,upper=upper,sites=points,site_losses=losses,site_encoded_losses=surface.values,
        reservoir=data.points,candidate_owner=data.owners.astype(np.int32),candidate_owner_distance=data.distances,
        cell_sizes=data.sizes,selected_sites=selected,pareto_dominated=dominated,canonical_dominator=witness,
        cell_group=cell_group,hull_groups=hull,hull_position=hull_position,hull_epsilon_pass=epsilon,group_representatives=reps,
        candidate_neighbor_indices=np.asarray(neighbor_idx,np.int32),candidate_neighbor_distances=neighbor_dist,
        candidate_surrogate_prediction=predictions,candidate_role=role,candidate_score=score,candidate_score_rank=score_rank,candidate_is_production_winner=winner,candidate_routing_status=candidate_status,
        selected_cell_weights=weights,largest_selected_position=np.int64(largest_pos))

    # Diverse canonical examples. Existing checked Kepler candidates are forced in.
    choices=[]
    def add(index,source):
        index=int(index)
        if all(index!=x[0] for x in choices): choices.append((index,source))
    win=np.flatnonzero(winner)
    if len(win): add(win[0],"production_selected"); add(win[-1],"production_selected")
    dom=np.flatnonzero(dominated[data.owners] & ~winner)
    if len(dom): add(dom[int(np.argmin(predictions[dom]))],"pareto_dominated_best_surrogate"); add(dom[int(np.argmax(data.distances[dom]))],"pareto_dominated_farthest")
    nonh=np.flatnonzero((hull_position[data.owners]<0) & ~dominated[data.owners] & ~winner)
    if len(nonh): add(nonh[int(np.argmin(predictions[nonh]))],"non_hull_best_surrogate")
    routed_loser=np.flatnonzero((role>0)&~winner)
    if len(routed_loser): add(routed_loser[int(np.argmin(np.where(score_rank[routed_loser]>1,score_rank[routed_loser],10**9)))],"selected_cell_poor_rank")
    unselected=np.flatnonzero(role==0)
    if len(unselected): add(unselected[int(np.argmin(predictions[unselected]))],"global_attractive_unselected")
    # Force previously labeled identical queries.
    for key,record in prior_lookup.items():
        psystem,pcase,pcount,pbytes=key
        if psystem==spec["system"] and pcase==spec["case_id"] and pcount==spec["count"]:
            matches=np.flatnonzero(np.all(data.points==np.frombuffer(pbytes,np.float64)[None,:],axis=1))
            if len(matches)==1:add(matches[0],"reused_prior_checked")
    fill=np.linspace(0,len(data.points)-1,8,dtype=int)
    for i in fill:
        if len(choices)>=8: break
        add(i,"canonical_reservoir_spread")

    # Six evaluated sites, retaining identities and current-count semantics.
    site_choices=[]
    def sadd(index,source):
        index=int(index)
        if all(index!=x[0] for x in site_choices):site_choices.append((index,source))
    loss_order=np.lexsort((np.arange(len(losses)),losses)); sadd(loss_order[0],"incumbent");sadd(loss_order[min(1,len(loss_order)-1)],"second_loss")
    sadd(int(np.argmax(data.sizes)),"largest_cell_owner")
    if len(selected):sadd(selected[0],"first_selected_cell");sadd(selected[-1],"last_selected_cell")
    sadd(len(points)-1,"most_recent_site")
    for i in np.linspace(0,len(points)-1,8,dtype=int):
        if len(site_choices)>=6:break
        sadd(i,"canonical_history_spread")

    state_hash=digest_arrays(points,losses,data.points,data.owners,data.distances,data.sizes)
    meta=dict(state_id=sid,system=spec["system"],case_id=spec["case_id"],split=split,
              evaluation_count=spec["count"],iteration=spec["iteration"],generator_id=spec["generator_id"],
              state_npz=str(path),state_sha256=state_hash,site_count=len(points),reservoir_count=len(data.points),
              selected_cell_count=len(selected),candidate_choices=[dict(index=i,source=s) for i,s in choices],
              handoff_site_choices=[dict(index=i,source=s) for i,s in site_choices],
              production_n2_radius=((upper-lower)/float(spec["count"])**(1.0/len(lower))).tolist())
    atomic_json(meta_path,meta); return meta


def example_key(row):
    raw=f"{row['task']}|{row['system']}|{row['case_id']}|{row['state_id']}|{row['entity_index']}|{row['evaluation_count']}".encode()
    return hashlib.sha256(raw).hexdigest()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);STATE_DIR.mkdir(parents=True,exist_ok=True);CACHE_DIR.mkdir(parents=True,exist_ok=True)
    prior_rows=json.loads(PRIOR_AUDIT.read_text()) if PRIOR_AUDIT.exists() else []
    prior_lookup={}
    for r in prior_rows:
        key=("kepler5",int(r["case_id"]),int(r["state_reference"]["history_count"]),np.asarray(r["candidate_unit"],np.float64).tobytes())
        prior_lookup[key]=r
    specs=state_specs(); splits=split_map(specs); states=[]; examples=[]
    for number,spec in enumerate(specs,1):
        print(f"state {number}/{len(specs)} {spec['system']} {spec['case_id']} n={spec['count']}",flush=True)
        state=build_state(spec,splits[(spec["system"],spec["case_id"])],prior_lookup);states.append(state)
        z=np.load(state["state_npz"]); lower=z["lower"];upper=z["upper"]
        for choice in state["candidate_choices"]:
            idx=int(choice["index"]); unit=np.asarray(z["reservoir"][idx]); theta=lower+(upper-lower)*unit; owner=int(z["candidate_owner"][idx])
            row=dict(task="candidate_selection",system=state["system"],case_id=state["case_id"],split=state["split"],state_id=state["state_id"],state_npz=state["state_npz"],state_sha256=state["state_sha256"],entity_index=idx,entity_source=choice["source"],unit_point=unit.tolist(),theta=theta.tolist(),owner_site=owner,evaluation_count=state["evaluation_count"]+1,current_routing_status=int(z["candidate_routing_status"][idx]),current_routing_status_name=routing_name(owner in set(map(int,z["selected_sites"])),int(z["hull_position"][owner]),bool(z["pareto_dominated"][owner]),bool(np.isfinite(z["candidate_score"][idx])),bool(z["candidate_is_production_winner"][idx])))
            row["example_id"]=example_key(row);examples.append(row)
        for choice in state["handoff_site_choices"]:
            idx=int(choice["index"]);unit=np.asarray(z["sites"][idx]);theta=lower+(upper-lower)*unit
            row=dict(task="handoff_promotion",system=state["system"],case_id=state["case_id"],split=state["split"],state_id=state["state_id"],state_npz=state["state_npz"],state_sha256=state["state_sha256"],entity_index=idx,entity_source=choice["source"],unit_point=unit.tolist(),theta=theta.tolist(),observed_loss=float(z["site_losses"][idx]),evaluation_count=state["evaluation_count"],current_routing_status_name="EVALUATED_SITE")
            row["example_id"]=example_key(row);examples.append(row)
    # Exact query dedup only; state/count differences remain distinct by design.
    unique={}; duplicates=0
    for row in examples:
        if row["example_id"] in unique:duplicates+=1
        else:unique[row["example_id"]]=row
    manifest=list(unique.values())
    # Seed cache with identical prior candidate labels.
    reused=0
    for row in manifest:
        if row["task"]!="candidate_selection" or row["system"]!="kepler5":continue
        key=("kepler5",row["case_id"],row["evaluation_count"]-1,np.asarray(row["unit_point"],np.float64).tobytes())
        old=prior_lookup.get(key)
        if old is None:continue
        cache=CACHE_DIR/f"{row['example_id']}.json"
        lower,upper,_=production._config(row["system"]);theta=np.asarray(row["theta"],np.float64)
        radius=(upper-lower)/float(row["evaluation_count"])**(1.0/len(theta))
        result=dict(example_id=row["example_id"],reused=True,label=bool(old["label"]=="GOOD"),
                    recovered=bool(old["label"]=="GOOD"),full_pipeline_pass=bool(old["label"]=="GOOD"),
                    objective_loss=float(old["promotion_candidate_loss"]),final_loss=float(old["offline_final_loss"]),
                    final_parameter_l2_error=float(old["offline_final_parameter_l2_error"]),continuation_max_error=float(old["offline_rollout_max_error"]),
                    n2_box_lower=np.maximum(lower,theta-radius).tolist(),
                    n2_box_upper=np.minimum(upper,theta+radius).tolist(),
                    n2_evaluations=None,n3_objective_evaluations=None,n3_accepted_steps=None,cache_source="prior_kepler_candidate_audit")
        atomic_json(cache,result);reused+=1
    atomic_json(OUT/"states.json",states);atomic_json(OUT/"examples_unlabeled.json",manifest)
    atomic_json(OUT/"preparation.json",dict(states=len(states),trajectories=len({(s['system'],s['case_id']) for s in states}),examples=len(manifest),candidate_examples=sum(r['task']=='candidate_selection' for r in manifest),handoff_examples=sum(r['task']=='handoff_promotion' for r in manifest),exact_duplicates_removed=duplicates,reused_labels=reused))
    return manifest


def label_case(system: str,case_id: int):
    manifest=json.loads((OUT/"examples_unlabeled.json").read_text()); rows=[r for r in manifest if r["system"]==system and int(r["case_id"])==case_id]
    todo=[r for r in rows if not (CACHE_DIR/f"{r['example_id']}.json").exists()]
    if not todo:return
    trajectory,_=(production._ball_case(case_id) if system=="reflection_ball" else production._kepler_case(case_id))
    objective,torch_objective=production._objectives(system,trajectory);lower,upper,teacher=production._config(system);width=upper-lower
    for number,row in enumerate(todo,1):
        started=time.monotonic();theta=np.asarray(row["theta"],np.float64)
        loss=float(row["observed_loss"]) if row["task"]=="handoff_promotion" else float(objective(theta))
        n=int(row["evaluation_count"]);radius=width/n**(1./5.)
        endpoint=dict(theta=theta,loss=loss,evaluations=n,lower=np.maximum(lower,theta-radius),upper=np.minimum(upper,theta+radius),stop_reason="offline_supervision_query")
        finish=production._finish(system,trajectory,objective,torch_objective,endpoint,teacher,f"dataset_{row['example_id']}")
        result=dict(example_id=row["example_id"],reused=False,label=bool(finish["recovered"] and finish["rollout_finite"]),objective_loss=loss,label_seconds=float(time.monotonic()-started),**finish)
        atomic_json(CACHE_DIR/f"{row['example_id']}.json",result)
        print(f"{system} {case_id} {number}/{len(todo)} {row['task']} {'GOOD' if result['label'] else 'BAD'}",flush=True)


def run_labels(workers: int):
    manifest=json.loads((OUT/"examples_unlabeled.json").read_text()); tasks=sorted({(r["system"],int(r["case_id"])) for r in manifest if not (CACHE_DIR/f"{r['example_id']}.json").exists()})
    workers=max(1,min(workers,4))
    for offset in range(0,len(tasks),workers):
        batch=tasks[offset:offset+workers];children=[]
        for system,case_id in batch:
            cmd=[sys.executable,"-m",__spec__.name,"--label-case",system,str(case_id)]
            children.append(((system,case_id),subprocess.Popen(cmd,env=os.environ.copy())))
        failed=[(task,p.wait()) for task,p in children];failed=[x for x in failed if x[1]]
        if failed:raise RuntimeError(failed)


def distribution(values):
    a=np.asarray(values,float)
    if not len(a):return dict(count=0)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9,method="higher")),maximum=float(a.max()))


def finalize(before):
    examples=json.loads((OUT/"examples_unlabeled.json").read_text()); labeled=[]
    for row in examples:
        p=CACHE_DIR/f"{row['example_id']}.json"
        if not p.exists():raise RuntimeError(f"missing {p}")
        label=json.loads(p.read_text())
        labeled.append({**row, "target_good": bool(label["label"]),
                        **{f"label_{k}":v for k,v in label.items() if k!="example_id"}})
    atomic_json(OUT/"dataset.json",labeled)
    # JSONL index for streaming; ragged tensors remain referenced losslessly by state_npz.
    with (OUT/"dataset.jsonl").open("w") as f:
        for row in labeled:f.write(json.dumps(row,separators=(",",":"))+"\n")
    summaries={}
    for task in ("candidate_selection","handoff_promotion"):
        rows=[r for r in labeled if r["task"]==task]
        by_system={}
        for system in ("reflection_ball","kepler5"):
            sr=[r for r in rows if r["system"]==system]
            by_system[system]=dict(total=len(sr),good=sum(r["label_label"] for r in sr),bad=sum(not r["label_label"] for r in sr),trajectories=len({r["case_id"] for r in sr}))
        routing={}
        for r in rows:
            routing.setdefault(r["current_routing_status_name"],dict(total=0,good=0,bad=0));q=routing[r["current_routing_status_name"]];q["total"]+=1;q["good" if r["label_label"] else "bad"]+=1
        summaries[task]=dict(total=len(rows),good=sum(r["label_label"] for r in rows),bad=sum(not r["label_label"] for r in rows),by_system=by_system,
            distinct_trajectories=len({(r["system"],r["case_id"]) for r in rows}),split_cases={s:len({(r["system"],r["case_id"]) for r in rows if r["split"]==s}) for s in ("train","held_out","fresh_test")},
            split_examples={s:sum(r["split"]==s for r in rows) for s in ("train","held_out","fresh_test")},
            evaluation_progress=distribution([r["evaluation_count"] for r in rows]),routing_status=routing,
            useful_currently_rejected=sum(r["label_label"] and r["current_routing_status_name"] not in ("SELECTED_CELL_WINNER","EVALUATED_SITE") for r in rows))
    hard=[r["example_id"] for r in labeled if (r["system"]=="kepler5" or
          (r["task"]=="candidate_selection" and r["label_label"] and r["current_routing_status_name"] in ("PARETO_DOMINATED_OWNER","NON_HULL_OWNER")) or
          (r["task"]=="handoff_promotion" and r["label_label"] and r["entity_source"]!="incumbent"))]
    reused=sum(r.get("label_reused",False) for r in labeled); seconds=sum(float(r.get("label_label_seconds",0.)) for r in labeled); n2=sum((r.get("label_n2_evaluations") or 0) for r in labeled);n3=sum((r.get("label_n3_objective_evaluations") or 0) for r in labeled)
    after=hashes();atomic_json(OUT/"checkpoint_hashes_after.json",after)
    report=dict(schema_version=SCHEMA_VERSION,state_storage="lossless ragged NPZ plus JSONL state references",systems=["reflection_ball","kepler5"],summaries=summaries,hard_case_examples=len(hard),hard_case_example_ids=hard,
        caching=dict(total=len(labeled),reused_existing_labels=reused,new_labels=len(labeled)-reused,cache_files=len(list(CACHE_DIR.glob('*.json')))),
        downstream_labeling_cost=dict(wall_label_seconds_sum=seconds,n2_objective_evaluations=n2,n3_objective_evaluations=n3,candidate_physical_evaluations=sum(r['task']=='candidate_selection' and not r.get('label_reused',False) for r in labeled)),
        checkpoint_hashes_unchanged=before==after)
    atomic_json(OUT/"report.json",report);atomic_json(OUT/"schema.json",dict(
        version=SCHEMA_VERSION,label_definition="unchanged N2->N3 reaches existing teacher-equivalent recovery and finite rollout",ground_truth_is_input=False,
        state_arrays="see each states/*.npz; variable lengths are preserved without pooling",example_index="dataset.jsonl",splitting="whole (system,case_id) trajectory"))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--label-case",nargs=2);parser.add_argument("--workers",type=int,default=3);parser.add_argument("--finalize",action="store_true");args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.label_case:
        label_case(args.label_case[0],int(args.label_case[1]));return
    before=hashes();atomic_json(OUT/"checkpoint_hashes_before.json",before)
    if args.prepare or not (OUT/"examples_unlabeled.json").exists():prepare()
    if not args.prepare and not args.finalize:run_labels(args.workers)
    if args.finalize or (not args.prepare):
        report=finalize(before);print(json.dumps(jsonable(report),indent=2),flush=True)


if __name__=="__main__":main()
