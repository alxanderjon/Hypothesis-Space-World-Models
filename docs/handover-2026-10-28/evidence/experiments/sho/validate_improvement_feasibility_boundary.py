#!/usr/bin/env python3
"""Replace only lossy normalized feasibility routing with exact raw-margin sign."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.audit_stage1b_same_size_grouping as grouping_teacher
import experiments.sho.run_stage1b_edirect_vs_direct as production
import experiments.sho.run_stage1b_hull_multiplicative_smoke as smoke
import experiments.sho.run_stage1b_hull_turn_smokescreen as turn
import experiments.sho.run_stage1b_improvement_and as improvement
from experiments.geometry.train_addition_operator import Addition
from experiments.geometry.train_stationary_magnitude_threshold import MagnitudeThreshold
from experiments.geometry.train_subtraction_operator import Subtraction
from experiments.sho.validate_candidate_voronoi_assignment import recovered_pool
from experiments.sho.validate_lower_hull_construction import learned_hull, load_hull
from experiments.sho.validate_representative_tie_selector import StrictLossSelector, fixed_grouping
from experiments.sho.validate_same_size_grouping_representative import load_fast_cells, load_grouping
from experiments.sho.validate_sobol_batching_truncation import LearnedSobol


BASE = Path("outputs/sho/stage1B")
OUT = BASE / "improvement_feasibility_boundary"
ADD_MANIFEST = BASE / "addition_operator_smoke/candidate_manifest.json"
SUB_MANIFEST = BASE / "subtraction_operator_training/subtraction_operator_locked.json"
ZERO_MANIFEST = BASE / "stationary_magnitude_threshold_smoke/zero_anchor_converged/candidate_manifest.json"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


class RawBoundary:
    def __init__(self) -> None:
        self.am = json.loads(ADD_MANIFEST.read_text()); self.sm = json.loads(SUB_MANIFEST.read_text())
        self.zm = json.loads(ZERO_MANIFEST.read_text())
        assert self.am["status"] == "FROZEN_CANDIDATE"
        assert self.sm["status"] == "LOCKED" and self.zm["status"] == "FROZEN_CANDIDATE"
        self.add_path = Path(self.am["checkpoint"]); self.sub_path = Path(self.sm["checkpoint"]); self.zero_path = Path(self.zm["checkpoint"])
        assert sha(self.add_path) == self.am["sha256"] and sha(self.sub_path) == self.sm["sha256"] and sha(self.zero_path) == self.zm["sha256"]
        self.add = Addition(); self.sub = Subtraction(); self.zero = MagnitudeThreshold()
        self.add.load_state_dict(torch.load(self.add_path, map_location="cpu", weights_only=False)["state_dict"])
        self.sub.load_state_dict(torch.load(self.sub_path, map_location="cpu", weights_only=False)["state_dict"])
        self.zero.load_state_dict(torch.load(self.zero_path, map_location="cpu", weights_only=False)["state_dict"])
        for model in (self.add, self.sub, self.zero): model.eval().requires_grad_(False)

    def evaluate(self, upper: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        upper = np.asarray(upper, np.float64); lower = np.asarray(lower, np.float64)
        with torch.no_grad():
            summed = self.add(torch.from_numpy(np.column_stack((upper, np.full(len(upper), turn.HULL_TOLERANCE)))))
            margin = self.sub(torch.column_stack((summed, torch.from_numpy(lower))))
            state = self.zero.decision(margin)
        return margin.numpy(), state.numpy(), state.numpy() >= 0


def load_and() -> nn.Linear:
    checkpoint = torch.load(improvement.AND_PATH, map_location="cpu", weights_only=False)
    assert checkpoint["status"] == "LOCKED_COMPONENT"
    model = nn.Linear(2, 1, bias=True).to(torch.float64)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval().requires_grad_(False)


def repaired_pool(states: list[dict[str, object]], split: str, boundary: RawBoundary,
                  and_model: nn.Linear) -> dict[str, np.ndarray]:
    feasibility = improvement.feasibility_margin._build_pool(states, split)
    upstream = feasibility["upstream"]
    finite = feasibility["finite_event_indices"]; endpoints = feasibility["endpoint_event_indices"]
    margin, sign_state, finite_feasible = boundary.evaluate(feasibility["locked_K_upper"], feasibility["locked_K_lower"])
    if not np.array_equal(margin.view(np.uint64), feasibility["raw_margin"].view(np.uint64)):
        raise RuntimeError("locked arithmetic changed production raw-margin bits")
    feasible = np.ones(len(upstream["A"]), bool); feasible[finite] = finite_feasible
    positive = np.ones(len(upstream["A"]), bool)
    positive[finite] = upstream["locked_sign_next"][finite] > 0.0
    inputs = np.column_stack((positive.astype(np.float64), feasible.astype(np.float64)))
    with torch.no_grad(): prediction = and_model(torch.from_numpy(inputs)).squeeze(-1).numpy() >= 0.0
    return {"prediction": prediction, "teacher": upstream["exact_pass"].astype(bool),
            "state_position": upstream["state_position"], "hull_position": upstream["hull_position"],
            "finite": finite, "endpoints": endpoints, "margin": margin, "sign_state": sign_state,
            "feasible": feasible, "positive": positive}


def replay(kind: str, states: list[dict[str, object]], boundary: RawBoundary,
           and_model: nn.Linear) -> dict[str, int]:
    pool = repaired_pool(states, kind, boundary, and_model)
    decision_errors = int(np.sum(pool["prediction"] != pool["teacher"]))
    selected_errors = rejected_errors = payload_errors = 0
    for position, state in enumerate(states):
        mask = pool["state_position"] == position
        hull = np.asarray(state["teacher_hull"], np.int64); sizes=np.asarray(state["sizes"]); losses=np.asarray(state["losses"]); reps=np.asarray(state["representatives"])
        picked = [int(hull[int(p)]) for p, keep in zip(pool["hull_position"][mask], pool["prediction"][mask]) if keep]
        if not picked: picked.append(int(np.argmax(sizes)))
        got=np.asarray(sorted(picked,key=lambda i:(sizes[i],losses[i],i)),np.int64)
        expected=production._potentially_optimal_cells(sizes,losses)
        selected_errors += int(not np.array_equal(got,expected))
        rejected_errors += int(not np.array_equal(np.setdiff1d(hull,got,assume_unique=True),np.setdiff1d(hull,expected,assume_unique=True)))
        payload_errors += int(not all(np.array_equal(a,b) for a,b in zip((sizes[got],losses[got],reps[got]),(sizes[expected],losses[expected],reps[expected]))))
    return {"states":len(states),"events":len(pool["teacher"]),"pass_events":int(pool["teacher"].sum()),"fail_events":int((~pool["teacher"]).sum()),
            "endpoint_events":len(pool["endpoints"]),"decision_errors":decision_errors,"selected_vertex_state_errors":selected_errors,
            "rejected_vertex_state_errors":rejected_errors,"downstream_scoring_input_state_errors":payload_errors}


def boundary_controls(boundary: RawBoundary, and_model: nn.Linear) -> dict[str, int]:
    rng=np.random.default_rng(13112026)
    anchors=np.r_[np.array([-1e100,-1.,-0.,0.,1.,1e100]),np.ldexp(rng.uniform(-2,2,20000),rng.integers(-900,900,20000))]
    anchors=anchors[np.isfinite(anchors)]
    upper=[]; lower=[]; labels=[]; classes=[]
    for u in anchors:
        b=np.float64(u+turn.HULL_TOLERANCE)
        if not np.isfinite(b): continue
        for steps,name in ((-4,"feasible"),(-1,"immediate_feasible"),(0,"equality"),(1,"immediate_infeasible"),(4,"infeasible")):
            v=b
            for _ in range(abs(steps)): v=np.nextafter(v,-np.inf if steps<0 else np.inf)
            upper.append(u);lower.append(v);labels.append(((u+turn.HULL_TOLERANCE)-v)>=0.0);classes.append(name)
    upper=np.asarray(upper);lower=np.asarray(lower);labels=np.asarray(labels);classes=np.asarray(classes)
    margin,state,pred=boundary.evaluate(upper,lower);native=(upper+turn.HULL_TOLERANCE)-lower
    errors=pred!=labels
    # final AND truth table remains unchanged and exact
    x=np.asarray([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    with torch.no_grad(): and_pred=and_model(torch.from_numpy(x)).squeeze(-1).numpy()>=0
    result={"cases":len(labels),"errors":int(errors.sum()),"arithmetic_bit_errors":int(np.sum(margin.view(np.uint64)!=native.view(np.uint64))),
            "signed_state_errors":int(np.sum(state!=np.sign(native).astype(np.int64))),"equality_cases":int(np.sum(classes=="equality")),
            "equality_errors":int(np.sum(errors&(classes=="equality"))),"immediate_feasible_cases":int(np.sum(classes=="immediate_feasible")),
            "immediate_feasible_errors":int(np.sum(errors&(classes=="immediate_feasible"))),"immediate_infeasible_cases":int(np.sum(classes=="immediate_infeasible")),
            "immediate_infeasible_errors":int(np.sum(errors&(classes=="immediate_infeasible"))),"and_truth_table_errors":int(np.sum(and_pred!=np.asarray([0,0,0,1],bool)))}
    return result


def main() -> None:
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    for p in (OUT/"improvement_feasibility_boundary_locked.json",OUT/"improvement_condition_locked.json"):
        if p.exists(): raise RuntimeError(f"Refusing to overwrite {p}")
    torch.set_num_threads(1); grouping,_=load_grouping(); selector=StrictLossSelector(); hull=load_hull(); boundary=RawBoundary(); and_model=load_and()
    protected=[boundary.add_path,boundary.sub_path,boundary.zero_path,improvement.AND_PATH,improvement.FEASIBILITY_MARGIN_PATH,
               improvement.FEASIBILITY_THRESHOLD_PATH,improvement.SLOPE_SIGN_PATH,BASE/"lower_hull_construction/lower_hull_construction_locked.json",Path(production.__file__)]
    before={str(p):sha(p) for p in protected};dump(OUT/"protected_before.json",before)
    payload=torch.load(turn.RAW_TRACE_PATH,map_location="cpu",weights_only=False)
    train=turn._stratified_state_indices(payload["states"],"train",turn.TRAIN_SOURCE_STATES,smoke.DATA_SEED)
    held=turn._stratified_state_indices(payload["states"],"validation",turn.VALIDATION_SOURCE_STATES,smoke.DATA_SEED+1)
    raw=grouping_teacher._build_states(payload,[(i,"archived_training") for i in train]+[(i,"held_out") for i in held],workers=8)
    by={"archived_training":[],"held_out":[],"fresh":[]}
    for item in raw:
        g,_=fixed_grouping(item["sizes"],item["losses"],grouping,selector);h=learned_hull(g["grouped_sizes"],g["grouped_losses"],hull)
        by[item["split"]].append({"source_state_index":item["source_state_index"],"trajectory_id":item["trajectory_id"],"iteration":item["iteration"],
            "state_bin":turn._state_bin(len(item["sizes"])),"sizes":g["grouped_sizes"],"losses":g["grouped_losses"],"representatives":g["representatives"],"teacher_hull":h})
    # Reuse the immediately preceding production-replay fresh fixture exactly,
    # preserving its audited 23 hull events and the requested 4,002 total.
    fast=load_fast_cells();sobol=LearnedSobol();rng=np.random.default_rng(12112026)
    for i,n in enumerate((6,10,23,41,64)):
        sites=rng.random((n,2));
        if i%2==0:sites[:4]=np.asarray([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        losses=(sites[:,0]-.29)**2+1.3*(sites[:,1]-.67)**2+np.arange(n)*1e-12+1e-8;gid,it=996001+i,81+i
        probes=recovered_pool(n,gid,it,sobol);cell=fast.cells(sites,probes);truth=production._cell_data(sites,losses,gid,it)
        assert np.array_equal(cell["probe_owner"],truth.owners) and np.array_equal(cell["sizes"],truth.sizes)
        g,_=fixed_grouping(cell["sizes"],losses,grouping,selector);h=learned_hull(g["grouped_sizes"],g["grouped_losses"],hull)
        by["fresh"].append({"source_state_index":910000+i,"trajectory_id":gid,"iteration":it,"state_bin":turn._state_bin(n),"sizes":g["grouped_sizes"],"losses":g["grouped_losses"],"representatives":g["representatives"],"teacher_hull":h})
    replay_results={k:replay(k,v,boundary,and_model) for k,v in by.items()}; controls=boundary_controls(boundary,and_model)
    after={str(p):sha(p) for p in protected};changes=[p for p in before if before[p]!=after[p]]
    errors=sum(sum(v[k] for k in ("decision_errors","selected_vertex_state_errors","rejected_vertex_state_errors","downstream_scoring_input_state_errors")) for v in replay_results.values())
    passed=errors==0 and all(v==0 for k,v in controls.items() if k.endswith("errors")) and not changes
    result={"status":"LOCKED_ASSEMBLY" if passed else "FAILED","name":"improvement_feasibility_boundary","new_parameters":0,
        "production_order":"raw_margin = addition(K_next,5e-14); raw_margin = subtraction(raw_margin,K_lower); feasible iff sign is ZERO or POSITIVE",
        "raw_margin_sign_sufficient":True,"operators":{"addition":{"status":"FROZEN_CANDIDATE","path":str(boundary.add_path),"sha256":sha(boundary.add_path)},
        "subtraction":{"status":"LOCKED","path":str(boundary.sub_path),"sha256":sha(boundary.sub_path)},"zero_sign":{"status":"FROZEN_CANDIDATE reused by locked routing","path":str(boundary.zero_path),"sha256":sha(boundary.zero_path)},
        "final_AND":{"status":"LOCKED_UNCHANGED","path":str(improvement.AND_PATH),"sha256":sha(improvement.AND_PATH)}},"replay":replay_results,"controls":controls,
        "protected_changes":changes,"seconds":time.time()-started,"next_operation":"cell scoring (not run)"}
    dump(OUT/"results.json",result)
    if not passed: raise AssertionError(result)
    common={"status":"LOCKED_ASSEMBLY","new_parameters":0,"results_sha256":sha(OUT/"results.json"),"implementation_sha256":sha(Path(__file__)),
            "component_hashes":{k:v["sha256"] for k,v in result["operators"].items()}}
    dump(OUT/"improvement_feasibility_boundary_locked.json",{"name":"improvement_feasibility_boundary",**common})
    dump(OUT/"improvement_condition_locked.json",{"name":"improvement_condition",**common})
    print("PASS",json.dumps({"replay":replay_results,"controls":controls},sort_keys=True))


if __name__=="__main__":main()
