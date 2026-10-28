#!/usr/bin/env python3
"""Learn only topology validity and numerical stencil-scale selection for N3."""
from __future__ import annotations

import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from experiments.bouncing_ball import run_first_transfer as ball
from experiments.sho import run_stage1b_n3_gradient_polishing as n3
from experiments.sho import train_n3_derivative_representation as prior


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/n3_learned_polisher/stencil_scale_selector")
BALL_ROWS = Path("outputs/bouncing_ball/second_broad_n1_n2_n3_transfer/per_trajectory.csv")
EXPONENTS = (8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 36, 38, 40)
SEED = 20_261_003
FEATURES = 14


@dataclass
class ScaleState:
    family: str
    split: str
    case_id: str
    theta: torch.Tensor
    objective: Callable[[torch.Tensor], torch.Tensor]
    topology: Callable[[np.ndarray], tuple] | None


@dataclass
class DerivativeGroup:
    state: ScaleState
    order: int
    indices: tuple[int, ...]
    target: float
    estimates: np.ndarray
    cancellation: np.ndarray
    topology_valid: np.ndarray
    features: torch.Tensor | None = None
    teacher_index: int = -1


class TopologyRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2, dtype=torch.float64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class ScaleRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURES, 16, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(16, 1, dtype=torch.float64),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def states() -> list[ScaleState]:
    output = []
    for state in prior.make_states():
        if state.family == "ball":
            continue
        output.append(ScaleState(state.family, state.split, state.case_id,
                                 state.theta, state.objective, None))
    with BALL_ROWS.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=len(rows), n_timesteps=ball.TOTAL_STEPS,
        dt=ball.DT, seed=20_261_001
    )
    for row, trajectory in zip(rows, trajectories):
        trajectory_np = np.asarray(trajectory, dtype=np.float64)
        trajectory_t = torch.from_numpy(trajectory_np)
        trajectory_id = int(row["trajectory_id"])
        split = "train" if trajectory_id < 4 else "heldout" if trajectory_id < 6 else "fresh"
        def topology(theta, p=trajectory_np):
            return tuple(ball.numpy_k_event_signature(p[index], theta)
                         for index in range(ball.PREFIX_STATES-1))
        objective = lambda x, p=trajectory_t: ball.prefix_loss_torch(x, p)
        output.append(ScaleState("ball", split, f"{trajectory_id}_n2",
                                 torch.tensor((float(row["n2_a"]), float(row["n2_b"]))),
                                 objective, topology))
        output.append(ScaleState("ball", split, f"{trajectory_id}_final",
                                 torch.tensor((float(row["final_a"]), float(row["final_b"]))),
                                 objective, topology))
    return output


def exact(state: ScaleState) -> tuple[np.ndarray, np.ndarray]:
    x = state.theta.detach().clone().requires_grad_(True)
    value = state.objective(x)
    gradient = torch.autograd.grad(value, x, create_graph=True)[0]
    hessian = torch.stack([
        torch.autograd.grad(gradient[index], x, retain_graph=True)[0]
        for index in range(len(x))
    ])
    hessian = 0.5*(hessian+hessian.T)
    return gradient.detach().numpy(), hessian.detach().numpy()


def same_topology(state: ScaleState, points: list[torch.Tensor]) -> bool:
    if state.topology is None:
        return True
    center = state.topology(state.theta.detach().numpy())
    return all(state.topology(point.detach().numpy()) == center for point in points)


def materialize_state(state: ScaleState) -> list[DerivativeGroup]:
    theta = state.theta.detach()
    dimension = len(theta)
    true_g, true_h = exact(state)
    gradient_estimates = [[] for _ in range(dimension)]
    gradient_cancellation = [[] for _ in range(dimension)]
    gradient_topology = [[] for _ in range(dimension)]
    hessian_indices = [(i,j) for i in range(dimension) for j in range(i,dimension)]
    hessian_estimates = [[] for _ in hessian_indices]
    hessian_cancellation = [[] for _ in hessian_indices]
    hessian_topology = [[] for _ in hessian_indices]
    with torch.no_grad():
        f0 = state.objective(theta)
        for exponent in EXPONENTS:
            steps = (2.0**(-exponent))*torch.maximum(torch.ones_like(theta),torch.abs(theta))
            plus=[]; minus=[]; plus_points=[]; minus_points=[]
            for axis in range(dimension):
                offset=torch.zeros_like(theta); offset[axis]=steps[axis]
                pp=theta+offset; pm=theta-offset
                plus_points.append(pp); minus_points.append(pm)
                plus.append(state.objective(pp)); minus.append(state.objective(pm))
                numerator=plus[-1]-minus[-1]
                gradient_estimates[axis].append(float(numerator/(2.0*steps[axis])))
                denominator=abs(float(plus[-1]))+abs(float(minus[-1]))+2.0*abs(float(f0))+1e-300
                gradient_cancellation[axis].append(abs(float(numerator))/denominator)
                gradient_topology[axis].append(same_topology(state,[pp,pm]))
            serial=0
            for left,right in hessian_indices:
                if left==right:
                    numerator=plus[left]-2.0*f0+minus[left]
                    estimate=numerator/(steps[left]*steps[left])
                    points=[plus_points[left],minus_points[left]]
                    denominator=abs(float(plus[left]))+2.0*abs(float(f0))+abs(float(minus[left]))+1e-300
                else:
                    oi=torch.zeros_like(theta);oi[left]=steps[left]
                    oj=torch.zeros_like(theta);oj[right]=steps[right]
                    points=[theta+oi+oj,theta+oi-oj,theta-oi+oj,theta-oi-oj]
                    values=[state.objective(point) for point in points]
                    numerator=values[0]-values[1]-values[2]+values[3]
                    estimate=numerator/(4.0*steps[left]*steps[right])
                    denominator=sum(abs(float(value)) for value in values)+1e-300
                hessian_estimates[serial].append(float(estimate))
                hessian_cancellation[serial].append(abs(float(numerator))/denominator)
                hessian_topology[serial].append(same_topology(state,points))
                serial+=1
    groups=[]
    for axis in range(dimension):
        groups.append(DerivativeGroup(state,1,(axis,),float(true_g[axis]),
                                      np.asarray(gradient_estimates[axis]),
                                      np.asarray(gradient_cancellation[axis]),
                                      np.asarray(gradient_topology[axis],dtype=bool)))
    for serial,(left,right) in enumerate(hessian_indices):
        groups.append(DerivativeGroup(state,2,(left,right),float(true_h[left,right]),
                                      np.asarray(hessian_estimates[serial]),
                                      np.asarray(hessian_cancellation[serial]),
                                      np.asarray(hessian_topology[serial],dtype=bool)))
    return groups


def logmag(value: np.ndarray | float) -> np.ndarray | float:
    return np.log2(np.maximum(np.abs(value), np.finfo(np.float64).tiny))


def build_features(group: DerivativeGroup) -> None:
    values=group.estimates; count=len(values); rows=[]
    magnitude=np.maximum(np.abs(values),np.finfo(np.float64).tiny)
    for index,(exponent,value) in enumerate(zip(EXPONENTS,values)):
        coarse=values[index-1] if index>0 else value
        fine=values[index+1] if index+1<count else value
        dc=abs(value-coarse); df=abs(fine-value)
        relative_coarse=dc/magnitude[index]
        relative_fine=df/magnitude[index]
        rows.append((
            exponent/40.0,
            float(group.order==1),float(group.order==2),
            float(np.sign(value)),
            float(np.clip(logmag(value)/1024.0,-1.0,1.0)),
            float(np.clip(logmag(dc+np.finfo(np.float64).tiny)/1024.0,-1.0,1.0)),
            float(np.clip(logmag(df+np.finfo(np.float64).tiny)/1024.0,-1.0,1.0)),
            float(np.clip(np.log2(relative_coarse+np.finfo(np.float64).tiny)/64.0,-1.0,1.0)),
            float(np.clip(np.log2(relative_fine+np.finfo(np.float64).tiny)/64.0,-1.0,1.0)),
            float(np.clip(logmag(group.cancellation[index]+np.finfo(np.float64).tiny)/1024.0,-1.0,1.0)),
            float(df<dc),float(index==0),float(index==count-1),
            float(group.topology_valid[index]),
        ))
    group.features=torch.tensor(rows,dtype=torch.float64)
    candidates=np.flatnonzero(group.topology_valid)
    if not len(candidates):
        group.teacher_index=-1;return
    errors=np.abs(values[candidates]-group.target)
    # Exact best available teacher stencil; canonical coarser index wins ties.
    group.teacher_index=int(candidates[int(np.argmin(errors))])


def fit_topology() -> tuple[TopologyRouter,dict]:
    torch.manual_seed(SEED)
    model=TopologyRouter()
    x=torch.tensor([[0.0],[1.0]],dtype=torch.float64);y=torch.tensor([0,1])
    opt=torch.optim.LBFGS(model.parameters(),lr=.8,max_iter=200,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad();loss=nn.functional.cross_entropy(model(x),y);loss.backward();return loss
    opt.step(closure)
    with torch.no_grad():pred=model(x).argmax(1)
    return model.eval().requires_grad_(False),dict(parameters=sum(p.numel() for p in model.parameters()),
        truth_table_errors=int((pred!=y).sum()),weights={k:v.tolist() for k,v in model.state_dict().items()})


def fit_ranker(groups:list[DerivativeGroup])->tuple[ScaleRanker,dict]:
    torch.manual_seed(SEED+1);model=ScaleRanker()
    training=[g for g in groups if g.state.split=="train" and g.teacher_index>=0]
    feature_batch=torch.stack([g.features for g in training])
    valid_batch=torch.stack([torch.from_numpy(g.topology_valid) for g in training])
    target_batch=torch.tensor([g.teacher_index for g in training],dtype=torch.long)
    opt=torch.optim.Adam(model.parameters(),lr=3e-3)
    best=None;best_errors=10**9;patience=0
    for epoch in range(4000):
        opt.zero_grad()
        logits=model(feature_batch.reshape(-1,FEATURES)).reshape(len(training),len(EXPONENTS))
        logits=torch.where(valid_batch,logits,torch.full_like(logits,-1e30))
        loss=nn.functional.cross_entropy(logits,target_batch)
        loss.backward();opt.step()
        if epoch%25==0:
            with torch.no_grad():
                logits=model(feature_batch.reshape(-1,FEATURES)).reshape(len(training),len(EXPONENTS))
                errors=int((torch.argmax(torch.where(valid_batch,logits,torch.full_like(logits,-1e30)),dim=1)!=target_batch).sum())
            if errors<best_errors:
                best_errors=errors;best={k:v.detach().clone() for k,v in model.state_dict().items()};patience=0
            else:patience+=25
            if errors==0 or patience>=500:break
    model.load_state_dict(best)
    return model.eval().requires_grad_(False),dict(parameters=sum(p.numel() for p in model.parameters()),
        epochs=epoch+1,training_groups=len(training),training_selection_errors=best_errors,
        state_dict={k:v.tolist() for k,v in model.state_dict().items()})


def select(model:ScaleRanker,group:DerivativeGroup)->int:
    with torch.no_grad():scores=model(group.features)
    scores=torch.where(torch.from_numpy(group.topology_valid),scores,torch.full_like(scores,-1e30))
    return int(torch.argmax(scores))


def evaluate(topology_model:TopologyRouter,ranker:ScaleRanker,groups:list[DerivativeGroup])->dict:
    topology_errors=0;selection=defaultdict(lambda:dict(groups=0,selection_errors=0,wrong_sign=0,no_valid=0))
    distributions=defaultdict(Counter);selected={}
    for group in groups:
        with torch.no_grad():pred=topology_model(torch.from_numpy(group.topology_valid.astype(np.float64))[:,None]).argmax(1).numpy().astype(bool)
        topology_errors+=int(np.count_nonzero(pred!=group.topology_valid))
        key=f"{group.state.family}_{group.state.split}"
        summary=selection[key];summary["groups"]+=1
        if group.teacher_index<0:
            summary["no_valid"]+=1;continue
        index=select(ranker,group);selected[id(group)]=index
        summary["selection_errors"]+=int(index!=group.teacher_index)
        if group.target!=0.0:
            summary["wrong_sign"]+=int(np.signbit(group.estimates[index])!=np.signbit(group.target))
        distributions[f"{key}_order{group.order}"][EXPONENTS[index]]+=1

    # Diagnostic raw reconstruction from selected scales only.
    diagnostic=defaultdict(lambda:dict(states=0,gradient_sign_errors=0,curvature_route_errors=0,
        worst_gradient_relative_error=0.0,worst_direction_relative_error=0.0))
    by_state=defaultdict(list)
    for group in groups:by_state[id(group.state)].append(group)
    for state_groups in by_state.values():
        state=state_groups[0].state;dimension=len(state.theta);g=np.empty(dimension);h=np.empty((dimension,dimension))
        valid=True
        for group in state_groups:
            if id(group) not in selected:valid=False;break
            value=group.estimates[selected[id(group)]]
            if group.order==1:g[group.indices[0]]=value
            else:
                i,j=group.indices;h[i,j]=h[j,i]=value
        if not valid:continue
        true_g,true_h=exact(state);key=f"{state.family}_{state.split}";item=diagnostic[key];item["states"]+=1
        item["gradient_sign_errors"]+=int(np.count_nonzero(np.signbit(g)!=np.signbit(true_g)))
        item["worst_gradient_relative_error"]=max(item["worst_gradient_relative_error"],float(np.linalg.norm(g-true_g)/max(np.linalg.norm(true_g),1e-300)))
        tg=torch.from_numpy(true_g);th=torch.from_numpy(true_h);pg=torch.from_numpy(g);ph=torch.from_numpy(h)
        true_route,true_dir=prior.direct_route(tg,th);route,direction=prior.direct_route(pg,ph)
        item["curvature_route_errors"]+=int(route!=true_route)
        relative=math.inf if direction is None or true_dir is None else float(torch.linalg.vector_norm(direction-true_dir)/max(float(torch.linalg.vector_norm(true_dir)),1e-300))
        item["worst_direction_relative_error"]=max(item["worst_direction_relative_error"],relative)

    difficult={}
    hard_ids={"132","230","9","10","17","236"}
    for group in groups:
        if group.state.family=="sho" and group.state.case_id in hard_ids and group.order==1:
            difficult[f"sho_{group.state.case_id}_axis{group.indices[0]}"]=EXPONENTS[selected[id(group)]] if id(group) in selected else None
        if group.state.family=="ball" and group.order==1:
            difficult[f"ball_{group.state.case_id}_axis{group.indices[0]}"]=EXPONENTS[selected[id(group)]] if id(group) in selected else None
    known_truncation_selected=0
    for group in groups:
        if id(group) not in selected:continue
        index=selected[id(group)]
        if (group.state.family=="sho" and group.state.case_id=="5"
                and group.order==1 and EXPONENTS[index]<=24):
            known_truncation_selected+=1
        if (group.state.family=="ball" and group.state.case_id=="0_n2"
                and group.order==1 and group.indices==(0,) and EXPONENTS[index]==8):
            known_truncation_selected+=1
    totals=dict(topology_errors=topology_errors,
        selection_errors=sum(v["selection_errors"] for v in selection.values()),
        wrong_sign=sum(v["wrong_sign"] for v in selection.values()),
        no_valid=sum(v["no_valid"] for v in selection.values()),
        known_invalid_or_truncation_selected=known_truncation_selected)
    passed=all(value==0 for value in totals.values())
    return dict(passed=passed,totals=totals,selection=dict(selection),
        distributions={k:dict(sorted(v.items())) for k,v in distributions.items()},
        diagnostic=dict(diagnostic),difficult_cases=difficult)


def main()->None:
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    state_list=states();groups=[]
    for index,state in enumerate(state_list):
        groups.extend(materialize_state(state))
        if (index+1)%10==0:print(f"stencils {index+1}/{len(state_list)}",flush=True)
    for group in groups:build_features(group)
    topology,topology_report=fit_topology();ranker,ranker_report=fit_ranker(groups)
    validation=evaluate(topology,ranker,groups)
    passed=validation["passed"]
    checkpoints={}
    if passed:
        top_path=OUT/"n3_stencil_topology_validity.pt";scale_path=OUT/"n3_stencil_scale_selector.pt"
        torch.save(dict(status="LOCKED",state_dict=topology.state_dict(),architecture="Linear(1,2)"),top_path)
        torch.save(dict(status="FROZEN_CANDIDATE",state_dict=ranker.state_dict(),exponents=EXPONENTS,features=FEATURES),scale_path)
        checkpoints=dict(topology=str(top_path),scale=str(scale_path))
    report=dict(status="PASSED" if passed else "STOPPED_SCALE_SELECTOR_FAILED",
        teacher_rules=dict(
            topology="valid iff every +/- stencil point has the exact same complete event-signature sequence as the center; smooth objectives are valid",
            numerical="among topology-valid scales, label the scale with minimum absolute error to the autograd teacher derivative; canonical coarser scale wins exact ties"),
        systems=dict(states=len(state_list),groups=len(groups),dimensions="2 plus synthetic 3-6",exponents=EXPONENTS),
        topology_router=topology_report,scale_selector=ranker_report,validation=validation,
        checkpoints=checkpoints,existing_checkpoints_modified=False,
        diagnosis="" if passed else "The smallest shared selector did not generalize the exact teacher scale choice; stop before derivative magnitude learning.",
        elapsed_seconds=time.time()-started)
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
