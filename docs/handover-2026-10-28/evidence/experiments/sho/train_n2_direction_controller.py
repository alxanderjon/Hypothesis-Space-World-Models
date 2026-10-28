#!/usr/bin/env python3
"""Learn and replay only the categorical control in the N2 Powell scaffold.

The frozen scalar line controller and all line-search/state/stopping machinery are
imported unchanged.  This file deliberately keeps the continuous Powell
arithmetic as teacher scaffolding and learns only its finite routing decisions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiments.sho.run_stage1b_learned_powell_2d as base
from experiments.sho.validate_representative_tie_selector import StrictLossSelector


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/sho/stage1B/n2_direction_controller")
LOCAL_CHECKPOINT = OUT / "local_direction_routers.pt"
LINE_CHECKPOINT = base.LINE_CHECKPOINT
SOURCE_500 = Path("outputs/sho/stage1B/learned_powell_2d/per_trajectory_results.csv")
SOURCE_1500 = Path("outputs/sho/stage1B/learned_powell_2d_continuation/continuation_per_trajectory.csv")
SEED = 20260929


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TinyRouter(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def fit_truth_table(name: str, x: torch.Tensor, y: torch.Tensor, outputs: int) -> tuple[TinyRouter, dict]:
    torch.manual_seed(SEED + sum(map(ord, name)))
    model = TinyRouter(x.shape[1], outputs)
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=300,
                            tolerance_grad=1e-14, tolerance_change=1e-16,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        pred = model(x).argmax(1)
    errors = int((pred != y).sum())
    if errors:
        raise AssertionError(f"{name} truth table has {errors} errors")
    changed = any(not torch.equal(initial[k], model.state_dict()[k]) for k in initial)
    return model.eval().requires_grad_(False), {
        "name": name, "cases": len(y), "errors": errors,
        "parameters": sum(p.numel() for p in model.parameters()),
        "random_initialization_changed": changed,
        "weights": {k: v.detach().tolist() for k, v in model.state_dict().items()},
    }


def train_local_routers() -> tuple[dict[str, TinyRouter], list[dict]]:
    # Same-coordinate -> ACTIVE is shared across every dimension and direction.
    basis_x = torch.tensor([[0.0], [1.0]])
    basis_y = torch.tensor([0, 1])
    # Shared conjunction for pre-extrapolation and replacement eligibility.
    and_x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    and_y = torch.tensor([0, 0, 0, 1])
    # Exact comparison is supplied categorically as NEGATIVE/ZERO/POSITIVE.
    sign_x = torch.eye(3, dtype=torch.float64)
    negative_y = torch.tensor([1, 0, 0])
    # Per output slot: KEEP, COPY_OLD_LAST, USE_DISPLACEMENT.
    update_x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    update_y = torch.tensor([0, 1, 2, 2])
    specifications = (
        ("basis_membership", basis_x, basis_y, 2),
        ("boolean_and", and_x, and_y, 2),
        ("strict_negative", sign_x, negative_y, 2),
        ("direction_update_role", update_x, update_y, 3),
    )
    models, reports = {}, []
    for name, x, y, outputs in specifications:
        model, report = fit_truth_table(name, x, y, outputs)
        models[name] = model
        reports.append(report)
    return models, reports


class LearnedDirectionController:
    def __init__(self, models: dict[str, TinyRouter]):
        self.models = models
        self.compare = StrictLossSelector()
        self.counts = {k: 0 for k in (
            "basis", "basis_errors", "rank", "rank_errors", "and", "and_errors",
            "negative", "negative_errors", "update", "update_errors")}

    def _route(self, name: str, values: list[float]) -> int:
        with torch.no_grad():
            return int(self.models[name](torch.tensor([values])).argmax(1)[0])

    def initial_directions(self, dimension: int) -> list[np.ndarray]:
        output = []
        for direction_index in range(dimension):
            vector = np.empty(dimension, dtype=np.float64)
            for coordinate_index in range(dimension):
                same = float(direction_index == coordinate_index)
                value = self._route("basis_membership", [same])
                teacher = int(bool(same))
                self.counts["basis"] += 1
                self.counts["basis_errors"] += int(value != teacher)
                vector[coordinate_index] = float(value)
            output.append(vector)
        return output

    def _negative(self, value: float) -> bool:
        # Reuse locked exact subtraction/sign.  Decide whether value < 0.
        replace, _, state = self.compare.decide(np.array([0.0]), np.array([value]))
        state_value = int(state[0])
        category = {-1: 0, 0: 1, 1: 2}[state_value]
        got = bool(self._route("strict_negative", torch.eye(3)[category].tolist()))
        teacher = value < 0.0
        self.counts["negative"] += 1
        self.counts["negative_errors"] += int(got != teacher)
        return got

    def largest_decrease_index(self, decreases: list[float]) -> int:
        current = 0
        for candidate in range(1, len(decreases)):
            # candidate > current iff current-candidate is strictly negative.
            replace = self._negative(float(decreases[current] - decreases[candidate]))
            teacher = decreases[candidate] > decreases[current]
            self.counts["rank"] += 1
            self.counts["rank_errors"] += int(replace != teacher)
            if replace:
                current = candidate
        return current

    def conjunction(self, left: bool, right: bool) -> bool:
        got = bool(self._route("boolean_and", [float(left), float(right)]))
        teacher = left and right
        self.counts["and"] += 1
        self.counts["and_errors"] += int(got != teacher)
        return got

    def should_replace(self, test: float) -> bool:
        return self._negative(test)

    def update(self, directions: list[np.ndarray], biggest: int,
               displacement: np.ndarray) -> list[np.ndarray]:
        old = [v.copy() for v in directions]
        last = len(old) - 1
        result = []
        for index, vector in enumerate(old):
            flags = [float(index == biggest), float(index == last)]
            role = self._route("direction_update_role", flags)
            teacher = 2 if index == last else (1 if index == biggest else 0)
            self.counts["update"] += 1
            self.counts["update_errors"] += int(role != teacher)
            result.append(vector if role == 0 else (old[last].copy() if role == 1 else displacement.copy()))
        return result


def load_controller() -> LearnedDirectionController:
    payload = torch.load(LOCAL_CHECKPOINT, map_location="cpu", weights_only=False)
    models = {}
    for name, shape in payload["shapes"].items():
        model = TinyRouter(*shape)
        model.load_state_dict(payload["state_dicts"][name])
        models[name] = model.eval().requires_grad_(False)
    return LearnedDirectionController(models)


def alpha_interval(theta: np.ndarray, direction: np.ndarray, lower: np.ndarray,
                   upper: np.ndarray) -> tuple[float, float] | None:
    lo, hi = -math.inf, math.inf
    for axis in range(len(theta)):
        if abs(direction[axis]) <= 1e-18:
            if theta[axis] < lower[axis] or theta[axis] > upper[axis]: return None
            continue
        a = (lower[axis] - theta[axis]) / direction[axis]
        b = (upper[axis] - theta[axis]) / direction[axis]
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return None if (not math.isfinite(lo) or not math.isfinite(hi) or hi-lo <= 1e-15) else (float(lo), float(hi))


def run_trajectory(task: tuple, max_evaluations: int = 1500) -> tuple[base.TrajectoryResult, dict]:
    trajectory_id, position, lower, upper, starting_theta, starting_loss, n1_source = task
    control = load_controller()
    state = base.EvaluationState(position, lower, upper, starting_theta.copy(), starting_loss,
                                 max_evaluations=max_evaluations,
                                 snapshot_budgets=(500, 750, 1000, 1500))
    theta, theta_loss = starting_theta.copy(), starting_loss
    directions = control.initial_directions(len(theta))
    cycles = line_searches = line_converged = line_caps = line_improved = 0
    arbitrary_lines = arbitrary_converged = arbitrary_improved = 0
    displacement_searches = displacement_improved = displacement_outperformed = 0
    direction_updates = line_serial = 0
    stopped = False; stagnant_cycles = 0
    line_events = []; direction_update_evaluations = []; displacement_improvement_evaluations = []
    def account(result):
        nonlocal line_searches,line_converged,line_caps,line_improved,arbitrary_lines,arbitrary_converged,arbitrary_improved
        line_searches += 1; line_converged += int(result.converged); line_caps += int(result.hit_line_cap); line_improved += int(result.improved)
        line_events.append(dict(start_evaluation=state.count-result.evaluations,end_evaluation=state.count,converged=result.converged,hit_line_cap=result.hit_line_cap,improved=result.improved,arbitrary_direction=result.arbitrary_direction))
        if result.arbitrary_direction:
            arbitrary_lines += 1; arbitrary_converged += int(result.converged); arbitrary_improved += int(result.improved)
    while state.count < state.max_evaluations and state.best_loss >= base.SUCCESS_LOSS:
        cycles += 1; state.current_cycle = cycles
        theta_start, loss_start = theta.copy(), theta_loss
        decreases = []
        for direction in directions:
            if state.count >= state.max_evaluations or state.best_loss < base.SUCCESS_LOSS: break
            before = theta_loss; line_serial += 1
            result = base._line_search(state,theta,theta_loss,direction,base.LINE_SEED+100_003*trajectory_id+line_serial)
            account(result); theta,theta_loss=result.theta,result.loss
            decreases.append(max(0.0,before-theta_loss))
        if state.best_loss < base.SUCCESS_LOSS or state.count >= state.max_evaluations or len(decreases)<len(directions): break
        theta_end, loss_end = theta.copy(), theta_loss
        displacement = theta_end-theta_start; displacement_norm=float(np.linalg.norm(displacement))
        biggest_index=control.largest_decrease_index(decreases); biggest_decrease=decreases[biggest_index]
        significant=displacement_norm > base.DISPLACEMENT_TOLERANCE*max(1.0,float(np.linalg.norm(upper-lower)))
        extrapolated=theta_end+displacement
        inside=bool(np.all(extrapolated>=lower) and np.all(extrapolated<=upper))
        if control.conjunction(significant,inside):
            extrapolated_loss=state.evaluate(extrapolated)
            if state.best_loss < base.SUCCESS_LOSS: theta,theta_loss=extrapolated,extrapolated_loss; break
            better=extrapolated_loss<loss_start
            first=loss_start-loss_end-biggest_decrease; second=loss_start-extrapolated_loss
            test=2.0*(loss_start-2.0*loss_end+extrapolated_loss)*first*first-biggest_decrease*second*second
            if control.conjunction(better,control.should_replace(test)) and state.count<state.max_evaluations:
                before=theta_loss; line_serial+=1
                result=base._line_search(state,theta,theta_loss,displacement,base.LINE_SEED+100_003*trajectory_id+line_serial)
                account(result); displacement_searches+=1
                improvement=max(0.0,before-result.loss); displacement_improved+=int(result.improved)
                if result.improved: displacement_improvement_evaluations.append(state.count)
                displacement_outperformed+=int(improvement>max(decreases) if decreases else improvement>0)
                theta,theta_loss=result.theta,result.loss
                directions=control.update(directions,biggest_index,displacement)
                direction_updates+=1; direction_update_evaluations.append(state.count)
        cycle_relative=(loss_start-theta_loss)/max(abs(loss_start),1e-300)
        if cycle_relative<=1e-10 and displacement_norm<=1e-10*max(1.0,float(np.linalg.norm(upper-lower))): stagnant_cycles+=1
        else: stagnant_cycles=0
        if stagnant_cycles>=2: stopped=True
    result=base.TrajectoryResult(trajectory_id,starting_theta,starting_loss,lower,upper,n1_source,state.best_theta,state.best_loss,state.best_loss<base.SUCCESS_LOSS,state.count,cycles,state.first_threshold,state.best_loss_history,line_searches,line_converged,line_caps,line_improved,arbitrary_lines,arbitrary_converged,arbitrary_improved,displacement_searches,displacement_improved,displacement_outperformed,direction_updates,stopped,state.snapshots,state.last_best_evaluation,line_events,direction_update_evaluations,displacement_improvement_evaluations)
    return result, control.counts


def worker(task): return run_trajectory(task)


def main() -> None:
    started=time.time(); OUT.mkdir(parents=True,exist_ok=True)
    before=sha(LINE_CHECKPOINT)
    models,reports=train_local_routers()
    shapes={name:(model.linear.in_features,model.linear.out_features) for name,model in models.items()}
    torch.save(dict(status="LOCAL_OPERATORS_LOCKED",shapes=shapes,state_dicts={n:m.state_dict() for n,m in models.items()},reports=reports,seed=SEED),LOCAL_CHECKPOINT)

    # Generic fresh controls across dimensions and direction-set sizes.
    controls=[]
    for dimension in range(1,9):
        c=LearnedDirectionController(models); dirs=c.initial_directions(dimension)
        assert np.array_equal(np.stack(dirs),np.eye(dimension))
        rng=np.random.default_rng(SEED+dimension)
        for _ in range(1000):
            decreases=rng.integers(0,8,size=dimension).astype(float).tolist()
            assert c.largest_decrease_index(decreases)==int(np.argmax(decreases))
            left, right = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
            assert c.conjunction(left, right) == (left and right)
            biggest=int(np.argmax(decreases)); disp=rng.normal(size=dimension)
            got=c.update(dirs,biggest,disp); expected=[v.copy() for v in dirs]; expected[biggest]=expected[-1]; expected[-1]=disp
            assert all(np.array_equal(a,b) for a,b in zip(got,expected))
        controls.append(dict(dimension=dimension,decisions=c.counts))

    source=list(csv.DictReader(SOURCE_500.open()))
    failure_ids=[int(r["trajectory_id"]) for r in source if r["success_1e-20"]=="0"]
    tasks,_=base._prepare_tasks(); selected=[tasks[i] for i in failure_ids]
    results=[]; routing={}
    workers=min(8,os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,t) for t in selected]
        for future in as_completed(futures):
            result,counts=future.result(); results.append(result); routing[result.trajectory_id]=counts
            if len(results)%10==0: print(f"replay {len(results)}/{len(selected)}",flush=True)
    results.sort(key=lambda r:r.trajectory_id)
    baseline1500={int(r["trajectory_id"]):r for r in csv.DictReader(SOURCE_1500.open()) if r["checkpoint_budget"]=="1500"}
    rows=[]
    for r in results:
        old=baseline1500[r.trajectory_id]
        rows.append(dict(trajectory_id=r.trajectory_id,success=int(r.success),evaluations=r.evaluations,final_loss=r.final_loss,
                         baseline_loss=float(old["best_prefix_loss"]),loss_exact=int(r.final_loss==float(old["best_prefix_loss"])),
                         theta_exact=int(r.final_theta[0]==float(old["best_a"]) and r.final_theta[1]==float(old["best_b"])),
                         **routing[r.trajectory_id]))
    total_errors=sum(sum(v for k,v in item.items() if k.endswith("_errors")) for item in routing.values())
    learned_success=sum(r.success for r in results); full_success=186+learned_success
    previous_success_losses=[float(r["final_loss"]) for r in source if r["success_1e-20"]=="1"]
    result=dict(status="FAILED_FULL_GATE",local_operators_status="LOCKED",overall_checkpoint=None,
                line_checkpoint=str(LINE_CHECKPOINT),line_checkpoint_sha256_before=before,line_checkpoint_sha256_after=sha(LINE_CHECKPOINT),
                line_checkpoint_unchanged=before==sha(LINE_CHECKPOINT),local_reports=reports,total_new_parameters=sum(r["parameters"] for r in reports),
                generic_controls=controls,archived_continuation_trajectories=len(results),routing_errors=total_errors,
                exact_loss_replays=sum(r["loss_exact"] for r in rows),exact_theta_replays=sum(r["theta_exact"] for r in rows),
                continued_successes=learned_success,full_256_successes=full_success,previous_full_successes=250,
                prior_success_median_evaluations=float(np.median([int(r["objective_evaluations"]) for r in source if r["success_1e-20"]=="1"])),
                prior_success_median_final_loss=float(np.median(previous_success_losses)),
                unresolved_ids=[r.trajectory_id for r in results if not r.success],
                diagnosis="Exact teacher-equivalent routing reproduces the deterministic Powell trajectory and therefore cannot eliminate its six direction-update-stagnation outcomes. A distinct improvement teacher/counterfactual direction target is absent.",
                seconds=time.time()-started)
    (OUT/"local_operator_report.json").write_text(json.dumps(dict(status="LOCKED",checkpoint=str(LOCAL_CHECKPOINT),checkpoint_sha256=sha(LOCAL_CHECKPOINT),reports=reports,generic_controls=controls),indent=2)+"\n")
    with (OUT/"continuation_replay.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
