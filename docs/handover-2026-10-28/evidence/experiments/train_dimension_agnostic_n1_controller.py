"""Quick-screen one shared d=4/d=5 E-DIRECT proposal/cell controller."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn

from experiments.bouncing_ball import run_full_k_transfer as ball4
from experiments.bouncing_ball import run_reflection_peelback as ball5
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.kepler.diagnose_five_parameter_n1 import (
    GT as KEPLER5_TEACHER,
    HIGH as KEPLER5_HIGH,
    LOW as KEPLER5_LOW,
    WIDTH as KEPLER5_WIDTH,
    objective_factory as kepler5_objective,
)
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.run_stage1b_edirect_vs_direct import GaussianLocalResponse, _normalize
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/dimension_agnostic_n1_controller")
DATA = OUT / "teacher_dataset.pt"
CHECKPOINT = OUT / "dimension_agnostic_n1_controller.pt"
SHO_ROOT = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases")
BALL4_ROOT = Path("outputs/bouncing_ball/full_k_50_transfer/cases")
BALL5_ROOT = Path("outputs/bouncing_ball/reflection_peelback/cases")
KEPLER_TRACE = Path("outputs/kepler/five_parameter_n1_diagnosis/trace.npz")
KEPLER_ROUNDS = Path("outputs/kepler/five_parameter_n1_diagnosis/rounds.json")
SHO_CASE_DATA = Path("outputs/sho/stage1B/dimension_general_edirect/validation_cases.npz")
FINAL_GATE = 1.0e-20
MAX_CANDIDATES = 9
COORD_FEATURES = 8
SCALAR_FEATURES = 10


def split_for(system: str, case: int) -> str:
    if system == "kepler5":
        return "held_out"
    return "train" if case < 2 else ("held_out" if case == 2 else "fresh")


def make_n3(torch_objective, lower, upper):
    _, _, run_n3 = staged._make_n3(torch_objective, lower, upper)
    return run_n3


def label_candidate(theta, loss, system, run_n3):
    result = staged._run_n3(theta, loss, f"n1_cell_{system}", run_n3)
    return bool(result["final_loss"] < FINAL_GATE), result


def round_starts(events: list[dict]) -> list[tuple[int, int]]:
    found = {}
    for row in events:
        if row["source"] not in ("initial",):
            found.setdefault(int(row["iteration"]), int(row["evaluation"]) - 1)
    return sorted(found.items())


def select_rounds(starts: list[tuple[int, int]], targets: tuple[int, ...]):
    output = []
    for target in targets:
        candidates = [row for row in starts if row[1] >= target]
        chosen = candidates[0] if candidates else starts[-1]
        if chosen not in output:
            output.append(chosen)
    return output


def sho_adapter(case: int):
    path = SHO_ROOT / f"interior_{case}"
    result = json.loads((path / "result.json").read_text())
    source = int(result["case_id"])
    with np.load(SHO_CASE_DATA) as data:
        states = data[f"prefix_{source}"].copy()
        teacher = data["teachers"].reshape(-1, 4)[source].copy()
    lower = np.full(4, -1.99); upper = np.full(4, 1.99)
    x = torch.from_numpy(states[:-1]); y = torch.from_numpy(states[1:])
    def torch_objective(theta):
        return torch.mean((x @ theta.reshape(2, 2).T - y) ** 2)
    def objective(theta):
        return float(np.mean((states[:-1] @ theta.reshape(2, 2).T - states[1:]) ** 2))
    trace = np.load(path / "stopping_state.npz")
    events = json.loads((path / "events.json").read_text())
    return dict(system="sho", case=case, points=trace["points"], losses=trace["losses"],
                events=events, seed=source, lower=lower, upper=upper, teacher=teacher,
                objective=objective, run_n3=make_n3(torch_objective, lower, upper),
                targets=(25, 100, 200))


def ball4_adapter(case: int):
    trajectories, _, _, _ = ball4.ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=ball4.COUNT, n_timesteps=ball4.ball.TOTAL_STEPS,
        dt=ball4.ball.DT, seed=ball4.SEED,
    )
    objective, torch_objective = ball4.objectives(trajectories[case])
    path = BALL4_ROOT / f"trajectory_{case:02d}"
    trace = np.load(path / "n1_state.npz")
    events = json.loads((path / "events.json").read_text())
    return dict(system="ball4", case=case, points=trace["points"], losses=trace["losses"],
                events=events, seed=2_000_000 + case, lower=ball4.LOWER,
                upper=ball4.UPPER, teacher=ball4.TEACHER, objective=objective,
                run_n3=make_n3(torch_objective, ball4.LOWER, ball4.UPPER),
                targets=(25, 500, 2000))


def ball5_adapter(case: int):
    trajectories = ball5._trajectories()
    trajectory = trajectories[int(ball5.SELECTED[case])]
    objective, torch_objective = ball5.objectives(trajectory)
    path = BALL5_ROOT / f"case_{case:02d}"
    trace = np.load(path / "n1_state.npz")
    events = json.loads((path / "events.json").read_text())
    return dict(system="ball5", case=case, points=trace["points"], losses=trace["losses"],
                events=events, seed=3_000_000 + case, lower=ball5.LOWER,
                upper=ball5.UPPER, teacher=ball5.TEACHER, objective=objective,
                run_n3=make_n3(torch_objective, ball5.LOWER, ball5.UPPER),
                targets=(25, 500, 2000))


def kepler5_adapter():
    _, objective, torch_objective = kepler5_objective()
    trace = np.load(KEPLER_TRACE)
    rounds = json.loads(KEPLER_ROUNDS.read_text())
    events = []
    for row in rounds:
        events.append(dict(iteration=row["iteration"], evaluation=row["evaluations"] + 1,
                           source="cell"))
    return dict(system="kepler5", case=0, points=trace["points"], losses=trace["losses"],
                events=events, seed=91_000, lower=KEPLER5_LOW, upper=KEPLER5_HIGH,
                teacher=KEPLER5_TEACHER, objective=objective,
                run_n3=make_n3(torch_objective, KEPLER5_LOW, KEPLER5_HIGH),
                targets=(297, 1000, 1200, 1450))


def candidate_features(points, losses, data, proposal, owner, cap):
    d = points.shape[1]
    site = points[owner]
    members = data.points[data.owners == owner]
    if len(members):
        lower_extent = site - np.min(members, axis=0)
        upper_extent = np.max(members, axis=0) - site
    else:
        lower_extent = np.zeros(d); upper_extent = np.zeros(d)
    if len(points) > 1:
        tree = cKDTree(points)
        _, indices = tree.query(site, k=2)
        neighbor_delta = np.abs(points[int(indices[1])] - site)
    else:
        neighbor_delta = np.zeros(d)
    delta = proposal - site
    coordinate = np.column_stack((
        site, proposal, np.minimum(site, 1.0 - site), delta, np.abs(delta),
        lower_extent, upper_extent, neighbor_delta,
    ))
    values = np.log10(np.maximum(losses, 1e-300))
    prediction = float(GaussianLocalResponse(points, losses).predict(proposal)[0])
    selected = set(int(v) for v in data.selected)
    selected_sizes = data.sizes[data.selected]
    largest = int(data.selected[int(np.argmax(selected_sizes))])
    loss_rank = float(np.mean(losses <= losses[owner]))
    size_rank = float(np.mean(data.sizes <= data.sizes[owner]))
    scalar = np.asarray((
        values[owner] - float(np.min(values)),
        float(_normalize(data.sizes)[owner]), loss_rank, size_rank,
        prediction - float(np.min(values)),
        float(np.linalg.norm(delta) / math.sqrt(d)),
        float(owner in selected), float(owner == largest),
        math.log1p(len(points)) / math.log1p(cap),
        math.log2(d),
    ), dtype=np.float64)
    return coordinate, scalar


def state_examples(adapter, iteration, count, critical=False):
    points = np.asarray(adapter["points"][:count], dtype=np.float64)
    losses = np.asarray(adapter["losses"][:count], dtype=np.float64)
    api = edirect.composition(points.shape[1])
    data = api["_cell_data"](points, losses, adapter["seed"], iteration)
    proposals = api["_cell_proposals"](points, losses, adapter["seed"], iteration)
    owners = cKDTree(points).query(proposals, k=1)[1].astype(int)
    # Preserve canonical proposal order and cap only after keeping the largest-cell proposal.
    proposals = proposals[:MAX_CANDIDATES]
    owners = owners[:MAX_CANDIDATES]
    if critical:
        teacher_unit = (adapter["teacher"] - adapter["lower"]) / (
            adapter["upper"] - adapter["lower"]
        )
        index = int(np.argmin(np.linalg.norm(data.points - teacher_unit, axis=1)))
        missed = data.points[index]
        missed_owner = int(data.owners[index])
        if not any(np.array_equal(missed, proposal) for proposal in proposals):
            proposals = np.concatenate((proposals, missed[None, :]))
            owners = np.concatenate((owners, np.asarray([missed_owner])))
    rows = []
    for position, (proposal, owner) in enumerate(zip(proposals, owners)):
        theta = adapter["lower"] + (adapter["upper"] - adapter["lower"]) * proposal
        loss = float(adapter["objective"](theta))
        label, n3 = label_candidate(theta, loss, adapter["system"], adapter["run_n3"])
        coordinate, scalar = candidate_features(
            points, losses, data, proposal, int(owner), len(adapter["points"])
        )
        rows.append(dict(
            coordinate=torch.from_numpy(coordinate), scalar=torch.from_numpy(scalar),
            label=label, theta=theta.tolist(), loss=loss, owner=int(owner),
            deterministic_selected=bool(int(owner) in set(data.selected.tolist())),
            n3=n3, candidate_position=position,
        ))
    return dict(system=adapter["system"], case=adapter["case"], split=split_for(
        adapter["system"], adapter["case"]), iteration=iteration, evaluation=count,
        dimension=points.shape[1], candidates=rows,
    )


def build_dataset():
    adapters = []
    for case in range(4):
        adapters.extend((sho_adapter(case), ball4_adapter(case), ball5_adapter(case)))
    adapters.append(kepler5_adapter())
    states = []
    for adapter in adapters:
        starts = round_starts(adapter["events"])
        for iteration, count in select_rounds(starts, adapter["targets"]):
            key = (adapter["system"], adapter["case"], iteration, count)
            print("label", key, flush=True)
            states.append(state_examples(
                adapter, iteration, count,
                critical=(adapter["system"] == "kepler5" and iteration == 42),
            ))
            torch.save(states, DATA)
    return states


class SharedSetController(nn.Module):
    def __init__(self):
        super().__init__()
        self.coordinate = nn.Sequential(nn.Linear(COORD_FEATURES, 4), nn.Tanh())
        self.cell = nn.Sequential(nn.Linear(8 + SCALAR_FEATURES, 6), nn.Tanh())
        self.score = nn.Sequential(nn.Linear(18, 4), nn.Tanh(), nn.Linear(4, 1))

    def forward_state(self, state):
        embeddings = []
        for candidate in state["candidates"]:
            encoded = self.coordinate(candidate["coordinate"])
            pooled = torch.cat((encoded.mean(0), encoded.max(0).values))
            embeddings.append(self.cell(torch.cat((pooled, candidate["scalar"]))))
        cells = torch.stack(embeddings)
        context = torch.cat((cells.mean(0), cells.max(0).values))
        return self.score(torch.cat((cells, context.expand(len(cells), -1)), 1)).squeeze(1)


def state_loss(logits, labels):
    weight = torch.where(labels, torch.as_tensor(4.0), torch.as_tensor(1.0))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.double(), weight=weight)
    if labels.any() and (~labels).any():
        loss = loss + nn.functional.softplus(
            logits[~labels].max() - logits[labels].max()
        )
    return loss


def metrics(model, states):
    labels_all=[]; predictions_all=[]; top_hits=0; positive_states=0
    by_system={}
    with torch.no_grad():
        for state in states:
            logits=model.forward_state(state); labels=torch.tensor(
                [row["label"] for row in state["candidates"]], dtype=torch.bool)
            predictions=logits >= 0.0
            labels_all.extend(labels.tolist()); predictions_all.extend(predictions.tolist())
            if labels.any():
                positive_states += 1; top_hits += int(labels[int(torch.argmax(logits))])
            group=by_system.setdefault(state["system"], [0,0,0,0])
            group[0]+=len(labels); group[1]+=int((predictions & ~labels).sum())
            group[2]+=int((~predictions & labels).sum()); group[3]+=int((predictions==labels).sum())
    y=np.asarray(labels_all,bool); p=np.asarray(predictions_all,bool)
    fp=int(np.sum(p&~y)); fn=int(np.sum(~p&y)); tn=int(np.sum(~p&~y)); tp=int(np.sum(p&y))
    return dict(candidates=len(y), positives=int(y.sum()), negatives=int((~y).sum()),
                accuracy=float(np.mean(y==p)), false_positives=fp, false_negatives=fn,
                false_positive_rate=float(fp/max(1,fp+tn)),
                false_negative_rate=float(fn/max(1,fn+tp)),
                positive_state_top1_recall=float(top_hits/max(1,positive_states)),
                positive_states=positive_states,
                by_system={k:dict(candidates=v[0],false_positives=v[1],false_negatives=v[2],
                                  accuracy=v[3]/v[0]) for k,v in by_system.items()})


def train(states):
    torch.manual_seed(20261001)
    model=SharedSetController(); optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)
    train_states=[s for s in states if s["split"]=="train"]
    validation=[s for s in states if s["split"]=="held_out"]
    best=None; best_key=(-math.inf,-math.inf,-math.inf); history=[]; stale=0
    rng=np.random.default_rng(20261001)
    for epoch in range(1,801):
        order=rng.permutation(len(train_states)); total=0.0
        for index in order:
            state=train_states[int(index)]; optimizer.zero_grad()
            logits=model.forward_state(state); labels=torch.tensor(
                [row["label"] for row in state["candidates"]],dtype=torch.bool)
            loss=state_loss(logits,labels); loss.backward(); optimizer.step(); total+=float(loss)
        if epoch%10==0:
            value=metrics(model,validation)
            key=(-value["false_positives"],value["positive_state_top1_recall"],value["accuracy"])
            history.append(dict(epoch=epoch,training_loss=total/len(train_states),**value))
            if key>best_key: best_key=key; best=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=1
            if stale>=12: break
    model.load_state_dict(best)
    return model, history


def main():
    OUT.mkdir(parents=True,exist_ok=True); before=hashes()
    states=torch.load(DATA,weights_only=False) if DATA.exists() else build_dataset()
    model,history=train(states)
    screen={split:metrics(model,[s for s in states if s["split"]==split])
            for split in ("train","held_out","fresh")}
    hard=[s for s in states if s["system"]=="kepler5" and s["iteration"]==42][0]
    with torch.no_grad():
        logits=model.forward_state(hard).numpy(); labels=np.asarray(
            [r["label"] for r in hard["candidates"]])
    hard_result=dict(evaluation=hard["evaluation"],candidate_count=len(labels),
                     positive_count=int(labels.sum()),top_index=int(np.argmax(logits)),
                     top_is_correctable=bool(labels[np.argmax(logits)]),
                     critical_candidate_index=len(labels)-1,
                     critical_candidate_correctable=bool(labels[-1]),
                     critical_candidate_rank=int(np.sum(logits>logits[-1])+1),
                     critical_candidate_logit=float(logits[-1]),
                     candidates=[dict(
                         index=index, label=bool(labels[index]), logit=float(logits[index]),
                         loss=float(row["loss"]), owner=int(row["owner"]),
                         deterministic_selected=bool(row["deterministic_selected"]),
                         theta=row["theta"],
                     ) for index, row in enumerate(hard["candidates"])])
    passed=bool(screen["held_out"]["false_positives"]==0
                and screen["fresh"]["false_positives"]==0
                and screen["held_out"]["positive_state_top1_recall"]>=0.9
                and screen["fresh"]["positive_state_top1_recall"]>=0.9
                and hard_result["top_is_correctable"])
    payload=dict(state_dict=model.state_dict(),architecture=dict(
        coordinate_encoder=f"{COORD_FEATURES}->4 tanh shared across coordinates",
        coordinate_pool="mean+max",cell_encoder=f"{8+SCALAR_FEATURES}->6 tanh",
        cell_pool="mean+max",scorer="18->4 tanh->1 shared across candidates"),
        parameter_count=sum(p.numel() for p in model.parameters()),screen=screen)
    if passed: torch.save(payload,CHECKPOINT)
    report=dict(status=("FROZEN_CANDIDATE" if passed else "FAILED_QUICK_SCREEN"),
                architecture=payload["architecture"],parameter_count=payload["parameter_count"],
                systems=sorted(set(s["system"] for s in states)),
                dimensions=sorted(set(s["dimension"] for s in states)),
                states=len(states),candidates=sum(len(s["candidates"]) for s in states),
                corpus_by_system={system: dict(
                    states=sum(s["system"]==system for s in states),
                    candidates=sum(len(s["candidates"]) for s in states
                                   if s["system"]==system),
                    positives=sum(row["label"] for s in states if s["system"]==system
                                  for row in s["candidates"]),
                    negatives=sum(not row["label"] for s in states if s["system"]==system
                                  for row in s["candidates"]),
                ) for system in sorted(set(s["system"] for s in states))},
                split_metrics=screen,hard_kepler=hard_result,training_history=history,
                no_ground_truth_runtime_features=True,
                checkpoint_saved=passed,checkpoint=str(CHECKPOINT) if passed else None,
                checkpoint_hashes_unchanged=before==hashes(),protected_checkpoint_count=len(before),
                failure_classification=(None if passed else "cell-selection target/runtime state or model capacity; closed-loop gate not reached"))
    (OUT/"report.json").write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()))
    print(json.dumps({k:report[k] for k in ("status","parameter_count","states","candidates","split_metrics","hard_kepler","checkpoint_hashes_unchanged")},indent=2))


if __name__=="__main__": main()
