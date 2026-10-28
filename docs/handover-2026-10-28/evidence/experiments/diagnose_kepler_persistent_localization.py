#!/usr/bin/env python3
"""Diagnose saved d=5 Kepler localization failures without advancing N1."""
from __future__ import annotations

import argparse
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
from experiments.continue_kepler_incomplete_handoffs import SavedPrefixObjective
from experiments.sho import audit_stage1b_po_cell_selection as po_audit
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

CASE_IDS = (9, 22, 28, 30, 33, 35, 41, 53, 54, 58)
CAP = 20_000
TARGET_COUNTS = (5_000, 10_000, 15_000, 20_000)
SOURCE = Path("outputs/d5_handoff_continuation_20000/cases")
OUT = Path("outputs/d5_kepler_persistent_localization_diagnosis")
CASES = OUT / "cases"


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _reconstruct(case_id, objective, lower, upper):
    saved = np.load(SOURCE / f"case_{case_id:03d}" / "n1_state.npz")
    points = np.asarray(saved["points"], dtype=np.float64)
    losses = np.asarray(saved["losses"], dtype=np.float64)
    if points.shape != (CAP, 5) or losses.shape != (CAP,):
        raise RuntimeError(f"case {case_id}: invalid saved 20k state")
    replay = SavedPrefixObjective(lower, upper, points, losses, objective)
    runner = production.N1Runner(replay, lower, upper, 6_300_000 + case_id)
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        runner.run_to(CAP)
    finally:
        production.N1_GATE = old_gate
    if not np.array_equal(np.asarray(runner.points), points):
        raise RuntimeError(f"case {case_id}: 20k point replay mismatch")
    if not np.array_equal(np.asarray(runner.losses), losses):
        raise RuntimeError(f"case {case_id}: 20k loss replay mismatch")
    if replay.replayed != CAP or replay.live_calls:
        raise RuntimeError(f"case {case_id}: replay call accounting mismatch")
    return runner


def _round_starts(events):
    starts = {}
    for event in events:
        if event["source"] == "initial":
            continue
        iteration = int(event["iteration"])
        starts.setdefault(iteration, int(event["evaluation"]) - 1)
    return starts


def _representatives(events):
    starts = _round_starts(events)
    choices = []
    used = set()
    for target in TARGET_COUNTS:
        candidates = [(abs(count - target), count, iteration)
                      for iteration, count in starts.items() if count < CAP]
        _, count, iteration = min(candidates)
        if iteration not in used:
            choices.append((target, count, iteration)); used.add(iteration)
    return choices


def _rejection_reason(owner, selected, metadata):
    if int(owner) in set(map(int, selected)):
        return "OWNER_POTENTIALLY_OPTIMAL_WITHIN_CELL_CANDIDATE_NOT_CHOSEN"
    group_index = int(metadata["cell_group"][owner])
    group = metadata["groups"][group_index]
    if int(group["sample_index"]) != int(owner):
        return "SAME_SIZE_NONREPRESENTATIVE"
    if bool(metadata["pareto_dominated"][owner]):
        return "PARETO_DOMINATED_SIZE_LOSS"
    if group_index not in metadata["hull"]:
        return "NOT_LOWER_RIGHT_HULL"
    if not bool(group.get("epsilon_pass", False)):
        return "FAILED_EPSILON_SLOPE_TEST"
    return "NOT_IN_POTENTIALLY_OPTIMAL_OUTPUT"


def _candidate_key(point):
    return np.asarray(point, dtype=np.float64).tobytes()


def _audit_state(runner, full_tree, objective, teacher, lower, upper,
                 target, count, iteration):
    points = np.asarray(runner.points[:count], dtype=np.float64)
    losses = np.asarray(runner.losses[:count], dtype=np.float64)
    data = runner.api["_cell_data"](points, losses, runner.case_id, iteration)
    selected = np.asarray(data.selected, dtype=np.int64)
    production_proposals = runner.api["_cell_proposals"](
        points, losses, runner.case_id, iteration
    )
    independent, metadata = po_audit._standalone_po_selection(data.sizes, losses)
    if not np.array_equal(selected, independent):
        raise RuntimeError("potentially-optimal audit mismatch")
    surface = production.edirect.teacher.GaussianLocalResponse(points, losses)
    candidate_theta = lower + (upper - lower) * data.points
    gt_unit = (teacher - lower) / (upper - lower)
    normalized_gt_distance = np.linalg.norm(data.points - gt_unit[None, :], axis=1)
    current_best = int(np.argmin(losses))
    incumbent = points[current_best]
    scalar_halfwidth = (count + 1) ** (-1.0 / len(lower))
    teacher_compatible = np.all(np.abs(data.points - gt_unit[None, :]) <= scalar_halfwidth,
                                axis=1)
    selected_owner = np.isin(data.owners, selected)
    selected_proposal_compatible = int(sum(
        np.all(np.abs(point - gt_unit) <= scalar_halfwidth)
        for point in production_proposals
    ))
    # Candidates that remained unevaluated throughout the complete saved run.
    ever_distance = full_tree.query(data.points, k=1)[0]
    unevaluated = ever_distance > 0.0
    proposal_keys = {_candidate_key(point) for point in production_proposals}
    not_production_proposal = np.asarray(
        [_candidate_key(point) not in proposal_keys for point in data.points], dtype=bool
    )
    eligible = unevaluated & not_production_proposal
    unselected_owner = ~selected_owner
    predictions = surface.predict(data.points)

    shortlist = []
    def add(indices, source):
        for value in np.atleast_1d(indices):
            index = int(value)
            if not eligible[index]:
                continue
            key = _candidate_key(data.points[index])
            if any(row["key"] == key for row in shortlist):
                continue
            shortlist.append(dict(index=index, source=source, key=key))

    index_set = np.flatnonzero(eligible & unselected_owner)
    if len(index_set):
        add(index_set[np.argsort(normalized_gt_distance[index_set])[:4]],
            "nearest_teacher_unselected_cell")
        add(index_set[np.argsort(predictions[index_set])[:4]],
            "best_surrogate_unselected_cell")
        owner_order = np.unique(data.owners[index_set])
        owner_order = owner_order[np.argsort(losses[owner_order])[:4]]
        for owner in owner_order:
            local = index_set[data.owners[index_set] == owner]
            add([local[int(np.argmin(predictions[local]))]],
                "best_surrogate_low_loss_unselected_owner")
    selected_set = np.flatnonzero(eligible & selected_owner)
    if len(selected_set):
        add(selected_set[np.argsort(normalized_gt_distance[selected_set])[:2]],
            "nearest_teacher_selected_cell_not_chosen")

    records = []
    for item in shortlist:
        index = item["index"]
        theta = candidate_theta[index]
        loss = float(objective(theta))
        owner = int(data.owners[index])
        records.append(dict(
            candidate_index=index, shortlist_source=item["source"],
            unit_point=data.points[index].tolist(), theta=theta.tolist(),
            objective_loss=loss, would_be_incumbent=bool(loss < losses[current_best]),
            normalized_ground_truth_distance=float(normalized_gt_distance[index]),
            owner_site_index=owner, owner_site_loss=float(losses[owner]),
            owner_cell_size=float(data.sizes[owner]),
            owner_potentially_optimal=bool(owner in set(map(int, selected))),
            production_rejection_reason=_rejection_reason(owner, selected, metadata),
            surrogate_prediction=float(predictions[index]),
            candidate_teacher_compatible_box=bool(teacher_compatible[index]),
        ))

    trace = next((trace for trace in runner.traces
                  if int(trace["iteration"]) == iteration), None)
    greedy = None
    if trace is not None:
        greedy_point = np.asarray(trace["point"], dtype=np.float64)
        matches = np.flatnonzero(np.all(np.asarray(runner.points) == greedy_point, axis=1))
        greedy = dict(
            site_count=int(trace["site_count"]), chosen_start=int(trace["chosen_start"]),
            prediction=float(trace["prediction"]), unit_point=greedy_point.tolist(),
            normalized_ground_truth_distance=float(np.linalg.norm(greedy_point - gt_unit)),
            evaluated_at=(int(matches[0]) + 1 if len(matches) else None),
            evaluated_loss=(float(runner.losses[int(matches[0])]) if len(matches) else None),
        )
    nearest_raw = int(np.argmin(normalized_gt_distance))
    return dict(
        target_evaluation=target, round_start_evaluations=count, iteration=iteration,
        evaluated_sites=count, incumbent_index=current_best,
        incumbent_loss=float(losses[current_best]),
        incumbent_normalized_ground_truth_distance=float(np.linalg.norm(incumbent-gt_unit)),
        raw_candidate_count=int(len(data.points)),
        potentially_optimal_cell_count=int(len(selected)),
        production_cell_proposal_count=int(len(production_proposals)),
        unevaluated_raw_candidate_count=int(np.sum(unevaluated)),
        nearest_raw_candidate_normalized_ground_truth_distance=float(
            normalized_gt_distance[nearest_raw]),
        nearest_raw_candidate_owner=int(data.owners[nearest_raw]),
        nearest_raw_candidate_owner_potentially_optimal=bool(selected_owner[nearest_raw]),
        teacher_compatible_raw_candidate_count=int(np.sum(teacher_compatible)),
        teacher_compatible_owner_count=int(len(np.unique(data.owners[teacher_compatible]))),
        teacher_compatible_candidates_with_selected_owner=int(
            np.sum(teacher_compatible & selected_owner)),
        teacher_compatible_production_proposal_count=selected_proposal_compatible,
        selected_cells=selected.tolist(), production_proposals=production_proposals.tolist(),
        continuous_surrogate_proposal=greedy, shortlist=records,
    )


def _candidate_handoff(record, state, objective, torch_objective, trajectory,
                       teacher, lower, upper, case_id):
    evaluation = int(state["round_start_evaluations"]) + 1
    theta = np.asarray(record["theta"], dtype=np.float64)
    width = upper - lower
    radius = width / evaluation ** (1.0 / len(lower))
    endpoint = dict(
        theta=theta, loss=float(record["objective_loss"]), evaluations=evaluation,
        lower=np.maximum(lower, theta-radius), upper=np.minimum(upper, theta+radius),
        stop_reason="offline_unevaluated_candidate_diagnostic",
    )
    finish = production._finish(
        "kepler5", trajectory, objective, torch_objective, endpoint, teacher,
        f"unevaluated_candidate_{case_id}_{evaluation}",
    )
    return dict(
        target_evaluation=state["target_evaluation"],
        round_start_evaluations=state["round_start_evaluations"],
        counterfactual_evaluation=evaluation,
        candidate={key: value for key, value in record.items()},
        offline_teacher_inside_n2_box=bool(
            np.all(teacher >= endpoint["lower"]) and
            np.all(teacher <= endpoint["upper"])
        ),
        **finish,
    )


def run_case(case_id: int) -> dict:
    if case_id not in CASE_IDS:
        raise ValueError(case_id)
    torch.set_num_threads(1)
    result_path = CASES / f"case_{case_id:03d}" / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trajectory, spec = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    runner = _reconstruct(case_id, objective, lower, upper)
    points = np.asarray(runner.points, dtype=np.float64)
    full_tree = cKDTree(points)
    states = [
        _audit_state(runner, full_tree, objective, teacher, lower, upper, *choice)
        for choice in _representatives(runner.events)
    ]
    checks = []
    first = None
    for state in sorted(states, key=lambda row: row["round_start_evaluations"]):
        improving = [row for row in state["shortlist"] if row["would_be_incumbent"]]
        if not improving:
            continue
        # At most three checks per representative state: best true loss,
        # closest to teacher for coverage diagnosis, and best surrogate score.
        chosen = []
        for key in (lambda row: row["objective_loss"],
                    lambda row: row["normalized_ground_truth_distance"],
                    lambda row: row["surrogate_prediction"]):
            row = min(improving, key=key)
            if row not in chosen:
                chosen.append(row)
        for candidate in chosen:
            check = _candidate_handoff(candidate, state, objective, torch_objective,
                                       trajectory, teacher, lower, upper, case_id)
            checks.append(check)
            if check["recovered"] and first is None:
                first = check
        if first is not None:
            break

    if first is not None:
        classification = (
            "AVAILABLE_BUT_NOT_SELECTED"
            if not first["candidate"]["owner_potentially_optimal"] else
            "SELECTED_BUT_NOT_REFINED_ENOUGH"
        )
    else:
        any_coverage = any(state["teacher_compatible_raw_candidate_count"] > 0
                           for state in states)
        classification = "STILL_UNRESOLVED" if any_coverage else "PROPOSAL/COVERAGE_FAILURE"
    result = dict(
        case_id=case_id, initial_condition=spec,
        exact_20000_reconstruction=True, representative_states=states,
        candidate_handoff_checks=checks, first_successful_unevaluated_candidate=first,
        primary_classification=classification,
        elapsed_seconds=float(time.monotonic()-started),
    )
    _atomic_json(result_path, result)
    return result


def recheck_direct_candidates(case_id: int) -> dict:
    """Test exposed candidate regions directly, retaining incumbent eligibility."""
    result_path = CASES / f"case_{case_id:03d}" / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("direct_candidate_checks_complete", False):
        return result
    trajectory, _ = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    checks = []
    first = None
    for state in sorted(result["representative_states"],
                        key=lambda row: row["round_start_evaluations"]):
        candidates = state["shortlist"]
        if not candidates:
            continue
        chosen = []
        for key in (lambda row: row["objective_loss"],
                    lambda row: row["normalized_ground_truth_distance"],
                    lambda row: row["surrogate_prediction"]):
            row = min(candidates, key=key)
            if row not in chosen:
                chosen.append(row)
        for candidate in chosen:
            check = _candidate_handoff(candidate, state, objective, torch_objective,
                                       trajectory, teacher, lower, upper, case_id)
            checks.append(check)
            if check["recovered"] and first is None:
                first = check
        if first is not None:
            break
    result["candidate_handoff_checks"] = checks
    result["first_successful_unevaluated_candidate"] = first
    result["direct_candidate_checks_complete"] = True
    if first is not None:
        result["primary_classification"] = (
            "AVAILABLE_BUT_NOT_SELECTED"
            if not first["candidate"]["owner_potentially_optimal"] else
            "SELECTED_BUT_NOT_REFINED_ENOUGH"
        )
    else:
        any_coverage = any(state["teacher_compatible_raw_candidate_count"] > 0
                           for state in result["representative_states"])
        result["primary_classification"] = (
            "STILL_UNRESOLVED" if any_coverage else "PROPOSAL/COVERAGE_FAILURE"
        )
    _atomic_json(result_path, result)
    return result


def check_nearest_evaluated(case_id: int) -> dict:
    result_path = CASES / f"case_{case_id:03d}" / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("nearest_evaluated_point_handoff") is not None:
        return result
    trajectory, _ = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    saved = np.load(SOURCE / f"case_{case_id:03d}" / "n1_state.npz")
    points = np.asarray(saved["points"], dtype=np.float64)
    losses = np.asarray(saved["losses"], dtype=np.float64)
    gt_unit = (teacher-lower)/(upper-lower)
    distances = np.linalg.norm(points-gt_unit[None,:],axis=1)
    index = int(np.argmin(distances)); evaluation=index+1
    theta=lower+(upper-lower)*points[index]
    radius=(upper-lower)/evaluation**(1.0/len(lower))
    endpoint=dict(theta=theta,loss=float(losses[index]),evaluations=evaluation,
        lower=np.maximum(lower,theta-radius),upper=np.minimum(upper,theta+radius),
        stop_reason="offline_nearest_evaluated_point_diagnostic")
    finish=production._finish("kepler5",trajectory,objective,torch_objective,
        endpoint,teacher,f"nearest_evaluated_{case_id}_{evaluation}")
    result["nearest_evaluated_point_handoff"]=dict(
        evaluation=evaluation,theta=theta.tolist(),loss=float(losses[index]),
        normalized_ground_truth_distance=float(distances[index]),
        was_incumbent_at_evaluation=bool(losses[index] == np.min(losses[:evaluation])),
        offline_teacher_inside_n2_box=bool(np.all(teacher>=endpoint["lower"]) and
            np.all(teacher<=endpoint["upper"])),**finish)
    _atomic_json(result_path,result);return result


def _run_tasks(indices, workers):
    for offset in range(0, len(indices), workers):
        children=[]
        for index in indices[offset:offset+workers]:
            command=[sys.executable,"-m",__spec__.name,"--case",str(index)]
            children.append((index,subprocess.Popen(command,env=os.environ.copy())))
        failed=[(index,child.wait()) for index,child in children]
        failed=[item for item in failed if item[1]]
        if failed: raise RuntimeError(f"diagnostic case failures: {failed}")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",type=int)
    parser.add_argument("--direct-candidate-case",type=int)
    parser.add_argument("--direct-candidate-all",action="store_true")
    parser.add_argument("--nearest-evaluated-case",type=int)
    parser.add_argument("--nearest-evaluated-all",action="store_true")
    parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);CASES.mkdir(parents=True,exist_ok=True)
    if args.case is not None:
        row=run_case(args.case);first=row["first_successful_unevaluated_candidate"]
        print(json.dumps(dict(case=args.case,classification=row["primary_classification"],
            first_round=None if first is None else first["round_start_evaluations"],
            checks=len(row["candidate_handoff_checks"]),seconds=row["elapsed_seconds"]),indent=2),flush=True)
        return
    if args.direct_candidate_case is not None:
        row=recheck_direct_candidates(args.direct_candidate_case)
        first=row["first_successful_unevaluated_candidate"]
        print(json.dumps(dict(case=args.direct_candidate_case,
            classification=row["primary_classification"],
            first_round=None if first is None else first["round_start_evaluations"],
            checks=len(row["candidate_handoff_checks"])),indent=2),flush=True)
        return
    if args.direct_candidate_all:
        before=hashes();_run_direct_tasks(list(CASE_IDS),max(1,args.workers))
        rows=[json.loads((CASES/f"case_{i:03d}"/"result.json").read_text()) for i in CASE_IDS]
        after=hashes();_write_report(rows,before,after,0.0)
        return
    if args.nearest_evaluated_case is not None:
        row=check_nearest_evaluated(args.nearest_evaluated_case)
        check=row["nearest_evaluated_point_handoff"]
        print(json.dumps(dict(case=args.nearest_evaluated_case,
            evaluation=check["evaluation"],recovered=check["recovered"]),indent=2),flush=True)
        return
    if args.nearest_evaluated_all:
        before=hashes();_run_named_tasks(list(CASE_IDS),max(1,args.workers),"--nearest-evaluated-case")
        rows=[json.loads((CASES/f"case_{i:03d}"/"result.json").read_text()) for i in CASE_IDS]
        after=hashes();_write_report(rows,before,after,0.0);return
    before=hashes();_atomic_json(OUT/"checkpoint_hashes_before.json",before)
    started=time.monotonic();_run_tasks(list(CASE_IDS),max(1,args.workers))
    rows=[json.loads((CASES/f"case_{i:03d}"/"result.json").read_text()) for i in CASE_IDS]
    after=hashes();_write_report(rows,before,after,float(time.monotonic()-started))


def _run_direct_tasks(indices,workers):
    _run_named_tasks(indices,workers,"--direct-candidate-case")


def _run_named_tasks(indices,workers,flag):
    for offset in range(0,len(indices),workers):
        children=[]
        for index in indices[offset:offset+workers]:
            command=[sys.executable,"-m",__spec__.name,flag,str(index)]
            children.append((index,subprocess.Popen(command,env=os.environ.copy())))
        failed=[(index,child.wait()) for index,child in children]
        failed=[item for item in failed if item[1]]
        if failed: raise RuntimeError(f"direct candidate check failures: {failed}")


def _write_report(rows,before,after,elapsed):
    _atomic_json(OUT/"checkpoint_hashes_after.json",after)
    classes={name:sum(row["primary_classification"]==name for row in rows) for name in
             ("AVAILABLE_BUT_NOT_SELECTED","SELECTED_BUT_NOT_REFINED_ENOUGH",
              "PROPOSAL/COVERAGE_FAILURE","STILL_UNRESOLVED")}
    report=dict(cases=len(rows),classifications=classes,
        successful_unevaluated_candidates=sum(row["first_successful_unevaluated_candidate"] is not None for row in rows),
        checkpoint_hashes_unchanged=before==after,elapsed_seconds=float(elapsed),cases_detail=rows)
    _atomic_json(OUT/"report.json",report)
    print(json.dumps({key:report[key] for key in ("cases","classifications",
        "successful_unevaluated_candidates","checkpoint_hashes_unchanged")},indent=2),flush=True)


if __name__=="__main__":main()
