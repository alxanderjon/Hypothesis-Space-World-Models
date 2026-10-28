#!/usr/bin/env python3
"""Audit immediate candidate labels against short unchanged-N1 continuations.

This is an offline counterfactual diagnostic.  One saved reservoir candidate is
forced into an exact saved N1 state, after which the same deterministic N1
state machine continues.  Every subsequently evaluated point is tested using
the production count-derived N2 box and unchanged N2/N3.  The first successful
handoff is absorbing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.train_n1_candidate_relational_depth_screen import (
    DeepCandidateModel, GraphState,
)


torch.set_num_threads(1)
ROOT = Path("outputs/n1_candidate_handoff_supervision")
DATASET = ROOT / "dataset.json"
STATES = ROOT / "states.json"
HARD = ROOT / "hard_cases.jsonl"
MODEL = Path("outputs/n1_candidate_top_of_list_screen/candidate_selection_hardest_bad_screen_only.pt")
OUT = Path("outputs/n1_candidate_long_horizon_label_audit")
BRANCHES = OUT / "branches"
PERSISTENT = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_id(row: dict) -> str:
    return f"{row['state_id']}__{row['example_id'][:16]}"


def load_corpus():
    rows = [row for row in json.loads(DATASET.read_text())
            if row["task"] == "candidate_selection"
            and row["system"] == "kepler5"
            and int(row["case_id"]) in PERSISTENT]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    metadata = {row["state_id"]: row for row in json.loads(STATES.read_text())}
    state_objects = {state_id: GraphState(metadata[state_id], examples)
                     for state_id, examples in grouped.items()}
    checkpoint = torch.load(MODEL, weights_only=False)
    model = DeepCandidateModel(32, 3)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scores = {}
    with torch.no_grad():
        for state_id, examples in grouped.items():
            values = model(state_objects[state_id], examples).cpu().numpy()
            scores[state_id] = [float(value) for value in values]
    hard_tags = defaultdict(set)
    with HARD.open() as stream:
        for line in stream:
            item = json.loads(line)
            hard_tags[item["example_id"]].update(item["hard_case_tags"])
    handoff = defaultdict(list)
    for row in json.loads(DATASET.read_text()):
        if row["task"] == "handoff_promotion":
            handoff[row["state_id"]].append(row)
    # Importing the frozen float32 ranker module changes Torch's process-wide
    # default.  Restore the production float64 numerical contract before any
    # E-DIRECT/Powell replay.
    torch.set_default_dtype(torch.float64)
    return grouped, metadata, scores, hard_tags, handoff


def chosen_branches(grouped, scores, hard_tags, handoff, only_failed=True):
    baseline_report = json.loads(Path(
        "outputs/n1_candidate_top_of_list_screen/report.json").read_text())
    failed = {row["state_id"] for row in
              baseline_report["persistent_kepler"]["remaining_bad_wins"]}
    chosen = []
    excluded = []
    for state_id, examples in sorted(grouped.items()):
        if only_failed and state_id not in failed:
            continue
        if not any(row["target_good"] for row in examples):
            continue
        known_handoff = [row for row in handoff.get(state_id, ()) if row["target_good"]]
        if known_handoff:
            excluded.append(dict(state_id=state_id,
                                 reason="SAVED_EVALUATED_HANDOFF_ALREADY_VALID",
                                 example_ids=[row["example_id"] for row in known_handoff]))
            continue
        values = np.asarray(scores[state_id], np.float64)
        good = np.asarray([bool(row["target_good"]) for row in examples])
        selected = {}

        def add(index, role):
            index = int(index)
            selected.setdefault(index, []).append(role)

        for index in np.flatnonzero(good):
            add(index, "CURRENT_IMMEDIATE_GOOD")
        bad = np.flatnonzero(~good)
        add(bad[int(np.argmax(values[bad]))], "CURRENT_RANKER_WINNING_BAD")
        aliases = [index for index in bad
                   if "low_loss_bad_alias" in hard_tags.get(examples[int(index)]["example_id"], ())]
        aliases.sort(key=lambda index: (float(examples[int(index)]["label_objective_loss"]),
                                        -values[int(index)], int(index)))
        for index in aliases[:2]:
            add(index, "IMPORTANT_LOW_LOSS_BAD_ALIAS")
        remaining = [index for index in bad if int(index) not in selected]
        if remaining:
            # One deterministic structurally rejected control, preferring a
            # Pareto/non-hull alternative over another selected-cell proposal.
            remaining.sort(key=lambda index: (
                examples[int(index)]["current_routing_status_name"] not in
                ("PARETO_DOMINATED_OWNER", "NON_HULL_OWNER"), int(index)))
            add(remaining[0], "REPRESENTATIVE_REMAINING_BAD")
        for index, roles in sorted(selected.items()):
            row = dict(examples[index])
            row["selection_roles"] = roles
            row["baseline_model_score"] = float(values[index])
            chosen.append(row)
    return chosen, excluded


def reconstruct(row: dict, objective, lower, upper):
    """Restore the exact lossless round-boundary snapshot without replay."""
    saved = np.load(row["state_npz"])
    points = np.asarray(saved["sites"], np.float64)
    losses = np.asarray(saved["site_losses"], np.float64)
    count = int(row["evaluation_count"]) - 1
    if points.shape != (count, len(lower)) or losses.shape != (count,):
        raise RuntimeError(f"{row['state_id']}: malformed lossless snapshot")
    runner = production.N1Runner(
        objective, lower, upper, int(saved["generator_id"]))
    runner.points = [point.copy() for point in points]
    runner.losses = [float(value) for value in losses]
    runner.iteration = int(saved["iteration"])
    # Candidate states were deliberately taken at production round boundaries.
    # A forced action changes that boundary state, so proposals are recomputed.
    runner.phase = "next_round"
    runner.pending = np.empty((0, len(lower)), np.float64)
    runner.next_pending = 0
    runner.stop_reason = None
    return runner


def endpoint_for(theta, loss, evaluations, lower, upper):
    theta = np.asarray(theta, np.float64)
    width = np.asarray(upper) - np.asarray(lower)
    radius = width / float(evaluations) ** (1.0 / len(theta))
    return dict(theta=theta, loss=float(loss), evaluations=int(evaluations),
                lower=np.maximum(lower, theta - radius),
                upper=np.minimum(upper, theta + radius),
                stop_reason="counterfactual_candidate_long_horizon_audit")


def handoff_check(case_id, trajectory, objective, torch_objective, teacher,
                  lower, upper, theta, loss, evaluations, label):
    endpoint = endpoint_for(theta, loss, evaluations, lower, upper)
    finish = production._finish("kepler5", trajectory, objective, torch_objective,
                                endpoint, teacher, label)
    success = bool(finish["recovered"] and finish["rollout_finite"])
    return success, endpoint, finish


def run_branch(row: dict, continuation: int) -> dict:
    path = BRANCHES / f"{branch_id(row)}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if int(prior["continuation_budget"]) >= continuation:
            return prior
    started = time.monotonic()
    case_id = int(row["case_id"])
    trajectory, _ = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    start_count = int(row["evaluation_count"]) - 1
    runner = reconstruct(row, objective, lower, upper)
    start_best = float(np.min(runner.losses))
    unit = np.asarray(row["unit_point"], np.float64)
    theta = lower + (upper - lower) * unit
    if np.any(np.all(np.asarray(runner.points) == unit[None, :], axis=1)):
        raise RuntimeError(f"{row['example_id']}: forced candidate already evaluated")
    runner.evaluate(unit, "forced_candidate")
    forced_loss = float(runner.losses[-1])
    expected_loss = float(row["label_objective_loss"])
    # The old label corpus can differ from the currently evaluated objective by
    # a few terminal float64 bits.  Search continuation must use the current
    # objective value; retain the corpus value and exact delta for audit.
    forced_loss_difference = forced_loss - expected_loss

    lineage = [False] * start_count + [True]
    parent_indices = [-1] * start_count + [int(row["owner_site"])]
    evolution = [dict(additional_evaluations=0, total_evaluations=len(runner.losses),
                      source="forced_candidate", evaluated_loss=forced_loss,
                      best_loss=float(np.min(runner.losses)))]
    success, endpoint, finish = handoff_check(
        case_id, trajectory, objective, torch_objective, teacher, lower, upper,
        theta, forced_loss, len(runner.losses), f"long_horizon_forced_{row['example_id']}")
    successful = None
    if success:
        successful = dict(additional_evaluations=0,
                          total_evaluations=len(runner.losses),
                          source="forced_candidate", theta=theta.tolist(),
                          loss=forced_loss, nearest_parent_index=int(row["owner_site"]),
                          nearest_parent_in_forced_lineage=True,
                          endpoint={"lower": endpoint["lower"].tolist(),
                                    "upper": endpoint["upper"].tolist()}, **finish)
    no_progress = False
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        for step in range(1, continuation + 1):
            if successful is not None:
                break
            before = len(runner.losses)
            prior_points = np.asarray(runner.points, np.float64)
            runner.run_to(before + 1)
            if len(runner.losses) == before:
                no_progress = True
                break
            point = np.asarray(runner.points[-1], np.float64)
            distances = np.linalg.norm(prior_points - point[None, :], axis=1)
            parent = int(np.argmin(distances))
            parent_indices.append(parent)
            lineage.append(bool(lineage[parent]))
            value = float(runner.losses[-1])
            theta_new = lower + (upper - lower) * point
            evolution.append(dict(additional_evaluations=step,
                                  total_evaluations=len(runner.losses),
                                  source=runner.events[-1]["source"],
                                  evaluated_loss=value,
                                  best_loss=float(np.min(runner.losses)),
                                  nearest_parent_index=parent,
                                  nearest_parent_in_forced_lineage=bool(lineage[-1])))
            success, endpoint, finish = handoff_check(
                case_id, trajectory, objective, torch_objective, teacher, lower, upper,
                theta_new, value, len(runner.losses),
                f"long_horizon_{row['example_id']}_{step}")
            if success:
                successful = dict(
                    additional_evaluations=step,
                    total_evaluations=len(runner.losses),
                    source=runner.events[-1]["source"], theta=theta_new.tolist(),
                    loss=value, nearest_parent_index=parent,
                    nearest_parent_in_forced_lineage=bool(lineage[-1]),
                    endpoint={"lower": endpoint["lower"].tolist(),
                              "upper": endpoint["upper"].tolist()}, **finish)
                break
    finally:
        production.N1_GATE = old_gate

    if successful is not None and successful["additional_evaluations"] == 0:
        label = "IMMEDIATE_RECOVERY"
    elif successful is not None:
        label = "LEADS_TO_RECOVERY"
    else:
        label = "RECOVERY_NOT_OBSERVED_WITHIN_CONTINUATION"
    result = dict(
        branch_id=branch_id(row), state_id=row["state_id"], case_id=case_id,
        split=row["split"], example_id=row["example_id"],
        entity_index=int(row["entity_index"]), entity_source=row["entity_source"],
        selection_roles=row["selection_roles"],
        baseline_model_score=float(row["baseline_model_score"]),
        old_immediate_label="GOOD" if row["target_good"] else "BAD",
        new_diagnostic_label=label, start_evaluations=start_count,
        forced_evaluation=start_count + 1, continuation_budget=int(continuation),
        additional_evaluations_run=int(len(evolution) - 1),
        no_progress=bool(no_progress), forced_theta=theta.tolist(),
        forced_loss=forced_loss, old_label_objective_loss=expected_loss,
        forced_loss_difference=forced_loss_difference, start_best_loss=start_best,
        final_branch_best_loss=float(np.min(runner.losses)),
        first_successful_handoff=successful, n1_loss_evolution=evolution,
        success_absorbing=bool(successful is not None),
        elapsed_seconds=float(time.monotonic() - started),
    )
    atomic_json(path, result)
    return result


def run_unforced_control(row: dict, continuation: int) -> dict:
    """Continue the same state without the counterfactual candidate action."""
    path = OUT / "controls" / f"{row['state_id']}.json"
    if path.exists():
        prior = json.loads(path.read_text())
        if int(prior["continuation_budget"]) >= continuation:
            return prior
    trajectory, _ = production._kepler_case(int(row["case_id"]))
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    runner = reconstruct(row, objective, lower, upper)
    evolution = []
    successful = None
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        for step in range(1, continuation + 1):
            before = len(runner.losses)
            runner.run_to(before + 1)
            if len(runner.losses) == before:
                break
            point = np.asarray(runner.points[-1], np.float64)
            theta = lower + (upper - lower) * point
            value = float(runner.losses[-1])
            evolution.append(dict(additional_evaluations=step,
                                  total_evaluations=len(runner.losses),
                                  source=runner.events[-1]["source"],
                                  evaluated_loss=value,
                                  best_loss=float(np.min(runner.losses))))
            success, endpoint, finish = handoff_check(
                int(row["case_id"]), trajectory, objective, torch_objective,
                teacher, lower, upper, theta, value, len(runner.losses),
                f"unforced_control_{row['state_id']}_{step}")
            if success:
                successful = dict(additional_evaluations=step,
                                  total_evaluations=len(runner.losses),
                                  source=runner.events[-1]["source"],
                                  theta=theta.tolist(), loss=value,
                                  endpoint={"lower": endpoint["lower"].tolist(),
                                            "upper": endpoint["upper"].tolist()},
                                  **finish)
                break
    finally:
        production.N1_GATE = old_gate
    result = dict(state_id=row["state_id"], case_id=int(row["case_id"]),
                  start_evaluations=int(row["evaluation_count"])-1,
                  continuation_budget=int(continuation),
                  first_successful_handoff=successful,
                  recovery_observed=bool(successful is not None),
                  n1_loss_evolution=evolution)
    atomic_json(path, result)
    return result


def distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0}
    return dict(count=int(len(values)), minimum=float(values.min()),
                median=float(np.median(values)),
                p90=float(np.quantile(values, .9, method="higher")),
                maximum=float(values.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation", type=int, default=25)
    parser.add_argument("--all-positive", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); BRANCHES.mkdir(parents=True, exist_ok=True)
    before = hashes(); model_hash = sha256(MODEL)
    grouped, metadata, scores, hard_tags, handoff = load_corpus()
    selected, excluded = chosen_branches(
        grouped, scores, hard_tags, handoff, only_failed=not args.all_positive)
    if args.limit is not None:
        selected = selected[:args.limit]
    if args.controls_only:
        state_rows = {}
        for row in selected:
            state_rows.setdefault(row["state_id"], row)
        controls = []
        for number, row in enumerate(state_rows.values(), 1):
            print(json.dumps(dict(control=number, total=len(state_rows),
                                  state=row["state_id"])), flush=True)
            controls.append(run_unforced_control(row, args.continuation))
        atomic_json(OUT / "unforced_controls.json", dict(
            continuation_budget=args.continuation, states=len(controls),
            recovery_observed=sum(row["recovery_observed"] for row in controls),
            controls=controls))
        print(json.dumps(dict(states=len(controls), recovery_observed=sum(
            row["recovery_observed"] for row in controls)), indent=2))
        return
    results = []
    for number, row in enumerate(selected, 1):
        print(json.dumps(dict(branch=number, total=len(selected), state=row["state_id"],
                              roles=row["selection_roles"], old_good=row["target_good"])),
              flush=True)
        result = run_branch(row, args.continuation)
        results.append(result)
        print(json.dumps(dict(branch=result["branch_id"],
                              label=result["new_diagnostic_label"],
                              delay=(None if result["first_successful_handoff"] is None else
                                     result["first_successful_handoff"]["additional_evaluations"]),
                              seconds=result["elapsed_seconds"])), flush=True)
    counts = {name: sum(row["new_diagnostic_label"] == name for row in results)
              for name in ("IMMEDIATE_RECOVERY", "LEADS_TO_RECOVERY",
                           "RECOVERY_NOT_OBSERVED_WITHIN_CONTINUATION")}
    delayed_bad = [row for row in results
                   if row["old_immediate_label"] == "BAD"
                   and row["new_diagnostic_label"] == "LEADS_TO_RECOVERY"]
    delays = [row["first_successful_handoff"]["additional_evaluations"]
              for row in results if row["new_diagnostic_label"] == "LEADS_TO_RECOVERY"]
    cases = defaultdict(lambda: defaultdict(int))
    for row in results:
        cases[str(row["case_id"])][row["new_diagnostic_label"]] += 1
    after = hashes()
    report = dict(
        status="DIAGNOSTIC_ONLY_NO_TRAINING", scope=("all_positive_persistent_states"
            if args.all_positive else "eleven_persistent_states_misranked_by_hardest_BAD_model"),
        continuation_budget_per_branch=int(args.continuation), branches=len(results),
        states=len({row["state_id"] for row in results}), excluded_states=excluded,
        labels=counts, old_bad_leads_to_recovery=len(delayed_bad),
        old_bad_delayed_branch_ids=[row["branch_id"] for row in delayed_bad],
        delay_to_recovery=distribution(delays), by_case=dict(cases),
        results=results, model_checkpoint=str(MODEL), model_checkpoint_sha256=model_hash,
        checkpoint_hashes_unchanged=before == after,
        dataset_sha256=sha256(DATASET), created_unix=time.time())
    atomic_json(OUT / ("report_all_positive.json" if args.all_positive else "report.json"), report)
    print(json.dumps({key: report[key] for key in
                      ("scope", "continuation_budget_per_branch", "branches", "states",
                       "labels", "old_bad_leads_to_recovery", "delay_to_recovery",
                       "by_case", "checkpoint_hashes_unchanged")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
