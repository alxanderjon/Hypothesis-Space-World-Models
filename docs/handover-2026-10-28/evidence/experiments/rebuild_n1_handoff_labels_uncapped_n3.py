#!/usr/bin/env python3
"""Rebuild only N1 handoff-promotion labels with N3's step cap removed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SOURCE = Path("outputs/n1_candidate_handoff_supervision")
OUT = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
CACHE = OUT / "label_cache"
PROGRESS = OUT / "uncapped_progress"
PERSISTENT_KEPLER = {9, 22, 28, 30, 33, 35, 41, 53, 54, 58}
SPLITS = ("train", "held_out", "fresh_test")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def uncapped(run):
    namespace = dict(run.__globals__)
    namespace["MAX_ACCEPTED_STEPS"] = math.inf
    return types.FunctionType(run.__code__, namespace)


def checkpoint_chunk_runner(run, accepted_steps=1_000):
    """Checkpoint boundary only; max-step results are always resumed."""
    namespace = dict(run.__globals__)
    namespace["MAX_ACCEPTED_STEPS"] = accepted_steps
    return types.FunctionType(run.__code__, namespace)


def source_rows():
    return json.loads((SOURCE / "dataset.json").read_text())


def case_rows(system: str, case_id: int):
    return [row for row in source_rows()
            if row["task"] == "handoff_promotion" and
            row["system"] == system and int(row["case_id"]) == case_id]


def label_case(system: str, case_id: int, single_chunk: bool = False) -> None:
    rows = case_rows(system, case_id)
    if not rows:
        return
    trajectory, _ = (production._ball_case(case_id) if system == "reflection_ball"
                     else production._kepler_case(case_id))
    _, torch_objective = production._objectives(system, trajectory)
    lower, upper, teacher = production._config(system)
    _, _, production_run = staged._make_n3(torch_objective, lower, upper)
    run = checkpoint_chunk_runner(production_run)
    for number, row in enumerate(rows, 1):
        path = CACHE / system / f"case_{case_id:03d}" / f'{row["example_id"]}.json'
        if path.exists():
            continue
        started = time.monotonic()
        old_stop = row["label_n3_stop_reason"]
        if old_stop != "max_accepted_steps":
            final = np.asarray(row["label_final_theta"], np.float64)
            record = dict(
                example_id=row["example_id"], system=system, case_id=case_id,
                state_id=row["state_id"], entity_index=int(row["entity_index"]),
                entity_source=row["entity_source"], split=row["split"],
                old_target_good=bool(row["target_good"]),
                corrected_target_good=bool(row["target_good"]), label_changed=False,
                n2_reused_exactly=True,
                n2_endpoint=row["label_n2_endpoint"], n2_loss=row["label_n2_loss"],
                n2_box_lower=row["label_n2_box_lower"],
                n2_box_upper=row["label_n2_box_upper"],
                n2_evaluations=row["label_n2_evaluations"],
                n2_stop_reason=row["label_n2_stop_reason"],
                n3_step_cap=None, n3_resume_mode="existing_genuine_stop_reused",
                n3_stop_reason=old_stop,
                n3_accepted_steps=int(row["label_n3_accepted_steps"]),
                n3_objective_evaluations=int(row["label_n3_objective_evaluations"]),
                n3_rejected_newton_steps=None, n3_backtracking_rejections=None,
                n3_accepted_with_backtracking=None, n3_damping_steps=None,
                n3_direct_routes=None, n3_damped_routes=None,
                final_theta=final.tolist(), final_loss=float(row["label_final_loss"]),
                final_entrywise_parameter_error=row["label_final_entrywise_parameter_error"],
                final_max_parameter_error=float(row["label_final_max_parameter_error"]),
                final_parameter_l2_error=float(row["label_final_parameter_l2_error"]),
                continuation_max_error=row["label_continuation_max_error"],
                continuation_mse=row["label_continuation_mse"],
                collision_topology_regressions=row["label_collision_topology_regressions"],
                rollout_finite=bool(row["label_rollout_finite"]),
                elapsed_seconds=float(time.monotonic() - started),
            )
            atomic_json(path, record)
            print(system, case_id, f"{number}/{len(rows)}", "GENUINE STOP REUSED",
                  old_stop, flush=True)
            continue

        # The first 20 accepted steps are already saved bit-exactly. Resume
        # from that exact production final state, changing only the step cap.
        theta = np.asarray(row["label_final_theta"], np.float64)
        endpoint = polish.N2Endpoint(
            0, theta.copy(), float(row["label_final_loss"]),
            "saved_exact_step20_n3_state", "uncapped_n3_handoff_relabel", False)
        progress_path = (PROGRESS / system / f"case_{case_id:03d}" /
                         f'{row["example_id"]}.json')
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            endpoint = polish.N2Endpoint(
                0, np.asarray(progress["theta"], np.float64), float(progress["loss"]),
                "uncapped_checkpoint_resume", "uncapped_n3_handoff_relabel", False)
        else:
            progress = dict(additional_accepted_steps=0, objective_evaluations=0,
                            rejected_newton_steps=0, backtracking_rejections=0,
                            accepted_with_backtracking=0, damping_steps=0,
                            direct_routes=0, damped_routes=0, chunks=0)
        while True:
            result = run(endpoint, torch.empty(0))
            chunk_damping = np.asarray(result.damping_values, np.float64)
            progress["additional_accepted_steps"] += int(result.accepted_steps)
            progress["objective_evaluations"] += int(result.actual_loss_evaluations)
            progress["rejected_newton_steps"] += int(result.rejected_newton_steps)
            progress["backtracking_rejections"] += int(result.backtracking_rejections)
            progress["accepted_with_backtracking"] += int(result.accepted_with_backtracking)
            progress["damping_steps"] += int(result.damping_steps)
            progress["direct_routes"] += int(np.count_nonzero(chunk_damping == 0.0))
            progress["damped_routes"] += int(np.count_nonzero(chunk_damping > 0.0))
            progress["chunks"] += 1
            progress["theta"] = np.asarray(result.theta_history[-1], np.float64).tolist()
            progress["loss"] = float(result.loss_history[-1])
            progress["last_stop_reason"] = result.stop_reason
            atomic_json(progress_path, progress)
            if result.stop_reason != "max_accepted_steps":
                break
            print(system, case_id, f"{number}/{len(rows)}",
                  "UNCAPPED CHECKPOINT", progress["additional_accepted_steps"],
                  f'{progress["loss"]:.6e}', flush=True)
            if single_chunk:
                result = None
                break
            endpoint = polish.N2Endpoint(
                0, np.asarray(progress["theta"], np.float64), float(progress["loss"]),
                "uncapped_checkpoint_resume", "uncapped_n3_handoff_relabel", False)
        if result is None:
            continue
        final = np.asarray(result.theta_history[-1], np.float64)
        rollout = production._rollout(system, trajectory, final)
        difference = final - teacher
        corrected = bool(result.loss_history[-1] < staged.FINAL_GATE and
                         rollout["rollout_finite"])
        record = dict(
            example_id=row["example_id"], system=system, case_id=case_id,
            state_id=row["state_id"], entity_index=int(row["entity_index"]),
            entity_source=row["entity_source"], split=row["split"],
            old_target_good=bool(row["target_good"]),
            corrected_target_good=corrected,
            label_changed=bool(corrected != bool(row["target_good"])),
            n2_reused_exactly=True,
            n2_endpoint=theta.tolist(), n2_loss=float(row["label_n2_loss"]),
            n2_box_lower=row["label_n2_box_lower"],
            n2_box_upper=row["label_n2_box_upper"],
            n2_evaluations=row["label_n2_evaluations"],
            n2_stop_reason=row["label_n2_stop_reason"],
            n3_step_cap=None,
            n3_resume_mode="exact_saved_step20_plus_uncapped_continuation",
            n3_stop_reason=result.stop_reason,
            n3_accepted_steps=int(row["label_n3_accepted_steps"] +
                                  progress["additional_accepted_steps"]),
            n3_additional_accepted_steps=int(progress["additional_accepted_steps"]),
            n3_objective_evaluations=int(row["label_n3_objective_evaluations"] +
                                         progress["objective_evaluations"]),
            n3_rejected_newton_steps=int(progress["rejected_newton_steps"]),
            n3_backtracking_rejections=int(progress["backtracking_rejections"]),
            n3_accepted_with_backtracking=int(progress["accepted_with_backtracking"]),
            n3_damping_steps=int(progress["damping_steps"]),
            n3_direct_routes=int(progress["direct_routes"]),
            n3_damped_routes=int(progress["damped_routes"]),
            n3_checkpoint_chunks=int(progress["chunks"]),
            final_theta=final.tolist(), final_loss=float(result.loss_history[-1]),
            final_entrywise_parameter_error=difference.tolist(),
            final_max_parameter_error=float(np.max(np.abs(difference))),
            final_parameter_l2_error=float(np.linalg.norm(difference)),
            continuation_max_error=rollout["continuation_max_error"],
            continuation_mse=rollout["continuation_mse"],
            collision_topology_regressions=rollout["collision_topology_regressions"],
            rollout_finite=bool(rollout["rollout_finite"]),
            elapsed_seconds=float(time.monotonic() - started),
        )
        atomic_json(path, record)
        print(system, case_id, f"{number}/{len(rows)}",
              "GOOD" if record["old_target_good"] else "BAD", "->",
              "GOOD" if corrected else "BAD", result.accepted_steps,
              result.stop_reason, flush=True)


def run_cases(workers: int) -> None:
    rows = source_rows()
    cases = sorted({(row["system"], int(row["case_id"])) for row in rows
                    if row["task"] == "handoff_promotion"})
    workers = max(1, min(int(workers), 4))
    for offset in range(0, len(cases), workers):
        children = []
        for system, case_id in cases[offset:offset + workers]:
            command = [sys.executable, "-m",
                       "experiments.rebuild_n1_handoff_labels_uncapped_n3",
                       "--label-case", system, str(case_id)]
            children.append(((system, case_id), subprocess.Popen(command,
                            env=os.environ.copy())))
        failed = [(case, child.wait()) for case, child in children]
        failed = [item for item in failed if item[1]]
        if failed:
            raise RuntimeError(failed)


def dist(values):
    values = np.asarray(list(values), np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=int(len(values)), minimum=float(values.min()),
                median=float(np.median(values)),
                p90=float(np.quantile(values, .9, method="higher")),
                p99=float(np.quantile(values, .99, method="higher")),
                maximum=float(values.max()))


def nearest_label_disagreements(rows, label_key):
    """Leave-one-out nearest theta label disagreement; no distance threshold."""
    count = 0
    by_system = Counter()
    for system in ("reflection_ball", "kepler5"):
        current = [row for row in rows if row["system"] == system]
        lower, upper, _ = production._config(system)
        theta = np.asarray([row["theta"] for row in current], np.float64)
        theta = (theta - lower) / (upper - lower)
        labels = np.asarray([bool(row[label_key]) for row in current])
        for index in range(len(current)):
            distance = np.sum((theta - theta[index]) ** 2, axis=1)
            distance[index] = math.inf
            nearest = int(np.argmin(distance))
            if labels[index] != labels[nearest]:
                count += 1
                by_system[system] += 1
    return dict(total=count, by_system=dict(by_system))


def finalize() -> None:
    source = source_rows()
    corrections = {}
    for path in CACHE.rglob("*.json"):
        row = json.loads(path.read_text())
        corrections[row["example_id"]] = row
    handoff_source = [row for row in source if row["task"] == "handoff_promotion"]
    missing = [row["example_id"] for row in handoff_source
               if row["example_id"] not in corrections]
    if missing:
        raise RuntimeError(f"missing {len(missing)} corrected labels")

    corrected_handoff = []
    full = []
    for row in source:
        if row["task"] != "handoff_promotion":
            # Candidate rows and labels are preserved verbatim.
            full.append(row)
            continue
        correction = corrections[row["example_id"]]
        updated = dict(row)
        updated["old_target_good"] = bool(row["target_good"])
        updated["old_label_fields"] = {
            key: value for key, value in row.items() if key.startswith("label_")
        }
        updated["target_good"] = bool(correction["corrected_target_good"])
        for key, value in correction.items():
            if key not in ("example_id", "system", "case_id", "state_id",
                           "entity_index", "entity_source", "split"):
                updated[f"corrected_{key}"] = value
        corrected_handoff.append(updated)
        full.append(updated)

    atomic_json(OUT / "handoff_promotion_dataset.json", corrected_handoff)
    atomic_jsonl(OUT / "handoff_promotion_dataset.jsonl", corrected_handoff)
    atomic_json(OUT / "dataset.json", full)
    atomic_jsonl(OUT / "dataset.jsonl", full)
    for task in ("candidate_selection", "handoff_promotion"):
        for split in SPLITS:
            rows = [row for row in full if row["task"] == task and row["split"] == split]
            atomic_jsonl(OUT / f"{task}_{split}.jsonl", rows)

    # State-level absorbing target over every evaluated-site promotion example
    # represented by the existing dataset for that saved N1 state.
    state_groups = defaultdict(list)
    for row in corrected_handoff:
        state_groups[row["state_id"]].append(row)
    archive = []
    for state_id, rows in sorted(state_groups.items()):
        successful = [row for row in rows if row["target_good"]]
        archive.append(dict(
            state_id=state_id, system=rows[0]["system"], case_id=rows[0]["case_id"],
            split=rows[0]["split"], evaluation_count=rows[0]["evaluation_count"],
            absorbing_any_successful_handoff=bool(successful),
            successful_site_indices=[int(row["entity_index"]) for row in successful],
            successful_example_ids=[row["example_id"] for row in successful],
            successful_site_sources=[row["entity_source"] for row in successful],
            represented_handoff_sites=len(rows),
            total_evaluated_sites=int(np.load(rows[0]["state_npz"])["sites"].shape[0]),
            target_scope="existing handoff-promotion examples for this saved state",
            success_is_absorbing=True,
        ))
    atomic_json(OUT / "archive_state_targets.json", archive)
    atomic_jsonl(OUT / "archive_state_targets.jsonl", archive)

    changed = [row for row in corrected_handoff if row["corrected_label_changed"]]
    bad_good = [row for row in changed if not row["old_target_good"] and row["target_good"]]
    good_bad = [row for row in changed if row["old_target_good"] and not row["target_good"]]
    by_system = {}
    for system in ("reflection_ball", "kepler5"):
        rows = [row for row in corrected_handoff if row["system"] == system]
        changes = [row for row in rows if row["corrected_label_changed"]]
        by_system[system] = dict(
            total=len(rows), old_good=sum(row["old_target_good"] for row in rows),
            old_bad=sum(not row["old_target_good"] for row in rows),
            corrected_good=sum(row["target_good"] for row in rows),
            corrected_bad=sum(not row["target_good"] for row in rows),
            bad_to_good=sum((not row["old_target_good"]) and row["target_good"] for row in rows),
            good_to_bad=sum(row["old_target_good"] and not row["target_good"] for row in rows),
            changed=len(changes),
        )
    old_mixed = sum(any(row["old_target_good"] for row in rows) and
                    any(not row["old_target_good"] for row in rows)
                    for rows in state_groups.values())
    new_mixed = sum(any(row["target_good"] for row in rows) and
                    any(not row["target_good"] for row in rows)
                    for rows in state_groups.values())
    old_empty = sum(not any(row["old_target_good"] for row in rows)
                    for rows in state_groups.values())
    new_empty = sum(not any(row["target_good"] for row in rows)
                    for rows in state_groups.values())
    steps_over_20 = [row["corrected_n3_accepted_steps"] for row in corrected_handoff
                     if row["corrected_n3_accepted_steps"] > 20]
    report = dict(
        label_definition=("unchanged saved N2 endpoint -> N3 without accepted-step cap; "
                          "GOOD iff existing final_loss < 1e-20 and rollout is finite"),
        total_handoff_examples=len(corrected_handoff),
        old_good=sum(row["old_target_good"] for row in corrected_handoff),
        old_bad=sum(not row["old_target_good"] for row in corrected_handoff),
        corrected_good=sum(row["target_good"] for row in corrected_handoff),
        corrected_bad=sum(not row["target_good"] for row in corrected_handoff),
        bad_to_good=len(bad_good), good_to_bad=len(good_bad),
        by_system=by_system,
        changes=dict(
            nonincumbent=sum(row["entity_index"] != int(np.argmin(
                np.load(row["state_npz"])["site_losses"])) for row in changed),
            low_loss_old_bad_alias=sum((not row["old_target_good"]) and
                float(row["observed_loss"]) <= 1.0e-4 for row in changed),
            persistent_kepler=sum(row["system"] == "kepler5" and
                int(row["case_id"]) in PERSISTENT_KEPLER for row in changed),
            by_split=dict(Counter(row["split"] for row in changed)),
        ),
        accepted_steps=dict(all=dist(row["corrected_n3_accepted_steps"]
                                     for row in corrected_handoff),
                            beyond_20=dist(steps_over_20),
                            examples_beyond_20=len(steps_over_20)),
        stop_reasons=dict(Counter(row["corrected_n3_stop_reason"]
                                  for row in corrected_handoff)),
        neighborhood_conflict_diagnostics=dict(
            states=len(state_groups), old_mixed_label_states=old_mixed,
            corrected_mixed_label_states=new_mixed,
            old_states_without_good=old_empty,
            corrected_states_without_good=new_empty,
            old_leave_one_out_nearest_theta_label_disagreements=
                nearest_label_disagreements(corrected_handoff, "old_target_good"),
            corrected_leave_one_out_nearest_theta_label_disagreements=
                nearest_label_disagreements(corrected_handoff, "target_good"),
        ),
        archive_targets=dict(
            states=len(archive), any_successful=sum(
                row["absorbing_any_successful_handoff"] for row in archive),
            no_successful_represented_site=sum(
                not row["absorbing_any_successful_handoff"] for row in archive),
            scope="existing handoff-promotion site set; positives are certified, negatives are not exhaustive over every raw archived site",
        ),
        candidate_rows_preserved=sum(row["task"] == "candidate_selection" for row in full),
    )
    atomic_json(OUT / "report.json", report)
    atomic_json(OUT / "schema.json", dict(
        version=2,
        source_dataset=str(SOURCE / "dataset.jsonl"),
        corrected_task="handoff_promotion",
        candidate_selection_labels="preserved verbatim from source dataset",
        n2="saved exact output of unchanged production N2 is reused",
        n3="production N3 with only MAX_ACCEPTED_STEPS suppressed to infinity",
        corrected_label=("existing recovery rule: final_loss < 1e-20 and finite rollout; "
                         "parameter and rollout errors are diagnostics, not thresholds"),
        split="preserved whole-trajectory train/held_out/fresh_test split",
        archive_state_target="absorbing ANY GOOD over existing promotion examples in that saved state",
        archive_scope_limitation=("existing promotion corpus has canonical evaluated-site samples, "
                                  "not every raw archived site; negative archive targets are not exhaustive certificates"),
        files=dict(full_dataset="dataset.jsonl",
                   corrected_handoff="handoff_promotion_dataset.jsonl",
                   archive_targets="archive_state_targets.jsonl"),
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-case", nargs=2)
    parser.add_argument("--label-range", nargs=3)
    parser.add_argument("--single-chunk", action="store_true")
    parser.add_argument("--passes", type=int, default=1,
                        help="Repeat a range so checkpointed unresolved rows advance again.")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.label_case:
        label_case(args.label_case[0], int(args.label_case[1]), args.single_chunk)
        return
    if args.label_range:
        system, first, last = args.label_range
        for _ in range(args.passes):
            for case_id in range(int(first), int(last) + 1):
                label_case(system, case_id, args.single_chunk)
        return
    before = hashes()
    atomic_json(OUT / "checkpoint_hashes_before.json", before)
    if not args.finalize:
        run_cases(args.workers)
    finalize()
    after = hashes()
    atomic_json(OUT / "checkpoint_hashes_after.json", after)
    atomic_json(OUT / "checkpoint_hash_status.json",
                dict(unchanged=before == after, before=before, after=after))


if __name__ == "__main__":
    main()
