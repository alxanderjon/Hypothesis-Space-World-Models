#!/usr/bin/env python3
"""Rebuild only N1 handoff-promotion labels with corrected deterministic N3."""
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

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from experiments import test_n3_active_set_repair as active
from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.n3_box_qp import dimension_general_box_qp_direction
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

SOURCE = Path("outputs/n1_candidate_handoff_supervision")
PRIOR = Path("outputs/n1_candidate_handoff_supervision_uncapped_n3")
OUT = Path("outputs/n1_candidate_handoff_supervision_box_qp_n3")
CACHE = OUT / "label_cache"
ACTIVE_REPORT = PRIOR / "active_set_n3_diagnostic" / "report.json"
SPLITS = ("train", "held_out", "fresh_test")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def source_handoff_rows():
    return [row for row in json.loads((SOURCE / "dataset.json").read_text())
            if row["task"] == "handoff_promotion"]


def known_alias_ids():
    return {row["example_id"] for row in
            json.loads(ACTIVE_REPORT.read_text())["aliases"]}


def label_case(system, case_id):
    rows = [row for row in source_handoff_rows()
            if row["system"] == system and int(row["case_id"]) == case_id]
    if not rows:
        return
    trajectory, _ = (production._ball_case(case_id)
                     if system == "reflection_ball"
                     else production._kepler_case(case_id))
    _, torch_objective = production._objectives(system, trajectory)
    lower, upper, teacher = production._config(system)
    derivatives, direction_fn, run_n3 = staged._make_n3(
        torch_objective, lower, upper)
    aliases = known_alias_ids()
    for number, row in enumerate(rows, 1):
        path = CACHE / system / f"case_{case_id:03d}" / f'{row["example_id"]}.json'
        if path.exists():
            continue
        started = time.monotonic()
        n2_theta = np.asarray(row["label_n2_endpoint"], np.float64)
        endpoint = polish.N2Endpoint(
            0, n2_theta.copy(), float(row["label_n2_loss"]),
            "saved_exact_unchanged_n2", "corrected_box_qp_n3_relabel", False)
        result = run_n3(endpoint, torch.empty(0))
        final = np.asarray(result.theta_history[-1], np.float64)
        rollout = production._rollout(system, trajectory, final)
        difference = final - teacher
        corrected = bool(result.loss_history[-1] < staged.FINAL_GATE and
                         rollout["rollout_finite"])

        final_tensor = torch.from_numpy(final.copy())
        _, gradient, hessian = derivatives(final_tensor, torch.empty(0))
        _, qp_info = dimension_general_box_qp_direction(
            final_tensor, gradient, hessian,
            torch.from_numpy(lower.copy()), torch.from_numpy(upper.copy()),
            direction_fn)
        exact_active = np.flatnonzero((final == lower) | (final == upper)).tolist()
        if row["example_id"] in aliases:
            alias_status = "CONFIRMED_CONSTRAINED_ALIAS"
        elif not corrected and exact_active:
            alias_status = "BOUND_CONSTRAINED_BAD_GENUINE_STOP"
        else:
            alias_status = "NONE"
        damping = np.asarray(result.damping_values, np.float64)
        record = {
            "example_id": row["example_id"], "system": system,
            "case_id": case_id, "state_id": row["state_id"],
            "entity_index": int(row["entity_index"]),
            "entity_source": row["entity_source"], "split": row["split"],
            "old_target_good": bool(row["target_good"]),
            "corrected_target_good": corrected,
            "label_changed": corrected != bool(row["target_good"]),
            "n2_reused_exactly": True,
            "n2_endpoint": n2_theta.tolist(),
            "n2_loss": float(row["label_n2_loss"]),
            "n2_box_lower": row["label_n2_box_lower"],
            "n2_box_upper": row["label_n2_box_upper"],
            "n2_evaluations": int(row["label_n2_evaluations"]),
            "n2_stop_reason": row["label_n2_stop_reason"],
            "n3_semantic_step_cap": None,
            "n3_stop_reason": result.stop_reason,
            "n3_accepted_steps": int(result.accepted_steps),
            "n3_objective_evaluations": int(result.actual_loss_evaluations),
            "n3_rejected_newton_steps": int(result.rejected_newton_steps),
            "n3_backtracking_rejections": int(result.backtracking_rejections),
            "n3_accepted_with_backtracking": int(result.accepted_with_backtracking),
            "n3_damping_steps": int(result.damping_steps),
            "n3_direct_routes": int(np.count_nonzero(damping == 0.0)),
            "n3_damped_routes": int(np.count_nonzero(damping > 0.0)),
            "final_theta": final.tolist(),
            "final_loss": float(result.loss_history[-1]),
            "final_entrywise_parameter_error": difference.tolist(),
            "final_max_parameter_error": float(np.max(np.abs(difference))),
            "final_parameter_l2_error": float(np.linalg.norm(difference)),
            "continuation_max_error": rollout["continuation_max_error"],
            "continuation_mse": rollout["continuation_mse"],
            "collision_topology_regressions": rollout[
                "collision_topology_regressions"],
            "rollout_finite": bool(rollout["rollout_finite"]),
            "constrained_alias_status": alias_status,
            "final_exact_active_bound_coordinates": exact_active,
            "final_box_qp_face": qp_info["winning_face_status"],
            "elapsed_seconds": float(time.monotonic() - started),
        }
        atomic_json(path, record)
        print(system, case_id, f"{number}/{len(rows)}",
              "GOOD" if corrected else "BAD", result.accepted_steps,
              result.stop_reason, alias_status, flush=True)


def run_cases(workers):
    cases = sorted({(row["system"], int(row["case_id"]))
                    for row in source_handoff_rows()})
    workers = max(1, min(int(workers), 4))
    for offset in range(0, len(cases), workers):
        children = []
        for system, case_id in cases[offset:offset + workers]:
            command = [sys.executable, "-m", __spec__.name,
                       "--label-case", system, str(case_id)]
            children.append(((system, case_id), subprocess.Popen(
                command, env=os.environ.copy())))
        failures = [(case, child.wait()) for case, child in children]
        failures = [item for item in failures if item[1]]
        if failures:
            raise RuntimeError(failures)


def distribution(values):
    values = np.asarray(list(values), np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "minimum": float(values.min()),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9, method="higher")),
            "p99": float(np.quantile(values, .99, method="higher")),
            "maximum": float(values.max())}


def finalize():
    source = source_handoff_rows()
    corrections = {}
    for path in CACHE.rglob("*.json"):
        correction = json.loads(path.read_text())
        corrections[correction["example_id"]] = correction
    missing = [row["example_id"] for row in source
               if row["example_id"] not in corrections]
    if missing:
        raise RuntimeError(f"missing {len(missing)} labels")

    corrected = []
    for row in source:
        correction = corrections[row["example_id"]]
        updated = dict(row)
        updated["old_target_good"] = bool(row["target_good"])
        updated["old_label_fields"] = {
            key: value for key, value in row.items() if key.startswith("label_")}
        updated["target_good"] = bool(correction["corrected_target_good"])
        for key, value in correction.items():
            if key not in ("example_id", "system", "case_id", "state_id",
                           "entity_index", "entity_source", "split"):
                updated[f"corrected_{key}"] = value
        corrected.append(updated)
    atomic_json(OUT / "handoff_promotion_dataset.json", corrected)
    atomic_jsonl(OUT / "handoff_promotion_dataset.jsonl", corrected)
    for split in SPLITS:
        atomic_jsonl(OUT / f"handoff_promotion_{split}.jsonl",
                     [row for row in corrected if row["split"] == split])

    # Chronological absorbing target over every evaluated-site handoff example
    # represented in this corpus.  Site indices are stable insertion indices.
    state_groups = defaultdict(list)
    for row in corrected:
        state_groups[row["state_id"]].append(row)
    trajectory_groups = defaultdict(list)
    for rows in state_groups.values():
        trajectory_groups[(rows[0]["system"], int(rows[0]["case_id"]))].append(rows)
    archive = []
    first_success = {}
    for trajectory, states in trajectory_groups.items():
        states.sort(key=lambda rows: (int(rows[0]["evaluation_count"]),
                                     rows[0]["state_id"]))
        absorbing = False
        accumulated_successes = {}
        first_evaluation = None
        for rows in states:
            current_success = [row for row in rows if row["target_good"]]
            for row in current_success:
                accumulated_successes[int(row["entity_index"])] = row["example_id"]
            if current_success and not absorbing:
                absorbing = True
                first_evaluation = int(rows[0]["evaluation_count"])
            total_sites = int(np.load(rows[0]["state_npz"])["sites"].shape[0])
            archive.append({
                "state_id": rows[0]["state_id"], "system": trajectory[0],
                "case_id": trajectory[1], "split": rows[0]["split"],
                "evaluation_count": int(rows[0]["evaluation_count"]),
                "absorbing_any_successful_handoff": absorbing,
                "first_successful_archive_evaluation": first_evaluation,
                "current_successful_site_indices": sorted(
                    int(row["entity_index"]) for row in current_success),
                "current_successful_example_ids": [
                    row["example_id"] for row in current_success],
                "absorbed_successful_site_indices": sorted(accumulated_successes),
                "absorbed_successful_example_ids": [
                    accumulated_successes[index]
                    for index in sorted(accumulated_successes)],
                "represented_handoff_sites": len(rows),
                "total_evaluated_sites_available": total_sites,
                "success_is_absorbing": True,
                "target_scope": (
                    "all evaluated-site handoff examples represented by the "
                    "saved supervision corpus"),
            })
        first_success[f"{trajectory[0]}:{trajectory[1]}"] = first_evaluation
    archive.sort(key=lambda row: (row["system"], int(row["case_id"]),
                                  row["evaluation_count"]))
    atomic_json(OUT / "archive_state_targets.json", archive)
    atomic_jsonl(OUT / "archive_state_targets.jsonl", archive)

    prior_completed = {
        row["example_id"]: row for row in
        json.loads((PRIOR / "handoff_promotion_dataset_completed.json").read_text())}
    prior_unresolved = {
        row["example_id"] for row in
        json.loads((PRIOR / "handoff_promotion_unresolved.json").read_text())}
    source_by_id = {row["example_id"]: row for row in source}
    prior_bad_to_good = {
        example_id for example_id, row in prior_completed.items()
        if not bool(source_by_id[example_id]["target_good"])
        and bool(row["target_good"])}
    changed = [row for row in corrected
               if row["old_target_good"] != row["target_good"]]
    by_system = {}
    for system in ("reflection_ball", "kepler5"):
        rows = [row for row in corrected if row["system"] == system]
        by_system[system] = {
            "total": len(rows), "good": sum(row["target_good"] for row in rows),
            "bad": sum(not row["target_good"] for row in rows),
            "bad_to_good": sum(not row["old_target_good"] and row["target_good"]
                               for row in rows),
            "good_to_bad": sum(row["old_target_good"] and not row["target_good"]
                               for row in rows),
        }
    unresolved_results = [row for row in corrected
                          if row["example_id"] in prior_unresolved]
    prior_completed_differences = []
    for row in corrected:
        old = prior_completed.get(row["example_id"])
        if old is not None and bool(old["target_good"]) != bool(row["target_good"]):
            prior_completed_differences.append(row["example_id"])
    original_good_to_bad = [
        row["example_id"] for row in corrected
        if row["old_target_good"] and not row["target_good"]]
    archive_by_system = {}
    for system in ("reflection_ball", "kepler5"):
        rows = [row for row in archive if row["system"] == system]
        archive_by_system[system] = {
            "states": len(rows),
            "current_positive": sum(bool(row["current_successful_site_indices"])
                                    for row in rows),
            "current_negative": sum(not row["current_successful_site_indices"]
                                    for row in rows),
            "absorbing_positive": sum(
                row["absorbing_any_successful_handoff"] for row in rows),
            "absorbing_negative": sum(
                not row["absorbing_any_successful_handoff"] for row in rows),
        }
    report = {
        "label_definition": (
            "saved unchanged production N2 endpoint -> integrated dimension-general "
            "box-QP N3 with no semantic accepted-step cap -> existing final_loss "
            "< 1e-20 and finite-rollout recovered-law outcome"),
        "handoff_examples": len(corrected),
        "good": sum(row["target_good"] for row in corrected),
        "bad": sum(not row["target_good"] for row in corrected),
        "bad_to_good_from_original": sum(
            not row["old_target_good"] and row["target_good"] for row in corrected),
        "good_to_bad_from_original": sum(
            row["old_target_good"] and not row["target_good"] for row in corrected),
        "changed_from_original": len(changed),
        "by_system": by_system,
        "previous_bad_to_good_count": len(prior_bad_to_good),
        "previous_290_bad_to_good_preserved": sum(
            bool(corrected_row["target_good"])
            for corrected_row in corrected
            if corrected_row["example_id"] in prior_bad_to_good),
        "previous_290_bad_to_good_reversed": sorted(
            corrected_row["example_id"]
            for corrected_row in corrected
            if corrected_row["example_id"] in prior_bad_to_good
            and not corrected_row["target_good"]),
        "original_good_to_bad_example_ids": original_good_to_bad,
        "prior_completed_label_differences": prior_completed_differences,
        "previously_unresolved": {
            "count": len(unresolved_results),
            "good": sum(row["target_good"] for row in unresolved_results),
            "bad": sum(not row["target_good"] for row in unresolved_results),
            "confirmed_constrained_alias": sum(
                row["corrected_constrained_alias_status"] ==
                "CONFIRMED_CONSTRAINED_ALIAS" for row in unresolved_results),
            "stop_reasons": dict(Counter(
                row["corrected_n3_stop_reason"] for row in unresolved_results)),
        },
        "remaining_unresolved_labels": 0,
        "accepted_steps": distribution(
            row["corrected_n3_accepted_steps"] for row in corrected),
        "stop_reasons": dict(Counter(
            row["corrected_n3_stop_reason"] for row in corrected)),
        "constrained_alias_status": dict(Counter(
            row["corrected_constrained_alias_status"] for row in corrected)),
        "archive": {
            "states": len(archive),
            "positive": sum(row["absorbing_any_successful_handoff"]
                            for row in archive),
            "negative": sum(not row["absorbing_any_successful_handoff"]
                            for row in archive),
            "trajectories": len(trajectory_groups),
            "by_system": archive_by_system,
            "represented_site_occurrences": sum(
                row["represented_handoff_sites"] for row in archive),
            "total_evaluated_site_occurrences": sum(
                row["total_evaluated_sites_available"] for row in archive),
            "first_successful_evaluation_by_trajectory": first_success,
            "scope": "represented handoff examples; raw unlabelled archive sites are retained in NPZ but were not exhaustively relabelled",
            "site_selector_training_suitable": bool(
                any(row["target_good"] for row in corrected)
                and any(not row["target_good"] for row in corrected)),
            "archive_any_none_gate_training_suitable": bool(
                any(row["absorbing_any_successful_handoff"] for row in archive)
                and any(not row["absorbing_any_successful_handoff"]
                        for row in archive)),
            "suitability_note": (
                "site-level handoff promotion has both classes; the represented "
                "chronological states are all absorbing-positive and cannot alone "
                "train an ANY-vs-NONE archive gate"),
        },
    }
    atomic_json(OUT / "report.json", report)
    atomic_json(OUT / "schema.json", {
        "version": 1, "source": str(SOURCE / "dataset.json"),
        "task": "handoff_promotion only",
        "n1_n2": "unchanged; saved exact N2 endpoints reused",
        "n3": "integrated dimension-general active-set box-QP, no accepted-step cap",
        "splits": "original whole-trajectory splits preserved",
        "candidate_selection_labels_modified": False,
        "archive_success": "chronological absorbing ANY corrected GOOD",
        "archive_identity_scope": (
            "evaluated-site handoff examples represented in the source corpus; "
            "unlabelled raw archive sites are retained but not enumerated as "
            "successful identities"),
        "training_suitability": (
            "site-level promotion has both classes; represented archive-state "
            "ANY/NONE targets are all positive and need negative states before "
            "training a state gate"),
    })
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-case", nargs=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.label_case:
        label_case(args.label_case[0], int(args.label_case[1]))
        return
    before = hashes()
    atomic_json(OUT / "checkpoint_hashes_before.json", before)
    if not args.finalize:
        run_cases(args.workers)
    finalize()
    after = hashes()
    atomic_json(OUT / "checkpoint_hashes_after.json", after)
    atomic_json(OUT / "checkpoint_hash_status.json", {
        "unchanged": before == after, "before": before, "after": after})


if __name__ == "__main__":
    main()
