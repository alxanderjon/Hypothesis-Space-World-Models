#!/usr/bin/env python3
"""Build exact chronological archive-handoff supervision.

Every label belongs to (evaluated site, current N1 evaluation count).  Search
histories, N2, objectives, and bounds are unchanged.  The currently integrated
dimension-general box-QP N3 supplies all final labels and runs to a genuine
numerical stop.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
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

from experiments import audit_kepler_archive_absorbing_handoff as kepler_archive
from experiments import audit_recovery_precision_and_reflection_archive as ball_archive
from experiments import validate_d5_production_pipeline as production
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import run_stage1b_n3_gradient_polishing as polish
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.validate_lossless_edirect_representation import RoundState, encode


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/chronological_archive_handoff_supervision")
CACHE = OUT / "query_cache"
CASES = OUT / "cases"
STATES = OUT / "states"
SOURCE_DATASET = Path("outputs/n1_candidate_handoff_supervision/dataset.json")
IDENTIFIABILITY = Path("outputs/d5_handoff_characterization/reflection_identifiability.json")
SYSTEMS = ("reflection_ball", "kepler5")
SPLITS = ("train", "held_out", "fresh_test")
SCHEMA_VERSION = 1


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
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_lookup() -> dict[tuple[str, int], str]:
    rows = json.loads(SOURCE_DATASET.read_text())
    answer: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (row["system"], int(row["case_id"]))
        previous = answer.setdefault(key, row["split"])
        if previous != row["split"]:
            raise RuntimeError(f"trajectory split leakage in source: {key}")
    return answer


def identifiable_ball_ids() -> list[int]:
    rows = json.loads(IDENTIFIABILITY.read_text())["rows"]
    return sorted(int(row["case_id"]) for row in rows
                  if row["classification"] == "IDENTIFIABILITY_COMPLETE")


def case_specs() -> list[tuple[str, int]]:
    return ([('reflection_ball', case_id) for case_id in identifiable_ball_ids()]
            + [('kepler5', case_id) for case_id in range(64)])


def history_path(system: str, case_id: int) -> Path:
    if system == "reflection_ball":
        return (Path("outputs/d5_production_pipeline_validation/cases/") /
                system / f"case_{case_id:03d}" / "n1_state.npz")
    return kepler_archive.history_path(case_id)


def prior_query_path(system: str, case_id: int, n: int, index: int) -> Path:
    name = f"n{n:05d}_i{index + 1:05d}.json"
    if system == "reflection_ball":
        return (Path("outputs/recovery_precision_and_reflection_archive_audit/") /
                "reflection_ball_checks" / f"case_{case_id:03d}" / name)
    return (Path("outputs/kepler_archive_absorbing_handoff/checks") /
            f"case_{case_id:03d}" / name)


def query_path(system: str, case_id: int, n: int, index: int) -> Path:
    return CACHE / system / f"case_{case_id:03d}" / f"n{n:05d}_i{index + 1:05d}.json"


def corrected_n3_from_saved_n2(system: str, trajectory: np.ndarray,
                               torch_objective, teacher: np.ndarray,
                               saved: dict, label: str) -> dict:
    lower, upper, _ = production._config(system)
    _, _, run_n3 = staged._make_n3(torch_objective, lower, upper)
    endpoint = polish.N2Endpoint(
        0, np.asarray(saved["n2_endpoint"], np.float64),
        float(saved["n2_loss"]), "saved_exact_unchanged_n2",
        "chronological_archive_handoff", False)
    result = run_n3(endpoint, torch.empty(0))
    final = np.asarray(result.theta_history[-1], np.float64)
    difference = final - teacher
    rollout = production._rollout(system, trajectory, final)
    recovered = bool(result.loss_history[-1] < production.FINAL_GATE)
    return {
        "n2_box_lower": saved["n2_box_lower"],
        "n2_box_upper": saved["n2_box_upper"],
        "n2_evaluations": int(saved["n2_evaluations"]),
        "n2_endpoint": saved["n2_endpoint"],
        "n2_loss": float(saved["n2_loss"]),
        "n2_stop_reason": saved["n2_stop_reason"],
        "n2_historical_gate": bool(saved["n2_historical_gate"]),
        "n3_accepted_steps": int(result.accepted_steps),
        "n3_objective_evaluations": int(result.actual_loss_evaluations),
        "n3_stop_reason": result.stop_reason,
        "final_theta": final.tolist(),
        "final_loss": float(result.loss_history[-1]),
        "final_entrywise_parameter_error": difference.tolist(),
        "final_max_parameter_error": float(np.max(np.abs(difference))),
        "final_parameter_l2_error": float(np.linalg.norm(difference)),
        "recovered": recovered,
        "n3_recovered": recovered,
        "full_pipeline_pass": bool(recovered and rollout["rollout_finite"]),
        **rollout,
    }


class ChronologicalCase:
    def __init__(self, system: str, case_id: int):
        self.system = system
        self.case_id = int(case_id)
        if system == "reflection_ball":
            self.trajectory, self.initial_condition = production._ball_case(case_id)
        else:
            self.trajectory, self.initial_condition = production._kepler_case(case_id)
        self.objective, self.torch_objective = production._objectives(
            system, self.trajectory)
        self.lower, self.upper, self.teacher = production._config(system)
        self.width = self.upper - self.lower
        self.history = history_path(system, case_id)
        saved = np.load(self.history)
        self.points = np.asarray(saved["points"], np.float64)
        self.losses = np.asarray(saved["losses"], np.float64)
        self.thetas = self.lower + self.width * self.points
        if self.points.shape != (len(self.losses), 5):
            raise RuntimeError(f"malformed d=5 history: {self.history}")
        self.new_cache_hits = 0
        self.prior_n2_hits = 0
        self.n2_computed = 0

    def endpoint(self, n: int, index: int) -> dict:
        theta = self.thetas[index]
        radius = self.width / float(n) ** (1.0 / 5.0)
        return {
            "theta": theta, "loss": float(self.losses[index]),
            "evaluations": int(n),
            "lower": np.maximum(self.lower, theta - radius),
            "upper": np.minimum(self.upper, theta + radius),
            "stop_reason": "chronological_archive_handoff_supervision",
        }

    def check(self, n: int, index: int) -> dict:
        path = query_path(self.system, self.case_id, n, index)
        if path.exists():
            self.new_cache_hits += 1
            return json.loads(path.read_text())
        started = time.monotonic()
        endpoint = self.endpoint(n, index)
        prior = prior_query_path(self.system, self.case_id, n, index)
        if prior.exists():
            saved = json.loads(prior.read_text())
            if not (np.array_equal(np.asarray(saved["n2_box_lower"]), endpoint["lower"])
                    and np.array_equal(np.asarray(saved["n2_box_upper"]), endpoint["upper"])):
                raise RuntimeError(f"cached N2 box mismatch: {prior}")
            finish = corrected_n3_from_saved_n2(
                self.system, self.trajectory, self.torch_objective, self.teacher,
                saved, f"archive_{self.system}_{self.case_id}_{n}_{index}")
            n2_source = "reused_exact_prior_cache"
            self.prior_n2_hits += 1
        else:
            finish = production._finish(
                self.system, self.trajectory, self.objective, self.torch_objective,
                endpoint, self.teacher,
                f"archive_{self.system}_{self.case_id}_{n}_{index}")
            n2_source = "computed_unchanged_n2"
            self.n2_computed += 1
        if finish["n3_stop_reason"] == "max_accepted_steps":
            raise RuntimeError("semantic accepted-step cap reached corrected N3")
        row = {
            "schema_version": SCHEMA_VERSION,
            "system": self.system, "case_id": self.case_id,
            "current_evaluation_count": int(n),
            "archive_index": int(index),
            "insertion_evaluation": int(index + 1),
            "canonical_site_order": int(index),
            "theta": self.thetas[index].tolist(),
            "archived_loss": float(self.losses[index]),
            "target_good": bool(finish["full_pipeline_pass"]),
            "n2_source": n2_source,
            "n3_semantic_accepted_step_cap": None,
            "elapsed_seconds": float(time.monotonic() - started),
            **finish,
        }
        atomic_json(path, row)
        return row

    def save_state(self, n: int, split: str, labels: list[dict]) -> dict:
        state_id = f"{self.system}_case_{self.case_id:03d}_n{n:05d}"
        npz_path = STATES / f"{state_id}.npz"
        binary_path = STATES / f"{state_id}.edstate"
        state = RoundState(
            dimension=5,
            trajectory_id=(6_100_000 if self.system == "reflection_ball"
                           else 6_200_000) + self.case_id,
            # The canonical state carries the real chronological N1 count.
            # This is part of the runtime handoff state because the production
            # N2 box depends on it.
            iteration=n, lower=self.lower.copy(), upper=self.upper.copy(),
            points=self.points[:n].copy(), losses=self.losses[:n].copy())
        payload = encode(state)
        if not binary_path.exists():
            binary_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = binary_path.with_suffix(".edstate.tmp")
            temporary.write_bytes(payload)
            temporary.replace(binary_path)
        if binary_path.read_bytes() != payload:
            raise RuntimeError(f"lossless state mismatch: {binary_path}")
        if not npz_path.exists():
            np.savez_compressed(
                npz_path, schema_version=np.int64(SCHEMA_VERSION),
                system=np.asarray(self.system), case_id=np.int64(self.case_id),
                split=np.asarray(split), dimension=np.int64(5),
                evaluation_count=np.int64(n), iteration=np.int64(n),
                lower=self.lower, upper=self.upper,
                normalized_sites=self.points[:n],
                physical_sites=self.thetas[:n], site_losses=self.losses[:n],
                insertion_order=np.arange(n, dtype=np.int64))
        good = [row for row in labels if row["target_good"]]
        incumbent = int(np.argmin(self.losses[:n]))
        return {
            "schema_version": SCHEMA_VERSION,
            "state_id": state_id, "system": self.system,
            "case_id": self.case_id, "split": split,
            "dimension": 5, "evaluation_count": int(n),
            "state_target": "ANY_GOOD_HANDOFF_NOW" if good else "NONE_CONTINUE_N1",
            "target_any_good": bool(good),
            "canonical_site_order": "archive insertion order, zero-based index",
            "site_count": int(n),
            "site_query_ids": [
                f"{self.system}:{self.case_id}:{n}:{row['archive_index']}"
                for row in labels],
            "good_site_indices": [int(row["archive_index"]) for row in good],
            "bad_site_indices": [int(row["archive_index"]) for row in labels
                                 if not row["target_good"]],
            "incumbent_site_index": incumbent,
            "state_npz": str(npz_path),
            "lossless_state_binary": str(binary_path),
            "lossless_state_sha256": hashlib.sha256(payload).hexdigest(),
            "source_history": str(self.history),
            "source_history_sha256": file_hash(self.history),
            "ground_truth_absent_from_runtime_state": True,
        }


def run_case(system: str, case_id: int) -> dict:
    result_path = CASES / system / f"case_{case_id:03d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    splits = split_lookup()
    split = splits[(system, case_id)]
    case = ChronologicalCase(system, case_id)
    started = time.monotonic()
    states = []
    labels = []
    first_positive = None
    for n in range(1, len(case.losses) + 1):
        current = [case.check(n, index) for index in range(n)]
        labels.extend(current)
        state = case.save_state(n, split, current)
        states.append(state)
        if state["target_any_good"]:
            first_positive = state
            break
    if first_positive is None:
        raise RuntimeError(
            f"{system} case {case_id} did not reach ANY_GOOD in saved history")
    n = int(first_positive["evaluation_count"])
    good = [row for row in labels
            if row["current_evaluation_count"] == n and row["target_good"]]
    canonical = min(good, key=lambda row: row["archive_index"])
    incumbent = int(first_positive["incumbent_site_index"])
    transitions = []
    by_site = defaultdict(list)
    for row in labels:
        by_site[int(row["archive_index"])].append(row)
    for index, rows in by_site.items():
        rows.sort(key=lambda row: row["current_evaluation_count"])
        for earlier, later in zip(rows, rows[1:]):
            if not earlier["target_good"] and later["target_good"]:
                transitions.append({
                    "archive_index": index,
                    "from_count": earlier["current_evaluation_count"],
                    "to_count": later["current_evaluation_count"],
                    "theta": later["theta"],
                    "site_unchanged": earlier["theta"] == later["theta"],
                    "only_runtime_change_is_count_derived_n2_box": True,
                })
                break
    result = {
        "schema_version": SCHEMA_VERSION,
        "system": system, "case_id": case_id, "split": split,
        "initial_condition": case.initial_condition,
        "state_ids": [row["state_id"] for row in states],
        "query_cache_paths": [str(query_path(
            system, case_id, row["current_evaluation_count"], row["archive_index"]))
            for row in labels],
        "chronological_states": len(states),
        "none_states": sum(not row["target_any_good"] for row in states),
        "first_any_good_evaluation": n,
        "first_positive_good_site_count": len(good),
        "first_positive_good_site_indices": first_positive["good_site_indices"],
        "canonical_first_good_site_index": int(canonical["archive_index"]),
        "canonical_first_good_is_newest": bool(canonical["archive_index"] == n - 1),
        "canonical_first_good_is_older": bool(canonical["archive_index"] < n - 1),
        "canonical_first_good_is_incumbent": bool(canonical["archive_index"] == incumbent),
        "canonical_first_good_is_nonincumbent": bool(canonical["archive_index"] != incumbent),
        "older_site_bad_to_good_box_scale_transitions": transitions,
        "site_labels": len(labels),
        "good_site_labels": sum(row["target_good"] for row in labels),
        "bad_site_labels": sum(not row["target_good"] for row in labels),
        "new_query_cache_hits": case.new_cache_hits,
        "prior_exact_n2_cache_reuse": case.prior_n2_hits,
        "unchanged_n2_computations": case.n2_computed,
        "elapsed_seconds": float(time.monotonic() - started),
        "stopped_at_first_positive": True,
    }
    atomic_json(result_path, result)
    print(system, case_id, split, f"first={n}",
          f"good={len(good)}/{n}", flush=True)
    return result


def run_cases(workers: int) -> None:
    specs = case_specs()
    workers = max(1, min(int(workers), 4))
    for offset in range(0, len(specs), workers):
        children = []
        for system, case_id in specs[offset:offset + workers]:
            command = [sys.executable, "-m", __spec__.name,
                       "--case", system, str(case_id)]
            children.append(((system, case_id), subprocess.Popen(
                command, env=os.environ.copy())))
        failures = [(spec, child.wait()) for spec, child in children]
        failures = [value for value in failures if value[1]]
        if failures:
            raise RuntimeError(failures)


def distribution(values) -> dict:
    values = np.asarray(list(values), np.float64)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .9, method="higher")),
        "maximum": float(values.max()),
        "histogram": {str(int(key)): int(value) for key, value in
                      sorted(Counter(values.astype(int)).items())},
    }


def finalize(before_hashes: dict) -> dict:
    results = []
    for system, case_id in case_specs():
        path = CASES / system / f"case_{case_id:03d}.json"
        if not path.exists():
            raise RuntimeError(f"missing result: {path}")
        results.append(json.loads(path.read_text()))
    states = []
    sites = []
    for result in results:
        for state_id in result["state_ids"]:
            state_path = STATES / f"{state_id}.npz"
            state = np.load(state_path)
            n = int(state["evaluation_count"])
            state_rows = [json.loads(query_path(
                result["system"], int(result["case_id"]), n, index).read_text())
                          for index in range(n)]
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "state_id": state_id, "system": result["system"],
                "case_id": int(result["case_id"]), "split": result["split"],
                "dimension": 5, "evaluation_count": n,
                "state_target": ("ANY_GOOD_HANDOFF_NOW" if
                                 any(row["target_good"] for row in state_rows)
                                 else "NONE_CONTINUE_N1"),
                "target_any_good": any(row["target_good"] for row in state_rows),
                "site_count": n,
                "good_site_indices": [row["archive_index"] for row in state_rows
                                      if row["target_good"]],
                "bad_site_indices": [row["archive_index"] for row in state_rows
                                     if not row["target_good"]],
                "incumbent_site_index": int(np.argmin(state["site_losses"])),
                "canonical_site_order": "archive insertion order, zero-based index",
                "state_npz": str(state_path),
                "lossless_state_binary": str(STATES / f"{state_id}.edstate"),
                "lossless_state_sha256": file_hash(STATES / f"{state_id}.edstate"),
                "ground_truth_absent_from_runtime_state": True,
            }
            states.append(metadata)
            for row in state_rows:
                item = dict(row)
                item.update({"state_id": state_id, "split": result["split"],
                             "state_npz": str(state_path),
                             "lossless_state_binary": metadata[
                                 "lossless_state_binary"]})
                sites.append(item)
    states.sort(key=lambda row: (row["system"], row["case_id"],
                                 row["evaluation_count"]))
    sites.sort(key=lambda row: (row["system"], row["case_id"],
                                row["current_evaluation_count"],
                                row["archive_index"]))
    atomic_json(OUT / "archive_states.json", states)
    atomic_jsonl(OUT / "archive_states.jsonl", states)
    atomic_json(OUT / "archive_site_labels.json", sites)
    atomic_jsonl(OUT / "archive_site_labels.jsonl", sites)
    for split in SPLITS:
        atomic_jsonl(OUT / f"archive_states_{split}.jsonl",
                     [row for row in states if row["split"] == split])
        atomic_jsonl(OUT / f"archive_site_labels_{split}.jsonl",
                     [row for row in sites if row["split"] == split])

    sections = {}
    for system in SYSTEMS:
        for split in SPLITS + ("all",):
            rows = [row for row in results if row["system"] == system and
                    (split == "all" or row["split"] == split)]
            state_rows = [row for row in states if row["system"] == system and
                          (split == "all" or row["split"] == split)]
            site_rows = [row for row in sites if row["system"] == system and
                         (split == "all" or row["split"] == split)]
            sections[f"{system}:{split}"] = {
                "trajectories": len(rows),
                "archive_states": len(state_rows),
                "none_continue_states": sum(not row["target_any_good"]
                                            for row in state_rows),
                "first_any_good_states": sum(row["target_any_good"]
                                             for row in state_rows),
                "first_success_evaluation": distribution(
                    row["first_any_good_evaluation"] for row in rows),
                "good_site_labels": sum(row["target_good"] for row in site_rows),
                "bad_site_labels": sum(not row["target_good"] for row in site_rows),
                "canonical_first_good_newest": sum(
                    row["canonical_first_good_is_newest"] for row in rows),
                "canonical_first_good_older": sum(
                    row["canonical_first_good_is_older"] for row in rows),
                "canonical_first_good_incumbent": sum(
                    row["canonical_first_good_is_incumbent"] for row in rows),
                "canonical_first_good_nonincumbent": sum(
                    row["canonical_first_good_is_nonincumbent"] for row in rows),
                "older_site_bad_to_good_box_scale_transitions": sum(
                    len(row["older_site_bad_to_good_box_scale_transitions"])
                    for row in rows),
                "good_sites_at_first_positive": distribution(
                    row["first_positive_good_site_count"] for row in rows),
                "all_trajectories_reached_any_good": all(
                    row["first_any_good_evaluation"] is not None for row in rows),
                "labeling_cost": {
                    "site_count_queries": len(site_rows),
                    "new_query_cache_hits": sum(row["new_query_cache_hits"]
                                                for row in rows),
                    "prior_exact_n2_cache_reuse": sum(
                        row["prior_exact_n2_cache_reuse"] for row in rows),
                    "unchanged_n2_computations": sum(
                        row["unchanged_n2_computations"] for row in rows),
                    "n2_objective_evaluations": sum(
                        row["n2_evaluations"] for row in site_rows),
                    "n3_objective_evaluations": sum(
                        row["n3_objective_evaluations"] for row in site_rows),
                    "n3_accepted_steps": sum(
                        row["n3_accepted_steps"] for row in site_rows),
                },
            }
    after_hashes = hashes()
    report = {
        "label_definition": (
            "(archive site, current real N1 evaluation count) -> current-count "
            "production N2 box -> unchanged N2 -> integrated box-QP uncapped N3 "
            "-> existing final_loss < 1e-20 and finite-rollout outcome"),
        "systems": ["identifiable reflection_ball d=5", "kepler5 d=5"],
        "canonical_site_order": "archive insertion order",
        "state_stop": "first ANY_GOOD state; positive is absorbing",
        "whole_trajectory_splits_preserved": True,
        "ground_truth_runtime_features": False,
        "sections": sections,
        "total_trajectories": len(results),
        "total_archive_states": len(states),
        "total_site_labels": len(sites),
        "checkpoint_hashes_unchanged": before_hashes == after_hashes,
        "dataset_sha256": file_hash(OUT / "archive_states.json"),
        "site_labels_sha256": file_hash(OUT / "archive_site_labels.json"),
    }
    atomic_json(OUT / "report.json", report)
    atomic_json(OUT / "schema.json", {
        "version": SCHEMA_VERSION,
        "state_task": "archive state -> NONE_CONTINUE_N1 or ANY_GOOD_HANDOFF_NOW",
        "site_task": "(archive site, current evaluation count) -> BAD or GOOD",
        "state_input": "canonical lossless .edstate plus ragged .npz",
        "runtime_input_excludes_ground_truth": True,
        "site_order": "insertion order",
        "split_unit": "whole physical trajectory",
        "n2": "unchanged production count-derived box and local refinement",
        "n3": "dimension-general active-set box-QP; no semantic accepted-step cap",
    })
    atomic_json(OUT / "checkpoint_hashes_before.json", before_hashes)
    atomic_json(OUT / "checkpoint_hashes_after.json", after_hashes)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    CASES.mkdir(parents=True, exist_ok=True)
    STATES.mkdir(parents=True, exist_ok=True)
    if args.case:
        run_case(args.case[0], int(args.case[1]))
        return
    before = hashes()
    if not args.finalize:
        run_cases(args.workers)
    finalize(before)


if __name__ == "__main__":
    main()
