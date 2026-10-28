#!/usr/bin/env python3
"""Expand chronological archive-handoff data with new real d=5 trajectories.

No optimizer decision is changed.  N1 is advanced one real evaluation at a
time, with termination delegated to the archive ANY_GOOD label requested for
this dataset.  Every site/count query runs unchanged N2 and the integrated
dimension-general box-QP N3 to a genuine numerical stop.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
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
from scipy.stats import qmc
import torch

from experiments import validate_d5_production_pipeline as production
from experiments.bouncing_ball import run_first_transfer as ball_teacher
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.kepler import deterministic_teacher as kepler_teacher
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.validate_lossless_edirect_representation import RoundState, encode


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

BASE = Path("outputs/chronological_archive_handoff_supervision")
OUT = Path("outputs/chronological_archive_handoff_supervision_expanded")
CASES = OUT / "cases"
CACHE = OUT / "query_cache"
STATES = OUT / "states"
TRAJECTORIES = OUT / "trajectories"
MANIFEST = OUT / "manifest.json"
SPLITS = ("train", "held_out", "fresh_test")
SYSTEMS = ("reflection_ball", "kepler5")
BATCH = 64
SCHEMA_VERSION = 1


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"),
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sobol_point(index: int, dimension: int, seed: int) -> np.ndarray:
    power = max(1, int(math.ceil(math.log2(index + 1))))
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(power)[index]


def ball_case(index: int) -> tuple[np.ndarray, dict]:
    u = sobol_point(index, 4, production.BALL_SEED)
    x = production.BALL_IC_BOUNDS["x"][0] + np.float64(u[0]) * 9.0
    y = production.BALL_IC_BOUNDS["y"][0] + np.float64(u[1]) * 9.0
    speed = np.float64(0.5) + np.float64(u[2]) * np.float64(2.5)
    angle = np.float64(u[3]) * np.float64(2.0 * math.pi)
    vx = np.float64(speed * np.cos(angle)); vy = np.float64(speed * np.sin(angle))
    trajectory = ball_teacher.TEACHER.simulate_bouncing_ball(
        x, y, vx, vy, ball_teacher.DT, ball_teacher.TOTAL_STEPS,
        xmin=ball_teacher.XMIN, xmax=ball_teacher.XMAX,
        ymin=ball_teacher.YMIN, ymax=ball_teacher.YMAX)
    return np.asarray(trajectory, np.float64), {
        "x": float(x), "y": float(y), "speed": float(speed),
        "direction": float(angle), "vx": float(vx), "vy": float(vy),
        "sobol_index": int(index), "sobol_unit": u.tolist(),
        "seed": production.BALL_SEED,
    }


def kepler_case(index: int) -> tuple[np.ndarray, dict]:
    u = sobol_point(index, 4, production.KEPLER_SEED)
    eccentricity = np.float64(0.18) + np.float64(u[0]) * np.float64(0.50)
    periapsis = np.float64(0.72) + np.float64(u[1]) * np.float64(0.80)
    orientation = np.float64(u[2]) * np.float64(2.0 * math.pi)
    phase_steps = 13 + min(198, int(math.floor(float(u[3]) * 199.0)))
    initial = kepler_teacher.initial_state(eccentricity, periapsis, orientation)
    phased = kepler_teacher.numpy_rollout(initial, steps=phase_steps)[-1]
    trajectory = kepler_teacher.numpy_rollout(phased, steps=kepler_teacher.TOTAL_STEPS)
    return np.asarray(trajectory, np.float64), {
        "eccentricity": float(eccentricity), "periapsis": float(periapsis),
        "orientation": float(orientation), "phase_steps": int(phase_steps),
        "sobol_index": int(index), "sobol_unit": u.tolist(),
        "seed": production.KEPLER_SEED,
    }


def ball_identifiability(index: int) -> dict:
    trajectory, spec = ball_case(index)
    prefix = trajectory[:ball.ball.PREFIX_STATES - 1]
    signatures = [ball.event_signature(state, ball.TEACHER) for state in prefix]
    impacts = int(sum(len(value) for value in signatures))
    if impacts == 0:
        return {"sobol_index": index, "initial_condition": spec,
                "prefix_impacts": 0, "jacobian_rank": None,
                "classification": "REJECT_ZERO_IMPACT"}
    inputs = torch.from_numpy(prefix.copy())
    theta = torch.from_numpy(ball.TEACHER.copy())

    def transition_map(parameters: torch.Tensor) -> torch.Tensor:
        return torch.stack([ball.torch_step(state, parameters)
                            for state in inputs]).reshape(-1)

    jacobian = torch.autograd.functional.jacobian(
        transition_map, theta, vectorize=True, strategy="forward-mode"
    ).detach().numpy()
    singular = np.linalg.svd(jacobian, compute_uv=False)
    tolerance = max(jacobian.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "sobol_index": index, "initial_condition": spec,
        "prefix_impacts": impacts,
        "prefix_corner_steps": int(sum(any(mask == 3 for mask in value)
                                       for value in signatures)),
        "prefix_multiple_impact_steps": int(sum(len(value) > 1 for value in signatures)),
        "jacobian_rank": rank, "rank_tolerance": float(tolerance),
        "singular_values": singular.tolist(),
        "classification": ("ACCEPT_IDENTIFIABLE" if rank == 5
                           else "REJECT_RANK_DEFICIENT"),
    }


def cohort_splits(system: str, indices: list[int], stage: int) -> dict[int, str]:
    ordered = sorted(indices, key=lambda value: hashlib.sha256(
        f"expanded-stage-{stage}:{system}:{value}".encode()).hexdigest())
    train = int(0.70 * len(ordered)); held = int(0.15 * len(ordered))
    return {value: ("train" if offset < train else
                    "held_out" if offset < train + held else "fresh_test")
            for offset, value in enumerate(ordered)}


def ensure_manifest(stage: int) -> dict:
    existing = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {
        "version": 1, "base_sobol_indices": [0, 63], "stages": [],
        "ball_rejections": [], "trajectories": [],
    }
    completed = {int(row["stage"]) for row in existing["stages"]}
    if stage in completed:
        return existing
    if stage != len(completed) + 1:
        raise RuntimeError("stages must be created consecutively")
    used_ball = {int(row["sobol_index"]) for row in existing["trajectories"]
                 if row["system"] == "reflection_ball"}
    used_kepler = {int(row["sobol_index"]) for row in existing["trajectories"]
                   if row["system"] == "kepler5"}
    next_ball = max({63, *used_ball, *(int(row["sobol_index"])
                    for row in existing["ball_rejections"])}) + 1
    accepted_ball = []
    while len(accepted_ball) < BATCH:
        audit = ball_identifiability(next_ball)
        if audit["classification"] == "ACCEPT_IDENTIFIABLE":
            accepted_ball.append(audit)
        else:
            existing["ball_rejections"].append(audit)
        next_ball += 1
    first_kepler = max({63, *used_kepler}) + 1
    kepler_indices = list(range(first_kepler, first_kepler + BATCH))
    ball_indices = [int(row["sobol_index"]) for row in accepted_ball]
    ball_splits = cohort_splits("reflection_ball", ball_indices, stage)
    kepler_splits = cohort_splits("kepler5", kepler_indices, stage)
    for audit in accepted_ball:
        index = int(audit["sobol_index"])
        existing["trajectories"].append({
            "stage": stage, "system": "reflection_ball", "case_id": index,
            "sobol_index": index, "split": ball_splits[index],
            "identifiability": audit,
        })
    for index in kepler_indices:
        _, spec = kepler_case(index)
        existing["trajectories"].append({
            "stage": stage, "system": "kepler5", "case_id": index,
            "sobol_index": index, "split": kepler_splits[index],
            "initial_condition": spec,
        })
    existing["stages"].append({
        "stage": stage, "new_reflection_ball": len(ball_indices),
        "new_kepler5": len(kepler_indices),
        "reflection_ball_indices": ball_indices,
        "kepler5_indices": kepler_indices,
        "split_assignment_fixed_before_labels": True,
    })
    existing["trajectories"].sort(key=lambda row: (row["stage"], row["system"],
                                                     row["case_id"]))
    atomic_json(MANIFEST, existing)
    return existing


class ArchiveN1Runner(production.N1Runner):
    """Production N1 decisions, advanced one evaluation at a time.

    The obsolete loss gate is not consulted because archive ANY_GOOD is the
    stopping target being labeled.  Proposal generation and selection are the
    unchanged production operations.
    """
    def __init__(self, objective, lower, upper, case_id):
        super().__init__(objective, lower, upper, case_id)
        self._design = edirect.initial_design(self.dimension)

    def advance_one(self) -> None:
        while True:
            if self.phase == "initial":
                if self.next_pending < len(self._design):
                    self.evaluate(self._design[self.next_pending], "initial")
                    self.next_pending += 1
                    if self.next_pending == len(self._design):
                        self.phase = "next_round"; self.next_pending = 0
                    return
                self.phase = "next_round"; self.next_pending = 0
            if self.phase == "next_round":
                self.pending = self.api["_cell_proposals"](
                    np.asarray(self.points), np.asarray(self.losses),
                    self.case_id, self.iteration)
                self.next_pending = 0; self.phase = "cell"
            if self.phase == "cell" and self.next_pending < len(self.pending):
                self.evaluate(self.pending[self.next_pending], "cell")
                self.next_pending += 1
                return
            if self.phase == "cell":
                self.phase = "greedy"
            if self.phase == "greedy":
                unit = self.api["_greedy_proposal"](
                    np.asarray(self.points), np.asarray(self.losses),
                    self.case_id, self.iteration)
                self.iteration += 1; self.phase = "next_round"
                if unit is not None:
                    self.evaluate(unit, "greedy")
                    return
                if len(self.pending) == 0:
                    raise RuntimeError("unchanged N1 reached no-progress before archive success")


def trajectory_for(spec: dict) -> tuple[np.ndarray, dict]:
    return (ball_case(spec["sobol_index"]) if spec["system"] == "reflection_ball"
            else kepler_case(spec["sobol_index"]))


def query_path(spec: dict, n: int, index: int) -> Path:
    return (CACHE / spec["system"] / f"case_{spec['case_id']:05d}" /
            f"n{n:05d}_i{index + 1:05d}.json")


def state_id(spec: dict, n: int) -> str:
    return f"{spec['system']}_new_case_{spec['case_id']:05d}_n{n:05d}"


def check_site(spec: dict, trajectory: np.ndarray, objective, torch_objective,
               lower: np.ndarray, upper: np.ndarray, teacher: np.ndarray,
               runner: ArchiveN1Runner, n: int, index: int) -> dict:
    path = query_path(spec, n, index)
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    theta = lower + (upper - lower) * runner.points[index]
    radius = (upper - lower) / float(n) ** (1.0 / len(lower))
    endpoint = {
        "theta": theta, "loss": float(runner.losses[index]), "evaluations": n,
        "lower": np.maximum(lower, theta - radius),
        "upper": np.minimum(upper, theta + radius),
        "stop_reason": "chronological_archive_handoff_expansion",
    }
    finish = production._finish(
        spec["system"], trajectory, objective, torch_objective, endpoint, teacher,
        f"expanded_archive_{spec['system']}_{spec['case_id']}_{n}_{index}")
    if finish["n3_stop_reason"] == "max_accepted_steps":
        raise RuntimeError("corrected N3 used the removed accepted-step cap")
    row = {
        "schema_version": SCHEMA_VERSION, "cohort": f"new_stage_{spec['stage']}",
        "system": spec["system"], "case_id": spec["case_id"], "split": spec["split"],
        "current_evaluation_count": n, "archive_index": index,
        "insertion_evaluation": index + 1, "canonical_site_order": index,
        "theta": theta.tolist(), "archived_loss": float(runner.losses[index]),
        "target_good": bool(finish["full_pipeline_pass"]),
        "n3_semantic_accepted_step_cap": None,
        "n2_source": "computed_unchanged_n2",
        "elapsed_seconds": float(time.monotonic() - started), **finish,
    }
    atomic_json(path, row)
    return row


def save_state(spec: dict, runner: ArchiveN1Runner, lower: np.ndarray,
               upper: np.ndarray, n: int, labels: list[dict]) -> dict:
    identifier = state_id(spec, n)
    binary = STATES / f"{identifier}.edstate"; npz = STATES / f"{identifier}.npz"
    points = np.asarray(runner.points[:n], np.float64)
    losses = np.asarray(runner.losses[:n], np.float64)
    payload = encode(RoundState(
        5, (7_100_000 if spec["system"] == "reflection_ball" else 7_200_000)
        + int(spec["case_id"]), n, lower.copy(), upper.copy(), points, losses))
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        temporary = binary.with_suffix(".edstate.tmp")
        temporary.write_bytes(payload); temporary.replace(binary)
    if binary.read_bytes() != payload:
        raise RuntimeError(f"state mismatch {binary}")
    physical = lower + (upper - lower) * points
    if not npz.exists():
        np.savez_compressed(
            npz, schema_version=np.int64(SCHEMA_VERSION),
            system=np.asarray(spec["system"]), case_id=np.int64(spec["case_id"]),
            split=np.asarray(spec["split"]), dimension=np.int64(5),
            evaluation_count=np.int64(n), iteration=np.int64(n),
            lower=lower, upper=upper, normalized_sites=points,
            physical_sites=physical, site_losses=losses,
            insertion_order=np.arange(n, dtype=np.int64))
    good = [row for row in labels if row["target_good"]]
    return {
        "schema_version": SCHEMA_VERSION, "cohort": f"new_stage_{spec['stage']}",
        "state_id": identifier, "system": spec["system"],
        "case_id": spec["case_id"], "split": spec["split"], "dimension": 5,
        "evaluation_count": n,
        "state_target": "ANY_GOOD_HANDOFF_NOW" if good else "NONE_CONTINUE_N1",
        "target_any_good": bool(good), "site_count": n,
        "good_site_indices": [row["archive_index"] for row in good],
        "bad_site_indices": [row["archive_index"] for row in labels
                             if not row["target_good"]],
        "incumbent_site_index": int(np.argmin(losses)),
        "canonical_site_order": "archive insertion order, zero-based index",
        "state_npz": str(npz), "lossless_state_binary": str(binary),
        "lossless_state_sha256": hashlib.sha256(payload).hexdigest(),
        "ground_truth_absent_from_runtime_state": True,
    }


def run_case(spec: dict) -> dict:
    result_path = CASES / spec["system"] / f"case_{spec['case_id']:05d}.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    torch.set_num_threads(1)
    trajectory, initial = trajectory_for(spec)
    objective, torch_objective = production._objectives(spec["system"], trajectory)
    lower, upper, teacher = production._config(spec["system"])
    runner = ArchiveN1Runner(
        objective, lower, upper,
        (7_100_000 if spec["system"] == "reflection_ball" else 7_200_000)
        + int(spec["case_id"]))
    states = []; labels = []; started = time.monotonic()
    while True:
        runner.advance_one(); n = len(runner.losses)
        current = [check_site(spec, trajectory, objective, torch_objective,
                              lower, upper, teacher, runner, n, index)
                   for index in range(n)]
        labels.extend(current)
        state = save_state(spec, runner, lower, upper, n, current); states.append(state)
        if state["target_any_good"]:
            break
    n = len(runner.losses); current = labels[-n:]
    good = [row for row in current if row["target_good"]]
    newest = n - 1; incumbent = int(states[-1]["incumbent_site_index"])
    newest_good = any(row["archive_index"] == newest for row in good)
    incumbent_good = any(row["archive_index"] == incumbent for row in good)
    mixed = bool(good) and len(good) < n
    tags = {
        "newest_site_bad": not newest_good,
        "incumbent_bad": not incumbent_good,
        "older_site_required": not newest_good,
        "nonincumbent_site_required": not incumbent_good,
        "mixed_good_bad": mixed,
        "mixed_not_solved_by_newest_or_incumbent": bool(
            mixed and (not newest_good or not incumbent_good)),
        "multiple_good_sites": len(good) > 1,
    }
    tags["hard"] = any(tags[key] for key in (
        "newest_site_bad", "incumbent_bad", "older_site_required",
        "nonincumbent_site_required", "mixed_not_solved_by_newest_or_incumbent"))
    trajectory_path = TRAJECTORIES / spec["system"] / f"case_{spec['case_id']:05d}.npz"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    if not trajectory_path.exists():
        np.savez_compressed(trajectory_path, trajectory=trajectory)
    result = {
        "schema_version": SCHEMA_VERSION, "cohort": f"new_stage_{spec['stage']}",
        "stage": spec["stage"], "system": spec["system"],
        "case_id": spec["case_id"], "sobol_index": spec["sobol_index"],
        "split": spec["split"], "initial_condition": initial,
        "trajectory_path": str(trajectory_path), "trajectory_sha256": hashlib.sha256(
            np.ascontiguousarray(trajectory).tobytes()).hexdigest(),
        "state_ids": [row["state_id"] for row in states],
        "chronological_states": len(states), "none_states": len(states) - 1,
        "first_any_good_evaluation": n,
        "first_positive_good_site_indices": [row["archive_index"] for row in good],
        "first_positive_good_site_count": len(good),
        "newest_site_index": newest, "incumbent_site_index": incumbent,
        "hard_tags": tags, "site_labels": len(labels),
        "good_site_labels": sum(row["target_good"] for row in labels),
        "bad_site_labels": sum(not row["target_good"] for row in labels),
        "n1_events": runner.events, "n1_continuous_traces": runner.traces,
        "elapsed_seconds": float(time.monotonic() - started),
        "stopped_at_first_positive": True,
    }
    atomic_json(result_path, result)
    print(spec["system"], spec["case_id"], spec["split"],
          f"first={n}", f"good={len(good)}/{n}",
          f"hard={tags['hard']}", flush=True)
    return result


def worker(system: str, case_id: int) -> None:
    manifest = json.loads(MANIFEST.read_text())
    spec = next(row for row in manifest["trajectories"]
                if row["system"] == system and int(row["case_id"]) == case_id)
    run_case(spec)


def run_stage(stage: int, workers: int) -> None:
    manifest = ensure_manifest(stage)
    specs = [row for row in manifest["trajectories"] if int(row["stage"]) == stage]
    workers = max(1, min(int(workers), 4))
    for offset in range(0, len(specs), workers):
        children = []
        for spec in specs[offset:offset + workers]:
            command = [sys.executable, "-m", __spec__.name, "--case",
                       spec["system"], str(spec["case_id"])]
            children.append((spec, subprocess.Popen(command, env=os.environ.copy())))
        failures = [(spec["system"], spec["case_id"], child.wait())
                    for spec, child in children]
        failures = [row for row in failures if row[2]]
        if failures:
            raise RuntimeError(failures)


def distribution(values) -> dict:
    values = list(map(int, values)); counts = Counter(values)
    return {"count": len(values), "minimum": min(values),
            "median": float(np.median(values)), "maximum": max(values),
            "histogram": {str(key): counts[key] for key in sorted(counts)}}


def finalize(stage: int, before_hashes: dict) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    specs = [row for row in manifest["trajectories"] if int(row["stage"]) <= stage]
    result_paths = [(spec, CASES / spec["system"] /
                     f"case_{spec['case_id']:05d}.json") for spec in specs]
    unresolved_specs = [spec for spec, path in result_paths if not path.exists()]
    results = [json.loads(path.read_text()) for _, path in result_paths if path.exists()]
    new_states = []; new_sites = []
    for result in results:
        for identifier in result["state_ids"]:
            npz_path = STATES / f"{identifier}.npz"
            n = int(np.load(npz_path)["evaluation_count"])
            rows = [json.loads(query_path(result, n, index).read_text())
                    for index in range(n)]
            good = [row["archive_index"] for row in rows if row["target_good"]]
            new_states.append({
                "schema_version": SCHEMA_VERSION, "cohort": result["cohort"],
                "state_id": identifier, "system": result["system"],
                "case_id": result["case_id"], "split": result["split"],
                "dimension": 5, "evaluation_count": n,
                "state_target": ("ANY_GOOD_HANDOFF_NOW" if good
                                 else "NONE_CONTINUE_N1"),
                "target_any_good": bool(good), "site_count": n,
                "good_site_indices": good,
                "bad_site_indices": [row["archive_index"] for row in rows
                                     if not row["target_good"]],
                "incumbent_site_index": int(np.argmin(np.load(npz_path)["site_losses"])),
                "canonical_site_order": "archive insertion order, zero-based index",
                "state_npz": str(npz_path),
                "lossless_state_binary": str(STATES / f"{identifier}.edstate"),
                "lossless_state_sha256": file_hash(STATES / f"{identifier}.edstate"),
                "ground_truth_absent_from_runtime_state": True,
            })
            for row in rows:
                new_sites.append(dict(row, state_id=identifier,
                                      state_npz=str(npz_path),
                                      lossless_state_binary=str(STATES / f"{identifier}.edstate")))
    new_states.sort(key=lambda row: (row["system"], row["case_id"], row["evaluation_count"]))
    new_sites.sort(key=lambda row: (row["system"], row["case_id"],
                                    row["current_evaluation_count"], row["archive_index"]))
    base_states = [json.loads(line) for line in (BASE / "archive_states.jsonl").read_text().splitlines()]
    base_sites = [json.loads(line) for line in (BASE / "archive_site_labels.jsonl").read_text().splitlines()]
    expanded_states = base_states + new_states; expanded_sites = base_sites + new_sites
    for prefix, rows in (("new_archive_states", new_states),
                         ("new_archive_site_labels", new_sites),
                         ("expanded_archive_states", expanded_states),
                         ("expanded_archive_site_labels", expanded_sites)):
        atomic_jsonl(OUT / f"{prefix}.jsonl", rows)
        for split in SPLITS:
            atomic_jsonl(OUT / f"{prefix}_{split}.jsonl",
                         [row for row in rows if row["split"] == split])

    positives = [row for row in new_states if row["target_any_good"]]
    by_key = {(row["system"], row["case_id"]): row for row in results}
    for row in positives:
        row["hard_tags"] = by_key[(row["system"], row["case_id"])]["hard_tags"]
    hard = [row for row in positives if row["hard_tags"]["hard"]]
    atomic_jsonl(OUT / "new_hard_first_positive_states.jsonl", hard)
    for split in SPLITS:
        atomic_jsonl(OUT / f"new_hard_first_positive_states_{split}.jsonl",
                     [row for row in hard if row["split"] == split])

    # Preserve the original corpus in the separately tagged hard subset too.
    # The tags are derived solely from labels and bookkeeping present at the
    # first positive state; no ground-truth quantity enters the runtime state.
    base_site_by_state = {}
    for site in base_sites:
        base_site_by_state.setdefault(site["state_id"], []).append(site)
    base_hard = []
    for source in base_states:
        if not source["target_any_good"]:
            continue
        row = dict(source, cohort="base")
        sites = base_site_by_state[row["state_id"]]
        good_indices = {int(site["archive_index"]) for site in sites
                        if site["target_good"]}
        newest = int(row["evaluation_count"]) - 1
        incumbent = int(row["incumbent_site_index"])
        newest_good = newest in good_indices
        incumbent_good = incumbent in good_indices
        mixed = bool(good_indices) and len(good_indices) < int(row["site_count"])
        tags = {
            "newest_site_bad": not newest_good,
            "incumbent_bad": not incumbent_good,
            "older_site_required": not newest_good,
            "nonincumbent_site_required": not incumbent_good,
            "mixed_good_bad": mixed,
            "mixed_not_solved_by_newest_or_incumbent": bool(
                mixed and (not newest_good or not incumbent_good)),
            "multiple_good_sites": len(good_indices) > 1,
        }
        tags["hard"] = any(tags[key] for key in (
            "newest_site_bad", "incumbent_bad", "older_site_required",
            "nonincumbent_site_required",
            "mixed_not_solved_by_newest_or_incumbent"))
        row["hard_tags"] = tags
        if tags["hard"]:
            base_hard.append(row)
    expanded_hard = base_hard + hard
    atomic_jsonl(OUT / "expanded_hard_first_positive_states.jsonl", expanded_hard)
    for split in SPLITS:
        atomic_jsonl(OUT / f"expanded_hard_first_positive_states_{split}.jsonl",
                     [row for row in expanded_hard if row["split"] == split])

    stage_reports = {}
    stage_results = [row for row in results if int(row["stage"]) == stage]
    for system in SYSTEMS:
        rows = [row for row in stage_results if row["system"] == system]
        planned = [row for row in specs if int(row["stage"]) == stage
                   and row["system"] == system]
        unresolved = [row for row in unresolved_specs if int(row["stage"]) == stage
                      and row["system"] == system]
        state_rows = [row for row in new_states if row["system"] == system
                      and row["cohort"] == f"new_stage_{stage}"]
        site_rows = [row for row in new_sites if row["system"] == system
                     and row["cohort"] == f"new_stage_{stage}"]
        first = [row for row in state_rows if row["target_any_good"]]
        hard_rows = [row for row in rows if row["hard_tags"]["hard"]]
        stage_reports[system] = {
            "planned_trajectories": len(planned),
            "new_trajectories": len(rows),
            "unresolved_trajectories": len(unresolved),
            "unresolved_case_ids": [row["case_id"] for row in unresolved],
            "chronological_states": len(state_rows),
            "none_states": sum(not row["target_any_good"] for row in state_rows),
            "first_any_states": len(first),
            "first_success_evaluation": distribution(
                row["first_any_good_evaluation"] for row in rows),
            "good_site_labels": sum(row["target_good"] for row in site_rows),
            "bad_site_labels": sum(not row["target_good"] for row in site_rows),
            "newest_site_successes": sum(not row["hard_tags"]["newest_site_bad"] for row in rows),
            "incumbent_successes": sum(not row["hard_tags"]["incumbent_bad"] for row in rows),
            "hard_first_positive_states": len(hard_rows),
            "older_site_required": sum(row["hard_tags"]["older_site_required"] for row in rows),
            "nonincumbent_site_required": sum(
                row["hard_tags"]["nonincumbent_site_required"] for row in rows),
            "multiple_good_cases": sum(row["hard_tags"]["multiple_good_sites"] for row in rows),
            "hard_by_split": {split: sum(row["split"] == split for row in hard_rows)
                              for split in SPLITS},
            "all_stopped_at_first_any": all(row["stopped_at_first_positive"] for row in rows),
        }
    cumulative_hard = Counter((row["system"], row["split"]) for row in hard)
    after_hashes = hashes()
    report = {
        "stage": stage, "stage_report": stage_reports,
        "planned_new_trajectories": len(specs),
        "unresolved_new_trajectories": len(unresolved_specs),
        "unresolved_cases": [{"system": row["system"], "case_id": row["case_id"],
                              "stage": row["stage"], "split": row["split"]}
                             for row in unresolved_specs],
        "cumulative_new_trajectories": len(results),
        "cumulative_new_states": len(new_states),
        "cumulative_new_site_labels": len(new_sites),
        "cumulative_hard_first_positive_states": len(hard),
        "expanded_hard_first_positive_states": len(expanded_hard),
        "cumulative_hard_by_system_split": {
            f"{system}:{split}": cumulative_hard[(system, split)]
            for system in SYSTEMS for split in SPLITS},
        "expanded_total_states": len(expanded_states),
        "expanded_total_site_labels": len(expanded_sites),
        "whole_trajectory_splits": True,
        "split_assignment_fixed_before_labels": True,
        "checkpoint_hashes_unchanged": before_hashes == after_hashes,
    }
    atomic_json(OUT / f"stage_{stage:02d}_report.json", report)
    atomic_json(OUT / "latest_report.json", report)
    atomic_json(OUT / "checkpoint_hashes_after.json", after_hashes)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case", nargs=2)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    for path in (OUT, CASES, CACHE, STATES, TRAJECTORIES):
        path.mkdir(parents=True, exist_ok=True)
    before = hashes()
    if not (OUT / "checkpoint_hashes_before.json").exists():
        atomic_json(OUT / "checkpoint_hashes_before.json", before)
    if args.case:
        worker(args.case[0], int(args.case[1])); return
    ensure_manifest(args.stage)
    if not args.finalize:
        run_stage(args.stage, args.workers)
    finalize(args.stage, json.loads((OUT / "checkpoint_hashes_before.json").read_text()))


if __name__ == "__main__":
    main()
