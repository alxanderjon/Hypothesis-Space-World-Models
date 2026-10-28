#!/usr/bin/env python3
"""Continue only the saved incomplete five-parameter Kepler N1 searches."""
from __future__ import annotations

import argparse
import copy
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
import torch

from experiments import characterize_d5_handoff as prior
from experiments import validate_d5_production_pipeline as production
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

CASE_IDS = (7, 9, 10, 22, 28, 30, 33, 34, 35, 41, 53, 54, 57, 58, 60)
START = 5_000
CAP = 20_000
COARSE = 500
FINE = 100
SOURCE = Path("outputs/d5_handoff_characterization/kepler_cases")
OUT = Path("outputs/d5_handoff_continuation_20000")
CASES = OUT / "cases"


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class SavedPrefixObjective:
    """Replay saved objective values, then delegate only genuinely new states."""

    def __init__(self, lower, upper, points, losses, live, start_index=0):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.width = np.asarray(upper, dtype=np.float64) - self.lower
        self.points = np.asarray(points, dtype=np.float64)
        self.losses = np.asarray(losses, dtype=np.float64)
        self.live = live
        self.index = int(start_index)
        self.replayed = 0
        self.live_calls = 0

    def __call__(self, theta) -> float:
        theta = np.asarray(theta, dtype=np.float64)
        if self.index < len(self.points):
            expected = self.lower + self.width * self.points[self.index]
            if not np.array_equal(theta, expected):
                difference = float(np.max(np.abs(theta - expected)))
                raise RuntimeError(
                    f"saved N1 replay diverged at evaluation {self.index + 1}: "
                    f"max theta difference {difference:.17g}"
                )
            value = float(self.losses[self.index])
            self.replayed += 1
        else:
            value = float(self.live(theta))
            self.live_calls += 1
        self.index += 1
        return value


def _snapshot(runner) -> dict:
    return dict(
        points=[np.asarray(point).copy() for point in runner.points],
        losses=list(runner.losses), events=copy.deepcopy(runner.events),
        traces=copy.deepcopy(runner.traces), iteration=int(runner.iteration),
        phase=str(runner.phase), pending=np.asarray(runner.pending).copy(),
        next_pending=int(runner.next_pending), stop_reason=runner.stop_reason,
    )


def _restore(snapshot: dict, objective, lower, upper, case_seed: int):
    runner = production.N1Runner(objective, lower, upper, case_seed)
    runner.points = [point.copy() for point in snapshot["points"]]
    runner.losses = list(snapshot["losses"])
    runner.events = copy.deepcopy(snapshot["events"])
    runner.traces = copy.deepcopy(snapshot["traces"])
    runner.iteration = int(snapshot["iteration"])
    runner.phase = str(snapshot["phase"])
    runner.pending = np.asarray(snapshot["pending"]).copy()
    runner.next_pending = int(snapshot["next_pending"])
    runner.stop_reason = snapshot["stop_reason"]
    return runner


def _reconstruct(case_id: int, live_objective, lower, upper):
    saved = np.load(SOURCE / f"case_{case_id:03d}" / "n1_state.npz")
    points = np.asarray(saved["points"], dtype=np.float64)
    losses = np.asarray(saved["losses"], dtype=np.float64)
    if points.shape != (START, len(lower)) or losses.shape != (START,):
        raise RuntimeError(f"case {case_id}: unexpected saved state shapes")
    replay = SavedPrefixObjective(lower, upper, points, losses, live_objective)
    runner = production.N1Runner(replay, lower, upper, 6_300_000 + case_id)
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        runner.run_to(START)
    finally:
        production.N1_GATE = old_gate
    actual_points = np.asarray(runner.points)
    actual_losses = np.asarray(runner.losses)
    if not np.array_equal(actual_points, points):
        raise RuntimeError(f"case {case_id}: regenerated points differ from saved state")
    if not np.array_equal(actual_losses, losses):
        raise RuntimeError(f"case {case_id}: regenerated losses differ from saved state")
    if replay.replayed != START or replay.live_calls:
        raise RuntimeError(f"case {case_id}: saved replay call accounting mismatch")
    runner.objective = live_objective
    return runner, dict(
        exact_points=True, exact_losses=True, evaluations=START,
        iteration=int(runner.iteration), phase=runner.phase,
        next_pending=int(runner.next_pending), pending_count=int(len(runner.pending)),
    )


def _handoff(runner, objective, torch_objective, trajectory, teacher, case_id,
             spacing: str) -> dict:
    endpoint = runner.endpoint()
    finish = production._finish(
        "kepler5", trajectory, objective, torch_objective,
        endpoint, teacher, f"continued_kepler_{case_id}_{endpoint['evaluations']}",
    )
    theta = np.asarray(endpoint["theta"], dtype=np.float64)
    radius = (np.asarray(endpoint["upper"]) - np.asarray(endpoint["lower"]))
    nominal_halfwidth = runner.width / endpoint["evaluations"] ** (1.0 / runner.dimension)
    return dict(
        spacing=spacing, n1_evaluations=int(endpoint["evaluations"]),
        n1_incumbent=theta.tolist(), n1_loss=float(endpoint["loss"]),
        offline_n1_parameter_distance=float(np.linalg.norm(theta - teacher)),
        n2_box_nominal_halfwidth=nominal_halfwidth.tolist(),
        n2_box_actual_width=radius.tolist(),
        offline_teacher_inside_n2_box=bool(
            np.all(teacher >= endpoint["lower"]) and
            np.all(teacher <= endpoint["upper"])
        ),
        **finish,
    )


def _refine_interval(lower_snapshot: dict, coarse_runner, coarse_target: int,
                     objective, torch_objective, trajectory, teacher, lower, upper,
                     case_id: int):
    # Reuse the already evaluated coarse interval as objective supervision.  This
    # repeats proposal/state construction but performs no physical re-evaluation.
    replay = SavedPrefixObjective(
        lower, upper, np.asarray(coarse_runner.points),
        np.asarray(coarse_runner.losses), objective,
        start_index=len(lower_snapshot["losses"]),
    )
    runner = _restore(lower_snapshot, replay, lower, upper, 6_300_000 + case_id)
    checks = []
    start = len(lower_snapshot["losses"])
    for target in range(start + FINE, coarse_target + 1, FINE):
        runner.run_to(target)
        expected_points = np.asarray(coarse_runner.points[:target])
        expected_losses = np.asarray(coarse_runner.losses[:target])
        if not np.array_equal(np.asarray(runner.points), expected_points):
            raise RuntimeError(f"case {case_id}: fine replay point mismatch at {target}")
        if not np.array_equal(np.asarray(runner.losses), expected_losses):
            raise RuntimeError(f"case {case_id}: fine replay loss mismatch at {target}")
        check = _handoff(runner, objective, torch_objective, trajectory, teacher,
                         case_id, "fine_100")
        checks.append(check)
        if check["recovered"]:
            return check, checks, replay
    raise RuntimeError(f"case {case_id}: coarse success was absent in fine replay")


def run_case(case_id: int) -> dict:
    if case_id not in CASE_IDS:
        raise ValueError(case_id)
    torch.set_num_threads(1)
    case_dir = CASES / f"case_{case_id:03d}"
    result_path = case_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trajectory, spec = production._kepler_case(case_id)
    objective, torch_objective = production._objectives("kepler5", trajectory)
    lower, upper, teacher = production._config("kepler5")
    runner, reconstruction = _reconstruct(case_id, objective, lower, upper)
    coarse_checks = []
    refinement_checks = []
    first = None
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        for target in range(START + COARSE, CAP + 1, COARSE):
            lower_snapshot = _snapshot(runner)
            runner.run_to(target)
            check = _handoff(runner, objective, torch_objective, trajectory,
                             teacher, case_id, "coarse_500")
            coarse_checks.append(check)
            if check["recovered"]:
                first, refinement_checks, replay = _refine_interval(
                    lower_snapshot, runner, target, objective, torch_objective,
                    trajectory, teacher, lower, upper, case_id,
                )
                if replay.live_calls:
                    raise RuntimeError(f"case {case_id}: fine interval repeated physical calls")
                break
            if runner.stop_reason == "no_progress":
                break
    finally:
        production.N1_GATE = old_gate
    if first is not None:
        status = "SUCCESSFUL_HANDOFF_FOUND"
        final_points = np.asarray(runner.points[:first["n1_evaluations"]])
        final_losses = np.asarray(runner.losses[:first["n1_evaluations"]])
    elif len(runner.losses) >= CAP:
        status = "SEARCH_INCOMPLETE_AT_20000"
        final_points = np.asarray(runner.points)
        final_losses = np.asarray(runner.losses)
    else:
        status = "SEARCH_INCOMPLETE_NO_PROGRESS"
        final_points = np.asarray(runner.points)
        final_losses = np.asarray(runner.losses)
    np.savez_compressed(case_dir / "n1_state.npz", points=final_points,
                        losses=final_losses)
    result = dict(
        case_id=case_id, initial_condition=spec, status=status,
        reconstruction=reconstruction, first_successful_handoff=first,
        coarse_checks=coarse_checks, refinement_checks=refinement_checks,
        final_n1_evaluations=int(len(final_losses)),
        continuation_physical_evaluations=int(len(final_losses) - START),
        elapsed_seconds=float(time.monotonic() - started),
    )
    _atomic_json(result_path, result)
    return result


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=int(len(values)), minimum=float(values.min()),
                median=float(np.median(values)),
                p90=float(np.quantile(values, 0.9)), maximum=float(values.max()))


def _run_tasks(indices, workers):
    for offset in range(0, len(indices), workers):
        batch = indices[offset:offset + workers]
        children = []
        for index in batch:
            command = [sys.executable, "-m", __spec__.name, "--case", str(index)]
            children.append((index, subprocess.Popen(command, env=os.environ.copy())))
        failed = [(index, child.wait()) for index, child in children]
        failed = [(index, code) for index, code in failed if code]
        if failed:
            raise RuntimeError(f"continuation case failures: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CASES.mkdir(parents=True, exist_ok=True)
    if args.case is not None:
        result = run_case(args.case)
        first = result["first_successful_handoff"]
        print(json.dumps(dict(
            case=args.case, status=result["status"],
            evaluation=None if first is None else first["n1_evaluations"],
            loss=None if first is None else first["n1_loss"],
            seconds=result["elapsed_seconds"],
        ), indent=2), flush=True)
        return
    before = hashes(); _atomic_json(OUT / "checkpoint_hashes_before.json", before)
    started = time.monotonic()
    _run_tasks(list(CASE_IDS), max(1, args.workers))
    rows = [json.loads((CASES / f"case_{case_id:03d}" / "result.json").read_text())
            for case_id in CASE_IDS]
    successes = [row for row in rows if row["first_successful_handoff"] is not None]
    first = [row["first_successful_handoff"] for row in successes]
    after = hashes(); _atomic_json(OUT / "checkpoint_hashes_after.json", after)
    report = dict(
        status="COMPLETE" if len(successes) == len(rows) else "SEARCH_INCOMPLETE_CASES",
        cases=len(rows), successful_handoffs=len(successes),
        search_incomplete_at_20000=sum(row["status"] == "SEARCH_INCOMPLETE_AT_20000"
                                       for row in rows),
        search_incomplete_no_progress=sum(row["status"] == "SEARCH_INCOMPLETE_NO_PROGRESS"
                                          for row in rows),
        incomplete_case_ids=[row["case_id"] for row in rows
                             if row["first_successful_handoff"] is None],
        successful_handoff_evaluations=_distribution(
            [item["n1_evaluations"] for item in first]),
        successful_handoff_losses=_distribution([item["n1_loss"] for item in first]),
        successful_handoff_normalized_box_halfwidth=_distribution(
            [item["n1_evaluations"] ** (-1.0 / 5.0) for item in first]),
        checkpoint_hashes_unchanged=before == after,
        elapsed_seconds=float(time.monotonic() - started), cases_detail=rows,
    )
    _atomic_json(OUT / "report.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "cases", "successful_handoffs", "search_incomplete_at_20000",
        "search_incomplete_no_progress", "incomplete_case_ids",
        "successful_handoff_evaluations", "successful_handoff_losses",
        "checkpoint_hashes_unchanged")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
