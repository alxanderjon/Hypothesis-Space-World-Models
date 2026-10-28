#!/usr/bin/env python3
"""Broad, checkpointed validation of the unchanged production d=5 pipeline.

The only new machinery in this file is experiment orchestration and Sobol
sampling of valid physical initial conditions.  N1 uses the existing
continuous E-DIRECT operations and its production stop conditions; N2 and N3
are called through their existing implementations.  Ground truth is used only
afterward for diagnostics and the established recovery check.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scipy.stats import qmc
import torch

from experiments import audit_known_structure as known
from experiments.bouncing_ball import run_first_transfer as ball_teacher
from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.kepler import deterministic_teacher as kepler_teacher
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho import continuous_surrogate_edirect as continuous
from experiments.sho import dimension_general_edirect as edirect
from experiments.sho.deterministic_full_k_framework import hashes


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/d5_production_pipeline_validation")
CASES = OUT / "cases"
STAGE_1 = 64
STAGE_2 = 128
PRODUCTION_CAP = 5_000
DIAGNOSTIC_CAP = 10_000
N1_GATE = 1.0e-4
FINAL_GATE = 1.0e-20
BALL_SEED = 20_261_103
KEPLER_SEED = 20_261_104

# These are the existing valid physical initial-condition ranges.  The ball
# values are the defaults of build_bouncing_ball_trajectories.  The Kepler
# values are the envelope already represented by the existing fresh transfer
# cases; orientation spans its full natural range and phase remains integral.
BALL_IC_BOUNDS = dict(x=(-4.5, 4.5), y=(-4.5, 4.5), speed=(0.5, 3.0),
                      direction=(0.0, 2.0 * math.pi))
KEPLER_IC_BOUNDS = dict(eccentricity=(0.18, 0.68), periapsis=(0.72, 1.52),
                         orientation=(0.0, 2.0 * math.pi), phase_steps=(13, 211))


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sobol(count: int, dimension: int, seed: int) -> np.ndarray:
    power = int(math.ceil(math.log2(count)))
    return qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(power)[:count]


def _ball_case(index: int) -> tuple[np.ndarray, dict]:
    u = _sobol(STAGE_2, 4, BALL_SEED)[index]
    x = BALL_IC_BOUNDS["x"][0] + np.float64(u[0]) * 9.0
    y = BALL_IC_BOUNDS["y"][0] + np.float64(u[1]) * 9.0
    speed = 0.5 + np.float64(u[2]) * 2.5
    angle = np.float64(u[3]) * np.float64(2.0 * math.pi)
    vx = speed * np.cos(angle); vy = speed * np.sin(angle)
    trajectory = ball_teacher.TEACHER.simulate_bouncing_ball(
        x, y, vx, vy, ball_teacher.DT, ball_teacher.TOTAL_STEPS,
        xmin=ball_teacher.XMIN, xmax=ball_teacher.XMAX,
        ymin=ball_teacher.YMIN, ymax=ball_teacher.YMAX,
    )
    spec = dict(x=float(x), y=float(y), speed=float(speed), direction=float(angle),
                vx=float(vx), vy=float(vy), sobol_index=index,
                sobol_unit=u.tolist(), seed=BALL_SEED)
    return np.asarray(trajectory, dtype=np.float64), spec


def _kepler_case(index: int) -> tuple[np.ndarray, dict]:
    u = _sobol(STAGE_2, 4, KEPLER_SEED)[index]
    eccentricity = 0.18 + np.float64(u[0]) * 0.50
    periapsis = 0.72 + np.float64(u[1]) * 0.80
    orientation = np.float64(u[2]) * np.float64(2.0 * math.pi)
    # Map the fourth Sobol coordinate uniformly to every integer in [13, 211].
    phase_steps = 13 + min(198, int(math.floor(float(u[3]) * 199.0)))
    initial = kepler_teacher.initial_state(eccentricity, periapsis, orientation)
    phased = kepler_teacher.numpy_rollout(initial, steps=phase_steps)[-1]
    trajectory = kepler_teacher.numpy_rollout(phased, steps=kepler_teacher.TOTAL_STEPS)
    spec = dict(eccentricity=float(eccentricity), periapsis=float(periapsis),
                orientation=float(orientation), phase_steps=phase_steps,
                sobol_index=index, sobol_unit=u.tolist(), seed=KEPLER_SEED)
    return trajectory, spec


def _objectives(system: str, trajectory: np.ndarray):
    if system == "reflection_ball":
        return ball.objectives(trajectory)
    inputs = trajectory[:kepler_teacher.PREFIX_STATES - 1]
    targets = trajectory[1:kepler_teacher.PREFIX_STATES]
    torch_inputs = torch.from_numpy(inputs.copy())
    torch_targets = torch.from_numpy(targets.copy())

    def numpy_objective(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            residual = known.step(inputs, np.asarray(theta, dtype=np.float64), np) - targets
            value = float(np.mean(residual * residual))
        return value if math.isfinite(value) else 1.0e100

    def torch_objective(theta: torch.Tensor) -> torch.Tensor:
        return (known.step(torch_inputs, theta, torch) - torch_targets).square().mean()

    return numpy_objective, torch_objective


def _config(system: str):
    if system == "reflection_ball":
        return ball.LOWER.copy(), ball.UPPER.copy(), ball.TEACHER.copy()
    if system == "kepler5":
        return known.LOW.copy(), known.HIGH.copy(), known.GT.copy()
    raise ValueError(system)


class N1Runner:
    """Exact current continuous-E-DIRECT state machine with resumable cap state."""

    def __init__(self, objective, lower, upper, case_id):
        self.objective = objective
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.width = self.upper - self.lower
        self.dimension = len(self.lower)
        self.case_id = int(case_id)
        self.api = edirect.composition(self.dimension)
        self.traces: list[dict] = []
        self.api["_greedy_proposal"] = (
            lambda p, f, c, it: continuous.proposal(p, f, c, it, self.traces)
        )
        self.points: list[np.ndarray] = []
        self.losses: list[float] = []
        self.events: list[dict] = []
        self.iteration = 0
        self.phase = "initial"
        self.pending = np.empty((0, self.dimension), dtype=np.float64)
        self.next_pending = 0
        self.stop_reason: str | None = None

    def evaluate(self, unit, source):
        unit = np.asarray(unit, dtype=np.float64)
        value = float(self.objective(self.lower + self.width * unit))
        if not math.isfinite(value):
            raise RuntimeError("nonfinite physical objective")
        self.points.append(unit.copy()); self.losses.append(value)
        best = int(np.argmin(self.losses))
        self.events.append(dict(
            evaluation=len(self.losses), iteration=self.iteration, source=source,
            point=unit.tolist(), theta=(self.lower + self.width * unit).tolist(),
            loss=value, best_index=best, best_loss=float(self.losses[best]),
        ))

    def run_to(self, cap: int):
        self.stop_reason = None
        if self.phase == "initial":
            design = edirect.initial_design(self.dimension)
            while self.next_pending < len(design) and len(self.losses) < cap:
                self.evaluate(design[self.next_pending], "initial")
                self.next_pending += 1
                if min(self.losses) <= N1_GATE:
                    self.stop_reason = "loss_gate"
                    return
            if self.next_pending == len(design):
                self.phase = "next_round"; self.next_pending = 0
        while len(self.losses) < cap:
            before = len(self.losses)
            if self.phase == "next_round":
                self.pending = self.api["_cell_proposals"](
                    np.asarray(self.points), np.asarray(self.losses),
                    self.case_id, self.iteration,
                )
                self.next_pending = 0; self.phase = "cell"
            while self.phase == "cell" and self.next_pending < len(self.pending):
                if len(self.losses) >= cap:
                    self.stop_reason = "resource_cap"
                    return
                self.evaluate(self.pending[self.next_pending], "cell")
                self.next_pending += 1
                if min(self.losses) <= N1_GATE:
                    self.stop_reason = "loss_gate"
                    return
            if self.phase == "cell":
                self.phase = "greedy"
            if self.phase == "greedy":
                if len(self.losses) >= cap:
                    self.stop_reason = "resource_cap"
                    return
                unit = self.api["_greedy_proposal"](
                    np.asarray(self.points), np.asarray(self.losses),
                    self.case_id, self.iteration,
                )
                if unit is not None:
                    self.evaluate(unit, "greedy")
                    if min(self.losses) <= N1_GATE:
                        self.stop_reason = "loss_gate"
                        return
                self.iteration += 1; self.phase = "next_round"
            if len(self.losses) == before:
                self.stop_reason = "no_progress"
                return
        self.stop_reason = "resource_cap"

    def endpoint(self) -> dict:
        values = np.asarray(self.losses, dtype=np.float64)
        points = np.asarray(self.points, dtype=np.float64)
        best = int(np.argmin(values))
        theta = self.lower + self.width * points[best]
        radius = self.width / len(values) ** (1.0 / self.dimension)
        return dict(theta=theta, loss=float(values[best]), best_index=best,
                    evaluations=len(values), iterations=self.iteration,
                    lower=np.maximum(self.lower, theta - radius),
                    upper=np.minimum(self.upper, theta + radius),
                    stop_reason=self.stop_reason,
                    cell_evaluations=sum(e["source"] == "cell" for e in self.events),
                    continuous_evaluations=sum(e["source"] == "greedy" for e in self.events),
                    continuous_calls=len(self.traces),
                    duplicate_rejections=sum(t["exact_duplicate_rejected"] for t in self.traces))


def _n3_result(theta, loss, run_n3, label):
    return staged._run_n3(np.asarray(theta), float(loss), label, run_n3)


def _first_correctable_incumbent(events: list[dict], lower, upper, run_n3) -> dict | None:
    """Established diagnostic: test only strict incumbent improvements."""
    best_loss = math.inf
    for event in events:
        if float(event["best_loss"]) >= best_loss:
            continue
        best_loss = float(event["best_loss"])
        theta = np.asarray(event["theta"] if event["loss"] == event["best_loss"] else
                           events[int(event["best_index"])]["theta"], dtype=np.float64)
        result = _n3_result(theta, best_loss, run_n3, "n1_correctability_diagnostic")
        if result["recovered"]:
            return dict(evaluation=int(event["evaluation"]), theta=theta.tolist(),
                        loss=best_loss, n3_accepted_steps=result["accepted_steps"],
                        n3_stop_reason=result["stop_reason"])
    return None


def _rollout(system: str, trajectory: np.ndarray, theta: np.ndarray) -> dict:
    start = (ball_teacher.PREFIX_STATES - 1 if system == "reflection_ball"
             else kepler_teacher.PREFIX_STATES - 1)
    truth = trajectory[start:]
    predicted = np.empty_like(truth); predicted[0] = truth[0]
    topology_regressions = 0
    for index in range(len(truth) - 1):
        if system == "reflection_ball":
            predicted[index + 1] = ball.numpy_step(predicted[index], theta)
            topology_regressions += int(
                ball.event_signature(predicted[index], theta) !=
                ball.event_signature(truth[index], ball.TEACHER)
            )
        else:
            predicted[index + 1] = known.step(predicted[index:index + 1], theta, np)[0]
    error = predicted - truth
    maximum = float(np.max(np.abs(error)))
    mse = float(np.mean(error * error))
    return dict(continuation_max_error=(maximum if math.isfinite(maximum) else None),
                continuation_mse=(mse if math.isfinite(mse) else None),
                collision_topology_regressions=int(topology_regressions),
                rollout_finite=bool(np.isfinite(predicted).all()))


def _finish(system: str, trajectory: np.ndarray, objective, torch_objective,
            endpoint: dict, teacher: np.ndarray, label: str) -> dict:
    _, _, run_n3 = staged._make_n3(torch_objective, *(_config(system)[:2]))
    n2 = staged._run_n2(objective, endpoint["theta"], endpoint["loss"],
                        endpoint["lower"], endpoint["upper"])
    n3 = _n3_result(n2["theta"], n2["loss"], run_n3, label)
    final = np.asarray(n3["final_theta"], dtype=np.float64)
    difference = final - teacher
    rollout = _rollout(system, trajectory, final)
    return dict(
        n2_box_lower=endpoint["lower"].tolist(), n2_box_upper=endpoint["upper"].tolist(),
        n2_evaluations=int(n2["evaluations"]), n2_endpoint=n2["theta"].tolist(),
        n2_loss=float(n2["loss"]), n2_stop_reason=n2["message"],
        n2_historical_gate=bool(n2["gate"]),
        n3_accepted_steps=int(n3["accepted_steps"]),
        n3_objective_evaluations=int(n3["objective_evaluations"]),
        n3_stop_reason=n3["stop_reason"], final_theta=final.tolist(),
        final_loss=float(n3["final_loss"]),
        final_entrywise_parameter_error=difference.tolist(),
        final_max_parameter_error=float(np.max(np.abs(difference))),
        final_parameter_l2_error=float(np.linalg.norm(difference)),
        recovered=bool(n3["recovered"]), n3_recovered=bool(n3["recovered"]),
        full_pipeline_pass=bool(n3["recovered"] and rollout["rollout_finite"]),
        **rollout,
    )


def _classification(endpoint: dict, finish: dict) -> str:
    if finish.get("n3_recovered", False) and not finish.get("rollout_finite", True):
        return "FAIL_NUMERICAL_NONFINITE_ROLLOUT"
    if finish["recovered"]:
        return "PASS"
    if endpoint["stop_reason"] == "resource_cap":
        return "FAIL_SEARCH_INCOMPLETE_RESOURCE_CAP"
    if endpoint["stop_reason"] == "no_progress":
        return "FAIL_SEARCH_INCOMPLETE_NO_PROGRESS"
    if finish["n2_stop_reason"] == "Maximum number of function evaluations has been exceeded.":
        return "FAIL_N2_CAP_OR_STAGNATION"
    return "FAIL_N3_RECOVERY"


def run_case(system: str, index: int) -> dict:
    torch.set_num_threads(1)
    case_dir = CASES / system / f"case_{index:03d}"
    result_path = case_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        production = existing["production"]
        production["recovered"] = bool(production.get("n3_recovered",
                                                       production["recovered"]))
        production["full_pipeline_pass"] = bool(
            production["recovered"] and production.get("rollout_finite", True)
        )
        if production["recovered"] and not production.get("rollout_finite", True):
            production["classification"] = "FAIL_NUMERICAL_NONFINITE_ROLLOUT"
        _atomic_json(result_path, existing)
        return existing
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    trajectory, initial_condition = (_ball_case(index) if system == "reflection_ball"
                                     else _kepler_case(index))
    lower, upper, teacher = _config(system)
    objective, torch_objective = _objectives(system, trajectory)
    _, _, run_n3 = staged._make_n3(torch_objective, lower, upper)
    case_id = (6_100_000 if system == "reflection_ball" else 6_200_000) + index
    runner = N1Runner(objective, lower, upper, case_id)
    runner.run_to(PRODUCTION_CAP)
    production_endpoint = runner.endpoint()
    production_finish = _finish(system, trajectory, objective, torch_objective,
                                production_endpoint, teacher, "production_n2_endpoint")
    first_correctable = _first_correctable_incumbent(runner.events, lower, upper, run_n3)
    production = dict(
        n1_evaluations=production_endpoint["evaluations"],
        n1_iterations=production_endpoint["iterations"],
        n1_endpoint=production_endpoint["theta"].tolist(),
        n1_loss=production_endpoint["loss"],
        n1_stop_reason=production_endpoint["stop_reason"],
        n1_cell_evaluations=production_endpoint["cell_evaluations"],
        n1_continuous_evaluations=production_endpoint["continuous_evaluations"],
        n1_continuous_calls=production_endpoint["continuous_calls"],
        n1_duplicate_rejections=production_endpoint["duplicate_rejections"],
        first_n3_correctable_incumbent=first_correctable,
        **production_finish,
    )
    production["classification"] = _classification(production_endpoint, production_finish)
    diagnostic = None
    if (not production_finish["recovered"] and
            production_endpoint["stop_reason"] in ("resource_cap", "no_progress")):
        if production_endpoint["stop_reason"] == "resource_cap":
            runner.run_to(DIAGNOSTIC_CAP)
            diagnostic_endpoint = runner.endpoint()
            diagnostic_finish = _finish(
                system, trajectory, objective, torch_objective, diagnostic_endpoint,
                teacher, "diagnostic_continuation_n2_endpoint",
            )
            diagnostic = dict(
                resource_allowance=DIAGNOSTIC_CAP,
                n1_evaluations=diagnostic_endpoint["evaluations"],
                n1_iterations=diagnostic_endpoint["iterations"],
                n1_endpoint=diagnostic_endpoint["theta"].tolist(),
                n1_loss=diagnostic_endpoint["loss"],
                n1_stop_reason=diagnostic_endpoint["stop_reason"],
                **diagnostic_finish,
            )
        else:
            diagnostic = dict(resource_allowance=DIAGNOSTIC_CAP,
                              not_continued_reason="no_progress is terminal under unchanged search")
    np.savez_compressed(case_dir / "n1_state.npz",
                        points=np.asarray(runner.points), losses=np.asarray(runner.losses))
    _atomic_json(case_dir / "events.json", runner.events)
    _atomic_json(case_dir / "continuous.json", runner.traces)
    row = dict(system=system, case_index=index, initial_condition=initial_condition,
               trajectory_sha256=_array_digest(trajectory), hypothesis_bounds=[lower.tolist(), upper.tolist()],
               teacher=teacher.tolist(), production=production,
               diagnostic_continuation=diagnostic,
               elapsed_seconds=float(time.monotonic() - started))
    _atomic_json(result_path, row)
    return row


def _run_tasks(tasks: list[tuple[str, int]], workers: int) -> None:
    """Use independent interpreters; forking a Torch-loaded parent is unsafe."""
    workers = max(1, workers)
    for offset in range(0, len(tasks), workers):
        batch = tasks[offset:offset + workers]
        children = []
        for system, index in batch:
            command = [sys.executable, "-m", __spec__.name, "--case", system, str(index)]
            children.append(((system, index), subprocess.Popen(command, env=os.environ.copy())))
        failures = [(task, child.wait()) for task, child in children]
        failures = [(task, code) for task, code in failures if code]
        if failures:
            raise RuntimeError(f"case subprocess failures: {failures}")


def _distribution(values) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return dict(count=0)
    return dict(count=len(data), minimum=float(np.min(data)), median=float(np.median(data)),
                p90=float(np.quantile(data, 0.9, method="higher")),
                maximum=float(np.max(data)), total=float(np.sum(data)))


def _system_summary(system: str, rows: list[dict]) -> dict:
    prod = [row["production"] for row in rows]
    recovered = [row for row in rows if row["production"]["recovered"]]
    diagnostics = [row["diagnostic_continuation"] for row in rows
                   if row["diagnostic_continuation"] is not None]
    recovered_diag = [row for row in diagnostics if row.get("recovered", False)]
    return dict(
        new_cases=len(rows), production_recoveries=len(recovered),
        production_failures=len(rows) - len(recovered),
        full_pipeline_recoveries=sum(r.get("full_pipeline_pass", r["recovered"])
                                     for r in prod),
        genuine_pipeline_failures=sum(not r.get("full_pipeline_pass", r["recovered"])
                                      for r in prod),
        n1_evaluations=_distribution([r["n1_evaluations"] for r in prod]),
        n2_evaluations=_distribution([r["n2_evaluations"] for r in prod]),
        n3_accepted_steps=_distribution([r["n3_accepted_steps"] for r in prod]),
        n1_resource_cap=sum(r["n1_stop_reason"] == "resource_cap" for r in prod),
        n1_no_progress=sum(r["n1_stop_reason"] == "no_progress" for r in prod),
        n1_loss_gate=sum(r["n1_stop_reason"] == "loss_gate" for r in prod),
        classifications={name: sum(r["classification"] == name for r in prod)
                         for name in sorted({r["classification"] for r in prod})},
        diagnostic_continuations=len(diagnostics),
        recovered_only_after_diagnostic_continuation=len(recovered_diag),
        worst_final_parameter_l2_error=(max(r["final_parameter_l2_error"] for r in prod)
                                        if prod else None),
        worst_final_max_parameter_error=(max(r["final_max_parameter_error"] for r in prod)
                                         if prod else None),
        worst_continuation_max_error=(max((r["continuation_max_error"] for r in prod
                                           if r["continuation_max_error"] is not None),
                                          default=None)),
        worst_continuation_mse=(max((r["continuation_mse"] for r in prod
                                     if r["continuation_mse"] is not None), default=None)),
        collision_topology_regressions=sum(r["collision_topology_regressions"] for r in prod),
        cases=[row["case_index"] for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, metavar=("SYSTEM", "INDEX"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stage1-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CASES.mkdir(parents=True, exist_ok=True)
    if args.case:
        row = run_case(args.case[0], int(args.case[1]))
        print(json.dumps(dict(system=row["system"], case=row["case_index"],
                              n1=row["production"]["n1_evaluations"],
                              n1_stop=row["production"]["n1_stop_reason"],
                              recovered=row["production"]["recovered"],
                              classification=row["production"]["classification"],
                              seconds=row["elapsed_seconds"]), indent=2), flush=True)
        return
    before = hashes(); _atomic_json(OUT / "checkpoint_hashes_before.json", before)
    started = time.monotonic()
    tasks = [(system, index) for system in ("reflection_ball", "kepler5")
             for index in range(STAGE_1)]
    _run_tasks(tasks, args.workers)
    stage1_rows = {system: [json.loads((CASES / system / f"case_{i:03d}" / "result.json").read_text())
                            for i in range(STAGE_1)]
                   for system in ("reflection_ball", "kepler5")}
    stage1_pass = all(row["production"].get("full_pipeline_pass",
                                             row["production"]["recovered"])
                      for rows in stage1_rows.values() for row in rows)
    extended = False
    if stage1_pass and not args.stage1_only:
        tasks = [(system, index) for system in ("reflection_ball", "kepler5")
                 for index in range(STAGE_1, STAGE_2)]
        _run_tasks(tasks, args.workers)
        extended = True
    count = STAGE_2 if extended else STAGE_1
    rows = {system: [json.loads((CASES / system / f"case_{i:03d}" / "result.json").read_text())
                     for i in range(count)]
            for system in ("reflection_ball", "kepler5")}
    after = hashes(); _atomic_json(OUT / "checkpoint_hashes_after.json", after)
    report = dict(
        status=("COMPLETE_STAGE_2" if extended else "COMPLETE_STAGE_1_STOPPED_ON_FAILURE"),
        pipeline="continuous E-DIRECT N1 -> production count-derived N2 box -> unchanged N2 -> unchanged N3",
        n1_gate=N1_GATE, production_cap=PRODUCTION_CAP,
        diagnostic_continuation_cap=DIAGNOSTIC_CAP,
        initial_condition_sampling=dict(method="scrambled Sobol low discrepancy",
                                        ball_bounds=BALL_IC_BOUNDS,
                                        kepler_bounds=KEPLER_IC_BOUNDS,
                                        ball_seed=BALL_SEED, kepler_seed=KEPLER_SEED),
        stage1_all_recovered=stage1_pass, extended_to_stage2=extended,
        systems={name: _system_summary(name, value) for name, value in rows.items()},
        checkpoint_hashes_unchanged=before == after,
        checkpoint_count=len(before), elapsed_seconds=float(time.monotonic() - started),
    )
    _atomic_json(OUT / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
