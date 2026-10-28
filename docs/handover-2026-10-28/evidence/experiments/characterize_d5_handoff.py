#!/usr/bin/env python3
"""Characterize legitimate d=5 handoff states without changing search decisions."""
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
import torch

from experiments.bouncing_ball import run_reflection_peelback as ball
from experiments.kepler import run_four_parameter_peelback as staged
from experiments.sho.deterministic_full_k_framework import hashes
from experiments import validate_d5_production_pipeline as production


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/d5_handoff_characterization")
CASES = OUT / "kepler_cases"
COUNT = 64
CAP = 5_000
CHECKPOINTS = (20,) + tuple(range(100, CAP + 1, 100))


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reflection_identifiability() -> dict:
    rows = []
    for case_id in range(COUNT):
        trajectory, spec = production._ball_case(case_id)
        inputs = torch.from_numpy(trajectory[:ball.ball.PREFIX_STATES - 1].copy())
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
        rho_norm = float(np.linalg.norm(jacobian[:, 4]))
        signatures = [ball.event_signature(state, ball.TEACHER)
                      for state in trajectory[:ball.ball.PREFIX_STATES - 1]]
        impacts = int(sum(len(value) for value in signatures))
        row = dict(
            case_id=case_id, initial_condition=spec,
            prefix_impacts=impacts,
            prefix_corner_steps=int(sum(any(mask == 3 for mask in value)
                                        for value in signatures)),
            prefix_multiple_impact_steps=int(sum(len(value) > 1 for value in signatures)),
            jacobian_shape=list(jacobian.shape), jacobian_rank=rank,
            rank_tolerance=float(tolerance), singular_values=singular.tolist(),
            rho_jacobian_column_norm=rho_norm,
            classification=("IDENTIFIABILITY_COMPLETE" if rank == 5 else
                            "IDENTIFIABILITY_INCOMPLETE"),
        )
        rows.append(row)
    result = dict(
        cases=COUNT,
        rank_rule="rank = count(singular_value > max(J.shape)*eps*largest_singular_value)",
        complete=sum(row["classification"] == "IDENTIFIABILITY_COMPLETE" for row in rows),
        incomplete=sum(row["classification"] == "IDENTIFIABILITY_INCOMPLETE" for row in rows),
        zero_impact=sum(row["prefix_impacts"] == 0 for row in rows),
        zero_impact_rank_deficient=sum(row["prefix_impacts"] == 0 and row["jacobian_rank"] < 5
                                       for row in rows),
        impact_positive_rank_five=sum(row["prefix_impacts"] > 0 and row["jacobian_rank"] == 5
                                      for row in rows),
        rows=rows,
    )
    _atomic_json(OUT / "reflection_identifiability.json", result)
    return result


def _handoff(endpoint: dict, objective, torch_objective, teacher: np.ndarray,
             trajectory: np.ndarray, case_id: int) -> dict:
    n2 = staged._run_n2(objective, endpoint["theta"], endpoint["loss"],
                        endpoint["lower"], endpoint["upper"])
    _, _, run_n3 = staged._make_n3(torch_objective, production.known.LOW,
                                    production.known.HIGH)
    n3 = staged._run_n3(n2["theta"], n2["loss"],
                        f"kepler_handoff_{case_id}_{endpoint['evaluations']}", run_n3)
    final = np.asarray(n3["final_theta"], dtype=np.float64)
    box_width = endpoint["upper"] - endpoint["lower"]
    return dict(
        evaluation=int(endpoint["evaluations"]), n1_theta=endpoint["theta"].tolist(),
        n1_loss=float(endpoint["loss"]),
        offline_n1_parameter_distance=float(np.linalg.norm(endpoint["theta"] - teacher)),
        n2_box_lower=endpoint["lower"].tolist(),
        n2_box_upper=endpoint["upper"].tolist(),
        n2_box_width=box_width.tolist(),
        n2_box_normalized_halfwidth=float(endpoint["evaluations"] ** (-1.0 / 5.0)),
        offline_teacher_inside_n2_box=bool(np.all(teacher >= endpoint["lower"]) and
                                           np.all(teacher <= endpoint["upper"])),
        n2_evaluations=int(n2["evaluations"]), n2_endpoint=n2["theta"].tolist(),
        n2_loss=float(n2["loss"]), n2_stop_reason=n2["message"],
        n2_historical_gate=bool(n2["gate"]),
        n3_recovered=bool(n3["recovered"]),
        n3_accepted_steps=int(n3["accepted_steps"]),
        n3_objective_evaluations=int(n3["objective_evaluations"]),
        n3_stop_reason=n3["stop_reason"], n3_final_theta=final.tolist(),
        n3_final_loss=float(n3["final_loss"]),
        offline_final_parameter_distance=float(np.linalg.norm(final - teacher)),
    )


def run_kepler_case(case_id: int) -> dict:
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
    # Suppress only termination.  The same candidate pools, cell decisions,
    # surrogate minimizer, case seed, evaluation order and objective remain.
    old_gate = production.N1_GATE
    production.N1_GATE = -math.inf
    try:
        runner = production.N1Runner(objective, lower, upper, 6_300_000 + case_id)
        checks = []
        first = None
        for target in CHECKPOINTS:
            runner.run_to(target)
            endpoint = runner.endpoint()
            check = _handoff(endpoint, objective, torch_objective, teacher,
                             trajectory, case_id)
            checks.append(check)
            if check["n3_recovered"]:
                first = check
                break
            if endpoint["stop_reason"] == "no_progress":
                break
    finally:
        production.N1_GATE = old_gate
    status = ("SUCCESSFUL_HANDOFF_FOUND" if first is not None else
              ("SEARCH_INCOMPLETE_RESOURCE_CAP" if len(runner.losses) >= CAP else
               "SEARCH_INCOMPLETE_NO_PROGRESS"))
    np.savez_compressed(case_dir / "n1_state.npz",
                        points=np.asarray(runner.points), losses=np.asarray(runner.losses))
    result = dict(
        case_id=case_id, initial_condition=spec, status=status,
        checkpoints_tested=len(checks), first_successful_handoff=first,
        last_checkpoint=checks[-1], actual_n1_evaluations=len(runner.losses),
        n1_iterations=int(runner.iteration), n1_stop_reason=runner.stop_reason,
        n1_cell_evaluations=sum(event["source"] == "cell" for event in runner.events),
        n1_continuous_evaluations=sum(event["source"] == "greedy" for event in runner.events),
        continuous_surrogate_calls=len(runner.traces),
        continuous_duplicate_rejections=sum(trace["exact_duplicate_rejected"]
                                             for trace in runner.traces),
        staged_handoffs=checks, elapsed_seconds=float(time.monotonic() - started),
    )
    _atomic_json(result_path, result)
    return result


def _run_tasks(indices: list[int], workers: int) -> None:
    for offset in range(0, len(indices), workers):
        batch = indices[offset:offset + workers]
        children = []
        for index in batch:
            command = [sys.executable, "-m", __spec__.name,
                       "--kepler-case", str(index)]
            children.append((index, subprocess.Popen(command, env=os.environ.copy())))
        failed = [(index, child.wait()) for index, child in children]
        failed = [(index, code) for index, code in failed if code]
        if failed:
            raise RuntimeError(f"Kepler case failures: {failed}")


def _distribution(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict(count=0)
    return dict(count=len(values), minimum=float(values.min()),
                median=float(np.median(values)),
                p90=float(np.quantile(values, 0.9, method="higher")),
                maximum=float(values.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflection-only", action="store_true")
    parser.add_argument("--kepler-case", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CASES.mkdir(parents=True, exist_ok=True)
    if args.kepler_case is not None:
        result = run_kepler_case(args.kepler_case)
        first = result["first_successful_handoff"]
        print(json.dumps(dict(case=args.kepler_case, status=result["status"],
                              evaluation=None if first is None else first["evaluation"],
                              n1_loss=None if first is None else first["n1_loss"],
                              seconds=result["elapsed_seconds"]), indent=2), flush=True)
        return
    before = hashes(); _atomic_json(OUT / "checkpoint_hashes_before.json", before)
    reflection = reflection_identifiability()
    if args.reflection_only:
        print(json.dumps({key: reflection[key] for key in
                          ("cases", "complete", "incomplete", "zero_impact",
                           "zero_impact_rank_deficient", "impact_positive_rank_five")},
                         indent=2))
        return
    started = time.monotonic()
    _run_tasks(list(range(COUNT)), max(1, args.workers))
    rows = [json.loads((CASES / f"case_{i:03d}" / "result.json").read_text())
            for i in range(COUNT)]
    successful = [row for row in rows if row["first_successful_handoff"] is not None]
    evaluations = [row["first_successful_handoff"]["evaluation"] for row in successful]
    losses = [row["first_successful_handoff"]["n1_loss"] for row in successful]
    after = hashes(); _atomic_json(OUT / "checkpoint_hashes_after.json", after)
    report = dict(
        status=("COMPLETE" if len(successful) == COUNT else "SEARCH_INCOMPLETE_CASES"),
        reflection={key: reflection[key] for key in
                    ("cases", "complete", "incomplete", "zero_impact",
                     "zero_impact_rank_deficient", "impact_positive_rank_five")},
        kepler=dict(cases=COUNT, successful_handoffs=len(successful),
                     search_incomplete=COUNT - len(successful),
                     evaluations_to_first_successful_handoff=_distribution(evaluations),
                     loss_at_first_successful_handoff=_distribution(losses),
                     cases_detail=rows),
        checkpoints=CHECKPOINTS, production_cap=CAP,
        checkpoint_hashes_unchanged=before == after,
        checkpoint_count=len(before), elapsed_seconds=float(time.monotonic() - started),
    )
    _atomic_json(OUT / "report.json", report)
    print(json.dumps({
        "status": report["status"], "reflection": report["reflection"],
        "kepler": {key: report["kepler"][key] for key in
                    ("cases", "successful_handoffs", "search_incomplete",
                     "evaluations_to_first_successful_handoff",
                     "loss_at_first_successful_handoff")},
        "checkpoint_hashes_unchanged": report["checkpoint_hashes_unchanged"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
