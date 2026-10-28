"""Cross-system closed-loop regression for the recovered scalar exponential.

Only ``GaussianLocalResponse._weights`` is substituted.  Production artifacts
are read as the A path; every B-path case is rerun and persisted independently.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

from experiments.sho import run_stage1b_edirect_vs_direct as teacher_module
from experiments.sho.deterministic_full_k_framework import hashes
from experiments.sho.validate_n1_exp_substitution_closed_loop import (
    CorrectlyRoundedGaussianLocalResponse,
    assembled_exp_compiled,
    earliest_correctable as sho_earliest_correctable,
    finish as sho_finish,
    run_search as sho_run_search,
)
from experiments.sho.recover_n1_scalar_exp import assembled_exp


OUT = Path("outputs/sho/stage1B/n1_scalar_exp_cross_system_closed_loop")
CASES = OUT / "cases"
PRODUCTION_SHO = Path("outputs/sho/stage1B/checkpointed_resource_edirect/cases")
PRODUCTION_BALL4 = Path("outputs/bouncing_ball/full_k_50_transfer/cases")
PRODUCTION_BALL5 = Path("outputs/bouncing_ball/reflection_peelback/cases")
PRODUCTION_KEPLER = Path("outputs/kepler/five_parameter_n1_diagnosis")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def install_exp() -> None:
    teacher_module.GaussianLocalResponse = CorrectlyRoundedGaussianLocalResponse


def verify_compiled_graph() -> None:
    control = np.linspace(-9.0, np.nextafter(0.0, -1.0), 1_000_001,
                          dtype=np.float64)
    got = assembled_exp_compiled(control)
    expected = assembled_exp(control)
    if not np.array_equal(got.view(np.uint64), expected.view(np.uint64)):
        raise RuntimeError("compiled exp differs from the recovered assembly")


def _production_metrics(row: dict, system: str) -> dict:
    if system == "sho4":
        return dict(
            recovered=bool(row["final_loss"] < 1.0e-20),
            n1_evaluations=int(row["total_n1_evaluations"]),
            n2_evaluations=int(row["n2_evaluations"]),
            n3_accepted_steps=int(row["n3_accepted_steps"]),
            final_parameter_error=float(row["matrix_error"]),
            final_loss=float(row["final_loss"]),
            rollout_error=float(row["rollout_max_error"]),
        )
    error_key = ("final_frobenius_error" if system == "ball4"
                 else "final_parameter_l2_error")
    return dict(
        recovered=bool(row["recovered"]),
        n1_evaluations=int(row["n1_evaluations"]),
        n2_evaluations=int(row["n2_evaluations"]),
        n3_accepted_steps=int(row["n3_accepted_steps"]),
        final_parameter_error=float(row[error_key]),
        final_loss=float(row["final_loss"]),
        rollout_error=float(row["continuation_max_error"]),
    )


def _first_correctable_ball(events: list[dict], lower: np.ndarray,
                            upper: np.ndarray, numpy_objective, run_n3) -> dict | None:
    """Label incumbent improvements using the unchanged production N3."""
    from experiments.kepler import run_four_parameter_peelback as staged
    best_loss = math.inf
    for event in events:
        loss = float(event["loss"])
        if loss >= best_loss:
            continue
        best_loss = loss
        point = np.asarray(event["point"], dtype=np.float64)
        theta = lower + (upper - lower) * point
        result = staged._run_n3(theta, loss, "n1_incumbent", run_n3)
        if result["recovered"]:
            return dict(evaluation=int(event["evaluation"]), loss=loss,
                        theta=theta.tolist(), accepted_steps=result["accepted_steps"])
    return None


def run_sho(index: int) -> dict:
    from experiments.sho import validate_n1_exp_substitution_closed_loop as validation
    from experiments.sho.generate_checkpointed_edirect_teacher_data import (
        DATA, _case_objectives,
    )
    from experiments.sho.train_newton_correctability_router import _case_math

    with np.load(DATA) as data:
        name = str(data["names"][index])
        category = str(data["categories"][index])
        teacher = data["teachers"][index].ravel()
        states = data[f"prefix_{index}"].copy()
    production_dir = PRODUCTION_SHO / name
    production = json.loads((production_dir / "result.json").read_text())
    production_events = json.loads((production_dir / "events.json").read_text())
    production_first = sho_earliest_correctable(production_events, states, category)

    validation.CASE_INDEX = index
    search = sho_run_search(_case_objectives(states)[0], teacher)
    candidate_first = sho_earliest_correctable(search["events"], states, category)
    objective, torch_objective = _case_objectives(states)
    candidate = sho_finish(search, teacher, states, objective, torch_objective)
    candidate_metrics = dict(
        recovered=bool(candidate["recovered"]),
        n1_evaluations=int(candidate["n1_evaluations"]),
        n2_evaluations=int(candidate["n2_evaluations"]),
        n3_accepted_steps=int(candidate["n3_accepted_steps"]),
        final_parameter_error=float(candidate["matrix_error"]),
        final_loss=float(candidate["final_loss"]),
        rollout_error=float(candidate["rollout_max_error"]),
    )
    return dict(
        system="sho4", case=name, case_index=index, category=category,
        production=dict(**_production_metrics(production, "sho4"),
                        first_newton_correctable=production_first),
        recovered_exp=dict(**candidate_metrics,
                           first_newton_correctable=candidate_first),
        candidate_events=search["events"], candidate_traces=search["traces"],
    )


def run_ball4(index: int) -> dict:
    from experiments.bouncing_ball import run_full_k_transfer as module
    from experiments.bouncing_ball import run_first_transfer as ball
    from experiments.kepler import run_four_parameter_peelback as staged
    case_root = CASES / "ball4"
    module.CASES = case_root
    trajectories, _, _, _ = ball.TEACHER.build_bouncing_ball_trajectories(
        n_trajectories=module.COUNT, n_timesteps=ball.TOTAL_STEPS,
        dt=ball.DT, seed=module.SEED)
    candidate = module.run_case(index, trajectories[index])
    production_path = PRODUCTION_BALL4 / f"trajectory_{index:02d}"
    production = json.loads((production_path / "result.json").read_text())
    numpy_objective, torch_objective = module.objectives(trajectories[index])
    _, _, run_n3 = staged._make_n3(torch_objective, module.LOWER, module.UPPER)
    production_events = json.loads((production_path / "events.json").read_text())
    candidate_events = json.loads((case_root / f"trajectory_{index:02d}" / "events.json").read_text())
    prod_first = _first_correctable_ball(production_events, module.LOWER, module.UPPER,
                                         numpy_objective, run_n3)
    cand_first = _first_correctable_ball(candidate_events, module.LOWER, module.UPPER,
                                         numpy_objective, run_n3)
    return dict(
        system="ball4", case=f"trajectory_{index:02d}", case_index=index,
        production=dict(**_production_metrics(production, "ball4"),
                        first_newton_correctable=prod_first),
        recovered_exp=dict(**_production_metrics(candidate, "ball4"),
                           first_newton_correctable=cand_first),
    )


def run_ball5(index: int) -> dict:
    from experiments.bouncing_ball import run_reflection_peelback as module
    case_root = CASES / "ball5"
    module.CASES = case_root
    trajectories = module._trajectories()
    candidate = module.run_case(index, trajectories[module.SELECTED[index]])
    production_path = PRODUCTION_BALL5 / f"case_{index:02d}"
    production = json.loads((production_path / "result.json").read_text())
    numpy_objective, torch_objective = module.objectives(trajectories[module.SELECTED[index]])
    _, _, run_n3 = module.staged._make_n3(torch_objective, module.LOWER, module.UPPER)
    production_events = json.loads((production_path / "events.json").read_text())
    candidate_events = json.loads((case_root / f"case_{index:02d}" / "events.json").read_text())
    prod_first = _first_correctable_ball(production_events, module.LOWER, module.UPPER,
                                         numpy_objective, run_n3)
    cand_first = _first_correctable_ball(candidate_events, module.LOWER, module.UPPER,
                                         numpy_objective, run_n3)
    return dict(
        system="ball5", case=f"case_{index:02d}", case_index=index,
        production=dict(**_production_metrics(production, "ball5"),
                        first_newton_correctable=prod_first),
        recovered_exp=dict(**_production_metrics(candidate, "ball5"),
                           first_newton_correctable=cand_first),
    )


def _kepler_finish(points: np.ndarray, losses: np.ndarray, events: list[dict],
                   trajectory: np.ndarray, objective, torch_objective) -> dict:
    from experiments.audit_known_structure import GT, HIGH, LOW, step
    from experiments.kepler import run_four_parameter_peelback as staged
    count = len(losses)
    best = int(np.argmin(losses))
    theta = LOW + (HIGH - LOW) * points[best]
    radius = (HIGH - LOW) / count ** (1.0 / len(LOW))
    box_lower = np.maximum(LOW, theta - radius)
    box_upper = np.minimum(HIGH, theta + radius)
    n2 = staged._run_n2(objective, theta, float(losses[best]), box_lower, box_upper)
    _, _, run_n3 = staged._make_n3(torch_objective, LOW, HIGH)
    n3 = staged._run_n3(n2["theta"], n2["loss"], "kepler5_exp_regression", run_n3)
    final = np.asarray(n3["final_theta"], dtype=np.float64)
    predicted = np.empty_like(trajectory)
    predicted[0] = trajectory[0]
    for index in range(len(trajectory) - 1):
        predicted[index + 1] = step(predicted[index:index + 1], final, np)[0]
    error = predicted - trajectory
    return dict(
        recovered=bool(n3["recovered"]), n1_evaluations=count,
        n2_evaluations=int(n2["evaluations"]),
        n3_accepted_steps=int(n3["accepted_steps"]),
        final_parameter_error=float(np.linalg.norm(final - GT)),
        final_loss=float(n3["final_loss"]),
        rollout_error=float(np.max(np.abs(error))),
        final_theta=final.tolist(), n1_theta=theta.tolist(),
        n1_loss=float(losses[best]), n2_theta=np.asarray(n2["theta"]).tolist(),
        n2_loss=float(n2["loss"]),
    )


def _kepler_first_correctable(events: list[dict], torch_objective) -> dict | None:
    from experiments.audit_known_structure import HIGH, LOW
    from experiments.kepler import run_four_parameter_peelback as staged
    _, _, run_n3 = staged._make_n3(torch_objective, LOW, HIGH)
    best_loss = math.inf
    for event in events:
        loss = float(event["loss"])
        if loss >= best_loss:
            continue
        best_loss = loss
        if "theta" in event:
            theta = np.asarray(event["theta"], dtype=np.float64)
        else:
            theta = LOW + (HIGH - LOW) * np.asarray(event["point"], dtype=np.float64)
        result = staged._run_n3(theta, loss, "kepler5_n1_incumbent", run_n3)
        if result["recovered"]:
            return dict(evaluation=int(event["evaluation"]), loss=loss,
                        theta=theta.tolist(), accepted_steps=result["accepted_steps"])
    return None


def _run_kepler_search(objective, torch_objective) -> dict:
    from experiments.audit_known_structure import HIGH, LOW
    from experiments.kepler import run_four_parameter_peelback as staged
    from experiments.sho import continuous_surrogate_edirect as continuous
    from experiments.sho import dimension_general_edirect as edirect
    width = HIGH - LOW
    api = edirect.composition(5)
    traces: list[dict] = []
    api["_greedy_proposal"] = lambda p, f, c, it: continuous.proposal(p, f, c, it, traces)
    points: list[np.ndarray] = []
    losses: list[float] = []
    events: list[dict] = []
    iteration = 0

    def evaluate(unit, source):
        theta = LOW + width * np.asarray(unit, dtype=np.float64)
        value = float(objective(theta))
        points.append(np.asarray(unit, dtype=np.float64).copy())
        losses.append(value)
        events.append(dict(evaluation=len(losses), iteration=iteration, source=source,
                           point=np.asarray(unit).tolist(), theta=theta.tolist(), loss=value))

    for unit in edirect.initial_design(5):
        evaluate(unit, "initial")
    targets = (1_000, 1_500, 2_000, 3_000, 5_000)
    target_index = 0
    _, _, run_n3 = staged._make_n3(torch_objective, LOW, HIGH)
    selected_count = None
    while len(losses) < 5_000 and selected_count is None:
        proposals = api["_cell_proposals"](np.asarray(points), np.asarray(losses),
                                            91_000, iteration)
        for unit in proposals:
            if len(losses) >= 5_000:
                break
            evaluate(unit, "cell")
        if len(losses) < 5_000:
            unit = api["_greedy_proposal"](np.asarray(points), np.asarray(losses),
                                             91_000, iteration)
            if unit is not None:
                evaluate(unit, "continuous_surrogate")
        iteration += 1
        while target_index < len(targets) and len(losses) >= targets[target_index]:
            target = targets[target_index]
            prefix_losses = np.asarray(losses[:target])
            best = int(np.argmin(prefix_losses))
            theta = LOW + width * np.asarray(points[best])
            n3 = staged._run_n3(theta, float(prefix_losses[best]),
                                f"kepler5_n1_{target}", run_n3)
            if n3["recovered"]:
                selected_count = target
                break
            target_index += 1
    if selected_count is None:
        selected_count = min(5_000, len(losses))
    return dict(points=np.asarray(points[:selected_count]),
                losses=np.asarray(losses[:selected_count]),
                events=events[:selected_count], traces=traces, iterations=iteration)


def run_kepler() -> dict:
    from experiments.kepler.diagnose_five_parameter_n1 import objective_factory
    trajectory, objective, torch_objective = objective_factory()
    production_state = np.load(PRODUCTION_KEPLER / "trace.npz")
    production_events = json.loads((PRODUCTION_KEPLER / "events.json").read_text())
    # The established trajectory first passed the unchanged N3 criterion at the
    # 1,500 checkpoint; use precisely that saved prefix.
    production_points = production_state["points"][:1500]
    production_losses = production_state["losses"][:1500]
    production = _kepler_finish(production_points, production_losses,
                                 production_events[:1500], trajectory,
                                 objective, torch_objective)
    production_first = _kepler_first_correctable(production_events[:1500], torch_objective)
    search = _run_kepler_search(objective, torch_objective)
    candidate = _kepler_finish(search["points"], search["losses"], search["events"],
                               trajectory, objective, torch_objective)
    candidate_first = _kepler_first_correctable(search["events"], torch_objective)
    return dict(
        system="kepler5", case="five_parameter_case_00", case_index=0,
        production=dict(**production, first_newton_correctable=production_first),
        recovered_exp=dict(**candidate, first_newton_correctable=candidate_first),
    )


def run_one(system: str, index: int) -> dict:
    torch.set_num_threads(1)
    install_exp()
    before = hashes()
    started = time.monotonic()
    if system == "sho4":
        result = run_sho(index)
    elif system == "ball4":
        result = run_ball4(index)
    elif system == "ball5":
        result = run_ball5(index)
    elif system == "kepler5":
        if index != 0:
            raise ValueError("Kepler corpus has one established five-parameter case")
        result = run_kepler()
    else:
        raise ValueError(system)
    after = hashes()
    result["comparison"] = dict(
        final_recovery_preserved=(result["production"]["recovered"] ==
                                  result["recovered_exp"]["recovered"]),
        n1_evaluation_delta=(result["recovered_exp"]["n1_evaluations"] -
                             result["production"]["n1_evaluations"]),
        n2_evaluation_delta=(result["recovered_exp"]["n2_evaluations"] -
                             result["production"]["n2_evaluations"]),
        parameter_error_delta=(result["recovered_exp"]["final_parameter_error"] -
                               result["production"]["final_parameter_error"]),
        loss_delta=(result["recovered_exp"]["final_loss"] -
                    result["production"]["final_loss"]),
        rollout_error_delta=(result["recovered_exp"]["rollout_error"] -
                             result["production"]["rollout_error"]),
    )
    result["checkpoint_hashes_unchanged"] = before == after
    result["elapsed_seconds"] = time.monotonic() - started
    result.pop("candidate_events", None)
    result.pop("candidate_traces", None)
    path = CASES / system / f"{index:03d}.json"
    atomic_json(path, result)
    if result["production"]["recovered"] and not result["recovered_exp"]["recovered"]:
        raise RuntimeError(f"final recovery regression: {system}:{index}")
    if not result["checkpoint_hashes_unchanged"]:
        raise RuntimeError(f"checkpoint hash changed: {system}:{index}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("sho4", "ball4", "ball5", "kepler5"), required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--verify-exp", action="store_true")
    args = parser.parse_args()
    if args.verify_exp:
        verify_compiled_graph()
    result = run_one(args.system, args.index)
    print(json.dumps(dict(system=args.system, index=args.index,
                          production_recovered=result["production"]["recovered"],
                          candidate_recovered=result["recovered_exp"]["recovered"],
                          n1=result["recovered_exp"]["n1_evaluations"],
                          n2=result["recovered_exp"]["n2_evaluations"],
                          elapsed=result["elapsed_seconds"])), flush=True)


if __name__ == "__main__":
    main()
