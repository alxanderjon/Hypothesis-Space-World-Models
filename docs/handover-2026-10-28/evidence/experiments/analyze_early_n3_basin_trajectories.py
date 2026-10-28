#!/usr/bin/env python3
"""Trace early unchanged-N3 trajectories for ball and Kepler Sobol starts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

from experiments import map_direct_n3_correctability_geometry as basin
from experiments.sho import run_stage1b_n3_gradient_polishing as polish


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUT = Path("outputs/early_n3_basin_trajectory_geometry")
TRACES = OUT / "traces.jsonl"
SOURCE = Path("outputs/direct_n3_correctability_geometry/points.jsonl")
SYSTEMS = ("bouncing_ball", "kepler")
DETAIL_ACCEPTED_STEPS = 3


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _damping_trials(gradient: torch.Tensor, hessian: torch.Tensor,
                     damping: float, status: str) -> int:
    if status == "nonfinite_derivatives":
        return 0
    if status == "no_descent_direction":
        return polish.MAX_DAMPING_TRIALS
    if damping == 0.0:
        return 1
    singular = torch.linalg.svdvals(hessian)
    smallest = float(torch.min(singular)); largest = float(torch.max(singular))
    condition = largest / smallest if smallest > 0.0 else math.inf
    eigenvalues = torch.linalg.eigvalsh(hessian)
    minimum = float(torch.min(eigenvalues))
    scale = max(float(torch.max(torch.abs(eigenvalues))), torch.finfo(torch.float64).tiny)
    direct_allowed = minimum > 0.0 and condition <= polish.CONDITION_LIMIT
    initial = max(scale * polish.DAMPING_RELATIVE_FLOOR,
                  -minimum + scale * polish.DAMPING_RELATIVE_FLOOR)
    base = scale * polish.DAMPING_RELATIVE_FLOOR if direct_allowed else initial
    exponent = int(round(math.log10(damping / base)))
    return exponent + (2 if direct_allowed else 1)


def _iteration_state(theta: torch.Tensor, current_loss: float, config: dict) -> tuple[dict, torch.Tensor, torch.Tensor | None]:
    loss_t, gradient, hessian = config["derivatives"](theta, torch.empty(0))
    width = config["upper"] - config["lower"]
    gradient_np = gradient.numpy(); hessian_np = hessian.numpy()
    scaled_gradient = gradient_np * width
    scaled_hessian = hessian_np * np.outer(width, width)
    eigenvalues = np.linalg.eigvalsh(hessian_np)
    scaled_eigenvalues = np.linalg.eigvalsh(scaled_hessian)
    singular = np.linalg.svd(hessian_np, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else math.inf
    direction, teacher_condition, damping, status = config["direction_fn"](gradient, hessian)
    direction_np = None if direction is None else direction.numpy()
    if direction_np is None:
        step_norm = normalized_l2 = normalized_linf = dot = cosine = predicted = math.nan
        full_inside = False
    else:
        step_norm = float(np.linalg.norm(direction_np))
        normalized = direction_np / width
        normalized_l2 = float(np.linalg.norm(normalized))
        normalized_linf = float(np.max(np.abs(normalized)))
        dot = float(np.dot(gradient_np, direction_np))
        gradient_norm = float(np.linalg.norm(gradient_np))
        cosine = (-dot / (gradient_norm * step_norm)
                  if gradient_norm > 0.0 and step_norm > 0.0 else 0.0)
        predicted = max(0.0, -dot)
        candidate = theta.numpy() + direction_np
        full_inside = bool(np.all(candidate >= config["lower"]) and
                           np.all(candidate <= config["upper"]))
    normalized_margin = np.minimum(theta.numpy() - config["lower"],
                                   config["upper"] - theta.numpy()) / width
    record = dict(
        theta=theta.numpy().tolist(),
        recomputed_loss=float(loss_t),
        current_loss=float(current_loss),
        loss_replay_exact=bool(float(loss_t) == float(current_loss)),
        gradient_norm=float(np.linalg.norm(gradient_np)),
        scaled_gradient_norm=float(np.linalg.norm(scaled_gradient)),
        hessian_eigenvalues=eigenvalues.tolist(),
        scaled_hessian_eigenvalues=scaled_eigenvalues.tolist(),
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=_finite(condition),
        direction_condition=_finite(teacher_condition),
        route=("NONE" if direction is None else
               ("DIRECT" if float(damping) == 0.0 else "DAMPED")),
        damping=_finite(damping),
        damping_trials=_damping_trials(gradient, hessian, float(damping), status),
        direction_status=status,
        newton_direction=(None if direction_np is None else direction_np.tolist()),
        newton_step_norm=_finite(step_norm),
        normalized_step_l2=_finite(normalized_l2),
        normalized_step_linf=_finite(normalized_linf),
        gradient_dot_direction=_finite(dot),
        descent_cosine=_finite(cosine),
        predicted_decrease=_finite(predicted),
        predicted_decrease_over_loss=(None if not math.isfinite(predicted) or current_loss == 0.0
                                      else float(predicted / current_loss)),
        normalized_bound_margins=normalized_margin.tolist(),
        minimum_bound_margin=float(np.min(normalized_margin)),
        mean_bound_margin=float(np.mean(normalized_margin)),
        full_step_in_bounds=full_inside,
    )
    return record, gradient, direction


def _trace(source: dict, config: dict) -> dict:
    theta = torch.tensor(source["theta"], dtype=torch.float64)
    with torch.no_grad():
        current_loss = float(config["objective"](theta))
    initial_loss = current_loss
    theta_history = [theta.numpy().copy()]
    loss_history = [current_loss]
    iterations = []
    terminal_iteration = None
    rejected_newton_steps = actual_loss_evaluations = 0
    backtracking_rejections = accepted_with_backtracking = 0
    hessian_evaluations = damping_steps = 0
    stop_reason = "max_accepted_steps"

    while len(loss_history) - 1 < polish.MAX_ACCEPTED_STEPS:
        iteration_index = len(loss_history) - 1
        state, gradient, direction = _iteration_state(theta, current_loss, config)
        hessian_evaluations += 1
        gradient_norm = state["gradient_norm"]
        detailed = iteration_index < DETAIL_ACCEPTED_STEPS
        state["iteration_index"] = iteration_index
        state["backtracking_attempts"] = []
        state["accepted"] = False
        state["accepted_alpha"] = None
        state["accepted_backtrack_index"] = None
        state["accepted_loss"] = None
        state["actual_decrease"] = None
        state["actual_decrease_fraction"] = None

        if not math.isfinite(gradient_norm):
            rejected_newton_steps += 1; stop_reason = "nonfinite_gradient"
            terminal_iteration = state; break
        if gradient_norm == 0.0:
            rejected_newton_steps += 1; stop_reason = "floating_point_zero_gradient"
            terminal_iteration = state; break
        damping = state["damping"]
        if damping is not None and damping > 0.0:
            damping_steps += 1
        if direction is None:
            rejected_newton_steps += 1; stop_reason = state["direction_status"]
            terminal_iteration = state; break

        directional_derivative = float(torch.dot(gradient, direction))
        accepted = False; unchanged_candidate = False
        attempts = []
        for backtrack in range(polish.MAX_BACKTRACKS + 1):
            alpha = 2.0 ** (-backtrack)
            candidate = theta + alpha * direction
            attempt = dict(backtrack_index=backtrack, alpha=alpha,
                           candidate=candidate.numpy().tolist())
            if bool(torch.equal(candidate, theta)):
                unchanged_candidate = True
                attempt.update(unchanged_candidate=True, in_bounds=True,
                               evaluated=False, accepted=False)
                if detailed: attempts.append(attempt)
                break
            inside = bool(torch.all(candidate >= torch.from_numpy(config["lower"])) and
                          torch.all(candidate <= torch.from_numpy(config["upper"])))
            attempt.update(unchanged_candidate=False, in_bounds=inside)
            if not inside:
                backtracking_rejections += 1
                attempt.update(evaluated=False, accepted=False)
                if detailed: attempts.append(attempt)
                continue
            with torch.no_grad():
                candidate_loss = float(config["objective"](candidate))
            actual_loss_evaluations += 1
            actual_decrease = current_loss - candidate_loss
            is_accepted = candidate_loss < current_loss
            attempt.update(evaluated=True, candidate_loss=candidate_loss,
                           actual_decrease=actual_decrease,
                           actual_decrease_fraction=(actual_decrease / current_loss
                                                     if current_loss != 0.0 else None),
                           accepted=is_accepted)
            if detailed: attempts.append(attempt)
            if is_accepted:
                previous_loss = current_loss
                theta = candidate.detach(); current_loss = candidate_loss
                theta_history.append(theta.numpy().copy()); loss_history.append(current_loss)
                accepted_with_backtracking += int(backtrack > 0)
                accepted = True
                state.update(accepted=True, accepted_alpha=alpha,
                             accepted_backtrack_index=backtrack,
                             accepted_loss=candidate_loss,
                             actual_decrease=actual_decrease,
                             actual_decrease_fraction=(actual_decrease / previous_loss
                                                       if previous_loss != 0.0 else None))
                break
            backtracking_rejections += 1
        state["backtracking_attempts"] = attempts
        if detailed:
            iterations.append(state)

        if not accepted:
            rejected_newton_steps += 1
            predicted_decrease = max(0.0, -directional_derivative)
            loss_spacing = abs(float(np.spacing(np.float64(current_loss))))
            if unchanged_candidate or predicted_decrease <= 32.0 * loss_spacing:
                stop_reason = "floating_point_no_further_improvement"
            else:
                stop_reason = "backtracking_no_improvement"
            terminal_iteration = state
            break

    saved = source["n3"]
    final_theta = theta_history[-1]
    replay = dict(
        passed=bool(loss_history[-1] < basin.FINAL_GATE),
        final_theta=final_theta.tolist(), final_loss=float(loss_history[-1]),
        stop_reason=stop_reason, accepted_steps=len(loss_history) - 1,
        rejected_newton_steps=rejected_newton_steps,
        objective_evaluations=actual_loss_evaluations,
        backtracking_rejections=backtracking_rejections,
        accepted_with_backtracking=accepted_with_backtracking,
        hessian_evaluations=hessian_evaluations, damping_steps=damping_steps,
        theta_history=[value.tolist() for value in theta_history],
        loss_history=[float(value) for value in loss_history],
    )
    checks = dict(
        label=bool(replay["passed"] == saved["passed"]),
        final_theta=bool(np.array_equal(final_theta, np.asarray(saved["final_theta"]))),
        final_loss=bool(replay["final_loss"] == saved["final_loss"]),
        stop_reason=bool(stop_reason == saved["stop_reason"]),
        accepted_steps=bool(replay["accepted_steps"] == saved["accepted_steps"]),
        objective_evaluations=bool(replay["objective_evaluations"] == saved["objective_evaluations"]),
        backtracking_rejections=bool(replay["backtracking_rejections"] == saved["backtracking_rejections"]),
        damping_steps=bool(replay["damping_steps"] == saved["damping_steps"]),
    )
    if not all(checks.values()):
        raise RuntimeError(f"N3 replay mismatch {source['system']}:{source['sobol_index']}: {checks}")
    return dict(
        system=source["system"], objective_case=source["objective_case"],
        sobol_index=source["sobol_index"], theta=source["theta"],
        label_PASS=source["n3"]["passed"], initial_loss=initial_loss,
        detailed_iterations=iterations, terminal_iteration=terminal_iteration,
        replay=replay, exact_replay=checks,
    )


def _load(path: Path) -> list[dict]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with TRACES.open("a") as handle:
        handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _quantiles(values: list[float]) -> dict:
    array = np.asarray([value for value in values if value is not None and math.isfinite(value)],
                       dtype=np.float64)
    if not len(array): return dict(count=0)
    return dict(count=len(array), minimum=float(array.min()), q10=float(np.quantile(array,.1)),
                median=float(np.median(array)), q90=float(np.quantile(array,.9)),
                maximum=float(array.max()))


def _step_value(row: dict, step: int, kind: str) -> float | None:
    history = row["replay"]["loss_history"]
    if len(history) <= step: return None
    if kind == "ratio": return float(history[step] / history[0]) if history[0] else 0.0
    if kind == "reduction": return float((history[0] - history[step]) / history[0]) if history[0] else 0.0
    if kind == "loss": return float(history[step])
    raise ValueError(kind)


def _nearest_pair(source_report: dict, system: str) -> tuple[int, int]:
    pair = source_report["systems"][system]["nearest_opposite_label_pair"]
    return int(pair["pass_sobol_index"]), int(pair["fail_sobol_index"])


def _summarize(rows: list[dict]) -> dict:
    source_report = json.loads(Path("outputs/direct_n3_correctability_geometry/report.json").read_text())
    output = dict(systems={}, exact_replay_all=all(all(r["exact_replay"].values()) for r in rows))
    for system in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        passed = [row for row in selected if row["label_PASS"]]
        failed = [row for row in selected if not row["label_PASS"]]
        progress = {}
        for step in (1, 2, 3):
            progress[str(step)] = {
                "PASS_loss_ratio": _quantiles([_step_value(row, step, "ratio") for row in passed]),
                "FAIL_loss_ratio": _quantiles([_step_value(row, step, "ratio") for row in failed]),
                "PASS_reduction_fraction": _quantiles([_step_value(row, step, "reduction") for row in passed]),
                "FAIL_reduction_fraction": _quantiles([_step_value(row, step, "reduction") for row in failed]),
                "PASS_reached_step": sum(len(row["replay"]["loss_history"]) > step for row in passed),
                "FAIL_reached_step": sum(len(row["replay"]["loss_history"]) > step for row in failed),
            }
        first = lambda row: row["detailed_iterations"][0]
        first_stats = {}
        for key in ("accepted_alpha", "accepted_backtrack_index", "actual_decrease_fraction",
                    "scaled_gradient_norm", "minimum_eigenvalue", "condition_number",
                    "damping", "damping_trials", "normalized_step_l2", "descent_cosine",
                    "predicted_decrease_over_loss", "minimum_bound_margin"):
            first_stats[key] = {
                "PASS": _quantiles([first(row).get(key) for row in passed]),
                "FAIL": _quantiles([first(row).get(key) for row in failed]),
            }
        route_sequences = {}
        for row in selected:
            sequence = "-".join(item["route"] for item in row["detailed_iterations"][:3])
            key = ("PASS:" if row["label_PASS"] else "FAIL:") + sequence
            route_sequences[key] = route_sequences.get(key, 0) + 1
        pass_index, fail_index = _nearest_pair(source_report, system)
        pair_rows = [next(row for row in selected if row["sobol_index"] == index)
                     for index in (pass_index, fail_index)]
        pair = {
            "PASS": pair_rows[0], "FAIL": pair_rows[1],
            "normalized_parameter_distance": source_report["systems"][system]
                ["nearest_opposite_label_pair"]["normalized_parameter_distance"],
        }
        # Counterexamples based only on the observed early reduction quantity.
        strongest_fail = min((row for row in failed if _step_value(row,3,"ratio") is not None),
                             key=lambda row: _step_value(row,3,"ratio"), default=None)
        weakest_pass = max((row for row in passed if _step_value(row,3,"ratio") is not None),
                           key=lambda row: _step_value(row,3,"ratio"), default=None)
        output["systems"][system] = dict(
            samples=len(selected), PASS=len(passed), FAIL=len(failed),
            progress=progress, first_iteration=first_stats,
            route_sequences_first_three=route_sequences,
            nearest_opposite_pair=pair,
            strongest_three_step_FAIL=(None if strongest_fail is None else dict(
                sobol_index=strongest_fail["sobol_index"],
                loss_ratio=_step_value(strongest_fail,3,"ratio"),
                final_loss=strongest_fail["replay"]["final_loss"],
                accepted_steps=strongest_fail["replay"]["accepted_steps"])),
            weakest_three_step_PASS=(None if weakest_pass is None else dict(
                sobol_index=weakest_pass["sobol_index"],
                loss_ratio=_step_value(weakest_pass,3,"ratio"),
                final_loss=weakest_pass["replay"]["final_loss"],
                accepted_steps=weakest_pass["replay"]["accepted_steps"])),
        )
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = [row for row in _load(SOURCE) if row["system"] in SYSTEMS]
    existing = _load(TRACES); done = {(row["system"],row["sobol_index"]) for row in existing}
    configs = {name: basin._system_config(name) for name in SYSTEMS}
    for source_row in source:
        key = (source_row["system"], source_row["sobol_index"])
        if key in done: continue
        trace = _trace(source_row, configs[source_row["system"]])
        _append(trace); done.add(key)
        print(json.dumps(dict(system=key[0], index=key[1], PASS=trace["label_PASS"],
                              steps=trace["replay"]["accepted_steps"],
                              stop=trace["replay"]["stop_reason"])), flush=True)
    rows = _load(TRACES)
    report = _summarize(rows)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({system: {key:value for key,value in values.items()
                              if key in ("samples","PASS","FAIL")}
                      for system,values in report["systems"].items()}, indent=2), flush=True)


if __name__ == "__main__": main()
