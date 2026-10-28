#!/usr/bin/env python3
"""Deterministic Powell directions with the frozen learned 1-D N2 line search."""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_edirect_vs_direct import (
    _prefix_losses_numpy,
    _read_csv,
    _write_csv,
)
from experiments.sho.run_stage1b_network2_2d_from_edirect import _load_n1_inputs
from experiments.sho.run_stage1b_network2_powell_like_local_search import (
    ExploreExploitController,
    _history_statistics,
    _sample_query,
)
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    _posthoc_reference,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_powell_2d")
LINE_CHECKPOINT = Path(
    "outputs/sho/stage1B/network2_powell_like_local_search/best_checkpoint.pt"
)
NESTED_SOURCE = Path(
    "outputs/sho/stage1B/nested1d_and_network2_2d/nested1d_rescue.csv"
)
TRAJECTORY_COUNT = 256
TOTAL_EVALUATION_CAP = 500
LINE_EVALUATION_CAP = 120
MIN_LEARNED_LINE_QUERIES = 20
SUCCESS_LOSS = 1.0e-20
THRESHOLDS = (1.0e-4, 1.0e-8, 1.0e-12, 1.0e-16, 1.0e-20)
LINE_SEED = SEED + 9_710_000
DISPLACEMENT_TOLERANCE = 1.0e-14
LOCAL_BRACKET_TOLERANCE = 1.0e-8
LOCAL_STAGNATION_TOLERANCE = 1.0e-6

LINE_RULE = (
    "permanently frozen replay-verified scalar ExploreExploitController; four "
    "initial alpha probes at feasible lower/midpoint/upper/zero; unchanged "
    "DeepSets history statistics and stochastic alpha proposal; stop at global "
    "loss<1e-20 or local convergence (normalized supported bracket<=1e-8 and "
    "relative improvement<=1e-6 after >=20 learned queries); safety cap=120"
)
POWELL_RULE = (
    "deterministic 2-D Powell directions initialized to coordinate axes; exact "
    "sequential frozen learned line searches; standard extrapolation test and "
    "largest-decrease direction replacement by cycle displacement; no library "
    "Powell, gradients, parameter supervision, or learned direction selector"
)


_WORKER_MODEL: ExploreExploitController | None = None


def _get_line_model() -> ExploreExploitController:
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        checkpoint = torch.load(LINE_CHECKPOINT, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("model_class") != "ExploreExploitController"
            or checkpoint.get("status") != "permanently_frozen_replay_verified"
            or not checkpoint.get("replay_verification", {}).get("all_passed", False)
        ):
            raise RuntimeError("Scalar Network 2 checkpoint is not replay-verified")
        model = ExploreExploitController()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _WORKER_MODEL = model
    return _WORKER_MODEL


@dataclass(slots=True)
class EvaluationState:
    position: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    best_theta: np.ndarray
    best_loss: float
    max_evaluations: int = TOTAL_EVALUATION_CAP
    snapshot_budgets: tuple[int, ...] = ()
    count: int = 0
    current_cycle: int = 0
    last_best_evaluation: int = 0
    first_threshold: dict[float, int | None] = field(default_factory=dict)
    best_loss_history: list[float] = field(default_factory=list)
    snapshots: dict[int, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.first_threshold = {
            threshold: (0 if self.best_loss < threshold else None)
            for threshold in THRESHOLDS
        }
        self.best_loss_history = [self.best_loss]

    def evaluate(self, theta: np.ndarray) -> float:
        if self.count >= self.max_evaluations:
            raise RuntimeError("Objective evaluation cap exceeded")
        if not bool(
            np.all(theta >= self.lower - 1.0e-13)
            and np.all(theta <= self.upper + 1.0e-13)
        ):
            raise RuntimeError("Learned line search proposed outside N1 domain")
        theta = np.minimum(np.maximum(theta, self.lower), self.upper)
        loss = float(_prefix_losses_numpy(theta[None, :], self.position)[0])
        self.count += 1
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_theta = theta.copy()
            self.last_best_evaluation = self.count
        for threshold in THRESHOLDS:
            if self.first_threshold[threshold] is None and self.best_loss < threshold:
                self.first_threshold[threshold] = self.count
        self.best_loss_history.append(self.best_loss)
        if self.count in self.snapshot_budgets:
            self.snapshots[self.count] = {
                "best_loss": self.best_loss,
                "best_theta": self.best_theta.copy(),
                "cycles": self.current_cycle,
                "last_best_evaluation": self.last_best_evaluation,
            }
        return loss


@dataclass(slots=True)
class LineResult:
    theta: np.ndarray
    loss: float
    evaluations: int
    converged: bool
    hit_line_cap: bool
    improved: bool
    arbitrary_direction: bool
    proposal_count: int


@dataclass(slots=True)
class TrajectoryResult:
    trajectory_id: int
    starting_theta: np.ndarray
    starting_loss: float
    lower: np.ndarray
    upper: np.ndarray
    n1_source: str
    final_theta: np.ndarray
    final_loss: float
    success: bool
    evaluations: int
    cycles: int
    first_threshold: dict[float, int | None]
    best_loss_history: list[float]
    line_searches: int
    line_converged: int
    line_caps: int
    line_improved: int
    arbitrary_lines: int
    arbitrary_converged: int
    arbitrary_improved: int
    displacement_searches: int
    displacement_improved: int
    displacement_outperformed_coordinates: int
    direction_updates: int
    stopped_for_powell_convergence: bool
    snapshots: dict[int, dict[str, object]] = field(default_factory=dict)
    last_best_evaluation: int = 0
    line_events: list[dict[str, object]] = field(default_factory=list)
    direction_update_evaluations: list[int] = field(default_factory=list)
    displacement_improvement_evaluations: list[int] = field(default_factory=list)


def _alpha_interval(
    theta: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[float, float] | None:
    alpha_lower = -math.inf
    alpha_upper = math.inf
    for axis in range(2):
        if abs(direction[axis]) <= 1.0e-18:
            if theta[axis] < lower[axis] or theta[axis] > upper[axis]:
                return None
            continue
        first = (lower[axis] - theta[axis]) / direction[axis]
        second = (upper[axis] - theta[axis]) / direction[axis]
        alpha_lower = max(alpha_lower, min(first, second))
        alpha_upper = min(alpha_upper, max(first, second))
    if not math.isfinite(alpha_lower) or not math.isfinite(alpha_upper):
        return None
    if alpha_upper - alpha_lower <= 1.0e-15:
        return None
    return float(alpha_lower), float(alpha_upper)


def _supported_bracket_width(sampled_u: torch.Tensor, best_u: float) -> float:
    values = np.unique(sampled_u.detach().cpu().numpy().reshape(-1))
    left = values[values < best_u - 1.0e-15]
    right = values[values > best_u + 1.0e-15]
    left_gap = best_u - float(left[-1]) if len(left) else best_u
    right_gap = float(right[0]) - best_u if len(right) else 1.0 - best_u
    return max(0.0, left_gap) + max(0.0, right_gap)


def _line_search(
    state: EvaluationState,
    theta: np.ndarray,
    theta_loss: float,
    direction: np.ndarray,
    line_seed: int,
) -> LineResult:
    start_evaluations = state.count
    direction_norm = float(np.linalg.norm(direction))
    arbitrary = bool(
        direction_norm > 0.0
        and np.all(np.abs(direction / direction_norm) > 1.0e-10)
    )
    interval = _alpha_interval(theta, direction, state.lower, state.upper)
    if interval is None or direction_norm <= 1.0e-18:
        return LineResult(theta.copy(), theta_loss, 0, True, False, False, arbitrary, 0)
    alpha_lower, alpha_upper = interval
    alpha_width = alpha_upper - alpha_lower
    line_physical_width = alpha_width * direction_norm
    alpha_zero_u = float(np.clip((0.0 - alpha_lower) / alpha_width, 0.0, 1.0))
    initial_u_values = (0.0, 0.5, 1.0, alpha_zero_u)
    sampled_u_values: list[float] = []
    sampled_losses: list[float] = []
    line_best_theta = theta.copy()
    line_best_loss = theta_loss

    for normalized in initial_u_values:
        if state.count >= state.max_evaluations:
            break
        alpha = alpha_lower + normalized * alpha_width
        proposal = theta + alpha * direction
        loss = state.evaluate(proposal)
        sampled_u_values.append(normalized)
        sampled_losses.append(loss)
        if loss < line_best_loss:
            line_best_loss = loss
            line_best_theta = proposal.copy()
        if state.best_loss < SUCCESS_LOSS:
            break

    proposal_count = 0
    converged = False
    hit_line_cap = False
    loss_trace = [line_best_loss]
    scale_trace: list[float] = []
    model = _get_line_model()
    saved_rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(line_seed)
        while (
            state.count < state.max_evaluations
            and state.best_loss >= SUCCESS_LOSS
            and state.count - start_evaluations < LINE_EVALUATION_CAP
        ):
            sampled_u = torch.tensor(sampled_u_values, dtype=DTYPE)[None, :]
            sampled_loss = torch.tensor(sampled_losses, dtype=DTYPE)[None, :]
            mask = torch.ones_like(sampled_u, dtype=torch.bool)
            best_index = int(torch.argmin(sampled_loss, dim=1)[0])
            best_u_tensor = sampled_u[:, best_index]
            best_loss_tensor = sampled_loss[:, best_index]
            statistics = _history_statistics(
                sampled_u,
                sampled_loss,
                mask,
                best_u_tensor,
                best_loss_tensor,
                torch.tensor([min(sampled_losses)], dtype=DTYPE),
                torch.tensor([line_physical_width], dtype=DTYPE),
            )
            with torch.no_grad():
                raw = model(statistics.features, mask, statistics.global_features)
                query_u, _, scale, _, _ = _sample_query(raw, statistics, best_u_tensor)
            normalized = float(query_u[0])
            alpha = alpha_lower + normalized * alpha_width
            proposal = theta + alpha * direction
            loss = state.evaluate(proposal)
            sampled_u_values.append(normalized)
            sampled_losses.append(loss)
            proposal_count += 1
            scale_trace.append(float(scale[0]) * line_physical_width)
            if loss < line_best_loss:
                line_best_loss = loss
                line_best_theta = proposal.copy()
            loss_trace.append(line_best_loss)
            if state.best_loss < SUCCESS_LOSS:
                break
            if proposal_count >= MIN_LEARNED_LINE_QUERIES and len(loss_trace) >= 13:
                recent_start = loss_trace[-13]
                relative_improvement = (recent_start - line_best_loss) / max(
                    abs(recent_start), 1.0e-300
                )
                best_normalized = sampled_u_values[int(np.argmin(sampled_losses))]
                bracket_width = _supported_bracket_width(
                    torch.tensor(sampled_u_values, dtype=DTYPE), best_normalized
                )
                recent_scale = float(np.median(scale_trace[-8:])) if scale_trace else math.inf
                if (
                    bracket_width <= LOCAL_BRACKET_TOLERANCE
                    and relative_improvement <= LOCAL_STAGNATION_TOLERANCE
                ) or (
                    recent_scale <= 1.0e-10 * max(1.0, line_physical_width)
                    and relative_improvement <= LOCAL_STAGNATION_TOLERANCE
                ):
                    converged = True
                    break
        if (
            not converged
            and state.best_loss >= SUCCESS_LOSS
            and state.count - start_evaluations >= LINE_EVALUATION_CAP
        ):
            hit_line_cap = True
    finally:
        torch.random.set_rng_state(saved_rng)
    return LineResult(
        theta=line_best_theta,
        loss=line_best_loss,
        evaluations=state.count - start_evaluations,
        converged=converged or state.best_loss < SUCCESS_LOSS,
        hit_line_cap=hit_line_cap,
        improved=line_best_loss < theta_loss,
        arbitrary_direction=arbitrary,
        proposal_count=proposal_count,
    )


def _run_trajectory(
    arguments: tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        str,
    ],
    max_evaluations: int = TOTAL_EVALUATION_CAP,
    snapshot_budgets: tuple[int, ...] = (),
) -> TrajectoryResult:
    trajectory_id, position, lower, upper, starting_theta, starting_loss, n1_source = arguments
    state = EvaluationState(
        position=position,
        lower=lower,
        upper=upper,
        best_theta=starting_theta.copy(),
        best_loss=starting_loss,
        max_evaluations=max_evaluations,
        snapshot_budgets=snapshot_budgets,
    )
    theta = starting_theta.copy()
    theta_loss = starting_loss
    directions = [np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))]
    cycles = 0
    line_searches = 0
    line_converged = 0
    line_caps = 0
    line_improved = 0
    arbitrary_lines = 0
    arbitrary_converged = 0
    arbitrary_improved = 0
    displacement_searches = 0
    displacement_improved = 0
    displacement_outperformed = 0
    direction_updates = 0
    stopped_for_convergence = False
    line_serial = 0
    line_events: list[dict[str, object]] = []
    direction_update_evaluations: list[int] = []
    displacement_improvement_evaluations: list[int] = []

    def account(result: LineResult) -> None:
        nonlocal line_searches, line_converged, line_caps, line_improved
        nonlocal arbitrary_lines, arbitrary_converged, arbitrary_improved
        line_searches += 1
        line_converged += int(result.converged)
        line_caps += int(result.hit_line_cap)
        line_improved += int(result.improved)
        line_events.append(
            {
                "start_evaluation": state.count - result.evaluations,
                "end_evaluation": state.count,
                "converged": result.converged,
                "hit_line_cap": result.hit_line_cap,
                "improved": result.improved,
                "arbitrary_direction": result.arbitrary_direction,
            }
        )
        if result.arbitrary_direction:
            arbitrary_lines += 1
            arbitrary_converged += int(result.converged)
            arbitrary_improved += int(result.improved)

    stagnant_cycles = 0
    while state.count < state.max_evaluations and state.best_loss >= SUCCESS_LOSS:
        cycles += 1
        state.current_cycle = cycles
        theta_start = theta.copy()
        loss_start = theta_loss
        decreases: list[float] = []
        for direction in directions:
            if state.count >= state.max_evaluations or state.best_loss < SUCCESS_LOSS:
                break
            before = theta_loss
            line_serial += 1
            result = _line_search(
                state,
                theta,
                theta_loss,
                direction,
                LINE_SEED + 100_003 * trajectory_id + line_serial,
            )
            account(result)
            theta, theta_loss = result.theta, result.loss
            decreases.append(max(0.0, before - theta_loss))
        if state.best_loss < SUCCESS_LOSS or state.count >= state.max_evaluations:
            break
        if len(decreases) < 2:
            break

        theta_end = theta.copy()
        loss_end = theta_loss
        displacement = theta_end - theta_start
        displacement_norm = float(np.linalg.norm(displacement))
        biggest_index = int(np.argmax(decreases))
        biggest_decrease = decreases[biggest_index]
        if displacement_norm > DISPLACEMENT_TOLERANCE * max(
            1.0, float(np.linalg.norm(upper - lower))
        ):
            extrapolated = theta_end + displacement
            if bool(np.all(extrapolated >= lower) and np.all(extrapolated <= upper)):
                extrapolated_loss = state.evaluate(extrapolated)
                if state.best_loss < SUCCESS_LOSS:
                    theta, theta_loss = extrapolated, extrapolated_loss
                    break
                if extrapolated_loss < loss_start:
                    first = loss_start - loss_end - biggest_decrease
                    second = loss_start - extrapolated_loss
                    test = (
                        2.0
                        * (loss_start - 2.0 * loss_end + extrapolated_loss)
                        * first
                        * first
                        - biggest_decrease * second * second
                    )
                    if test < 0.0 and state.count < state.max_evaluations:
                        before = theta_loss
                        line_serial += 1
                        result = _line_search(
                            state,
                            theta,
                            theta_loss,
                            displacement,
                            LINE_SEED + 100_003 * trajectory_id + line_serial,
                        )
                        account(result)
                        displacement_searches += 1
                        improvement = max(0.0, before - result.loss)
                        displacement_improved += int(result.improved)
                        if result.improved:
                            displacement_improvement_evaluations.append(state.count)
                        displacement_outperformed += int(
                            improvement > max(decreases) if decreases else improvement > 0.0
                        )
                        theta, theta_loss = result.theta, result.loss
                        directions[biggest_index] = directions[-1]
                        directions[-1] = displacement
                        direction_updates += 1
                        direction_update_evaluations.append(state.count)

        cycle_relative = (loss_start - theta_loss) / max(abs(loss_start), 1.0e-300)
        if cycle_relative <= 1.0e-10 and displacement_norm <= 1.0e-10 * max(
            1.0, float(np.linalg.norm(upper - lower))
        ):
            stagnant_cycles += 1
        else:
            stagnant_cycles = 0
        if stagnant_cycles >= 2:
            stopped_for_convergence = True
            # The experiment's overall stopping rule is the numerical gate or
            # the 500-evaluation emergency cap.  Record Powell stagnation for
            # diagnosis, but keep giving the frozen stochastic line module a
            # chance to improve on a later cycle.

    return TrajectoryResult(
        trajectory_id=trajectory_id,
        starting_theta=starting_theta,
        starting_loss=starting_loss,
        lower=lower,
        upper=upper,
        n1_source=n1_source,
        final_theta=state.best_theta,
        final_loss=state.best_loss,
        success=state.best_loss < SUCCESS_LOSS,
        evaluations=state.count,
        cycles=cycles,
        first_threshold=state.first_threshold,
        best_loss_history=state.best_loss_history,
        line_searches=line_searches,
        line_converged=line_converged,
        line_caps=line_caps,
        line_improved=line_improved,
        arbitrary_lines=arbitrary_lines,
        arbitrary_converged=arbitrary_converged,
        arbitrary_improved=arbitrary_improved,
        displacement_searches=displacement_searches,
        displacement_improved=displacement_improved,
        displacement_outperformed_coordinates=displacement_outperformed,
        direction_updates=direction_updates,
        stopped_for_powell_convergence=stopped_for_convergence,
        snapshots=state.snapshots,
        last_best_evaluation=state.last_best_evaluation,
        line_events=line_events,
        direction_update_evaluations=direction_update_evaluations,
        displacement_improvement_evaluations=displacement_improvement_evaluations,
    )


def _prepare_tasks() -> tuple[list[tuple], np.ndarray]:
    positions, _, hidden_c = _make_exact_labeled_split(TRAJECTORY_COUNT, FRESH_TEST_SEED)
    positions_numpy = positions.to(dtype=DTYPE).cpu().numpy()
    references = _posthoc_reference(hidden_c).cpu().numpy()
    n1 = _load_n1_inputs()
    best_theta = n1.best_theta.cpu().numpy().copy()
    best_loss = n1.best_loss.cpu().numpy().copy()
    source = list(n1.source)
    nested = {int(row["trajectory_id"]): row for row in _read_csv(NESTED_SOURCE)}
    for trajectory_id in (36, 230):
        row = nested[trajectory_id]
        if row["crossed_gate"] != "1":
            raise RuntimeError("Nested 1-D N1 result did not pass")
        best_theta[trajectory_id] = (
            float(row["final_a"]),
            float(row["final_b"]),
        )
        best_loss[trajectory_id] = float(row["final_loss"])
        source[trajectory_id] = "nested1d_rescue"
    tasks = [
        (
            index,
            positions_numpy[index],
            n1.lower[index].cpu().numpy(),
            n1.upper[index].cpu().numpy(),
            best_theta[index],
            float(best_loss[index]),
            source[index],
        )
        for index in range(TRAJECTORY_COUNT)
    ]
    return tasks, references


def _run_all(tasks: list[tuple], workers: int) -> list[TrajectoryResult]:
    if workers <= 1:
        results = []
        for task in tasks:
            result = _run_trajectory(task)
            results.append(result)
            print(
                f"learned-Powell {len(results)}/{len(tasks)} success="
                f"{sum(item.success for item in results)}",
                flush=True,
            )
        return results
    results: list[TrajectoryResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_trajectory, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 8 == 0 or len(results) == len(tasks):
                print(
                    f"learned-Powell {len(results)}/{len(tasks)} success="
                    f"{sum(item.success for item in results)}",
                    flush=True,
                )
    results.sort(key=lambda result: result.trajectory_id)
    return results


def _posthoc_failure_diagnosis(
    result: TrajectoryResult, reference: np.ndarray
) -> str:
    if result.success:
        return "success"
    inside = bool(
        np.all(reference >= result.lower - 1.0e-14)
        and np.all(reference <= result.upper + 1.0e-14)
    )
    if not inside:
        return "bad_n1_domain_excluded_reference_posthoc"
    if result.line_searches and result.line_caps / result.line_searches >= 0.5:
        return "line_search_failure_or_nonconvergence"
    if result.displacement_searches and result.displacement_improved == 0:
        return "poor_direction_update"
    if result.evaluations >= TOTAL_EVALUATION_CAP:
        return "evaluation_cap"
    return "powell_local_convergence_above_gate"


def _trajectory_rows(
    results: list[TrajectoryResult], references: np.ndarray
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result, reference in zip(results, references):
        inside = bool(
            np.all(reference >= result.lower - 1.0e-14)
            and np.all(reference <= result.upper + 1.0e-14)
        )
        row: dict[str, object] = {
            "trajectory_id": result.trajectory_id,
            "n1_source": result.n1_source,
            "n1_starting_loss": result.starting_loss,
            "n1_starting_a": result.starting_theta[0],
            "n1_starting_b": result.starting_theta[1],
            "reference_inside_n1_domain_posthoc": int(inside),
            "success_1e-20": int(result.success),
            "objective_evaluations": result.evaluations,
            "hit_evaluation_cap": int(result.evaluations >= TOTAL_EVALUATION_CAP),
            "powell_cycles": result.cycles,
            "final_loss": result.final_loss,
            "final_a": result.final_theta[0],
            "final_b": result.final_theta[1],
            "final_parameter_error_posthoc": math.hypot(
                result.final_theta[0] - reference[0],
                result.final_theta[1] - reference[1],
            ),
            "line_searches": result.line_searches,
            "line_converged": result.line_converged,
            "line_caps": result.line_caps,
            "line_improved": result.line_improved,
            "arbitrary_direction_line_searches": result.arbitrary_lines,
            "arbitrary_direction_converged": result.arbitrary_converged,
            "arbitrary_direction_improved": result.arbitrary_improved,
            "displacement_line_searches": result.displacement_searches,
            "displacement_line_improved": result.displacement_improved,
            "displacement_outperformed_coordinate_directions": result.displacement_outperformed_coordinates,
            "direction_updates": result.direction_updates,
            "stopped_for_powell_convergence": int(result.stopped_for_powell_convergence),
            "failure_diagnosis": _posthoc_failure_diagnosis(result, reference),
            "line_rule": LINE_RULE,
            "powell_rule": POWELL_RULE,
        }
        for threshold in THRESHOLDS:
            key = f"{threshold:.0e}".replace("-0", "-")
            evaluation = result.first_threshold[threshold]
            row[f"reached_{key}"] = int(evaluation is not None)
            row[f"evaluations_to_{key}"] = evaluation if evaluation is not None else ""
        rows.append(row)
    return rows


def _summary_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = (
        ("all", rows),
        ("n1_normal_edirect", [row for row in rows if row["n1_source"] == "normal_edirect_nominal_local_spacing"]),
        ("n1_adaptive_strip", [row for row in rows if row["n1_source"] == "adaptive_strip_rescue_pre_part_a"]),
        ("n1_nested1d_ids_36_230", [row for row in rows if row["n1_source"] == "nested1d_rescue"]),
        ("trajectory_36", [row for row in rows if int(row["trajectory_id"]) == 36]),
        ("trajectory_230", [row for row in rows if int(row["trajectory_id"]) == 230]),
    )
    output: list[dict[str, object]] = []
    for group, selected in groups:
        successes = [row for row in selected if int(row["success_1e-20"]) == 1]
        summary: dict[str, object] = {
            "row_type": "group_summary",
            "group": group,
            "trajectories": len(selected),
            "success_count_1e-20": len(successes),
            "success_fraction_1e-20": len(successes) / len(selected) if selected else "",
            "failure_count": len(selected) - len(successes),
            "median_objective_evaluations_among_successes": float(np.median([int(row["objective_evaluations"]) for row in successes])) if successes else "",
            "mean_objective_evaluations_among_successes": float(np.mean([int(row["objective_evaluations"]) for row in successes])) if successes else "",
            "worst_objective_evaluations_among_successes": max([int(row["objective_evaluations"]) for row in successes], default=""),
            "median_powell_cycles": float(np.median([int(row["powell_cycles"]) for row in selected])) if selected else "",
            "mean_powell_cycles": float(np.mean([int(row["powell_cycles"]) for row in selected])) if selected else "",
            "worst_powell_cycles": max([int(row["powell_cycles"]) for row in selected], default=""),
            "median_final_parameter_error_posthoc": float(np.median([float(row["final_parameter_error_posthoc"]) for row in selected])) if selected else "",
            "worst_final_parameter_error_posthoc": max([float(row["final_parameter_error_posthoc"]) for row in selected], default=""),
            "line_search_convergence_fraction": sum(int(row["line_converged"]) for row in selected) / max(1, sum(int(row["line_searches"]) for row in selected)),
            "arbitrary_direction_convergence_fraction": sum(int(row["arbitrary_direction_converged"]) for row in selected) / max(1, sum(int(row["arbitrary_direction_line_searches"]) for row in selected)),
            "arbitrary_direction_improvement_fraction": sum(int(row["arbitrary_direction_improved"]) for row in selected) / max(1, sum(int(row["arbitrary_direction_line_searches"]) for row in selected)),
            "displacement_improvement_fraction": sum(int(row["displacement_line_improved"]) for row in selected) / max(1, sum(int(row["displacement_line_searches"]) for row in selected)),
            "displacement_outperformed_coordinate_count": sum(int(row["displacement_outperformed_coordinate_directions"]) for row in selected),
            "evaluation_cap": TOTAL_EVALUATION_CAP,
            "line_evaluation_cap": LINE_EVALUATION_CAP,
            "line_rule": LINE_RULE,
            "powell_rule": POWELL_RULE,
        }
        for threshold in THRESHOLDS:
            key = f"{threshold:.0e}".replace("-0", "-")
            reached = [row for row in selected if int(row[f"reached_{key}"]) == 1]
            evaluations = [int(row[f"evaluations_to_{key}"]) for row in reached]
            summary[f"fraction_reaching_{key}"] = len(reached) / len(selected) if selected else ""
            summary[f"median_evaluations_to_{key}"] = float(np.median(evaluations)) if evaluations else ""
            summary[f"mean_evaluations_to_{key}"] = float(np.mean(evaluations)) if evaluations else ""
            summary[f"worst_evaluations_to_{key}"] = max(evaluations, default="")
        output.append(summary)

    for diagnosis in (
        "line_search_failure_or_nonconvergence",
        "poor_direction_update",
        "bad_n1_domain_excluded_reference_posthoc",
        "evaluation_cap",
        "powell_local_convergence_above_gate",
    ):
        output.append(
            {
                "row_type": "failure_diagnosis",
                "group": diagnosis,
                "trajectories": sum(row["failure_diagnosis"] == diagnosis for row in rows),
                "success_count_1e-20": "",
                "success_fraction_1e-20": "",
                "failure_count": "",
                "evaluation_cap": TOTAL_EVALUATION_CAP,
                "line_evaluation_cap": LINE_EVALUATION_CAP,
                "line_rule": LINE_RULE,
                "powell_rule": POWELL_RULE,
            }
        )
    return output


def _plot(path: Path, results: list[TrajectoryResult]) -> None:
    grid = np.arange(TOTAL_EVALUATION_CAP + 1)
    carried = np.empty((len(results), len(grid)), dtype=np.float64)
    for row_index, result in enumerate(results):
        history = np.asarray(result.best_loss_history)
        carried[row_index, : len(history)] = history
        carried[row_index, len(history) :] = history[-1]
    median = np.median(carried, axis=0)
    lower_quartile = np.quantile(carried, 0.25, axis=0)
    upper_quartile = np.quantile(carried, 0.75, axis=0)
    worst = np.max(carried, axis=0)
    figure, axis = plt.subplots(figsize=(9.4, 5.8))
    axis.semilogy(grid, median, color="tab:blue", linewidth=2.2, label="median")
    axis.fill_between(grid, lower_quartile, upper_quartile, color="tab:blue", alpha=0.18, label="IQR")
    axis.semilogy(grid, worst, color="0.4", linewidth=1.4, label="worst")
    for trajectory_id, color in ((36, "tab:orange"), (230, "tab:red")):
        axis.semilogy(grid, carried[trajectory_id], color=color, linewidth=1.5, label=f"ID {trajectory_id}")
    for threshold in THRESHOLDS:
        axis.axhline(threshold, color="0.78", linestyle="--", linewidth=0.8)
    axis.set_xlabel("objective evaluations")
    axis.set_ylabel("best prefix loss")
    axis.grid(alpha=0.23)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=TRAJECTORY_COUNT)
    args = parser.parse_args()
    for required in (LINE_CHECKPOINT, NESTED_SOURCE):
        if not required.exists():
            raise FileNotFoundError(required)
    _get_line_model()
    tasks, references = _prepare_tasks()
    tasks = tasks[: args.limit]
    references = references[: args.limit]
    results = _run_all(tasks, args.workers)
    if args.limit != TRAJECTORY_COUNT:
        print("smoke run complete; outputs not written", flush=True)
        return
    rows = _trajectory_rows(results, references)
    summary = _summary_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "learned_powell_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "per_trajectory_results.csv", rows)
    _plot(OUTPUT_DIR / "prefix_loss_vs_evaluations.png", results)
    print(
        f"learned-Powell final success={sum(result.success for result in results)}/256; "
        f"wrote {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
