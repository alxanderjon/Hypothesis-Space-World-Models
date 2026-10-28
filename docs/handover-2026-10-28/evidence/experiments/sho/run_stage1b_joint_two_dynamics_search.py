"""Discover two shared SHO dynamics from an unlabeled trajectory mixture."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from scipy.optimize import minimize_scalar
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P
from experiments.sho.run_stage1b_continuous_c_search_comparison import (
    LOWER,
    UPPER,
    _free_c_diagnostic,
    _losses,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/joint_two_dynamics_search")
FIT_PER_GROUP = 32
TEST_PER_GROUP = 64
COARSE_POINTS = 16
MAX_BRENT_SWEEPS = 12
BRENT_XATOL = 1e-14
BRENT_MAXITER = 500
CONVERGENCE_TOL = 1e-12
ADAM_LR = 1e-3
ADAM_STEPS = 500
ADAM_CHECKPOINTS = (0, 10, 20, 50, 100, 200, 500)
ADAM_INITIAL_PAIRS = (
    (-1.50, -1.00),
    (0.00, 0.05),
    (1.80, 1.85),
    (-1.00, 1.80),
)


@dataclass(frozen=True)
class MixtureData:
    positions: torch.Tensor
    group_ids: torch.Tensor


@dataclass(frozen=True)
class CoarseResult:
    grid: torch.Tensor
    loss_matrix: torch.Tensor
    joint_surface: torch.Tensor
    index_a: int
    index_b: int
    c_a: float
    c_b: float
    bracket_a: tuple[float, float]
    bracket_b: tuple[float, float]
    objective: float


def _make_mixture(per_group: int, seed: int) -> MixtureData:
    """Generate balanced data; frequency values are confined to this generator."""
    generator = torch.Generator().manual_seed(seed)
    group_ids = torch.cat(
        (
            torch.zeros(per_group, dtype=torch.long),
            torch.ones(per_group, dtype=torch.long),
        )
    )
    generating_frequencies = torch.where(
        group_ids == 0,
        torch.tensor(2.0, dtype=DTYPE),
        torch.tensor(4.0, dtype=DTYPE),
    )
    initial_conditions = 2.0 * torch.rand(
        2 * per_group, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    slowest_frequency = float(generating_frequencies.min().item())
    future_steps = math.ceil(10.0 * math.tau / (slowest_frequency * float(DELTA_T)))
    times = torch.arange(P + future_steps + 1, dtype=DTYPE).unsqueeze(0)
    times = times.expand(2 * per_group, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, generating_frequencies, times)
    permutation = torch.randperm(2 * per_group, generator=generator)
    return MixtureData(positions[permutation], group_ids[permutation])


def _posthoc_group_metadata(group_id: int) -> tuple[str, float, float, int]:
    """Return label-only metadata; this function is called only after fitting."""
    omega = 2.0 if group_id == 0 else 4.0
    true_c = 2.0 * math.cos(omega * float(DELTA_T))
    future_steps = math.ceil(10.0 * math.tau / (omega * float(DELTA_T)))
    return f"omega_{int(omega)}", omega, true_c, future_steps


def _candidate_losses(value: float | torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        scalar = value
    else:
        scalar = torch.tensor(value, dtype=DTYPE)
    return _losses(scalar.expand(positions.shape[0]), positions[:, :P])


def _joint_from_losses(loss_a: torch.Tensor, loss_b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(loss_a, loss_b).mean()


def _coarse_search(positions: torch.Tensor) -> CoarseResult:
    grid = torch.linspace(LOWER, UPPER, COARSE_POINTS, dtype=DTYPE)
    loss_matrix = torch.stack(
        [_candidate_losses(value, positions).detach() for value in grid], dim=1
    )
    joint_surface = torch.full((COARSE_POINTS, COARSE_POINTS), torch.nan, dtype=DTYPE)
    best_objective = float("inf")
    best_indices = (-1, -1)
    for index_a in range(COARSE_POINTS):
        for index_b in range(index_a + 1, COARSE_POINTS):
            objective = float(
                _joint_from_losses(
                    loss_matrix[:, index_a], loss_matrix[:, index_b]
                ).item()
            )
            joint_surface[index_a, index_b] = objective
            if objective < best_objective:
                best_objective = objective
                best_indices = (index_a, index_b)
    index_a, index_b = best_indices
    bracket_a = (
        float(grid[max(0, index_a - 1)].item()),
        float(grid[min(COARSE_POINTS - 1, index_a + 1)].item()),
    )
    bracket_b = (
        float(grid[max(0, index_b - 1)].item()),
        float(grid[min(COARSE_POINTS - 1, index_b + 1)].item()),
    )
    return CoarseResult(
        grid=grid,
        loss_matrix=loss_matrix,
        joint_surface=joint_surface,
        index_a=index_a,
        index_b=index_b,
        c_a=float(grid[index_a].item()),
        c_b=float(grid[index_b].item()),
        bracket_a=bracket_a,
        bracket_b=bracket_b,
        objective=best_objective,
    )


def _bounded_coordinate_refinement(
    positions: torch.Tensor,
    bounds: tuple[float, float],
    fixed_losses: torch.Tensor,
    current_value: float,
    ordering_ok: Callable[[float], bool],
) -> tuple[float, torch.Tensor, float, int, bool]:
    cache: dict[float, torch.Tensor] = {}

    def losses_at(value: float) -> torch.Tensor:
        key = float(value)
        if key not in cache:
            cache[key] = _candidate_losses(key, positions).detach()
        return cache[key]

    current_losses = losses_at(current_value)
    current_objective = float(_joint_from_losses(current_losses, fixed_losses).item())

    def objective(value: float) -> float:
        if not ordering_ok(value):
            return float("inf")
        return float(_joint_from_losses(losses_at(value), fixed_losses).item())

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    proposed_value = float(result.x)
    proposed_losses = losses_at(proposed_value)
    proposed_objective = float(_joint_from_losses(proposed_losses, fixed_losses).item())
    if proposed_objective <= current_objective:
        return proposed_value, proposed_losses, proposed_objective, len(cache), bool(result.success)
    return current_value, current_losses, current_objective, len(cache), bool(result.success)


def _alternating_brent(
    positions: torch.Tensor, coarse: CoarseResult
) -> tuple[float, float, list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    c_a, c_b = coarse.c_a, coarse.c_b
    loss_a = coarse.loss_matrix[:, coarse.index_a]
    loss_b = coarse.loss_matrix[:, coarse.index_b]
    assignments = (loss_b < loss_a).to(torch.long)
    history: list[dict[str, Any]] = [
        {
            "method": "coarse_to_alternating_brent",
            "run": "main",
            "sweep_or_step": 0,
            "checkpoint": "coarse",
            "c_A": c_a,
            "c_B": c_b,
            "joint_objective": coarse.objective,
            "maximum_parameter_change": "",
            "objective_improvement": "",
            "assignments_changed": "",
            "assigned_A": int((assignments == 0).sum().item()),
            "assigned_B": int((assignments == 1).sum().item()),
            "candidate_evaluations_this_step": COARSE_POINTS,
            "trajectory_candidate_loss_evaluations_cumulative": positions.shape[0]
            * COARSE_POINTS,
            "optimizer_success": True,
            "stop_reason": "",
        }
    ]
    previous_objective = coarse.objective
    cumulative_candidate_evaluations = COARSE_POINTS
    for sweep in range(1, MAX_BRENT_SWEEPS + 1):
        previous_a, previous_b = c_a, c_b
        previous_assignments = assignments.clone()
        upper_a = min(coarse.bracket_a[1], math.nextafter(c_b, -math.inf))
        c_a, loss_a, _, evaluations_a, success_a = _bounded_coordinate_refinement(
            positions,
            (coarse.bracket_a[0], upper_a),
            loss_b,
            c_a,
            lambda value: value < c_b,
        )
        lower_b = max(coarse.bracket_b[0], math.nextafter(c_a, math.inf))
        c_b, loss_b, objective, evaluations_b, success_b = _bounded_coordinate_refinement(
            positions,
            (lower_b, coarse.bracket_b[1]),
            loss_a,
            c_b,
            lambda value: value > c_a,
        )
        assignments = (loss_b < loss_a).to(torch.long)
        maximum_change = max(abs(c_a - previous_a), abs(c_b - previous_b))
        improvement = previous_objective - objective
        assignments_changed = bool(torch.any(assignments != previous_assignments).item())
        cumulative_candidate_evaluations += evaluations_a + evaluations_b
        stop_reason = ""
        if maximum_change <= CONVERGENCE_TOL:
            stop_reason = "parameter_change_below_tolerance"
        elif sweep == MAX_BRENT_SWEEPS:
            stop_reason = "maximum_sweeps"
        history.append(
            {
                "method": "coarse_to_alternating_brent",
                "run": "main",
                "sweep_or_step": sweep,
                "checkpoint": "refinement_sweep",
                "c_A": c_a,
                "c_B": c_b,
                "joint_objective": objective,
                "maximum_parameter_change": maximum_change,
                "objective_improvement": improvement,
                "assignments_changed": assignments_changed,
                "assigned_A": int((assignments == 0).sum().item()),
                "assigned_B": int((assignments == 1).sum().item()),
                "candidate_evaluations_this_step": evaluations_a + evaluations_b,
                "trajectory_candidate_loss_evaluations_cumulative": positions.shape[0]
                * cumulative_candidate_evaluations,
                "optimizer_success": bool(success_a and success_b),
                "stop_reason": stop_reason,
            }
        )
        previous_objective = objective
        if stop_reason:
            break
    return c_a, c_b, history, loss_a, loss_b


def _adam_control(
    positions: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    histories: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    for run_index, initial_pair in enumerate(ADAM_INITIAL_PAIRS):
        pair = nn.Parameter(torch.tensor(initial_pair, dtype=DTYPE))
        optimizer = torch.optim.Adam([pair], lr=ADAM_LR)
        best_pair = pair.detach().clone()
        with torch.no_grad():
            best_objective = float(
                _joint_from_losses(
                    _candidate_losses(pair[0], positions),
                    _candidate_losses(pair[1], positions),
                ).item()
            )
        checkpoint_evaluations = 0
        for step in range(0, ADAM_STEPS + 1):
            if step > 0:
                optimizer.zero_grad()
                loss_a = _candidate_losses(pair[0], positions)
                loss_b = _candidate_losses(pair[1], positions)
                objective = _joint_from_losses(loss_a, loss_b)
                if float(objective.detach().item()) < best_objective:
                    best_objective = float(objective.detach().item())
                    best_pair = pair.detach().clone()
                objective.backward()
                optimizer.step()
                with torch.no_grad():
                    pair.clamp_(LOWER, UPPER)
                    pair.copy_(torch.sort(pair).values)
            if step in ADAM_CHECKPOINTS:
                with torch.no_grad():
                    loss_a = _candidate_losses(pair[0], positions)
                    loss_b = _candidate_losses(pair[1], positions)
                    objective_value = float(_joint_from_losses(loss_a, loss_b).item())
                    if objective_value < best_objective:
                        best_objective = objective_value
                        best_pair = pair.detach().clone()
                    best_loss_a = _candidate_losses(best_pair[0], positions)
                    best_loss_b = _candidate_losses(best_pair[1], positions)
                    best_assignments = (best_loss_b < best_loss_a).to(torch.long)
                checkpoint_evaluations += 4
                histories.append(
                    {
                        "method": "adam_control",
                        "run": f"bad_init_{run_index}",
                        "initial_c_A": initial_pair[0],
                        "initial_c_B": initial_pair[1],
                        "sweep_or_step": step,
                        "checkpoint": "adam_checkpoint",
                        "c_A": float(best_pair[0].item()),
                        "c_B": float(best_pair[1].item()),
                        "joint_objective": best_objective,
                        "maximum_parameter_change": "",
                        "objective_improvement": "",
                        "assignments_changed": "",
                        "assigned_A": int((best_assignments == 0).sum().item()),
                        "assigned_B": int((best_assignments == 1).sum().item()),
                        "candidate_evaluations_this_step": "",
                        "trajectory_candidate_loss_evaluations_cumulative": positions.shape[0]
                        * (2 * step + checkpoint_evaluations),
                        "optimizer_success": "",
                        "stop_reason": "" if step < ADAM_STEPS else "fixed_step_budget",
                    }
                )
        finals.append(
            {
                "run": f"bad_init_{run_index}",
                "initial_c_A": initial_pair[0],
                "initial_c_B": initial_pair[1],
                "c_A": float(best_pair[0].item()),
                "c_B": float(best_pair[1].item()),
                "joint_objective": best_objective,
            }
        )
    return histories, finals


def _match_components(
    c_a: float, c_b: float, references: dict[int, float]
) -> dict[int, str]:
    identity_error = abs(c_a - references[0]) + abs(c_b - references[1])
    swapped_error = abs(c_b - references[0]) + abs(c_a - references[1])
    if identity_error <= swapped_error:
        return {0: "A", 1: "B"}
    return {0: "B", 1: "A"}


def _trajectory_evaluations(
    split: str,
    data: MixtureData,
    free_c: torch.Tensor,
    c_a: float,
    c_b: float,
    component_for_group: dict[int, str],
) -> list[dict[str, Any]]:
    with torch.no_grad():
        loss_a = _candidate_losses(c_a, data.positions)
        loss_b = _candidate_losses(c_b, data.positions)
    assigned = torch.where(loss_a <= loss_b, 0, 1)
    rows: list[dict[str, Any]] = []
    for trajectory in range(data.positions.shape[0]):
        group_id = int(data.group_ids[trajectory].item())
        group_name, omega, true_c, future_steps = _posthoc_group_metadata(group_id)
        expected_component = component_for_group[group_id]
        assigned_component = "A" if int(assigned[trajectory].item()) == 0 else "B"
        assigned_c = c_a if assigned_component == "A" else c_b
        correct_loss = loss_a[trajectory] if expected_component == "A" else loss_b[trajectory]
        incorrect_loss = loss_b[trajectory] if expected_component == "A" else loss_a[trajectory]
        position = data.positions[trajectory]
        target = position[P : P + future_steps]
        direct_error = (
            assigned_c * position[P - 1 : P + future_steps - 1]
            - position[P - 2 : P + future_steps - 2]
            - target
        )
        previous, current = position[P - 2], position[P - 1]
        predictions = []
        for _ in range(future_steps):
            following = assigned_c * current - previous
            predictions.append(following)
            previous, current = current, following
        rollout_error = torch.stack(predictions) - target
        rows.append(
            {
                "split": split,
                "trajectory_index": trajectory,
                "true_group_posthoc": group_name,
                "true_omega_posthoc": omega,
                "true_c_posthoc": true_c,
                "c_free_posthoc": float(free_c[trajectory].item()),
                "expected_component_posthoc": expected_component,
                "assigned_component": assigned_component,
                "assigned_c": assigned_c,
                "assignment_correct": assigned_component == expected_component,
                "prefix_loss_A": float(loss_a[trajectory].item()),
                "prefix_loss_B": float(loss_b[trajectory].item()),
                "correct_prefix_loss_posthoc": float(correct_loss.item()),
                "incorrect_prefix_loss_posthoc": float(incorrect_loss.item()),
                "assignment_margin": float((incorrect_loss - correct_loss).item()),
                "future_steps_for_10_periods": future_steps,
                "max_h1_future_error": float(torch.abs(direct_error).max().item()),
                "rollout_10_period_mse": float(torch.mean(rollout_error**2).item()),
                "max_abs_10_period_trajectory_error": float(
                    torch.abs(rollout_error).max().item()
                ),
            }
        )
    return rows


def _assignment_summary(
    rows: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split]
    correct = sum(bool(row["assignment_correct"]) for row in selected)
    return {
        "section": "assignment",
        "split": split,
        "group": "overall",
        "trajectories": len(selected),
        "correct_assignments": correct,
        "incorrect_assignments": len(selected) - correct,
        "all_assignments_correct": correct == len(selected),
        "worst_assignment_margin": min(float(row["assignment_margin"]) for row in selected),
    }


def _future_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    groups = [
        ("overall", test_rows),
        ("learned_A", [row for row in test_rows if row["assigned_component"] == "A"]),
        ("learned_B", [row for row in test_rows if row["assigned_component"] == "B"]),
    ]
    result = []
    for name, selected in groups:
        result.append(
            {
                "section": "future_simulation",
                "split": "test",
                "group": name,
                "trajectories": len(selected),
                "worst_h1_error": max(float(row["max_h1_future_error"]) for row in selected),
                "worst_10_period_mse": max(
                    float(row["rollout_10_period_mse"]) for row in selected
                ),
                "worst_max_abs_10_period_trajectory_error": max(
                    float(row["max_abs_10_period_trajectory_error"])
                    for row in selected
                ),
            }
        )
    return result


def _control_summary_rows(
    controls: list[dict[str, Any]],
    fit_data: MixtureData,
    references: dict[int, float],
) -> list[dict[str, Any]]:
    rows = []
    for control in controls:
        c_a, c_b = float(control["c_A"]), float(control["c_B"])
        mapping = _match_components(c_a, c_b, references)
        with torch.no_grad():
            loss_a = _candidate_losses(c_a, fit_data.positions)
            loss_b = _candidate_losses(c_b, fit_data.positions)
        assigned = torch.where(loss_a <= loss_b, 0, 1)
        correct = 0
        errors = []
        for group_id in (0, 1):
            component = mapping[group_id]
            learned = c_a if component == "A" else c_b
            errors.append(abs(learned - references[group_id]))
        for index in range(fit_data.positions.shape[0]):
            group_id = int(fit_data.group_ids[index].item())
            expected = 0 if mapping[group_id] == "A" else 1
            correct += int(int(assigned[index].item()) == expected)
        rows.append(
            {
                "section": "adam_control",
                "method": control["run"],
                "initial_c_A": control["initial_c_A"],
                "initial_c_B": control["initial_c_B"],
                "learned_c_A": c_a,
                "learned_c_B": c_b,
                "joint_objective": control["joint_objective"],
                "maximum_dynamics_error": max(errors),
                "fit_correct_assignments": correct,
                "fit_all_assignments_correct": correct == fit_data.positions.shape[0],
                "both_dynamics_within_1e-6": max(errors) <= 1e-6,
            }
        )
    return rows


def _plot_heatmap(
    coarse: CoarseResult, refined_a: float, refined_b: float
) -> None:
    grid = coarse.grid.cpu().numpy()
    surface = coarse.joint_surface.cpu().numpy()
    masked = np.ma.masked_invalid(surface)
    positive = masked.compressed()
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.pcolormesh(
        grid,
        grid,
        masked,
        shading="nearest",
        norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())),
        cmap="viridis",
    )
    axis.scatter(coarse.c_b, coarse.c_a, marker="*", s=170, color="white", edgecolor="black", label="best coarse pair")
    axis.scatter(refined_b, refined_a, marker="x", s=90, color="red", linewidths=2, label="refined pair")
    axis.plot([LOWER, UPPER], [LOWER, UPPER], color="gray", linestyle="--", linewidth=1)
    axis.set_xlim(LOWER, UPPER)
    axis.set_ylim(LOWER, UPPER)
    axis.set_xlabel("c_B")
    axis.set_ylabel("c_A")
    axis.set_title("Unlabeled joint prefix objective, c_A < c_B")
    axis.legend(loc="lower right")
    figure.colorbar(image, ax=axis, label="J(c_A, c_B)")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "joint_objective_heatmap.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_data = _make_mixture(FIT_PER_GROUP, TRAIN_DATA_SEED + 1510000)
    test_data = _make_mixture(TEST_PER_GROUP, TEST_DATA_SEED + 1511000)

    print("Stage 1: unlabeled 16x16 joint coarse search", flush=True)
    coarse = _coarse_search(fit_data.positions)
    print("Stage 2: alternating bounded Brent refinement", flush=True)
    c_a, c_b, refinement_history, _, _ = _alternating_brent(
        fit_data.positions, coarse
    )
    print("Control: ordinary Adam from deliberately bad pairs", flush=True)
    adam_history, adam_finals = _adam_control(fit_data.positions)

    print("Post-hoc only: individual free-c references", flush=True)
    fit_free, _ = _free_c_diagnostic(fit_data.positions[:, :P])
    test_free, _ = _free_c_diagnostic(test_data.positions[:, :P])
    fit_references = {
        group_id: float(torch.median(fit_free[fit_data.group_ids == group_id]).item())
        for group_id in (0, 1)
    }
    component_for_group = _match_components(c_a, c_b, fit_references)

    assignment_rows = _trajectory_evaluations(
        "fit", fit_data, fit_free, c_a, c_b, component_for_group
    )
    assignment_rows.extend(
        _trajectory_evaluations(
            "test", test_data, test_free, c_a, c_b, component_for_group
        )
    )

    summary: list[dict[str, Any]] = [
        {
            "section": "coarse_search",
            "method": "16x16_valid_ordered_pairs",
            "learned_c_A": coarse.c_a,
            "learned_c_B": coarse.c_b,
            "bracket_A_low": coarse.bracket_a[0],
            "bracket_A_high": coarse.bracket_a[1],
            "bracket_B_low": coarse.bracket_b[0],
            "bracket_B_high": coarse.bracket_b[1],
            "joint_objective": coarse.objective,
            "unique_candidate_values": COARSE_POINTS,
            "valid_pair_objectives": COARSE_POINTS * (COARSE_POINTS - 1) // 2,
            "trajectory_candidate_prefix_loss_evaluations": fit_data.positions.shape[0]
            * COARSE_POINTS,
        },
        {
            "section": "refined_solution",
            "method": "alternating_bounded_brent",
            "learned_c_A": c_a,
            "learned_c_B": c_b,
            "joint_objective": refinement_history[-1]["joint_objective"],
            "refinement_sweeps": refinement_history[-1]["sweep_or_step"],
            "stopping_reason": refinement_history[-1]["stop_reason"],
        },
    ]
    for component, learned in (("A", c_a), ("B", c_b)):
        group_id = next(
            group for group, matched_component in component_for_group.items()
            if matched_component == component
        )
        group_name, omega, true_c, _ = _posthoc_group_metadata(group_id)
        fit_values = fit_free[fit_data.group_ids == group_id]
        test_values = test_free[test_data.group_ids == group_id]
        reference = fit_references[group_id]
        bracket = coarse.bracket_a if component == "A" else coarse.bracket_b
        summary.append(
            {
                "section": "dynamics_discovery",
                "method": "coarse_to_alternating_brent",
                "group": f"learned_{component}_matched_{group_name}",
                "learned_c": learned,
                "fit_reference_c_median": reference,
                "fit_reference_c_min": float(fit_values.min().item()),
                "fit_reference_c_max": float(fit_values.max().item()),
                "test_reference_c_median": float(torch.median(test_values).item()),
                "true_omega_posthoc": omega,
                "true_c_posthoc": true_c,
                "abs_error_vs_fit_reference": abs(learned - reference),
                "abs_error_vs_true_c": abs(learned - true_c),
                "reference_inside_selected_bracket": bracket[0] <= reference <= bracket[1],
            }
        )
    summary.append(_assignment_summary(assignment_rows, "fit"))
    summary.append(_assignment_summary(assignment_rows, "test"))
    summary.extend(_future_summary_rows(assignment_rows))
    summary.extend(_control_summary_rows(adam_finals, fit_data, fit_references))

    dynamics_rows = [row for row in summary if row["section"] == "dynamics_discovery"]
    maximum_dynamics_error = max(float(row["abs_error_vs_fit_reference"]) for row in dynamics_rows)
    brackets_correct = all(bool(row["reference_inside_selected_bracket"]) for row in dynamics_rows)
    fit_assignments_correct = all(
        bool(row["assignment_correct"])
        for row in assignment_rows
        if row["split"] == "fit"
    )
    test_assignments_correct = all(
        bool(row["assignment_correct"])
        for row in assignment_rows
        if row["split"] == "test"
    )
    worst_rollout = max(
        float(row["max_abs_10_period_trajectory_error"])
        for row in assignment_rows
        if row["split"] == "test"
    )
    if not brackets_correct:
        limiting_component = "incorrect_global_basin_selection"
    elif maximum_dynamics_error > 1e-6:
        limiting_component = "inaccurate_local_refinement"
    elif not (fit_assignments_correct and test_assignments_correct):
        limiting_component = "incorrect_trajectory_assignment"
    elif worst_rollout > 1e-6:
        limiting_component = "rollout_sensitivity_after_correct_discovery"
    else:
        limiting_component = "no_failure_observed_at_requested_precision"
    summary.append(
        {
            "section": "diagnosis",
            "method": "coarse_to_alternating_brent",
            "coarse_brackets_contain_references": brackets_correct,
            "maximum_dynamics_error": maximum_dynamics_error,
            "fit_all_assignments_correct": fit_assignments_correct,
            "test_all_assignments_correct": test_assignments_correct,
            "worst_test_10_period_trajectory_error": worst_rollout,
            "limiting_component": limiting_component,
        }
    )

    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "per_trajectory_assignments.csv", assignment_rows)
    _write_csv(OUTPUT_DIR / "refinement_history.csv", refinement_history + adam_history)
    _plot_heatmap(coarse, c_a, c_b)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "learned_c_A": c_a,
            "learned_c_B": c_b,
            "fit_correct": sum(row["assignment_correct"] for row in assignment_rows if row["split"] == "fit"),
            "test_correct": sum(row["assignment_correct"] for row in assignment_rows if row["split"] == "test"),
            "limiting_component": limiting_component,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
