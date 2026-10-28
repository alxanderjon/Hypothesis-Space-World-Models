"""Discover the number of shared structured SHO dynamics from prefixes."""

from __future__ import annotations

import itertools
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
import torch
from scipy.optimize import minimize_scalar

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P
from experiments.sho.run_stage1b_continuous_c_search_comparison import LOWER, UPPER, _losses
from experiments.sho.run_stage1b_joint_two_dynamics_search import (
    MixtureData,
    _make_mixture,
    _posthoc_group_metadata,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/unknown_shared_dynamics_count")
FIT_PER_GROUP = 32
VALIDATION_PER_GROUP = 32
TEST_PER_GROUP = 64
CANDIDATE_COUNTS = (1, 2, 3, 4)
COARSE_POINTS = 16
PREFIX_GATE = 1e-12
MAX_BRENT_SWEEPS = 12
BRENT_XATOL = 1e-14
BRENT_MAXITER = 500
CONVERGENCE_TOL = 1e-12


@dataclass(frozen=True)
class CountFit:
    count: int
    coarse_indices: tuple[int, ...]
    coarse_values: tuple[float, ...]
    brackets: tuple[tuple[float, float], ...]
    coarse_objective: float
    values: tuple[float, ...]
    objective: float
    sweeps: int
    stop_reason: str
    history: list[dict[str, Any]]


def _candidate_losses(value: float, positions: torch.Tensor) -> torch.Tensor:
    scalar = torch.tensor(value, dtype=DTYPE)
    return _losses(scalar.expand(positions.shape[0]), positions[:, :P]).detach()


def _loss_matrix(values: tuple[float, ...], positions: torch.Tensor) -> torch.Tensor:
    return torch.stack([_candidate_losses(value, positions) for value in values], dim=1)


def _assign(loss_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assigned_losses, assignments = torch.min(loss_matrix, dim=1)
    return assigned_losses, assignments


def _component_counts(assignments: torch.Tensor, count: int) -> tuple[int, ...]:
    return tuple(int((assignments == component).sum().item()) for component in range(count))


def _component_fields(
    values: tuple[float, ...], counts: tuple[int, ...], maximum_count: int = 4
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "c_values": ";".join(f"{value:.17g}" for value in values),
        "assignment_counts": ";".join(str(value) for value in counts),
    }
    for index in range(maximum_count):
        fields[f"c_{index + 1}"] = values[index] if index < len(values) else ""
        fields[f"assigned_{index + 1}"] = counts[index] if index < len(counts) else ""
    return fields


def _precompute_coarse(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grid = torch.linspace(LOWER, UPPER, COARSE_POINTS, dtype=DTYPE)
    matrix = torch.stack(
        [_candidate_losses(float(value.item()), positions) for value in grid], dim=1
    )
    return grid, matrix


def _best_coarse_combination(
    count: int, grid: torch.Tensor, coarse_losses: torch.Tensor
) -> tuple[tuple[int, ...], float]:
    best_indices: tuple[int, ...] | None = None
    best_objective = float("inf")
    for indices in itertools.combinations(range(COARSE_POINTS), count):
        objective = float(torch.min(coarse_losses[:, indices], dim=1).values.mean().item())
        if objective < best_objective:
            best_objective = objective
            best_indices = indices
    if best_indices is None:
        raise RuntimeError(f"No coarse combination found for M={count}")
    return best_indices, best_objective


def _brackets(
    grid: torch.Tensor, indices: tuple[int, ...]
) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            float(grid[max(0, index - 1)].item()),
            float(grid[min(COARSE_POINTS - 1, index + 1)].item()),
        )
        for index in indices
    )


def _history_row(
    count: int,
    sweep: int,
    stage: str,
    values: tuple[float, ...],
    objective: float,
    assignments: torch.Tensor,
    maximum_change: float | str,
    assignments_changed: bool | str,
    candidate_evaluations: int,
    cumulative_trajectory_evaluations: int,
    stop_reason: str,
) -> dict[str, Any]:
    counts = _component_counts(assignments, count)
    return {
        "M": count,
        "sweep": sweep,
        "stage": stage,
        **_component_fields(values, counts),
        "joint_objective": objective,
        "maximum_parameter_change": maximum_change,
        "assignments_changed": assignments_changed,
        "active_components": sum(value > 0 for value in counts),
        "unused_components": sum(value == 0 for value in counts),
        "candidate_evaluations_this_sweep": candidate_evaluations,
        "trajectory_candidate_loss_evaluations_cumulative": cumulative_trajectory_evaluations,
        "stop_reason": stop_reason,
    }


def _coordinate_refinement(
    positions: torch.Tensor,
    bounds: tuple[float, float],
    current_value: float,
    fixed_best_losses: torch.Tensor | None,
    ordering_ok: Callable[[float], bool],
) -> tuple[float, torch.Tensor, float, int, bool]:
    cache: dict[float, torch.Tensor] = {}

    def losses_at(value: float) -> torch.Tensor:
        key = float(value)
        if key not in cache:
            cache[key] = _candidate_losses(key, positions)
        return cache[key]

    def combined(candidate_losses: torch.Tensor) -> torch.Tensor:
        if fixed_best_losses is None:
            return candidate_losses
        return torch.minimum(candidate_losses, fixed_best_losses)

    current_losses = losses_at(current_value)
    current_objective = float(combined(current_losses).mean().item())

    def objective(value: float) -> float:
        if not ordering_ok(value):
            return float("inf")
        return float(combined(losses_at(value)).mean().item())

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    proposed_value = float(result.x)
    proposed_losses = losses_at(proposed_value)
    proposed_objective = float(combined(proposed_losses).mean().item())
    if proposed_objective <= current_objective:
        return proposed_value, proposed_losses, proposed_objective, len(cache), bool(result.success)
    return current_value, current_losses, current_objective, len(cache), bool(result.success)


def _fit_count(
    count: int,
    positions: torch.Tensor,
    grid: torch.Tensor,
    coarse_losses: torch.Tensor,
) -> CountFit:
    indices, coarse_objective = _best_coarse_combination(count, grid, coarse_losses)
    values = tuple(float(grid[index].item()) for index in indices)
    brackets = _brackets(grid, indices)
    component_losses = [coarse_losses[:, index].clone() for index in indices]
    assigned_losses, assignments = _assign(torch.stack(component_losses, dim=1))
    cumulative_candidates = COARSE_POINTS
    history = [
        _history_row(
            count,
            0,
            "coarse",
            values,
            float(assigned_losses.mean().item()),
            assignments,
            "",
            "",
            COARSE_POINTS,
            positions.shape[0] * cumulative_candidates,
            "",
        )
    ]
    previous_objective = coarse_objective
    stop_reason = ""
    for sweep in range(1, MAX_BRENT_SWEEPS + 1):
        previous_values = values
        previous_assignments = assignments.clone()
        values_list = list(values)
        candidate_evaluations = 0
        optimizer_success = True
        for component in range(count):
            other_losses = [
                component_losses[index]
                for index in range(count)
                if index != component
            ]
            fixed_best = (
                torch.stack(other_losses, dim=1).min(dim=1).values
                if other_losses
                else None
            )
            lower = brackets[component][0]
            upper = brackets[component][1]
            if component > 0:
                lower = max(lower, math.nextafter(values_list[component - 1], math.inf))
            if component + 1 < count:
                upper = min(upper, math.nextafter(values_list[component + 1], -math.inf))
            value, losses, _, evaluations, success = _coordinate_refinement(
                positions,
                (lower, upper),
                values_list[component],
                fixed_best,
                lambda candidate, component=component: (
                    (component == 0 or candidate > values_list[component - 1])
                    and (component + 1 == count or candidate < values_list[component + 1])
                ),
            )
            values_list[component] = value
            component_losses[component] = losses
            candidate_evaluations += evaluations
            optimizer_success = optimizer_success and success
        values = tuple(values_list)
        assigned_losses, assignments = _assign(torch.stack(component_losses, dim=1))
        objective = float(assigned_losses.mean().item())
        maximum_change = max(
            abs(value - previous) for value, previous in zip(values, previous_values)
        )
        assignments_changed = bool(torch.any(assignments != previous_assignments).item())
        cumulative_candidates += candidate_evaluations
        if not optimizer_success:
            stop_reason = "optimizer_failure"
        elif maximum_change <= CONVERGENCE_TOL:
            stop_reason = "parameter_change_below_tolerance"
        elif sweep == MAX_BRENT_SWEEPS:
            stop_reason = "maximum_sweeps"
        history.append(
            _history_row(
                count,
                sweep,
                "alternating_bounded_brent",
                values,
                objective,
                assignments,
                maximum_change,
                assignments_changed,
                candidate_evaluations,
                positions.shape[0] * cumulative_candidates,
                stop_reason,
            )
        )
        previous_objective = objective
        if stop_reason:
            break
    return CountFit(
        count=count,
        coarse_indices=indices,
        coarse_values=tuple(float(grid[index].item()) for index in indices),
        brackets=brackets,
        coarse_objective=coarse_objective,
        values=values,
        objective=previous_objective,
        sweeps=int(history[-1]["sweep"]),
        stop_reason=stop_reason,
        history=history,
    )


def _evaluate_prefixes(
    values: tuple[float, ...], positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    matrix = _loss_matrix(values, positions)
    assigned_losses, assignments = _assign(matrix)
    return assigned_losses, assignments, _component_counts(assignments, len(values))


def _match_selected_to_hidden_groups(
    values: tuple[float, ...]
) -> dict[int, int]:
    true_values = {
        group_id: _posthoc_group_metadata(group_id)[2]
        for group_id in (0, 1)
    }
    if len(values) == 1:
        return {0: 0, 1: 0}
    best_mapping: dict[int, int] | None = None
    best_error = float("inf")
    for components in itertools.permutations(range(len(values)), 2):
        error = sum(
            abs(values[components[group_id]] - true_values[group_id])
            for group_id in (0, 1)
        )
        if error < best_error:
            best_error = error
            best_mapping = {0: components[0], 1: components[1]}
    if best_mapping is None:
        raise RuntimeError("Unable to match selected dynamics to hidden groups")
    return best_mapping


def _selected_assignment_rows(
    split: str,
    data: MixtureData,
    values: tuple[float, ...],
    component_for_group: dict[int, int],
) -> list[dict[str, Any]]:
    matrix = _loss_matrix(values, data.positions)
    assigned_losses, assignments = _assign(matrix)
    rows: list[dict[str, Any]] = []
    for trajectory in range(data.positions.shape[0]):
        group_id = int(data.group_ids[trajectory].item())
        group_name, omega, true_c, future_steps = _posthoc_group_metadata(group_id)
        assigned_component = int(assignments[trajectory].item())
        expected_component = component_for_group[group_id]
        row: dict[str, Any] = {
            "split": split,
            "trajectory_index": trajectory,
            "true_group_posthoc": group_name,
            "true_omega_posthoc": omega,
            "true_c_posthoc": true_c,
            "assigned_component": assigned_component + 1,
            "expected_component_posthoc": expected_component + 1,
            "assigned_c": values[assigned_component],
            "assignment_correct_posthoc": assigned_component == expected_component,
            "assigned_prefix_loss": float(assigned_losses[trajectory].item()),
            "component_prefix_losses": ";".join(
                f"{float(value.item()):.17g}" for value in matrix[trajectory]
            ),
            "max_h1_future_error": "",
            "rollout_10_period_mse": "",
            "max_abs_10_period_trajectory_error": "",
        }
        if split == "test":
            position = data.positions[trajectory]
            assigned_c = values[assigned_component]
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
            row.update(
                {
                    "max_h1_future_error": float(torch.abs(direct_error).max().item()),
                    "rollout_10_period_mse": float(torch.mean(rollout_error**2).item()),
                    "max_abs_10_period_trajectory_error": float(
                        torch.abs(rollout_error).max().item()
                    ),
                }
            )
        rows.append(row)
    return rows


def _plot_model_selection(summary: list[dict[str, Any]]) -> None:
    counts = [int(row["M"]) for row in summary]
    fit = [float(row["worst_fit_prefix_loss"]) for row in summary]
    validation = [float(row["worst_validation_prefix_loss"]) for row in summary]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(counts, fit, marker="o", label="fitting prefixes")
    axis.plot(counts, validation, marker="s", label="validation prefixes")
    axis.axhline(PREFIX_GATE, color="black", linestyle="--", label="1e-12 selection gate")
    axis.set_yscale("log")
    axis.set_xticks(counts)
    axis.set_xlabel("candidate dynamics count M")
    axis.set_ylabel("worst assigned multi-step prefix loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "max_prefix_error_vs_M.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit_data = _make_mixture(FIT_PER_GROUP, TRAIN_DATA_SEED + 1520000)
    validation_data = _make_mixture(
        VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1521000
    )
    test_data = _make_mixture(TEST_PER_GROUP, TEST_DATA_SEED + 1522000)

    print("Precomputing the 16-candidate fitting prefix-loss table", flush=True)
    grid, coarse_losses = _precompute_coarse(fit_data.positions)
    fits: list[CountFit] = []
    for count in CANDIDATE_COUNTS:
        print(f"Fitting candidate dynamics count M={count}", flush=True)
        fits.append(_fit_count(count, fit_data.positions, grid, coarse_losses))

    summary: list[dict[str, Any]] = []
    evaluations: dict[int, dict[str, Any]] = {}
    for fit in fits:
        fit_losses, fit_assignments, fit_counts = _evaluate_prefixes(
            fit.values, fit_data.positions
        )
        validation_losses, validation_assignments, validation_counts = _evaluate_prefixes(
            fit.values, validation_data.positions
        )
        fit_gate = float(fit_losses.max().item()) <= PREFIX_GATE
        validation_gate = float(validation_losses.max().item()) <= PREFIX_GATE
        evaluations[fit.count] = {
            "fit_losses": fit_losses,
            "fit_assignments": fit_assignments,
            "fit_counts": fit_counts,
            "validation_losses": validation_losses,
            "validation_assignments": validation_assignments,
            "validation_counts": validation_counts,
            "fit_gate": fit_gate,
            "validation_gate": validation_gate,
        }
        summary.append(
            {
                "M": fit.count,
                "coarse_c_values": ";".join(f"{value:.17g}" for value in fit.coarse_values),
                "coarse_brackets": ";".join(
                    f"[{low:.17g},{high:.17g}]" for low, high in fit.brackets
                ),
                "coarse_objective": fit.coarse_objective,
                **_component_fields(fit.values, fit_counts),
                "refined_fit_objective": fit.objective,
                "refinement_sweeps": fit.sweeps,
                "stopping_reason": fit.stop_reason,
                "fit_assignment_counts": ";".join(str(value) for value in fit_counts),
                "validation_assignment_counts": ";".join(
                    str(value) for value in validation_counts
                ),
                "fit_active_components": sum(value > 0 for value in fit_counts),
                "validation_active_components": sum(
                    value > 0 for value in validation_counts
                ),
                "fit_unused_components": sum(value == 0 for value in fit_counts),
                "validation_unused_components": sum(
                    value == 0 for value in validation_counts
                ),
                "worst_fit_prefix_loss": float(fit_losses.max().item()),
                "worst_validation_prefix_loss": float(validation_losses.max().item()),
                "fit_prefix_gate_passed": fit_gate,
                "validation_prefix_gate_passed": validation_gate,
                "both_prefix_gates_passed": fit_gate and validation_gate,
                "selected": False,
                "unique_coarse_candidate_values": COARSE_POINTS,
                "fitting_trajectory_candidate_loss_evaluations": fit_data.positions.shape[0]
                * COARSE_POINTS,
            }
        )

    passing_counts = [
        row["M"] for row in summary if bool(row["both_prefix_gates_passed"])
    ]
    selected_count = min(passing_counts) if passing_counts else None
    selected_rows: list[dict[str, Any]] = []
    if selected_count is not None:
        selected_fit = next(fit for fit in fits if fit.count == selected_count)
        selected_summary = next(row for row in summary if row["M"] == selected_count)
        selected_summary["selected"] = True

        print(f"Prefix-only selection chose M={selected_count}", flush=True)
        component_for_group = _match_selected_to_hidden_groups(selected_fit.values)
        selected_rows.extend(
            _selected_assignment_rows(
                "fit", fit_data, selected_fit.values, component_for_group
            )
        )
        selected_rows.extend(
            _selected_assignment_rows(
                "validation",
                validation_data,
                selected_fit.values,
                component_for_group,
            )
        )
        selected_rows.extend(
            _selected_assignment_rows(
                "test", test_data, selected_fit.values, component_for_group
            )
        )
        selected_summary["posthoc_true_count"] = 2
        selected_summary["posthoc_count_correct"] = selected_count == 2
        true_values = {
            group_id: _posthoc_group_metadata(group_id)[2]
            for group_id in (0, 1)
        }
        errors = {
            group_id: abs(
                selected_fit.values[component_for_group[group_id]]
                - true_values[group_id]
            )
            for group_id in (0, 1)
        }
        selected_summary["posthoc_matched_true_c_values"] = ";".join(
            f"{true_values[group_id]:.17g}" for group_id in (0, 1)
        )
        selected_summary["posthoc_abs_dynamics_errors"] = ";".join(
            f"{errors[group_id]:.17g}" for group_id in (0, 1)
        )
        for split in ("fit", "validation", "test"):
            split_rows = [row for row in selected_rows if row["split"] == split]
            correct = sum(
                bool(row["assignment_correct_posthoc"]) for row in split_rows
            )
            selected_summary[f"posthoc_{split}_correct_assignments"] = correct
            selected_summary[f"posthoc_{split}_assignment_accuracy"] = correct / len(
                split_rows
            )
        test_rows = [row for row in selected_rows if row["split"] == "test"]
        selected_summary["posthoc_worst_test_h1_error"] = max(
            float(row["max_h1_future_error"]) for row in test_rows
        )
        selected_summary["posthoc_worst_test_10_period_mse"] = max(
            float(row["rollout_10_period_mse"]) for row in test_rows
        )
        selected_summary["posthoc_worst_test_max_abs_10_period_error"] = max(
            float(row["max_abs_10_period_trajectory_error"])
            for row in test_rows
        )
    else:
        print("No candidate M passed the fixed prefix gates", flush=True)

    refinement_history = [row for fit in fits for row in fit.history]
    _write_csv(OUTPUT_DIR / "model_count_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "refinement_history.csv", refinement_history)
    _write_csv(OUTPUT_DIR / "selected_model_assignments.csv", selected_rows)
    _plot_model_selection(summary)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "selected_M": selected_count,
            "candidate_gate_results": {
                int(row["M"]): bool(row["both_prefix_gates_passed"])
                for row in summary
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
