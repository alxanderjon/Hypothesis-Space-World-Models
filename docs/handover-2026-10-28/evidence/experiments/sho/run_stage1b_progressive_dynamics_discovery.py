"""Progressively discover shared SHO dynamics on the failed M=16 dataset."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import direct, linear_sum_assignment, minimize

import experiments.sho.run_stage1b_sixteen_dynamics_scaling_test as scaling
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/progressive_dynamics_discovery")
DIRECT_MAXFUN_PER_ADDITION = 200
POWELL_MAXFEV_PER_ADDITION = 5_000
MAX_COMPONENT_ATTEMPTS = 20
NEWTON_MAX_ITERATIONS = 50
NEWTON_MAX_BACKTRACKS = 60
STABILITY_SCALE = 1e-6
PREFIX_GATE = 1e-20


class DirectBudgetReached(RuntimeError):
    """Enforce an exact scipy DIRECT objective-call ceiling."""


@dataclass(frozen=True)
class NewtonResult:
    values: tuple[float, ...]
    objective: float
    objective_evaluations: int
    accepted_iterations: int
    runtime_seconds: float
    stop_reason: str
    minimum_hessian: float


def _value_string(values: list[float] | tuple[float, ...]) -> str:
    return ";".join(f"{value:.17g}" for value in values)


def _count_string(values: list[int] | tuple[int, ...]) -> str:
    return ";".join(str(value) for value in values)


def _sorted_values(values: Any) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in np.asarray(values, dtype=float)))


def _counts(assignments: torch.Tensor, component_count: int) -> tuple[int, ...]:
    return tuple(
        int((assignments == component).sum().item())
        for component in range(component_count)
    )


class ScalarLossCache:
    """Share expensive L_i(c) calculations across all progressive stages."""

    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.cache: dict[float, torch.Tensor] = {}

    def losses(self, value: float) -> torch.Tensor:
        key = float(value)
        if key not in self.cache:
            self.cache[key] = scaling._candidate_losses(
                key, self.positions
            ).detach()
        return self.cache[key]

    def loss_matrix(self, values: tuple[float, ...]) -> torch.Tensor:
        return torch.stack([self.losses(value) for value in values], dim=1)

    def assigned(
        self, values: tuple[float, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
        matrix = self.loss_matrix(values)
        assigned_losses, assignments = torch.min(matrix, dim=1)
        return assigned_losses, assignments, _counts(assignments, len(values))


class AddObjective:
    """One-dimensional marginal-improvement objective J_add(c)."""

    def __init__(
        self,
        cache: ScalarLossCache,
        existing_best_losses: torch.Tensor | None,
        addition: int,
    ) -> None:
        self.cache = cache
        self.existing_best_losses = existing_best_losses
        self.addition = addition
        self.evaluations = 0
        self.hard_limit = DIRECT_MAXFUN_PER_ADDITION
        self.best_objective = float("inf")
        self.best_c = 0.0
        self.best_losses: torch.Tensor | None = None
        self.cache_size_at_start = len(cache.cache)

    def __call__(self, raw_value: Any) -> float:
        if self.evaluations >= self.hard_limit:
            raise DirectBudgetReached(
                f"DIRECT addition {self.addition} reached {self.hard_limit} calls"
            )
        value = float(np.asarray(raw_value, dtype=float).reshape(-1)[0])
        candidate_losses = self.cache.losses(value)
        combined = (
            candidate_losses
            if self.existing_best_losses is None
            else torch.minimum(self.existing_best_losses, candidate_losses)
        )
        objective = float(combined.mean().item())
        self.evaluations += 1
        if objective < self.best_objective:
            self.best_objective = objective
            self.best_c = value
            self.best_losses = combined.clone()
        return objective

    @property
    def new_scalar_evaluations(self) -> int:
        return len(self.cache.cache) - self.cache_size_at_start


class JointObjective:
    """Permutation-invariant joint prefix objective for one Powell stage."""

    def __init__(self, cache: ScalarLossCache) -> None:
        self.cache = cache
        self.evaluations = 0
        self.cache_size_at_start = len(cache.cache)
        self.best_objective = float("inf")
        self.best_raw_values: tuple[float, ...] = tuple()
        self.best_assignments: torch.Tensor | None = None
        self.best_counts: tuple[int, ...] = tuple()
        self.callback_iterations = 0
        self.callback_assignment_changes = 0
        self.previous_callback_assignments: torch.Tensor | None = None

    def inspect(
        self, raw_values: Any
    ) -> tuple[float, torch.Tensor, tuple[int, ...]]:
        values = tuple(float(value) for value in np.asarray(raw_values, dtype=float))
        matrix = torch.stack([self.cache.losses(value) for value in values], dim=1)
        assigned_losses, assignments = torch.min(matrix, dim=1)
        return (
            float(assigned_losses.mean().item()),
            assignments,
            _counts(assignments, len(values)),
        )

    def __call__(self, raw_values: Any) -> float:
        objective, assignments, counts = self.inspect(raw_values)
        self.evaluations += 1
        if objective < self.best_objective:
            self.best_objective = objective
            self.best_raw_values = tuple(
                float(value) for value in np.asarray(raw_values, dtype=float)
            )
            self.best_assignments = assignments.clone()
            self.best_counts = counts
        return objective

    def callback(self, raw_values: np.ndarray) -> None:
        _, assignments, _ = self.inspect(raw_values)
        self.callback_iterations += 1
        if self.previous_callback_assignments is not None and bool(
            torch.any(assignments != self.previous_callback_assignments).item()
        ):
            self.callback_assignment_changes += 1
        self.previous_callback_assignments = assignments.clone()

    @property
    def new_scalar_evaluations(self) -> int:
        return len(self.cache.cache) - self.cache_size_at_start


def _run_direct_addition(
    cache: ScalarLossCache,
    existing_best_losses: torch.Tensor | None,
    addition: int,
) -> tuple[AddObjective, float, str, float]:
    objective = AddObjective(cache, existing_best_losses, addition)
    start = time.perf_counter()
    termination = ""
    try:
        result = direct(
            objective,
            bounds=[(scaling.LOWER, scaling.UPPER)],
            maxfun=DIRECT_MAXFUN_PER_ADDITION,
            maxiter=10_000,
            locally_biased=False,
            vol_tol=1e-30,
            len_tol=1e-12,
        )
        termination = str(result.message)
    except (DirectBudgetReached, SystemError):
        if objective.evaluations < DIRECT_MAXFUN_PER_ADDITION:
            raise
        termination = (
            f"hard objective-evaluation budget "
            f"{DIRECT_MAXFUN_PER_ADDITION} reached"
        )
    if objective.best_losses is None:
        raise RuntimeError("DIRECT did not evaluate any new-component candidate")
    return objective, objective.best_c, termination, time.perf_counter() - start


def _run_joint_powell(
    cache: ScalarLossCache,
    initial: tuple[float, ...],
) -> tuple[JointObjective, Any, float]:
    objective = JointObjective(cache)
    start = time.perf_counter()
    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="Powell",
        bounds=[(scaling.LOWER, scaling.UPPER)] * len(initial),
        callback=objective.callback,
        options={
            "xtol": 1e-12,
            "ftol": 1e-15,
            "maxiter": 400,
            "maxfev": POWELL_MAXFEV_PER_ADDITION,
            "disp": False,
        },
    )
    return objective, result, time.perf_counter() - start


def _assign_from_values(
    values: tuple[float, ...], positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    matrix = torch.stack(
        [scaling._candidate_losses(value, positions).detach() for value in values],
        dim=1,
    )
    assigned_losses, assignments = torch.min(matrix, dim=1)
    return assigned_losses, assignments, _counts(assignments, len(values))


def _assignment_stability(
    values: tuple[float, ...],
    cache: ScalarLossCache,
) -> tuple[bool, torch.Tensor, float, int, int]:
    matrix = cache.loss_matrix(values)
    sorted_losses, sorted_indices = torch.sort(matrix, dim=1)
    assignments = sorted_indices[:, 0]
    minimum_margin = (
        float((sorted_losses[:, 1] - sorted_losses[:, 0]).min().item())
        if len(values) > 1
        else float("inf")
    )
    maximum_changes = 0
    tested = 0
    for component in range(len(values)):
        for direction in (-1.0, 1.0):
            perturbed = list(values)
            perturbed[component] += direction * STABILITY_SCALE
            if not scaling.LOWER <= perturbed[component] <= scaling.UPPER:
                continue
            tested += 1
            perturbed_matrix = cache.loss_matrix(tuple(perturbed))
            perturbed_assignments = torch.argmin(perturbed_matrix, dim=1)
            maximum_changes = max(
                maximum_changes,
                int((perturbed_assignments != assignments).sum().item()),
            )
    return maximum_changes == 0, assignments, minimum_margin, maximum_changes, tested


def _fixed_objective_float(
    values: tuple[float, ...],
    positions: torch.Tensor,
    assignments: torch.Tensor,
) -> float:
    total = torch.zeros((), dtype=DTYPE)
    for component, value in enumerate(values):
        mask = assignments == component
        total = total + scaling._candidate_losses(value, positions[mask]).sum()
    return float((total / positions.shape[0]).item())


def _newton_polish(
    initial: tuple[float, ...],
    positions: torch.Tensor,
    assignments: torch.Tensor,
) -> NewtonResult:
    values = initial
    evaluations = 1
    accepted_iterations = 0
    objective = _fixed_objective_float(values, positions, assignments)
    stop_reason = "maximum_iterations"
    minimum_hessian = float("nan")
    start = time.perf_counter()
    for _ in range(NEWTON_MAX_ITERATIONS):
        gradients: list[torch.Tensor] = []
        hessians: list[torch.Tensor] = []
        for component, value in enumerate(values):
            parameter = torch.tensor(value, dtype=DTYPE, requires_grad=True)
            mask = assignments == component
            term = scaling._candidate_losses(parameter, positions[mask]).sum()
            term = term / positions.shape[0]
            first = torch.autograd.grad(term, parameter, create_graph=True)[0]
            second = torch.autograd.grad(first, parameter)[0]
            gradients.append(first.detach())
            hessians.append(second.detach())
        gradient = torch.stack(gradients)
        hessian = torch.stack(hessians)
        evaluations += 1
        minimum_hessian = float(hessian.min().item())
        if minimum_hessian <= 0.0 or not bool(torch.isfinite(hessian).all()):
            stop_reason = "nonpositive_or_invalid_hessian"
            break
        proposed = -gradient / hessian
        if not bool(torch.isfinite(proposed).all()):
            stop_reason = "invalid_newton_step"
            break
        accepted = False
        scale = 1.0
        for _ in range(NEWTON_MAX_BACKTRACKS + 1):
            trial = tuple(
                value + scale * float(delta.item())
                for value, delta in zip(values, proposed)
            )
            admissible = (
                all(scaling.LOWER <= value <= scaling.UPPER for value in trial)
                and all(
                    trial[index] < trial[index + 1]
                    for index in range(len(trial) - 1)
                )
            )
            if admissible and trial != values:
                trial_objective = _fixed_objective_float(
                    trial, positions, assignments
                )
                evaluations += 1
                if trial_objective < objective:
                    values = trial
                    objective = trial_objective
                    accepted_iterations += 1
                    accepted = True
                    break
            scale *= 0.5
        if not accepted:
            negligible = all(
                value + float(delta.item()) == value
                for value, delta in zip(values, proposed)
            )
            stop_reason = (
                "numerically_negligible_update"
                if negligible
                else "no_float64_decreasing_step"
            )
            break
        stop_reason = ""
    if not stop_reason:
        stop_reason = "maximum_iterations"
    return NewtonResult(
        values=values,
        objective=objective,
        objective_evaluations=evaluations,
        accepted_iterations=accepted_iterations,
        runtime_seconds=time.perf_counter() - start,
        stop_reason=stop_reason,
        minimum_hessian=minimum_hessian,
    )


def _split_metrics(
    values: tuple[float, ...], positions: torch.Tensor
) -> tuple[float, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    assigned_losses, assignments, counts = _assign_from_values(values, positions)
    return float(assigned_losses.max().item()), assigned_losses, assignments, counts


def _posthoc_component_mapping(
    values: tuple[float, ...], fit: scaling.DynamicsData
) -> dict[int, int]:
    _, assignments, _ = _assign_from_values(values, fit.positions)
    true_count = scaling.M
    confusion = np.zeros((true_count, len(values)), dtype=np.int64)
    for group in range(true_count):
        selected = assignments[fit.group_ids == group]
        for component in range(len(values)):
            confusion[group, component] = int((selected == component).sum().item())
    groups, components = linear_sum_assignment(-confusion)
    return {
        int(group): int(component)
        for group, component in zip(groups, components)
    }


def _posthoc_c_errors(
    values: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    references = np.asarray(
        [scaling._posthoc_metadata(group)[2] for group in range(scaling.M)]
    )
    learned = np.asarray(values)
    groups, components = linear_sum_assignment(
        np.abs(references[:, None] - learned[None, :])
    )
    matched_references = [float("nan")] * len(values)
    errors = [float("nan")] * len(values)
    for group, component in zip(groups, components):
        matched_references[int(component)] = float(references[int(group)])
        errors[int(component)] = abs(
            values[int(component)] - float(references[int(group)])
        )
    return tuple(matched_references), tuple(errors)


def _final_evaluation(
    values: tuple[float, ...],
    datasets: dict[str, scaling.DynamicsData],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping = _posthoc_component_mapping(values, datasets["fit"])
    references, errors = _posthoc_c_errors(values)
    summary: dict[str, Any] = {
        "dynamics_discovered": len(values),
        "recovered_c_values": _value_string(values),
        "matched_reference_c_values_posthoc": _value_string(references),
        "absolute_c_errors_posthoc": _value_string(errors),
        "worst_absolute_c_error_posthoc": max(errors),
    }
    assignment_rows: list[dict[str, Any]] = []
    for split, data in datasets.items():
        matrix = torch.stack(
            [
                scaling._candidate_losses(value, data.positions).detach()
                for value in values
            ],
            dim=1,
        )
        assigned_losses, assignments = torch.min(matrix, dim=1)
        counts = _counts(assignments, len(values))
        correct = 0
        h1_errors: list[float] = []
        rollout_mses: list[float] = []
        rollout_errors: list[float] = []
        for trajectory in range(data.positions.shape[0]):
            group_id = int(data.group_ids[trajectory].item())
            assigned_component = int(assignments[trajectory].item())
            expected_component = mapping.get(group_id, -1)
            assignment_correct = assigned_component == expected_component
            correct += int(assignment_correct)
            group_name, omega, reference_c, future_steps = (
                scaling._posthoc_metadata(group_id)
            )
            position = data.positions[trajectory]
            assigned_c = values[assigned_component]
            target = position[scaling.P : scaling.P + future_steps]
            h1_error = (
                assigned_c
                * position[
                    scaling.P - 1 : scaling.P + future_steps - 1
                ]
                - position[
                    scaling.P - 2 : scaling.P + future_steps - 2
                ]
                - target
            )
            previous, current = position[scaling.P - 2], position[scaling.P - 1]
            predictions = []
            for _ in range(future_steps):
                following = assigned_c * current - previous
                predictions.append(following)
                previous, current = current, following
            rollout_error = torch.stack(predictions) - target
            max_h1 = float(torch.abs(h1_error).max().item())
            rollout_mse = float(torch.mean(rollout_error**2).item())
            max_rollout = float(torch.abs(rollout_error).max().item())
            h1_errors.append(max_h1)
            rollout_mses.append(rollout_mse)
            rollout_errors.append(max_rollout)
            assignment_rows.append(
                {
                    "split": split,
                    "trajectory_index": trajectory,
                    "true_group_posthoc": group_name,
                    "true_omega_posthoc": omega,
                    "true_c_posthoc": reference_c,
                    "assigned_component": assigned_component + 1,
                    "expected_component_posthoc": expected_component + 1,
                    "assigned_c": assigned_c,
                    "assignment_correct_posthoc": assignment_correct,
                    "assigned_prefix_loss": float(
                        assigned_losses[trajectory].item()
                    ),
                    "component_prefix_losses": _value_string(
                        [float(item.item()) for item in matrix[trajectory]]
                    ),
                    "max_h1_error_10_period_evaluation": max_h1,
                    "rollout_10_period_mse": rollout_mse,
                    "max_abs_10_period_rollout_error": max_rollout,
                }
            )
        summary[f"{split}_assignment_counts"] = _count_string(counts)
        summary[f"{split}_active_components"] = sum(count > 0 for count in counts)
        summary[f"{split}_assignment_accuracy"] = correct / data.positions.shape[0]
        summary[f"worst_{split}_prefix_loss"] = float(
            assigned_losses.max().item()
        )
        summary[f"worst_{split}_h1_error"] = max(h1_errors)
        summary[f"worst_{split}_10_period_mse"] = max(rollout_mses)
        summary[f"worst_{split}_rollout_error"] = max(rollout_errors)
    return summary, assignment_rows


def _plot_history(rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    x = [int(row["active_component_count_after_refinement"]) for row in rows]
    fit = [max(float(row["worst_fit_prefix_loss_after_refinement"]), 1e-30) for row in rows]
    validation = [
        max(float(row["worst_validation_prefix_loss_after_refinement"]), 1e-30)
        for row in rows
    ]
    axis.plot(x, fit, marker="o", label="fit")
    axis.plot(x, validation, marker="s", label="validation")
    axis.axhline(PREFIX_GATE, color="black", linestyle="--", label=r"$10^{-20}$ gate")
    axis.set_yscale("log")
    axis.set_xlabel("active component count")
    axis.set_ylabel("worst per-trajectory prefix loss")
    axis.set_xticks(sorted(set(x)))
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "worst_prefix_loss_vs_component_count.png", dpi=180
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Reuse the exact deterministic datasets from the failed simultaneous run.
    fit = scaling._make_data(
        scaling.FIT_PER_GROUP, TRAIN_DATA_SEED + 1_600_000
    )
    validation = scaling._make_data(
        scaling.VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1_601_000
    )
    test = scaling._make_data(
        scaling.TEST_PER_GROUP, TEST_DATA_SEED + 1_602_000
    )
    datasets = {"fit": fit, "validation": validation, "test": test}
    cache = ScalarLossCache(fit.positions)

    values: tuple[float, ...] = tuple()
    fit_best_losses: torch.Tensor | None = None
    component_history: list[dict[str, Any]] = []
    stop_reason = "maximum_component_attempts"
    total_direct_evaluations = 0
    total_powell_evaluations = 0
    total_newton_evaluations = 0
    total_runtime = 0.0

    for addition in range(1, MAX_COMPONENT_ATTEMPTS + 1):
        worst_fit_before = (
            float("inf")
            if fit_best_losses is None
            else float(fit_best_losses.max().item())
        )
        worst_validation_before = (
            float("inf")
            if not values
            else _split_metrics(values, validation.positions)[0]
        )
        print(
            f"Addition {addition:02d}: 1D DIRECT with {len(values)} active components",
            flush=True,
        )
        add_objective, proposed_c, direct_termination, direct_runtime = (
            _run_direct_addition(cache, fit_best_losses, addition)
        )
        direct_new_scalar_evaluations = add_objective.new_scalar_evaluations
        total_direct_evaluations += add_objective.evaluations
        total_runtime += direct_runtime
        initial_raw = tuple(values) + (proposed_c,)
        pre_powell_matrix = cache.loss_matrix(initial_raw)
        pre_powell_losses, pre_powell_assignments = torch.min(
            pre_powell_matrix, dim=1
        )
        pre_powell_counts = _counts(pre_powell_assignments, len(initial_raw))
        proposed_count_before_powell = pre_powell_counts[-1]

        print(
            f"  proposed c={proposed_c:.12g}; joint Powell over {len(initial_raw)}",
            flush=True,
        )
        joint, powell_result, powell_runtime = _run_joint_powell(
            cache, initial_raw
        )
        powell_new_scalar_evaluations = joint.new_scalar_evaluations
        total_powell_evaluations += joint.evaluations
        total_runtime += powell_runtime
        if not joint.best_raw_values or joint.best_assignments is None:
            raise RuntimeError("Powell did not evaluate the joint objective")
        raw_values = joint.best_raw_values
        raw_counts = joint.best_counts
        previously_active_became_unused = any(
            raw_counts[index] == 0 for index in range(len(values))
        )
        new_component_count_after_powell = raw_counts[-1]
        unused_raw_indices = tuple(
            index for index, count in enumerate(raw_counts) if count == 0
        )
        active_values = tuple(
            value
            for index, value in enumerate(raw_values)
            if index not in unused_raw_indices
        )
        active_values = _sorted_values(active_values)
        _, _, active_counts_after_removal = cache.assigned(active_values)

        stable, stable_assignments, margin, changes, perturbations = (
            _assignment_stability(active_values, cache)
        )
        if stable:
            newton = _newton_polish(
                active_values, fit.positions, stable_assignments
            )
            polished_values = _sorted_values(newton.values)
        else:
            newton = NewtonResult(
                values=active_values,
                objective=float("nan"),
                objective_evaluations=0,
                accepted_iterations=0,
                runtime_seconds=0.0,
                stop_reason="skipped_assignments_unstable",
                minimum_hessian=float("nan"),
            )
            polished_values = active_values
        total_newton_evaluations += newton.objective_evaluations
        total_runtime += newton.runtime_seconds

        values = polished_values
        worst_fit_after, fit_best_losses, _, final_fit_counts = _split_metrics(
            values, fit.positions
        )
        (
            worst_validation_after,
            _,
            _,
            final_validation_counts,
        ) = _split_metrics(values, validation.positions)
        gate_passed = (
            worst_fit_after < PREFIX_GATE
            and worst_validation_after < PREFIX_GATE
        )
        component_history.append(
            {
                "addition_attempt": addition,
                "component_count_before_addition": len(initial_raw) - 1,
                "new_c_proposed_by_DIRECT": proposed_c,
                "DIRECT_best_add_objective": add_objective.best_objective,
                "DIRECT_objective_evaluations": add_objective.evaluations,
                "DIRECT_new_scalar_loss_evaluations": direct_new_scalar_evaluations,
                "DIRECT_runtime_seconds": direct_runtime,
                "DIRECT_termination": direct_termination,
                "worst_fit_prefix_loss_before_addition": worst_fit_before,
                "worst_validation_prefix_loss_before_addition": worst_validation_before,
                "worst_fit_prefix_loss_after_DIRECT_addition": float(
                    pre_powell_losses.max().item()
                ),
                "new_component_assignment_count_before_Powell": proposed_count_before_powell,
                "new_component_received_trajectories_before_Powell": (
                    proposed_count_before_powell > 0
                ),
                "Powell_objective_evaluations": joint.evaluations,
                "Powell_new_scalar_loss_evaluations": powell_new_scalar_evaluations,
                "Powell_callback_iterations": joint.callback_iterations,
                "Powell_assignment_changes_between_callbacks": (
                    joint.callback_assignment_changes
                ),
                "Powell_runtime_seconds": powell_runtime,
                "Powell_success": bool(powell_result.success),
                "Powell_termination": str(powell_result.message),
                "new_component_assignment_count_after_Powell": (
                    new_component_count_after_powell
                ),
                "new_component_received_trajectories_after_Powell": (
                    new_component_count_after_powell > 0
                ),
                "Powell_caused_previously_active_component_to_become_unused": (
                    previously_active_became_unused
                ),
                "unused_component_indices_removed_after_Powell": _count_string(
                    tuple(index + 1 for index in unused_raw_indices)
                ),
                "unused_component_count_removed_after_Powell": len(
                    unused_raw_indices
                ),
                "active_values_before_Newton": _value_string(active_values),
                "active_assignment_counts_before_Newton": _count_string(
                    active_counts_after_removal
                ),
                "assignments_stable_before_Newton": stable,
                "minimum_assignment_margin_before_Newton": margin,
                "maximum_assignment_changes_under_perturbation": changes,
                "stability_perturbations_tested": perturbations,
                "Newton_objective_evaluations": newton.objective_evaluations,
                "Newton_accepted_iterations": newton.accepted_iterations,
                "Newton_runtime_seconds": newton.runtime_seconds,
                "Newton_stop_reason": newton.stop_reason,
                "Newton_minimum_hessian": newton.minimum_hessian,
                "active_c_values_after_refinement": _value_string(values),
                "active_component_count_after_refinement": len(values),
                "fit_assignment_counts_after_refinement": _count_string(
                    final_fit_counts
                ),
                "validation_assignment_counts_after_refinement": _count_string(
                    final_validation_counts
                ),
                "worst_fit_prefix_loss_after_refinement": worst_fit_after,
                "worst_validation_prefix_loss_after_refinement": (
                    worst_validation_after
                ),
                "fit_and_validation_1e-20_gate_passed": gate_passed,
            }
        )
        print(
            f"  active={len(values)}, worst fit={worst_fit_after:.3e}, "
            f"validation={worst_validation_after:.3e}, counts={final_fit_counts}",
            flush=True,
        )
        if gate_passed:
            stop_reason = "fit_and_validation_1e-20_gate_passed"
            break

    posthoc_summary, final_assignments = _final_evaluation(values, datasets)
    progressive_summary = {
        "stop_reason": stop_reason,
        "addition_attempts": len(component_history),
        "maximum_component_attempts": MAX_COMPONENT_ATTEMPTS,
        "DIRECT_budget_per_addition": DIRECT_MAXFUN_PER_ADDITION,
        "Powell_maxfev_per_addition": POWELL_MAXFEV_PER_ADDITION,
        "total_DIRECT_objective_evaluations": total_direct_evaluations,
        "total_Powell_objective_evaluations": total_powell_evaluations,
        "total_Newton_objective_evaluations": total_newton_evaluations,
        "total_recorded_optimizer_runtime_seconds": total_runtime,
        "final_fit_and_validation_1e-20_gate_passed": (
            component_history[-1]["fit_and_validation_1e-20_gate_passed"]
        ),
        **posthoc_summary,
    }
    _write_csv(OUTPUT_DIR / "progressive_summary.csv", [progressive_summary])
    _write_csv(OUTPUT_DIR / "component_history.csv", component_history)
    _write_csv(OUTPUT_DIR / "final_assignments.csv", final_assignments)
    _plot_history(component_history)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "stop_reason": stop_reason,
            "attempts": len(component_history),
            "active_components": len(values),
            "worst_fit": posthoc_summary["worst_fit_prefix_loss"],
            "worst_validation": posthoc_summary[
                "worst_validation_prefix_loss"
            ],
            "worst_test": posthoc_summary["worst_test_prefix_loss"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
