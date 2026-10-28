"""Stage 1B scaling test: recover 16 shared SHO dynamics without labels.

The fitting path is deliberately restricted to prefix positions and the
prefix-only multi-step loss:

    DIRECT (10,000 joint calls) -> bounded joint Powell -> safeguarded Newton.

Hidden group metadata is retained only for post-hoc evaluation.
"""

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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P
from experiments.sho.run_stage1b_continuous_c_search_comparison import LOWER, UPPER, _losses
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

OUTPUT_DIR = Path("outputs/sho/stage1B/sixteen_dynamics_scaling_test")
M = 16
FIT_PER_GROUP = 32
VALIDATION_PER_GROUP = 8
TEST_PER_GROUP = 8
DIRECT_MAXFUN = 10_000
POWELL_MAXFEV = 20_000
CONTROL_MAXFEV = 20_000
NEWTON_MAX_ITERATIONS = 50
NEWTON_MAX_BACKTRACKS = 60
PRIMARY_GATE = 1e-20
SECONDARY_GATE = 1e-24
STABILITY_SCALE = 1e-6

# The generator spans the interior of the admissible interval uniformly in c.
# These values are never provided to any search routine.
HIDDEN_C_MIN = -1.90
HIDDEN_C_MAX = 1.90


def _linspace_tuple(lower: float, upper: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(lower, upper, M))


CONTROL_STARTS = (
    ("clustered_near_zero", _linspace_tuple(-0.15, 0.15)),
    ("clustered_near_left_side", _linspace_tuple(-1.85, -1.45)),
    ("evenly_spaced_generic", _linspace_tuple(-1.75, 1.75)),
)


@dataclass(frozen=True)
class DynamicsData:
    positions: torch.Tensor
    group_ids: torch.Tensor


@dataclass(frozen=True)
class StageResult:
    method: str
    stage: str
    values: tuple[float, ...]
    costs: dict[str, Any]
    details: dict[str, Any]


@dataclass(frozen=True)
class NewtonResult:
    values: tuple[float, ...]
    objective: float
    objective_evaluations: int
    underlying_loss_calculations: int
    accepted_iterations: int
    stop_reason: str
    minimum_hessian_eigenvalue: float
    maximum_hessian_offdiagonal: float
    runtime_seconds: float
    history: list[dict[str, Any]]


class DirectBudgetReached(RuntimeError):
    """Hard stop for scipy DIRECT's otherwise approximate maxfun budget."""


def _reference_cs() -> torch.Tensor:
    """Post-hoc/generator helper; never called by an optimizer."""
    return torch.linspace(HIDDEN_C_MIN, HIDDEN_C_MAX, M, dtype=DTYPE)


def _make_data(per_group: int, seed: int) -> DynamicsData:
    """Generate shuffled trajectories; hidden metadata stays in this function."""
    generator = torch.Generator().manual_seed(seed)
    hidden_cs = _reference_cs()
    hidden_omegas = torch.acos(hidden_cs / 2.0) / float(DELTA_T)
    group_ids = torch.arange(M, dtype=torch.long).repeat_interleave(per_group)
    generating_omegas = hidden_omegas[group_ids]
    initial_conditions = 2.0 * torch.rand(
        M * per_group, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    slowest_frequency = float(hidden_omegas.min().item())
    future_steps = math.ceil(
        10.0 * math.tau / (slowest_frequency * float(DELTA_T))
    )
    times = torch.arange(P + future_steps + 1, dtype=DTYPE).unsqueeze(0)
    times = times.expand(M * per_group, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, generating_omegas, times)
    permutation = torch.randperm(M * per_group, generator=generator)
    return DynamicsData(positions[permutation], group_ids[permutation])


def _posthoc_metadata(group_id: int) -> tuple[str, float, float, int]:
    reference_c = float(_reference_cs()[group_id].item())
    omega = math.acos(reference_c / 2.0) / float(DELTA_T)
    future_steps = math.ceil(10.0 * math.tau / (omega * float(DELTA_T)))
    return f"dynamics_{group_id + 1:02d}", omega, reference_c, future_steps


def _candidate_losses(
    value: float | torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    scalar = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=DTYPE)
    return _losses(scalar.expand(positions.shape[0]), positions[:, :P])


def _sorted_values(values: Any) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in np.asarray(values, dtype=float)))


def _value_string(values: tuple[float, ...] | list[float]) -> str:
    return ";".join(f"{value:.17g}" for value in values)


def _summary_fields(
    prefix: str, values: tuple[float, ...] | list[float]
) -> dict[str, Any]:
    fields: dict[str, Any] = {f"{prefix}_values": _value_string(values)}
    for index, value in enumerate(values):
        fields[f"{prefix}_{index + 1:02d}"] = value
    return fields


def _counts(assignments: torch.Tensor) -> tuple[int, ...]:
    return tuple(int((assignments == index).sum().item()) for index in range(M))


class JointObjective:
    """Cached J(theta) with exact call counts and best-so-far history."""

    def __init__(self, positions: torch.Tensor, method: str, run: str) -> None:
        self.positions = positions
        self.method = method
        self.run = run
        self.scalar_cache: dict[float, torch.Tensor] = {}
        self.phase = ""
        self.total_evaluations = 0
        self.phase_evaluations = 0
        self.phase_new_scalars = 0
        self.phase_hard_limit: int | None = None
        self.best_objective = float("inf")
        self.best_values = tuple(0.0 for _ in range(M))
        self.best_counts = tuple(0 for _ in range(M))
        self.history: list[dict[str, Any]] = []

    def set_phase(self, phase: str, hard_limit: int | None = None) -> None:
        self.phase = phase
        self.phase_evaluations = 0
        self.phase_new_scalars = 0
        self.phase_hard_limit = hard_limit

    def losses(self, value: float) -> torch.Tensor:
        key = float(value)
        if key not in self.scalar_cache:
            self.scalar_cache[key] = _candidate_losses(key, self.positions).detach()
            self.phase_new_scalars += 1
        return self.scalar_cache[key]

    def inspect(
        self, values: tuple[float, ...]
    ) -> tuple[float, torch.Tensor, tuple[int, ...]]:
        matrix = torch.stack([self.losses(value) for value in values], dim=1)
        assigned_losses, assignments = torch.min(matrix, dim=1)
        return float(assigned_losses.mean().item()), assignments, _counts(assignments)

    def __call__(self, raw_values: Any) -> float:
        if (
            self.phase_hard_limit is not None
            and self.phase_evaluations >= self.phase_hard_limit
        ):
            raise DirectBudgetReached(
                f"Hard objective-evaluation budget {self.phase_hard_limit} reached"
            )
        values = _sorted_values(raw_values)
        objective, _, counts = self.inspect(values)
        self.total_evaluations += 1
        self.phase_evaluations += 1
        if objective < self.best_objective:
            self.best_objective = objective
            self.best_values = values
            self.best_counts = counts
        self.history.append(
            {
                "method": self.method,
                "run": self.run,
                "phase": self.phase,
                "record_type": "objective_evaluation",
                "objective_evaluations_cumulative": self.total_evaluations,
                "phase_objective_evaluations": self.phase_evaluations,
                "current_objective": objective,
                "current_c_values": _value_string(values),
                "best_objective": self.best_objective,
                "best_c_values": _value_string(self.best_values),
                "best_assignment_counts": ";".join(str(value) for value in self.best_counts),
                "active_best_components": sum(value > 0 for value in self.best_counts),
                "underlying_L_i_c_calculations_cumulative": len(self.scalar_cache)
                * self.positions.shape[0],
                "iteration": "",
                "assignments_changed": "",
                "stop_reason": "",
            }
        )
        if self.run == "main" and self.phase == "DIRECT" and self.phase_evaluations % 1000 == 0:
            print(
                f"  DIRECT {self.phase_evaluations}/{DIRECT_MAXFUN}: "
                f"best J={self.best_objective:.6e}, active={sum(v > 0 for v in self.best_counts)}",
                flush=True,
            )
        return objective

    def iteration_row(
        self, raw_values: Any, iteration: int, assignments_changed: bool | str
    ) -> None:
        values = _sorted_values(raw_values)
        objective, _, counts = self.inspect(values)
        self.history.append(
            {
                "method": self.method,
                "run": self.run,
                "phase": self.phase,
                "record_type": "optimizer_iteration",
                "objective_evaluations_cumulative": self.total_evaluations,
                "phase_objective_evaluations": self.phase_evaluations,
                "current_objective": objective,
                "current_c_values": _value_string(values),
                "best_objective": self.best_objective,
                "best_c_values": _value_string(self.best_values),
                "best_assignment_counts": ";".join(str(value) for value in counts),
                "active_best_components": sum(value > 0 for value in counts),
                "underlying_L_i_c_calculations_cumulative": len(self.scalar_cache)
                * self.positions.shape[0],
                "iteration": iteration,
                "assignments_changed": assignments_changed,
                "stop_reason": "",
            }
        )


def _run_powell(
    evaluator: JointObjective,
    initial: tuple[float, ...],
    maxfev: int,
) -> tuple[tuple[float, ...], Any, float, int]:
    evaluator.set_phase("Powell")
    previous_assignments: torch.Tensor | None = None
    iterations = 0

    def callback(raw_values: np.ndarray) -> None:
        nonlocal previous_assignments, iterations
        iterations += 1
        values = _sorted_values(raw_values)
        _, assignments, _ = evaluator.inspect(values)
        changed: bool | str = "" if previous_assignments is None else bool(
            torch.any(assignments != previous_assignments).item()
        )
        evaluator.iteration_row(values, iterations, changed)
        previous_assignments = assignments.clone()

    start = time.perf_counter()
    result = minimize(
        evaluator,
        np.asarray(initial, dtype=float),
        method="Powell",
        bounds=[(LOWER, UPPER)] * M,
        callback=callback,
        options={
            "xtol": 1e-12,
            "ftol": 1e-15,
            "maxiter": 400,
            "maxfev": maxfev,
            "disp": False,
        },
    )
    return evaluator.best_values, result, time.perf_counter() - start, iterations


def _assignment_stability(
    values: tuple[float, ...], positions: torch.Tensor
) -> tuple[bool, torch.Tensor, float, int, int]:
    """Test local assignments without the infeasible 2**16 corner sweep."""
    matrix = torch.stack(
        [_candidate_losses(value, positions).detach() for value in values], dim=1
    )
    sorted_losses, sorted_indices = torch.sort(matrix, dim=1)
    assignments = sorted_indices[:, 0]
    minimum_margin = float((sorted_losses[:, 1] - sorted_losses[:, 0]).min().item())
    perturbations: list[np.ndarray] = []
    for component in range(M):
        for direction in (-1.0, 1.0):
            delta = np.zeros(M)
            delta[component] = direction * STABILITY_SCALE
            perturbations.append(delta)
    structured = (
        np.ones(M),
        -np.ones(M),
        np.asarray([1 if index % 2 == 0 else -1 for index in range(M)]),
        np.asarray([-1 if index % 2 == 0 else 1 for index in range(M)]),
        np.asarray([1 if index < M // 2 else -1 for index in range(M)]),
        np.asarray([-1 if index < M // 2 else 1 for index in range(M)]),
    )
    perturbations.extend(STABILITY_SCALE * pattern for pattern in structured)
    generator = np.random.default_rng(SEED + 160)
    perturbations.extend(
        STABILITY_SCALE * generator.choice((-1.0, 1.0), size=M) for _ in range(16)
    )
    maximum_changes = 0
    tested = 0
    for delta in perturbations:
        perturbed = tuple(value + float(change) for value, change in zip(values, delta))
        if not (
            all(LOWER <= value <= UPPER for value in perturbed)
            and all(perturbed[index] < perturbed[index + 1] for index in range(M - 1))
        ):
            continue
        tested += 1
        perturbed_matrix = torch.stack(
            [_candidate_losses(value, positions).detach() for value in perturbed], dim=1
        )
        perturbed_assignments = torch.argmin(perturbed_matrix, dim=1)
        maximum_changes = max(
            maximum_changes,
            int((perturbed_assignments != assignments).sum().item()),
        )
    stable = maximum_changes == 0
    return stable, assignments, minimum_margin, maximum_changes, tested


def _fixed_assignment_objective(
    parameters: torch.Tensor,
    positions: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros((), dtype=DTYPE)
    for component in range(M):
        mask = assignments == component
        if bool(mask.any()):
            total = total + _candidate_losses(
                parameters[component], positions[mask]
            ).sum()
    return total / positions.shape[0]


def _active_fixed_assignment_objective(
    active_parameters: torch.Tensor,
    active_components: tuple[int, ...],
    positions: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    """Fixed objective over active coordinates; unused components stay frozen."""
    total = torch.zeros((), dtype=DTYPE)
    for parameter, component in zip(active_parameters, active_components):
        mask = assignments == component
        total = total + _candidate_losses(parameter, positions[mask]).sum()
    return total / positions.shape[0]


def _fixed_objective_float(
    values: tuple[float, ...], positions: torch.Tensor, assignments: torch.Tensor
) -> float:
    with torch.no_grad():
        parameters = torch.tensor(values, dtype=DTYPE)
        return float(
            _fixed_assignment_objective(parameters, positions, assignments).item()
        )


def _joint_newton(
    initial: tuple[float, ...],
    positions: torch.Tensor,
    assignments: torch.Tensor,
    evaluation_offset: int,
) -> NewtonResult:
    values = initial
    active_components = tuple(
        component
        for component, count in enumerate(_counts(assignments))
        if count > 0
    )
    rows: list[dict[str, Any]] = []
    objective_evaluations = 1
    accepted_iterations = 0
    stop_reason = "maximum_iterations"
    minimum_eigenvalue = float("nan")
    maximum_offdiagonal = float("nan")
    best_objective = _fixed_objective_float(values, positions, assignments)
    start = time.perf_counter()
    for iteration in range(1, NEWTON_MAX_ITERATIONS + 1):
        # Frozen assignments make the local objective exactly separable. Build
        # each scalar derivative independently rather than traversing a large
        # 16-branch autograd graph once per Hessian row.
        objective_terms: list[torch.Tensor] = []
        gradient_terms: list[torch.Tensor] = []
        hessian_terms: list[torch.Tensor] = []
        for component in active_components:
            parameter = torch.tensor(
                values[component], dtype=DTYPE, requires_grad=True
            )
            mask = assignments == component
            term = _candidate_losses(parameter, positions[mask]).sum()
            term = term / positions.shape[0]
            first = torch.autograd.grad(term, parameter, create_graph=True)[0]
            second = torch.autograd.grad(first, parameter)[0]
            objective_terms.append(term.detach())
            gradient_terms.append(first.detach())
            hessian_terms.append(second.detach())
        objective = torch.stack(objective_terms).sum()
        gradient = torch.stack(gradient_terms)
        hessian = torch.diag(torch.stack(hessian_terms))
        objective_evaluations += 1
        hessian_detached = hessian.detach()
        gradient_detached = gradient.detach()
        eigenvalues = torch.linalg.eigvalsh(hessian_detached)
        minimum_eigenvalue = float(eigenvalues.min().item())
        offdiagonal = hessian_detached - torch.diag(torch.diag(hessian_detached))
        maximum_offdiagonal = float(torch.abs(offdiagonal).max().item())
        current_objective = float(objective.detach().item())
        proposed = torch.full(
            (len(active_components),), torch.nan, dtype=DTYPE
        )
        accepted_delta = torch.zeros(len(active_components), dtype=DTYPE)
        backtracks = 0
        if minimum_eigenvalue <= 0.0 or not bool(torch.isfinite(hessian_detached).all()):
            stop_reason = "hessian_not_positive_definite"
        else:
            proposed = -torch.linalg.solve(hessian_detached, gradient_detached)
            if not bool(torch.isfinite(proposed).all()):
                stop_reason = "invalid_newton_step"
            else:
                accepted = False
                scale = 1.0
                for backtracks in range(NEWTON_MAX_BACKTRACKS + 1):
                    current_tensor = torch.tensor(values, dtype=DTYPE)
                    trial_tensor = current_tensor.clone()
                    trial_tensor[list(active_components)] += scale * proposed
                    trial = tuple(float(value.item()) for value in trial_tensor)
                    admissible = (
                        all(LOWER <= value <= UPPER for value in trial)
                        and all(
                            trial[index] < trial[index + 1]
                            for index in range(M - 1)
                        )
                    )
                    if admissible and trial != values:
                        trial_objective = _fixed_objective_float(
                            trial, positions, assignments
                        )
                        objective_evaluations += 1
                        if trial_objective < current_objective:
                            accepted = True
                            accepted_delta = trial_tensor - current_tensor
                            values = trial
                            best_objective = trial_objective
                            accepted_iterations += 1
                            stop_reason = ""
                            break
                    scale *= 0.5
                if not accepted:
                    if all(
                        float(values[component]) + float(proposed[index].item())
                        == values[component]
                        for index, component in enumerate(active_components)
                    ):
                        stop_reason = "numerically_negligible_update"
                    else:
                        stop_reason = "no_float64_decreasing_step"
        counts = _counts(assignments)
        rows.append(
            {
                "method": "direct_powell_newton",
                "run": "main",
                "phase": "Newton",
                "record_type": "newton_iteration",
                "objective_evaluations_cumulative": evaluation_offset
                + objective_evaluations,
                "phase_objective_evaluations": objective_evaluations,
                "current_objective": best_objective,
                "current_c_values": _value_string(values),
                "best_objective": best_objective,
                "best_c_values": _value_string(values),
                "best_assignment_counts": ";".join(str(value) for value in counts),
                "active_best_components": sum(value > 0 for value in counts),
                "underlying_L_i_c_calculations_cumulative": objective_evaluations
                * positions.shape[0],
                "iteration": iteration,
                "assignments_changed": False,
                "gradient_norm": float(torch.linalg.vector_norm(gradient_detached).item()),
                "minimum_hessian_eigenvalue": minimum_eigenvalue,
                "maximum_hessian_offdiagonal": maximum_offdiagonal,
                "proposed_step_norm": float(torch.linalg.vector_norm(proposed).item()),
                "accepted_step_norm": float(
                    torch.linalg.vector_norm(accepted_delta).item()
                ),
                "backtracks": backtracks,
                "polished_active_components": len(active_components),
                "stop_reason": stop_reason,
            }
        )
        if stop_reason:
            break
    return NewtonResult(
        values=values,
        objective=best_objective,
        objective_evaluations=objective_evaluations,
        underlying_loss_calculations=objective_evaluations * positions.shape[0],
        accepted_iterations=accepted_iterations,
        stop_reason=stop_reason,
        minimum_hessian_eigenvalue=minimum_eigenvalue,
        maximum_hessian_offdiagonal=maximum_offdiagonal,
        runtime_seconds=time.perf_counter() - start,
        history=rows,
    )


def _posthoc_component_match(
    values: tuple[float, ...]
) -> tuple[dict[int, int], tuple[float, ...], tuple[float, ...]]:
    references = np.asarray(
        [_posthoc_metadata(group_id)[2] for group_id in range(M)]
    )
    learned = np.asarray(values)
    group_indices, component_indices = linear_sum_assignment(
        np.abs(references[:, None] - learned[None, :])
    )
    mapping = {
        int(group): int(component)
        for group, component in zip(group_indices, component_indices)
    }
    component_references = [float("nan")] * M
    component_errors = [float("nan")] * M
    for group, component in mapping.items():
        component_references[component] = float(references[group])
        component_errors[component] = abs(values[component] - float(references[group]))
    return mapping, tuple(component_references), tuple(component_errors)


def _posthoc_assignment_mapping(
    values: tuple[float, ...], fit_data: DynamicsData
) -> dict[int, int]:
    matrix = torch.stack(
        [_candidate_losses(value, fit_data.positions).detach() for value in values], dim=1
    )
    assignments = torch.argmin(matrix, dim=1)
    confusion = np.zeros((M, M), dtype=np.int64)
    for group in range(M):
        selected = assignments[fit_data.group_ids == group]
        for component in range(M):
            confusion[group, component] = int((selected == component).sum().item())
    group_indices, component_indices = linear_sum_assignment(-confusion)
    return {
        int(group): int(component)
        for group, component in zip(group_indices, component_indices)
    }


def _evaluate_stage(
    result: StageResult,
    datasets: dict[str, DynamicsData],
    save_assignments: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = result.values
    _, references, c_errors = _posthoc_component_match(values)
    assignment_mapping = _posthoc_assignment_mapping(values, datasets["fit"])
    aggregate: dict[str, Any] = {
        "method": result.method,
        "stage": result.stage,
        **_summary_fields("learned_c", values),
        **_summary_fields("matched_reference_c", references),
        "absolute_c_errors": _value_string(c_errors),
        "worst_absolute_c_error": max(c_errors),
        **result.costs,
        **result.details,
    }
    assignment_rows: list[dict[str, Any]] = []
    for split, data in datasets.items():
        matrix = torch.stack(
            [_candidate_losses(value, data.positions).detach() for value in values],
            dim=1,
        )
        assigned_losses, assignments = torch.min(matrix, dim=1)
        counts = _counts(assignments)
        correct = 0
        represented_groups: set[int] = set()
        split_h1: list[float] = []
        split_mse: list[float] = []
        split_rollout: list[float] = []
        for trajectory in range(data.positions.shape[0]):
            group_id = int(data.group_ids[trajectory].item())
            expected_component = assignment_mapping[group_id]
            assigned_component = int(assignments[trajectory].item())
            assignment_correct = assigned_component == expected_component
            correct += int(assignment_correct)
            if assignment_correct:
                represented_groups.add(group_id)
            group_name, omega, true_c, future_steps = _posthoc_metadata(group_id)
            position = data.positions[trajectory]
            assigned_c = values[assigned_component]
            target = position[P : P + future_steps]
            h1_error = (
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
            max_h1 = float(torch.abs(h1_error).max().item())
            rollout_mse = float(torch.mean(rollout_error**2).item())
            max_rollout = float(torch.abs(rollout_error).max().item())
            split_h1.append(max_h1)
            split_mse.append(rollout_mse)
            split_rollout.append(max_rollout)
            if save_assignments:
                assignment_rows.append(
                    {
                        "method": result.method,
                        "stage": result.stage,
                        "split": split,
                        "trajectory_index": trajectory,
                        "true_group_posthoc": group_name,
                        "true_omega_posthoc": omega,
                        "true_c_posthoc": true_c,
                        "assigned_component": assigned_component + 1,
                        "expected_component_posthoc": expected_component + 1,
                        "assigned_c": assigned_c,
                        "assignment_correct_posthoc": assignment_correct,
                        "assigned_prefix_loss": float(assigned_losses[trajectory].item()),
                        "component_prefix_losses": _value_string(
                            [float(loss.item()) for loss in matrix[trajectory]]
                        ),
                        "max_h1_error_10_period_evaluation": max_h1,
                        "rollout_10_period_mse": rollout_mse,
                        "max_abs_10_period_rollout_error": max_rollout,
                    }
                )
        aggregate[f"{split}_assignment_counts"] = ";".join(
            str(value) for value in counts
        )
        aggregate[f"{split}_active_components"] = sum(value > 0 for value in counts)
        aggregate[f"{split}_assignment_accuracy"] = correct / data.positions.shape[0]
        aggregate[f"{split}_all_hidden_groups_represented"] = (
            len(represented_groups) == M
        )
        aggregate[f"worst_{split}_prefix_loss"] = float(assigned_losses.max().item())
        aggregate[f"worst_{split}_h1_error"] = max(split_h1)
        aggregate[f"worst_{split}_10_period_mse"] = max(split_mse)
        aggregate[f"worst_{split}_rollout_error"] = max(split_rollout)
    aggregate["primary_1e-20_gate_passed"] = (
        float(aggregate["worst_fit_prefix_loss"]) < PRIMARY_GATE
        and float(aggregate["worst_validation_prefix_loss"]) < PRIMARY_GATE
    )
    aggregate["secondary_1e-24_milestone_passed"] = (
        float(aggregate["worst_fit_prefix_loss"]) < SECONDARY_GATE
        and float(aggregate["worst_validation_prefix_loss"]) < SECONDARY_GATE
    )
    return aggregate, assignment_rows


def _costs(
    global_evaluations: int = 0,
    local_evaluations: int = 0,
    newton_evaluations: int = 0,
    global_underlying: int = 0,
    local_underlying: int = 0,
    newton_underlying: int = 0,
    global_runtime: float = 0.0,
    local_runtime: float = 0.0,
    newton_runtime: float = 0.0,
) -> dict[str, Any]:
    return {
        "global_search_objective_evaluations": global_evaluations,
        "local_refinement_objective_evaluations": local_evaluations,
        "newton_objective_evaluations": newton_evaluations,
        "total_objective_evaluations": global_evaluations
        + local_evaluations
        + newton_evaluations,
        "global_underlying_L_i_c_calculations": global_underlying,
        "local_underlying_L_i_c_calculations": local_underlying,
        "newton_underlying_L_i_c_calculations": newton_underlying,
        "total_underlying_L_i_c_calculations": global_underlying
        + local_underlying
        + newton_underlying,
        "global_runtime_seconds": global_runtime,
        "local_runtime_seconds": local_runtime,
        "newton_runtime_seconds": newton_runtime,
        "total_runtime_seconds": global_runtime + local_runtime + newton_runtime,
    }


def _plot_progress(history: list[dict[str, Any]], direct_boundary: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    series = (
        ("direct_powell_newton", "main", "DIRECT → Powell → Newton", "tab:blue"),
        ("powell_without_direct", "clustered_near_zero", "Powell: near zero", "tab:orange"),
        (
            "powell_without_direct",
            "clustered_near_left_side",
            "Powell: one side",
            "tab:red",
        ),
        (
            "powell_without_direct",
            "evenly_spaced_generic",
            "Powell: generic spacing",
            "tab:green",
        ),
    )
    for method, run, label, color in series:
        selected = [
            row
            for row in history
            if row.get("method") == method
            and row.get("run") == run
            and row.get("record_type")
            in {"objective_evaluation", "newton_iteration"}
        ]
        if not selected:
            continue
        x = [int(row["objective_evaluations_cumulative"]) for row in selected]
        y = [max(float(row["best_objective"]), 1e-32) for row in selected]
        axis.plot(x, y, linewidth=1.35, color=color, label=label)
    axis.axvline(
        direct_boundary,
        color="tab:blue",
        linestyle="--",
        alpha=0.7,
        label="main Powell begins",
    )
    axis.set_yscale("log")
    axis.set_xlabel("joint/fixed-assignment objective evaluations")
    axis.set_ylabel("best prefix objective")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "best_objective_vs_evaluations.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fit = _make_data(FIT_PER_GROUP, TRAIN_DATA_SEED + 1_600_000)
    validation = _make_data(VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1_601_000)
    test = _make_data(TEST_PER_GROUP, TEST_DATA_SEED + 1_602_000)
    datasets = {"fit": fit, "validation": validation, "test": test}
    history: list[dict[str, Any]] = []
    stage_results: list[StageResult] = []

    print("Method A1: unbiased 16D DIRECT (hard 10,000-call budget)", flush=True)
    evaluator = JointObjective(fit.positions, "direct_powell_newton", "main")
    evaluator.set_phase("DIRECT", hard_limit=DIRECT_MAXFUN)
    direct_start = time.perf_counter()
    try:
        direct_result = direct(
            evaluator,
            bounds=[(LOWER, UPPER)] * M,
            maxfun=DIRECT_MAXFUN,
            maxiter=100_000,
            locally_biased=False,
            vol_tol=1e-30,
            len_tol=1e-12,
        )
        direct_success = bool(direct_result.success)
        direct_termination = str(direct_result.message)
    except (DirectBudgetReached, SystemError):
        if evaluator.phase_evaluations < DIRECT_MAXFUN:
            raise
        direct_success = False
        direct_termination = (
            f"hard objective-evaluation budget {DIRECT_MAXFUN} reached"
        )
    direct_runtime = time.perf_counter() - direct_start
    direct_values = evaluator.best_values
    direct_best_counts = evaluator.best_counts
    direct_evaluations = evaluator.phase_evaluations
    direct_unique_scalars = evaluator.phase_new_scalars
    history.extend(evaluator.history)
    stage_results.append(
        StageResult(
            "direct_powell_newton",
            "after_DIRECT",
            direct_values,
            _costs(
                global_evaluations=direct_evaluations,
                global_underlying=direct_unique_scalars * fit.positions.shape[0],
                global_runtime=direct_runtime,
            ),
            {
                "DIRECT_requested_maxfun": DIRECT_MAXFUN,
                "DIRECT_actual_evaluations": direct_evaluations,
                "DIRECT_success": direct_success,
                "DIRECT_termination": direct_termination,
                "DIRECT_unique_scalar_c_values": direct_unique_scalars,
            },
        )
    )

    print("Method A2: bounded joint 16D Powell", flush=True)
    history_before = len(evaluator.history)
    powell_values, powell_result, powell_runtime, powell_iterations = _run_powell(
        evaluator, direct_values, POWELL_MAXFEV
    )
    history.extend(evaluator.history[history_before:])
    powell_evaluations = evaluator.phase_evaluations
    powell_unique_scalars = evaluator.phase_new_scalars
    stable, assignments, margin, changes, perturbations_tested = _assignment_stability(
        powell_values, fit.positions
    )
    powell_counts = _counts(assignments)
    stage_results.append(
        StageResult(
            "direct_powell_newton",
            "after_joint_Powell_before_Newton",
            powell_values,
            _costs(
                global_evaluations=direct_evaluations,
                local_evaluations=powell_evaluations,
                global_underlying=direct_unique_scalars * fit.positions.shape[0],
                local_underlying=powell_unique_scalars * fit.positions.shape[0],
                global_runtime=direct_runtime,
                local_runtime=powell_runtime,
            ),
            {
                "Powell_requested_maxfev": POWELL_MAXFEV,
                "Powell_iterations": powell_iterations,
                "Powell_success": bool(powell_result.success),
                "Powell_termination": str(powell_result.message),
                "assignments_stable_before_Newton": stable,
                "all_components_active_before_Newton": all(
                    value > 0 for value in powell_counts
                ),
                "stability_perturbations_tested": perturbations_tested,
                "maximum_assignment_changes_under_perturbation": changes,
                "minimum_assignment_margin": margin,
                "active_components_before_Newton": sum(v > 0 for v in powell_counts),
            },
        )
    )

    print("Method A3: frozen-assignment safeguarded Newton", flush=True)
    if stable:
        newton = _joint_newton(
            powell_values,
            fit.positions,
            assignments,
            direct_evaluations + powell_evaluations,
        )
    else:
        newton = NewtonResult(
            values=powell_values,
            objective=evaluator.best_objective,
            objective_evaluations=0,
            underlying_loss_calculations=0,
            accepted_iterations=0,
            stop_reason="skipped_assignments_unstable",
            minimum_hessian_eigenvalue=float("nan"),
            maximum_hessian_offdiagonal=float("nan"),
            runtime_seconds=0.0,
            history=[],
        )
    history.extend(newton.history)
    stage_results.append(
        StageResult(
            "direct_powell_newton",
            "after_Newton",
            newton.values,
            _costs(
                global_evaluations=direct_evaluations,
                local_evaluations=powell_evaluations,
                newton_evaluations=newton.objective_evaluations,
                global_underlying=direct_unique_scalars * fit.positions.shape[0],
                local_underlying=powell_unique_scalars * fit.positions.shape[0],
                newton_underlying=newton.underlying_loss_calculations,
                global_runtime=direct_runtime,
                local_runtime=powell_runtime,
                newton_runtime=newton.runtime_seconds,
            ),
            {
                "Newton_accepted_iterations": newton.accepted_iterations,
                "Newton_polished_active_components": sum(
                    value > 0 for value in powell_counts
                ) if stable else 0,
                "Newton_stop_reason": newton.stop_reason,
                "Newton_minimum_hessian_eigenvalue": newton.minimum_hessian_eigenvalue,
                "Newton_maximum_hessian_offdiagonal": newton.maximum_hessian_offdiagonal,
                "fixed_assignment_objective_separable": (
                    newton.maximum_hessian_offdiagonal == 0.0
                ),
            },
        )
    )

    print("Controls: bounded joint Powell without DIRECT", flush=True)
    for run_name, initial in CONTROL_STARTS:
        print(f"  {run_name}", flush=True)
        control = JointObjective(fit.positions, "powell_without_direct", run_name)
        values, result, runtime, iterations = _run_powell(
            control, initial, CONTROL_MAXFEV
        )
        history.extend(control.history)
        stage_results.append(
            StageResult(
                "powell_without_direct",
                run_name,
                values,
                _costs(
                    local_evaluations=control.phase_evaluations,
                    local_underlying=control.phase_new_scalars
                    * fit.positions.shape[0],
                    local_runtime=runtime,
                ),
                {
                    "initial_values": _value_string(initial),
                    "Powell_requested_maxfev": CONTROL_MAXFEV,
                    "Powell_iterations": iterations,
                    "Powell_success": bool(result.success),
                    "Powell_termination": str(result.message),
                },
            )
        )

    print("Post-hoc evaluation; no refitting", flush=True)
    summary: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    final_stages = {
        ("direct_powell_newton", "after_Newton"),
        *(("powell_without_direct", name) for name, _ in CONTROL_STARTS),
    }
    for result in stage_results:
        save = (result.method, result.stage) in final_stages
        aggregate, rows = _evaluate_stage(result, datasets, save)
        summary.append(aggregate)
        assignment_rows.extend(rows)

    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "search_history.csv", history)
    _write_csv(OUTPUT_DIR / "component_assignments.csv", assignment_rows)
    _plot_progress(history, direct_evaluations)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "DIRECT_evaluations": direct_evaluations,
            "DIRECT_active": sum(v > 0 for v in direct_best_counts),
            "Powell_active": sum(v > 0 for v in powell_counts),
            "final_values": newton.values,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
