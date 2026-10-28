"""Recover four shared SHO dynamics with a genuine four-dimensional search."""

from __future__ import annotations

import itertools
import math
import os
import time
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
from scipy.optimize import direct, minimize, minimize_scalar

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

OUTPUT_DIR = Path("outputs/sho/stage1B/four_dynamics_multidimensional_search")
FIT_PER_GROUP = 32
VALIDATION_PER_GROUP = 16
TEST_PER_GROUP = 32
M = 4
DIRECT_MAXFUN = int(os.environ.get("FOUR_DYNAMICS_DIRECT_MAXFUN", "5000"))
POWELL_MAXFEV = int(os.environ.get("FOUR_DYNAMICS_POWELL_MAXFEV", "5000"))
CONTROL_MAXFEV = int(os.environ.get("FOUR_DYNAMICS_CONTROL_MAXFEV", "1500"))
COARSE_POINTS = 16
BRENT_XATOL = 1e-14
MAX_BRENT_SWEEPS = 12
NEWTON_MAX_ITERATIONS = 50
NEWTON_MAX_BACKTRACKS = 60
PRIMARY_GATE = 1e-20
SECONDARY_GATE = 1e-24
STABILITY_SCALE = 1e-6
CONTROL_STARTS = (
    ("clustered_near_zero", (-0.15, -0.05, 0.05, 0.15)),
    ("clustered_high_c", (1.55, 1.60, 1.65, 1.70)),
    ("evenly_spaced_generic", (-1.50, -0.50, 0.50, 1.50)),
)


@dataclass(frozen=True)
class FourDynamicsData:
    positions: torch.Tensor
    group_ids: torch.Tensor


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


@dataclass(frozen=True)
class StageResult:
    method: str
    stage: str
    values: tuple[float, ...]
    costs: dict[str, Any]
    details: dict[str, Any]


class DirectBudgetReached(RuntimeError):
    """Hard stop for scipy DIRECT's otherwise approximate maxfun budget."""


def _make_four_dynamics_data(per_group: int, seed: int) -> FourDynamicsData:
    """Generate data; hidden frequency values are confined to this function."""
    generator = torch.Generator().manual_seed(seed)
    group_ids = torch.arange(M, dtype=torch.long).repeat_interleave(per_group)
    generating_frequencies = group_ids.to(DTYPE) + 1.0
    initial_conditions = 2.0 * torch.rand(
        M * per_group, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    slowest_frequency = float(generating_frequencies.min().item())
    future_steps = math.ceil(10.0 * math.tau / (slowest_frequency * float(DELTA_T)))
    times = torch.arange(P + future_steps + 1, dtype=DTYPE).unsqueeze(0)
    times = times.expand(M * per_group, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, generating_frequencies, times)
    permutation = torch.randperm(M * per_group, generator=generator)
    return FourDynamicsData(positions[permutation], group_ids[permutation])


def _posthoc_metadata(group_id: int) -> tuple[str, float, float, int]:
    """Return hidden metadata only after all searches have finished."""
    omega = float(group_id + 1)
    reference_c = 2.0 * math.cos(omega * float(DELTA_T))
    future_steps = math.ceil(10.0 * math.tau / (omega * float(DELTA_T)))
    return f"omega_{group_id + 1}", omega, reference_c, future_steps


def _candidate_losses(value: float | torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    scalar = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=DTYPE)
    return _losses(scalar.expand(positions.shape[0]), positions[:, :P])


def _sorted_values(values: Any) -> tuple[float, ...]:
    return tuple(sorted(float(value) for value in np.asarray(values, dtype=float)))


def _fields(prefix: str, values: tuple[float, ...] | list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"{prefix}_values": ";".join(f"{value:.17g}" for value in values)
    }
    for index in range(M):
        result[f"{prefix}_{index + 1}"] = values[index]
    return result


def _counts(assignments: torch.Tensor) -> tuple[int, ...]:
    return tuple(int((assignments == index).sum().item()) for index in range(M))


class JointObjective:
    """Cached J(theta) evaluator with per-evaluation best-so-far history."""

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

    def inspect(self, values: tuple[float, ...]) -> tuple[float, torch.Tensor, tuple[int, ...]]:
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
        objective, assignments, counts = self.inspect(values)
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
                **_fields("current_c", values),
                "best_objective": self.best_objective,
                **_fields("best_c", self.best_values),
                "best_assignment_counts": ";".join(str(value) for value in self.best_counts),
                "active_best_components": sum(value > 0 for value in self.best_counts),
                "underlying_L_i_c_calculations_cumulative": len(self.scalar_cache)
                * self.positions.shape[0],
                "iteration": "",
                "assignments_changed": "",
                "stop_reason": "",
            }
        )
        return objective

    def iteration_row(
        self,
        raw_values: Any,
        iteration: int,
        assignments_changed: bool | str,
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
                **_fields("current_c", values),
                "best_objective": self.best_objective,
                **_fields("best_c", self.best_values),
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
            "maxiter": 200,
            "maxfev": maxfev,
            "disp": False,
        },
    )
    runtime = time.perf_counter() - start
    return evaluator.best_values, result, runtime, iterations


def _assignment_stability(
    values: tuple[float, ...], positions: torch.Tensor
) -> tuple[bool, torch.Tensor, float, int]:
    matrix = torch.stack(
        [_candidate_losses(value, positions).detach() for value in values], dim=1
    )
    sorted_losses, sorted_indices = torch.sort(matrix, dim=1)
    assignments = sorted_indices[:, 0]
    minimum_margin = float((sorted_losses[:, 1] - sorted_losses[:, 0]).min().item())
    maximum_changes = 0
    perturbations = []
    for component in range(M):
        for direction in (-1.0, 1.0):
            delta = [0.0] * M
            delta[component] = direction * STABILITY_SCALE
            perturbations.append(delta)
    perturbations.extend(
        [list(signs) for signs in itertools.product((-STABILITY_SCALE, STABILITY_SCALE), repeat=M)]
    )
    for delta in perturbations:
        perturbed = tuple(value + change for value, change in zip(values, delta))
        if not (
            all(LOWER <= value <= UPPER for value in perturbed)
            and all(perturbed[index] < perturbed[index + 1] for index in range(M - 1))
        ):
            continue
        perturbed_matrix = torch.stack(
            [_candidate_losses(value, positions).detach() for value in perturbed], dim=1
        )
        perturbed_assignments = torch.argmin(perturbed_matrix, dim=1)
        maximum_changes = max(
            maximum_changes,
            int((perturbed_assignments != assignments).sum().item()),
        )
    return maximum_changes == 0, assignments, minimum_margin, maximum_changes


def _fixed_assignment_objective(
    parameters: torch.Tensor,
    positions: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    total = torch.zeros((), dtype=DTYPE)
    trajectory_count = positions.shape[0]
    for component in range(M):
        mask = assignments == component
        if bool(mask.any()):
            total = total + _candidate_losses(parameters[component], positions[mask]).sum()
    return total / trajectory_count


def _fixed_objective_float(
    values: tuple[float, ...], positions: torch.Tensor, assignments: torch.Tensor
) -> float:
    with torch.no_grad():
        parameters = torch.tensor(values, dtype=DTYPE)
        return float(_fixed_assignment_objective(parameters, positions, assignments).item())


def _joint_newton(
    method: str,
    initial: tuple[float, ...],
    positions: torch.Tensor,
    assignments: torch.Tensor,
    evaluation_offset: int,
) -> NewtonResult:
    values = initial
    rows: list[dict[str, Any]] = []
    objective_evaluations = 0
    accepted_iterations = 0
    stop_reason = "maximum_iterations"
    minimum_eigenvalue = float("nan")
    maximum_offdiagonal = float("nan")
    best_objective = _fixed_objective_float(values, positions, assignments)
    objective_evaluations += 1
    start = time.perf_counter()
    for iteration in range(1, NEWTON_MAX_ITERATIONS + 1):
        parameters = torch.tensor(values, dtype=DTYPE, requires_grad=True)
        objective = _fixed_assignment_objective(parameters, positions, assignments)
        gradient = torch.autograd.grad(objective, parameters, create_graph=True)[0]
        hessian = torch.stack(
            [
                torch.autograd.grad(gradient[index], parameters, retain_graph=True)[0]
                for index in range(M)
            ]
        )
        objective_evaluations += 1
        hessian_detached = hessian.detach()
        eigenvalues = torch.linalg.eigvalsh(hessian_detached)
        minimum_eigenvalue = float(eigenvalues.min().item())
        offdiagonal = hessian_detached - torch.diag(torch.diag(hessian_detached))
        maximum_offdiagonal = float(torch.abs(offdiagonal).max().item())
        gradient_detached = gradient.detach()
        current_objective = float(objective.detach().item())
        proposed = torch.full((M,), torch.nan, dtype=DTYPE)
        accepted_delta = torch.zeros(M, dtype=DTYPE)
        backtracks = 0
        if minimum_eigenvalue <= 0.0 or not bool(torch.isfinite(hessian_detached).all()):
            stop_reason = "hessian_not_positive_definite"
        else:
            proposed = -torch.linalg.solve(hessian_detached, gradient_detached)
            if not bool(torch.isfinite(proposed).all()):
                stop_reason = "invalid_newton_step"
            else:
                scale = 1.0
                accepted = False
                for backtracks in range(NEWTON_MAX_BACKTRACKS + 1):
                    trial_tensor = torch.tensor(values, dtype=DTYPE) + scale * proposed
                    trial = tuple(float(value.item()) for value in trial_tensor)
                    admissible = (
                        all(LOWER <= value <= UPPER for value in trial)
                        and all(trial[index] < trial[index + 1] for index in range(M - 1))
                    )
                    if admissible and trial != values:
                        trial_objective = _fixed_objective_float(trial, positions, assignments)
                        objective_evaluations += 1
                        if trial_objective < current_objective:
                            accepted = True
                            accepted_delta = trial_tensor - torch.tensor(values, dtype=DTYPE)
                            values = trial
                            best_objective = trial_objective
                            accepted_iterations += 1
                            stop_reason = ""
                            break
                    scale *= 0.5
                if not accepted:
                    if all(float(values[index]) + float(proposed[index].item()) == values[index] for index in range(M)):
                        stop_reason = "numerically_negligible_update"
                    else:
                        stop_reason = "no_float64_decreasing_step"
        counts = _counts(assignments)
        rows.append(
            {
                "method": method,
                "run": "main",
                "phase": "Newton",
                "record_type": "newton_iteration",
                "objective_evaluations_cumulative": evaluation_offset + objective_evaluations,
                "phase_objective_evaluations": objective_evaluations,
                "current_objective": best_objective,
                **_fields("current_c", values),
                "best_objective": best_objective,
                **_fields("best_c", values),
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
                "accepted_step_norm": float(torch.linalg.vector_norm(accepted_delta).item()),
                "backtracks": backtracks,
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


class BaselineCache:
    def __init__(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.cache: dict[float, torch.Tensor] = {}
        self.phase_new_scalars = 0

    def set_phase(self) -> None:
        self.phase_new_scalars = 0

    def losses(self, value: float) -> torch.Tensor:
        key = float(value)
        if key not in self.cache:
            self.cache[key] = _candidate_losses(key, self.positions).detach()
            self.phase_new_scalars += 1
        return self.cache[key]


def _baseline_global(
    positions: torch.Tensor,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, float], ...],
    list[torch.Tensor],
    BaselineCache,
    list[dict[str, Any]],
    int,
    float,
]:
    start = time.perf_counter()
    cache = BaselineCache(positions)
    cache.set_phase()
    grid = tuple(float(value.item()) for value in torch.linspace(LOWER, UPPER, COARSE_POINTS, dtype=DTYPE))
    grid_losses = [cache.losses(value) for value in grid]
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_indices: tuple[int, ...] | None = None
    best_counts = tuple(0 for _ in range(M))
    evaluations = 0
    for indices in itertools.combinations(range(COARSE_POINTS), M):
        evaluations += 1
        matrix = torch.stack([grid_losses[index] for index in indices], dim=1)
        assigned_losses, assignments = torch.min(matrix, dim=1)
        objective = float(assigned_losses.mean().item())
        if objective < best_objective:
            best_objective = objective
            best_indices = indices
            best_counts = _counts(assignments)
        best_values = tuple(grid[index] for index in best_indices) if best_indices else tuple()
        current_values = tuple(grid[index] for index in indices)
        history.append(
            {
                "method": "coarse_brent_newton",
                "run": "main",
                "phase": "coarse_grid_combinations",
                "record_type": "objective_evaluation",
                "objective_evaluations_cumulative": evaluations,
                "phase_objective_evaluations": evaluations,
                "current_objective": objective,
                **_fields("current_c", current_values),
                "best_objective": best_objective,
                **_fields("best_c", best_values),
                "best_assignment_counts": ";".join(str(value) for value in best_counts),
                "active_best_components": sum(value > 0 for value in best_counts),
                "underlying_L_i_c_calculations_cumulative": len(cache.cache)
                * positions.shape[0],
                "iteration": "",
                "assignments_changed": "",
                "stop_reason": "",
            }
        )
    if best_indices is None:
        raise RuntimeError("The baseline coarse search found no candidate combination")
    values = tuple(grid[index] for index in best_indices)
    brackets = tuple(
        (grid[max(0, index - 1)], grid[min(COARSE_POINTS - 1, index + 1)])
        for index in best_indices
    )
    component_losses = [grid_losses[index] for index in best_indices]
    return values, brackets, component_losses, cache, history, evaluations, time.perf_counter() - start


def _baseline_alternating_brent(
    positions: torch.Tensor,
    initial: tuple[float, ...],
    brackets: tuple[tuple[float, float], ...],
    component_losses: list[torch.Tensor],
    cache: BaselineCache,
    evaluation_offset: int,
) -> tuple[tuple[float, ...], list[dict[str, Any]], int, int, int, str, float]:
    cache.set_phase()
    values = list(initial)
    matrix = torch.stack(component_losses, dim=1)
    assigned_losses, assignments = torch.min(matrix, dim=1)
    best_objective = float(assigned_losses.mean().item())
    best_values = tuple(values)
    local_evaluations = 0
    history: list[dict[str, Any]] = []
    stop_reason = "maximum_sweeps"
    start = time.perf_counter()
    for sweep in range(1, MAX_BRENT_SWEEPS + 1):
        previous_values = tuple(values)
        previous_assignments = assignments.clone()
        for component in range(M):
            fixed_losses = torch.stack(
                [component_losses[index] for index in range(M) if index != component],
                dim=1,
            ).min(dim=1).values
            lower, upper = brackets[component]
            if component > 0:
                lower = max(lower, math.nextafter(values[component - 1], math.inf))
            if component + 1 < M:
                upper = min(upper, math.nextafter(values[component + 1], -math.inf))

            def objective(candidate: float) -> float:
                nonlocal local_evaluations, best_objective, best_values
                local_evaluations += 1
                losses = cache.losses(candidate)
                value = float(torch.minimum(losses, fixed_losses).mean().item())
                trial_values = list(values)
                trial_values[component] = float(candidate)
                if value < best_objective:
                    best_objective = value
                    best_values = tuple(trial_values)
                trial_matrix = torch.stack(
                    [losses if index == component else component_losses[index] for index in range(M)],
                    dim=1,
                )
                trial_assignments = torch.argmin(trial_matrix, dim=1)
                counts = _counts(trial_assignments)
                history.append(
                    {
                        "method": "coarse_brent_newton",
                        "run": "main",
                        "phase": "alternating_bounded_brent",
                        "record_type": "objective_evaluation",
                        "objective_evaluations_cumulative": evaluation_offset + local_evaluations,
                        "phase_objective_evaluations": local_evaluations,
                        "current_objective": value,
                        **_fields("current_c", tuple(trial_values)),
                        "best_objective": best_objective,
                        **_fields("best_c", best_values),
                        "best_assignment_counts": ";".join(str(item) for item in counts),
                        "active_best_components": sum(item > 0 for item in counts),
                        "underlying_L_i_c_calculations_cumulative": len(cache.cache)
                        * positions.shape[0],
                        "iteration": sweep,
                        "assignments_changed": "",
                        "stop_reason": "",
                    }
                )
                return value

            current_objective = float(
                torch.minimum(component_losses[component], fixed_losses).mean().item()
            )
            result = minimize_scalar(
                objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": BRENT_XATOL, "maxiter": 500},
            )
            proposed_losses = cache.losses(float(result.x))
            proposed_objective = float(torch.minimum(proposed_losses, fixed_losses).mean().item())
            if proposed_objective <= current_objective:
                values[component] = float(result.x)
                component_losses[component] = proposed_losses
        matrix = torch.stack(component_losses, dim=1)
        assigned_losses, assignments = torch.min(matrix, dim=1)
        objective_value = float(assigned_losses.mean().item())
        best_objective = min(best_objective, objective_value)
        if objective_value <= best_objective:
            best_values = tuple(values)
        maximum_change = max(abs(a - b) for a, b in zip(values, previous_values))
        assignments_changed = bool(torch.any(assignments != previous_assignments).item())
        counts = _counts(assignments)
        if maximum_change <= 1e-12:
            stop_reason = "parameter_change_below_tolerance"
        history.append(
            {
                "method": "coarse_brent_newton",
                "run": "main",
                "phase": "alternating_bounded_brent",
                "record_type": "optimizer_iteration",
                "objective_evaluations_cumulative": evaluation_offset + local_evaluations,
                "phase_objective_evaluations": local_evaluations,
                "current_objective": objective_value,
                **_fields("current_c", tuple(values)),
                "best_objective": best_objective,
                **_fields("best_c", best_values),
                "best_assignment_counts": ";".join(str(item) for item in counts),
                "active_best_components": sum(item > 0 for item in counts),
                "underlying_L_i_c_calculations_cumulative": len(cache.cache)
                * positions.shape[0],
                "iteration": sweep,
                "assignments_changed": assignments_changed,
                "stop_reason": stop_reason if maximum_change <= 1e-12 else "",
            }
        )
        if maximum_change <= 1e-12:
            break
    return (
        tuple(values),
        history,
        local_evaluations,
        cache.phase_new_scalars,
        sweep,
        stop_reason,
        time.perf_counter() - start,
    )


def _posthoc_match(values: tuple[float, ...]) -> tuple[dict[int, int], tuple[float, ...], tuple[float, ...]]:
    references = tuple(_posthoc_metadata(group_id)[2] for group_id in range(M))
    best_mapping: dict[int, int] | None = None
    best_error = float("inf")
    for components in itertools.permutations(range(M)):
        error = sum(abs(values[components[group]] - references[group]) for group in range(M))
        if error < best_error:
            best_error = error
            best_mapping = {group: components[group] for group in range(M)}
    if best_mapping is None:
        raise RuntimeError("Post-hoc matching failed")
    component_references = [0.0] * M
    component_errors = [0.0] * M
    for group, component in best_mapping.items():
        component_references[component] = references[group]
        component_errors[component] = abs(values[component] - references[group])
    return best_mapping, tuple(component_references), tuple(component_errors)


def _posthoc_assignment_mapping(
    values: tuple[float, ...], fit_data: FourDynamicsData
) -> dict[int, int]:
    matrix = torch.stack(
        [_candidate_losses(value, fit_data.positions).detach() for value in values], dim=1
    )
    assignments = torch.argmin(matrix, dim=1)
    best_mapping: dict[int, int] | None = None
    best_correct = -1
    for components in itertools.permutations(range(M)):
        correct = sum(
            int(
                (
                    assignments[fit_data.group_ids == group_id]
                    == components[group_id]
                ).sum().item()
            )
            for group_id in range(M)
        )
        if correct > best_correct:
            best_correct = correct
            best_mapping = {group_id: components[group_id] for group_id in range(M)}
    if best_mapping is None:
        raise RuntimeError("Post-hoc assignment matching failed")
    return best_mapping


def _evaluate_stage(
    stage_result: StageResult,
    datasets: dict[str, FourDynamicsData],
    save_assignments: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = stage_result.values
    _, references, errors = _posthoc_match(values)
    mapping = _posthoc_assignment_mapping(values, datasets["fit"])
    aggregate: dict[str, Any] = {
        "method": stage_result.method,
        "stage": stage_result.stage,
        **_fields("learned_c", values),
        **_fields("matched_reference_c", references),
        "absolute_c_errors": ";".join(f"{error:.17g}" for error in errors),
        "worst_absolute_c_error": max(errors),
        **stage_result.costs,
        **stage_result.details,
    }
    trajectory_rows: list[dict[str, Any]] = []
    for split, data in datasets.items():
        matrix = torch.stack(
            [_candidate_losses(value, data.positions).detach() for value in values], dim=1
        )
        assigned_losses, assignments = torch.min(matrix, dim=1)
        counts = _counts(assignments)
        correct = 0
        represented = set()
        split_rows = []
        for trajectory in range(data.positions.shape[0]):
            group_id = int(data.group_ids[trajectory].item())
            expected_component = mapping[group_id]
            assigned_component = int(assignments[trajectory].item())
            assignment_correct = assigned_component == expected_component
            correct += int(assignment_correct)
            if assignment_correct:
                represented.add(group_id)
            row: dict[str, Any] = {
                "method": stage_result.method,
                "stage": stage_result.stage,
                "split": split,
                "trajectory_index": trajectory,
                "true_group_posthoc": _posthoc_metadata(group_id)[0],
                "true_omega_posthoc": _posthoc_metadata(group_id)[1],
                "true_c_posthoc": _posthoc_metadata(group_id)[2],
                "assigned_component": assigned_component + 1,
                "expected_component_posthoc": expected_component + 1,
                "assigned_c": values[assigned_component],
                "assignment_correct_posthoc": assignment_correct,
                "assigned_prefix_loss": float(assigned_losses[trajectory].item()),
                "component_prefix_losses": ";".join(
                    f"{float(loss.item()):.17g}" for loss in matrix[trajectory]
                ),
                "max_h1_future_error": "",
                "rollout_10_period_mse": "",
                "max_abs_10_period_rollout_error": "",
            }
            if split == "test":
                future_steps = _posthoc_metadata(group_id)[3]
                position = data.positions[trajectory]
                value = values[assigned_component]
                target = position[P : P + future_steps]
                direct_error = (
                    value * position[P - 1 : P + future_steps - 1]
                    - position[P - 2 : P + future_steps - 2]
                    - target
                )
                previous, current = position[P - 2], position[P - 1]
                predictions = []
                for _ in range(future_steps):
                    following = value * current - previous
                    predictions.append(following)
                    previous, current = current, following
                rollout_error = torch.stack(predictions) - target
                row["max_h1_future_error"] = float(torch.abs(direct_error).max().item())
                row["rollout_10_period_mse"] = float(torch.mean(rollout_error**2).item())
                row["max_abs_10_period_rollout_error"] = float(torch.abs(rollout_error).max().item())
            split_rows.append(row)
        aggregate[f"{split}_assignment_counts"] = ";".join(str(value) for value in counts)
        aggregate[f"{split}_active_components"] = sum(value > 0 for value in counts)
        aggregate[f"{split}_assignment_accuracy"] = correct / data.positions.shape[0]
        aggregate[f"{split}_all_hidden_groups_represented"] = len(represented) == M
        aggregate[f"worst_{split}_prefix_loss"] = float(assigned_losses.max().item())
        if save_assignments:
            trajectory_rows.extend(split_rows)
        if split == "test":
            aggregate["worst_test_h1_error"] = max(float(row["max_h1_future_error"]) for row in split_rows)
            aggregate["worst_test_10_period_mse"] = max(float(row["rollout_10_period_mse"]) for row in split_rows)
            aggregate["worst_test_max_abs_10_period_rollout_error"] = max(
                float(row["max_abs_10_period_rollout_error"]) for row in split_rows
            )
    aggregate["primary_1e-20_gate_passed"] = (
        float(aggregate["worst_fit_prefix_loss"]) < PRIMARY_GATE
        and float(aggregate["worst_validation_prefix_loss"]) < PRIMARY_GATE
    )
    aggregate["secondary_1e-24_milestone_passed"] = (
        float(aggregate["worst_fit_prefix_loss"]) < SECONDARY_GATE
        and float(aggregate["worst_validation_prefix_loss"]) < SECONDARY_GATE
    )
    return aggregate, trajectory_rows


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
        "total_objective_evaluations": global_evaluations + local_evaluations + newton_evaluations,
        "global_underlying_L_i_c_calculations": global_underlying,
        "local_underlying_L_i_c_calculations": local_underlying,
        "newton_underlying_L_i_c_calculations": newton_underlying,
        "total_underlying_L_i_c_calculations": global_underlying + local_underlying + newton_underlying,
        "global_runtime_seconds": global_runtime,
        "local_runtime_seconds": local_runtime,
        "newton_runtime_seconds": newton_runtime,
        "total_runtime_seconds": global_runtime + local_runtime + newton_runtime,
    }


def _plot_progress(history: list[dict[str, Any]], direct_boundary: int, baseline_boundary: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    series = (
        ("direct_powell_newton", "DIRECT → Powell → Newton", "tab:blue"),
        ("coarse_brent_newton", "coarse grid → alternating Brent → Newton", "tab:orange"),
    )
    for method, label, color in series:
        selected = [
            row for row in history
            if row.get("method") == method
            and row.get("run") == "main"
            and row.get("record_type") in {"objective_evaluation", "newton_iteration"}
        ]
        x = [int(row["objective_evaluations_cumulative"]) for row in selected]
        y = [max(float(row["best_objective"]), 1e-32) for row in selected]
        axis.plot(x, y, color=color, linewidth=1.5, label=label)
    axis.axvline(direct_boundary, color="tab:blue", linestyle="--", alpha=.7, label="Powell begins")
    axis.axvline(baseline_boundary, color="tab:orange", linestyle="--", alpha=.7, label="alternating Brent begins")
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
    fit = _make_four_dynamics_data(FIT_PER_GROUP, TRAIN_DATA_SEED + 1530000)
    validation = _make_four_dynamics_data(VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1531000)
    test = _make_four_dynamics_data(TEST_PER_GROUP, TEST_DATA_SEED + 1532000)
    datasets = {"fit": fit, "validation": validation, "test": test}
    search_history: list[dict[str, Any]] = []
    stage_results: list[StageResult] = []

    print("Method A1: unbiased 4D DIRECT", flush=True)
    main_objective = JointObjective(fit.positions, "direct_powell_newton", "main")
    main_objective.set_phase("DIRECT", hard_limit=DIRECT_MAXFUN)
    direct_start = time.perf_counter()
    try:
        direct_result = direct(
            main_objective,
            bounds=[(LOWER, UPPER)] * M,
            maxfun=DIRECT_MAXFUN,
            maxiter=10000,
            locally_biased=False,
            vol_tol=1e-30,
            len_tol=1e-12,
        )
        direct_success = bool(direct_result.success)
        direct_termination = str(direct_result.message)
    except (DirectBudgetReached, SystemError):
        if main_objective.phase_evaluations < DIRECT_MAXFUN:
            raise
        direct_result = None
        direct_success = False
        direct_termination = f"hard objective-evaluation budget {DIRECT_MAXFUN} reached"
    direct_runtime = time.perf_counter() - direct_start
    direct_values = main_objective.best_values
    direct_evaluations = main_objective.phase_evaluations
    direct_unique_scalars = main_objective.phase_new_scalars
    search_history.extend(main_objective.history)
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

    print("Method A2: joint 4D Powell", flush=True)
    history_before_powell = len(main_objective.history)
    powell_values, powell_result, powell_runtime, powell_iterations = _run_powell(
        main_objective, direct_values, POWELL_MAXFEV
    )
    search_history.extend(main_objective.history[history_before_powell:])
    powell_evaluations = main_objective.phase_evaluations
    powell_unique_scalars = main_objective.phase_new_scalars
    stable_a, assignments_a, margin_a, changes_a = _assignment_stability(
        powell_values, fit.positions
    )
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
                "Powell_iterations": powell_iterations,
                "Powell_success": bool(powell_result.success),
                "Powell_termination": str(powell_result.message),
                "assignments_stable_before_Newton": stable_a,
                "maximum_assignment_changes_under_perturbation": changes_a,
                "minimum_assignment_margin": margin_a,
            },
        )
    )

    print("Method A3: fixed-assignment joint Newton", flush=True)
    if stable_a:
        newton_a = _joint_newton(
            "direct_powell_newton",
            powell_values,
            fit.positions,
            assignments_a,
            direct_evaluations + powell_evaluations,
        )
    else:
        newton_a = NewtonResult(
            powell_values, main_objective.best_objective, 0, 0, 0,
            "skipped_assignments_unstable", float("nan"), float("nan"), 0.0, []
        )
    search_history.extend(newton_a.history)
    stage_results.append(
        StageResult(
            "direct_powell_newton",
            "after_Newton",
            newton_a.values,
            _costs(
                global_evaluations=direct_evaluations,
                local_evaluations=powell_evaluations,
                newton_evaluations=newton_a.objective_evaluations,
                global_underlying=direct_unique_scalars * fit.positions.shape[0],
                local_underlying=powell_unique_scalars * fit.positions.shape[0],
                newton_underlying=newton_a.underlying_loss_calculations,
                global_runtime=direct_runtime,
                local_runtime=powell_runtime,
                newton_runtime=newton_a.runtime_seconds,
            ),
            {
                "Newton_accepted_iterations": newton_a.accepted_iterations,
                "Newton_stop_reason": newton_a.stop_reason,
                "Newton_minimum_hessian_eigenvalue": newton_a.minimum_hessian_eigenvalue,
                "Newton_maximum_hessian_offdiagonal": newton_a.maximum_hessian_offdiagonal,
                "fixed_assignment_objective_separable": newton_a.maximum_hessian_offdiagonal == 0.0,
            },
        )
    )

    print("Method B: coarse combinations -> alternating Brent -> Newton", flush=True)
    (
        baseline_coarse_values,
        baseline_brackets,
        baseline_component_losses,
        baseline_cache,
        baseline_global_history,
        baseline_global_evaluations,
        baseline_global_runtime,
    ) = _baseline_global(fit.positions)
    search_history.extend(baseline_global_history)
    baseline_global_underlying = len(baseline_cache.cache) * fit.positions.shape[0]
    stage_results.append(
        StageResult(
            "coarse_brent_newton",
            "after_16_point_coarse_combinations",
            baseline_coarse_values,
            _costs(
                global_evaluations=baseline_global_evaluations,
                global_underlying=baseline_global_underlying,
                global_runtime=baseline_global_runtime,
            ),
            {
                "coarse_unique_c_values": COARSE_POINTS,
                "coarse_distinct_combinations": math.comb(COARSE_POINTS, M),
            },
        )
    )
    (
        baseline_brent_values,
        baseline_brent_history,
        baseline_local_evaluations,
        baseline_local_unique,
        baseline_sweeps,
        baseline_stop,
        baseline_local_runtime,
    ) = _baseline_alternating_brent(
        fit.positions,
        baseline_coarse_values,
        baseline_brackets,
        baseline_component_losses,
        baseline_cache,
        baseline_global_evaluations,
    )
    search_history.extend(baseline_brent_history)
    stable_b, assignments_b, margin_b, changes_b = _assignment_stability(
        baseline_brent_values, fit.positions
    )
    stage_results.append(
        StageResult(
            "coarse_brent_newton",
            "after_alternating_Brent_before_Newton",
            baseline_brent_values,
            _costs(
                global_evaluations=baseline_global_evaluations,
                local_evaluations=baseline_local_evaluations,
                global_underlying=baseline_global_underlying,
                local_underlying=baseline_local_unique * fit.positions.shape[0],
                global_runtime=baseline_global_runtime,
                local_runtime=baseline_local_runtime,
            ),
            {
                "alternating_Brent_sweeps": baseline_sweeps,
                "alternating_Brent_stop_reason": baseline_stop,
                "assignments_stable_before_Newton": stable_b,
                "maximum_assignment_changes_under_perturbation": changes_b,
                "minimum_assignment_margin": margin_b,
            },
        )
    )
    if stable_b:
        newton_b = _joint_newton(
            "coarse_brent_newton",
            baseline_brent_values,
            fit.positions,
            assignments_b,
            baseline_global_evaluations + baseline_local_evaluations,
        )
    else:
        newton_b = NewtonResult(
            baseline_brent_values, float("nan"), 0, 0, 0,
            "skipped_assignments_unstable", float("nan"), float("nan"), 0.0, []
        )
    search_history.extend(newton_b.history)
    stage_results.append(
        StageResult(
            "coarse_brent_newton",
            "after_Newton",
            newton_b.values,
            _costs(
                global_evaluations=baseline_global_evaluations,
                local_evaluations=baseline_local_evaluations,
                newton_evaluations=newton_b.objective_evaluations,
                global_underlying=baseline_global_underlying,
                local_underlying=baseline_local_unique * fit.positions.shape[0],
                newton_underlying=newton_b.underlying_loss_calculations,
                global_runtime=baseline_global_runtime,
                local_runtime=baseline_local_runtime,
                newton_runtime=newton_b.runtime_seconds,
            ),
            {
                "Newton_accepted_iterations": newton_b.accepted_iterations,
                "Newton_stop_reason": newton_b.stop_reason,
                "Newton_minimum_hessian_eigenvalue": newton_b.minimum_hessian_eigenvalue,
                "Newton_maximum_hessian_offdiagonal": newton_b.maximum_hessian_offdiagonal,
                "fixed_assignment_objective_separable": newton_b.maximum_hessian_offdiagonal == 0.0,
            },
        )
    )

    print("Method C: Powell without global search controls", flush=True)
    for run_name, initial in CONTROL_STARTS:
        evaluator = JointObjective(fit.positions, "powell_without_global", run_name)
        values, result, runtime, iterations = _run_powell(evaluator, initial, CONTROL_MAXFEV)
        search_history.extend(evaluator.history)
        stage_results.append(
            StageResult(
                "powell_without_global",
                run_name,
                values,
                _costs(
                    local_evaluations=evaluator.phase_evaluations,
                    local_underlying=evaluator.phase_new_scalars * fit.positions.shape[0],
                    local_runtime=runtime,
                ),
                {
                    "initial_values": ";".join(f"{value:.17g}" for value in initial),
                    "Powell_iterations": iterations,
                    "Powell_success": bool(result.success),
                    "Powell_termination": str(result.message),
                    "Powell_requested_maxfev": CONTROL_MAXFEV,
                },
            )
        )

    print("Post-hoc evaluation on fitting, validation, and test data", flush=True)
    summary: list[dict[str, Any]] = []
    per_trajectory: list[dict[str, Any]] = []
    final_stage_names = {
        ("direct_powell_newton", "after_Newton"),
        ("coarse_brent_newton", "after_Newton"),
        *(("powell_without_global", name) for name, _ in CONTROL_STARTS),
    }
    for stage_result in stage_results:
        save = (stage_result.method, stage_result.stage) in final_stage_names
        aggregate, rows = _evaluate_stage(stage_result, datasets, save)
        summary.append(aggregate)
        per_trajectory.extend(rows)

    _write_csv(OUTPUT_DIR / "four_dynamics_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "search_history.csv", search_history)
    _write_csv(OUTPUT_DIR / "per_trajectory_assignments.csv", per_trajectory)
    _plot_progress(search_history, direct_evaluations, baseline_global_evaluations)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "DIRECT_evaluations": direct_evaluations,
            "DIRECT_best": direct_values,
            "Powell_best": powell_values,
            "main_final": newton_a.values,
            "baseline_final": newton_b.values,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
