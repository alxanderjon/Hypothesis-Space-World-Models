"""Measure the local float64 precision floor of the discovered M=2 solution."""

from __future__ import annotations

import csv
import itertools
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P
from experiments.sho.run_stage1b_continuous_c_search_comparison import _losses
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

SOURCE_SUMMARY = Path(
    "outputs/sho/stage1B/unknown_shared_dynamics_count/model_count_summary.csv"
)
OUTPUT_DIR = Path("outputs/sho/stage1B/m2_precision_floor")
FIT_PER_GROUP = 32
VALIDATION_PER_GROUP = 32
TEST_PER_GROUP = 64
STABILITY_SCALES = (1e-8, 1e-7, 1e-6)
BRENT_XATOL = 1e-18
BRENT_MAXITER = 1000
NEWTON_MAX_ITERATIONS = 50
NEWTON_MAX_BACKTRACKS = 60
MILESTONES = (1e-12, 1e-16, 1e-20, 1e-24, 1e-28)


@dataclass(frozen=True)
class DiscoveredM2:
    values: tuple[float, float]
    brackets: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class BrentResult:
    value: float
    objective: float
    evaluations: int
    success: bool
    message: str
    last_evaluation_change: float
    evaluation_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class StencilResult:
    lower_loss_exists: bool
    best_value: float
    best_objective: float
    best_offset: float
    smallest_resolvable_scale: float | None
    smallest_lower_loss_scale: float | None
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class NewtonResult:
    value: float
    objective: float
    accepted_iterations: int
    stop_reason: str
    rows: list[dict[str, Any]]


def _load_discovered_m2() -> DiscoveredM2:
    with SOURCE_SUMMARY.open(encoding="utf-8", newline="") as file:
        candidates = [row for row in csv.DictReader(file) if int(row["M"]) == 2]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one M=2 row in {SOURCE_SUMMARY}")
    row = candidates[0]
    values = tuple(float(value) for value in row["c_values"].split(";"))
    matches = re.findall(r"\[([^,]+),([^\]]+)\]", row["coarse_brackets"])
    brackets = tuple((float(low), float(high)) for low, high in matches)
    if len(values) != 2 or len(brackets) != 2 or not values[0] < values[1]:
        raise RuntimeError("The source row is not a valid ordered M=2 discovery output")
    return DiscoveredM2(
        values=(values[0], values[1]),
        brackets=(brackets[0], brackets[1]),
    )


def _candidate_losses(value: float | torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    scalar = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=DTYPE)
    return _losses(scalar.expand(positions.shape[0]), positions[:, :P])


def _assignment_state(
    values: tuple[float, float], positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.stack(
        [_candidate_losses(value, positions).detach() for value in values], dim=1
    )
    assigned_losses, assignments = torch.min(matrix, dim=1)
    return assigned_losses, assignments


def _verify_assignment_stability(
    discovered: DiscoveredM2, positions: torch.Tensor
) -> tuple[torch.Tensor, list[dict[str, Any]], float, int]:
    base_losses, base_assignments = _assignment_state(discovered.values, positions)
    other_losses = torch.stack(
        [
            _candidate_losses(discovered.values[1], positions).detach(),
            _candidate_losses(discovered.values[0], positions).detach(),
        ],
        dim=1,
    )
    base_margin = float((other_losses.gather(1, base_assignments[:, None]).squeeze(1) - base_losses).min().item())
    rows: list[dict[str, Any]] = []
    maximum_changes = 0
    for scale in STABILITY_SCALES:
        for direction_a, direction_b in itertools.product((-1, 0, 1), repeat=2):
            if direction_a == 0 and direction_b == 0:
                continue
            perturbed = (
                discovered.values[0] + direction_a * scale,
                discovered.values[1] + direction_b * scale,
            )
            if not (
                discovered.brackets[0][0] <= perturbed[0] <= discovered.brackets[0][1]
                and discovered.brackets[1][0] <= perturbed[1] <= discovered.brackets[1][1]
                and perturbed[0] < perturbed[1]
            ):
                continue
            _, assignments = _assignment_state(perturbed, positions)
            changed = int((assignments != base_assignments).sum().item())
            maximum_changes = max(maximum_changes, changed)
            rows.append(
                {
                    "operation": "assignment_stability",
                    "component": "both",
                    "iteration_or_evaluation": len(rows) + 1,
                    "c": "",
                    "objective": "",
                    "offset": "",
                    "gradient": "",
                    "hessian": "",
                    "proposed_delta": "",
                    "accepted_delta": "",
                    "backtracks": "",
                    "assignment_perturbation_A": direction_a * scale,
                    "assignment_perturbation_B": direction_b * scale,
                    "assignments_changed": changed,
                    "stop_reason": "",
                }
            )
    if maximum_changes:
        raise RuntimeError(
            f"Discovered assignments are not stable: up to {maximum_changes} changed"
        )
    return base_assignments, rows, base_margin, maximum_changes


def _group_objective_float(value: float, positions: torch.Tensor) -> float:
    with torch.no_grad():
        return float(_candidate_losses(value, positions).mean().item())


def _tight_brent(
    component: str, initial_value: float, bracket: tuple[float, float], positions: torch.Tensor
) -> BrentResult:
    evaluations: list[tuple[float, float]] = []

    def objective(value: float) -> float:
        loss = _group_objective_float(value, positions)
        evaluations.append((float(value), loss))
        return loss

    result = minimize_scalar(
        objective,
        bounds=bracket,
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    last_change = (
        abs(evaluations[-1][0] - evaluations[-2][0])
        if len(evaluations) >= 2
        else float("nan")
    )
    rows = [
        {
            "operation": "tight_bounded_brent",
            "component": component,
            "iteration_or_evaluation": index,
            "c": value,
            "objective": loss,
            "offset": value - initial_value,
            "gradient": "",
            "hessian": "",
            "proposed_delta": "",
            "accepted_delta": "",
            "backtracks": "",
            "assignment_perturbation_A": "",
            "assignment_perturbation_B": "",
            "assignments_changed": "",
            "stop_reason": str(result.message) if index == len(evaluations) else "",
        }
        for index, (value, loss) in enumerate(evaluations, start=1)
    ]
    return BrentResult(
        value=float(result.x),
        objective=float(result.fun),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        last_evaluation_change=last_change,
        evaluation_rows=rows,
    )


def _stencil_offsets(center: float) -> tuple[float, ...]:
    ulp = abs(float(np.spacing(np.float64(center))))
    offsets = {ulp * multiplier for multiplier in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)}
    for exponent in range(-16, -5):
        offsets.add(10.0**exponent)
        offsets.add(3.0 * 10.0**exponent)
    return tuple(sorted(offset for offset in offsets if offset > 0.0))


def _local_stencil(
    operation: str,
    component: str,
    center: float,
    bracket: tuple[float, float],
    positions: torch.Tensor,
) -> StencilResult:
    center_loss = _group_objective_float(center, positions)
    candidates = [(center, 0.0, center_loss)]
    for magnitude in _stencil_offsets(center):
        for direction in (-1.0, 1.0):
            value = center + direction * magnitude
            if bracket[0] <= value <= bracket[1] and value != center:
                candidates.append((value, value - center, _group_objective_float(value, positions)))
    best_value, best_offset, best_loss = min(candidates, key=lambda item: item[2])
    changed_scales = sorted(
        abs(offset) for _, offset, loss in candidates if offset and loss != center_loss
    )
    lower_scales = sorted(
        abs(offset) for _, offset, loss in candidates if offset and loss < center_loss
    )
    rows = [
        {
            "operation": operation,
            "component": component,
            "iteration_or_evaluation": index,
            "c": value,
            "objective": loss,
            "offset": offset,
            "gradient": "",
            "hessian": "",
            "proposed_delta": "",
            "accepted_delta": "",
            "backtracks": "",
            "assignment_perturbation_A": "",
            "assignment_perturbation_B": "",
            "assignments_changed": "",
            "stop_reason": "",
        }
        for index, (value, offset, loss) in enumerate(candidates, start=1)
    ]
    return StencilResult(
        lower_loss_exists=best_loss < center_loss,
        best_value=best_value,
        best_objective=best_loss,
        best_offset=best_offset,
        smallest_resolvable_scale=changed_scales[0] if changed_scales else None,
        smallest_lower_loss_scale=lower_scales[0] if lower_scales else None,
        rows=rows,
    )


def _safeguarded_newton(
    component: str,
    initial_value: float,
    bracket: tuple[float, float],
    positions: torch.Tensor,
) -> NewtonResult:
    value = initial_value
    rows: list[dict[str, Any]] = []
    accepted_iterations = 0
    stop_reason = "maximum_iterations"
    for iteration in range(1, NEWTON_MAX_ITERATIONS + 1):
        parameter = torch.tensor(value, dtype=DTYPE, requires_grad=True)
        objective = _candidate_losses(parameter, positions).mean()
        gradient = torch.autograd.grad(objective, parameter, create_graph=True)[0]
        hessian = torch.autograd.grad(gradient, parameter)[0]
        objective_value = float(objective.detach().item())
        gradient_value = float(gradient.detach().item())
        hessian_value = float(hessian.detach().item())
        proposed_delta = -gradient_value / hessian_value if hessian_value > 0.0 else float("nan")
        accepted_delta = 0.0
        backtracks = 0
        candidate_value = value
        candidate_objective = objective_value
        if not math.isfinite(proposed_delta) or hessian_value <= 0.0:
            stop_reason = "nonpositive_or_invalid_hessian"
        elif value + proposed_delta == value:
            stop_reason = "numerically_negligible_update"
        else:
            scale = 1.0
            accepted = False
            for backtracks in range(NEWTON_MAX_BACKTRACKS + 1):
                trial = value + scale * proposed_delta
                if bracket[0] <= trial <= bracket[1] and trial != value:
                    trial_objective = _group_objective_float(trial, positions)
                    if trial_objective < objective_value:
                        candidate_value = trial
                        candidate_objective = trial_objective
                        accepted_delta = trial - value
                        accepted = True
                        break
                scale *= 0.5
            if accepted:
                value = candidate_value
                accepted_iterations += 1
                stop_reason = ""
            else:
                stop_reason = "no_float64_decreasing_step"
        rows.append(
            {
                "operation": "safeguarded_newton",
                "component": component,
                "iteration_or_evaluation": iteration,
                "c": candidate_value,
                "objective": candidate_objective,
                "offset": candidate_value - initial_value,
                "gradient": gradient_value,
                "hessian": hessian_value,
                "proposed_delta": proposed_delta,
                "accepted_delta": accepted_delta,
                "backtracks": backtracks,
                "assignment_perturbation_A": "",
                "assignment_perturbation_B": "",
                "assignments_changed": "",
                "stop_reason": stop_reason,
            }
        )
        if stop_reason:
            break
    return NewtonResult(
        value=value,
        objective=_group_objective_float(value, positions),
        accepted_iterations=accepted_iterations,
        stop_reason=stop_reason,
        rows=rows,
    )


def _test_rollout_metrics(values: tuple[float, float], data: MixtureData) -> tuple[float, float, float]:
    _, assignments = _assignment_state(values, data.positions)
    worst_h1 = 0.0
    worst_mse = 0.0
    worst_maximum = 0.0
    for trajectory in range(data.positions.shape[0]):
        group_id = int(data.group_ids[trajectory].item())
        future_steps = _posthoc_group_metadata(group_id)[3]
        value = values[int(assignments[trajectory].item())]
        position = data.positions[trajectory]
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
        worst_h1 = max(worst_h1, float(torch.abs(direct_error).max().item()))
        worst_mse = max(worst_mse, float(torch.mean(rollout_error**2).item()))
        worst_maximum = max(worst_maximum, float(torch.abs(rollout_error).max().item()))
    return worst_h1, worst_mse, worst_maximum


def _stage_summary(
    stage: str,
    values: tuple[float, float],
    fit_data: MixtureData,
    validation_data: MixtureData,
    test_data: MixtureData,
    frozen_fit_assignments: torch.Tensor,
) -> dict[str, Any]:
    fit_matrix = torch.stack(
        [_candidate_losses(value, fit_data.positions).detach() for value in values], dim=1
    )
    fit_losses = fit_matrix.gather(1, frozen_fit_assignments[:, None]).squeeze(1)
    reassigned = torch.argmin(fit_matrix, dim=1)
    validation_losses, _ = _assignment_state(values, validation_data.positions)
    worst_fit = float(fit_losses.max().item())
    worst_validation = float(validation_losses.max().item())
    worst_combined = max(worst_fit, worst_validation)
    worst_h1, worst_rollout_mse, worst_rollout_maximum = _test_rollout_metrics(
        values, test_data
    )
    row: dict[str, Any] = {
        "stage": stage,
        "c_A": values[0],
        "c_B": values[1],
        "worst_fitting_prefix_loss": worst_fit,
        "worst_validation_prefix_loss": worst_validation,
        "worst_fit_validation_prefix_loss": worst_combined,
        "fitting_assignment_changes_from_frozen": int(
            (reassigned != frozen_fit_assignments).sum().item()
        ),
        "worst_test_h1_error": worst_h1,
        "worst_test_10_period_mse": worst_rollout_mse,
        "worst_test_max_abs_10_period_error": worst_rollout_maximum,
    }
    for milestone in MILESTONES:
        exponent = int(round(-math.log10(milestone)))
        row[f"worst_prefix_below_1e-{exponent}"] = worst_combined < milestone
    return row


def _posthoc_reference_errors(values: tuple[float, float]) -> tuple[float, float, float, float]:
    references = tuple(sorted(_posthoc_group_metadata(group_id)[2] for group_id in (0, 1)))
    direct = abs(values[0] - references[0]) + abs(values[1] - references[1])
    swapped = abs(values[0] - references[1]) + abs(values[1] - references[0])
    if direct <= swapped:
        return references[0], references[1], abs(values[0] - references[0]), abs(values[1] - references[1])
    return references[1], references[0], abs(values[0] - references[1]), abs(values[1] - references[0])


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    discovered = _load_discovered_m2()
    fit_data = _make_mixture(FIT_PER_GROUP, TRAIN_DATA_SEED + 1520000)
    validation_data = _make_mixture(VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1521000)
    test_data = _make_mixture(TEST_PER_GROUP, TEST_DATA_SEED + 1522000)

    print("Step 1: verifying unsupervised assignment stability", flush=True)
    frozen_assignments, history, minimum_margin, maximum_changes = _verify_assignment_stability(
        discovered, fit_data.positions
    )
    group_positions = tuple(
        fit_data.positions[frozen_assignments == component]
        for component in range(2)
    )
    if any(group.shape[0] == 0 for group in group_positions):
        raise RuntimeError("Cannot polish an unused discovered component")

    print("Steps 2-3: tighter independent bounded Brent", flush=True)
    brent = tuple(
        _tight_brent(
            component,
            discovered.values[index],
            discovered.brackets[index],
            group_positions[index],
        )
        for index, component in enumerate(("A", "B"))
    )
    for result in brent:
        history.extend(result.evaluation_rows)

    print("Step 4: local float64 stencils and safeguarded Newton", flush=True)
    brent_stencils = tuple(
        _local_stencil(
            "post_brent_stencil",
            component,
            brent[index].value,
            discovered.brackets[index],
            group_positions[index],
        )
        for index, component in enumerate(("A", "B"))
    )
    for stencil in brent_stencils:
        history.extend(stencil.rows)
    newton = []
    for index, component in enumerate(("A", "B")):
        if brent_stencils[index].lower_loss_exists:
            result = _safeguarded_newton(
                component,
                brent[index].value,
                discovered.brackets[index],
                group_positions[index],
            )
        else:
            result = NewtonResult(
                value=brent[index].value,
                objective=brent[index].objective,
                accepted_iterations=0,
                stop_reason="not_needed_brent_at_local_stencil_floor",
                rows=[],
            )
        newton.append(result)
        history.extend(result.rows)
    newton_tuple = (newton[0], newton[1])
    final_stencils = tuple(
        _local_stencil(
            "post_newton_floor_stencil",
            component,
            newton_tuple[index].value,
            discovered.brackets[index],
            group_positions[index],
        )
        for index, component in enumerate(("A", "B"))
    )
    for stencil in final_stencils:
        history.extend(stencil.rows)

    print("Step 5: stage metrics and post-hoc references", flush=True)
    stage_values = (
        ("original_discovered", discovered.values),
        ("tighter_bounded_brent", (brent[0].value, brent[1].value)),
        ("safeguarded_newton_polish", (newton_tuple[0].value, newton_tuple[1].value)),
    )
    summary = [
        _stage_summary(
            stage,
            values,
            fit_data,
            validation_data,
            test_data,
            frozen_assignments,
        )
        for stage, values in stage_values
    ]
    summary[0].update(
        {
            "assignment_stability_scales": ";".join(
                f"{value:.1e}" for value in STABILITY_SCALES
            ),
            "maximum_assignment_changes_under_perturbation": maximum_changes,
            "minimum_discovered_assignment_margin": minimum_margin,
        }
    )
    summary[1].update(
        {
            "requested_brent_xatol": BRENT_XATOL,
            "brent_A_objective": brent[0].objective,
            "brent_B_objective": brent[1].objective,
            "brent_A_function_evaluations": brent[0].evaluations,
            "brent_B_function_evaluations": brent[1].evaluations,
            "brent_A_success": brent[0].success,
            "brent_B_success": brent[1].success,
            "brent_A_termination": brent[0].message,
            "brent_B_termination": brent[1].message,
            "brent_A_final_evaluation_change": brent[0].last_evaluation_change,
            "brent_B_final_evaluation_change": brent[1].last_evaluation_change,
            "brent_A_nearby_lower_loss_exists": brent_stencils[0].lower_loss_exists,
            "brent_B_nearby_lower_loss_exists": brent_stencils[1].lower_loss_exists,
            "brent_A_best_stencil_offset": brent_stencils[0].best_offset,
            "brent_B_best_stencil_offset": brent_stencils[1].best_offset,
            "brent_A_smallest_resolvable_scale": brent_stencils[0].smallest_resolvable_scale,
            "brent_B_smallest_resolvable_scale": brent_stencils[1].smallest_resolvable_scale,
            "brent_A_smallest_lower_loss_scale": brent_stencils[0].smallest_lower_loss_scale,
            "brent_B_smallest_lower_loss_scale": brent_stencils[1].smallest_lower_loss_scale,
        }
    )
    reference_a, reference_b, error_a, error_b = _posthoc_reference_errors(stage_values[-1][1])
    summary[2].update(
        {
            "newton_A_accepted_iterations": newton_tuple[0].accepted_iterations,
            "newton_B_accepted_iterations": newton_tuple[1].accepted_iterations,
            "newton_A_stop_reason": newton_tuple[0].stop_reason,
            "newton_B_stop_reason": newton_tuple[1].stop_reason,
            "final_A_discrete_stencil_minimum": not final_stencils[0].lower_loss_exists,
            "final_B_discrete_stencil_minimum": not final_stencils[1].lower_loss_exists,
            "c_A_reference_posthoc": reference_a,
            "c_B_reference_posthoc": reference_b,
            "abs_c_A_error_posthoc": error_a,
            "abs_c_B_error_posthoc": error_b,
            "measured_limiting_factor": "bounded_Brent_algorithmic_termination_before_local_floor; then float64 arithmetic and rollout amplification",
            "recommended_future_exact_recovery_gate": 1e-20,
            "recommended_secondary_precision_milestone": 1e-24,
        }
    )

    _write_csv(OUTPUT_DIR / "precision_floor_summary.csv", summary)
    _write_csv(OUTPUT_DIR / "precision_polish_history.csv", history)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "assignment_stable": maximum_changes == 0,
            "brent_values": [result.value for result in brent],
            "newton_values": [result.value for result in newton_tuple],
            "final_worst_prefix_loss": summary[-1]["worst_fit_validation_prefix_loss"],
            "final_parameter_errors": [error_a, error_b],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
