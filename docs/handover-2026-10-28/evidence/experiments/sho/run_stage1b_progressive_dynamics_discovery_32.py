"""Scale successful progressive shared-dynamics discovery from 16 to 32."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import experiments.sho.run_stage1b_progressive_dynamics_discovery as progressive
import experiments.sho.run_stage1b_sixteen_dynamics_scaling_test as scaling
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/progressive_dynamics_discovery_32")
HIDDEN_DYNAMICS = 32
FIT_PER_GROUP = 16
VALIDATION_PER_GROUP = 4
TEST_PER_GROUP = 4
MAX_COMPONENT_ATTEMPTS = 40
HIDDEN_C_MIN = -1.90
HIDDEN_C_MAX = 1.90


def _reference_cs() -> torch.Tensor:
    """Generator/post-hoc helper; never passed to the fitting procedure."""
    return torch.linspace(
        HIDDEN_C_MIN, HIDDEN_C_MAX, HIDDEN_DYNAMICS, dtype=DTYPE
    )


def _make_data(per_group: int, seed: int) -> scaling.DynamicsData:
    """Generate and fully shuffle a 32-dynamics split."""
    generator = torch.Generator().manual_seed(seed)
    hidden_cs = _reference_cs()
    hidden_omegas = torch.acos(hidden_cs / 2.0) / float(DELTA_T)
    group_ids = torch.arange(HIDDEN_DYNAMICS, dtype=torch.long).repeat_interleave(
        per_group
    )
    generating_omegas = hidden_omegas[group_ids]
    initial_conditions = 2.0 * torch.rand(
        HIDDEN_DYNAMICS * per_group,
        2,
        generator=generator,
        dtype=DTYPE,
    ) - 1.0
    slowest_frequency = float(hidden_omegas.min().item())
    future_steps = math.ceil(
        10.0 * math.tau / (slowest_frequency * float(DELTA_T))
    )
    times = torch.arange(
        scaling.P + future_steps + 1, dtype=DTYPE
    ).unsqueeze(0)
    times = times.expand(HIDDEN_DYNAMICS * per_group, -1) * float(DELTA_T)
    positions, _ = _state_at_time(
        initial_conditions, generating_omegas, times
    )
    permutation = torch.randperm(
        HIDDEN_DYNAMICS * per_group, generator=generator
    )
    return scaling.DynamicsData(
        positions[permutation], group_ids[permutation]
    )


def _posthoc_metadata(group_id: int) -> tuple[str, float, float, int]:
    reference_c = float(_reference_cs()[group_id].item())
    omega = math.acos(reference_c / 2.0) / float(DELTA_T)
    future_steps = math.ceil(10.0 * math.tau / (omega * float(DELTA_T)))
    return f"dynamics_{group_id + 1:02d}", omega, reference_c, future_steps


def _posthoc_component_mapping(
    values: tuple[float, ...], fit: scaling.DynamicsData
) -> dict[int, int]:
    _, assignments, _ = progressive._assign_from_values(values, fit.positions)
    confusion = np.zeros(
        (HIDDEN_DYNAMICS, len(values)), dtype=np.int64
    )
    for group in range(HIDDEN_DYNAMICS):
        selected = assignments[fit.group_ids == group]
        for component in range(len(values)):
            confusion[group, component] = int(
                (selected == component).sum().item()
            )
    groups, components = linear_sum_assignment(-confusion)
    return {
        int(group): int(component)
        for group, component in zip(groups, components)
    }


def _posthoc_c_errors(
    values: tuple[float, ...]
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    references = np.asarray(
        [_posthoc_metadata(group)[2] for group in range(HIDDEN_DYNAMICS)]
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
    unmatched = len(values) - len(components)
    return tuple(matched_references), tuple(errors), unmatched


def _final_evaluation(
    values: tuple[float, ...],
    datasets: dict[str, scaling.DynamicsData],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mapping = _posthoc_component_mapping(values, datasets["fit"])
    references, errors, unmatched = _posthoc_c_errors(values)
    finite_errors = [error for error in errors if math.isfinite(error)]
    summary: dict[str, Any] = {
        "hidden_dynamics_count_posthoc": HIDDEN_DYNAMICS,
        "dynamics_discovered": len(values),
        "stopped_at_32_posthoc": len(values) == HIDDEN_DYNAMICS,
        "unmatched_extra_components_posthoc": unmatched,
        "recovered_c_values": progressive._value_string(values),
        "matched_reference_c_values_posthoc": progressive._value_string(
            references
        ),
        "absolute_c_errors_posthoc": progressive._value_string(errors),
        "worst_absolute_c_error_posthoc": (
            max(finite_errors) if finite_errors else float("nan")
        ),
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
        counts = progressive._counts(assignments, len(values))
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
            group_name, omega, reference_c, future_steps = _posthoc_metadata(
                group_id
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
                    "component_prefix_losses": progressive._value_string(
                        [float(item.item()) for item in matrix[trajectory]]
                    ),
                    "max_h1_error_10_period_evaluation": max_h1,
                    "rollout_10_period_mse": rollout_mse,
                    "max_abs_10_period_rollout_error": max_rollout,
                }
            )
        summary[f"{split}_assignment_counts"] = progressive._count_string(
            counts
        )
        summary[f"{split}_active_components"] = sum(
            count > 0 for count in counts
        )
        summary[f"{split}_assignment_accuracy"] = (
            correct / data.positions.shape[0]
        )
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
    fit = [
        max(float(row["worst_fit_prefix_loss_after_refinement"]), 1e-30)
        for row in rows
    ]
    validation = [
        max(
            float(row["worst_validation_prefix_loss_after_refinement"]),
            1e-30,
        )
        for row in rows
    ]
    axis.plot(x, fit, marker="o", markersize=4, label="fit")
    axis.plot(x, validation, marker="s", markersize=4, label="validation")
    axis.axhline(
        progressive.PREFIX_GATE,
        color="black",
        linestyle="--",
        label=r"$10^{-20}$ gate",
    )
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
    fit = _make_data(FIT_PER_GROUP, TRAIN_DATA_SEED + 1_700_000)
    validation = _make_data(
        VALIDATION_PER_GROUP, TRAIN_DATA_SEED + 1_701_000
    )
    test = _make_data(TEST_PER_GROUP, TEST_DATA_SEED + 1_702_000)
    datasets = {"fit": fit, "validation": validation, "test": test}

    # From here until the loop stops, only positions and prefix losses are used.
    cache = progressive.ScalarLossCache(fit.positions)
    values: tuple[float, ...] = tuple()
    fit_best_losses: torch.Tensor | None = None
    component_history: list[dict[str, Any]] = []
    stop_reason = "maximum_component_attempts"
    total_direct_evaluations = 0
    total_powell_evaluations = 0
    total_newton_evaluations = 0
    total_direct_runtime = 0.0
    total_powell_runtime = 0.0
    total_newton_runtime = 0.0

    for addition in range(1, MAX_COMPONENT_ATTEMPTS + 1):
        worst_fit_before = (
            float("inf")
            if fit_best_losses is None
            else float(fit_best_losses.max().item())
        )
        worst_validation_before = (
            float("inf")
            if not values
            else progressive._split_metrics(values, validation.positions)[0]
        )
        print(
            f"Addition {addition:02d}: 1D DIRECT with {len(values)} active components",
            flush=True,
        )
        add_objective, proposed_c, direct_termination, direct_runtime = (
            progressive._run_direct_addition(
                cache, fit_best_losses, addition
            )
        )
        direct_new_scalar_evaluations = add_objective.new_scalar_evaluations
        total_direct_evaluations += add_objective.evaluations
        total_direct_runtime += direct_runtime

        initial_raw = tuple(values) + (proposed_c,)
        pre_powell_matrix = cache.loss_matrix(initial_raw)
        pre_powell_losses, pre_powell_assignments = torch.min(
            pre_powell_matrix, dim=1
        )
        pre_powell_counts = progressive._counts(
            pre_powell_assignments, len(initial_raw)
        )
        new_count_before_powell = pre_powell_counts[-1]

        print(
            f"  proposed c={proposed_c:.12g}; joint Powell over {len(initial_raw)}",
            flush=True,
        )
        joint, powell_result, powell_runtime = progressive._run_joint_powell(
            cache, initial_raw
        )
        powell_new_scalar_evaluations = joint.new_scalar_evaluations
        total_powell_evaluations += joint.evaluations
        total_powell_runtime += powell_runtime
        if not joint.best_raw_values or joint.best_assignments is None:
            raise RuntimeError("Powell did not evaluate the joint objective")

        raw_values = joint.best_raw_values
        raw_counts = joint.best_counts
        previously_active_became_unused = any(
            raw_counts[index] == 0 for index in range(len(values))
        )
        new_count_after_powell = raw_counts[-1]
        unused_raw_indices = tuple(
            index for index, count in enumerate(raw_counts) if count == 0
        )
        active_values = tuple(
            value
            for index, value in enumerate(raw_values)
            if index not in unused_raw_indices
        )
        active_values = progressive._sorted_values(active_values)
        _, _, counts_after_removal = cache.assigned(active_values)

        stable, stable_assignments, margin, changes, perturbations = (
            progressive._assignment_stability(active_values, cache)
        )
        if stable:
            newton = progressive._newton_polish(
                active_values, fit.positions, stable_assignments
            )
            polished_values = progressive._sorted_values(newton.values)
        else:
            newton = progressive.NewtonResult(
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
        total_newton_runtime += newton.runtime_seconds

        values = polished_values
        (
            worst_fit_after,
            fit_best_losses,
            _,
            final_fit_counts,
        ) = progressive._split_metrics(values, fit.positions)
        (
            worst_validation_after,
            _,
            _,
            final_validation_counts,
        ) = progressive._split_metrics(values, validation.positions)
        gate_passed = (
            worst_fit_after < progressive.PREFIX_GATE
            and worst_validation_after < progressive.PREFIX_GATE
        )
        component_history.append(
            {
                "addition_attempt": addition,
                "component_count_before_addition": len(initial_raw) - 1,
                "spawned_c_by_DIRECT": proposed_c,
                "DIRECT_best_add_objective": add_objective.best_objective,
                "DIRECT_objective_evaluations": add_objective.evaluations,
                "DIRECT_new_scalar_loss_evaluations": direct_new_scalar_evaluations,
                "DIRECT_runtime_seconds": direct_runtime,
                "DIRECT_termination": direct_termination,
                "worst_fit_prefix_loss_before_addition": worst_fit_before,
                "worst_validation_prefix_loss_before_addition": (
                    worst_validation_before
                ),
                "worst_fit_prefix_loss_after_DIRECT_addition": float(
                    pre_powell_losses.max().item()
                ),
                "new_component_assignment_count_before_Powell": (
                    new_count_before_powell
                ),
                "new_component_received_trajectories_before_Powell": (
                    new_count_before_powell > 0
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
                    new_count_after_powell
                ),
                "new_component_received_trajectories_after_Powell": (
                    new_count_after_powell > 0
                ),
                "previously_discovered_components_remain_active": (
                    not previously_active_became_unused
                ),
                "Powell_caused_previously_active_component_to_become_unused": (
                    previously_active_became_unused
                ),
                "unused_component_indices_removed_after_Powell": (
                    progressive._count_string(
                        tuple(index + 1 for index in unused_raw_indices)
                    )
                ),
                "unused_component_count_removed_after_Powell": len(
                    unused_raw_indices
                ),
                "active_values_before_Newton": progressive._value_string(
                    active_values
                ),
                "active_assignment_counts_before_Newton": (
                    progressive._count_string(counts_after_removal)
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
                "active_c_values_after_refinement": progressive._value_string(
                    values
                ),
                "active_component_count_after_refinement": len(values),
                "fit_assignment_counts_after_refinement": (
                    progressive._count_string(final_fit_counts)
                ),
                "validation_assignment_counts_after_refinement": (
                    progressive._count_string(final_validation_counts)
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

    # Hidden metadata becomes visible only after label-blind fitting has stopped.
    posthoc_summary, final_assignments = _final_evaluation(values, datasets)
    any_spawned_unused = any(
        row["new_component_assignment_count_after_Powell"] == 0
        for row in component_history
    )
    removed_total = sum(
        int(row["unused_component_count_removed_after_Powell"])
        for row in component_history
    )
    any_previous_lost = any(
        bool(row["Powell_caused_previously_active_component_to_become_unused"])
        for row in component_history
    )
    summary = {
        "stop_reason": stop_reason,
        "addition_attempts": len(component_history),
        "maximum_component_attempts": MAX_COMPONENT_ATTEMPTS,
        "DIRECT_budget_per_addition": (
            progressive.DIRECT_MAXFUN_PER_ADDITION
        ),
        "Powell_maxfev_per_addition": (
            progressive.POWELL_MAXFEV_PER_ADDITION
        ),
        "total_DIRECT_objective_evaluations": total_direct_evaluations,
        "total_Powell_objective_evaluations": total_powell_evaluations,
        "total_Newton_objective_evaluations": total_newton_evaluations,
        "total_DIRECT_runtime_seconds": total_direct_runtime,
        "total_Powell_runtime_seconds": total_powell_runtime,
        "total_Newton_runtime_seconds": total_newton_runtime,
        "total_recorded_optimizer_runtime_seconds": (
            total_direct_runtime + total_powell_runtime + total_newton_runtime
        ),
        "any_spawned_component_unused_after_Powell": any_spawned_unused,
        "total_unused_components_removed_after_Powell": removed_total,
        "any_previously_discovered_component_became_unused": any_previous_lost,
        "final_fit_and_validation_1e-20_gate_passed": (
            component_history[-1]["fit_and_validation_1e-20_gate_passed"]
        ),
        **posthoc_summary,
    }
    _write_csv(OUTPUT_DIR / "summary.csv", [summary])
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
