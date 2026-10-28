"""Diagnose separability versus component allocation in the failed M=16 run."""

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
from scipy.optimize import minimize_scalar

import experiments.sho.run_stage1b_sixteen_dynamics_scaling_test as scaling
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    "outputs/sho/stage1B/sixteen_dynamics_separability_diagnostic"
)
BRENT_XATOL = 1e-14
BRENT_MAXITER = 1_000
NEWTON_MAX_BACKTRACKS = 60
GATE = 1e-20


def _group_objective_float(c: float, positions: torch.Tensor) -> float:
    return float(scaling._candidate_losses(c, positions).mean().item())


def _one_safeguarded_newton_step(
    initial: float, positions: torch.Tensor
) -> tuple[float, float, float, float, bool, int, str, float]:
    """Apply at most one accepted Newton step with monotone backtracking."""
    start = time.perf_counter()
    parameter = torch.tensor(initial, dtype=DTYPE, requires_grad=True)
    objective = scaling._candidate_losses(parameter, positions).mean()
    gradient = torch.autograd.grad(objective, parameter, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient, parameter)[0]
    current_objective = float(objective.detach().item())
    gradient_value = float(gradient.detach().item())
    hessian_value = float(hessian.detach().item())
    if not math.isfinite(hessian_value) or hessian_value <= 0.0:
        return (
            initial,
            current_objective,
            gradient_value,
            hessian_value,
            False,
            0,
            "nonpositive_or_invalid_hessian",
            time.perf_counter() - start,
        )
    proposed = -gradient_value / hessian_value
    if not math.isfinite(proposed):
        return (
            initial,
            current_objective,
            gradient_value,
            hessian_value,
            False,
            0,
            "invalid_newton_step",
            time.perf_counter() - start,
        )
    scale = 1.0
    for backtracks in range(NEWTON_MAX_BACKTRACKS + 1):
        trial = initial + scale * proposed
        if scaling.LOWER <= trial <= scaling.UPPER and trial != initial:
            trial_objective = _group_objective_float(trial, positions)
            if trial_objective < current_objective:
                return (
                    trial,
                    trial_objective,
                    gradient_value,
                    hessian_value,
                    True,
                    backtracks,
                    "accepted",
                    time.perf_counter() - start,
                )
        scale *= 0.5
    return (
        initial,
        current_objective,
        gradient_value,
        hessian_value,
        False,
        NEWTON_MAX_BACKTRACKS,
        "no_float64_decreasing_step",
        time.perf_counter() - start,
    )


def _recover_group(
    group_id: int, positions: torch.Tensor
) -> tuple[dict[str, Any], float]:
    evaluations = 0

    def objective(c: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return _group_objective_float(c, positions)

    brent_start = time.perf_counter()
    brent = minimize_scalar(
        objective,
        bounds=(scaling.LOWER, scaling.UPPER),
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    brent_runtime = time.perf_counter() - brent_start
    brent_c = float(brent.x)
    (
        recovered_c,
        recovered_mean_loss,
        gradient,
        hessian,
        newton_accepted,
        newton_backtracks,
        newton_status,
        newton_runtime,
    ) = _one_safeguarded_newton_step(brent_c, positions)
    losses = scaling._candidate_losses(recovered_c, positions).detach()
    group_name, omega, reference_c, _ = scaling._posthoc_metadata(group_id)
    row: dict[str, Any] = {
        "group_index": group_id + 1,
        "group_name": group_name,
        "previously_merged_low_c_group": group_id < 3,
        "fitting_trajectory_count": positions.shape[0],
        "reference_omega_posthoc": omega,
        "reference_c_posthoc": reference_c,
        "brent_c": brent_c,
        "brent_mean_prefix_loss": float(brent.fun),
        "brent_objective_evaluations": evaluations,
        "brent_success": bool(brent.success),
        "brent_message": str(brent.message),
        "brent_runtime_seconds": brent_runtime,
        "newton_c": recovered_c,
        "newton_mean_prefix_loss": recovered_mean_loss,
        "newton_gradient_at_brent_c": gradient,
        "newton_hessian_at_brent_c": hessian,
        "newton_step_accepted": newton_accepted,
        "newton_backtracks": newton_backtracks,
        "newton_status": newton_status,
        "newton_runtime_seconds": newton_runtime,
        "recovered_c": recovered_c,
        "absolute_c_error": abs(recovered_c - reference_c),
        "mean_group_prefix_loss": float(losses.mean().item()),
        "worst_group_prefix_loss": float(losses.max().item()),
        "group_1e-20_gate_passed": float(losses.max().item()) < GATE,
    }
    return row, recovered_c


def _plot_cross_loss(matrix: np.ndarray) -> None:
    log_matrix = np.log10(np.maximum(matrix, 1e-30))
    figure, axis = plt.subplots(figsize=(10, 8.5))
    image = axis.imshow(log_matrix, cmap="viridis", aspect="equal")
    ticks = np.arange(scaling.M)
    labels = [str(index + 1) for index in range(scaling.M)]
    axis.set_xticks(ticks, labels)
    axis.set_yticks(ticks, labels)
    axis.set_xlabel("recovered dynamics b")
    axis.set_ylabel("true group a")
    axis.set_title(r"Cross-prefix loss: $\log_{10} M[a,b]$")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}$ mean prefix loss")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "cross_loss_matrix_heatmap.png", dpi=180)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Exact fitting data and seed from the failed scaling experiment.
    fit = scaling._make_data(
        scaling.FIT_PER_GROUP, TRAIN_DATA_SEED + 1_600_000
    )

    group_rows: list[dict[str, Any]] = []
    recovered_cs: list[float] = []
    grouped_positions: list[torch.Tensor] = []
    for group_id in range(scaling.M):
        positions = fit.positions[fit.group_ids == group_id]
        grouped_positions.append(positions)
        print(f"Recovering labeled group {group_id + 1:02d}/16", flush=True)
        row, recovered_c = _recover_group(group_id, positions)
        group_rows.append(row)
        recovered_cs.append(recovered_c)

    cross_loss = np.empty((scaling.M, scaling.M), dtype=np.float64)
    trajectory_margins: list[torch.Tensor] = []
    for group_id, positions in enumerate(grouped_positions):
        matrix = torch.stack(
            [
                scaling._candidate_losses(recovered_c, positions).detach()
                for recovered_c in recovered_cs
            ],
            dim=1,
        )
        cross_loss[group_id] = matrix.mean(dim=0).numpy()
        own = matrix[:, group_id]
        incorrect = torch.cat(
            (matrix[:, :group_id], matrix[:, group_id + 1 :]), dim=1
        )
        best_incorrect, best_incorrect_indices = torch.min(incorrect, dim=1)
        # Map indices in the 15-column incorrect matrix back to 16 components.
        best_incorrect_groups = best_incorrect_indices + (
            best_incorrect_indices >= group_id
        ).to(best_incorrect_indices.dtype)
        margins = best_incorrect - own
        trajectory_margins.append(margins)
        incorrect_group_means = cross_loss[group_id].copy()
        incorrect_group_means[group_id] = np.inf
        nearest_group = int(np.argmin(incorrect_group_means))
        group_rows[group_id].update(
            {
                "own_recovered_mean_cross_loss": cross_loss[group_id, group_id],
                "best_incorrect_mean_cross_loss": float(
                    incorrect_group_means[nearest_group]
                ),
                "nearest_incorrect_recovered_group": nearest_group + 1,
                "group_mean_cross_loss_margin": float(
                    incorrect_group_means[nearest_group]
                    - cross_loss[group_id, group_id]
                ),
                "minimum_trajectory_cross_loss_margin": float(
                    margins.min().item()
                ),
                "mean_trajectory_cross_loss_margin": float(
                    margins.mean().item()
                ),
                "maximum_trajectory_cross_loss_margin": float(
                    margins.max().item()
                ),
                "trajectory_cross_loss_margins": ";".join(
                    f"{float(value.item()):.17g}" for value in margins
                ),
                "best_incorrect_group_by_trajectory": ";".join(
                    str(int(value.item()) + 1) for value in best_incorrect_groups
                ),
            }
        )

    all_margins = torch.cat(trajectory_margins)
    global_minimum_margin = float(all_margins.min().item())
    global_mean_margin = float(all_margins.mean().item())
    all_groups_pass = all(bool(row["group_1e-20_gate_passed"]) for row in group_rows)
    for row in group_rows:
        row["all_16_groups_1e-20_gate_passed"] = all_groups_pass
        row["global_minimum_trajectory_cross_loss_margin"] = global_minimum_margin
        row["global_mean_trajectory_cross_loss_margin"] = global_mean_margin

    matrix_rows: list[dict[str, Any]] = []
    for group_id in range(scaling.M):
        row: dict[str, Any] = {
            "true_group_index": group_id + 1,
            "true_group_name": scaling._posthoc_metadata(group_id)[0],
            "reference_c_posthoc": scaling._posthoc_metadata(group_id)[2],
        }
        for recovered_group in range(scaling.M):
            row[f"loss_using_recovered_group_{recovered_group + 1:02d}"] = (
                cross_loss[group_id, recovered_group]
            )
        matrix_rows.append(row)

    _write_csv(OUTPUT_DIR / "group_recovery_summary.csv", group_rows)
    _write_csv(OUTPUT_DIR / "cross_loss_matrix.csv", matrix_rows)
    _plot_cross_loss(cross_loss)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "all_groups_pass": all_groups_pass,
            "worst_c_error": max(float(row["absolute_c_error"]) for row in group_rows),
            "worst_prefix_loss": max(
                float(row["worst_group_prefix_loss"]) for row in group_rows
            ),
            "global_minimum_trajectory_margin": global_minimum_margin,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
