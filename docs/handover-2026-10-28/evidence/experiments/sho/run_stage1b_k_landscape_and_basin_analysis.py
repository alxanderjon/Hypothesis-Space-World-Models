"""Stage 1B K-landscape and K-initialization basin diagnostics.

This is an operator-space diagnostic, not a neural model.
"""

from __future__ import annotations

import csv
import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import DIAGNOSTIC_NUM_TEST
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import _angle_error, _eig_angle, _true_k
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import (
    OBS_PREFIX_TRANSITIONS,
    ROLLOUT_PERIODS,
    TEST_DATA_SEED,
    _make_prefix_future_data,
    _rollout_truth,
    _state_windows,
    _trajectory_errors,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/k_landscape_and_basin_analysis")
HEATMAP_DIR = OUTPUT_DIR / "fixed_K_single_trajectory_heatmaps"
DTYPE = torch.float64
SEED = 0

THETA_DELTA_GRID = np.linspace(-0.06, 0.06, 121)
R_GRID = np.linspace(0.97, 1.03, 121)
DENSE_DELTA_GRID = np.linspace(-0.02, 0.02, 121)
TRAIN_LR = 3e-2
TRAIN_STEPS = 250
COV_PRETRAIN_STEPS = 100
CURRICULUM_STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
BASIN_THRESHOLDS = [1e-3, 1e-6, 1e-9, 1e-12]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_k(k_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return torch.bmm(k_matrix, states.unsqueeze(-1)).squeeze(-1)


def _rollout_matrix(k_matrix: torch.Tensor, state0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [state0]
    state = state0
    for _ in range(steps):
        state = _apply_k(k_matrix, state)
        states.append(state)
    return torch.stack(states, dim=1)


def _candidate_k(radius: np.ndarray, theta: np.ndarray) -> np.ndarray:
    k = np.zeros(radius.shape + (2, 2), dtype=np.float64)
    k[..., 0, 1] = 1.0
    k[..., 1, 0] = -(radius**2)
    k[..., 1, 1] = 2.0 * radius * np.cos(theta)
    return k


def _rollout_np(k_matrix: np.ndarray, state0: np.ndarray, steps: int) -> np.ndarray:
    states = np.empty((steps + 1, 2), dtype=np.float64)
    states[0] = state0
    for step in range(steps):
        states[step + 1] = k_matrix @ states[step]
    return states


def _rollout_losses_many_np(k_matrices: np.ndarray, true_states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out many fixed 2x2 K matrices against one trajectory."""
    flat_k = k_matrices.reshape(-1, 2, 2)
    state = np.broadcast_to(true_states[0], (flat_k.shape[0], 2)).copy()
    sum_sq = np.zeros(flat_k.shape[0], dtype=np.float64)
    max_abs = np.zeros(flat_k.shape[0], dtype=np.float64)
    final_abs = np.zeros(flat_k.shape[0], dtype=np.float64)
    for step in range(1, true_states.shape[0]):
        state = np.einsum("mij,mj->mi", flat_k, state)
        error = state[:, 1] - true_states[step, 1]
        sum_sq += error**2
        max_abs = np.maximum(max_abs, np.abs(error))
        final_abs = np.abs(error)
    mse = sum_sq / (true_states.shape[0] - 1)
    return mse.reshape(k_matrices.shape[:-2]), max_abs.reshape(k_matrices.shape[:-2]), final_abs.reshape(k_matrices.shape[:-2])


def _aggregate_losses_phase_radius_np(
    true_states: np.ndarray,
    omegas: np.ndarray,
    r_grid: np.ndarray,
    delta_grid: np.ndarray,
) -> np.ndarray:
    """Vectorized aggregate rollout MSE over [r, delta_theta, trajectory]."""
    theta = omegas[None, None, :] * float(DELTA_T) + delta_grid[None, :, None]
    radius = r_grid[:, None, None]
    k = np.zeros((len(r_grid), len(delta_grid), len(omegas), 2, 2), dtype=np.float64)
    k[..., 0, 1] = 1.0
    k[..., 1, 0] = -(radius**2)
    k[..., 1, 1] = 2.0 * radius * np.cos(theta)
    flat_k = k.reshape(-1, 2, 2)
    n_traj = len(omegas)
    state0 = np.broadcast_to(true_states[:, 0], (len(r_grid), len(delta_grid), n_traj, 2)).reshape(-1, 2).copy()
    state = state0
    sum_sq = np.zeros(flat_k.shape[0], dtype=np.float64)
    flat_truth = np.broadcast_to(true_states[None, None, :, :, :], (len(r_grid), len(delta_grid), n_traj, true_states.shape[1], 2)).reshape(-1, true_states.shape[1], 2)
    for step in range(1, true_states.shape[1]):
        state = np.einsum("mij,mj->mi", flat_k, state)
        error = state[:, 1] - flat_truth[:, step, 1]
        sum_sq += error**2
    per_candidate = sum_sq.reshape(len(r_grid), len(delta_grid), n_traj) / (true_states.shape[1] - 1)
    return np.mean(per_candidate, axis=2)


def _rollout_loss_np(k_matrix: np.ndarray, true_states: np.ndarray) -> tuple[float, float, float]:
    predicted = _rollout_np(k_matrix, true_states[0], true_states.shape[0] - 1)
    error = predicted[1:, 1] - true_states[1:, 1]
    return float(np.mean(error**2)), float(np.max(np.abs(error))), float(abs(error[-1]))


def _true_k_np(omegas: torch.Tensor) -> np.ndarray:
    return _true_k(omegas).detach().cpu().numpy()


def _representative_indices(omegas: torch.Tensor) -> dict[str, int]:
    values = omegas.detach().cpu().numpy()
    order = np.argsort(values)
    return {
        "low_omega": int(order[0]),
        "medium_omega": int(order[len(order) // 2]),
        "high_omega": int(order[-1]),
    }


def _plot_heatmap(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    true_x: float,
    true_y: float,
    log_scale: bool,
) -> None:
    plt.figure(figsize=(7.0, 5.4))
    data = np.log10(np.maximum(values, 1e-300)) if log_scale else values
    mesh = plt.pcolormesh(x, y, data, shading="auto", cmap="viridis")
    plt.scatter([true_x], [true_y], s=45, c="red", marker="x", label="true K")
    plt.colorbar(mesh, label="log10 MSE" if log_scale else "MSE")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _single_trajectory_landscapes(true_states: torch.Tensor, omegas: torch.Tensor) -> list[dict[str, Any]]:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    states_np = true_states.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for label, index in _representative_indices(omegas).items():
        omega = float(omegas[index].item())
        theta_true = omega * float(DELTA_T)
        theta_grid = theta_true + THETA_DELTA_GRID
        theta_mesh, r_mesh = np.meshgrid(theta_grid, R_GRID)
        k_grid = _candidate_k(r_mesh, theta_mesh)
        losses, _, _ = _rollout_losses_many_np(k_grid, states_np[index])
        np.save(HEATMAP_DIR / f"{label}_loss.npy", losses)
        _plot_heatmap(
            losses,
            theta_grid,
            R_GRID,
            HEATMAP_DIR / f"{label}_loss_heatmap.png",
            title=f"{label}: fixed K 10-period MSE",
            xlabel="theta",
            ylabel="radius r",
            true_x=theta_true,
            true_y=1.0,
            log_scale=False,
        )
        _plot_heatmap(
            losses,
            theta_grid,
            R_GRID,
            HEATMAP_DIR / f"{label}_log_loss_heatmap.png",
            title=f"{label}: fixed K 10-period log-MSE",
            xlabel="theta",
            ylabel="radius r",
            true_x=theta_true,
            true_y=1.0,
            log_scale=True,
        )
        min_index = np.unravel_index(int(np.argmin(losses)), losses.shape)
        rows.append(
            {
                "analysis": "single_trajectory_phase_radius_landscape",
                "trajectory_label": label,
                "trajectory_index": index,
                "omega": omega,
                "theta_true": theta_true,
                "min_mse": float(losses[min_index]),
                "min_theta": float(theta_grid[min_index[1]]),
                "min_r": float(R_GRID[min_index[0]]),
                "mse_at_true_grid_point": float(losses[np.argmin(np.abs(R_GRID - 1.0)), np.argmin(np.abs(theta_grid - theta_true))]),
            }
        )
    return rows


def _basin_width(mask: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, center_x_idx: int, center_y_idx: int) -> dict[str, float]:
    if not mask[center_y_idx, center_x_idx]:
        return {"delta_theta_width": 0.0, "r_width": 0.0}
    left = center_x_idx
    right = center_x_idx
    while left > 0 and mask[center_y_idx, left - 1]:
        left -= 1
    while right < mask.shape[1] - 1 and mask[center_y_idx, right + 1]:
        right += 1
    low = center_y_idx
    high = center_y_idx
    while low > 0 and mask[low - 1, center_x_idx]:
        low -= 1
    while high < mask.shape[0] - 1 and mask[high + 1, center_x_idx]:
        high += 1
    return {
        "delta_theta_width": float(x_grid[right] - x_grid[left]),
        "r_width": float(y_grid[high] - y_grid[low]),
    }


def _aggregate_landscape(true_states: torch.Tensor, omegas: torch.Tensor) -> tuple[list[dict[str, Any]], np.ndarray]:
    states_np = true_states.detach().cpu().numpy()
    omegas_np = omegas.detach().cpu().numpy()
    losses = _aggregate_losses_phase_radius_np(states_np, omegas_np, R_GRID, THETA_DELTA_GRID)
    np.save(OUTPUT_DIR / "aggregate_phase_radius_landscape.npy", losses)
    _plot_heatmap(
        losses,
        THETA_DELTA_GRID,
        R_GRID,
        OUTPUT_DIR / "aggregate_phase_radius_landscape.png",
        title="Aggregate normalized fixed-K landscape",
        xlabel="delta theta",
        ylabel="radius r",
        true_x=0.0,
        true_y=1.0,
        log_scale=False,
    )
    _plot_heatmap(
        losses,
        THETA_DELTA_GRID,
        R_GRID,
        OUTPUT_DIR / "aggregate_log_landscape.png",
        title="Aggregate normalized fixed-K log-landscape",
        xlabel="delta theta",
        ylabel="radius r",
        true_x=0.0,
        true_y=1.0,
        log_scale=True,
    )
    center_x = int(np.argmin(np.abs(THETA_DELTA_GRID)))
    center_y = int(np.argmin(np.abs(R_GRID - 1.0)))
    rows = []
    for threshold in BASIN_THRESHOLDS:
        width = _basin_width(losses < threshold, THETA_DELTA_GRID, R_GRID, center_x, center_y)
        rows.append(
            {
                "threshold": threshold,
                "delta_theta_width": width["delta_theta_width"],
                "r_width": width["r_width"],
                "grid_fraction_below_threshold": float(np.mean(losses < threshold)),
            }
        )
    _write_csv(OUTPUT_DIR / "basin_width_vs_threshold.csv", rows)
    return rows, losses


def _matrix_error_row(k_matrix: np.ndarray, k_true: np.ndarray) -> dict[str, float]:
    eig = np.linalg.eigvals(k_matrix[None, ...])
    true_eig = np.linalg.eigvals(k_true[None, ...])
    return {
        "trace_error": float(abs(np.trace(k_matrix) - np.trace(k_true))),
        "determinant_error": float(abs(np.linalg.det(k_matrix) - np.linalg.det(k_true))),
        "spectral_radius_error": float(abs(np.abs(eig).max() - np.abs(true_eig).max())),
        "eigen_angle_error": float(_angle_error(_eig_angle(k_matrix[None, ...]), _eig_angle(k_true[None, ...]))[0]),
        "k_frobenius_error": float(np.linalg.norm(k_matrix - k_true)),
    }


def _dense_perturbation_slices(true_states: torch.Tensor, omegas: torch.Tensor) -> list[dict[str, Any]]:
    states_np = true_states.detach().cpu().numpy()
    k_true_all = _true_k_np(omegas)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    random_planes = []
    for plane_index in range(2):
        a = rng.normal(size=(2, 2))
        b = rng.normal(size=(2, 2))
        a = a / np.linalg.norm(a)
        b = b - np.sum(a * b) * a
        b = b / np.linalg.norm(b)
        random_planes.append((f"random_plane_{plane_index}", a, b))
    for traj_index, (trajectory_states, k_true) in enumerate(zip(states_np, k_true_all)):
        for entry_name, row_index, col_index in [("K00", 0, 0), ("K01", 0, 1), ("K10", 1, 0), ("K11", 1, 1)]:
            k_stack = np.broadcast_to(k_true, (len(DENSE_DELTA_GRID), 2, 2)).copy()
            k_stack[:, row_index, col_index] += DENSE_DELTA_GRID
            mse_values, max_values, final_values = _rollout_losses_many_np(k_stack, trajectory_states)
            for delta, k, mse, max_error, final_error in zip(DENSE_DELTA_GRID, k_stack, mse_values, max_values, final_values):
                rows.append(
                    {
                        "analysis": "dense_perturbation_slice",
                        "trajectory_index": traj_index,
                        "slice": entry_name,
                        "delta_a": float(delta),
                        "delta_b": 0.0,
                        "rollout_mse": mse,
                        "max_abs_error": max_error,
                        "final_time_error": final_error,
                        **_matrix_error_row(k, k_true),
                    }
                )
        for slice_name, transform in [
            ("trace_c", lambda d: np.array([[0.0, 0.0], [0.0, d]], dtype=np.float64)),
            ("radius_determinant", lambda d: _candidate_k(np.array(1.0 + d), np.array(math.acos(np.clip(k_true[1, 1] / 2.0, -1.0, 1.0)))) - k_true),
        ]:
            k_stack = np.stack([k_true + transform(float(delta)) for delta in DENSE_DELTA_GRID], axis=0)
            mse_values, max_values, final_values = _rollout_losses_many_np(k_stack, trajectory_states)
            for delta, k, mse, max_error, final_error in zip(DENSE_DELTA_GRID, k_stack, mse_values, max_values, final_values):
                rows.append(
                    {
                        "analysis": "dense_perturbation_slice",
                        "trajectory_index": traj_index,
                        "slice": slice_name,
                        "delta_a": float(delta),
                        "delta_b": 0.0,
                        "rollout_mse": mse,
                        "max_abs_error": max_error,
                        "final_time_error": final_error,
                        **_matrix_error_row(k, k_true),
                    }
                )
        plane_delta = np.linspace(-0.015, 0.015, 41)
        for plane_name, a, b in random_planes:
            aa, bb = np.meshgrid(plane_delta, plane_delta)
            k_stack = k_true[None, None, :, :] + aa[..., None, None] * a + bb[..., None, None] * b
            mse_values, max_values, final_values = _rollout_losses_many_np(k_stack, trajectory_states)
            for delta_a, delta_b, k, mse, max_error, final_error in zip(aa.reshape(-1), bb.reshape(-1), k_stack.reshape(-1, 2, 2), mse_values.reshape(-1), max_values.reshape(-1), final_values.reshape(-1)):
                    rows.append(
                        {
                            "analysis": "dense_perturbation_slice",
                            "trajectory_index": traj_index,
                            "slice": plane_name,
                            "delta_a": float(delta_a),
                            "delta_b": float(delta_b),
                            "rollout_mse": mse,
                            "max_abs_error": max_error,
                            "final_time_error": final_error,
                            **_matrix_error_row(k, k_true),
                        }
                    )
    summary_rows = []
    for slice_name in sorted({row["slice"] for row in rows}):
        selected = [row for row in rows if row["slice"] == slice_name]
        summary_rows.append(
            {
                "slice": slice_name,
                "median_mse": float(np.median([row["rollout_mse"] for row in selected])),
                "p90_mse": float(np.percentile([row["rollout_mse"] for row in selected], 90)),
                "max_mse": float(np.max([row["rollout_mse"] for row in selected])),
                "median_eigen_angle_error": float(np.median([row["eigen_angle_error"] for row in selected])),
                "median_determinant_error": float(np.median([row["determinant_error"] for row in selected])),
                "median_trace_error": float(np.median([row["trace_error"] for row in selected])),
            }
        )
    _write_csv(OUTPUT_DIR / "dense_perturbation_slices.csv", rows)
    _write_csv(OUTPUT_DIR / "dense_perturbation_slice_summary.csv", summary_rows)
    plt.figure(figsize=(9.0, 5.4))
    labels = [row["slice"] for row in summary_rows]
    values = [row["p90_mse"] for row in summary_rows]
    plt.bar(labels, values)
    plt.yscale("log")
    plt.ylabel("p90 rollout MSE over perturbation slice")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dense_perturbation_slices.png", dpi=160)
    plt.close()
    return summary_rows


def _cov_loss(k_matrix: torch.Tensor, prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> torch.Tensor:
    x = prefix_in.transpose(1, 2)
    y = prefix_out.transpose(1, 2)
    c00 = torch.bmm(x, x.transpose(1, 2)) / prefix_in.shape[1]
    c10 = torch.bmm(y, x.transpose(1, 2)) / prefix_in.shape[1]
    return torch.mean((torch.bmm(k_matrix, c00) - c10) ** 2)


def _rollout_loss_torch(k_matrix: torch.Tensor, true_states: torch.Tensor, stage: int | str) -> torch.Tensor:
    if stage == "full_10_period":
        target = true_states
    else:
        length = int(stage) + 1
        target = true_states[:, :length]
    predicted = _rollout_matrix(k_matrix, target[:, 0], target.shape[1] - 1)
    return torch.mean((predicted[..., 1] - target[..., 1]) ** 2)


def _make_initial_k(k_true: torch.Tensor, kind: str, value_a: float, value_b: float, rng: torch.Generator) -> torch.Tensor:
    if kind == "phase_radius_grid":
        theta_true = torch.arccos(torch.clamp(k_true[:, 1, 1] / 2.0, -1.0, 1.0))
        radius = 1.0 + value_a
        theta = theta_true + value_b
        k = torch.zeros_like(k_true)
        k[:, 0, 1] = 1.0
        k[:, 1, 0] = -(radius**2)
        k[:, 1, 1] = 2.0 * radius * torch.cos(theta)
        return k
    if kind == "dense_noise":
        noise = torch.randn(k_true.shape, dtype=k_true.dtype, generator=rng)
        return k_true + value_a * noise
    if kind == "random_stable":
        angle = torch.empty(k_true.shape[0], dtype=k_true.dtype).uniform_(0.25, 0.75, generator=rng)
        radius = torch.empty(k_true.shape[0], dtype=k_true.dtype).uniform_(0.85, 1.05, generator=rng)
        k = torch.zeros_like(k_true)
        k[:, 0, 1] = 1.0
        k[:, 1, 0] = -(radius**2)
        k[:, 1, 1] = 2.0 * radius * torch.cos(angle)
        return k
    raise ValueError(f"unknown init kind: {kind}")


def _classify(k_matrix: torch.Tensor, k_true: torch.Tensor, rollout_mse: float) -> str:
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    frob = float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2))))
    trace_error = float(np.mean(np.abs(np.trace(k_np, axis1=1, axis2=2) - np.trace(true_np, axis1=1, axis2=2))))
    det_error = float(np.mean(np.abs(np.linalg.det(k_np) - np.linalg.det(true_np))))
    angle_error = float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np))))
    spectral_radius = float(np.mean(np.abs(np.linalg.eigvals(k_np)).max(axis=1)))
    if not math.isfinite(rollout_mse) or spectral_radius > 1.2:
        return "unstable_divergent"
    if rollout_mse < 1e-12 and frob < 1e-8:
        return "exact_recovery"
    if rollout_mse < 1e-8:
        return "near_recovery"
    if angle_error > max(trace_error, det_error):
        return "phase_error_failure"
    if det_error > trace_error:
        return "determinant_radius_failure"
    if spectral_radius < 1.05 and rollout_mse > 1e-6:
        return "wrong_stable_operator"
    return "general_dense_failure"


def _train_k_protocol(
    protocol: str,
    k_init: torch.Tensor,
    prefix_in: torch.Tensor,
    prefix_out: torch.Tensor,
    true_states: torch.Tensor,
) -> torch.Tensor:
    parameter = torch.nn.Parameter(k_init.clone())
    optimizer = torch.optim.Adam([parameter], lr=TRAIN_LR)
    stages = CURRICULUM_STAGES if protocol in {"curriculum", "cov_pretrain_then_curriculum"} else ["full_10_period"]
    if protocol == "cov_pretrain_then_curriculum":
        for _ in range(COV_PRETRAIN_STEPS):
            optimizer.zero_grad()
            loss = _cov_loss(parameter, prefix_in, prefix_out)
            loss.backward()
            optimizer.step()
    for stage in stages:
        steps = TRAIN_STEPS // max(1, len(stages))
        if protocol == "covariance_only":
            steps = TRAIN_STEPS
        for _ in range(steps):
            optimizer.zero_grad()
            if protocol == "covariance_only":
                loss = _cov_loss(parameter, prefix_in, prefix_out)
            elif protocol == "covariance_plus_rollout":
                loss = _rollout_loss_torch(parameter, true_states, stage) + 10.0 * _cov_loss(parameter, prefix_in, prefix_out)
            else:
                loss = _rollout_loss_torch(parameter, true_states, stage)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
        if protocol == "covariance_only":
            break
    return parameter.detach()


def _k_metrics(k_matrix: torch.Tensor, k_true: torch.Tensor, true_states: torch.Tensor) -> dict[str, Any]:
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    metrics = _trajectory_errors(predicted, true_states)
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    metrics.update(
        {
            "k_frobenius_error": float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2)))),
            "trace_c_error": float(np.mean(np.abs(np.trace(k_np, axis1=1, axis2=2) - np.trace(true_np, axis1=1, axis2=2)))),
            "determinant_error": float(np.mean(np.abs(np.linalg.det(k_np) - np.linalg.det(true_np)))),
            "eigen_angle_error": float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np)))),
            "spectral_radius_error": float(np.mean(np.abs(np.abs(np.linalg.eigvals(k_np)).max(axis=1) - 1.0))),
        }
    )
    return metrics


def _basin_sweep(prefix_in: torch.Tensor, prefix_out: torch.Tensor, true_states: torch.Tensor, omegas: torch.Tensor) -> list[dict[str, Any]]:
    subset = torch.linspace(0, prefix_in.shape[0] - 1, 16).round().to(torch.long)
    prefix_in = prefix_in.index_select(0, subset)
    prefix_out = prefix_out.index_select(0, subset)
    true_states = true_states.index_select(0, subset)
    omegas = omegas.index_select(0, subset)
    k_true = _true_k(omegas)
    protocols = ["direct_rollout", "curriculum", "covariance_only", "cov_pretrain_then_curriculum", "covariance_plus_rollout"]
    init_specs: list[tuple[str, float, float]] = []
    for r_delta in [-0.03, 0.0, 0.03]:
        for theta_delta in [-0.03, 0.0, 0.03]:
            init_specs.append(("phase_radius_grid", r_delta, theta_delta))
    for sigma in [1e-3, 1e-2]:
        for seed_index in range(2):
            init_specs.append(("dense_noise", sigma, float(seed_index)))
    for seed_index in range(3):
        init_specs.append(("random_stable", 0.0, float(seed_index)))
    rows: list[dict[str, Any]] = []
    for protocol in protocols:
        for init_kind, value_a, value_b in init_specs:
            rng = torch.Generator().manual_seed(SEED + int(abs(value_a) * 1e6) + int(value_b * 1000) + len(rows))
            k_init = _make_initial_k(k_true, init_kind, value_a, value_b, rng)
            k_final = _train_k_protocol(protocol, k_init, prefix_in, prefix_out, true_states)
            metrics = _k_metrics(k_final, k_true, true_states)
            classification = _classify(k_final, k_true, metrics["rollout_mse"])
            rows.append(
                {
                    "analysis": "k_initialization_basin_sweep",
                    "num_trajectories": int(true_states.shape[0]),
                    "protocol": protocol,
                    "init_kind": init_kind,
                    "init_value_a": value_a,
                    "init_value_b": value_b,
                    "classification": classification,
                    **metrics,
                }
            )
    _write_csv(OUTPUT_DIR / "basin_sweep_runs.csv", rows)
    summary_rows = []
    for protocol in protocols:
        for init_kind in sorted({row["init_kind"] for row in rows}):
            selected = [row for row in rows if row["protocol"] == protocol and row["init_kind"] == init_kind]
            if not selected:
                continue
            success = [row for row in selected if row["classification"] in {"exact_recovery", "near_recovery"}]
            summary_rows.append(
                {
                    "protocol": protocol,
                    "init_kind": init_kind,
                    "num_runs": len(selected),
                    "success_rate": len(success) / len(selected),
                    "median_rollout_mse": float(np.median([row["rollout_mse"] for row in selected])),
                    "best_rollout_mse": float(np.min([row["rollout_mse"] for row in selected])),
                    "median_k_frobenius_error": float(np.median([row["k_frobenius_error"] for row in selected])),
                }
            )
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _plot_basin_success(summary_rows)
    return summary_rows


def _plot_basin_success(summary_rows: list[dict[str, Any]]) -> None:
    protocols = sorted({row["protocol"] for row in summary_rows})
    init_kinds = sorted({row["init_kind"] for row in summary_rows})
    x = np.arange(len(protocols))
    width = 0.8 / len(init_kinds)
    plt.figure(figsize=(10.0, 5.2))
    for idx, init_kind in enumerate(init_kinds):
        values = []
        for protocol in protocols:
            match = [row for row in summary_rows if row["protocol"] == protocol and row["init_kind"] == init_kind]
            values.append(match[0]["success_rate"] if match else 0.0)
        plt.bar(x + (idx - len(init_kinds) / 2) * width + width / 2, values, width=width, label=init_kind)
    plt.xticks(x, protocols, rotation=25, ha="right")
    plt.ylabel("success rate")
    plt.ylim(0, 1.05)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basin_success_direct_vs_curriculum.png", dpi=160)
    plt.savefig(OUTPUT_DIR / "basin_success_covariance_vs_rollout.png", dpi=160)
    plt.close()


def _write_html(
    single_rows: list[dict[str, Any]],
    basin_width_rows: list[dict[str, Any]],
    dense_summary_rows: list[dict[str, Any]],
    basin_summary_rows: list[dict[str, Any]],
) -> None:
    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        body = "\n".join(
            "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in columns)}</tr></thead><tbody>{body}</tbody></table>"

    best_protocol = min(basin_summary_rows, key=lambda row: row["median_rollout_mse"]) if basin_summary_rows else {}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B K Landscape and Basin Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Stage 1B K Landscape and Basin Analysis</h1>
  <p>Position-only state <code>s_t=(x_{{t-1}},x_t)</code>. No neural network, no velocity, no labels for fitting. True omega is used only for diagnostic landscape centers and K-error attribution.</p>
  <h2>Representative Fixed-K Landscapes</h2>
  {table(single_rows, ["trajectory_label", "omega", "theta_true", "min_mse", "min_theta", "min_r", "mse_at_true_grid_point"])}
  <div class="grid">
    <img src="fixed_K_single_trajectory_heatmaps/low_omega_log_loss_heatmap.png">
    <img src="fixed_K_single_trajectory_heatmaps/medium_omega_log_loss_heatmap.png">
    <img src="fixed_K_single_trajectory_heatmaps/high_omega_log_loss_heatmap.png">
  </div>
  <h2>Aggregate Basin Widths</h2>
  {table(basin_width_rows, ["threshold", "delta_theta_width", "r_width", "grid_fraction_below_threshold"])}
  <div class="grid">
    <img src="aggregate_phase_radius_landscape.png">
    <img src="aggregate_log_landscape.png">
    <img src="dense_perturbation_slices.png">
  </div>
  <h2>Dense Perturbation Sensitivity</h2>
  {table(dense_summary_rows, ["slice", "median_mse", "p90_mse", "max_mse", "median_eigen_angle_error", "median_determinant_error", "median_trace_error"])}
  <h2>K Initialization Basin Sweep</h2>
  <p>Best median protocol/init group: {_fmt(best_protocol.get("protocol", ""))} / {_fmt(best_protocol.get("init_kind", ""))}.</p>
  {table(basin_summary_rows, ["protocol", "init_kind", "num_runs", "success_rate", "median_rollout_mse", "best_rollout_mse", "median_k_frobenius_error"])}
  <div class="grid">
    <img src="basin_success_direct_vs_curriculum.png">
    <img src="basin_success_covariance_vs_rollout.png">
  </div>
  <h2>Implementation Appendix</h2>
  <h3>Dense Perturbation Slices</h3>
  <p>Entrywise slices used <code>delta in [{DENSE_DELTA_GRID[0]:.6g}, {DENSE_DELTA_GRID[-1]:.6g}]</code> with {len(DENSE_DELTA_GRID)} grid points. For <code>K00</code>, <code>K01</code>, <code>K10</code>, and <code>K11</code>, exactly one entry was perturbed and every other entry was kept at that trajectory's <code>K_true</code>.</p>
  <p>The trace/c slice added <code>[[0,0],[0,delta]]</code> to <code>K_true</code>, changing only <code>K11</code>. The radius/determinant slice rebuilt the companion-form operator with fixed true angle and perturbed radius: <code>K(1+delta, theta_true) = [[0,1],[-(1+delta)^2, 2(1+delta)cos(theta_true)]]</code>.</p>
  <p><code>random_plane_0</code> and <code>random_plane_1</code> are two random two-dimensional affine planes through <code>K_true</code>. For each plane, basis matrix <code>a</code> was sampled from a standard normal 2x2 matrix and normalized in Frobenius norm; basis matrix <code>b</code> was sampled independently, orthogonalized against <code>a</code> by Gram-Schmidt, and normalized. Plane coefficients used <code>delta_a, delta_b in [-0.015, 0.015]</code> with 41 points per axis. The RNG seed was <code>{SEED}</code>.</p>
  <h3>Covariance Objective</h3>
  <p><code>covariance_only</code> did not compute a closed-form least-squares solution. It optimized <code>K</code> directly with Adam on <code>L_cov = ||K C00 - C10||_F^2</code>, where <code>C00 = X X^T / m</code> and <code>C10 = Y X^T / m</code> from observed prefix transitions <code>s_t -> s_{{t+1}}</code>. This objective is the covariance normal-equation form corresponding to local transition fitting, but the diagnostic used gradient optimization to compare the landscape signal under the same trainable-K setup.</p>
  <h3>Basin Sweep Training Details</h3>
  <p>The K basin sweep used Adam with learning rate <code>{TRAIN_LR:g}</code>. Rollout/covariance protocols used <code>{TRAIN_STEPS}</code> optimization steps; covariance pretraining used <code>{COV_PRETRAIN_STEPS}</code> additional covariance-only steps before the rollout curriculum. The curriculum stages were <code>{escape(str(CURRICULUM_STAGES))}</code>. To keep this diagnostic short, the basin sweep used 16 representative test trajectories; the fixed-K landscapes used the full test set.</p>
  <p>Success classification thresholds were: <code>exact_recovery</code> if rollout MSE &lt; <code>1e-12</code> and mean Frobenius K error &lt; <code>1e-8</code>; <code>near_recovery</code> if rollout MSE &lt; <code>1e-8</code>. Runs with nonfinite loss or mean spectral radius above <code>1.2</code> were classified as unstable/divergent. Remaining failures were attributed by comparing eigen-angle, determinant/radius, and trace errors.</p>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions, omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    positions = positions.to(dtype=DTYPE)
    omegas = omegas.to(dtype=DTYPE)
    true_states = _rollout_truth(positions, "full_10_period")
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)

    setup = {
        "diagnostic": "stage1B_k_landscape_and_basin_analysis",
        "dtype": str(DTYPE),
        "position_only_state": "s_t=(x_{t-1},x_t)",
        "uses_neural_network": False,
        "uses_velocity": False,
        "uses_omega_labels_for_fitting": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder_in_model": False,
        "delta_t": float(DELTA_T),
        "num_test_trajectories": int(DIAGNOSTIC_NUM_TEST),
        "obs_prefix_transitions": OBS_PREFIX_TRANSITIONS,
        "rollout_periods": ROLLOUT_PERIODS,
        "theta_delta_grid": [float(THETA_DELTA_GRID[0]), float(THETA_DELTA_GRID[-1]), len(THETA_DELTA_GRID)],
        "r_grid": [float(R_GRID[0]), float(R_GRID[-1]), len(R_GRID)],
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    single_rows = _single_trajectory_landscapes(true_states, omegas)
    basin_width_rows, _ = _aggregate_landscape(true_states, omegas)
    dense_summary_rows = _dense_perturbation_slices(true_states, omegas)
    basin_summary_rows = _basin_sweep(prefix_in, prefix_out, true_states, omegas)

    _write_csv(OUTPUT_DIR / "single_trajectory_landscape_summary.csv", single_rows)
    _write_json(
        OUTPUT_DIR / "metrics.json",
        {
            "setup": setup,
            "single_trajectory_landscapes": single_rows,
            "basin_widths": basin_width_rows,
            "dense_perturbation_summary": dense_summary_rows,
            "basin_sweep_summary": basin_summary_rows,
        },
    )
    _write_html(single_rows, basin_width_rows, dense_summary_rows, basin_summary_rows)


if __name__ == "__main__":
    main()
