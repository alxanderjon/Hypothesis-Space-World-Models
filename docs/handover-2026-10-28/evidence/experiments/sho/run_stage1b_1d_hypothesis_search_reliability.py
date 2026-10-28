"""Reliability test for global-basin selection followed by local 1D refinement."""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize_scalar
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, _make_data
from experiments.sho.run_stage1b_continuous_c_search_comparison import LOWER, UPPER, _free_c_diagnostic, _future_metrics, _losses
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/hypothesis_search_1d_reliability")
NUM_TRAJECTORIES = int(os.environ.get("SEARCHREL_NUM_TRAJECTORIES", "256"))
COARSE_POINTS = 16
ADAM_LR = 1e-3
ADAM_CHECKPOINTS = (10, 20, 50, 100, 200)
BASIN_THRESHOLD = 1e-3
PRECISION_THRESHOLD = 1e-6


def _row(method: str, checkpoint: str, trajectory: int, c: float, loss: float, coarse_evals: int, local_evals: int, runtime: float, free_c: torch.Tensor, positions: torch.Tensor, interval: tuple[float, float] | None = None) -> dict[str, Any]:
    error = abs(c - float(free_c[trajectory].item()))
    rollout_mse, max_trajectory_error = _future_metrics(c, positions[trajectory])
    return {
        "method": method, "checkpoint": checkpoint, "trajectory_index": trajectory,
        "coarse_evaluations": coarse_evals, "local_evaluations": local_evals, "total_loss_evaluations": coarse_evals + local_evals,
        "selected_interval_low": "" if interval is None else interval[0], "selected_interval_high": "" if interval is None else interval[1],
        "c_free": float(free_c[trajectory].item()), "c_found": c, "abs_c_error": error,
        "basin_success": error <= BASIN_THRESHOLD, "precision_success": error <= PRECISION_THRESHOLD,
        "below_1e-3": error <= 1e-3, "below_1e-6": error <= 1e-6, "below_1e-9": error <= 1e-9, "below_1e-12": error <= 1e-12,
        "prefix_loss": loss, "rollout_10_period_mse": rollout_mse, "max_trajectory_error": max_trajectory_error,
        "runtime_seconds": runtime,
    }


def _full_brent(positions: torch.Tensor, free_c: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    for trajectory in range(positions.shape[0]):
        position = positions[trajectory : trajectory + 1]; count = 0; start = time.perf_counter()
        def objective(value: float) -> float:
            nonlocal count
            count += 1
            return float(_losses(torch.tensor([value], dtype=DTYPE), position)[0].item())
        result = minimize_scalar(objective, bounds=(LOWER, UPPER), method="bounded", options={"xatol": 1e-12, "maxiter": 500})
        rows.append(_row("full_interval_bounded_brent", "final", trajectory, float(result.x), float(result.fun), 0, count, time.perf_counter() - start, free_c, positions))
    return rows


def _coarse_intervals(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    n = positions.shape[0]; grid = torch.linspace(LOWER, UPPER, COARSE_POINTS, dtype=DTYPE)
    best_c = torch.empty(n, dtype=DTYPE); best_loss = torch.empty(n, dtype=DTYPE); lows = torch.empty(n, dtype=DTYPE); highs = torch.empty(n, dtype=DTYPE)
    start = time.perf_counter()
    for trajectory in range(n):
        losses = _losses(grid, positions[trajectory : trajectory + 1].expand(COARSE_POINTS, -1))
        index = int(torch.argmin(losses).item())
        best_c[trajectory] = grid[index]; best_loss[trajectory] = losses[index]
        lows[trajectory] = grid[max(0, index - 1)]; highs[trajectory] = grid[min(COARSE_POINTS - 1, index + 1)]
    return best_c, best_loss, lows, highs, (time.perf_counter() - start) / n


def _coarse_brent(positions: torch.Tensor, free_c: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor, coarse_runtime: float) -> list[dict[str, Any]]:
    rows = []
    for trajectory in range(positions.shape[0]):
        position = positions[trajectory : trajectory + 1]; count = 0; interval = (float(lows[trajectory].item()), float(highs[trajectory].item())); start = time.perf_counter()
        def objective(value: float) -> float:
            nonlocal count
            count += 1
            return float(_losses(torch.tensor([value], dtype=DTYPE), position)[0].item())
        result = minimize_scalar(objective, bounds=interval, method="bounded", options={"xatol": 1e-12, "maxiter": 500})
        rows.append(_row("coarse_to_bounded_brent", "final", trajectory, float(result.x), float(result.fun), COARSE_POINTS, count, coarse_runtime + time.perf_counter() - start, free_c, positions, interval))
    return rows


def _coarse_adam(positions: torch.Tensor, free_c: torch.Tensor, initial: torch.Tensor, initial_loss: torch.Tensor, lows: torch.Tensor, highs: torch.Tensor, coarse_runtime: float) -> list[dict[str, Any]]:
    c = nn.Parameter(initial.clone()); optimizer = torch.optim.Adam([c], lr=ADAM_LR)
    best_c = initial.clone(); best_loss = initial_loss.clone(); rows: list[dict[str, Any]] = []; extra_evaluations = 0; start = time.perf_counter()
    for step in range(1, max(ADAM_CHECKPOINTS) + 1):
        optimizer.zero_grad(); loss = _losses(c, positions)
        with torch.no_grad():
            improved = loss < best_loss
            best_loss[improved] = loss[improved]
            best_c[improved] = c.detach()[improved]
        loss.sum().backward(); optimizer.step()
        with torch.no_grad(): c.copy_(torch.maximum(lows, torch.minimum(highs, c)))
        if step in ADAM_CHECKPOINTS:
            with torch.no_grad(): current = _losses(c, positions)
            extra_evaluations += 1
            improved = current < best_loss; best_loss[improved] = current[improved]; best_c[improved] = c.detach()[improved]
            runtime = coarse_runtime + (time.perf_counter() - start) / positions.shape[0]
            for trajectory in range(positions.shape[0]):
                interval = (float(lows[trajectory].item()), float(highs[trajectory].item()))
                rows.append(_row("coarse_to_local_adam", f"adam_{step}", trajectory, float(best_c[trajectory].item()), float(best_loss[trajectory].item()), COARSE_POINTS, step + extra_evaluations, runtime, free_c, positions, interval))
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    keys = []
    for row in rows:
        key = (row["method"], row["checkpoint"])
        if key not in keys:
            keys.append(key)
    for method, checkpoint in keys:
        group = [row for row in rows if row["method"] == method and row["checkpoint"] == checkpoint]
        errors = [float(row["abs_c_error"]) for row in group]; evaluations = [int(row["total_loss_evaluations"]) for row in group]; runtimes = [float(row["runtime_seconds"]) for row in group]
        summary.append({
            "method": method, "checkpoint": checkpoint, "trajectories": len(group),
            "basin_success_count": sum(bool(row["basin_success"]) for row in group), "basin_success_fraction": sum(bool(row["basin_success"]) for row in group) / len(group),
            "precision_success_count": sum(bool(row["precision_success"]) for row in group), "precision_success_fraction": sum(bool(row["precision_success"]) for row in group) / len(group),
            "count_below_1e-3": sum(bool(row["below_1e-3"]) for row in group), "count_below_1e-6": sum(bool(row["below_1e-6"]) for row in group), "count_below_1e-9": sum(bool(row["below_1e-9"]) for row in group), "count_below_1e-12": sum(bool(row["below_1e-12"]) for row in group),
            "median_abs_c_error": statistics.median(errors), "worst_abs_c_error": max(errors),
            "worst_rollout_10_period_mse": max(float(row["rollout_10_period_mse"]) for row in group), "worst_max_trajectory_error": max(float(row["max_trajectory_error"]) for row in group),
            "median_loss_evaluations": statistics.median(evaluations), "max_loss_evaluations": max(evaluations),
            "median_runtime_seconds": statistics.median(runtimes),
        })
    return summary


def _plot(summary: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    styles = {"full_interval_bounded_brent": ("o", "tab:blue"), "coarse_to_bounded_brent": ("s", "tab:green"), "coarse_to_local_adam": ("^", "tab:orange")}
    for method in styles:
        group = [row for row in summary if row["method"] == method]
        group.sort(key=lambda row: float(row["median_loss_evaluations"]))
        x = [float(row["median_loss_evaluations"]) for row in group]; median = [float(row["median_abs_c_error"]) for row in group]; worst = [float(row["worst_abs_c_error"]) for row in group]
        marker, color = styles[method]; label = method.replace("_", " ")
        axis.plot(x, median, marker=marker, color=color, label=f"{label}: median")
        axis.plot(x, worst, marker=marker, color=color, linestyle="--", alpha=.75, label=f"{label}: worst")
    axis.axhline(1e-3, color="gray", linestyle=":", linewidth=1); axis.axhline(1e-6, color="black", linestyle=":", linewidth=1)
    axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("median prefix-loss evaluations"); axis.set_ylabel("absolute c error"); axis.legend(fontsize=8); figure.tight_layout(); figure.savefig(OUTPUT_DIR / "c_error_vs_evaluations.png", dpi=170); plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _make_data(NUM_TRAJECTORIES, TEST_DATA_SEED + 1408000)
    print("building independent prefix-only c_free references", flush=True)
    free_c, _ = _free_c_diagnostic(data.positions[:, :P])
    print("Method A: full-interval bounded Brent", flush=True); rows = _full_brent(data.positions, free_c)
    print("shared 16-point coarse basin search", flush=True); initial, initial_loss, lows, highs, coarse_runtime = _coarse_intervals(data.positions)
    print("Method B: coarse search -> bounded Brent", flush=True); rows.extend(_coarse_brent(data.positions, free_c, lows, highs, coarse_runtime))
    print("Method C: coarse search -> local Adam", flush=True); rows.extend(_coarse_adam(data.positions, free_c, initial, initial_loss, lows, highs, coarse_runtime))
    summary = _summaries(rows); _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "per_trajectory_results.csv", rows); _plot(summary)
    print({"output_dir": str(OUTPUT_DIR), "trajectories": NUM_TRAJECTORIES, "rows": len(rows)}, flush=True)


if __name__ == "__main__":
    main()
