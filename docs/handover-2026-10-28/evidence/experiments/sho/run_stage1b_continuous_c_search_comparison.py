"""Stage 1B continuous prefix-only search comparison for trajectory c values."""

from __future__ import annotations

import math
import json
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
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, _make_data, _steps
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import _prefix_loss_vector
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/continuous_c_search_comparison")
NUM_TRAJECTORIES = int(os.environ.get("CSEARCH_NUM_TRAJECTORIES", "32"))
LOWER = -1.99
UPPER = 1.99
ADAM_LR = 1e-3
ADAM_MILESTONES = (10, 20, 50, 100, 200, 500)
ADAM_INITIALIZATIONS = (-1.0, 0.0, 1.0)
FREE_STEPS = int(os.environ.get("CSEARCH_FREE_STEPS", "1000"))
FREE_LR = float(os.environ.get("CSEARCH_FREE_LR", "1e-2"))
FREE_GRID_POINTS = int(os.environ.get("CSEARCH_FREE_GRID_POINTS", "256"))
GRID_POINTS = int(os.environ.get("CSEARCH_LANDSCAPE_POINTS", "2001"))
COARSE_POINTS = 16
COARSE_LEVELS = 6


def _losses(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return _prefix_loss_vector(c, positions)


def _free_c_diagnostic(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Prefix-only global grid initialization makes this a reliable numerical
    # floor even when Adam's generic starts fall into other local basins.
    n = positions.shape[0]
    grid = torch.linspace(LOWER, UPPER, FREE_GRID_POINTS, dtype=DTYPE)
    initial = []
    for trajectory in range(n):
        parts = [_losses(chunk, positions[trajectory : trajectory + 1].expand(chunk.shape[0], -1)) for chunk in grid.split(64)]
        initial.append(grid[torch.argmin(torch.cat(parts))])
    c = nn.Parameter(torch.stack(initial))
    optimizer = torch.optim.Adam([c], lr=FREE_LR)
    best_c = c.detach().clone()
    best_loss = _losses(c, positions).detach()
    for _ in range(FREE_STEPS):
        optimizer.zero_grad()
        losses = _losses(c, positions)
        losses.sum().backward()
        optimizer.step()
        with torch.no_grad():
            c.clamp_(LOWER, UPPER)
            current = _losses(c, positions)
            improved = current < best_loss
            best_loss[improved] = current[improved]
            best_c[improved] = c.detach()[improved]
    return best_c, best_loss


def _future_metrics(c: float, positions: torch.Tensor) -> tuple[float, float]:
    steps = _steps(10.0)
    a = positions[P - 2]
    b = positions[P - 1]
    predictions = []
    for _ in range(steps):
        nxt = c * b - a
        predictions.append(nxt)
        a, b = b, nxt
    prediction = torch.stack(predictions)
    error = prediction - positions[P : P + steps]
    return float(torch.mean(error**2).item()), float(torch.abs(error).max().item())


def _result(method: str, trajectory: int, c: float, prefix_loss: float, evaluations: int, runtime: float, free_c: torch.Tensor, positions: torch.Tensor, **extra: Any) -> dict[str, Any]:
    rollout_mse, maximum = _future_metrics(c, positions[trajectory])
    row: dict[str, Any] = {
        "method": method, "trajectory_index": trajectory, "loss_evaluations": evaluations,
        "c_found": c, "c_free": float(free_c[trajectory].item()), "abs_c_error": abs(c - float(free_c[trajectory].item())),
        "prefix_loss": prefix_loss, "rollout_10_period_mse": rollout_mse, "max_trajectory_error": maximum,
        "runtime_seconds_per_trajectory": runtime,
    }
    row.update(extra)
    return row


def _adam_search(positions: torch.Tensor, free_c: torch.Tensor) -> list[dict[str, Any]]:
    n = positions.shape[0]
    repeated = positions.repeat(len(ADAM_INITIALIZATIONS), 1)
    initial = torch.tensor(ADAM_INITIALIZATIONS, dtype=DTYPE).repeat_interleave(n)
    c = nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([c], lr=ADAM_LR)
    best_c = c.detach().clone(); best_loss = _losses(c, repeated).detach(); evaluations = 1
    rows: list[dict[str, Any]] = []; start = time.perf_counter()
    for step in range(1, max(ADAM_MILESTONES) + 1):
        optimizer.zero_grad(); losses = _losses(c, repeated); losses.sum().backward(); optimizer.step()
        with torch.no_grad(): c.clamp_(LOWER, UPPER)
        if step in ADAM_MILESTONES:
            with torch.no_grad(): current = _losses(c, repeated)
            evaluations += 1
            improved = current < best_loss; best_loss[improved] = current[improved]; best_c[improved] = c.detach()[improved]
            elapsed_per_trajectory = (time.perf_counter() - start) / n
            for init_index, initialization in enumerate(ADAM_INITIALIZATIONS):
                offset = init_index * n
                for trajectory in range(n):
                    idx = offset + trajectory
                    rows.append(_result("adam", trajectory, float(best_c[idx].item()), float(best_loss[idx].item()), step + evaluations, elapsed_per_trajectory, free_c, positions, initialization=initialization, adam_steps=step, coarse_level="", local_minima_count=""))
    return rows


def _brent_search(positions: torch.Tensor, free_c: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in range(positions.shape[0]):
        position = positions[trajectory : trajectory + 1]
        count = 0; start = time.perf_counter()
        def objective(value: float) -> float:
            nonlocal count
            count += 1
            return float(_losses(torch.tensor([value], dtype=DTYPE), position)[0].item())
        result = minimize_scalar(objective, bounds=(LOWER, UPPER), method="bounded", options={"xatol": 1e-12, "maxiter": 500})
        runtime = time.perf_counter() - start
        rows.append(_result("bounded_brent", trajectory, float(result.x), float(result.fun), count, runtime, free_c, positions, initialization="", adam_steps="", coarse_level="", local_minima_count="", optimizer_success=bool(result.success)))
    return rows


def _coarse_to_fine(positions: torch.Tensor, free_c: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in range(positions.shape[0]):
        low, high = LOWER, UPPER; best_c = 0.0; best_loss = float("inf"); start = time.perf_counter()
        for level in range(1, COARSE_LEVELS + 1):
            grid = torch.linspace(low, high, COARSE_POINTS, dtype=DTYPE)
            repeated = positions[trajectory : trajectory + 1].expand(COARSE_POINTS, -1)
            losses = _losses(grid, repeated)
            index = int(torch.argmin(losses).item())
            if float(losses[index].item()) < best_loss:
                best_loss = float(losses[index].item()); best_c = float(grid[index].item())
            next_low = float(grid[max(0, index - 1)].item()); next_high = float(grid[min(COARSE_POINTS - 1, index + 1)].item())
            low, high = next_low, next_high
            rows.append(_result("coarse_to_fine", trajectory, best_c, best_loss, COARSE_POINTS * level, time.perf_counter() - start, free_c, positions, initialization="", adam_steps="", coarse_level=level, local_minima_count=""))
    return rows


def _landscapes(positions: torch.Tensor, free_c: torch.Tensor) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    grid = torch.linspace(LOWER, UPPER, GRID_POINTS, dtype=DTYPE)
    rows: list[dict[str, Any]] = []; curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for trajectory in range(3):
        parts = []
        start = time.perf_counter()
        for chunk in grid.split(128):
            parts.append(_losses(chunk, positions[trajectory : trajectory + 1].expand(chunk.shape[0], -1)))
        losses = torch.cat(parts)
        interior = (losses[1:-1] < losses[:-2]) & (losses[1:-1] < losses[2:])
        local_minima = int(interior.sum().item())
        index = int(torch.argmin(losses).item())
        rows.append(_result("landscape_grid", trajectory, float(grid[index].item()), float(losses[index].item()), GRID_POINTS, (time.perf_counter() - start), free_c, positions, initialization="", adam_steps="", coarse_level="", local_minima_count=local_minima))
        curves[trajectory] = (grid.numpy(), losses.numpy())
    return rows, curves


def _plot_landscapes(curves: dict[int, tuple[np.ndarray, np.ndarray]], rows: list[dict[str, Any]], free_c: torch.Tensor) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for trajectory, axis in enumerate(axes):
        grid, loss = curves[trajectory]; axis.plot(grid, np.maximum(loss, 1e-30), color="navy", linewidth=1)
        axis.axvline(float(free_c[trajectory].item()), color="black", linestyle="--", label="free-c diagnostic")
        markers = [("bounded_brent", None, "Brent", "o"), ("coarse_to_fine", 6, "coarse-to-fine L6", "s")]
        for method, level, label, marker in markers:
            candidates = [row for row in rows if row["trajectory_index"] == trajectory and row["method"] == method and (level is None or row["coarse_level"] == level)]
            if candidates:
                axis.scatter(candidates[-1]["c_found"], max(float(candidates[-1]["prefix_loss"]), 1e-30), marker=marker, s=42, label=label)
        for initialization, color in zip(ADAM_INITIALIZATIONS, ("tab:red", "tab:orange", "tab:green")):
            candidates = [row for row in rows if row["trajectory_index"] == trajectory and row["method"] == "adam" and row["initialization"] == initialization and row["adam_steps"] == 500]
            axis.scatter(candidates[0]["c_found"], max(float(candidates[0]["prefix_loss"]), 1e-30), marker="x", color=color, label=f"Adam c0={initialization:g}")
        axis.set_yscale("log"); axis.set_ylabel(f"trajectory {trajectory}\nprefix loss"); axis.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("c"); figure.tight_layout(); figure.savefig(OUTPUT_DIR / ".leaving.tmp.png", dpi=170)
    plt.close(figure)


def _plot_efficiency(rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    series = []
    for initialization in ADAM_INITIALIZATIONS:
        series.append((f"Adam c0={initialization:g}", [row for row in rows if row["method"] == "adam" and row["initialization"] == initialization]))
    series.extend((("bounded Brent", [row for row in rows if row["method"] == "bounded_brent"]), ("coarse-to-fine", [row for row in rows if row["method"] == "coarse_to_fine"])))
    for label, subset in series:
        evaluations = sorted({int(row["loss_evaluations"]) for row in subset})
        x, c_values, rollout_values = [], [], []
        for count in evaluations:
            group = [row for row in subset if int(row["loss_evaluations"]) == count]
            x.append(count); c_values.append(max(float(row["abs_c_error"]) for row in group)); rollout_values.append(max(float(row["rollout_10_period_mse"]) for row in group))
        axes[0].plot(x, c_values, marker="o", label=label); axes[1].plot(x, rollout_values, marker="o", label=label)
    for axis in axes: axis.set_xscale("log"); axis.set_yscale("log"); axis.set_xlabel("prefix-loss evaluations"); axis.legend(fontsize=8)
    axes[0].set_ylabel("worst |c - c_free|"); axes[1].set_ylabel("worst 10-period rollout MSE")
    figure.tight_layout(); figure.savefig(OUTPUT_DIR / "accuracy_vs_loss_evaluations.png", dpi=170); plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _make_data(NUM_TRAJECTORIES, TEST_DATA_SEED + 1297000)
    print("building prefix-only free-c numerical floor", flush=True)
    free_c, free_loss = _free_c_diagnostic(data.positions[:, :P])
    rows = [_result("free_c_diagnostic", i, float(free_c[i].item()), float(free_loss[i].item()), FREE_GRID_POINTS + FREE_STEPS + 1, 0.0, free_c, data.positions, initialization="prefix_grid", adam_steps=FREE_STEPS, coarse_level="", local_minima_count="") for i in range(NUM_TRAJECTORIES)]
    print("running Adam searches", flush=True); rows.extend(_adam_search(data.positions, free_c))
    print("running bounded Brent searches", flush=True); rows.extend(_brent_search(data.positions, free_c))
    print("running coarse-to-fine searches", flush=True); rows.extend(_coarse_to_fine(data.positions, free_c))
    print("evaluating dense landscapes", flush=True); landscape_rows, curves = _landscapes(data.positions, free_c); rows.extend(landscape_rows)
    _write_csv(OUTPUT_DIR / "results.csv", rows)
    _plot_landscapes(curves, rows, free_c)
    (OUTPUT_DIR / ".leaving.tmp.png").replace(OUTPUT_DIR / "prefix_loss_landscape.png")
    _plot_efficiency(rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "trajectories": NUM_TRAJECTORIES, "result_rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
