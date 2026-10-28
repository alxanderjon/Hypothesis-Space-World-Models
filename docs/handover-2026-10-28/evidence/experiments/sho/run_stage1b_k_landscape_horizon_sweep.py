"""Fixed-K horizon landscape sweep for Stage 1B SHO operators."""

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
from experiments.sho.run_stage1b_k_landscape_and_basin_analysis import (
    BASIN_THRESHOLDS,
    DTYPE,
    R_GRID,
    SEED,
    THETA_DELTA_GRID,
    _aggregate_losses_phase_radius_np,
    _basin_width,
    _plot_heatmap,
    _write_csv,
    _write_json,
)
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import TEST_DATA_SEED, _make_prefix_future_data
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/k_landscape_horizon_sweep")
HORIZONS_PERIODS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _states_for_horizon(positions: torch.Tensor, steps: int) -> torch.Tensor:
    x_prev = positions[:, : steps + 1]
    x_now = positions[:, 1 : steps + 2]
    return torch.stack((x_prev, x_now), dim=-1)


def _period_horizon_steps(periods: float) -> int:
    omega_min = float(OMEGA_RANGE[0])
    return int(math.ceil(periods * math.tau / (omega_min * float(DELTA_T))))


def _horizon_label(periods: float) -> str:
    return str(periods).replace(".", "p")


def _write_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "horizon_periods",
        "rollout_steps",
        "mse_at_true_grid_point",
        "global_min_mse",
        "global_min_delta_theta",
        "global_min_r",
        "basin_width_theta_mse_lt_1e-3",
        "basin_width_r_mse_lt_1e-3",
        "basin_width_theta_mse_lt_1e-6",
        "basin_width_r_mse_lt_1e-6",
        "basin_width_theta_mse_lt_1e-9",
        "basin_width_r_mse_lt_1e-9",
        "basin_width_theta_mse_lt_1e-12",
        "basin_width_r_mse_lt_1e-12",
    ]
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    images = "\n".join(
        f'<section><h3>{periods:g} periods</h3><div class="grid"><img src="landscape_{_horizon_label(periods)}.png"><img src="log_landscape_{_horizon_label(periods)}.png"></div></section>'
        for periods in HORIZONS_PERIODS
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Fixed-K Horizon Landscape Sweep</title>
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
  <h1>Stage 1B Fixed-K Horizon Landscape Sweep</h1>
  <p>For each horizon, candidate <code>K(r,theta)</code> was held fixed, rolled out, and compared against the true analytic position trajectory. The shared aggregate grid used <code>theta_i = theta_true_i + delta_theta</code> and <code>r</code>.</p>
  <table><thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in columns)}</tr></thead><tbody>{body}</tbody></table>
  <h2>Basin Width vs Horizon</h2>
  <img src="basin_width_vs_horizon.png">
  <h2>Heatmaps</h2>
  {images}
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _plot_basin_widths(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8.5, 5.0))
    horizons = [float(row["horizon_periods"]) for row in rows]
    for threshold in BASIN_THRESHOLDS:
        key = f"basin_width_theta_mse_lt_{threshold:.0e}"
        label = f"MSE < {threshold:g}"
        plt.plot(horizons, [float(row[key]) for row in rows], marker="o", label=label)
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel("horizon periods")
    plt.ylabel("connected delta-theta basin width")
    plt.title("K-landscape basin width vs rollout horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basin_width_vs_horizon.png", dpi=160)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_steps = _period_horizon_steps(max(HORIZONS_PERIODS))
    positions, omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    positions = positions[:, : max_steps + 2].to(dtype=DTYPE)
    omegas = omegas.to(dtype=DTYPE)
    omegas_np = omegas.detach().cpu().numpy()

    setup = {
        "diagnostic": "stage1B_k_landscape_horizon_sweep",
        "dtype": str(DTYPE),
        "seed": SEED,
        "position_only_state": "s_t=(x_{t-1},x_t)",
        "candidate_operator": "K(r,theta)=[[0,1],[-r^2,2*r*cos(theta)]]",
        "theta_delta_grid": [float(THETA_DELTA_GRID[0]), float(THETA_DELTA_GRID[-1]), len(THETA_DELTA_GRID)],
        "r_grid": [float(R_GRID[0]), float(R_GRID[-1]), len(R_GRID)],
        "thresholds": BASIN_THRESHOLDS,
        "horizons_periods": HORIZONS_PERIODS,
        "uses_neural_network": False,
        "uses_velocity": False,
        "uses_omega_labels_for_fitting": False,
        "uses_coefficient_labels": False,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    center_x = int(np.argmin(np.abs(THETA_DELTA_GRID)))
    center_y = int(np.argmin(np.abs(R_GRID - 1.0)))
    for periods in HORIZONS_PERIODS:
        steps = _period_horizon_steps(periods)
        states = _states_for_horizon(positions, steps).detach().cpu().numpy()
        losses = _aggregate_losses_phase_radius_np(states, omegas_np, R_GRID, THETA_DELTA_GRID)
        np.save(OUTPUT_DIR / f"landscape_{_horizon_label(periods)}.npy", losses)
        _plot_heatmap(
            losses,
            THETA_DELTA_GRID,
            R_GRID,
            OUTPUT_DIR / f"landscape_{_horizon_label(periods)}.png",
            title=f"Fixed-K landscape: {periods:g} periods",
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
            OUTPUT_DIR / f"log_landscape_{_horizon_label(periods)}.png",
            title=f"Fixed-K log-landscape: {periods:g} periods",
            xlabel="delta theta",
            ylabel="radius r",
            true_x=0.0,
            true_y=1.0,
            log_scale=True,
        )
        min_idx = np.unravel_index(int(np.argmin(losses)), losses.shape)
        row: dict[str, Any] = {
            "horizon_periods": periods,
            "rollout_steps": steps,
            "mse_at_true_grid_point": float(losses[center_y, center_x]),
            "global_min_mse": float(losses[min_idx]),
            "global_min_delta_theta": float(THETA_DELTA_GRID[min_idx[1]]),
            "global_min_r": float(R_GRID[min_idx[0]]),
        }
        for threshold in BASIN_THRESHOLDS:
            width = _basin_width(losses < threshold, THETA_DELTA_GRID, R_GRID, center_x, center_y)
            suffix = f"{threshold:.0e}"
            row[f"basin_width_theta_mse_lt_{suffix}"] = width["delta_theta_width"]
            row[f"basin_width_r_mse_lt_{suffix}"] = width["r_width"]
            row[f"grid_fraction_mse_lt_{suffix}"] = float(np.mean(losses < threshold))
        rows.append(row)

    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "rows": rows})
    _plot_basin_widths(rows)
    _write_html(rows)


if __name__ == "__main__":
    main()
