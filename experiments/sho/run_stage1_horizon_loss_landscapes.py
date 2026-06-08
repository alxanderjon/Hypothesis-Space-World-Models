"""Compare frozen-W Stage 1 omega loss landscapes across rollout horizons."""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from hypothesis_space_world_models.systems import generate_sho_data


TRUE_OMEGA = 2.0
OUTPUT_DIR = Path("outputs/sho/stage1/horizon_loss_landscapes")
HORIZONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _horizon_label(horizon: float) -> str:
    return f"{horizon:g}".replace(".", "p")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _trajectory_mse_for_omega_grid(
    coefficients: torch.Tensor,
    times: torch.Tensor,
    target_trajectories: torch.Tensor,
    omega_grid: np.ndarray,
    batch_size: int = 250,
) -> np.ndarray:
    """Evaluate fixed-coefficient trajectory MSE for a dense omega grid."""
    coefficients = coefficients.to(dtype=torch.float64)
    times = times.to(dtype=torch.float64)
    target_trajectories = target_trajectories.to(dtype=torch.float64)
    a = coefficients[:, 0].view(1, -1, 1)
    b = coefficients[:, 1].view(1, -1, 1)
    target = target_trajectories.view(1, *target_trajectories.shape)

    mse_values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(omega_grid), batch_size):
            omega_batch = torch.tensor(
                omega_grid[start : start + batch_size],
                dtype=torch.float64,
                device=times.device,
            ).view(-1, 1, 1)
            phases = omega_batch * times.view(1, 1, -1)
            prediction = a * torch.cos(phases) + b * torch.sin(phases)
            mse = torch.mean((prediction - target) ** 2, dim=(1, 2))
            mse_values.extend(float(value) for value in mse.cpu())
    return np.asarray(mse_values)


def _local_minima(omega_grid: np.ndarray, mse_values: np.ndarray) -> list[dict[str, float]]:
    minima = []
    for index in range(1, len(mse_values) - 1):
        if mse_values[index] <= mse_values[index - 1] and mse_values[index] <= mse_values[index + 1]:
            minima.append(
                {
                    "index": index,
                    "omega": float(omega_grid[index]),
                    "test_mse": float(mse_values[index]),
                }
            )
    minima.sort(key=lambda row: row["test_mse"])
    return minima


def _basin_width_around_true(
    omega_grid: np.ndarray,
    mse_values: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    true_index = int(np.argmin(np.abs(omega_grid - TRUE_OMEGA)))
    below = mse_values < threshold
    if not below[true_index]:
        return {
            "min": None,
            "max": None,
            "width": 0.0,
        }

    left = true_index
    while left > 0 and below[left - 1]:
        left -= 1
    right = true_index
    while right < len(below) - 1 and below[right + 1]:
        right += 1

    return {
        "min": float(omega_grid[left]),
        "max": float(omega_grid[right]),
        "width": float(omega_grid[right] - omega_grid[left]),
    }


def _mse_at_true_omega(data: Any) -> float:
    coefficients = data.coefficients.to(dtype=torch.float64)
    times = data.times.to(dtype=torch.float64)
    target = data.trajectories.to(dtype=torch.float64)
    a = coefficients[:, 0].unsqueeze(-1)
    b = coefficients[:, 1].unsqueeze(-1)
    prediction = a * torch.cos(TRUE_OMEGA * times) + b * torch.sin(TRUE_OMEGA * times)
    return float(torch.mean((prediction - target) ** 2).cpu())


def _secondary_minimum(
    omega_grid: np.ndarray,
    mse_values: np.ndarray,
) -> dict[str, float | None]:
    minima = _local_minima(omega_grid, mse_values)
    if len(minima) < 2:
        return {"omega": None, "test_mse": None}
    best_index = int(np.argmin(mse_values))
    step = float(omega_grid[1] - omega_grid[0])
    for minimum in minima:
        if abs(int(minimum["index"]) - best_index) > 3 and abs(minimum["omega"] - TRUE_OMEGA) > 3 * step:
            return {
                "omega": float(minimum["omega"]),
                "test_mse": float(minimum["test_mse"]),
            }
    return {"omega": None, "test_mse": None}


def _plot_landscape(
    omega_grid: np.ndarray,
    mse_values: np.ndarray,
    horizon: float,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(omega_grid, mse_values, linewidth=1.5)
    plt.axvline(TRUE_OMEGA, color="black", linestyle="--", label="true omega")
    plt.xlabel("omega")
    plt.ylabel("test trajectory MSE")
    plt.title(f"Stage 1 Frozen W: Omega Landscape, {horizon:g} Periods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"omega_landscape_{_horizon_label(horizon)}.png")
    plt.close()


def _plot_overlay(
    omega_grid: np.ndarray,
    landscapes: dict[float, np.ndarray],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for horizon, mse_values in landscapes.items():
        shifted = mse_values - np.min(mse_values)
        denominator = np.max(shifted)
        normalized = shifted / denominator if denominator > 0.0 else shifted
        plt.plot(omega_grid, normalized, label=f"{horizon:g} periods")
    plt.axvline(TRUE_OMEGA, color="black", linestyle="--", label="true omega")
    plt.xlabel("omega")
    plt.ylabel("normalized test MSE")
    plt.title("Stage 1 Frozen W: Normalized Omega Landscapes by Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "landscape_comparison_overlay.png")
    plt.close()


def _plot_basin_widths(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    horizons = [float(row["horizon_periods"]) for row in rows]
    widths_01 = [float(row["basin_width_mse_lt_0p1"]) for row in rows]
    widths_001 = [float(row["basin_width_mse_lt_0p01"]) for row in rows]
    plt.plot(horizons, widths_01, marker="o", label="MSE < 0.1")
    plt.plot(horizons, widths_001, marker="o", label="MSE < 0.01")
    plt.xscale("log")
    plt.xlabel("horizon in periods")
    plt.ylabel("connected basin width around true omega")
    plt.title("Stage 1 Frozen W: Basin Width vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "basin_width_vs_horizon.png")
    plt.close()


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "horizon_periods",
        "global_min_omega",
        "global_min_mse",
        "mse_at_true_omega",
        "basin_min_mse_lt_0p1",
        "basin_max_mse_lt_0p1",
        "basin_width_mse_lt_0p1",
        "basin_min_mse_lt_0p01",
        "basin_max_mse_lt_0p01",
        "basin_width_mse_lt_0p01",
        "strongest_secondary_min_omega",
        "strongest_secondary_min_mse",
    ]
    with (output_dir / "horizon_loss_landscape_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    landscape_images = "\n".join(
        (
            f'<section><h2>Omega Landscape, {row["horizon_periods"]:g} Periods</h2>'
            f'<img src="omega_landscape_{_horizon_label(float(row["horizon_periods"]))}.png" '
            f'alt="Omega landscape {row["horizon_periods"]:g} periods"></section>'
        )
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Horizon Loss Landscapes</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 1 Horizon Loss Landscapes</h1>
  <p>Frozen true coefficient map, no training, dense omega grid.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <section><h2>Normalized Landscape Overlay</h2><img src="landscape_comparison_overlay.png" alt="normalized landscape overlay"></section>
  <section><h2>Basin Width vs Horizon</h2><img src="basin_width_vs_horizon.png" alt="basin width vs horizon"></section>
  {landscape_images}
</body>
</html>
"""
    (output_dir / "horizon_loss_landscape_summary.html").write_text(html, encoding="utf-8")


def run_horizon_loss_landscapes(output_dir: Path = OUTPUT_DIR) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_test = 128
    num_time_steps = 100
    omega_grid = np.linspace(0.1, 5.0, 5000)
    rows = []
    landscapes: dict[float, np.ndarray] = {}

    setup = {
        "diagnostic": "stage1_horizon_loss_landscapes_frozen_true_W",
        "training": "none",
        "true_omega": TRUE_OMEGA,
        "W_true": [[1.0, 0.0], [0.0, 1.0 / TRUE_OMEGA]],
        "coefficient_map": "a = x0, b = v0 / true_omega",
        "omega_grid": "np.linspace(0.1, 5.0, 5000)",
        "omega_grid_min": float(omega_grid.min()),
        "omega_grid_max": float(omega_grid.max()),
        "omega_grid_size": int(len(omega_grid)),
        "horizons_periods": HORIZONS,
        "num_test_trajectories": num_test,
        "num_time_points_per_horizon": num_time_steps,
        "test_seed": seed + 1,
        "sho_period": math.tau / TRUE_OMEGA,
    }

    for horizon in HORIZONS:
        print(f"evaluating horizon landscape: {horizon:g} periods", flush=True)
        test_data = generate_sho_data(
            num_test,
            num_time_steps,
            omega=TRUE_OMEGA,
            num_periods=horizon,
            seed=seed + 1,
        )
        mse_values = _trajectory_mse_for_omega_grid(
            test_data.coefficients,
            test_data.times,
            test_data.trajectories,
            omega_grid,
        )
        landscapes[horizon] = mse_values
        best_index = int(np.argmin(mse_values))
        basin_01 = _basin_width_around_true(omega_grid, mse_values, 0.1)
        basin_001 = _basin_width_around_true(omega_grid, mse_values, 0.01)
        secondary = _secondary_minimum(omega_grid, mse_values)
        row = {
            "horizon_periods": horizon,
            "global_min_omega": float(omega_grid[best_index]),
            "global_min_mse": float(mse_values[best_index]),
            "mse_at_true_omega": _mse_at_true_omega(test_data),
            "basin_min_mse_lt_0p1": basin_01["min"],
            "basin_max_mse_lt_0p1": basin_01["max"],
            "basin_width_mse_lt_0p1": basin_01["width"],
            "basin_min_mse_lt_0p01": basin_001["min"],
            "basin_max_mse_lt_0p01": basin_001["max"],
            "basin_width_mse_lt_0p01": basin_001["width"],
            "strongest_secondary_min_omega": secondary["omega"],
            "strongest_secondary_min_mse": secondary["test_mse"],
        }
        rows.append(row)
        _plot_landscape(omega_grid, mse_values, horizon, output_dir)

    _plot_overlay(omega_grid, landscapes, output_dir)
    _plot_basin_widths(rows, output_dir)
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", {"horizons": rows})
    _write_summary(rows, output_dir)
    return rows


def main() -> None:
    rows = run_horizon_loss_landscapes()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
