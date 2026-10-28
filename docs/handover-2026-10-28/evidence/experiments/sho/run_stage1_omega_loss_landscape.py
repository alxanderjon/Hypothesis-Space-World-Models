"""Diagnose the Stage 1 omega-only loss landscape with exact coefficients."""

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
from hypothesis_space_world_models.systems.sho import decode_sho


TRUE_OMEGA = 2.0
OUTPUT_DIR = Path("outputs/sho/stage1/omega_loss_landscape")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _trajectory_mse_for_omega_grid(
    coefficients: torch.Tensor,
    times: torch.Tensor,
    target_trajectories: torch.Tensor,
    omega_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate trajectory MSE for each candidate omega with fixed true coefficients."""
    coefficients = coefficients.to(dtype=torch.float64)
    times = times.to(dtype=torch.float64)
    target_trajectories = target_trajectories.to(dtype=torch.float64)

    mse_values = []
    with torch.no_grad():
        for omega in omega_grid:
            prediction = decode_sho(coefficients, times, float(omega))
            mse = torch.mean((prediction - target_trajectories) ** 2)
            mse_values.append(float(mse.cpu()))
    return np.asarray(mse_values)


def _find_local_minima(omega_grid: np.ndarray, mse_values: np.ndarray) -> list[dict[str, float]]:
    minima = []
    for index in range(1, len(mse_values) - 1):
        if mse_values[index] <= mse_values[index - 1] and mse_values[index] <= mse_values[index + 1]:
            minima.append(
                {
                    "omega": float(omega_grid[index]),
                    "test_mse": float(mse_values[index]),
                }
            )
    minima.sort(key=lambda entry: entry["test_mse"])
    return minima


def _plot_landscape(
    omega_grid: np.ndarray,
    mse_values: np.ndarray,
    output_path: Path,
    title: str,
    xlim: tuple[float, float] | None = None,
    logscale: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(omega_grid, mse_values, linewidth=1.8)
    plt.axvline(TRUE_OMEGA, color="black", linestyle="--", label="true omega = 2.0")
    if xlim is not None:
        plt.xlim(*xlim)
    if logscale:
        plt.yscale("log")
    plt.xlabel("omega")
    plt.ylabel("test trajectory MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _write_csv(output_path: Path, omega_grid: np.ndarray, mse_values: np.ndarray) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["omega", "test_trajectory_mse"])
        writer.writeheader()
        for omega, mse in zip(omega_grid, mse_values):
            writer.writerow(
                {
                    "omega": f"{omega:.16g}",
                    "test_trajectory_mse": f"{mse:.16g}",
                }
            )


def _write_html(output_dir: Path, setup: dict[str, Any], summary: dict[str, Any]) -> None:
    def table(data: dict[str, Any]) -> str:
        rows = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, indent=2)
            else:
                value_text = str(value)
            rows.append(
                "<tr>"
                f"<th>{escape(str(key))}</th>"
                f"<td><pre>{escape(value_text)}</pre></td>"
                "</tr>"
            )
        return "<table>" + "\n".join(rows) + "</table>"

    images = [
        ("Omega vs Test MSE", "omega_vs_test_mse.png"),
        ("Omega vs Test MSE, Zoom Around True Omega", "omega_vs_test_mse_zoom_around_true.png"),
        ("Omega vs Test MSE, Log Scale", "omega_vs_test_mse_logscale.png"),
    ]
    image_html = "\n".join(
        f"<section><h2>{escape(title)}</h2><img src=\"{escape(filename)}\" alt=\"{escape(title)}\"></section>"
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SHO Stage 1 Omega Loss Landscape</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 300px; background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>SHO Stage 1 Omega Loss Landscape</h1>
  <h2>Setup</h2>
  {table(setup)}
  <h2>Summary</h2>
  {table(summary)}
  {image_html}
</body>
</html>
"""
    (output_dir / "omega_loss_landscape.html").write_text(html, encoding="utf-8")


def run_omega_loss_landscape(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = 0
    num_test = 128
    num_time_steps = 100
    num_periods = 10.0
    omega_grid = np.linspace(0.1, 5.0, 1000)

    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=TRUE_OMEGA,
        num_periods=num_periods,
        seed=seed + 1,
    )

    fixed_coefficients = test_data.coefficients
    mse_values = _trajectory_mse_for_omega_grid(
        fixed_coefficients,
        test_data.times,
        test_data.trajectories,
        omega_grid,
    )
    mse_at_true_omega = float(
        torch.mean(
            (
                decode_sho(
                    fixed_coefficients.to(dtype=torch.float64),
                    test_data.times.to(dtype=torch.float64),
                    TRUE_OMEGA,
                )
                - test_data.trajectories.to(dtype=torch.float64)
            )
            ** 2
        ).cpu()
    )
    local_minima = _find_local_minima(omega_grid, mse_values)
    best_index = int(np.argmin(mse_values))
    nearby_mask = (omega_grid >= 1.5) & (omega_grid <= 2.5)
    nearby_mse = mse_values[nearby_mask]
    nearby_omega = omega_grid[nearby_mask]
    nearby_best_index = int(np.argmin(nearby_mse))

    setup = {
        "diagnostic": "stage1_omega_loss_landscape_frozen_true_W",
        "true_omega": TRUE_OMEGA,
        "coefficient_map": "a = x0, b = v0 / true_omega",
        "W_true": [[1.0, 0.0], [0.0, 1.0 / TRUE_OMEGA]],
        "training": "none",
        "omega_grid": "np.linspace(0.1, 5.0, 1000)",
        "omega_grid_min": float(omega_grid.min()),
        "omega_grid_max": float(omega_grid.max()),
        "omega_grid_size": int(len(omega_grid)),
        "num_test_trajectories": num_test,
        "num_time_points_per_trajectory": num_time_steps,
        "num_periods_per_trajectory": num_periods,
        "test_seed": seed + 1,
        "test_period": math.tau / TRUE_OMEGA,
    }
    summary = {
        "best_grid_omega": float(omega_grid[best_index]),
        "best_grid_test_mse": float(mse_values[best_index]),
        "mse_at_true_omega": mse_at_true_omega,
        "nearest_grid_omega_to_true": float(omega_grid[np.argmin(np.abs(omega_grid - TRUE_OMEGA))]),
        "nearest_grid_mse_to_true": float(mse_values[np.argmin(np.abs(omega_grid - TRUE_OMEGA))]),
        "best_zoom_omega_1p5_to_2p5": float(nearby_omega[nearby_best_index]),
        "best_zoom_mse_1p5_to_2p5": float(nearby_mse[nearby_best_index]),
        "num_local_minima": len(local_minima),
        "top_10_local_minima": local_minima[:10],
    }

    _write_csv(output_dir / "omega_loss_landscape.csv", omega_grid, mse_values)
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", summary)
    _plot_landscape(
        omega_grid,
        mse_values,
        output_dir / "omega_vs_test_mse.png",
        "SHO Stage 1 Frozen True W: Omega Loss Landscape",
    )
    _plot_landscape(
        omega_grid,
        mse_values,
        output_dir / "omega_vs_test_mse_zoom_around_true.png",
        "SHO Stage 1 Frozen True W: Omega Loss Landscape Near True Omega",
        xlim=(1.5, 2.5),
    )
    _plot_landscape(
        omega_grid,
        mse_values,
        output_dir / "omega_vs_test_mse_logscale.png",
        "SHO Stage 1 Frozen True W: Omega Loss Landscape (Log MSE)",
        logscale=True,
    )
    _write_html(output_dir, setup, summary)
    return summary


def main() -> None:
    summary = run_omega_loss_landscape()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
