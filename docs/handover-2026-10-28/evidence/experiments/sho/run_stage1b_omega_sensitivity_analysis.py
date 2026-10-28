"""Stage 1B omega-sensitivity analysis with true coefficients.

No training, optimization, or curriculum is used. The analysis perturbs only
omega while keeping the true coefficient values fixed, then measures 10-period
trajectory reconstruction error on the Stage 1B variable-omega test set.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.sho.run_stage1b_two_state_omega import (
    NUM_TEST,
    NUM_TIME_STEPS,
    OMEGA_RANGE,
    SEED,
    _build_data_from_base,
    _make_base_conditions,
)


OUTPUT_DIR = Path("outputs/sho/stage1B/omega_sensitivity_analysis")
NUM_PERIODS = 10.0
DELTA_MAGNITUDES = [
    1e-1,
    3e-2,
    1e-2,
    3e-3,
    1e-3,
    3e-4,
    1e-4,
    3e-5,
    1e-5,
    3e-6,
    1e-6,
    3e-7,
    1e-7,
    3e-8,
    1e-8,
]
TRAJECTORY_MSE_TARGETS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-9, 1e-12, 1e-15]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _trajectory_mse_for_delta(data: Any, delta: float) -> torch.Tensor:
    coefficients = data.coefficients
    omega_test = data.omegas + delta
    a = coefficients[:, 0].unsqueeze(-1)
    b = coefficients[:, 1].unsqueeze(-1)
    phases = omega_test.unsqueeze(-1) * data.times
    predicted = a * torch.cos(phases) + b * torch.sin(phases)
    return (predicted - data.trajectories).square().mean(dim=1)


def _threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_abs_delta: dict[float, list[float]] = {}
    for row in rows:
        abs_delta = float(row["abs_delta"])
        if abs_delta == 0.0:
            continue
        by_abs_delta.setdefault(abs_delta, []).append(float(row["mean_trajectory_mse"]))

    threshold_rows = []
    for target in TRAJECTORY_MSE_TARGETS:
        passing = [
            abs_delta
            for abs_delta, mean_mses in by_abs_delta.items()
            if max(mean_mses) < target
        ]
        threshold_rows.append(
            {
                "trajectory_mse_target": target,
                "largest_tested_abs_delta_passing_both_signs": (
                    max(passing) if passing else None
                ),
                "smallest_tested_abs_delta": min(by_abs_delta),
            },
        )
    return threshold_rows


def _stage1b_reference_rows() -> list[dict[str, Any]]:
    rows = []
    for path in Path("outputs/sho/stage1B").glob("**/metrics.json"):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "omega_mse" not in metrics or "mean_abs_omega_error" not in metrics:
            continue
        if "analytic" in str(path):
            continue
        rows.append(
            {
                "run_path": str(path.parent),
                "omega_mse": metrics.get("omega_mse"),
                "mean_abs_omega_error": metrics.get("mean_abs_omega_error"),
                "max_omega_error": metrics.get("max_omega_error"),
                "trajectory_mse": metrics.get("trajectory_mse"),
            },
        )
    rows.sort(key=lambda row: float(row["mean_abs_omega_error"]))
    return rows[:10]


def _save_plots(rows: list[dict[str, Any]]) -> None:
    nonzero_rows = [row for row in rows if float(row["abs_delta"]) > 0.0]
    abs_deltas = [float(row["abs_delta"]) for row in nonzero_rows]
    means = [float(row["mean_trajectory_mse"]) for row in nonzero_rows]
    signs = [float(row["delta"]) > 0.0 for row in nonzero_rows]
    colors = ["#2563eb" if sign else "#dc2626" for sign in signs]

    plt.figure(figsize=(7, 4.5))
    plt.scatter(abs_deltas, means, c=colors, alpha=0.85)
    plt.xlabel("|omega error|")
    plt.ylabel("mean trajectory MSE")
    plt.title("Stage 1B Omega Sensitivity")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_mse_vs_abs_omega_error.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(abs_deltas, means, c=colors, alpha=0.85)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("|omega error|")
    plt.ylabel("mean trajectory MSE")
    plt.title("Stage 1B Omega Sensitivity (log-log)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "log_trajectory_mse_vs_log_abs_omega_error.png")
    plt.close()


def _write_html(
    rows: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> None:
    def table(data_rows: list[dict[str, Any]]) -> str:
        if not data_rows:
            return "<p>N/A</p>"
        columns = list(data_rows[0].keys())
        header = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
            + "</tr>"
            for row in data_rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Omega Sensitivity Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Omega Sensitivity Analysis</h1>
  <p>True coefficients are fixed. Only omega is perturbed. No training or optimization is used.</p>
  <h2>Setup</h2>
  {table([{
      "num_test_trajectories": NUM_TEST,
      "num_time_steps": NUM_TIME_STEPS,
      "num_periods": NUM_PERIODS,
      "omega_range": OMEGA_RANGE,
      "dtype": "float64",
      "seed": SEED + 1,
  }])}
  <h2>Omega Error Thresholds</h2>
  {table(thresholds)}
  <h2>Sensitivity Sweep</h2>
  {table(rows)}
  <h2>Best Existing Learned Stage 1B Omega Errors</h2>
  {table(references)}
  <section><h2>trajectory_mse_vs_abs_omega_error</h2><img src="trajectory_mse_vs_abs_omega_error.png" alt="trajectory_mse_vs_abs_omega_error"></section>
  <section><h2>log_trajectory_mse_vs_log_abs_omega_error</h2><img src="log_trajectory_mse_vs_log_abs_omega_error.png" alt="log_trajectory_mse_vs_log_abs_omega_error"></section>
</body>
</html>
"""
    (OUTPUT_DIR / "omega_sensitivity.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_initial_conditions, test_omegas = _make_base_conditions(
        NUM_TEST,
        seed=SEED + 1,
        dtype=torch.float64,
    )
    data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=NUM_PERIODS,
    )

    deltas = [0.0]
    for magnitude in DELTA_MAGNITUDES:
        deltas.extend([magnitude, -magnitude])

    rows = []
    for delta in deltas:
        trajectory_mse = _trajectory_mse_for_delta(data, delta)
        rows.append(
            {
                "delta": delta,
                "abs_delta": abs(delta),
                "mean_trajectory_mse": trajectory_mse.mean().item(),
                "median_trajectory_mse": trajectory_mse.median().item(),
                "max_trajectory_mse": trajectory_mse.max().item(),
            },
        )

    thresholds = _threshold_rows(rows)
    references = _stage1b_reference_rows()
    _write_csv(OUTPUT_DIR / "omega_sensitivity.csv", rows)
    _write_csv(OUTPUT_DIR / "omega_sensitivity_thresholds.csv", thresholds)
    _write_json(
        OUTPUT_DIR / "omega_sensitivity.json",
        {
            "setup": {
                "num_test_trajectories": NUM_TEST,
                "num_time_steps": NUM_TIME_STEPS,
                "num_periods": NUM_PERIODS,
                "omega_range": OMEGA_RANGE,
                "dtype": "float64",
                "seed": SEED + 1,
            },
            "sweep": rows,
            "thresholds": thresholds,
            "best_existing_stage1b_references": references,
        },
    )
    _save_plots(rows)
    _write_html(rows, thresholds, references)
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "thresholds": thresholds,
                "best_existing_stage1b_references": references[:5],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
