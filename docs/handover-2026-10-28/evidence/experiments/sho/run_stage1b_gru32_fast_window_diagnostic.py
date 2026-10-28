"""Fast Stage 1B GRU32 window diagnostic.

This diagnostic keeps reconstruction-only training and the exact SHO decoder,
but uses shorter per-stage budgets plus curriculum-level early stopping to
quickly test whether larger observation windows help variable-omega recovery.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from experiments.sho.run_stage1b_capacity_and_window_sweeps import (
    CURRICULUM_HORIZONS,
    SUCCESS_TRAJECTORY_MSE,
    _run_curriculum,
)


OUTPUT_DIR = Path("outputs/sho/stage1B/gru32_fast_window_diagnostic")
HIDDEN_SIZE = 32
WINDOW_SIZES = [6, 10, 20]
LEARNING_RATES = [1e-2, 3e-3]
MAX_EPOCHS_PER_STAGE = 50_000
STOPPING_PATIENCE = 10_000
STOPPING_MIN_DELTA = 0.0


def _lr_label(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("+", "").replace("-", "m")


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


def _save_grouped_plot(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    filename: str,
    log_y: bool,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for learning_rate in LEARNING_RATES:
        lr_rows = [row for row in rows if float(row["learning_rate"]) == learning_rate]
        lr_rows.sort(key=lambda row: int(row["window_size"]))
        plt.plot(
            [int(row["window_size"]) for row in lr_rows],
            [float(row[metric_key]) for row in lr_rows],
            marker="o",
            label=f"lr={learning_rate:g}",
        )
    plt.xlabel("observation window size in states")
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale("log")
    plt.title(f"Stage 1B GRU32 Fast Window Diagnostic: {ylabel}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def _write_summary_html(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "window_size",
        "learning_rate",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_abs_omega_error",
        "max_omega_error",
        "horizons_completed",
        "final_horizon_periods",
        "completed_final_horizon",
        "curriculum_stopped_because",
        "final_stage_best_epoch",
        "final_stage_stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    links = "".join(
        f'<li><a href="{escape(str(row["run_name"]))}/index.html">{escape(str(row["run_name"]))}</a></li>'
        for row in rows
    )
    plots = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in [
            "trajectory_mse_vs_window.png",
            "omega_mse_vs_window.png",
            "omega_corr_vs_window.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B GRU32 Fast Window Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B GRU32 Fast Window Diagnostic</h1>
  <p>Reconstruction loss only. No omega labels, coefficient labels, auxiliary losses, or regularizers.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plots}
  <h2>Run Directories</h2>
  <ul>{links}</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_jobs = len(WINDOW_SIZES) * len(LEARNING_RATES)
    max_stages = len(CURRICULUM_HORIZONS)
    print(
        (
            "Stage 1B GRU32 fast window diagnostic: "
            f"{total_jobs} jobs, at most {max_stages} curriculum stages per job."
        ),
        flush=True,
    )
    print(
        (
            f"windows={WINDOW_SIZES}, learning_rates={LEARNING_RATES}, "
            f"max_epochs_per_stage={MAX_EPOCHS_PER_STAGE}, "
            f"stopping_patience={STOPPING_PATIENCE}, "
            f"stopping_min_delta={STOPPING_MIN_DELTA}"
        ),
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    for window_size in WINDOW_SIZES:
        for learning_rate in LEARNING_RATES:
            label = (
                f"gru_hidden_{HIDDEN_SIZE}_window_{window_size}_"
                f"lr_{_lr_label(learning_rate)}"
            )
            print(f"running Stage 1B fast diagnostic: {label}", flush=True)
            metrics = _run_curriculum(
                output_dir=OUTPUT_DIR / label,
                hidden_size=HIDDEN_SIZE,
                window_size=window_size,
                label=label,
                experiment_name="gru32_fast_window_diagnostic",
                learning_rate=learning_rate,
                max_epochs_per_stage=MAX_EPOCHS_PER_STAGE,
                stopping_min_delta=STOPPING_MIN_DELTA,
                stopping_patience=STOPPING_PATIENCE,
                curriculum_early_stopping=True,
            )
            rows.append(
                {
                    **metrics,
                    "run_name": label,
                    "hidden_size": HIDDEN_SIZE,
                    "window_size": window_size,
                    "learning_rate": learning_rate,
                    "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
                    "stopping_patience": STOPPING_PATIENCE,
                    "stopping_min_delta": STOPPING_MIN_DELTA,
                    "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
                },
            )

    _write_csv(OUTPUT_DIR / "fast_window_diagnostic_summary.csv", rows)
    _write_json(OUTPUT_DIR / "fast_window_diagnostic_summary.json", {"runs": rows})
    _save_grouped_plot(
        output_dir=OUTPUT_DIR,
        rows=rows,
        metric_key="trajectory_mse",
        ylabel="trajectory MSE",
        filename="trajectory_mse_vs_window.png",
        log_y=True,
    )
    _save_grouped_plot(
        output_dir=OUTPUT_DIR,
        rows=rows,
        metric_key="omega_mse",
        ylabel="omega MSE",
        filename="omega_mse_vs_window.png",
        log_y=True,
    )
    _save_grouped_plot(
        output_dir=OUTPUT_DIR,
        rows=rows,
        metric_key="omega_correlation",
        ylabel="omega correlation",
        filename="omega_corr_vs_window.png",
        log_y=False,
    )
    _write_summary_html(OUTPUT_DIR / "fast_window_diagnostic_summary.html", rows)
    print(json.dumps({"runs": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
