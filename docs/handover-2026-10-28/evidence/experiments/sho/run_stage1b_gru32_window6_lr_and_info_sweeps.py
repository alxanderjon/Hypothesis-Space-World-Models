"""Stage 1B GRU32 window-6 optimization and information diagnostics.

Both experiments keep reconstruction-only training, the exact SHO decoder, the
late-horizon curriculum, and the existing stopping settings. Experiment B is run
only when Experiment A does not move substantially below the previous
approximately 2e-4 trajectory-MSE regime.
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
    LEARNING_RATE as DEFAULT_LEARNING_RATE,
    MAX_EPOCHS_PER_STAGE,
    STOPPING_MIN_DELTA,
    STOPPING_PATIENCE,
    SUCCESS_TRAJECTORY_MSE,
    _run_curriculum,
)


LR_SWEEP_OUTPUT_DIR = Path("outputs/sho/stage1B/gru32_window6_lr_sweep")
INFORMATION_SWEEP_OUTPUT_DIR = Path("outputs/sho/stage1B/gru32_information_sweep")
HIDDEN_SIZE = 32
WINDOW_SIZE = 6
LEARNING_RATES = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
INFORMATION_WINDOWS = [6, 10, 20]
SUBSTANTIAL_IMPROVEMENT_TRAJECTORY_MSE = 1e-4


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


def _save_summary_plots(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    title_prefix: str,
    filename_suffix: str,
    log_x: bool = False,
) -> None:
    x_values = [float(row[x_key]) for row in rows]
    specs = [
        (f"trajectory_mse_vs_{filename_suffix}.png", "trajectory_mse", "trajectory MSE", True),
        (f"omega_mse_vs_{filename_suffix}.png", "omega_mse", "omega MSE", True),
        (f"omega_corr_vs_{filename_suffix}.png", "omega_correlation", "omega correlation", False),
    ]
    for filename, metric_key, ylabel, log_y in specs:
        plt.figure(figsize=(7, 4.5))
        plt.plot(x_values, [float(row[metric_key]) for row in rows], marker="o")
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        if log_x:
            plt.xscale("log")
            plt.gca().invert_xaxis()
        if log_y:
            plt.yscale("log")
        plt.title(f"{title_prefix}: {ylabel}")
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def _write_summary_html(
    *,
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    plot_files: list[str],
    links: list[tuple[str, str]],
    note: str,
) -> None:
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in plot_files
    )
    link_html = "".join(
        f'<li><a href="{escape(href)}">{escape(label)}</a></li>' for label, href in links
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>{escape(note)}</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
  <h2>Run Directories</h2>
  <ul>{link_html}</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _with_sweep_metadata(
    row: dict[str, Any],
    *,
    run_name: str,
    learning_rate: float,
    experiment_group: str,
) -> dict[str, Any]:
    return {
        **row,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "experiment_group": experiment_group,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
    }


def run_lr_sweep() -> tuple[list[dict[str, Any]], float, bool]:
    LR_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for learning_rate in LEARNING_RATES:
        label = f"gru_hidden_32_window_6_lr_{_lr_label(learning_rate)}"
        print(f"running Stage 1B GRU32 window-6 LR sweep: {label}", flush=True)
        row = _run_curriculum(
            output_dir=LR_SWEEP_OUTPUT_DIR / label,
            hidden_size=HIDDEN_SIZE,
            window_size=WINDOW_SIZE,
            label=label,
            experiment_name="gru32_window6_lr_sweep",
            learning_rate=learning_rate,
        )
        rows.append(
            _with_sweep_metadata(
                row,
                run_name=label,
                learning_rate=learning_rate,
                experiment_group="gru32_window6_lr_sweep",
            ),
        )

    _write_csv(LR_SWEEP_OUTPUT_DIR / "gru32_window6_lr_sweep_summary.csv", rows)
    _write_json(LR_SWEEP_OUTPUT_DIR / "gru32_window6_lr_sweep_summary.json", {"runs": rows})
    _save_summary_plots(
        output_dir=LR_SWEEP_OUTPUT_DIR,
        rows=rows,
        x_key="learning_rate",
        x_label="learning rate",
        title_prefix="Stage 1B GRU32 Window-6 LR Sweep",
        filename_suffix="lr",
        log_x=True,
    )
    _write_summary_html(
        path=LR_SWEEP_OUTPUT_DIR / "gru32_window6_lr_sweep_summary.html",
        title="Stage 1B GRU32 Window-6 LR Sweep",
        rows=rows,
        columns=[
            "run_name",
            "learning_rate",
            "trajectory_mse",
            "coefficient_mse",
            "omega_mse",
            "omega_correlation",
            "mean_abs_omega_error",
            "max_omega_error",
            "final_stage_best_epoch",
            "final_stage_stopped_because",
        ],
        plot_files=[
            "trajectory_mse_vs_lr.png",
            "omega_mse_vs_lr.png",
            "omega_corr_vs_lr.png",
        ],
        links=[(str(row["learning_rate"]), f"{row['run_name']}/index.html") for row in rows],
        note="GRU hidden size 32, observation window 6, reconstruction loss only.",
    )
    best_row = min(rows, key=lambda row: float(row["trajectory_mse"]))
    best_lr = float(best_row["learning_rate"])
    needs_information_sweep = float(best_row["trajectory_mse"]) >= SUBSTANTIAL_IMPROVEMENT_TRAJECTORY_MSE
    return rows, best_lr, needs_information_sweep


def run_information_sweep(learning_rate: float) -> list[dict[str, Any]]:
    INFORMATION_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for window_size in INFORMATION_WINDOWS:
        label = f"gru_hidden_32_window_{window_size}_lr_{_lr_label(learning_rate)}"
        print(f"running Stage 1B GRU32 information sweep: {label}", flush=True)
        row = _run_curriculum(
            output_dir=INFORMATION_SWEEP_OUTPUT_DIR / label,
            hidden_size=HIDDEN_SIZE,
            window_size=window_size,
            label=label,
            experiment_name="gru32_information_sweep",
            learning_rate=learning_rate,
        )
        rows.append(
            _with_sweep_metadata(
                row,
                run_name=label,
                learning_rate=learning_rate,
                experiment_group="gru32_information_sweep",
            ),
        )

    _write_csv(INFORMATION_SWEEP_OUTPUT_DIR / "gru32_information_sweep_summary.csv", rows)
    _write_json(INFORMATION_SWEEP_OUTPUT_DIR / "gru32_information_sweep_summary.json", {"runs": rows})
    _save_summary_plots(
        output_dir=INFORMATION_SWEEP_OUTPUT_DIR,
        rows=rows,
        x_key="window_size",
        x_label="observation window size in states",
        title_prefix="Stage 1B GRU32 Information Sweep",
        filename_suffix="window",
    )
    _write_summary_html(
        path=INFORMATION_SWEEP_OUTPUT_DIR / "gru32_information_sweep_summary.html",
        title="Stage 1B GRU32 Information Sweep",
        rows=rows,
        columns=[
            "run_name",
            "window_size",
            "learning_rate",
            "trajectory_mse",
            "coefficient_mse",
            "omega_mse",
            "omega_correlation",
            "mean_abs_omega_error",
            "max_omega_error",
            "final_stage_best_epoch",
            "final_stage_stopped_because",
        ],
        plot_files=[
            "trajectory_mse_vs_window.png",
            "omega_mse_vs_window.png",
            "omega_corr_vs_window.png",
        ],
        links=[(f"window {row['window_size']}", f"{row['run_name']}/index.html") for row in rows],
        note=f"GRU hidden size 32, best LR from Experiment A = {learning_rate:g}.",
    )
    return rows


def main() -> None:
    lr_rows, best_lr, needs_information_sweep = run_lr_sweep()
    info_rows: list[dict[str, Any]] = []
    if needs_information_sweep:
        print(
            (
                "Experiment A did not substantially improve below "
                f"{SUBSTANTIAL_IMPROVEMENT_TRAJECTORY_MSE:.1e}; running Experiment B."
            ),
            flush=True,
        )
        info_rows = run_information_sweep(best_lr)
    else:
        print(
            (
                "Experiment A substantially improved the result; "
                "Experiment B was not run."
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "default_learning_rate_reference": DEFAULT_LEARNING_RATE,
                "lr_sweep": lr_rows,
                "best_lr_by_trajectory_mse": best_lr,
                "information_sweep_ran": needs_information_sweep,
                "information_sweep": info_rows,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
