"""Create the Stage 1 direct residual omega sweep output bundle."""

from __future__ import annotations

import csv
import json
import shutil
from html import escape
from pathlib import Path
from typing import Any

from experiments.sho.run_stage1_residual_omega_init_sweep import (
    OMEGA_INITIALIZATIONS,
    OUTPUT_DIR as SOURCE_OUTPUT_DIR,
    _omega_label,
    _plot_sweep,
    _read_json,
    _residual_variants,
    _summary_row,
    run_one,
)
from experiments.sho.run_stage1_followup_sweeps import RUN_DATE


OUTPUT_DIR = Path("outputs/sho/stage1/residual_direct_omega_sweep")


def _copy_or_run_source_run(variant: dict[str, Any], omega_init: float, target_dir: Path) -> None:
    source_run_name = f"{variant['model_name']}_omega_init_{_omega_label(omega_init)}"
    source_dir = SOURCE_OUTPUT_DIR / f"{RUN_DATE}_{source_run_name}"
    if (target_dir / "metrics.json").exists():
        return
    if (source_dir / "metrics.json").exists():
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        return
    run_one(variant=variant, omega_init=omega_init, output_dir=target_dir)


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "run_name",
        "model_name",
        "omega_init",
        "final_omega_hat",
        "best_omega_hat",
        "final_trajectory_mse",
        "best_trajectory_mse",
        "final_coefficient_mse",
        "mean_abs_predicted_a_test",
        "mean_abs_predicted_b_test",
        "mean_abs_true_a_test",
        "mean_abs_true_b_test",
        "residual_rms_test",
        "linear_weight_mse",
        "learning_rate",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
        "recovered_successfully",
    ]
    with (output_dir / "direct_residual_omega_sweep_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    html_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if column == "run_name":
                href = escape(str(Path(row["output_dir"]) / "index.html"))
                cells.append(f'<td><a href="{href}">{escape(str(value))}</a></td>')
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    images = [
        ("Omega Convergence by Model and Init", "omega_convergence_by_model_and_init.png"),
        ("Loss by Model and Init", "loss_by_model_and_init.png"),
        ("Success vs Omega Initialization by Model", "success_vs_omega_initialization_by_model.png"),
        ("Residual RMS by Model and Init", "residual_rms_by_model_and_init.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(filename)}" alt="{escape(title)}"></section>'
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Direct Residual Omega Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 1080px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 1 Direct Residual Omega Sweep</h1>
  <p>Direct 10-period training with trainable linear path plus SiLU residual branch.</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
  {image_html}
</body>
</html>
"""
    (output_dir / "direct_residual_omega_sweep_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def run_direct_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in _residual_variants():
        for omega_init in OMEGA_INITIALIZATIONS:
            run_name = f"{variant['model_name']}_omega_init_{_omega_label(omega_init)}"
            run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
            print(
                f"preparing direct residual sweep: {variant['model_name']} omega_init={omega_init:g}",
                flush=True,
            )
            _copy_or_run_source_run(variant, omega_init, run_dir)
            setup = _read_json(run_dir / "setup.json")
            setup["comparison_mode"] = "stage1_residual_direct_omega_sweep"
            setup["direct_sweep_materialized_from"] = str(SOURCE_OUTPUT_DIR)
            (run_dir / "setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
            rows.append(
                _summary_row(
                    run_name=run_name,
                    model_name=str(variant["model_name"]),
                    omega_init=omega_init,
                    learning_rate=float(variant["learning_rate"]),
                    run_dir=run_dir,
                )
            )
    _plot_sweep(rows, OUTPUT_DIR)
    _write_summary(rows, OUTPUT_DIR)
    return rows


def main() -> None:
    rows = run_direct_sweep()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
