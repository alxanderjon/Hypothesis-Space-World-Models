"""Run Stage 1 learned-global-omega initialization sweep."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from experiments.sho.stage0_exact_solution_decoder import (
    run_stage1_linear_encoder_global_omega,
)


RUN_DATE = "2026-06-06"
OUTPUT_DIR = Path("outputs/sho/stage1/omega_init_sweep")
OMEGA_INITIALIZATIONS = [0.5, 1.0, 1.5, 1.9, 2.1, 2.5, 3.0]
TRUE_OMEGA = 2.0


def _omega_label(value: float) -> str:
    return str(value).replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_entry_for_epoch(history: dict[str, Any], epoch: int) -> dict[str, Any]:
    for entry in history["losses"]:
        if int(entry["epoch"]) == int(epoch):
            return entry
    raise ValueError(f"epoch {epoch} not found in training history")


def _run_one(initial_omega_hat: float) -> dict[str, Any]:
    run_name = f"linear_encoder_omega_init_{_omega_label(initial_omega_hat)}"
    run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
    run_stage1_linear_encoder_global_omega(
        initial_omega_hat=initial_omega_hat,
        restore_best_checkpoint=False,
        output_dir=run_dir,
    )

    setup = _read_json(run_dir / "setup.json")
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_json(run_dir / "training_history.json")
    final_entry = history["losses"][-1]
    best_entry = _history_entry_for_epoch(history, int(history["best_epoch"]))

    return {
        "run_name": run_name,
        "omega_init": initial_omega_hat,
        "final_omega_hat": final_entry.get("omega_hat"),
        "best_omega_hat": best_entry.get("omega_hat"),
        "final_trajectory_mse": history["final_epoch_test_loss"],
        "best_trajectory_mse": history["best_test_loss"],
        "final_coefficient_mse": metrics.get("coefficient_mse"),
        "mean_abs_predicted_a_test": metrics.get("mean_abs_predicted_a_test"),
        "mean_abs_predicted_b_test": metrics.get("mean_abs_predicted_b_test"),
        "mean_abs_true_a_test": metrics.get("mean_abs_true_a_test"),
        "mean_abs_true_b_test": metrics.get("mean_abs_true_b_test"),
        "epochs_completed": setup.get("num_epochs_completed"),
        "best_epoch": history.get("best_epoch"),
        "stopped_because": setup.get("stopped_because"),
        "output_dir": str(run_dir),
    }


def _write_summary(rows: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_name",
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
        "epochs_completed",
        "best_epoch",
        "stopped_because",
        "output_dir",
    ]
    with (OUTPUT_DIR / "omega_init_sweep_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

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
        ("Omega Convergence by Initialization", "omega_convergence_by_init.png"),
        ("Loss by Initialization", "loss_by_init.png"),
        ("Final Omega vs Initialization", "final_omega_vs_initialization.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(filename)}" alt="{escape(title)}"></section>'
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Omega Initialization Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 1 Omega Initialization Sweep</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
  {image_html}
</body>
</html>
"""
    (OUTPUT_DIR / "omega_init_sweep_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def _plot_histories(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        omega_hats = [entry["omega_hat"] for entry in history["losses"]]
        plt.plot(epochs, omega_hats, label=f"init {row['omega_init']}")
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("epoch")
    plt.ylabel("omega_hat")
    plt.title("Stage 1: Omega Convergence by Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "omega_convergence_by_init.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        test_losses = [entry["test_loss"] for entry in history["losses"]]
        plt.plot(epochs, test_losses, label=f"init {row['omega_init']}")
    plt.xlabel("epoch")
    plt.ylabel("test loss")
    plt.yscale("log")
    plt.title("Stage 1: Test Loss by Omega Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_by_init.png")
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(
        [row["omega_init"] for row in rows],
        [row["final_omega_hat"] for row in rows],
        marker="o",
    )
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("omega initialization")
    plt.ylabel("final omega_hat")
    plt.title("Stage 1: Final Omega vs Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_omega_vs_initialization.png")
    plt.close()


def run_omega_init_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for initial_omega_hat in OMEGA_INITIALIZATIONS:
        print(f"running omega initialization sweep: {initial_omega_hat}", flush=True)
        rows.append(_run_one(initial_omega_hat))

    _plot_histories(rows)
    _write_summary(rows)
    return rows


if __name__ == "__main__":
    rows = run_omega_init_sweep()
    for row in rows:
        print(row)
