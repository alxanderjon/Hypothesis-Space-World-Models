"""Run Stage 0 learning-rate controls and residual learning-rate sweeps."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any, Callable

from experiments.sho.stage0_exact_solution_decoder import (
    run_linear_encoder_experiment,
    run_mlp_experiment,
    run_residual_encoder_experiment,
)


RUN_DATE = "2026-06-06"
STAGE0_OUTPUT_ROOT = Path("outputs/sho/stage0")


def _write_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    csv_name: str,
    html_name: str,
    title: str,
    columns: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / csv_name).open("w", encoding="utf-8", newline="") as file:
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
            if column == "run_name" and row.get("output_dir"):
                href = escape(str(Path(row["output_dir"]) / "index.html"))
                cells.append(f'<td><a href="{href}">{escape(str(value))}</a></td>')
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
</body>
</html>
"""
    (output_dir / html_name).write_text(html, encoding="utf-8")


def _read_row(run_name: str, output_dir: Path) -> dict[str, Any]:
    setup = json.loads((output_dir / "setup.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "run_name": run_name,
        "architecture_model_type": f"{setup.get('encoder_type')} hidden={setup.get('hidden_dim')}",
        "model_type": setup.get("model_type"),
        "best_test_loss": metrics.get("best_test_loss"),
        "coefficient_mse": metrics.get("coefficient_mse"),
        "trajectory_mse": metrics.get("trajectory_mse"),
        "residual_rms_test": metrics.get("residual_rms_test"),
        "linear_weight_mse": metrics.get("linear_weight_mse", metrics.get("weight_mse")),
        "epochs_completed": setup.get("num_epochs_completed"),
        "best_epoch": metrics.get("best_epoch"),
        "stopped_because": setup.get("stopped_because"),
        "optimizer": setup.get("optimizer"),
        "learning_rate": setup.get("learning_rate"),
        "comparison_mode": setup.get("comparison_mode"),
        "failed_due_to_nan_or_inf": metrics.get(
            "failed_due_to_nan_or_inf",
            setup.get("failed_due_to_nan_or_inf", False),
        ),
        "output_dir": str(output_dir),
    }


def _failure_row(
    run_name: str,
    output_dir: Path,
    learning_rate: float,
    architecture_model_type: str,
    error: Exception,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup = {
        "run_name": run_name,
        "architecture_model_type": architecture_model_type,
        "learning_rate": learning_rate,
        "comparison_mode": "lr_sweep_extended",
        "failed_due_to_nan_or_inf": True,
        "stopped_because": f"exception: {type(error).__name__}",
        "error": str(error),
    }
    metrics = {
        "failed_due_to_nan_or_inf": True,
        "best_test_loss": "",
        "coefficient_mse": "",
        "trajectory_mse": "",
    }
    (output_dir / "setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {
        "run_name": run_name,
        "architecture_model_type": architecture_model_type,
        "learning_rate": learning_rate,
        "best_test_loss": "",
        "coefficient_mse": "",
        "trajectory_mse": "",
        "residual_rms_test": "",
        "linear_weight_mse": "",
        "epochs_completed": "",
        "best_epoch": "",
        "stopped_because": setup["stopped_because"],
        "failed_due_to_nan_or_inf": True,
        "output_dir": str(output_dir),
    }


def run_lr_controlled() -> list[dict[str, Any]]:
    base = STAGE0_OUTPUT_ROOT / "best_achieved_lr_controlled"
    settings = {
        "comparison_mode": "best_achieved_lr_controlled",
        "learning_rate": 1e-2,
        "batch_size": 512,
        "stopping_criterion": "patience",
        "stopping_monitor": "test_loss",
        "stopping_min_delta": 1e-13,
        "stopping_patience": 10000,
        "target_loss": 1e-16,
        "restore_best_checkpoint": True,
    }
    runs: list[tuple[str, Callable[[], Any]]] = [
        (
            "exact_linear",
            lambda: run_linear_encoder_experiment(
                output_dir=base / f"{RUN_DATE}_exact_linear",
                **settings,
            ),
        ),
        (
            "linear_tiny_residual",
            lambda: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                hidden_dims=[4],
                output_dir=base / f"{RUN_DATE}_linear_tiny_residual",
                **settings,
            )[0],
        ),
        (
            "linear_moderate_residual",
            lambda: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                hidden_dims=[8, 8],
                output_dir=base / f"{RUN_DATE}_linear_moderate_residual",
                **settings,
            )[0],
        ),
        (
            "linear_large_residual",
            lambda: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                hidden_dims=[32, 32],
                output_dir=base / f"{RUN_DATE}_linear_large_residual",
                **settings,
            )[0],
        ),
        (
            "pure_mlp",
            lambda: run_mlp_experiment(
                "silu",
                hidden_dims=[32, 32],
                output_dir=base / f"{RUN_DATE}_pure_mlp",
                **settings,
            )[0],
        ),
    ]

    for run_name, run in runs:
        print(f"running lr-controlled {run_name}", flush=True)
        run()

    rows = [
        _read_row(run_name, base / f"{RUN_DATE}_{run_name}")
        for run_name, _ in runs
    ]
    _write_summary(
        rows=rows,
        output_dir=base,
        csv_name="comparison_summary.csv",
        html_name="comparison_summary.html",
        title="SHO Stage 0 LR-Controlled Comparison",
        columns=[
            "run_name",
            "model_type",
            "best_test_loss",
            "coefficient_mse",
            "trajectory_mse",
            "epochs_completed",
            "best_epoch",
            "stopped_because",
            "optimizer",
            "learning_rate",
            "comparison_mode",
        ],
    )
    return rows


def run_lr_sweep() -> list[dict[str, Any]]:
    base = STAGE0_OUTPUT_ROOT / "lr_sweep"
    variants = [
        ("linear_tiny_residual", [4]),
        ("linear_moderate_residual", [8, 8]),
        ("linear_large_residual", [32, 32]),
    ]
    learning_rates = [1e-2, 3e-3, 1e-3, 3e-4]
    rows = []

    for variant_name, hidden_dims in variants:
        for learning_rate in learning_rates:
            lr_label = f"lr_{learning_rate:.0e}".replace("-", "m")
            run_name = f"{variant_name}_{lr_label}"
            output_dir = base / f"{RUN_DATE}_{run_name}"
            print(f"running lr sweep {run_name}", flush=True)
            run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                comparison_mode="lr_sweep",
                learning_rate=learning_rate,
                hidden_dims=hidden_dims,
                output_dir=output_dir,
            )
            row = _read_row(run_name, output_dir)
            row["learning_rate"] = learning_rate
            rows.append(row)

    _write_summary(
        rows=rows,
        output_dir=base,
        csv_name="lr_sweep_summary.csv",
        html_name="lr_sweep_summary.html",
        title="SHO Stage 0 Residual Learning-Rate Sweep",
        columns=[
            "run_name",
            "learning_rate",
            "best_test_loss",
            "coefficient_mse",
            "trajectory_mse",
            "residual_rms_test",
            "linear_weight_mse",
            "epochs_completed",
            "stopped_because",
        ],
    )
    return rows


def run_lr_sweep_extended() -> list[dict[str, Any]]:
    base = STAGE0_OUTPUT_ROOT / "lr_sweep_extended"
    learning_rates = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    variants: list[tuple[str, str, Callable[[float, Path], Any]]] = [
        (
            "pure_mlp_tanh",
            "pure MLP Tanh 2->32->32->2",
            lambda learning_rate, output_dir: run_mlp_experiment(
                "tanh",
                comparison_mode="lr_sweep_extended",
                learning_rate=learning_rate,
                hidden_dims=[32, 32],
                output_dir=output_dir,
            )[0],
        ),
        (
            "pure_mlp_silu",
            "pure MLP SiLU 2->32->32->2",
            lambda learning_rate, output_dir: run_mlp_experiment(
                "silu",
                comparison_mode="lr_sweep_extended",
                learning_rate=learning_rate,
                hidden_dims=[32, 32],
                output_dir=output_dir,
            )[0],
        ),
        (
            "linear_tiny_residual",
            "trainable linear + SiLU residual 2->4->2",
            lambda learning_rate, output_dir: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                comparison_mode="lr_sweep_extended",
                learning_rate=learning_rate,
                hidden_dims=[4],
                output_dir=output_dir,
            )[0],
        ),
        (
            "linear_moderate_residual",
            "trainable linear + SiLU residual 2->8->8->2",
            lambda learning_rate, output_dir: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                comparison_mode="lr_sweep_extended",
                learning_rate=learning_rate,
                hidden_dims=[8, 8],
                output_dir=output_dir,
            )[0],
        ),
        (
            "linear_large_residual",
            "trainable linear + SiLU residual 2->32->32->2",
            lambda learning_rate, output_dir: run_residual_encoder_experiment(
                "trainable_linear_residual",
                "silu",
                comparison_mode="lr_sweep_extended",
                learning_rate=learning_rate,
                hidden_dims=[32, 32],
                output_dir=output_dir,
            )[0],
        ),
    ]
    rows = []
    for variant_name, architecture_model_type, run in variants:
        for learning_rate in learning_rates:
            lr_label = f"lr_{learning_rate:.0e}".replace("-", "m")
            run_name = f"{variant_name}_{lr_label}"
            output_dir = base / f"{RUN_DATE}_{run_name}"
            print(f"running extended lr sweep {run_name}", flush=True)
            try:
                run(learning_rate, output_dir)
                row = _read_row(run_name, output_dir)
                row["architecture_model_type"] = architecture_model_type
                row["learning_rate"] = learning_rate
                rows.append(row)
            except Exception as error:
                rows.append(
                    _failure_row(
                        run_name,
                        output_dir,
                        learning_rate,
                        architecture_model_type,
                        error,
                    )
                )

    _write_summary(
        rows=rows,
        output_dir=base,
        csv_name="lr_sweep_extended_summary.csv",
        html_name="lr_sweep_extended_summary.html",
        title="SHO Stage 0 Extended Learning-Rate Sweep",
        columns=[
            "run_name",
            "architecture_model_type",
            "learning_rate",
            "best_test_loss",
            "coefficient_mse",
            "trajectory_mse",
            "residual_rms_test",
            "linear_weight_mse",
            "epochs_completed",
            "best_epoch",
            "stopped_because",
            "failed_due_to_nan_or_inf",
        ],
    )
    return rows


def run_epsilon_sweep() -> list[dict[str, Any]]:
    base = STAGE0_OUTPUT_ROOT / "epsilon_sweep"
    epsilons = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    rows = []
    for epsilon in epsilons:
        epsilon_label = "eps_0" if epsilon == 0.0 else f"eps_{epsilon:.0e}".replace("-", "m")
        run_name = f"linear_tiny_residual_{epsilon_label}"
        output_dir = base / f"{RUN_DATE}_{run_name}"
        print(f"running epsilon sweep {run_name}", flush=True)
        run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            comparison_mode="epsilon_sweep",
            learning_rate=3e-2,
            hidden_dims=[4],
            residual_scale=epsilon,
            output_dir=output_dir,
        )
        row = _read_row(run_name, output_dir)
        row["architecture_model_type"] = (
            "trainable linear + epsilon-scaled SiLU residual 2->4->2"
        )
        row["epsilon"] = epsilon
        rows.append(row)

    _write_summary(
        rows=rows,
        output_dir=base,
        csv_name="epsilon_sweep_summary.csv",
        html_name="epsilon_sweep_summary.html",
        title="SHO Stage 0 Epsilon-Scaled Residual Sweep",
        columns=[
            "run_name",
            "architecture_model_type",
            "epsilon",
            "best_test_loss",
            "coefficient_mse",
            "trajectory_mse",
            "residual_rms_test",
            "linear_weight_mse",
            "epochs_completed",
            "best_epoch",
            "stopped_because",
        ],
    )
    return rows


if __name__ == "__main__":
    run_lr_controlled()
    run_lr_sweep()
    run_lr_sweep_extended()
    print("completed_stage0_lr_controls")
