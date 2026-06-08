"""Run a Stage 1 learning-rate sweep for frozen-W omega-only training."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from experiments.sho.run_stage1_followup_sweeps import (
    RUN_DATE,
    SHOFrozenTrueWLearnedOmegaModel,
    TRUE_OMEGA,
    _history_entry_for_epoch,
)
from experiments.sho.stage0_exact_solution_decoder import (
    BEST_ACHIEVED_MAX_EPOCHS,
    _base_setup,
    _evaluate_and_save_learned_model,
    _target_linear_weight_matrix,
    _train_model,
)
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.training import set_seed


OUTPUT_DIR = Path("outputs/sho/stage1/frozen_W_lr_sweep")
OMEGA_INIT = 1.0
LEARNING_RATES = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]


def _lr_label(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("-", "m")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one_lr(learning_rate: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_train = 512
    num_test = 128
    num_time_steps = 100
    num_periods = 10.0
    batch_size = 512
    stopping_criterion = "patience"
    stopping_monitor = "test_loss"
    stopping_min_delta = 1e-13
    stopping_patience = 10000

    set_seed(seed)
    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=TRUE_OMEGA,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=TRUE_OMEGA,
        num_periods=num_periods,
        seed=seed + 1,
    )
    model = SHOFrozenTrueWLearnedOmegaModel(TRUE_OMEGA, OMEGA_INIT)
    run_name = f"stage1_frozen_W_omega_init_1_lr_{_lr_label(learning_rate)}"
    setup = _base_setup(
        comparison_mode="stage1_frozen_W_lr_sweep",
        seed=seed,
        omega=TRUE_OMEGA,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=BEST_ACHIEVED_MAX_EPOCHS,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_frozen_true_W_learned_global_omega",
        encoder_type="frozen_exact_linear_encoder",
        activation=None,
        hidden_dim=None,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=None,
        restore_best_checkpoint=False,
    )
    setup.update(
        {
            "run_name": run_name,
            "initial_omega_hat_requested": OMEGA_INIT,
            "omega_hat_initial": float(model.omega_hat.detach().cpu()),
            "omega_parameterization": "softplus(raw_omega) + 1e-8",
            "W_trainable": False,
            "fixed_W_true": _target_linear_weight_matrix(TRUE_OMEGA).tolist(),
        }
    )
    history = _train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        num_train=num_train,
        max_epochs=BEST_ACHIEVED_MAX_EPOCHS,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_every=500,
        run_name=f"Stage 1 Frozen W LR Sweep lr={learning_rate:g}",
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=None,
        best_model_path=output_dir / "best_model.pt",
        restore_best_checkpoint=False,
    )
    final_entry = history["losses"][-1]
    best_entry = _history_entry_for_epoch(history, int(history["best_epoch"]))
    extra_metrics = {
        "omega_hat": float(final_entry["omega_hat"]),
        "best_omega_hat": float(best_entry["omega_hat"]),
        "true_omega": TRUE_OMEGA,
        "omega_error": float(final_entry["omega_hat"]) - TRUE_OMEGA,
        "best_omega_error": float(best_entry["omega_hat"]) - TRUE_OMEGA,
        "relative_omega_error": (float(final_entry["omega_hat"]) - TRUE_OMEGA) / TRUE_OMEGA,
        "omega_hat_initial": setup["omega_hat_initial"],
        "final_trajectory_mse": history["final_epoch_test_loss"],
        "best_trajectory_mse": history["best_test_loss"],
        "epochs_to_convergence": history["num_epochs_completed"],
        "plots_generated_from_restored_best_checkpoint": False,
        "W_trainable": False,
    }
    _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=history,
        output_dir=output_dir,
        run_name=f"Stage 1 Frozen W LR Sweep lr={learning_rate:g}",
        omega=TRUE_OMEGA,
        extra_metrics=extra_metrics,
        prediction_omega=float(final_entry["omega_hat"]),
    )


def _summary_row(run_name: str, learning_rate: float, run_dir: Path) -> dict[str, Any]:
    setup = _read_json(run_dir / "setup.json")
    history = _read_json(run_dir / "training_history.json")
    final_entry = history["losses"][-1]
    best_entry = _history_entry_for_epoch(history, int(history["best_epoch"]))
    return {
        "run_name": run_name,
        "learning_rate": learning_rate,
        "omega_init": setup["initial_omega_hat_requested"],
        "final_omega_hat": final_entry["omega_hat"],
        "best_omega_hat": best_entry["omega_hat"],
        "final_trajectory_mse": history["final_epoch_test_loss"],
        "best_trajectory_mse": history["best_test_loss"],
        "epochs_to_convergence": history["num_epochs_completed"],
        "best_epoch": history["best_epoch"],
        "stopped_because": history["stopped_because"],
        "output_dir": str(run_dir),
    }


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "run_name",
        "learning_rate",
        "omega_init",
        "final_omega_hat",
        "best_omega_hat",
        "final_trajectory_mse",
        "best_trajectory_mse",
        "epochs_to_convergence",
        "best_epoch",
        "stopped_because",
    ]
    with (output_dir / "lr_sweep_summary.csv").open("w", encoding="utf-8", newline="") as file:
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
        ("Omega Convergence by Learning Rate", "omega_convergence_by_lr.png"),
        ("Loss by Learning Rate", "loss_by_lr.png"),
        ("Final Omega vs Learning Rate", "final_omega_vs_lr.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(filename)}" alt="{escape(title)}"></section>'
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Frozen-W Learning-Rate Sweep</title>
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
  <h1>Stage 1 Frozen-W Learning-Rate Sweep</h1>
  <p>Frozen exact coefficient map, omega_init = 1.0, train omega only.</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
  {image_html}
</body>
</html>
"""
    (output_dir / "lr_sweep_summary.html").write_text(html, encoding="utf-8")


def _plot_lr_sweep(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        omega_hats = [entry["omega_hat"] for entry in history["losses"]]
        plt.plot(epochs, omega_hats, label=f"lr {row['learning_rate']:g}")
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("epoch")
    plt.ylabel("omega_hat")
    plt.title("Stage 1 Frozen W: Omega Convergence by Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_convergence_by_lr.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        test_losses = [entry["test_loss"] for entry in history["losses"]]
        plt.plot(epochs, test_losses, label=f"lr {row['learning_rate']:g}")
    plt.xlabel("epoch")
    plt.ylabel("test loss")
    plt.yscale("log")
    plt.title("Stage 1 Frozen W: Test Loss by Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_by_lr.png")
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    learning_rates = [row["learning_rate"] for row in rows]
    final_omegas = [row["final_omega_hat"] for row in rows]
    plt.plot(learning_rates, final_omegas, marker="o")
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("final omega_hat")
    plt.title("Stage 1 Frozen W: Final Omega vs Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_omega_vs_lr.png")
    plt.close()


def run_lr_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for learning_rate in LEARNING_RATES:
        run_name = f"frozen_W_omega_init_1_lr_{_lr_label(learning_rate)}"
        run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
        print(f"running frozen-W lr sweep: lr={learning_rate:g}", flush=True)
        if not (run_dir / "metrics.json").exists():
            run_one_lr(learning_rate, run_dir)
        rows.append(_summary_row(run_name, learning_rate, run_dir))

    _plot_lr_sweep(rows, OUTPUT_DIR)
    _write_summary(rows, OUTPUT_DIR)
    return rows


def main() -> None:
    rows = run_lr_sweep()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
