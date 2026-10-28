"""Run Stage 1 omega-basin follow-up sweeps."""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from experiments.sho.stage0_exact_solution_decoder import (
    BEST_ACHIEVED_MAX_EPOCHS,
    _base_setup,
    _evaluate_and_save_learned_model,
    _target_linear_weight_matrix,
    _train_model,
    run_stage1_linear_encoder_global_omega,
)
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.training import set_seed


RUN_DATE = "2026-06-06"
TRUE_OMEGA = 2.0
SUCCESS_TRAJECTORY_MSE = 1e-12
SUCCESS_OMEGA_TOLERANCE = 1e-6


def _inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value))))


class SHOFrozenTrueWLearnedOmegaModel(nn.Module):
    """Use exact coefficient map and learn only one global omega."""

    def __init__(self, true_omega: float, initial_omega_hat: float) -> None:
        super().__init__()
        self.register_buffer("exact_weight", _target_linear_weight_matrix(true_omega))
        self.raw_omega = nn.Parameter(_inverse_softplus(initial_omega_hat - 1e-8))

    @property
    def omega_hat(self) -> torch.Tensor:
        return F.softplus(self.raw_omega) + 1e-8

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = initial_conditions @ self.exact_weight.T
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = self.omega_hat * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories


def _omega_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _history_entry_for_epoch(history: dict[str, Any], epoch: int) -> dict[str, Any]:
    for entry in history["losses"]:
        if int(entry["epoch"]) == int(epoch):
            return entry
    raise ValueError(f"epoch {epoch} not found in training history")


def _success(best_trajectory_mse: float, best_omega_hat: float) -> bool:
    return (
        best_trajectory_mse < SUCCESS_TRAJECTORY_MSE
        and abs(best_omega_hat - TRUE_OMEGA) < SUCCESS_OMEGA_TOLERANCE
    )


def _summary_row(run_name: str, run_dir: Path, include_coefficients: bool) -> dict[str, Any]:
    setup = _read_json(run_dir / "setup.json")
    metrics = _read_json(run_dir / "metrics.json")
    history = _read_json(run_dir / "training_history.json")
    final_entry = history["losses"][-1]
    best_entry = _history_entry_for_epoch(history, int(history["best_epoch"]))
    row = {
        "run_name": run_name,
        "omega_init": setup["initial_omega_hat_requested"],
        "final_omega_hat": final_entry["omega_hat"],
        "best_omega_hat": best_entry["omega_hat"],
        "final_trajectory_mse": history["final_epoch_test_loss"],
        "best_trajectory_mse": history["best_test_loss"],
        "recovered_successfully": _success(
            float(history["best_test_loss"]),
            float(best_entry["omega_hat"]),
        ),
        "epochs_completed": setup.get("num_epochs_completed"),
        "best_epoch": history.get("best_epoch"),
        "stopped_because": setup.get("stopped_because"),
        "output_dir": str(run_dir),
    }
    if include_coefficients:
        row.update(
            {
                "final_coefficient_mse": metrics.get("coefficient_mse"),
                "mean_abs_predicted_a_test": metrics.get("mean_abs_predicted_a_test"),
                "mean_abs_predicted_b_test": metrics.get("mean_abs_predicted_b_test"),
                "mean_abs_true_a_test": metrics.get("mean_abs_true_a_test"),
                "mean_abs_true_b_test": metrics.get("mean_abs_true_b_test"),
            }
        )
    return row


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
        ("Success vs Omega Initialization", "success_vs_omega_initialization.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(image_title)}</h2><img src="{escape(filename)}" alt="{escape(image_title)}"></section>'
        for image_title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
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
  <h1>{escape(title)}</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
  {image_html}
</body>
</html>
"""
    (output_dir / html_name).write_text(html, encoding="utf-8")


def _plot_sweep(rows: list[dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    plt.figure(figsize=(8, 5))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        omega_hats = [entry["omega_hat"] for entry in history["losses"]]
        plt.plot(epochs, omega_hats, label=f"init {row['omega_init']}")
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("epoch")
    plt.ylabel("omega_hat")
    plt.title(f"{title_prefix}: Omega Convergence by Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_convergence_by_init.png")
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
    plt.title(f"{title_prefix}: Test Loss by Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_by_init.png")
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
    plt.title(f"{title_prefix}: Final Omega vs Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "final_omega_vs_initialization.png")
    plt.close()

    plt.figure(figsize=(6.5, 4.5))
    plt.step(
        [row["omega_init"] for row in rows],
        [1 if row["recovered_successfully"] else 0 for row in rows],
        where="mid",
    )
    plt.scatter(
        [row["omega_init"] for row in rows],
        [1 if row["recovered_successfully"] else 0 for row in rows],
    )
    plt.xlabel("omega initialization")
    plt.ylabel("recovered successfully")
    plt.yticks([0, 1], ["no", "yes"])
    plt.title(f"{title_prefix}: Success vs Initialization")
    plt.tight_layout()
    plt.savefig(output_dir / "success_vs_omega_initialization.png")
    plt.close()


def run_fine_omega_init_sweep() -> list[dict[str, Any]]:
    output_dir = Path("outputs/sho/stage1/fine_omega_init_sweep")
    omega_initializations = [1.60, 1.70, 1.80, 1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20, 2.30, 2.40]
    rows = []
    for omega_init in omega_initializations:
        run_name = f"linear_encoder_omega_init_{_omega_label(omega_init)}"
        run_dir = output_dir / f"{RUN_DATE}_{run_name}"
        print(f"running fine omega init sweep: {omega_init}", flush=True)
        if not (run_dir / "metrics.json").exists():
            run_stage1_linear_encoder_global_omega(
                initial_omega_hat=omega_init,
                restore_best_checkpoint=False,
                output_dir=run_dir,
            )
        rows.append(_summary_row(run_name, run_dir, include_coefficients=True))

    _plot_sweep(rows, output_dir, "Stage 1 Fine Omega Init Sweep")
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
        "recovered_successfully",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
    ]
    _write_summary(
        rows,
        output_dir,
        "fine_omega_init_sweep_summary.csv",
        "fine_omega_init_sweep_summary.html",
        "Stage 1 Fine Omega Initialization Sweep",
        columns,
    )
    return rows


def run_frozen_w_one(omega_init: float, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_train = 512
    num_test = 128
    num_time_steps = 100
    num_periods = 10.0
    batch_size = 512
    learning_rate = 1e-2
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
    model = SHOFrozenTrueWLearnedOmegaModel(TRUE_OMEGA, omega_init)
    setup = _base_setup(
        comparison_mode="stage1_frozen_W_omega_only",
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
            "run_name": "stage1_frozen_W_omega_only",
            "initial_omega_hat_requested": omega_init,
            "omega_hat_initial": float(model.omega_hat.detach().cpu()),
            "omega_parameterization": "softplus(raw_omega) + 1e-8",
            "W_trainable": False,
            "fixed_W_true": _target_linear_weight_matrix(TRUE_OMEGA).tolist(),
        }
    )
    history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        BEST_ACHIEVED_MAX_EPOCHS,
        batch_size,
        learning_rate,
        500,
        "Stage 1 Frozen W + Global Omega",
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        None,
        output_dir / "best_model.pt",
        False,
    )
    omega_hat = float(model.omega_hat.detach().cpu())
    extra_metrics = {
        "omega_hat": omega_hat,
        "true_omega": TRUE_OMEGA,
        "omega_error": omega_hat - TRUE_OMEGA,
        "relative_omega_error": (omega_hat - TRUE_OMEGA) / TRUE_OMEGA,
        "omega_hat_initial": setup["omega_hat_initial"],
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
        run_name="Stage 1 Frozen W + Global Omega",
        omega=TRUE_OMEGA,
        extra_metrics=extra_metrics,
        prediction_omega=omega_hat,
    )


def run_frozen_w_omega_only_sweep() -> list[dict[str, Any]]:
    output_dir = Path("outputs/sho/stage1/frozen_W_omega_only")
    omega_initializations = [0.5, 1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0]
    rows = []
    for omega_init in omega_initializations:
        run_name = f"frozen_W_omega_init_{_omega_label(omega_init)}"
        run_dir = output_dir / f"{RUN_DATE}_{run_name}"
        print(f"running frozen W omega-only sweep: {omega_init}", flush=True)
        if not (run_dir / "metrics.json").exists():
            run_frozen_w_one(omega_init, run_dir)
        rows.append(_summary_row(run_name, run_dir, include_coefficients=False))

    _plot_sweep(rows, output_dir, "Stage 1 Frozen W Omega-Only Sweep")
    columns = [
        "run_name",
        "omega_init",
        "final_omega_hat",
        "best_omega_hat",
        "final_trajectory_mse",
        "best_trajectory_mse",
        "recovered_successfully",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
    ]
    _write_summary(
        rows,
        output_dir,
        "frozen_W_omega_only_summary.csv",
        "frozen_W_omega_only_summary.html",
        "Stage 1 Frozen W Omega-Only Sweep",
        columns,
    )
    return rows


def main() -> None:
    fine_rows = run_fine_omega_init_sweep()
    frozen_rows = run_frozen_w_omega_only_sweep()
    print("fine omega init sweep")
    for row in fine_rows:
        print(row)
    print("frozen W omega-only sweep")
    for row in frozen_rows:
        print(row)


if __name__ == "__main__":
    main()
