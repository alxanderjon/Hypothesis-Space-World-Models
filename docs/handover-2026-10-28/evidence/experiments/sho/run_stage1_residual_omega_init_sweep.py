"""Run Stage 1 learned-global-omega sweeps with linear + residual encoders."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from experiments.sho.run_stage1_followup_sweeps import (
    RUN_DATE,
    TRUE_OMEGA,
    _history_entry_for_epoch,
    _inverse_softplus,
)
from experiments.sho.stage0_exact_solution_decoder import (
    BEST_ACHIEVED_MAX_EPOCHS,
    _add_linear_weight_metrics,
    _add_residual_metrics,
    _base_setup,
    _evaluate_and_save_learned_model,
    _target_linear_weight_matrix,
    _train_model,
)
from hypothesis_space_world_models.evaluation.metrics import mean_squared_error
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.training import set_seed


OUTPUT_DIR = Path("outputs/sho/stage1/residual_omega_init_sweep")
OMEGA_INITIALIZATIONS = [0.5, 1.0, 1.5, 1.8, 1.9, 2.0, 2.1, 2.2, 2.5, 3.0]
SUCCESS_TRAJECTORY_MSE = 1e-12
SUCCESS_OMEGA_TOLERANCE = 1e-6
NEAR_SUCCESS_TRAJECTORY_MSE = 1e-8
NEAR_SUCCESS_OMEGA_TOLERANCE = 1e-4


def _make_activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unsupported activation: {name}")


def _make_residual_encoder(hidden_dims: list[int], activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = 2
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(_make_activation(activation))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 2))
    return nn.Sequential(*layers)


class SHOTrainableLinearResidualLearnedOmegaModel(nn.Module):
    """Learn W x + f_theta(x) and one global SHO frequency."""

    def __init__(
        self,
        initial_omega_hat: float,
        hidden_dims: list[int],
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if initial_omega_hat <= 0:
            raise ValueError("initial_omega_hat must be positive")
        self.activation = activation
        self.hidden_dims = hidden_dims
        self.linear_encoder = nn.Linear(2, 2, bias=False)
        self.residual_encoder = _make_residual_encoder(hidden_dims, activation)
        self.raw_omega = nn.Parameter(_inverse_softplus(initial_omega_hat - 1e-8))

    @property
    def omega_hat(self) -> torch.Tensor:
        return F.softplus(self.raw_omega) + 1e-8

    def residual(self, initial_conditions: torch.Tensor) -> torch.Tensor:
        return self.residual_encoder(initial_conditions)

    def forward(
        self,
        initial_conditions: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.linear_encoder(initial_conditions) + self.residual(
            initial_conditions,
        )
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = self.omega_hat * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories


def _omega_label(value: float) -> str:
    return f"{value:.1f}".replace(".", "p")


def _lr_label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _residual_variants() -> list[dict[str, Any]]:
    return [
        {
            "model_name": "tiny_residual",
            "hidden_dims": [4],
            "learning_rate": 3e-2,
            "stage0_best_lr_source": "outputs/sho/stage0/lr_sweep_extended: linear_tiny_residual_lr_3em02",
        },
        {
            "model_name": "moderate_residual",
            "hidden_dims": [8, 8],
            "learning_rate": 3e-2,
            "stage0_best_lr_source": "outputs/sho/stage0/lr_sweep_extended: linear_moderate_residual_lr_3em02",
        },
        {
            "model_name": "large_residual",
            "hidden_dims": [32, 32],
            "learning_rate": 1e-2,
            "stage0_best_lr_source": "outputs/sho/stage0/lr_sweep_extended: linear_large_residual_lr_1em02",
        },
    ]


def _coefficient_mse(model: nn.Module, test_data: Any) -> float:
    model.eval()
    with torch.no_grad():
        coefficients, _ = model(test_data.initial_conditions, test_data.times)
    return mean_squared_error(coefficients, test_data.coefficients)


def _best_checkpoint_metrics(
    *,
    model: SHOTrainableLinearResidualLearnedOmegaModel,
    best_model_path: Path,
    test_data: Any,
) -> dict[str, Any]:
    final_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    best_omega_hat = float(model.omega_hat.detach().cpu())
    best_coefficient_mse = _coefficient_mse(model, test_data)
    model.load_state_dict(final_state)
    return {
        "best_omega_hat": best_omega_hat,
        "best_coefficient_mse": best_coefficient_mse,
    }


def _evaluate_run_metrics(
    model: SHOTrainableLinearResidualLearnedOmegaModel,
    train_data: Any,
    test_data: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    _add_linear_weight_metrics(metrics, model.linear_encoder.weight, TRUE_OMEGA)
    _add_residual_metrics(metrics, model, train_data, test_data)
    return metrics


def _success(best_trajectory_mse: float, best_omega_hat: float) -> bool:
    return (
        best_trajectory_mse < SUCCESS_TRAJECTORY_MSE
        and abs(best_omega_hat - TRUE_OMEGA) < SUCCESS_OMEGA_TOLERANCE
    )


def _near_success(best_trajectory_mse: float, best_omega_hat: float) -> bool:
    return (
        best_trajectory_mse < NEAR_SUCCESS_TRAJECTORY_MSE
        and abs(best_omega_hat - TRUE_OMEGA) < NEAR_SUCCESS_OMEGA_TOLERANCE
    )


def run_one(
    *,
    variant: dict[str, Any],
    omega_init: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_train = 512
    num_test = 128
    num_time_steps = 100
    num_periods = 10.0
    batch_size = 512
    max_epochs = BEST_ACHIEVED_MAX_EPOCHS
    learning_rate = float(variant["learning_rate"])
    stopping_criterion = "patience"
    stopping_monitor = "test_loss"
    stopping_min_delta = 1e-13
    stopping_patience = 10000
    activation = "silu"

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
    model = SHOTrainableLinearResidualLearnedOmegaModel(
        initial_omega_hat=omega_init,
        hidden_dims=list(variant["hidden_dims"]),
        activation=activation,
    )
    run_name = f"stage1_{variant['model_name']}_omega_init_{_omega_label(omega_init)}"
    setup = _base_setup(
        comparison_mode="stage1_residual_omega_init_sweep",
        seed=seed,
        omega=TRUE_OMEGA,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_trainable_linear_residual_learned_global_omega",
        encoder_type=f"trainable_linear_plus_{variant['model_name']}",
        activation=activation,
        hidden_dim=list(variant["hidden_dims"]),
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
            "initial_omega_hat_requested": omega_init,
            "omega_hat_initial": float(model.omega_hat.detach().cpu()),
            "omega_parameterization": "softplus(raw_omega) + 1e-8",
            "stage0_best_learning_rate_source": variant["stage0_best_lr_source"],
            "W_trainable": True,
            "residual_branch": f"2 -> {' -> '.join(str(dim) for dim in variant['hidden_dims'])} -> 2",
        }
    )
    best_model_path = output_dir / "best_model.pt"
    history = _train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        num_train=num_train,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_every=500,
        run_name=f"Stage 1 {variant['model_name']} omega_init={omega_init:g}",
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=None,
        best_model_path=best_model_path,
        restore_best_checkpoint=False,
    )

    final_entry = history["losses"][-1]
    best_entry = _history_entry_for_epoch(history, int(history["best_epoch"]))
    best_metrics = _best_checkpoint_metrics(
        model=model,
        best_model_path=best_model_path,
        test_data=test_data,
    )
    structural_metrics = _evaluate_run_metrics(model, train_data, test_data)
    best_omega_hat = float(best_metrics["best_omega_hat"])
    extra_metrics = {
        "omega_hat": float(final_entry["omega_hat"]),
        "best_omega_hat": best_omega_hat,
        "true_omega": TRUE_OMEGA,
        "omega_error": float(final_entry["omega_hat"]) - TRUE_OMEGA,
        "best_omega_error": best_omega_hat - TRUE_OMEGA,
        "relative_omega_error": (float(final_entry["omega_hat"]) - TRUE_OMEGA) / TRUE_OMEGA,
        "best_relative_omega_error": (best_omega_hat - TRUE_OMEGA) / TRUE_OMEGA,
        "final_trajectory_mse": history["final_epoch_test_loss"],
        "best_trajectory_mse": history["best_test_loss"],
        "final_coefficient_mse": _coefficient_mse(model, test_data),
        "best_coefficient_mse": best_metrics["best_coefficient_mse"],
        "recovered_successfully": _success(float(history["best_test_loss"]), best_omega_hat),
        "near_success": _near_success(float(history["best_test_loss"]), best_omega_hat),
        "omega_hat_initial": setup["omega_hat_initial"],
        "plots_generated_from_restored_best_checkpoint": False,
        "W_trainable": True,
    }
    extra_metrics.update(structural_metrics)
    _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=history,
        output_dir=output_dir,
        run_name=f"Stage 1 {variant['model_name']} Omega Init {omega_init:g}",
        omega=TRUE_OMEGA,
        extra_metrics=extra_metrics,
        prediction_omega=float(final_entry["omega_hat"]),
    )


def _summary_row(
    run_name: str,
    model_name: str,
    omega_init: float,
    learning_rate: float,
    run_dir: Path,
) -> dict[str, Any]:
    metrics = _read_json(run_dir / "metrics.json")
    setup = _read_json(run_dir / "setup.json")
    history = _read_json(run_dir / "training_history.json")
    return {
        "run_name": run_name,
        "model_name": model_name,
        "omega_init": omega_init,
        "final_omega_hat": metrics.get("omega_hat"),
        "best_omega_hat": metrics.get("best_omega_hat"),
        "true_omega": TRUE_OMEGA,
        "omega_error": metrics.get("omega_error"),
        "relative_omega_error": metrics.get("relative_omega_error"),
        "final_trajectory_mse": metrics.get("final_trajectory_mse"),
        "best_trajectory_mse": metrics.get("best_trajectory_mse"),
        "final_coefficient_mse": metrics.get("final_coefficient_mse"),
        "best_coefficient_mse": metrics.get("best_coefficient_mse"),
        "mean_abs_predicted_a_test": metrics.get("mean_abs_predicted_a_test"),
        "mean_abs_predicted_b_test": metrics.get("mean_abs_predicted_b_test"),
        "mean_abs_true_a_test": metrics.get("mean_abs_true_a_test"),
        "mean_abs_true_b_test": metrics.get("mean_abs_true_b_test"),
        "residual_rms_test": metrics.get("residual_rms_test"),
        "linear_weight_mse": metrics.get("linear_weight_mse"),
        "learning_rate": learning_rate,
        "epochs_completed": setup.get("num_epochs_completed"),
        "best_epoch": history.get("best_epoch"),
        "stopped_because": setup.get("stopped_because"),
        "recovered_successfully": metrics.get("recovered_successfully"),
        "near_success": metrics.get("near_success"),
        "output_dir": str(run_dir),
    }


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "run_name",
        "model_name",
        "omega_init",
        "final_omega_hat",
        "best_omega_hat",
        "true_omega",
        "omega_error",
        "relative_omega_error",
        "final_trajectory_mse",
        "best_trajectory_mse",
        "final_coefficient_mse",
        "best_coefficient_mse",
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
        "near_success",
    ]
    with (output_dir / "stage1_residual_omega_init_sweep_summary.csv").open(
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
  <title>Stage 1 Residual Omega Initialization Sweep</title>
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
  <h1>Stage 1 Residual Omega Initialization Sweep</h1>
  <p>Trainable linear path plus SiLU residual branch, learned global omega.</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{''.join(html_rows)}</tbody>
  </table>
  {image_html}
</body>
</html>
"""
    (output_dir / "stage1_residual_omega_init_sweep_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def _plot_sweep(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        omega_hats = [entry["omega_hat"] for entry in history["losses"]]
        plt.plot(
            epochs,
            omega_hats,
            label=f"{row['model_name']} init {row['omega_init']}",
            alpha=0.75,
        )
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("epoch")
    plt.ylabel("omega_hat")
    plt.title("Stage 1 Residual Models: Omega Convergence by Model and Init")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "omega_convergence_by_model_and_init.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for row in rows:
        history = _read_json(Path(row["output_dir"]) / "training_history.json")
        epochs = [entry["epoch"] for entry in history["losses"]]
        test_losses = [entry["test_loss"] for entry in history["losses"]]
        plt.plot(
            epochs,
            test_losses,
            label=f"{row['model_name']} init {row['omega_init']}",
            alpha=0.75,
        )
    plt.xlabel("epoch")
    plt.ylabel("test loss")
    plt.yscale("log")
    plt.title("Stage 1 Residual Models: Test Loss by Model and Init")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_by_model_and_init.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for model_name in sorted({row["model_name"] for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name]
        model_rows.sort(key=lambda row: float(row["omega_init"]))
        plt.plot(
            [row["omega_init"] for row in model_rows],
            [1 if row["recovered_successfully"] else 0 for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("omega initialization")
    plt.ylabel("recovered successfully")
    plt.yticks([0, 1], ["no", "yes"])
    plt.title("Stage 1 Residual Models: Success vs Omega Initialization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "success_vs_omega_initialization_by_model.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    for model_name in sorted({row["model_name"] for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name]
        model_rows.sort(key=lambda row: float(row["omega_init"]))
        plt.plot(
            [row["omega_init"] for row in model_rows],
            [row["residual_rms_test"] for row in model_rows],
            marker="o",
            label=model_name,
        )
    plt.xlabel("omega initialization")
    plt.ylabel("residual RMS on test set")
    plt.yscale("log")
    plt.title("Stage 1 Residual Models: Residual RMS by Model and Init")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_rms_by_model_and_init.png")
    plt.close()


def run_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in _residual_variants():
        for omega_init in OMEGA_INITIALIZATIONS:
            run_name = f"{variant['model_name']}_omega_init_{_omega_label(omega_init)}"
            run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
            print(
                f"running Stage 1 residual sweep: {variant['model_name']} omega_init={omega_init:g} lr={variant['learning_rate']:g}",
                flush=True,
            )
            if not (run_dir / "metrics.json").exists():
                run_one(variant=variant, omega_init=omega_init, output_dir=run_dir)
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
    rows = run_sweep()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
