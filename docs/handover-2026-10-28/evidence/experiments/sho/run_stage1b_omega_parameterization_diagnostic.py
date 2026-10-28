"""Stage 1B diagnostic: compare omega parameterizations.

This keeps the Stage 1B input, decoder, loss, architecture size, optimizer, and
late-horizon curriculum fixed. The only change is how the third network output
is transformed into a positive per-trajectory frequency.
"""

from __future__ import annotations

import copy
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

from experiments.sho.run_stage1b_two_state_omega import (
    BATCH_SIZE,
    DELTA_T,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    SUCCESS_OMEGA_TOLERANCE,
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
    _make_residual_branch,
    _metrics,
    _omega_correlation,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/omega_parameterization_diagnostic")
RUN_NAME = "linear_moderate_residual"
HIDDEN_DIMS = [8, 8]
LEARNING_RATE = 1e-2
MAX_EPOCHS_PER_STAGE = 100000
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 20000
SUCCESS_TRAJECTORY_MSE = 1e-15
LOG_EVERY = 10000
LATE_HORIZON_CURRICULUM = [
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
]


def _inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value))))


class SHOTwoStateOmegaParameterizedResidualModel(nn.Module):
    """Linear + moderate residual encoder with a selectable omega transform."""

    def __init__(
        self,
        *,
        hidden_dims: list[int],
        omega_parameterization: str,
        initial_omega_hat: float = 2.0,
    ) -> None:
        super().__init__()
        if initial_omega_hat <= 0:
            raise ValueError("initial_omega_hat must be positive")
        if omega_parameterization not in {"squared_frequency", "log_frequency"}:
            raise ValueError(f"unsupported omega_parameterization: {omega_parameterization}")
        self.hidden_dims = list(hidden_dims)
        self.omega_parameterization = omega_parameterization
        self.linear_encoder = nn.Linear(4, 3)
        self.residual_encoder = _make_residual_branch(self.hidden_dims)
        self._initialize_omega_bias(initial_omega_hat)

    def _initialize_omega_bias(self, initial_omega_hat: float) -> None:
        with torch.no_grad():
            if self.omega_parameterization == "squared_frequency":
                target_q = initial_omega_hat**2
                self.linear_encoder.bias[2].fill_(
                    _inverse_softplus(target_q - 1e-8).item(),
                )
            elif self.omega_parameterization == "log_frequency":
                self.linear_encoder.bias[2].fill_(math.log(initial_omega_hat))

    def _omega_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        if self.omega_parameterization == "squared_frequency":
            q_hat = F.softplus(raw) + 1e-8
            return torch.sqrt(q_hat)
        if self.omega_parameterization == "log_frequency":
            return torch.exp(raw)
        raise RuntimeError("unreachable omega parameterization")

    def forward(
        self,
        two_state_inputs: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_outputs = self.linear_encoder(two_state_inputs) + self.residual_encoder(
            two_state_inputs,
        )
        coefficients = raw_outputs[..., :2]
        omega_hat = self._omega_from_raw(raw_outputs[..., 2])
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = omega_hat.unsqueeze(-1) * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories, omega_hat


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


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


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _train_stage(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    stage_index: int,
    horizon_periods: float,
    variant_name: str,
) -> dict[str, Any]:
    num_train = train_data.two_state_inputs.shape[0]
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    final_train_loss = float("inf")
    final_test_loss = float("inf")
    epoch = 0

    for epoch in range(1, MAX_EPOCHS_PER_STAGE + 1):
        model.train()
        shuffled_indices = torch.randperm(num_train)
        loss_sum = 0.0
        example_count = 0
        for start in range(0, num_train, BATCH_SIZE):
            batch_indices = shuffled_indices[start : start + BATCH_SIZE]
            optimizer.zero_grad()
            _, trajectories, _ = model(
                train_data.two_state_inputs[batch_indices],
                train_data.times[batch_indices],
            )
            loss = mse_loss(trajectories, train_data.trajectories[batch_indices])
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            batch_count = len(batch_indices)
            loss_sum += loss.item() * batch_count
            example_count += batch_count

        if stopped_because == "nan_or_inf_loss":
            break

        final_train_loss = loss_sum / example_count
        model.eval()
        with torch.no_grad():
            _, test_trajectories, _ = model(test_data.two_state_inputs, test_data.times)
            final_test_loss = mse_loss(test_trajectories, test_data.trajectories).item()

        if final_test_loss < best_test_loss - STOPPING_MIN_DELTA:
            best_test_loss = final_test_loss
            best_train_loss = final_train_loss
            best_epoch = epoch
            best_model_state = _clone_state_dict(model.state_dict())
            best_optimizer_state = _clone_state_dict(optimizer.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                (
                    f"{variant_name} stage {stage_index:02d} horizon {horizon_periods:g} "
                    f"epoch {epoch:06d} train_loss={final_train_loss:.6e} "
                    f"test_loss={final_test_loss:.6e} best_test={best_test_loss:.6e}"
                ),
                flush=True,
            )

        if epochs_since_improvement >= STOPPING_PATIENCE:
            stopped_because = "patience_exhausted"
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)

    metrics, _, _, omega_hat = _evaluate_model(model, test_data)
    return {
        "stage_index": stage_index,
        "horizon_periods": horizon_periods,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "best_omega_mean": omega_hat.mean().item(),
        **metrics,
    }


def _save_stage_omega_plots(
    *,
    stage_dir: Path,
    stage_index: int,
    horizon_periods: float,
    omega_hat: torch.Tensor,
    true_omega: torch.Tensor,
    variant_name: str,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    omega_hat_cpu = omega_hat.detach().cpu()
    true_omega_cpu = true_omega.detach().cpu()
    error = omega_hat_cpu - true_omega_cpu

    plt.figure(figsize=(6, 6))
    plt.scatter(true_omega_cpu, omega_hat_cpu, s=18, alpha=0.75)
    plt.plot([1.5, 2.5], [1.5, 2.5], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("predicted omega")
    plt.title(
        f"Stage 1B {variant_name} Stage {stage_index}: {horizon_periods:g} periods",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(stage_dir / f"stage_{stage_index:02d}_predicted_vs_true_omega.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(error, bins=40)
    plt.xlabel("omega_hat - omega_true")
    plt.ylabel("count")
    plt.title(f"Stage 1B {variant_name} Stage {stage_index}: Omega Error")
    plt.tight_layout()
    plt.savefig(stage_dir / f"stage_{stage_index:02d}_omega_error_histogram.png")
    plt.close()


def _save_final_trajectory_examples(
    *,
    output_dir: Path,
    model: nn.Module,
    test_data: Any,
    variant_name: str,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        _, trajectories, _ = model(test_data.two_state_inputs, test_data.times)
    example_count = min(num_examples, test_data.trajectories.shape[0])
    fig, axes = plt.subplots(example_count, 1, figsize=(8, 2.4 * example_count))
    if example_count == 1:
        axes = [axes]
    for plot_index in range(example_count):
        ax = axes[plot_index]
        ax.plot(
            test_data.times[plot_index].detach().cpu(),
            test_data.trajectories[plot_index].detach().cpu(),
            label="target",
            linewidth=2,
        )
        ax.plot(
            test_data.times[plot_index].detach().cpu(),
            trajectories[plot_index].detach().cpu(),
            "--",
            label="prediction",
            linewidth=1.5,
        )
        ax.set_title(f"{variant_name}: final 10-period trajectory example {plot_index}")
        ax.set_xlabel("time")
        ax.set_ylabel("x(t)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_stage_trajectory_examples.png")
    plt.close(fig)


def _save_coefficient_scatter(
    *,
    output_dir: Path,
    coefficients: torch.Tensor,
    true_coefficients: torch.Tensor,
    title: str,
) -> None:
    pred = coefficients.detach().cpu()
    true = true_coefficients.detach().cpu()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, index, name in zip(axes, [0, 1], ["a", "b"]):
        axis.scatter(true[:, index], pred[:, index], s=18, alpha=0.75)
        min_value = min(true[:, index].min().item(), pred[:, index].min().item())
        max_value = max(true[:, index].max().item(), pred[:, index].max().item())
        axis.plot([min_value, max_value], [min_value, max_value], "k--")
        axis.set_xlabel(f"true {name}")
        axis.set_ylabel(f"predicted {name}")
        axis.set_title(f"{name} coefficient")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "coefficient_predicted_vs_true.png")
    plt.close(fig)


def _save_metric_plots(rows: list[dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    horizons = [float(row["horizon_periods"]) for row in rows]
    plot_specs = [
        ("trajectory_mse_vs_horizon.png", "trajectory_mse", "trajectory MSE"),
        ("coefficient_mse_vs_horizon.png", "coefficient_mse", "coefficient MSE"),
        ("omega_mse_vs_horizon.png", "omega_mse", "omega MSE"),
        ("omega_correlation_vs_horizon.png", "omega_correlation", "omega correlation"),
    ]
    for filename, key, ylabel in plot_specs:
        plt.figure(figsize=(7, 4.5))
        plt.plot(horizons, [float(row[key]) for row in rows], marker="o")
        plt.xlabel("horizon in periods")
        plt.ylabel(ylabel)
        if key != "omega_correlation":
            plt.yscale("log")
        plt.title(f"{title_prefix}: {ylabel}")
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def _write_html(
    *,
    output_dir: Path,
    title: str,
    setup: dict[str, Any],
    rows: list[dict[str, Any]],
    final_metrics: dict[str, Any],
) -> None:
    columns = [
        "stage_index",
        "horizon_periods",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_abs_omega_error",
        "max_omega_error",
        "best_epoch",
        "best_test_loss",
        "stopped_because",
    ]
    setup_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><pre>{escape(str(value))}</pre></td></tr>"
        for key, value in setup.items()
    )
    metric_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in final_metrics.items()
    )
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plots = [
        "trajectory_mse_vs_horizon.png",
        "coefficient_mse_vs_horizon.png",
        "omega_mse_vs_horizon.png",
        "omega_correlation_vs_horizon.png",
        "predicted_vs_true_omega.png",
        "omega_error_histogram.png",
        "coefficient_predicted_vs_true.png",
        "final_stage_trajectory_examples.png",
    ]
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in plots
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
    pre {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Final Metrics</h2>
  <table>{metric_rows}</table>
  <h2>Stage Metrics</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _save_final_omega_plots(
    *,
    output_dir: Path,
    omega_hat: torch.Tensor,
    true_omega: torch.Tensor,
    title_prefix: str,
) -> None:
    omega_hat_cpu = omega_hat.detach().cpu()
    true_omega_cpu = true_omega.detach().cpu()
    error = omega_hat_cpu - true_omega_cpu

    plt.figure(figsize=(6, 6))
    plt.scatter(true_omega_cpu, omega_hat_cpu, s=18, alpha=0.75)
    min_value = min(true_omega_cpu.min().item(), omega_hat_cpu.min().item())
    max_value = max(true_omega_cpu.max().item(), omega_hat_cpu.max().item())
    plt.plot([min_value, max_value], [min_value, max_value], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("predicted omega")
    plt.title(f"{title_prefix}: Predicted vs True Omega")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_true_omega.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(error, bins=40)
    plt.xlabel("omega_hat - omega_true")
    plt.ylabel("count")
    plt.title(f"{title_prefix}: Omega Error Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "omega_error_histogram.png")
    plt.close()


def _run_variant(omega_parameterization: str) -> dict[str, Any]:
    variant_output_dir = OUTPUT_DIR / omega_parameterization
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    model = SHOTwoStateOmegaParameterizedResidualModel(
        hidden_dims=HIDDEN_DIMS,
        omega_parameterization=omega_parameterization,
        initial_omega_hat=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = {
        "stage": "stage1B",
        "experiment": "omega_parameterization_diagnostic",
        "run_name": f"{RUN_NAME}_{omega_parameterization}",
        "input": "(x0, v0, x1, v1)",
        "output": "(a, b, omega_hat_i)",
        "omega_parameterization": omega_parameterization,
        "omega_transform": (
            "q_hat = softplus(raw_q) + 1e-8; omega_hat = sqrt(q_hat)"
            if omega_parameterization == "squared_frequency"
            else "omega_hat = exp(raw_l)"
        ),
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "model": "linear + moderate residual encoder",
        "hidden_dims": HIDDEN_DIMS,
        "activation": "SiLU",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "loss": "trajectory reconstruction only",
        "omega_supervision": False,
        "coefficient_supervision": False,
        "auxiliary_losses": False,
        "regularization": False,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
        "curriculum_horizons_periods": LATE_HORIZON_CURRICULUM,
        "random_seed": SEED,
        "run_date": RUN_DATE,
    }
    rows = []
    for stage_index, horizon_periods in enumerate(LATE_HORIZON_CURRICULUM, start=1):
        train_data = _build_data_from_base(
            train_initial_conditions,
            train_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
        )
        test_data = _build_data_from_base(
            test_initial_conditions,
            test_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
        )
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            variant_name=omega_parameterization,
        )
        rows.append(row)
        metrics, _, _, omega_hat = _evaluate_model(model, test_data)
        _save_stage_omega_plots(
            stage_dir=variant_output_dir / "stage_omega_plots",
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            omega_hat=omega_hat,
            true_omega=test_data.omegas,
            variant_name=omega_parameterization,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
                "stage_metrics_recomputed": metrics,
            },
            variant_output_dir / f"checkpoint_after_stage_{stage_index:02d}.pt",
        )

    final_test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=LATE_HORIZON_CURRICULUM[-1],
    )
    final_metrics, coefficients, _, omega_hat = _evaluate_model(model, final_test_data)
    final_metrics["recovered_successfully"] = (
        final_metrics["trajectory_mse"] < SUCCESS_TRAJECTORY_MSE
        and final_metrics["max_omega_error"] < SUCCESS_OMEGA_TOLERANCE
    )
    final_metrics["omega_parameterization"] = omega_parameterization
    final_metrics["run_name"] = setup["run_name"]
    final_metrics["epochs_completed_total"] = sum(int(row["epochs_completed"]) for row in rows)
    final_metrics["final_stage_best_epoch"] = rows[-1]["best_epoch"]
    final_metrics["final_stage_stopped_because"] = rows[-1]["stopped_because"]
    final_metrics["final_stage_best_test_loss"] = rows[-1]["best_test_loss"]
    final_metrics["omega_hat_mean"] = omega_hat.mean().item()
    final_metrics["omega_hat_std"] = omega_hat.std().item()

    _write_json(variant_output_dir / "setup.json", setup)
    _write_json(variant_output_dir / "metrics.json", final_metrics)
    _write_json(variant_output_dir / "curriculum_stage_summary.json", {"stages": rows})
    _write_csv(variant_output_dir / "curriculum_stage_summary.csv", rows)
    torch.save(model.state_dict(), variant_output_dir / "learned_parameters.pt")
    _save_metric_plots(
        rows,
        variant_output_dir,
        f"Stage 1B {omega_parameterization}",
    )
    _save_final_omega_plots(
        output_dir=variant_output_dir,
        omega_hat=omega_hat,
        true_omega=final_test_data.omegas,
        title_prefix=f"Stage 1B {omega_parameterization}",
    )
    _save_coefficient_scatter(
        output_dir=variant_output_dir,
        coefficients=coefficients,
        true_coefficients=final_test_data.coefficients,
        title=f"Stage 1B {omega_parameterization}: Coefficients",
    )
    _save_final_trajectory_examples(
        output_dir=variant_output_dir,
        model=model,
        test_data=final_test_data,
        variant_name=omega_parameterization,
    )
    _write_html(
        output_dir=variant_output_dir,
        title=f"Stage 1B Omega Parameterization: {omega_parameterization}",
        setup=setup,
        rows=rows,
        final_metrics=final_metrics,
    )
    return final_metrics


def _save_comparison_plots(rows: list[dict[str, Any]]) -> None:
    metric_specs = [
        ("final_trajectory_mse_by_parameterization.png", "trajectory_mse", "trajectory MSE", True),
        ("final_coefficient_mse_by_parameterization.png", "coefficient_mse", "coefficient MSE", True),
        ("final_omega_mse_by_parameterization.png", "omega_mse", "omega MSE", True),
        ("final_omega_correlation_by_parameterization.png", "omega_correlation", "omega correlation", False),
    ]
    labels = [str(row["omega_parameterization"]) for row in rows]
    for filename, key, ylabel, log_scale in metric_specs:
        plt.figure(figsize=(7, 4.5))
        plt.bar(labels, [float(row[key]) for row in rows])
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale("log")
        plt.title(f"Stage 1B Omega Parameterization Diagnostic: {ylabel}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "omega_parameterization",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_abs_omega_error",
        "max_omega_error",
        "final_stage_best_epoch",
        "final_stage_stopped_because",
        "epochs_completed_total",
        "recovered_successfully",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in [
            "final_trajectory_mse_by_parameterization.png",
            "final_coefficient_mse_by_parameterization.png",
            "final_omega_mse_by_parameterization.png",
            "final_omega_correlation_by_parameterization.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Omega Parameterization Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Omega Parameterization Diagnostic</h1>
  <p>Same two-state input, exact SHO decoder, reconstruction loss, linear + moderate residual encoder, and late-horizon curriculum. Only the omega parameterization differs.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
  <h2>Run Directories</h2>
  <ul>
    <li><a href="squared_frequency/index.html">squared_frequency</a></li>
    <li><a href="log_frequency/index.html">log_frequency</a></li>
  </ul>
</body>
</html>
"""
    (OUTPUT_DIR / "omega_parameterization_diagnostic_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def run_omega_parameterization_diagnostic() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_variant("squared_frequency"),
        _run_variant("log_frequency"),
    ]
    _write_csv(OUTPUT_DIR / "omega_parameterization_diagnostic_summary.csv", rows)
    _write_json(OUTPUT_DIR / "omega_parameterization_diagnostic_summary.json", {"runs": rows})
    _save_comparison_plots(rows)
    _write_summary_html(rows)
    return rows


def main() -> None:
    rows = run_omega_parameterization_diagnostic()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
