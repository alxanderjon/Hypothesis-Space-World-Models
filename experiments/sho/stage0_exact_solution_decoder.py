"""Runnable Stage 0 experiments for fixed-omega SHO coefficient recovery."""

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any
from html import escape

from hypothesis_space_world_models.evaluation.metrics import mean_squared_error
from hypothesis_space_world_models.evaluation.plots import (
    plot_omega_gradient_history,
    plot_omega_history,
    plot_phase_space_comparison,
    plot_phase_space_side_by_side,
    plot_test_loss_and_omega_history,
    plot_training_loss,
    plot_trajectory_comparison,
)
from hypothesis_space_world_models.models import (
    SHOFixedLinearResidualModel,
    SHOLearnedGlobalOmegaSolutionModel,
    SHOLinearExactSolutionModel,
    SHOExactSolutionModel,
    SHOTrainableLinearResidualModel,
    sho_target_weight_matrix,
)
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.systems.sho import decode_sho, decode_sho_velocity
from hypothesis_space_world_models.training import mse_loss, set_seed

import torch


DEFAULT_RUN_DATE = "2026-06-06"
DEFAULT_COMPARISON_MODE = "best_achieved"
SHARED_BUDGET_MAX_EPOCHS = 5000
BEST_ACHIEVED_MAX_EPOCHS = 100000
SHARED_BUDGET_LEARNING_RATE = 1e-3


def _run_dir(
    run_name: str,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    run_date: str = DEFAULT_RUN_DATE,
) -> Path:
    return Path("outputs/sho/stage0") / comparison_mode / f"{run_date}_{run_name}"


def _default_max_epochs(comparison_mode: str) -> int:
    if comparison_mode == "shared_budget":
        return SHARED_BUDGET_MAX_EPOCHS
    return BEST_ACHIEVED_MAX_EPOCHS


def _default_stopping_criterion(comparison_mode: str) -> str:
    if comparison_mode == "shared_budget":
        return "max_epochs"
    return "patience"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _html_table(data: dict[str, Any]) -> str:
    rows = []
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            value_text = json.dumps(value, indent=2)
        else:
            value_text = _format_value(value)
        rows.append(
            "<tr>"
            f"<th>{escape(str(key))}</th>"
            f"<td><pre>{escape(value_text)}</pre></td>"
            "</tr>"
        )
    return "<table>" + "\n".join(rows) + "</table>"


def _write_run_index(
    output_dir: Path,
    run_name: str,
    setup: dict[str, Any],
    metrics: dict[str, Any],
    training_summary: dict[str, Any],
    optimizer_analysis: str,
    images: list[tuple[str, str]],
) -> None:
    model_description = {
        "model_type": setup.get("model_type"),
        "encoder_type": setup.get("encoder_type"),
        "activation": setup.get("activation"),
        "hidden_dim": setup.get("hidden_dim"),
    }
    stage1_diagnostic_keys = [
        "mean_abs_predicted_a_test",
        "mean_abs_predicted_b_test",
        "mean_abs_true_a_test",
        "mean_abs_true_b_test",
        "coefficient_mse",
        "trajectory_mse",
        "omega_hat",
    ]
    stage1_diagnostics = {
        key: metrics[key]
        for key in stage1_diagnostic_keys
        if key in metrics
    }
    stage1_diagnostics_html = (
        "<h2>Stage 1 Diagnostics</h2>\n"
        f"{_html_table(stage1_diagnostics)}"
        if stage1_diagnostics
        else ""
    )
    image_html = "\n".join(
        f"<section><h2>{escape(title)}</h2><img src=\"{escape(path)}\" alt=\"{escape(title)}\"></section>"
        for title, path in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(run_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 0.4rem; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; max-width: 980px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 280px; background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>{escape(run_name)}</h1>
  <h2>Model Description</h2>
  {_html_table(model_description)}
  <h2>Setup / Configuration</h2>
  {_html_table(setup)}
  <h2>Training Metrics</h2>
  {_html_table(training_summary)}
  {stage1_diagnostics_html}
  <h2>Evaluation Metrics</h2>
  {_html_table(metrics)}
  {image_html}
  <h2>Optimizer Analysis</h2>
  <pre>{escape(optimizer_analysis)}</pre>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _optimizer_analysis_text(
    run_name: str,
    setup: dict[str, Any],
    metrics: dict[str, Any],
    training_history: dict[str, Any],
) -> str:
    losses = training_history["losses"]
    final_train_loss = training_history["final_train_loss"]
    final_test_loss = training_history["final_test_loss"]
    final_epoch_train_loss = training_history["final_epoch_train_loss"]
    final_epoch_test_loss = training_history["final_epoch_test_loss"]
    best_train_loss = training_history["best_train_loss"]
    best_test_loss = training_history["best_test_loss"]
    best_train_epoch = training_history["epoch_of_best_train_loss"]
    best_test_epoch = training_history["epoch_of_best_test_loss"]
    optimizer = setup["optimizer"]
    learning_rate = setup["learning_rate"]
    encoder_type = setup["encoder_type"]

    lines = [
        f"Run: SHO Stage 0: {run_name}",
        "",
        f"Comparison mode: {setup.get('comparison_mode')}",
        f"Optimizer: {optimizer}",
        f"Learning rate: {learning_rate}",
        "Learning-rate schedule: none; the learning rate is constant throughout training.",
        f"Stopping criterion: {setup.get('stopping_criterion')}",
        f"Stopping monitor: {setup.get('stopping_monitor')}",
        f"Stopped because: {setup.get('stopped_because')}",
        f"Epochs completed: {setup.get('num_epochs_completed')}",
        f"Final train loss: {final_train_loss:.6e}",
        f"Final test loss: {final_test_loss:.6e}",
        f"Final epoch train loss: {final_epoch_train_loss:.6e}",
        f"Final epoch test loss: {final_epoch_test_loss:.6e}",
        f"Best train loss: {best_train_loss:.6e}",
        f"Best test loss: {best_test_loss:.6e}",
        f"Best checkpoint epoch: {training_history.get('best_epoch')}",
        f"Reported metrics source: {training_history.get('reported_metrics_source')}",
        f"Epoch of best train loss: {best_train_epoch}",
        f"Epoch of best test loss: {best_test_epoch}",
        "",
    ]

    if encoder_type == "linear":
        losses_by_epoch = {int(entry["epoch"]): float(entry["train_loss"]) for entry in losses}
        plateau_samples = [
            (epoch, losses_by_epoch[epoch])
            for epoch in (350, 1000, 2000, 3300, 4500, 4650, 4750, 5000)
            if epoch in losses_by_epoch
        ]
        lines.extend(
            [
                "Linear encoder interpretation:",
                "The encoder is exactly capable of representing the target coefficient map,",
                "so the remaining behavior is optimizer dynamics rather than model expressivity.",
                "",
                "The loss drops rapidly at first and may then enter a near-flat region where",
                "the weights are already very close to the exact matrix. On a log scale this",
                "can look like a plateau, but sampled train losses reveal whether it is still",
                "making tiny improvements:",
            ]
        )
        for epoch, loss in plateau_samples:
            lines.append(f"  epoch {epoch}: {loss:.6e}")
        lines.extend(
            [
                "",
                "If a late drop appears after a long plateau, it is consistent with Adam operating",
                "on very small residuals: its adaptive moments and bias correction can keep making",
                "small parameter updates after the apparent plateau. Once those residual errors",
                "line up with the exact linear map, the trajectory loss can reach machine precision.",
                "",
                "This is expected optimizer behavior for an overdetermined but exactly solvable",
                "linear map trained with Adam at a fixed learning rate. The model is already",
                "functionally exact during much of the plateau; the late change mostly improves",
                "tiny floating-point residuals.",
                "",
                "SGD would likely behave more smoothly and may require careful learning-rate",
                "tuning; it would not usually show the same adaptive-moment late jump. LBFGS",
                "would likely solve this tiny deterministic linear problem in far fewer steps",
                "because it uses curvature information. The analytic solution is immediate:",
                "the target matrix is [[1, 0], [0, 1 / omega]], so it has zero optimization",
                "error by construction.",
                "",
                f"Final weight MSE: {metrics.get('weight_mse', 'N/A')}",
                f"Final coefficient MSE: {metrics.get('coefficient_mse', 'N/A')}",
                f"Final trajectory MSE: {metrics.get('trajectory_mse', 'N/A')}",
            ]
        )
    else:
        lines.extend(
            [
                "Optimizer interpretation:",
                "Adam is used with a constant learning rate and no scheduler. The full training",
                "history is saved in training_history.json, and training_loss.png is plotted from",
                "all epochs. This run is a nonlinear learned encoder, so any residual error is",
                "from optimization and the chosen MLP parameterization, not from the fixed analytic",
                "decoder.",
                "",
                f"Final coefficient MSE: {metrics.get('coefficient_mse', 'N/A')}",
                f"Final trajectory MSE: {metrics.get('trajectory_mse', 'N/A')}",
            ]
        )

    return "\n".join(lines)


def _target_linear_weight_matrix(omega: float) -> torch.Tensor:
    return sho_target_weight_matrix(omega)


def _energy_diagnostics(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    omega: float,
) -> dict[str, float]:
    energy = 0.5 * (velocities**2 + (omega * positions) ** 2)
    energy_drift = energy - energy[:, :1]

    amplitude = torch.sqrt(positions**2 + (velocities / omega) ** 2)
    amplitude_drift = amplitude - amplitude[:, :1]

    return {
        "initial_energy": energy[:, 0].mean().item(),
        "final_energy": energy[:, -1].mean().item(),
        "mean_energy_drift": energy_drift.abs().mean().item(),
        "max_energy_drift": energy_drift.abs().max().item(),
        "initial_amplitude": amplitude[:, 0].mean().item(),
        "final_amplitude": amplitude[:, -1].mean().item(),
        "mean_amplitude_drift": amplitude_drift.abs().mean().item(),
        "max_amplitude_drift": amplitude_drift.abs().max().item(),
    }


def _error_diagnostics(
    target_positions: torch.Tensor,
    target_velocities: torch.Tensor,
    prediction_positions: torch.Tensor,
    prediction_velocities: torch.Tensor,
) -> dict[str, float]:
    position_error = prediction_positions - target_positions
    velocity_error = prediction_velocities - target_velocities
    return {
        "max_abs_x_error": position_error.abs().max().item(),
        "max_abs_v_error": velocity_error.abs().max().item(),
        "rms_x_error": torch.sqrt(torch.mean(position_error**2)).item(),
        "rms_v_error": torch.sqrt(torch.mean(velocity_error**2)).item(),
    }


def _train_model(
    model: torch.nn.Module,
    train_data: Any,
    test_data: Any,
    num_train: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    log_every: int,
    run_name: str,
    stopping_criterion: str,
    stopping_monitor: str,
    stopping_min_delta: float,
    stopping_patience: int,
    target_loss: float | None,
    best_model_path: Path,
    restore_best_checkpoint: bool,
    start_epoch: int = 0,
    initial_losses: list[dict[str, float | int]] | None = None,
    initial_best_checkpoint_epoch: int = 0,
    initial_best_checkpoint_train_loss: float = float("inf"),
    initial_best_checkpoint_test_loss: float = float("inf"),
    initial_best_significant_monitored_loss: float = float("inf"),
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    raw_omega_in_optimizer = (
        hasattr(model, "raw_omega") and id(model.raw_omega) in optimizer_parameter_ids
    )
    losses: list[dict[str, float | int]] = list(initial_losses or [])
    best_significant_monitored_loss = initial_best_significant_monitored_loss
    best_checkpoint_monitored_loss = (
        initial_best_checkpoint_train_loss
        if stopping_monitor == "train_loss"
        else initial_best_checkpoint_test_loss
    )
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    best_checkpoint_epoch = initial_best_checkpoint_epoch
    best_checkpoint_train_loss = initial_best_checkpoint_train_loss
    best_checkpoint_test_loss = initial_best_checkpoint_test_loss
    best_train_loss_so_far = min(
        [float(entry["train_loss"]) for entry in losses],
        default=float("inf"),
    )
    best_test_loss_so_far = min(
        [float(entry["test_loss"]) for entry in losses],
        default=float("inf"),
    )
    failed_due_to_nan_or_inf = False
    if losses and math.isfinite(best_checkpoint_monitored_loss):
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_model_path)

    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        shuffled_indices = torch.randperm(num_train)
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        nonfinite_loss_seen = False
        omega_grad_abs = None
        coefficient_weight_grad_norm = None
        for start in range(0, num_train, batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]
            initial_conditions = train_data.initial_conditions[batch_indices]
            target_trajectories = train_data.trajectories[batch_indices]

            optimizer.zero_grad()
            _, pred_trajectories = model(initial_conditions, train_data.times)
            loss = mse_loss(pred_trajectories, target_trajectories)
            if not torch.isfinite(loss):
                nonfinite_loss_seen = True
                failed_due_to_nan_or_inf = True
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            if hasattr(model, "raw_omega") and model.raw_omega.grad is not None:
                omega_grad_abs = float(model.raw_omega.grad.detach().abs().cpu())
            coefficient_weight = None
            if hasattr(model, "encoder") and hasattr(model.encoder, "weight"):
                coefficient_weight = model.encoder.weight
            elif hasattr(model, "linear_encoder") and hasattr(
                model.linear_encoder,
                "weight",
            ):
                coefficient_weight = model.linear_encoder.weight
            if coefficient_weight is not None and coefficient_weight.grad is not None:
                coefficient_weight_grad_norm = float(
                    torch.linalg.vector_norm(
                        coefficient_weight.grad.detach(),
                    ).cpu()
                )
            optimizer.step()
            batch_count = len(batch_indices)
            epoch_loss_sum += loss.item() * batch_count
            epoch_example_count += batch_count

        if nonfinite_loss_seen or epoch_example_count == 0:
            train_loss = float("inf")
            test_loss = float("inf")
        else:
            train_loss = epoch_loss_sum / epoch_example_count
            model.eval()
            with torch.no_grad():
                _, test_pred_trajectories = model(
                    test_data.initial_conditions,
                    test_data.times,
                )
                test_loss = mse_loss(
                    test_pred_trajectories,
                    test_data.trajectories,
                ).item()
            if not math.isfinite(test_loss):
                failed_due_to_nan_or_inf = True
                stopped_because = "nan_or_inf_loss"

        if failed_due_to_nan_or_inf:
            loss_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "best_train_loss_so_far": best_train_loss_so_far,
                "best_test_loss_so_far": best_test_loss_so_far,
                "failed_due_to_nan_or_inf": True,
            }
            if hasattr(model, "omega_hat"):
                omega_hat = float(model.omega_hat.detach().cpu())
                raw_omega = float(model.raw_omega.detach().cpu())
                true_omega = float(train_data.omega)
                loss_entry.update(
                    {
                        "omega_hat": omega_hat,
                        "omega_error": omega_hat - true_omega,
                        "relative_omega_error": (omega_hat - true_omega) / true_omega,
                        "raw_omega": raw_omega,
                        "omega_grad_abs": omega_grad_abs,
                        "omega_grad_norm": omega_grad_abs,
                        "coefficient_weight_grad_norm": coefficient_weight_grad_norm,
                    }
                )
            losses.append(loss_entry)
            logging.info(
                "%s epoch %04d stopped: NaN or Inf loss",
                run_name,
                epoch,
            )
            break

        best_train_loss_so_far = min(best_train_loss_so_far, train_loss)
        best_test_loss_so_far = min(best_test_loss_so_far, test_loss)
        loss_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "best_train_loss_so_far": best_train_loss_so_far,
            "best_test_loss_so_far": best_test_loss_so_far,
        }
        if hasattr(model, "omega_hat"):
            omega_hat = float(model.omega_hat.detach().cpu())
            raw_omega = float(model.raw_omega.detach().cpu())
            true_omega = float(train_data.omega)
            loss_entry.update(
                {
                    "omega_hat": omega_hat,
                    "omega_error": omega_hat - true_omega,
                    "relative_omega_error": (omega_hat - true_omega) / true_omega,
                    "raw_omega": raw_omega,
                    "omega_grad_abs": omega_grad_abs,
                    "omega_grad_norm": omega_grad_abs,
                    "coefficient_weight_grad_norm": coefficient_weight_grad_norm,
                }
            )
        losses.append(loss_entry)
        monitored_loss = train_loss if stopping_monitor == "train_loss" else test_loss
        if monitored_loss < best_checkpoint_monitored_loss:
            best_checkpoint_monitored_loss = monitored_loss
            best_checkpoint_epoch = epoch
            best_checkpoint_train_loss = train_loss
            best_checkpoint_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)

        if monitored_loss < best_significant_monitored_loss - stopping_min_delta:
            best_significant_monitored_loss = monitored_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1 or epoch == max_epochs):
            if hasattr(model, "omega_hat"):
                logging.info(
                    (
                        "%s epoch %04d train_loss: %.6e test_loss: %.6e "
                        "omega_hat: %.6e raw_omega: %.6e "
                        "abs_dL_d_raw_omega: %s coefficient_weight_grad_norm: %s"
                    ),
                    run_name,
                    epoch,
                    train_loss,
                    test_loss,
                    float(model.omega_hat.detach().cpu()),
                    float(model.raw_omega.detach().cpu()),
                    _format_value(omega_grad_abs),
                    _format_value(coefficient_weight_grad_norm),
                )
            else:
                logging.info(
                    "%s epoch %04d train_loss: %.6e test_loss: %.6e",
                    run_name,
                    epoch,
                    train_loss,
                    test_loss,
                )

        if target_loss is not None and monitored_loss <= target_loss:
            stopped_because = f"{stopping_monitor}_target_reached"
            break
        if (
            stopping_criterion == "patience"
            and epochs_since_improvement >= stopping_patience
        ):
            stopped_because = "patience_exhausted"
            break

    best_train_entry = min(losses, key=lambda entry: float(entry["train_loss"]))
    best_test_entry = min(losses, key=lambda entry: float(entry["test_loss"]))
    if restore_best_checkpoint and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return {
        "losses": losses,
        "num_epochs_completed": losses[-1]["epoch"],
        "final_epoch_train_loss": losses[-1]["train_loss"],
        "final_epoch_test_loss": losses[-1]["test_loss"],
        "final_train_loss": best_checkpoint_train_loss,
        "final_test_loss": best_checkpoint_test_loss,
        "best_train_loss": best_train_entry["train_loss"],
        "best_test_loss": best_test_entry["test_loss"],
        "epoch_of_best_train_loss": best_train_entry["epoch"],
        "epoch_of_best_test_loss": best_test_entry["epoch"],
        "best_epoch": best_checkpoint_epoch,
        "best_checkpoint_train_loss": best_checkpoint_train_loss,
        "best_checkpoint_test_loss": best_checkpoint_test_loss,
        "restore_best_checkpoint": restore_best_checkpoint,
        "reported_metrics_source": "best_checkpoint" if restore_best_checkpoint else "final_epoch",
        "stopped_because": stopped_because,
        "stopping_monitor": stopping_monitor,
        "stopping_criterion": stopping_criterion,
        "stopping_min_delta": stopping_min_delta,
        "stopping_patience": stopping_patience,
        "target_loss": target_loss,
        "failed_due_to_nan_or_inf": failed_due_to_nan_or_inf,
        "raw_omega_in_optimizer": raw_omega_in_optimizer,
        "model_train_eval_modes_explicit": True,
    }


def _evaluate_loss(model: torch.nn.Module, data: Any) -> float:
    with torch.no_grad():
        _, pred_trajectories = model(data.initial_conditions, data.times)
        return mse_loss(pred_trajectories, data.trajectories).item()


def _train_model_adam_then_lbfgs(
    model: torch.nn.Module,
    train_data: Any,
    test_data: Any,
    num_train: int,
    adam_epochs: int,
    adam_learning_rate: float,
    lbfgs_max_iter: int,
    lbfgs_learning_rate: float,
    batch_size: int,
    log_every: int,
    run_name: str,
    stopping_monitor: str,
    best_model_path: Path,
    restore_best_checkpoint: bool,
) -> dict[str, Any]:
    """Train with Adam, then polish the same model with LBFGS."""
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=adam_learning_rate)
    losses: list[dict[str, float | int | str]] = []
    best_checkpoint_monitored_loss = float("inf")
    best_checkpoint_epoch = 0
    best_checkpoint_train_loss = float("inf")
    best_checkpoint_test_loss = float("inf")
    best_train_loss_so_far = float("inf")
    best_test_loss_so_far = float("inf")

    def record_epoch(epoch: int, phase: str, train_loss: float, test_loss: float) -> None:
        nonlocal best_checkpoint_monitored_loss
        nonlocal best_checkpoint_epoch
        nonlocal best_checkpoint_train_loss
        nonlocal best_checkpoint_test_loss
        nonlocal best_train_loss_so_far
        nonlocal best_test_loss_so_far

        best_train_loss_so_far = min(best_train_loss_so_far, train_loss)
        best_test_loss_so_far = min(best_test_loss_so_far, test_loss)
        losses.append(
            {
                "epoch": epoch,
                "phase": phase,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "best_train_loss_so_far": best_train_loss_so_far,
                "best_test_loss_so_far": best_test_loss_so_far,
            }
        )

        monitored_loss = train_loss if stopping_monitor == "train_loss" else test_loss
        if monitored_loss < best_checkpoint_monitored_loss:
            best_checkpoint_monitored_loss = monitored_loss
            best_checkpoint_epoch = epoch
            best_checkpoint_train_loss = train_loss
            best_checkpoint_test_loss = test_loss
            torch.save(model.state_dict(), best_model_path)

    for epoch in range(1, adam_epochs + 1):
        shuffled_indices = torch.randperm(num_train)
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        for start in range(0, num_train, batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]
            initial_conditions = train_data.initial_conditions[batch_indices]
            target_trajectories = train_data.trajectories[batch_indices]

            adam_optimizer.zero_grad()
            _, pred_trajectories = model(initial_conditions, train_data.times)
            loss = mse_loss(pred_trajectories, target_trajectories)
            loss.backward()
            adam_optimizer.step()
            batch_count = len(batch_indices)
            epoch_loss_sum += loss.item() * batch_count
            epoch_example_count += batch_count

        train_loss = epoch_loss_sum / epoch_example_count
        test_loss = _evaluate_loss(model, test_data)
        record_epoch(epoch, "adam", train_loss, test_loss)

        if log_every > 0 and (
            epoch % log_every == 0 or epoch == 1 or epoch == adam_epochs
        ):
            logging.info(
                "%s Adam epoch %04d train_loss: %.6e test_loss: %.6e",
                run_name,
                epoch,
                train_loss,
                test_loss,
            )

    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_learning_rate,
        max_iter=1,
        line_search_fn="strong_wolfe",
    )
    for step in range(1, lbfgs_max_iter + 1):
        epoch = adam_epochs + step

        def closure() -> torch.Tensor:
            lbfgs_optimizer.zero_grad()
            _, pred_trajectories = model(
                train_data.initial_conditions,
                train_data.times,
            )
            loss = mse_loss(pred_trajectories, train_data.trajectories)
            loss.backward()
            return loss

        lbfgs_optimizer.step(closure)
        train_loss = _evaluate_loss(model, train_data)
        test_loss = _evaluate_loss(model, test_data)
        record_epoch(epoch, "lbfgs", train_loss, test_loss)

        if log_every > 0 and (
            step % log_every == 0 or step == 1 or step == lbfgs_max_iter
        ):
            logging.info(
                "%s LBFGS step %04d train_loss: %.6e test_loss: %.6e",
                run_name,
                step,
                train_loss,
                test_loss,
            )

    best_train_entry = min(losses, key=lambda entry: float(entry["train_loss"]))
    best_test_entry = min(losses, key=lambda entry: float(entry["test_loss"]))
    if restore_best_checkpoint and best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return {
        "losses": losses,
        "num_epochs_completed": losses[-1]["epoch"],
        "final_epoch_train_loss": losses[-1]["train_loss"],
        "final_epoch_test_loss": losses[-1]["test_loss"],
        "final_train_loss": best_checkpoint_train_loss,
        "final_test_loss": best_checkpoint_test_loss,
        "best_train_loss": best_train_entry["train_loss"],
        "best_test_loss": best_test_entry["test_loss"],
        "epoch_of_best_train_loss": best_train_entry["epoch"],
        "epoch_of_best_test_loss": best_test_entry["epoch"],
        "best_epoch": best_checkpoint_epoch,
        "best_checkpoint_train_loss": best_checkpoint_train_loss,
        "best_checkpoint_test_loss": best_checkpoint_test_loss,
        "restore_best_checkpoint": restore_best_checkpoint,
        "reported_metrics_source": "best_checkpoint" if restore_best_checkpoint else "final_epoch",
        "stopped_because": "optimizer_phases_completed",
        "stopping_monitor": stopping_monitor,
        "stopping_criterion": "adam_then_lbfgs_phases",
        "stopping_min_delta": None,
        "stopping_patience": None,
        "target_loss": None,
        "adam_epochs": adam_epochs,
        "lbfgs_steps": lbfgs_max_iter,
    }


def _save_artifacts(
    output_dir: Path,
    setup: dict[str, Any],
    metrics: dict[str, Any],
    model: torch.nn.Module,
    test_data: Any,
    pred_coefficients: torch.Tensor,
    pred_trajectories: torch.Tensor,
    training_history: dict[str, Any],
    run_name: str,
    plotted_trajectory_index: int = 0,
    plot_num_periods: float = 10.0,
    plot_points_per_period: int = 500,
    prediction_omega: float | None = None,
) -> dict[str, torch.Tensor]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_num_time_points = int(plot_num_periods * plot_points_per_period)
    setup["plotted_trajectory_index"] = plotted_trajectory_index
    setup["plot_num_periods"] = plot_num_periods
    setup["plot_points_per_period"] = plot_points_per_period
    setup["plot_num_time_points"] = plot_num_time_points
    setup["num_epochs_completed"] = training_history["num_epochs_completed"]
    setup["stopped_because"] = training_history["stopped_because"]
    setup["restore_best_checkpoint"] = training_history["restore_best_checkpoint"]
    setup["failed_due_to_nan_or_inf"] = training_history.get(
        "failed_due_to_nan_or_inf",
        False,
    )
    metrics["final_epoch_train_loss"] = training_history["final_epoch_train_loss"]
    metrics["final_epoch_test_loss"] = training_history["final_epoch_test_loss"]
    metrics["best_epoch"] = training_history["best_epoch"]
    metrics["best_train_loss"] = training_history["best_train_loss"]
    metrics["best_test_loss"] = training_history["best_test_loss"]
    metrics["epoch_of_best_train_loss"] = training_history["epoch_of_best_train_loss"]
    metrics["epoch_of_best_test_loss"] = training_history["epoch_of_best_test_loss"]
    metrics["best_checkpoint_train_loss"] = training_history[
        "best_checkpoint_train_loss"
    ]
    metrics["best_checkpoint_test_loss"] = training_history["best_checkpoint_test_loss"]
    metrics["reported_metrics_source"] = training_history["reported_metrics_source"]
    metrics["failed_due_to_nan_or_inf"] = training_history.get(
        "failed_due_to_nan_or_inf",
        False,
    )
    metrics["raw_omega_in_optimizer"] = training_history.get(
        "raw_omega_in_optimizer",
    )
    metrics["model_train_eval_modes_explicit"] = training_history.get(
        "model_train_eval_modes_explicit",
    )
    metrics["plotted_trajectory_index"] = plotted_trajectory_index
    metrics["plot_num_periods"] = plot_num_periods
    metrics["plot_points_per_period"] = plot_points_per_period
    metrics["plot_num_time_points"] = plot_num_time_points

    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "training_history.json", training_history)
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    optimizer_analysis = _optimizer_analysis_text(
        run_name,
        setup,
        metrics,
        training_history,
    )
    (output_dir / "optimizer_analysis.txt").write_text(
        optimizer_analysis,
        encoding="utf-8",
    )

    plotted_num_trajectories = 1
    plot_times = torch.linspace(
        0.0,
        plot_num_periods * test_data.period,
        plot_num_time_points,
        device=test_data.times.device,
    )
    prediction_omega = test_data.omega if prediction_omega is None else prediction_omega
    target_plot_coefficients = test_data.coefficients[plotted_trajectory_index].unsqueeze(0)
    pred_plot_coefficients = pred_coefficients[plotted_trajectory_index].unsqueeze(0)
    target_plot_positions = decode_sho(
        target_plot_coefficients,
        plot_times,
        test_data.omega,
    )[0]
    pred_plot_positions = decode_sho(
        pred_plot_coefficients,
        plot_times,
        prediction_omega,
    )[0]
    target_plot_velocities = decode_sho_velocity(
        target_plot_coefficients,
        plot_times,
        test_data.omega,
    )[0]
    pred_plot_velocities = decode_sho_velocity(
        pred_plot_coefficients,
        plot_times,
        prediction_omega,
    )[0]

    logging.info(
        "Plotting %s from %s model_type=%s encoder_type=%s",
        run_name,
        output_dir,
        setup.get("model_type"),
        setup.get("encoder_type"),
    )
    logging.info("  plotted_num_trajectories: %d", plotted_num_trajectories)
    logging.info("  plotted_trajectory_index: %d", plotted_trajectory_index)
    logging.info("  plot_num_periods: %s", plot_num_periods)
    logging.info("  plot_points_per_period: %d", plot_points_per_period)
    logging.info("  plot_num_time_points: %d", plot_num_time_points)

    stage_label = (
        "SHO Stage 1"
        if str(setup.get("comparison_mode", "")).startswith("stage1")
        else "SHO Stage 0"
    )
    plot_trajectory_comparison(
        plot_times,
        target_plot_positions,
        pred_plot_positions,
        output_dir / "trajectory_comparison.png",
        title=f"{stage_label}: {run_name}\nTrajectory",
    )
    plot_phase_space_comparison(
        target_plot_positions,
        target_plot_velocities,
        pred_plot_positions,
        pred_plot_velocities,
        output_dir / "phase_space_comparison.png",
        title=f"{stage_label}: {run_name}\nPhase Space",
    )
    plot_training_loss(
        training_history["losses"],
        output_dir / "training_loss.png",
        title=f"{stage_label}: {run_name}\nTraining Loss",
    )
    images = [
        ("Trajectory Comparison", "trajectory_comparison.png"),
        ("Phase-Space Comparison", "phase_space_comparison.png"),
        ("Training Loss", "training_loss.png"),
    ]
    if training_history["losses"] and "omega_hat" in training_history["losses"][-1]:
        plot_omega_history(
            training_history["losses"],
            test_data.omega,
            output_dir / "omega_vs_epoch.png",
            title=f"{stage_label}: {run_name}\nOmega vs Epoch",
        )
        images.append(("Omega vs Epoch", "omega_vs_epoch.png"))
        plot_omega_gradient_history(
            training_history["losses"],
            output_dir / "omega_gradient_vs_epoch.png",
            title=f"{stage_label}: {run_name}\nOmega Gradient vs Epoch",
        )
        images.append(("Omega Gradient vs Epoch", "omega_gradient_vs_epoch.png"))
        plot_test_loss_and_omega_history(
            training_history["losses"],
            test_data.omega,
            output_dir / "test_loss_and_omega_vs_epoch.png",
            title=f"{stage_label}: {run_name}\nTest Loss and Omega vs Epoch",
        )
        images.append(
            (
                "Test Loss and Omega vs Epoch",
                "test_loss_and_omega_vs_epoch.png",
            )
        )
    if setup.get("make_final_20_percent_training_plot"):
        losses = training_history["losses"]
        final_slice_start = int(len(losses) * 0.8)
        final_slice = losses[final_slice_start:] or losses
        plot_training_loss(
            final_slice,
            output_dir / "training_loss_final_20_percent.png",
            title=f"{stage_label}: {run_name}\nTraining Loss Final 20%",
        )
        images.append(
            (
                "Training Loss Final 20%",
                "training_loss_final_20_percent.png",
            )
        )
    _write_run_index(
        output_dir=output_dir,
        run_name=f"{stage_label}: {run_name}",
        setup=setup,
        metrics=metrics,
        training_summary={
            "num_epochs_completed": training_history["num_epochs_completed"],
            "final_epoch_train_loss": training_history["final_epoch_train_loss"],
            "final_epoch_test_loss": training_history["final_epoch_test_loss"],
            "final_train_loss": training_history["final_train_loss"],
            "final_test_loss": training_history["final_test_loss"],
            "best_train_loss": training_history["best_train_loss"],
            "best_test_loss": training_history["best_test_loss"],
            "epoch_of_best_train_loss": training_history["epoch_of_best_train_loss"],
            "epoch_of_best_test_loss": training_history["epoch_of_best_test_loss"],
            "best_epoch": training_history["best_epoch"],
            "reported_metrics_source": training_history["reported_metrics_source"],
        },
        optimizer_analysis=optimizer_analysis,
        images=images,
    )

    return {
        "target_positions": target_plot_positions,
        "target_velocities": target_plot_velocities,
        "prediction_positions": pred_plot_positions,
        "prediction_velocities": pred_plot_velocities,
    }


def _base_setup(
    *,
    comparison_mode: str,
    seed: int,
    omega: float,
    num_train: int,
    num_test: int,
    num_time_steps: int,
    num_periods: float,
    max_epochs: int,
    learning_rate: Any,
    batch_size: int,
    model_type: str,
    encoder_type: str,
    activation: str | None,
    hidden_dim: Any,
    stopping_criterion: str,
    stopping_monitor: str,
    stopping_min_delta: float,
    stopping_patience: int,
    target_loss: float | None,
    restore_best_checkpoint: bool,
) -> dict[str, Any]:
    return {
        "comparison_mode": comparison_mode,
        "model_type": model_type,
        "encoder_type": encoder_type,
        "activation": activation,
        "hidden_dim": hidden_dim,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "num_epochs_completed": None,
        "stopping_criterion": stopping_criterion,
        "stopping_monitor": stopping_monitor,
        "stopping_min_delta": stopping_min_delta,
        "stopping_patience": stopping_patience,
        "target_loss": target_loss,
        "restore_best_checkpoint": restore_best_checkpoint,
        "stopped_because": None,
        "num_train_trajectories": num_train,
        "num_test_trajectories": num_test,
        "num_time_points_per_trajectory": num_time_steps,
        "omega": omega,
        "sho_period": math.tau / omega,
        "num_periods_per_trajectory": num_periods,
        "random_seed": seed,
    }


def _add_linear_weight_metrics(
    metrics: dict[str, Any],
    learned_weight_matrix: torch.Tensor,
    omega: float,
) -> None:
    target_weight_matrix = _target_linear_weight_matrix(omega)
    weight_error = learned_weight_matrix.detach().cpu() - target_weight_matrix
    metrics.update(
        {
            "learned_weight_matrix": learned_weight_matrix.detach().cpu().tolist(),
            "target_weight_matrix": target_weight_matrix.tolist(),
            "weight_error": weight_error.tolist(),
            "weight_mse": torch.mean(weight_error**2).item(),
            "linear_weight_matrix": learned_weight_matrix.detach().cpu().tolist(),
            "target_linear_weight_matrix": target_weight_matrix.tolist(),
            "linear_weight_error": weight_error.tolist(),
            "linear_weight_mse": torch.mean(weight_error**2).item(),
        }
    )


def _add_residual_metrics(
    metrics: dict[str, Any],
    model: torch.nn.Module,
    train_data: Any,
    test_data: Any,
) -> None:
    if not hasattr(model, "residual"):
        return
    with torch.no_grad():
        train_residual = model.residual(train_data.initial_conditions)
        test_residual = model.residual(test_data.initial_conditions)
    metrics.update(
        {
            "residual_norm": torch.sqrt(torch.mean(test_residual**2)).item(),
            "residual_norm_train": torch.linalg.vector_norm(train_residual).item(),
            "residual_norm_test": torch.linalg.vector_norm(test_residual).item(),
            "residual_rms_train": torch.sqrt(torch.mean(train_residual**2)).item(),
            "residual_rms_test": torch.sqrt(torch.mean(test_residual**2)).item(),
            "train_residual_mse": torch.mean(train_residual**2).item(),
            "test_residual_mse": torch.mean(test_residual**2).item(),
            "train_residual_rms": torch.sqrt(torch.mean(train_residual**2)).item(),
            "test_residual_rms": torch.sqrt(torch.mean(test_residual**2)).item(),
            "max_abs_test_residual": test_residual.abs().max().item(),
        }
    )


def _zero_initialize_residual_output(model: torch.nn.Module) -> None:
    if not hasattr(model, "residual_encoder"):
        raise ValueError("model does not have a residual_encoder")
    residual_encoder = model.residual_encoder
    for module in reversed(residual_encoder):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            return
    raise ValueError("residual_encoder does not contain a Linear output layer")


def _residual_rms(model: torch.nn.Module, initial_conditions: torch.Tensor) -> float:
    if not hasattr(model, "residual"):
        return float("nan")
    with torch.no_grad():
        residual = model.residual(initial_conditions)
        return torch.sqrt(torch.mean(residual**2)).item()


def _linear_weight_mse(model: torch.nn.Module, omega: float) -> float:
    if not hasattr(model, "linear_encoder"):
        return float("nan")
    learned_weight_matrix = model.linear_encoder.weight.detach().cpu()
    target_weight_matrix = _target_linear_weight_matrix(omega)
    return torch.mean((learned_weight_matrix - target_weight_matrix) ** 2).item()


def _add_lbfgs_phase_metrics(
    metrics: dict[str, Any],
    training_history: dict[str, Any],
) -> None:
    losses = training_history["losses"]
    adam_losses = [
        float(entry["test_loss"])
        for entry in losses
        if entry.get("phase") == "adam"
    ]
    lbfgs_losses = [
        float(entry["test_loss"])
        for entry in losses
        if entry.get("phase") == "lbfgs"
    ]
    if not adam_losses or not lbfgs_losses:
        return

    best_before = min(adam_losses)
    best_after = min(lbfgs_losses)
    improvement_factor = best_before / best_after if best_after > 0.0 else float("inf")
    metrics.update(
        {
            "adam_epochs_completed": len(adam_losses),
            "lbfgs_iterations_completed": len(lbfgs_losses),
            "best_metric_before_lbfgs": best_before,
            "best_metric_after_lbfgs": best_after,
            "lbfgs_improvement_factor": improvement_factor,
        }
    )


def _lbfgs_phase_metrics(training_history: dict[str, Any]) -> dict[str, Any]:
    losses = training_history["losses"]
    adam_losses = [
        float(entry["test_loss"])
        for entry in losses
        if entry.get("phase") == "adam"
    ]
    lbfgs_losses = [
        float(entry["test_loss"])
        for entry in losses
        if entry.get("phase") == "lbfgs"
    ]
    if not adam_losses or not lbfgs_losses:
        return {}

    best_before = min(adam_losses)
    best_after = min(lbfgs_losses)
    improvement_factor = best_before / best_after if best_after > 0.0 else float("inf")
    return {
        "adam_epochs_completed": len(adam_losses),
        "lbfgs_iterations_completed": len(lbfgs_losses),
        "best_metric_before_lbfgs": best_before,
        "best_metric_after_lbfgs": best_after,
        "lbfgs_improvement_factor": improvement_factor,
    }


def _evaluate_and_save_learned_model(
    *,
    model: torch.nn.Module,
    train_data: Any,
    test_data: Any,
    setup: dict[str, Any],
    training_history: dict[str, Any],
    output_dir: Path,
    run_name: str,
    omega: float,
    extra_metrics: dict[str, Any] | None = None,
    prediction_omega: float | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    prediction_omega = omega if prediction_omega is None else prediction_omega
    with torch.no_grad():
        pred_coefficients, pred_trajectories = model(
            test_data.initial_conditions,
            test_data.times,
        )
        target_velocities = decode_sho_velocity(
            test_data.coefficients,
            test_data.times,
            omega,
        )
        pred_velocities = decode_sho_velocity(
            pred_coefficients,
            test_data.times,
            prediction_omega,
        )

    metrics: dict[str, Any] = {
        "coefficient_mse": mean_squared_error(
            pred_coefficients,
            test_data.coefficients,
        ),
        "trajectory_mse": mean_squared_error(
            pred_trajectories,
            test_data.trajectories,
        ),
        "mean_abs_predicted_a_test": pred_coefficients[:, 0].abs().mean().item(),
        "mean_abs_predicted_b_test": pred_coefficients[:, 1].abs().mean().item(),
        "mean_abs_true_a_test": test_data.coefficients[:, 0].abs().mean().item(),
        "mean_abs_true_b_test": test_data.coefficients[:, 1].abs().mean().item(),
    }
    metrics.update(_energy_diagnostics(pred_trajectories, pred_velocities, omega))
    metrics.update(
        _error_diagnostics(
            test_data.trajectories,
            target_velocities,
            pred_trajectories,
            pred_velocities,
        )
    )
    metrics["plotted_num_trajectories"] = 1
    metrics["plotted_trajectory_index"] = 0

    if isinstance(model, SHOLinearExactSolutionModel):
        _add_linear_weight_metrics(metrics, model.encoder.weight, omega)
    elif isinstance(model, SHOTrainableLinearResidualModel):
        _add_linear_weight_metrics(metrics, model.linear_encoder.weight, omega)
    elif isinstance(model, SHOFixedLinearResidualModel):
        metrics["target_linear_weight_matrix"] = _target_linear_weight_matrix(
            omega
        ).tolist()

    _add_residual_metrics(metrics, model, train_data, test_data)
    _add_lbfgs_phase_metrics(metrics, training_history)
    if "residual_rms_test" in metrics:
        metrics["residual_rms_final"] = metrics["residual_rms_test"]
    if "linear_weight_mse" in metrics:
        metrics["linear_weight_mse_final"] = metrics["linear_weight_mse"]
    if extra_metrics is not None:
        metrics.update(extra_metrics)

    plot_data = _save_artifacts(
        output_dir,
        setup,
        metrics,
        model,
        test_data,
        pred_coefficients,
        pred_trajectories,
        training_history,
        run_name,
        prediction_omega=prediction_omega,
    )
    return metrics, plot_data


def run_mlp_experiment(
    activation: str,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    seed: int = 0,
    omega: float = 2.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int = 512,
    hidden_dim: int = 32,
    hidden_dims: list[int] | tuple[int, ...] | None = None,
    log_every: int = 500,
    stopping_criterion: str | None = None,
    stopping_monitor: str = "test_loss",
    stopping_min_delta: float = 1e-13,
    stopping_patience: int = 10000,
    target_loss: float | None = None,
    restore_best_checkpoint: bool = True,
    residual_zero_init: bool = False,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run learned MLP Stage 0 with one activation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_epochs is None:
        max_epochs = _default_max_epochs(comparison_mode)
    if learning_rate is None:
        learning_rate = SHARED_BUDGET_LEARNING_RATE
    if stopping_criterion is None:
        stopping_criterion = _default_stopping_criterion(comparison_mode)

    set_seed(seed)
    output_dir = (
        Path(output_dir)
        if output_dir
        else _run_dir(f"mlp_{activation}", comparison_mode=comparison_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed + 1,
    )

    model = SHOExactSolutionModel(
        omega=omega,
        hidden_dim=hidden_dim,
        activation=activation,
        hidden_dims=hidden_dims,
    )
    recorded_hidden_dims = list(hidden_dims) if hidden_dims is not None else [hidden_dim, hidden_dim]
    setup = _base_setup(
        comparison_mode=comparison_mode,
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type="mlp",
        activation=activation,
        hidden_dim=recorded_hidden_dims,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup["log_every"] = log_every

    run_name = f"MLP + {activation.upper() if activation == 'silu' else 'Tanh'}"
    logging.info("Stage 0 learned SHO setup (%s):", activation)
    for name, value in setup.items():
        logging.info("  %s: %s", name, value)

    training_history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        max_epochs,
        batch_size,
        learning_rate,
        log_every,
        run_name,
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        target_loss,
        output_dir / "best_model.pt",
        restore_best_checkpoint,
    )

    with torch.no_grad():
        pred_coefficients, pred_trajectories = model(
            test_data.initial_conditions,
            test_data.times,
        )
        target_velocities = decode_sho_velocity(
            test_data.coefficients,
            test_data.times,
            omega,
        )
        pred_velocities = decode_sho_velocity(
            pred_coefficients,
            test_data.times,
            omega,
        )

    metrics: dict[str, Any] = {
        "coefficient_mse": mean_squared_error(
            pred_coefficients,
            test_data.coefficients,
        ),
        "trajectory_mse": mean_squared_error(
            pred_trajectories,
            test_data.trajectories,
        ),
    }
    metrics.update(_energy_diagnostics(pred_trajectories, pred_velocities, omega))
    metrics.update(
        _error_diagnostics(
            test_data.trajectories,
            target_velocities,
            pred_trajectories,
            pred_velocities,
        )
    )
    metrics["plotted_num_trajectories"] = 1
    metrics["plotted_trajectory_index"] = 0

    plot_data = _save_artifacts(
        output_dir,
        setup,
        metrics,
        model,
        test_data,
        pred_coefficients,
        pred_trajectories,
        training_history,
        run_name,
    )
    return metrics, plot_data


def _running_best_slope(
    losses: list[dict[str, float | int]],
    *,
    fraction: float = 0.1,
) -> dict[str, float]:
    if not losses:
        return {
            "running_best_test_loss_final_10_percent_slope": float("nan"),
            "running_best_test_loss_final_10_percent_log10_slope": float("nan"),
        }

    start = int(len(losses) * (1.0 - fraction))
    window = losses[start:] or losses
    x_values = [float(entry["epoch"]) for entry in window]
    y_values = [float(entry["best_test_loss_so_far"]) for entry in window]
    n = len(x_values)
    if n < 2:
        return {
            "running_best_test_loss_final_10_percent_slope": 0.0,
            "running_best_test_loss_final_10_percent_log10_slope": 0.0,
        }

    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n
    denom = sum((x - mean_x) ** 2 for x in x_values)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
        / denom
        if denom > 0.0
        else 0.0
    )

    floor = torch.finfo(torch.float32).tiny
    log_y_values = [math.log10(max(y, floor)) for y in y_values]
    mean_log_y = sum(log_y_values) / n
    log_slope = (
        sum((x - mean_x) * (y - mean_log_y) for x, y in zip(x_values, log_y_values))
        / denom
        if denom > 0.0
        else 0.0
    )
    return {
        "running_best_test_loss_final_10_percent_slope": slope,
        "running_best_test_loss_final_10_percent_log10_slope": log_slope,
    }


def run_pure_mlp_silu_long_continuation(
    *,
    source_dir: str | Path = "outputs/sho/stage0/lr_sweep_extended/2026-06-05_pure_mlp_silu_lr_3em04",
    output_dir: str | Path = "outputs/sho/stage0/best_achieved_long/2026-06-06_pure_mlp_silu_1m_continuation",
    max_epochs: int = 1_000_000,
    log_every: int = 500,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Continue the best pure MLP SiLU Stage 0 checkpoint to a longer budget."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_setup = json.loads((source_dir / "setup.json").read_text(encoding="utf-8"))
    source_metrics = json.loads(
        (source_dir / "metrics.json").read_text(encoding="utf-8")
    )
    source_history = json.loads(
        (source_dir / "training_history.json").read_text(encoding="utf-8")
    )

    seed = int(source_setup["random_seed"])
    omega = float(source_setup["omega"])
    num_train = int(source_setup["num_train_trajectories"])
    num_test = int(source_setup["num_test_trajectories"])
    num_time_steps = int(source_setup["num_time_points_per_trajectory"])
    num_periods = float(source_setup["num_periods_per_trajectory"])
    batch_size = int(source_setup["batch_size"])
    learning_rate = float(source_setup["learning_rate"])
    stopping_criterion = str(source_setup["stopping_criterion"])
    stopping_monitor = str(source_setup["stopping_monitor"])
    stopping_min_delta = float(source_setup["stopping_min_delta"])
    stopping_patience = int(source_setup["stopping_patience"])
    target_loss = source_setup.get("target_loss")
    hidden_dims = list(source_setup.get("hidden_dim") or [32, 32])

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed + 1,
    )
    model = SHOExactSolutionModel(
        omega=omega,
        activation="silu",
        hidden_dims=hidden_dims,
    )
    model.load_state_dict(torch.load(source_dir / "best_model.pt", weights_only=True))

    setup = _base_setup(
        comparison_mode="best_achieved_long",
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type="mlp",
        activation="silu",
        hidden_dim=hidden_dims,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=True,
    )
    setup.update(
        {
            "log_every": log_every,
            "continuation_source_dir": str(source_dir),
            "continuation_source_best_epoch": source_metrics.get("best_epoch"),
            "continuation_source_best_test_loss": source_metrics.get("best_test_loss"),
            "continuation_source_epochs_completed": source_metrics.get(
                "num_epochs_completed",
            ),
            "optimizer_state_continued": False,
            "make_final_20_percent_training_plot": True,
        }
    )

    source_losses = source_history["losses"]
    start_epoch = int(source_history["num_epochs_completed"])
    source_best_test_loss = float(source_history["best_checkpoint_test_loss"])
    source_best_train_loss = float(source_history["best_checkpoint_train_loss"])
    source_best_checkpoint_epoch = int(source_history["best_epoch"])
    source_monitored_loss = (
        float(source_history["best_checkpoint_train_loss"])
        if stopping_monitor == "train_loss"
        else float(source_history["best_checkpoint_test_loss"])
    )
    run_name = "Pure MLP SiLU 1M Continuation"
    training_history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        max_epochs,
        batch_size,
        learning_rate,
        log_every,
        run_name,
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        target_loss,
        output_dir / "best_model.pt",
        True,
        start_epoch=start_epoch,
        initial_losses=source_losses,
        initial_best_checkpoint_epoch=source_best_checkpoint_epoch,
        initial_best_checkpoint_train_loss=source_best_train_loss,
        initial_best_checkpoint_test_loss=source_best_test_loss,
        initial_best_significant_monitored_loss=source_monitored_loss,
    )
    slope_metrics = _running_best_slope(training_history["losses"], fraction=0.1)
    previous_best = float(source_metrics["best_test_loss"])
    current_best = float(training_history["best_test_loss"])
    extra_metrics = {
        "previous_best_pure_mlp_silu_test_loss": previous_best,
        "improvement_factor_relative_to_previous_best_pure_mlp_silu": (
            previous_best / current_best if current_best > 0.0 else float("inf")
        ),
    }
    extra_metrics.update(slope_metrics)

    return _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=training_history,
        output_dir=output_dir,
        run_name=run_name,
        omega=omega,
        extra_metrics=extra_metrics,
    )


def run_linear_encoder_experiment(
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    seed: int = 0,
    omega: float = 2.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int = 512,
    log_every: int = 500,
    stopping_criterion: str | None = None,
    stopping_monitor: str = "test_loss",
    stopping_min_delta: float = 1e-13,
    stopping_patience: int = 10000,
    target_loss: float | None = None,
    restore_best_checkpoint: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run learned linear-encoder Stage 0."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_epochs is None:
        max_epochs = _default_max_epochs(comparison_mode)
    if learning_rate is None:
        learning_rate = (
            SHARED_BUDGET_LEARNING_RATE
            if comparison_mode == "shared_budget"
            else 1e-2
        )
    if stopping_criterion is None:
        stopping_criterion = _default_stopping_criterion(comparison_mode)
    if target_loss is None and comparison_mode == "best_achieved":
        target_loss = 1e-16

    set_seed(seed)
    output_dir = (
        Path(output_dir)
        if output_dir
        else _run_dir("linear_encoder", comparison_mode=comparison_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed + 1,
    )

    model = SHOLinearExactSolutionModel(omega=omega)
    setup = _base_setup(
        comparison_mode=comparison_mode,
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type="linear",
        activation=None,
        hidden_dim=None,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup["log_every"] = log_every

    logging.info("Stage 0 linear encoder SHO setup:")
    for name, value in setup.items():
        logging.info("  %s: %s", name, value)

    run_name = "Linear Encoder"
    training_history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        max_epochs,
        batch_size,
        learning_rate,
        log_every,
        run_name,
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        target_loss,
        output_dir / "best_model.pt",
        restore_best_checkpoint,
    )

    with torch.no_grad():
        pred_coefficients, pred_trajectories = model(
            test_data.initial_conditions,
            test_data.times,
        )
        target_velocities = decode_sho_velocity(
            test_data.coefficients,
            test_data.times,
            omega,
        )
        pred_velocities = decode_sho_velocity(
            pred_coefficients,
            test_data.times,
            omega,
        )

    learned_weight_matrix = model.encoder.weight.detach().clone()
    target_weight_matrix = _target_linear_weight_matrix(omega)
    weight_error = learned_weight_matrix - target_weight_matrix

    metrics: dict[str, Any] = {
        "learned_weight_matrix": learned_weight_matrix.tolist(),
        "target_weight_matrix": target_weight_matrix.tolist(),
        "weight_error": weight_error.tolist(),
        "weight_mse": torch.mean(weight_error**2).item(),
        "coefficient_mse": mean_squared_error(
            pred_coefficients,
            test_data.coefficients,
        ),
        "trajectory_mse": mean_squared_error(
            pred_trajectories,
            test_data.trajectories,
        ),
    }
    metrics.update(
        _error_diagnostics(
            test_data.trajectories,
            target_velocities,
            pred_trajectories,
            pred_velocities,
        )
    )
    metrics["plotted_num_trajectories"] = 1
    metrics["plotted_trajectory_index"] = 0

    _save_artifacts(
        output_dir,
        setup,
        metrics,
        model,
        test_data,
        pred_coefficients,
        pred_trajectories,
        training_history,
        run_name,
    )
    return metrics


def run_stage1_linear_encoder_global_omega(
    seed: int = 0,
    true_omega: float = 2.0,
    initial_omega_hat: float = 1.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    max_epochs: int = BEST_ACHIEVED_MAX_EPOCHS,
    learning_rate: float = 1e-2,
    batch_size: int = 512,
    log_every: int = 500,
    stopping_criterion: str = "patience",
    stopping_monitor: str = "test_loss",
    stopping_min_delta: float = 1e-13,
    stopping_patience: int = 10000,
    target_loss: float | None = None,
    restore_best_checkpoint: bool = True,
    output_dir: str | Path = "outputs/sho/stage1/unknown_global_omega/2026-06-06_linear_encoder_omega_init_1",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run Stage 1 with a linear coefficient encoder and learned global omega."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=true_omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=true_omega,
        num_periods=num_periods,
        seed=seed + 1,
    )

    model = SHOLearnedGlobalOmegaSolutionModel(
        initial_omega_hat=initial_omega_hat,
    )
    omega_hat_initial = float(model.omega_hat.detach().cpu())
    raw_omega_initial = float(model.raw_omega.detach().cpu())
    omega_initialization_verified = abs(omega_hat_initial - initial_omega_hat) < 1e-6
    omega_hat_requires_grad = bool(model.omega_hat.requires_grad)
    model.zero_grad()
    _, graph_probe_trajectories = model(
        train_data.initial_conditions,
        train_data.times,
    )
    graph_probe_loss = graph_probe_trajectories.square().mean()
    graph_probe_loss.backward()
    graph_probe_raw_omega_grad = (
        None
        if model.raw_omega.grad is None
        else float(model.raw_omega.grad.detach().cpu())
    )
    omega_hat_used_in_decoder_graph_verified = (
        graph_probe_raw_omega_grad is not None
        and math.isfinite(graph_probe_raw_omega_grad)
        and abs(graph_probe_raw_omega_grad) > 0.0
    )
    model.zero_grad()

    setup = _base_setup(
        comparison_mode="stage1_unknown_global_omega",
        seed=seed,
        omega=true_omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_learned_global_omega_solution",
        encoder_type="linear_encoder_global_omega",
        activation=None,
        hidden_dim=None,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup.update(
        {
            "run_name": "stage1_linear_encoder_global_omega",
            "true_omega_used_for_model": False,
            "true_omega_used_for_dataset_generation": True,
            "true_omega_used_for_evaluation_metrics": True,
            "initial_omega_hat_requested": initial_omega_hat,
            "omega_hat_initial": omega_hat_initial,
            "raw_omega_initial": raw_omega_initial,
            "omega_initialization_verified": omega_initialization_verified,
            "omega_hat_requires_grad": omega_hat_requires_grad,
            "omega_hat_used_in_decoder_graph_verified": (
                omega_hat_used_in_decoder_graph_verified
            ),
            "graph_probe_abs_dL_d_raw_omega": (
                None
                if graph_probe_raw_omega_grad is None
                else abs(graph_probe_raw_omega_grad)
            ),
            "omega_parameterization": "softplus(raw_omega) + 1e-8",
            "log_every": log_every,
        }
    )

    run_name = "Stage 1 Linear Encoder + Global Omega"
    training_history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        max_epochs,
        batch_size,
        learning_rate,
        log_every,
        run_name,
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        target_loss,
        output_dir / "best_model.pt",
        restore_best_checkpoint,
    )

    omega_hat = float(model.omega_hat.detach().cpu())
    omega_error = omega_hat - true_omega
    relative_omega_error = omega_error / true_omega
    extra_metrics = {
        "omega_hat": omega_hat,
        "true_omega": true_omega,
        "omega_error": omega_error,
        "relative_omega_error": relative_omega_error,
        "omega_hat_initial": omega_hat_initial,
        "raw_omega_initial": raw_omega_initial,
        "omega_initialization_verified": omega_initialization_verified,
        "omega_hat_requires_grad": omega_hat_requires_grad,
        "omega_hat_used_in_decoder_graph_verified": (
            omega_hat_used_in_decoder_graph_verified
        ),
        "graph_probe_abs_dL_d_raw_omega": (
            None
            if graph_probe_raw_omega_grad is None
            else abs(graph_probe_raw_omega_grad)
        ),
        "plots_generated_from_restored_best_checkpoint": restore_best_checkpoint,
    }
    _add_linear_weight_metrics(extra_metrics, model.encoder.weight, true_omega)

    return _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=training_history,
        output_dir=output_dir,
        run_name=run_name,
        omega=true_omega,
        extra_metrics=extra_metrics,
        prediction_omega=omega_hat,
    )


def run_mlp_adam_lbfgs_experiment(
    activation: str,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    seed: int = 0,
    omega: float = 2.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    adam_epochs: int = 100000,
    adam_learning_rate: float = 1e-3,
    lbfgs_max_iter: int = 500,
    lbfgs_learning_rate: float = 1.0,
    batch_size: int = 512,
    hidden_dim: int = 32,
    log_every: int = 500,
    stopping_monitor: str = "test_loss",
    restore_best_checkpoint: bool = True,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run pure MLP Stage 0 with Adam followed by LBFGS polish."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(seed)
    output_dir = (
        Path(output_dir)
        if output_dir
        else _run_dir(f"mlp_{activation}_adam_lbfgs", comparison_mode=comparison_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed + 1,
    )

    model = SHOExactSolutionModel(
        omega=omega,
        hidden_dim=hidden_dim,
        activation=activation,
    )
    setup = _base_setup(
        comparison_mode=comparison_mode,
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=adam_epochs + lbfgs_max_iter,
        learning_rate={
            "adam": adam_learning_rate,
            "lbfgs": lbfgs_learning_rate,
        },
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type="mlp_adam_lbfgs",
        activation=activation,
        hidden_dim=hidden_dim,
        stopping_criterion="adam_then_lbfgs_phases",
        stopping_monitor=stopping_monitor,
        stopping_min_delta=0.0,
        stopping_patience=0,
        target_loss=None,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup.update(
        {
            "optimizer": "Adam -> LBFGS",
            "optimizer_settings": {
                "adam_epochs": adam_epochs,
                "adam_learning_rate": adam_learning_rate,
                "lbfgs_max_iter": lbfgs_max_iter,
                "lbfgs_learning_rate": lbfgs_learning_rate,
                "lbfgs_line_search_fn": "strong_wolfe",
                "lbfgs_outer_steps_recorded_as_epochs": True,
            },
            "log_every": log_every,
        }
    )

    run_name = f"MLP + {activation.upper() if activation == 'silu' else 'Tanh'} Adam -> LBFGS"
    logging.info("Stage 0 MLP Adam -> LBFGS SHO setup (%s):", activation)
    for name, value in setup.items():
        logging.info("  %s: %s", name, value)

    training_history = _train_model_adam_then_lbfgs(
        model=model,
        train_data=train_data,
        test_data=test_data,
        num_train=num_train,
        adam_epochs=adam_epochs,
        adam_learning_rate=adam_learning_rate,
        lbfgs_max_iter=lbfgs_max_iter,
        lbfgs_learning_rate=lbfgs_learning_rate,
        batch_size=batch_size,
        log_every=log_every,
        run_name=run_name,
        stopping_monitor=stopping_monitor,
        best_model_path=output_dir / "best_model.pt",
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup.update(_lbfgs_phase_metrics(training_history))

    return _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=training_history,
        output_dir=output_dir,
        run_name=run_name,
        omega=omega,
    )


def run_residual_encoder_experiment(
    residual_type: str,
    activation: str,
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    seed: int = 0,
    omega: float = 2.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int = 512,
    hidden_dim: int = 32,
    hidden_dims: list[int] | tuple[int, ...] | None = None,
    log_every: int = 500,
    stopping_criterion: str | None = None,
    stopping_monitor: str = "test_loss",
    stopping_min_delta: float = 1e-13,
    stopping_patience: int = 10000,
    target_loss: float | None = None,
    restore_best_checkpoint: bool = True,
    residual_zero_init: bool = False,
    residual_scale: float = 1.0,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Run a linear-path-plus-residual Stage 0 variant."""
    if residual_type not in {"trainable_linear_residual", "fixed_linear_residual"}:
        raise ValueError(f"unsupported residual_type: {residual_type}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if max_epochs is None:
        max_epochs = _default_max_epochs(comparison_mode)
    if learning_rate is None:
        learning_rate = SHARED_BUDGET_LEARNING_RATE
    if stopping_criterion is None:
        stopping_criterion = _default_stopping_criterion(comparison_mode)

    set_seed(seed)
    output_dir = (
        Path(output_dir)
        if output_dir
        else _run_dir(f"{residual_type}_{activation}", comparison_mode=comparison_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = generate_sho_data(
        num_train,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )
    test_data = generate_sho_data(
        num_test,
        num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed + 1,
    )

    model_class = (
        SHOTrainableLinearResidualModel
        if residual_type == "trainable_linear_residual"
        else SHOFixedLinearResidualModel
    )
    model = model_class(
        omega=omega,
        hidden_dim=hidden_dim,
        activation=activation,
        hidden_dims=hidden_dims,
        residual_scale=residual_scale,
    )
    if residual_zero_init:
        _zero_initialize_residual_output(model)

    initial_residual_rms_train = _residual_rms(model, train_data.initial_conditions)
    initial_residual_rms_test = _residual_rms(model, test_data.initial_conditions)
    initial_linear_weight_mse = _linear_weight_mse(model, omega)
    recorded_hidden_dims = list(hidden_dims) if hidden_dims is not None else [hidden_dim, hidden_dim]
    encoder_type = (
        "trainable_linear_path_plus_mlp_residual"
        if residual_type == "trainable_linear_residual"
        else "fixed_exact_linear_path_plus_mlp_residual"
    )
    setup = _base_setup(
        comparison_mode=comparison_mode,
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type=encoder_type,
        activation=activation,
        hidden_dim=recorded_hidden_dims,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    setup["log_every"] = log_every
    setup["residual_zero_init"] = residual_zero_init
    setup["residual_scale"] = residual_scale
    if residual_type == "fixed_linear_residual":
        setup["fixed_exact_weight_matrix"] = _target_linear_weight_matrix(omega).tolist()

    run_label = (
        "Trainable Linear Path + MLP Residual"
        if residual_type == "trainable_linear_residual"
        else "Fixed Exact Linear Path + MLP Residual"
    )
    run_name = f"{run_label} ({activation.upper() if activation == 'silu' else 'Tanh'})"
    logging.info("Stage 0 residual SHO setup (%s, %s):", residual_type, activation)
    for name, value in setup.items():
        logging.info("  %s: %s", name, value)

    training_history = _train_model(
        model,
        train_data,
        test_data,
        num_train,
        max_epochs,
        batch_size,
        learning_rate,
        log_every,
        run_name,
        stopping_criterion,
        stopping_monitor,
        stopping_min_delta,
        stopping_patience,
        target_loss,
        output_dir / "best_model.pt",
        restore_best_checkpoint,
    )

    return _evaluate_and_save_learned_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        setup=setup,
        training_history=training_history,
        output_dir=output_dir,
        run_name=run_name,
        omega=omega,
        extra_metrics={
            "residual_rms_epoch0": initial_residual_rms_test,
            "residual_rms_train_epoch0": initial_residual_rms_train,
            "residual_rms_test_epoch0": initial_residual_rms_test,
            "linear_weight_mse_epoch0": initial_linear_weight_mse,
            "residual_zero_init_verified": initial_residual_rms_test == 0.0,
        },
    )


def run_activation_comparison(
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
    seed: int = 0,
    omega: float = 2.0,
    num_train: int = 512,
    num_test: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    max_epochs: int | None = None,
    learning_rate: float | None = None,
    batch_size: int = 512,
    hidden_dim: int = 32,
    log_every: int = 500,
    stopping_criterion: str | None = None,
    stopping_monitor: str = "test_loss",
    stopping_min_delta: float = 1e-13,
    stopping_patience: int = 10000,
    target_loss: float | None = None,
    restore_best_checkpoint: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run Tanh and SiLU MLP experiments and save side-by-side plot."""
    if max_epochs is None:
        max_epochs = _default_max_epochs(comparison_mode)
    if learning_rate is None:
        learning_rate = SHARED_BUDGET_LEARNING_RATE
    if stopping_criterion is None:
        stopping_criterion = _default_stopping_criterion(comparison_mode)

    output_dir = (
        Path(output_dir)
        if output_dir
        else _run_dir("mlp_activation_comparison", comparison_mode=comparison_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tanh_metrics, tanh_plot_data = run_mlp_experiment(
        activation="tanh",
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        log_every=log_every,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
        output_dir=_run_dir("mlp_tanh", comparison_mode=comparison_mode),
    )
    silu_metrics, silu_plot_data = run_mlp_experiment(
        activation="silu",
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        log_every=log_every,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
        output_dir=_run_dir("mlp_silu", comparison_mode=comparison_mode),
    )

    comparison_setup = _base_setup(
        comparison_mode=comparison_mode,
        seed=seed,
        omega=omega,
        num_train=num_train,
        num_test=num_test,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        model_type="sho_exact_solution_decoder",
        encoder_type="mlp_activation_comparison",
        activation="tanh_vs_silu",
        hidden_dim=hidden_dim,
        stopping_criterion=stopping_criterion,
        stopping_monitor=stopping_monitor,
        stopping_min_delta=stopping_min_delta,
        stopping_patience=stopping_patience,
        target_loss=target_loss,
        restore_best_checkpoint=restore_best_checkpoint,
    )
    comparison_setup["log_every"] = log_every
    comparison_metrics = {
        "tanh": tanh_metrics,
        "silu": silu_metrics,
    }

    _write_json(output_dir / "setup.json", comparison_setup)
    _write_json(output_dir / "metrics.json", comparison_metrics)
    plot_phase_space_side_by_side(
        [
            {"title": "MLP + Tanh", **tanh_plot_data},
            {"title": "MLP + SiLU", **silu_plot_data},
        ],
        output_dir / "phase_space_side_by_side.png",
    )
    _write_run_index(
        output_dir=output_dir,
        run_name="SHO Stage 0: MLP Activation Comparison",
        setup=comparison_setup,
        metrics=comparison_metrics,
        training_summary={
            "final_train_loss": "N/A",
            "final_test_loss": "N/A",
            "best_train_loss": "N/A",
            "best_test_loss": "N/A",
            "epoch_of_best_train_loss": "N/A",
            "epoch_of_best_test_loss": "N/A",
            "best_epoch": "N/A",
            "reported_metrics_source": "N/A",
        },
        optimizer_analysis="N/A: this directory compares completed model runs and does not train its own model.",
        images=[
            ("Phase-Space Side by Side", "phase_space_side_by_side.png"),
        ],
    )

    return comparison_metrics


def run_all(comparison_mode: str = DEFAULT_COMPARISON_MODE) -> dict[str, Any]:
    comparison_metrics = run_activation_comparison(comparison_mode=comparison_mode)
    linear_metrics = run_linear_encoder_experiment(comparison_mode=comparison_mode)
    return {
        "activation_comparison": comparison_metrics,
        "linear_encoder": linear_metrics,
    }


STAGE0_RUN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "exact_linear": {
        "stage": "0 sequence 1",
        "output_dir": str(_run_dir("exact_linear")),
        "description": "Learn only W in (a,b)=W(x0,v0). Exact linear baseline for the controlled capacity sequence.",
    },
    "linear_tiny_residual": {
        "stage": "0 sequence 2",
        "output_dir": str(_run_dir("linear_tiny_residual")),
        "description": "Learn W(x0,v0) plus a tiny SiLU residual branch with architecture 2->4->2.",
    },
    "linear_tiny_residual_random_init": {
        "stage": "0 zero-init comparison",
        "output_dir": str(_run_dir("linear_tiny_residual_random_init")),
        "description": "Tiny SiLU residual branch with standard random initialization.",
    },
    "linear_tiny_residual_zero_init": {
        "stage": "0 zero-init comparison",
        "output_dir": str(_run_dir("linear_tiny_residual_zero_init")),
        "description": "Tiny SiLU residual branch with output initialized to zero for every input.",
    },
    "linear_moderate_residual": {
        "stage": "0 sequence 3",
        "output_dir": str(_run_dir("linear_moderate_residual")),
        "description": "Learn W(x0,v0) plus a moderate SiLU residual branch with architecture 2->8->8->2.",
    },
    "linear_large_residual": {
        "stage": "0 sequence 4",
        "output_dir": str(_run_dir("linear_large_residual")),
        "description": "Learn W(x0,v0) plus a larger SiLU residual branch with architecture 2->32->32->2.",
    },
    "pure_mlp": {
        "stage": "0 sequence 5",
        "output_dir": str(_run_dir("pure_mlp")),
        "description": "Pure SiLU MLP coefficient encoder with architecture 2->32->32->2.",
    },
    "sanity_check_analytic": {
        "stage": "0A",
        "output_dir": "outputs/sho/stage0/sanity_check_analytic_solution",
        "description": "No learning: a=x0 and b=v0/omega. Verification baseline only.",
    },
    "linear_encoder": {
        "stage": "0B",
        "output_dir": str(_run_dir("linear_encoder")),
        "description": "Learn W in (a,b)=W(x0,v0). Tests exact linear recovery.",
    },
    "mlp_tanh": {
        "stage": "0C",
        "output_dir": str(_run_dir("mlp_tanh")),
        "description": "Pure MLP encoder with Tanh activation. Tests unconstrained nonlinear recovery.",
    },
    "mlp_silu": {
        "stage": "0C",
        "output_dir": str(_run_dir("mlp_silu")),
        "description": "Pure MLP encoder with SiLU activation. Tests unconstrained nonlinear recovery.",
    },
    "mlp_tanh_adam_lbfgs": {
        "stage": "0D",
        "output_dir": str(_run_dir("mlp_tanh_adam_lbfgs")),
        "description": "Pure Tanh MLP with Adam followed by LBFGS polish.",
    },
    "mlp_silu_adam_lbfgs": {
        "stage": "0D",
        "output_dir": str(_run_dir("mlp_silu_adam_lbfgs")),
        "description": "Pure SiLU MLP with Adam followed by LBFGS polish.",
    },
    "trainable_linear_residual_tanh": {
        "stage": "0E",
        "output_dir": str(_run_dir("trainable_linear_residual_tanh")),
        "description": "Learn W(x0,v0) plus a Tanh MLP residual. Exact recovery is W_exact plus zero residual.",
    },
    "trainable_linear_residual_silu": {
        "stage": "0E",
        "output_dir": str(_run_dir("trainable_linear_residual_silu")),
        "description": "Learn W(x0,v0) plus a SiLU MLP residual. Exact recovery is W_exact plus zero residual.",
    },
    "fixed_linear_residual_tanh": {
        "stage": "0F",
        "output_dir": str(_run_dir("fixed_linear_residual_tanh")),
        "description": "Use fixed W_exact(x0,v0) plus a Tanh MLP residual. Exact recovery is zero residual.",
    },
    "fixed_linear_residual_silu": {
        "stage": "0F",
        "output_dir": str(_run_dir("fixed_linear_residual_silu")),
        "description": "Use fixed W_exact(x0,v0) plus a SiLU MLP residual. Exact recovery is zero residual.",
    },
}


def available_stage0_runs() -> dict[str, dict[str, str]]:
    """Return selectable Stage 0 run names and their output folders."""
    return STAGE0_RUN_DESCRIPTIONS


def run_stage0_variant(run_name: str) -> Any:
    """Run exactly one named Stage 0 variant."""
    if run_name == "exact_linear":
        return run_linear_encoder_experiment(output_dir=_run_dir("exact_linear"))
    if run_name == "linear_tiny_residual":
        return run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            hidden_dims=[4],
            output_dir=_run_dir("linear_tiny_residual"),
        )[0]
    if run_name == "linear_tiny_residual_random_init":
        return run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            hidden_dims=[4],
            output_dir=_run_dir("linear_tiny_residual_random_init"),
        )[0]
    if run_name == "linear_tiny_residual_zero_init":
        return run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            hidden_dims=[4],
            residual_zero_init=True,
            output_dir=_run_dir("linear_tiny_residual_zero_init"),
        )[0]
    if run_name == "linear_moderate_residual":
        return run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            hidden_dims=[8, 8],
            output_dir=_run_dir("linear_moderate_residual"),
        )[0]
    if run_name == "linear_large_residual":
        return run_residual_encoder_experiment(
            "trainable_linear_residual",
            "silu",
            hidden_dims=[32, 32],
            output_dir=_run_dir("linear_large_residual"),
        )[0]
    if run_name == "pure_mlp":
        return run_mlp_experiment(
            "silu",
            hidden_dims=[32, 32],
            output_dir=_run_dir("pure_mlp"),
        )[0]
    if run_name == "sanity_check_analytic":
        from experiments.sho.sanity_check_analytic_solution import run_sanity_check

        return run_sanity_check()
    if run_name == "linear_encoder":
        return run_linear_encoder_experiment()
    if run_name == "mlp_tanh":
        return run_mlp_experiment("tanh")[0]
    if run_name == "mlp_silu":
        return run_mlp_experiment("silu")[0]
    if run_name == "mlp_tanh_adam_lbfgs":
        return run_mlp_adam_lbfgs_experiment("tanh")[0]
    if run_name == "mlp_silu_adam_lbfgs":
        return run_mlp_adam_lbfgs_experiment("silu")[0]
    if run_name == "trainable_linear_residual_tanh":
        return run_residual_encoder_experiment("trainable_linear_residual", "tanh")[0]
    if run_name == "trainable_linear_residual_silu":
        return run_residual_encoder_experiment("trainable_linear_residual", "silu")[0]
    if run_name == "fixed_linear_residual_tanh":
        return run_residual_encoder_experiment("fixed_linear_residual", "tanh")[0]
    if run_name == "fixed_linear_residual_silu":
        return run_residual_encoder_experiment("fixed_linear_residual", "silu")[0]
    raise ValueError(f"unknown Stage 0 run name: {run_name}")


def _summary_value(data: dict[str, Any], key: str) -> Any:
    value = data.get(key, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def create_comparison_summary(
    run_names: list[str],
    comparison_mode: str = DEFAULT_COMPARISON_MODE,
) -> list[dict[str, Any]]:
    """Write CSV and HTML summaries for completed Stage 0 runs."""
    summary_dir = Path("outputs/sho/stage0") / comparison_mode
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for run_name in run_names:
        if run_name == "sanity_check_analytic":
            output_dir = Path(STAGE0_RUN_DESCRIPTIONS[run_name]["output_dir"])
        else:
            output_dir = Path(STAGE0_RUN_DESCRIPTIONS[run_name]["output_dir"])
        setup_path = output_dir / "setup.json"
        metrics_path = output_dir / "metrics.json"
        if not setup_path.exists() or not metrics_path.exists():
            logging.warning("Skipping missing run artifacts: %s", run_name)
            continue

        setup = json.loads(setup_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": run_name,
                "model_type": _summary_value(setup, "model_type"),
                "best_test_loss": _summary_value(metrics, "best_test_loss"),
                "coefficient_mse": _summary_value(metrics, "coefficient_mse"),
                "trajectory_mse": _summary_value(metrics, "trajectory_mse"),
                "epochs_completed": _summary_value(setup, "num_epochs_completed"),
                "best_epoch": _summary_value(metrics, "best_epoch"),
                "stopped_because": _summary_value(setup, "stopped_because"),
                "optimizer": _summary_value(setup, "optimizer"),
                "learning_rate": _summary_value(setup, "learning_rate"),
                "comparison_mode": _summary_value(setup, "comparison_mode"),
                "output_dir": str(output_dir),
            }
        )

    csv_columns = [
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
    ]
    csv_path = summary_dir / "comparison_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in csv_columns})

    html_rows = []
    for row in rows:
        cells = []
        for column in csv_columns:
            value = row.get(column, "")
            if column == "run_name":
                href = escape(str(Path(row["output_dir"]) / "index.html"))
                value = f'<a href="{href}">{escape(str(value))}</a>'
                cells.append(f"<td>{value}</td>")
            else:
                cells.append(f"<td>{escape(str(value))}</td>")
        html_rows.append("<tr>" + "".join(cells) + "</tr>")

    header = "".join(f"<th>{escape(column)}</th>" for column in csv_columns)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SHO Stage 0 Comparison Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    a {{ color: #1d4ed8; }}
  </style>
</head>
<body>
  <h1>SHO Stage 0 Comparison Summary</h1>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>
      {"".join(html_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (summary_dir / "comparison_summary.html").write_text(html, encoding="utf-8")
    return rows


def run_stage0_variants(run_names: list[str]) -> dict[str, Any]:
    """Run multiple Stage 0 variants by name, then write comparison summaries."""
    results = {}
    for run_name in run_names:
        logging.info("Running Stage 0 variant: %s", run_name)
        results[run_name] = run_stage0_variant(run_name)
    create_comparison_summary(run_names)
    return results


def _print_available_runs() -> None:
    print("Available Stage 0 runs:")
    for name, info in available_stage0_runs().items():
        print(f"  {name}")
        print(f"    stage: {info['stage']}")
        print(f"    output: {info['output_dir']}")
        print(f"    tests: {info['description']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Run one selectable SHO Stage 0 variant."
    )
    parser.add_argument(
        "--run",
        choices=sorted(STAGE0_RUN_DESCRIPTIONS),
        help="Run exactly one Stage 0 variant by name.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List available Stage 0 run names and output folders.",
    )
    args = parser.parse_args()

    if args.list_runs or args.run is None:
        _print_available_runs()
    else:
        metrics = run_stage0_variant(args.run)
        print(json.dumps(metrics, indent=2))
