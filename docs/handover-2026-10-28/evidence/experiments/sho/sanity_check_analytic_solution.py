"""Sanity check for the analytic SHO coefficient map and decoder."""

import json
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from hypothesis_space_world_models.evaluation.metrics import mean_squared_error
from hypothesis_space_world_models.evaluation.plots import (
    plot_phase_space_comparison,
    plot_trajectory_comparison,
)
from hypothesis_space_world_models.systems.sho import (
    coefficients_from_initial_conditions,
    decode_sho,
    decode_sho_phase_space,
    generate_sho_data,
)


def run_sanity_check(
    seed: int = 0,
    omega: float = 2.0,
    num_trajectories: int = 128,
    num_time_steps: int = 100,
    num_periods: float = 10.0,
    output_dir: str | Path = "outputs/sho/stage0/sanity_check_analytic_solution",
) -> dict[str, float]:
    """Verify the SHO formulas, decoder, metrics, and time sampling."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    period = math.tau / omega
    setup = {
        "comparison_mode": "sanity_check",
        "model_type": "sho_analytic_sanity_check",
        "encoder_type": "analytic_formula",
        "activation": None,
        "hidden_dim": None,
        "optimizer": "N/A",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "max_epochs": 0,
        "num_epochs_completed": 0,
        "stopping_criterion": "N/A",
        "stopping_monitor": "N/A",
        "stopping_min_delta": "N/A",
        "stopping_patience": "N/A",
        "target_loss": "N/A",
        "restore_best_checkpoint": False,
        "stopped_because": "N/A",
        "num_trajectories": num_trajectories,
        "num_train_trajectories": "N/A",
        "num_test_trajectories": num_trajectories,
        "num_time_points_per_trajectory": num_time_steps,
        "omega": omega,
        "sho_period": period,
        "num_periods_per_trajectory": num_periods,
        "random_seed": seed,
    }

    logging.info("Analytic SHO sanity-check setup:")
    for name, value in setup.items():
        logging.info("  %s: %s", name, value)

    setup_path = output_dir / "setup.json"
    with setup_path.open("w", encoding="utf-8") as file:
        json.dump(setup, file, indent=2)

    data = generate_sho_data(
        num_trajectories=num_trajectories,
        num_time_steps=num_time_steps,
        omega=omega,
        num_periods=num_periods,
        seed=seed,
    )

    analytic_coefficients = coefficients_from_initial_conditions(
        data.initial_conditions,
        omega,
    )
    analytic_trajectories = decode_sho(
        analytic_coefficients,
        data.times,
        omega,
    )

    metrics = {
        "coefficient_mse": mean_squared_error(
            analytic_coefficients,
            data.coefficients,
        ),
        "trajectory_mse": mean_squared_error(
            analytic_trajectories,
            data.trajectories,
        ),
        "best_test_loss": mean_squared_error(
            analytic_trajectories,
            data.trajectories,
        ),
        "best_epoch": "N/A",
        "reported_metrics_source": "analytic_formula",
    }

    metrics_path = output_dir / "metrics.json"
    trajectory_plot_path = output_dir / "trajectory_comparison.png"
    phase_space_plot_path = output_dir / "phase_space_comparison.png"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    target_positions, target_velocities = decode_sho_phase_space(
        data.coefficients,
        data.times,
        omega,
    )
    analytic_positions, analytic_velocities = decode_sho_phase_space(
        analytic_coefficients,
        data.times,
        omega,
    )

    plot_trajectory_comparison(
        data.times,
        data.trajectories[0],
        analytic_trajectories[0],
        trajectory_plot_path,
    )
    plot_phase_space_comparison(
        target_positions[0],
        target_velocities[0],
        analytic_positions[0],
        analytic_velocities[0],
        phase_space_plot_path,
        title="SHO Stage 0A: Analytic Sanity Check\nPhase Space",
    )

    training_history = {
        "losses": [],
        "num_epochs_completed": 0,
        "final_epoch_train_loss": "N/A",
        "final_epoch_test_loss": "N/A",
        "final_train_loss": "N/A",
        "final_test_loss": metrics["trajectory_mse"],
        "best_train_loss": "N/A",
        "best_test_loss": metrics["trajectory_mse"],
        "epoch_of_best_train_loss": "N/A",
        "epoch_of_best_test_loss": "N/A",
        "best_epoch": "N/A",
        "reported_metrics_source": "analytic_formula",
    }
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(training_history, file, indent=2)

    torch.save({}, output_dir / "learned_parameters.pt")

    optimizer_analysis = (
        "N/A: this is an analytic sanity check with no learned parameters, "
        "optimizer, or training loop."
    )
    (output_dir / "optimizer_analysis.txt").write_text(
        optimizer_analysis,
        encoding="utf-8",
    )

    plt.figure(figsize=(7, 4.5))
    plt.text(
        0.5,
        0.5,
        "N/A: analytic sanity check\nNo training loss",
        ha="center",
        va="center",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.close()

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SHO Stage 0A: Analytic Sanity Check</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; max-width: 980px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ width: 280px; background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>SHO Stage 0A: Analytic Sanity Check</h1>
  <h2>Model Description</h2>
  <table>
    <tr><th>model_type</th><td><pre>{setup["model_type"]}</pre></td></tr>
    <tr><th>encoder_type</th><td><pre>{setup["encoder_type"]}</pre></td></tr>
  </table>
  <h2>Setup / Configuration</h2>
  <pre>{json.dumps(setup, indent=2)}</pre>
  <h2>Training Metrics</h2>
  <pre>{json.dumps(training_history, indent=2)}</pre>
  <h2>Evaluation Metrics</h2>
  <pre>{json.dumps(metrics, indent=2)}</pre>
  <section><h2>Trajectory Comparison</h2><img src="trajectory_comparison.png" alt="Trajectory Comparison"></section>
  <section><h2>Phase-Space Comparison</h2><img src="phase_space_comparison.png" alt="Phase-Space Comparison"></section>
  <section><h2>Training Loss</h2><img src="training_loss.png" alt="Training Loss"></section>
  <h2>Optimizer Analysis</h2>
  <pre>{optimizer_analysis}</pre>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics = run_sanity_check()
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")
