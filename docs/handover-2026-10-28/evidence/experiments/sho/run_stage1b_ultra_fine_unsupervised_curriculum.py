"""Stage 1B ultra-fine unsupervised horizon curriculum.

This run uses only trajectory reconstruction loss. It does not use omega labels,
coefficient labels, auxiliary losses, or regularization.
"""

from __future__ import annotations

import copy
import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.sho.run_stage1b_two_state_omega import (
    BATCH_SIZE,
    DELTA_T,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    SUCCESS_OMEGA_TOLERANCE,
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/ultra_fine_unsupervised_curriculum")
RUN_NAME = "linear_moderate_residual"
HIDDEN_DIMS = [8, 8]
LEARNING_RATE = 1e-2
MAX_EPOCHS_PER_STAGE = 100000
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 20000
SUCCESS_TRAJECTORY_MSE = 1e-15
ULTRA_FINE_HORIZONS = [
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
    8.0,
    10.0,
]
LOG_EVERY = 10000


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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    stage_index: int,
    horizon_periods: float,
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
                    f"{RUN_NAME} stage {stage_index:02d} horizon {horizon_periods:g} "
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

    metrics, _, _, _ = _evaluate_model(model, test_data)
    return {
        "stage_index": stage_index,
        "horizon_periods": horizon_periods,
        "epochs_completed": epoch if "epoch" in locals() else 0,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        **metrics,
    }


def _save_stage_omega_plots(
    *,
    stage_dir: Path,
    stage_index: int,
    horizon_periods: float,
    omega_hat: torch.Tensor,
    true_omega: torch.Tensor,
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
        f"Stage 1B Ultra-Fine Curriculum Stage {stage_index}: {horizon_periods:g} periods",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(stage_dir / f"stage_{stage_index:02d}_predicted_vs_true_omega.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(error, bins=40)
    plt.xlabel("omega_hat - omega_true")
    plt.ylabel("count")
    plt.title(
        f"Stage 1B Ultra-Fine Curriculum Stage {stage_index}: Omega Error",
    )
    plt.tight_layout()
    plt.savefig(stage_dir / f"stage_{stage_index:02d}_omega_error_histogram.png")
    plt.close()


def _save_final_trajectory_examples(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    test_data: Any,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        _, trajectories, _ = model(test_data.two_state_inputs, test_data.times)
    example_count = min(num_examples, test_data.trajectories.shape[0])
    fig, axes = plt.subplots(example_count, 1, figsize=(8, 2.4 * example_count), sharex=False)
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
        ax.set_title(f"Final 10-period trajectory example {plot_index}")
        ax.set_xlabel("time")
        ax.set_ylabel("x(t)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "final_stage_trajectory_examples.png")
    plt.close(fig)


def _save_metric_plots(rows: list[dict[str, Any]]) -> None:
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
        plt.title(f"Stage 1B Ultra-Fine Unsupervised Curriculum: {ylabel}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    columns = [
        "stage_index",
        "horizon_periods",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "mean_omega_error",
        "mean_abs_omega_error",
        "max_omega_error",
        "omega_correlation",
        "best_epoch",
        "best_test_loss",
        "stopped_because",
    ]
    setup_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><pre>{escape(str(value))}</pre></td></tr>"
        for key, value in setup.items()
    )
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
            "trajectory_mse_vs_horizon.png",
            "coefficient_mse_vs_horizon.png",
            "omega_mse_vs_horizon.png",
            "omega_correlation_vs_horizon.png",
            "final_stage_trajectory_examples.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Ultra-Fine Unsupervised Curriculum</title>
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
  <h1>Stage 1B Ultra-Fine Unsupervised Curriculum</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Stage Metrics</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (OUTPUT_DIR / "ultra_fine_unsupervised_curriculum_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def run_ultra_fine_curriculum() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = {
        "stage": "stage1B",
        "experiment": "ultra_fine_unsupervised_curriculum",
        "run_name": RUN_NAME,
        "input": "(x0, v0, x1, v1)",
        "output": "(a, b, omega_hat_i)",
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
        "curriculum_horizons_periods": ULTRA_FINE_HORIZONS,
        "random_seed": SEED,
        "run_date": RUN_DATE,
    }
    rows = []
    for stage_index, horizon_periods in enumerate(ULTRA_FINE_HORIZONS, start=1):
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
        )
        rows.append(row)
        metrics, _, _, omega_hat = _evaluate_model(model, test_data)
        _save_stage_omega_plots(
            stage_dir=OUTPUT_DIR / "stage_omega_plots",
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            omega_hat=omega_hat,
            true_omega=test_data.omegas,
        )
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
                "stage_metrics_recomputed": metrics,
            },
            OUTPUT_DIR / f"checkpoint_after_stage_{stage_index:02d}.pt",
        )

    final_test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=ULTRA_FINE_HORIZONS[-1],
    )
    _save_final_trajectory_examples(output_dir=OUTPUT_DIR, model=model, test_data=final_test_data)
    torch.save(model.state_dict(), OUTPUT_DIR / "learned_parameters.pt")
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "ultra_fine_unsupervised_curriculum_summary.json", {"stages": rows})
    _write_csv(OUTPUT_DIR / "ultra_fine_unsupervised_curriculum_summary.csv", rows)
    _save_metric_plots(rows)
    _write_html(rows, setup)
    return rows


def main() -> None:
    rows = run_ultra_fine_curriculum()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
