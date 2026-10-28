"""Stage 1B discovery curriculum diagnostic.

Compare joint reconstruction training from scratch against a two-phase run:

1. Discover trajectory-specific omega with supervised omega regression.
2. Remove omega supervision and train the same reconstruction objective/curriculum.

The architecture, decoder, learning rate, and stopping rules are unchanged.
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
    CURRICULUM_HORIZONS,
    DELTA_T,
    MAX_EPOCHS_PER_STAGE,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    STOPPING_MIN_DELTA,
    STOPPING_PATIENCE,
    SUCCESS_OMEGA_TOLERANCE,
    SUCCESS_TRAJECTORY_MSE,
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/discovery_curriculum")
RUN_NAME = "large_residual"
HIDDEN_DIMS = [32, 32]
LEARNING_RATE = 3e-2
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


def _r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual_sum = (prediction - target).square().sum()
    total_sum = (target - target.mean()).square().sum()
    if total_sum <= 0:
        return float("nan")
    return (1.0 - residual_sum / total_sum).item()


def _omega_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    pred = prediction.detach().cpu()
    true = target.detach().cpu()
    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * true_centered.square().sum(),
    )
    if denominator <= 0:
        return float("nan")
    return (pred_centered * true_centered).sum().div(denominator).item()


def _phase1_metrics(
    omega_hat: torch.Tensor,
    omega_true: torch.Tensor,
    *,
    train_loss: float,
    test_loss: float,
) -> dict[str, float]:
    error = omega_hat - omega_true
    return {
        "omega_mse": error.square().mean().item(),
        "mean_omega_error": error.mean().item(),
        "mean_abs_omega_error": error.abs().mean().item(),
        "max_omega_error": error.abs().max().item(),
        "omega_correlation": _omega_correlation(omega_hat, omega_true),
        "r2_score": _r2_score(omega_hat, omega_true),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }


def _train_omega_discovery_phase(
    *,
    model: SHOTwoStateTrajectoryOmegaLinearResidualModel,
    optimizer: torch.optim.Optimizer,
    train_inputs: torch.Tensor,
    train_omegas: torch.Tensor,
    test_inputs: torch.Tensor,
    test_omegas: torch.Tensor,
) -> dict[str, Any]:
    dummy_train_times = torch.zeros(train_inputs.shape[0], 1)
    dummy_test_times = torch.zeros(test_inputs.shape[0], 1)
    num_train = train_inputs.shape[0]
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
            _, _, omega_hat = model(
                train_inputs[batch_indices],
                dummy_train_times[batch_indices],
            )
            loss = mse_loss(omega_hat, train_omegas[batch_indices])
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
            _, _, test_omega_hat = model(test_inputs, dummy_test_times)
            final_test_loss = mse_loss(test_omega_hat, test_omegas).item()

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
                    f"phase1 omega discovery epoch {epoch:06d} "
                    f"train_loss={final_train_loss:.6e} "
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

    model.eval()
    with torch.no_grad():
        _, _, train_prediction = model(train_inputs, dummy_train_times)
        _, _, test_prediction = model(test_inputs, dummy_test_times)
        train_loss = mse_loss(train_prediction, train_omegas).item()
        test_loss = mse_loss(test_prediction, test_omegas).item()
    metrics = _phase1_metrics(
        test_prediction,
        test_omegas,
        train_loss=train_loss,
        test_loss=test_loss,
    )
    metrics.update(
        {
            "phase": "omega_discovery",
            "epochs_completed": epoch if "epoch" in locals() else 0,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "final_epoch_train_loss": final_train_loss,
            "final_epoch_test_loss": final_test_loss,
            "stopped_because": stopped_because,
        },
    )
    return metrics


def _train_reconstruction_stage(
    *,
    method: str,
    model: SHOTwoStateTrajectoryOmegaLinearResidualModel,
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
                    f"{method} stage {stage_index} horizon {horizon_periods:g} "
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
        "method": method,
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


def _make_model_and_optimizer() -> tuple[
    SHOTwoStateTrajectoryOmegaLinearResidualModel,
    torch.optim.Optimizer,
]:
    model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    return model, torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def _run_reconstruction_curriculum(
    *,
    method: str,
    model: SHOTwoStateTrajectoryOmegaLinearResidualModel,
    optimizer: torch.optim.Optimizer,
    train_initial_conditions: torch.Tensor,
    train_omegas: torch.Tensor,
    test_initial_conditions: torch.Tensor,
    test_omegas: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for stage_index, horizon_periods in enumerate(CURRICULUM_HORIZONS, start=1):
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
        row = _train_reconstruction_stage(
            method=method,
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
        )
        rows.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
            },
            OUTPUT_DIR / f"{method}_checkpoint_after_stage_{stage_index:02d}.pt",
        )
    return rows


def _save_comparison_plots(rows: list[dict[str, Any]]) -> None:
    methods = ["joint_from_scratch", "omega_discovery_then_reconstruction"]
    plot_specs = [
        ("omega_mse_vs_stage.png", "omega_mse", "omega MSE"),
        ("trajectory_mse_vs_stage.png", "trajectory_mse", "trajectory MSE"),
        ("coefficient_mse_vs_stage.png", "coefficient_mse", "coefficient MSE"),
    ]
    for filename, key, ylabel in plot_specs:
        plt.figure(figsize=(7, 4.5))
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            horizons = [float(row["horizon_periods"]) for row in method_rows]
            values = [float(row[key]) for row in method_rows]
            plt.plot(horizons, values, marker="o", label=method)
        plt.xlabel("horizon in periods")
        plt.ylabel(ylabel)
        plt.yscale("log")
        plt.title(f"Stage 1B Discovery Curriculum: {ylabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any], phase1: dict[str, Any]) -> None:
    columns = [
        "method",
        "stage_index",
        "horizon_periods",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_omega_error",
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
    phase1_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in phase1.items()
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
            "omega_mse_vs_stage.png",
            "trajectory_mse_vs_stage.png",
            "coefficient_mse_vs_stage.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Discovery Curriculum</title>
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
  <h1>Stage 1B Discovery Curriculum</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Phase 1 Omega Discovery Metrics</h2>
  <table>{phase1_rows}</table>
  <h2>Curriculum Comparison</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (OUTPUT_DIR / "discovery_curriculum_summary.html").write_text(html, encoding="utf-8")


def run_discovery_curriculum() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    train_two_state = _build_data_from_base(
        train_initial_conditions,
        train_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    ).two_state_inputs
    test_two_state = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    ).two_state_inputs

    setup = {
        "stage": "stage1B",
        "experiment": "discovery_curriculum",
        "input": "(x0, v0, x1, v1)",
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "architecture": "linear + large residual encoder",
        "hidden_dims": HIDDEN_DIMS,
        "activation": "SiLU",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS_PER_STAGE,
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "curriculum_horizons_periods": CURRICULUM_HORIZONS,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
        "random_seed": SEED,
        "phase2_omega_supervision": "none",
    }

    joint_model, joint_optimizer = _make_model_and_optimizer()
    joint_rows = _run_reconstruction_curriculum(
        method="joint_from_scratch",
        model=joint_model,
        optimizer=joint_optimizer,
        train_initial_conditions=train_initial_conditions,
        train_omegas=train_omegas,
        test_initial_conditions=test_initial_conditions,
        test_omegas=test_omegas,
    )
    torch.save(joint_model.state_dict(), OUTPUT_DIR / "joint_from_scratch_parameters.pt")

    discovery_model, discovery_optimizer = _make_model_and_optimizer()
    phase1_metrics = _train_omega_discovery_phase(
        model=discovery_model,
        optimizer=discovery_optimizer,
        train_inputs=train_two_state,
        train_omegas=train_omegas,
        test_inputs=test_two_state,
        test_omegas=test_omegas,
    )
    torch.save(
        {
            "model_state_dict": discovery_model.state_dict(),
            "optimizer_state_dict": discovery_optimizer.state_dict(),
            "metrics": phase1_metrics,
        },
        OUTPUT_DIR / "phase1_omega_discovery_checkpoint.pt",
    )
    discovery_rows = _run_reconstruction_curriculum(
        method="omega_discovery_then_reconstruction",
        model=discovery_model,
        optimizer=discovery_optimizer,
        train_initial_conditions=train_initial_conditions,
        train_omegas=train_omegas,
        test_initial_conditions=test_initial_conditions,
        test_omegas=test_omegas,
    )
    torch.save(
        discovery_model.state_dict(),
        OUTPUT_DIR / "omega_discovery_then_reconstruction_parameters.pt",
    )

    rows = joint_rows + discovery_rows
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "phase1_omega_discovery_metrics.json", phase1_metrics)
    _write_json(OUTPUT_DIR / "discovery_curriculum_summary.json", {"stages": rows})
    _write_csv(OUTPUT_DIR / "discovery_curriculum_summary.csv", rows)
    _save_comparison_plots(rows)
    _write_html(rows, setup, phase1_metrics)
    return {"phase1": phase1_metrics, "stages": rows}


def main() -> None:
    results = run_discovery_curriculum()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
