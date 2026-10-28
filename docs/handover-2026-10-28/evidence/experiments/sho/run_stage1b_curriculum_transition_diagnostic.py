"""Stage 1B curriculum transition diagnostic.

This is a localization run, not an architecture sweep. It uses the same
Stage 1B moderate residual model/settings that showed strong omega recovery
through 2 periods and collapse at the 2 -> 5 period transition, then inserts
intermediate horizons to find where the branch is lost.
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


OUTPUT_DIR = Path("outputs/sho/stage1B/curriculum_transition_diagnostic")
RUN_NAME = "linear_moderate_residual"
HIDDEN_DIMS = [8, 8]
LEARNING_RATE = 1e-2
TRANSITION_HORIZONS = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
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


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _train_stage_with_transition_records(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    stage_index: int,
    horizon_periods: float,
) -> dict[str, Any]:
    start_metrics, _, _, _ = _evaluate_model(model, test_data)
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
                    f"{RUN_NAME} stage {stage_index} horizon {horizon_periods:g} "
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

    restored_metrics, _, _, _ = _evaluate_model(model, test_data)
    discrepancy = {
        key: abs(restored_metrics[key] - best_test_loss)
        for key in ["trajectory_mse"]
    }
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
        "restored_checkpoint_trajectory_mse_discrepancy": discrepancy[
            "trajectory_mse"
        ],
        **_prefixed_metrics("start", start_metrics),
        **_prefixed_metrics("restored", restored_metrics),
    }


def _save_plots(rows: list[dict[str, Any]]) -> None:
    horizons = [float(row["horizon_periods"]) for row in rows]
    plot_specs = [
        ("omega_mse_vs_horizon.png", "restored_omega_mse", "omega MSE"),
        (
            "coefficient_mse_vs_horizon.png",
            "restored_coefficient_mse",
            "coefficient MSE",
        ),
        (
            "trajectory_mse_vs_horizon.png",
            "restored_trajectory_mse",
            "trajectory MSE",
        ),
        (
            "omega_correlation_vs_horizon.png",
            "restored_omega_correlation",
            "omega correlation",
        ),
    ]
    for filename, key, ylabel in plot_specs:
        values = [float(row[key]) for row in rows]
        plt.figure(figsize=(7, 4.5))
        plt.plot(horizons, values, marker="o")
        plt.xlabel("horizon in periods")
        plt.ylabel(ylabel)
        if "mse" in key:
            plt.yscale("log")
        plt.title(f"Stage 1B Transition Diagnostic: {ylabel}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    columns = [
        "stage_index",
        "horizon_periods",
        "restored_trajectory_mse",
        "restored_coefficient_mse",
        "restored_omega_mse",
        "restored_mean_omega_error",
        "restored_mean_abs_omega_error",
        "restored_max_omega_error",
        "restored_omega_correlation",
        "best_epoch",
        "best_test_loss",
        "stopped_because",
        "start_vs_previous_restored_trajectory_mse_delta",
        "start_vs_previous_restored_coefficient_mse_delta",
        "start_vs_previous_restored_omega_mse_delta",
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
            "omega_mse_vs_horizon.png",
            "coefficient_mse_vs_horizon.png",
            "trajectory_mse_vs_horizon.png",
            "omega_correlation_vs_horizon.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Curriculum Transition Diagnostic</title>
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
  <h1>Stage 1B Curriculum Transition Diagnostic</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Transition Summary</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (OUTPUT_DIR / "transition_summary.html").write_text(html, encoding="utf-8")


def run_transition_diagnostic() -> list[dict[str, Any]]:
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
        "experiment": "curriculum_transition_diagnostic",
        "run_name": RUN_NAME,
        "input": "(x0, v0, x1, v1)",
        "delta_t": DELTA_T,
        "omega_range": OMEGA_RANGE,
        "model": "(x0, v0, x1, v1) -> linear encoder + residual encoder -> (a, b, omega_hat_i)",
        "residual_hidden_dims": HIDDEN_DIMS,
        "activation": "SiLU",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "stopping_monitor": "test_loss",
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "curriculum_horizons_periods": TRANSITION_HORIZONS,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
        "random_seed": SEED,
        "note": "Localization run for the moderate residual model; no architecture sweep.",
    }
    rows = []
    previous_restored_metrics: dict[str, Any] | None = None
    for stage_index, horizon_periods in enumerate(TRANSITION_HORIZONS, start=1):
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
        row = _train_stage_with_transition_records(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
        )
        if previous_restored_metrics is None:
            row["start_vs_previous_restored_trajectory_mse_delta"] = ""
            row["start_vs_previous_restored_coefficient_mse_delta"] = ""
            row["start_vs_previous_restored_omega_mse_delta"] = ""
            row["start_vs_previous_restored_omega_correlation_delta"] = ""
        else:
            row["start_vs_previous_restored_trajectory_mse_delta"] = abs(
                row["start_trajectory_mse"]
                - previous_restored_metrics["trajectory_mse"],
            )
            row["start_vs_previous_restored_coefficient_mse_delta"] = abs(
                row["start_coefficient_mse"]
                - previous_restored_metrics["coefficient_mse"],
            )
            row["start_vs_previous_restored_omega_mse_delta"] = abs(
                row["start_omega_mse"] - previous_restored_metrics["omega_mse"],
            )
            row["start_vs_previous_restored_omega_correlation_delta"] = abs(
                row["start_omega_correlation"]
                - previous_restored_metrics["omega_correlation"],
            )
        previous_restored_metrics = {
            "trajectory_mse": row["restored_trajectory_mse"],
            "coefficient_mse": row["restored_coefficient_mse"],
            "omega_mse": row["restored_omega_mse"],
            "omega_correlation": row["restored_omega_correlation"],
        }
        rows.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
            },
            OUTPUT_DIR / f"checkpoint_after_stage_{stage_index:02d}.pt",
        )

    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "transition_summary.json", {"stages": rows})
    _write_csv(OUTPUT_DIR / "transition_summary.csv", rows)
    _save_plots(rows)
    _write_html(rows, setup)
    return rows


def main() -> None:
    rows = run_transition_diagnostic()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
