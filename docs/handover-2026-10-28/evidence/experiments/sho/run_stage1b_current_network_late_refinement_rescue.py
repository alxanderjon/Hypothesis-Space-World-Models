"""Stage 1B rescue diagnostic for the current feedforward residual model.

Start from the best restored 8-period checkpoint produced by the late-horizon
refinement curriculum, then refine only at the 10-period horizon with smaller
learning rates and reconstruction loss only.
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
    _build_data_from_base,
    _evaluate_model,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/current_network_late_refinement_rescue")
SOURCE_CHECKPOINT = Path(
    "outputs/sho/stage1B/late_horizon_refinement_curriculum/checkpoint_after_stage_14.pt",
)
RUN_NAME = "linear_moderate_residual"
HIDDEN_DIMS = [8, 8]
SOURCE_HORIZON_PERIODS = 8.0
TARGET_HORIZON_PERIODS = 10.0
LEARNING_RATES = [1e-3, 3e-4, 1e-4]
MAX_EPOCHS = 100000
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 20000
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


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _make_model_from_checkpoint() -> SHOTwoStateTrajectoryOmegaLinearResidualModel:
    checkpoint = _load_checkpoint(SOURCE_CHECKPOINT)
    model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
        hidden_dims=HIDDEN_DIMS,
        initial_omega_hat=2.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _train_10_period_refinement(
    *,
    learning_rate: float,
    train_data: Any,
    test_data: Any,
) -> tuple[dict[str, Any], SHOTwoStateTrajectoryOmegaLinearResidualModel]:
    model = _make_model_from_checkpoint()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

    source_metrics, _, _, _ = _evaluate_model(model, test_data)
    for epoch in range(1, MAX_EPOCHS + 1):
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
                    f"lr={learning_rate:g} epoch {epoch:06d} "
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

    metrics, _, _, omega_hat = _evaluate_model(model, test_data)
    row = {
        "experiment": "small_lr_10_period_refinement",
        "learning_rate": learning_rate,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_horizon_periods": SOURCE_HORIZON_PERIODS,
        "target_horizon_periods": TARGET_HORIZON_PERIODS,
        "source_10_period_trajectory_mse": source_metrics["trajectory_mse"],
        "source_10_period_coefficient_mse": source_metrics["coefficient_mse"],
        "source_10_period_omega_mse": source_metrics["omega_mse"],
        "epochs_completed": epoch if "epoch" in locals() else 0,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        **metrics,
    }
    run_dir = OUTPUT_DIR / f"lr_{learning_rate:g}".replace("-", "m").replace(".", "p")
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "metrics.json", row)
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _save_predicted_vs_true_omega(
        run_dir / "predicted_vs_true_omega.png",
        omega_hat,
        test_data.omegas,
        f"Stage 1B Rescue lr={learning_rate:g}",
    )
    _save_trajectory_examples(
        run_dir / "trajectory_examples.png",
        model,
        test_data,
        title=f"Stage 1B Rescue lr={learning_rate:g}",
    )
    return row, model


def _save_predicted_vs_true_omega(
    path: Path,
    omega_hat: torch.Tensor,
    true_omega: torch.Tensor,
    title: str,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(true_omega.detach().cpu(), omega_hat.detach().cpu(), s=18, alpha=0.75)
    plt.plot([1.5, 2.5], [1.5, 2.5], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("predicted omega")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_trajectory_examples(
    path: Path,
    model: torch.nn.Module,
    test_data: Any,
    *,
    title: str,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        _, trajectories, _ = model(test_data.two_state_inputs, test_data.times)
    example_count = min(num_examples, test_data.trajectories.shape[0])
    fig, axes = plt.subplots(example_count, 1, figsize=(8, 2.4 * example_count))
    if example_count == 1:
        axes = [axes]
    for index in range(example_count):
        ax = axes[index]
        ax.plot(
            test_data.times[index].detach().cpu(),
            test_data.trajectories[index].detach().cpu(),
            label="target",
        )
        ax.plot(
            test_data.times[index].detach().cpu(),
            trajectories[index].detach().cpu(),
            "--",
            label="prediction",
        )
        ax.set_title(f"{title}: example {index}")
        ax.set_xlabel("time")
        ax.set_ylabel("x(t)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_summary_plots(rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['learning_rate']:g}" for row in rows]
    plot_specs = [
        ("trajectory_mse_by_lr.png", "trajectory_mse", "trajectory MSE"),
        ("omega_mse_by_lr.png", "omega_mse", "omega MSE"),
        ("coefficient_mse_by_lr.png", "coefficient_mse", "coefficient MSE"),
    ]
    x_positions = list(range(len(rows)))
    for filename, key, ylabel in plot_specs:
        values = [float(row[key]) for row in rows]
        plt.figure(figsize=(7, 4.5))
        plt.bar(x_positions, values)
        plt.yscale("log")
        plt.xticks(x_positions, labels)
        plt.xlabel("learning rate")
        plt.ylabel(ylabel)
        plt.title(f"Stage 1B Current Network Late Refinement Rescue: {ylabel}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    columns = [
        "experiment",
        "learning_rate",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_omega_error",
        "mean_abs_omega_error",
        "max_omega_error",
        "best_epoch",
        "stopped_because",
        "source_10_period_trajectory_mse",
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
            "trajectory_mse_by_lr.png",
            "omega_mse_by_lr.png",
            "coefficient_mse_by_lr.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Current Network Late Refinement Rescue</title>
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
  <h1>Stage 1B Current Network Late Refinement Rescue</h1>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Summary</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
</body>
</html>
"""
    (OUTPUT_DIR / "late_refinement_rescue_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def _freeze_unfreeze_skip_row() -> dict[str, Any]:
    return {
        "experiment": "staged_freeze_unfreeze",
        "learning_rate": "N/A",
        "trajectory_mse": "N/A",
        "coefficient_mse": "N/A",
        "omega_mse": "N/A",
        "mean_omega_error": "N/A",
        "mean_abs_omega_error": "N/A",
        "max_omega_error": "N/A",
        "omega_correlation": "N/A",
        "best_epoch": "N/A",
        "stopped_because": "skipped",
        "skip_reason": (
            "The current linear_moderate_residual model has one shared linear "
            "encoder and one shared residual MLP whose final tensors jointly "
            "produce (a, b, raw_omega). It does not have cleanly separated "
            "coefficient and omega heads, so freezing coefficient-output "
            "parameters would require partial row-level parameter surgery rather "
            "than freezing an architectural component."
        ),
    }


def run_late_refinement_rescue() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    train_data = _build_data_from_base(
        train_initial_conditions,
        train_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=TARGET_HORIZON_PERIODS,
    )
    test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=TARGET_HORIZON_PERIODS,
    )
    setup = {
        "stage": "stage1B",
        "experiment": "current_network_late_refinement_rescue",
        "run_name": RUN_NAME,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_horizon_periods": SOURCE_HORIZON_PERIODS,
        "target_horizon_periods": TARGET_HORIZON_PERIODS,
        "input": "(x0, v0, x1, v1)",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "loss": "trajectory reconstruction only",
        "omega_supervision": False,
        "coefficient_supervision": False,
        "auxiliary_losses": False,
        "regularization": False,
        "model": "linear + moderate residual encoder",
        "hidden_dims": HIDDEN_DIMS,
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "optimizer": "Adam",
        "learning_rates": LEARNING_RATES,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "random_seed": SEED,
        "run_date": RUN_DATE,
        "experiment_b_status": "skipped",
        "experiment_b_skip_reason": _freeze_unfreeze_skip_row()["skip_reason"],
    }
    rows = []
    for learning_rate in LEARNING_RATES:
        print(f"running 10-period rescue refinement lr={learning_rate:g}", flush=True)
        row, _ = _train_10_period_refinement(
            learning_rate=learning_rate,
            train_data=train_data,
            test_data=test_data,
        )
        rows.append(row)
    rows.append(_freeze_unfreeze_skip_row())
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "late_refinement_rescue_summary.json", {"runs": rows})
    _write_csv(OUTPUT_DIR / "late_refinement_rescue_summary.csv", rows)
    numeric_rows = [row for row in rows if isinstance(row.get("trajectory_mse"), float)]
    _save_summary_plots(numeric_rows)
    _write_html(rows, setup)
    return rows


def main() -> None:
    rows = run_late_refinement_rescue()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
