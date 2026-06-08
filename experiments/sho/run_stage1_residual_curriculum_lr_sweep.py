"""Run Stage 1 residual time-horizon curricula across learning rates."""

from __future__ import annotations

import copy
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.sho.run_stage1_followup_sweeps import RUN_DATE, TRUE_OMEGA
from experiments.sho.run_stage1_residual_omega_init_sweep import (
    SHOTrainableLinearResidualLearnedOmegaModel,
    _omega_label,
    _residual_variants,
)
from experiments.sho.run_stage1_residual_time_horizon_curriculum import (
    CURRICULUM_HORIZONS,
    OMEGA_INITIALIZATIONS,
    _evaluate_stage,
)
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1/residual_curriculum_lr_sweep")
LEARNING_RATES = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
SUCCESS_TRAJECTORY_MSE = 1e-15
SUCCESS_OMEGA_TOLERANCE = 1e-6


def _lr_label(value: float) -> str:
    return f"{value:.0e}".replace("-", "m")


def _horizon_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _success(best_trajectory_mse: float, best_omega_hat: float) -> bool:
    return (
        best_trajectory_mse < SUCCESS_TRAJECTORY_MSE
        and abs(best_omega_hat - TRUE_OMEGA) < SUCCESS_OMEGA_TOLERANCE
    )


def _train_stage_with_best_restore(
    *,
    model: SHOTrainableLinearResidualLearnedOmegaModel,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    model_name: str,
    omega_init: float,
    learning_rate: float,
    stage_index: int,
    horizon_periods: float,
    max_epochs_per_stage: int,
    batch_size: int,
    stopping_min_delta: float,
    stopping_patience: int,
    log_every: int,
) -> dict[str, Any]:
    num_train = train_data.initial_conditions.shape[0]
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    final_train_loss = float("inf")
    final_test_loss = float("inf")
    final_omega_hat = float(model.omega_hat.detach().cpu())

    for epoch in range(1, max_epochs_per_stage + 1):
        model.train()
        shuffled_indices = torch.randperm(num_train)
        epoch_loss_sum = 0.0
        epoch_example_count = 0

        for start in range(0, num_train, batch_size):
            batch_indices = shuffled_indices[start : start + batch_size]
            optimizer.zero_grad()
            _, pred_trajectories = model(
                train_data.initial_conditions[batch_indices],
                train_data.times,
            )
            loss = mse_loss(
                pred_trajectories,
                train_data.trajectories[batch_indices],
            )
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            batch_count = len(batch_indices)
            epoch_loss_sum += loss.item() * batch_count
            epoch_example_count += batch_count

        if stopped_because == "nan_or_inf_loss":
            break

        final_train_loss = epoch_loss_sum / epoch_example_count
        model.eval()
        with torch.no_grad():
            _, test_pred_trajectories = model(
                test_data.initial_conditions,
                test_data.times,
            )
            final_test_loss = mse_loss(
                test_pred_trajectories,
                test_data.trajectories,
            ).item()

        final_omega_hat = float(model.omega_hat.detach().cpu())

        if final_test_loss < best_test_loss - stopping_min_delta:
            best_test_loss = final_test_loss
            best_train_loss = final_train_loss
            best_epoch = epoch
            best_model_state = _clone_state_dict(model.state_dict())
            best_optimizer_state = _clone_state_dict(optimizer.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if log_every > 0 and (
            epoch == 1 or epoch % log_every == 0 or epoch == max_epochs_per_stage
        ):
            print(
                (
                    f"{model_name} init {omega_init:g} lr {learning_rate:g} "
                    f"stage {stage_index} horizon {horizon_periods:g} "
                    f"epoch {epoch:06d} train_loss: {final_train_loss:.6e} "
                    f"test_loss: {final_test_loss:.6e} best_test: {best_test_loss:.6e} "
                    f"omega_hat: {final_omega_hat:.9g}"
                ),
                flush=True,
            )

        if epochs_since_improvement >= stopping_patience:
            stopped_because = "patience_exhausted"
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)

    restored_metrics = _evaluate_stage(model, train_data, test_data)
    best_omega_hat = float(model.omega_hat.detach().cpu())
    checkpoint_path = (
        f"checkpoint_after_stage_{stage_index:02d}_"
        f"{_horizon_label(horizon_periods)}_periods.pt"
    )
    return {
        "model": model_name,
        "omega_init": omega_init,
        "learning_rate": learning_rate,
        "stage_index": stage_index,
        "horizon_periods": horizon_periods,
        "final_omega_hat_before_restore": final_omega_hat,
        "best_omega_hat": best_omega_hat,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "final_trajectory_mse": final_test_loss,
        "best_trajectory_mse": restored_metrics["trajectory_mse"],
        "final_coefficient_mse": "",
        "best_coefficient_mse": restored_metrics["coefficient_mse"],
        "mean_abs_predicted_a_test": restored_metrics["mean_abs_predicted_a_test"],
        "mean_abs_predicted_b_test": restored_metrics["mean_abs_predicted_b_test"],
        "mean_abs_true_a_test": restored_metrics["mean_abs_true_a_test"],
        "mean_abs_true_b_test": restored_metrics["mean_abs_true_b_test"],
        "residual_rms_test": restored_metrics["residual_rms_test"],
        "linear_weight_mse": restored_metrics["linear_weight_mse"],
        "epochs_completed": epoch if "epoch" in locals() else 0,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "restored_best_checkpoint_before_next_stage": True,
        "optimizer_state_restored_from_best_epoch": best_optimizer_state is not None,
        "checkpoint_path": checkpoint_path,
    }


def run_one_curriculum(
    *,
    variant: dict[str, Any],
    omega_init: float,
    learning_rate: float,
    output_dir: Path,
) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_train = 512
    num_test = 128
    num_time_steps = 100
    batch_size = 512
    max_epochs_per_stage = 100000
    stopping_min_delta = 0.0
    stopping_patience = 20000
    log_every = 10000
    activation = "silu"

    set_seed(seed)
    model = SHOTrainableLinearResidualLearnedOmegaModel(
        initial_omega_hat=omega_init,
        hidden_dims=list(variant["hidden_dims"]),
        activation=activation,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    setup = {
        "comparison_mode": "stage1_residual_time_horizon_curriculum_lr_sweep",
        "model_type": "sho_trainable_linear_residual_learned_global_omega",
        "encoder_type": f"trainable_linear_plus_{variant['model_name']}",
        "activation": activation,
        "hidden_dim": list(variant["hidden_dims"]),
        "optimizer": "Adam",
        "optimizer_reset_between_stages": False,
        "optimizer_state_restored_from_best_epoch_between_stages": True,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs_per_stage": max_epochs_per_stage,
        "stopping_monitor": "test_loss",
        "stopping_min_delta": stopping_min_delta,
        "stopping_patience": stopping_patience,
        "true_omega": TRUE_OMEGA,
        "initial_omega_hat_requested": omega_init,
        "omega_hat_initial": float(model.omega_hat.detach().cpu()),
        "omega_parameterization": "softplus(raw_omega) + 1e-8",
        "num_train_trajectories": num_train,
        "num_test_trajectories": num_test,
        "num_time_points_per_stage": num_time_steps,
        "curriculum_horizons_periods": CURRICULUM_HORIZONS,
        "random_seed": seed,
    }

    rows = []
    for stage_index, horizon_periods in enumerate(CURRICULUM_HORIZONS, start=1):
        train_data = generate_sho_data(
            num_train,
            num_time_steps,
            omega=TRUE_OMEGA,
            num_periods=horizon_periods,
            seed=seed,
        )
        test_data = generate_sho_data(
            num_test,
            num_time_steps,
            omega=TRUE_OMEGA,
            num_periods=horizon_periods,
            seed=seed + 1,
        )
        row = _train_stage_with_best_restore(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            model_name=variant["model_name"],
            omega_init=omega_init,
            learning_rate=learning_rate,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            max_epochs_per_stage=max_epochs_per_stage,
            batch_size=batch_size,
            stopping_min_delta=stopping_min_delta,
            stopping_patience=stopping_patience,
            log_every=log_every,
        )
        rows.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
            },
            output_dir / row["checkpoint_path"],
        )

    final_stage = rows[-1]
    run_metrics = {
        **final_stage,
        "final_omega_hat": final_stage["final_omega_hat_before_restore"],
        "best_final_stage_omega_hat": final_stage["best_omega_hat"],
        "best_final_stage_trajectory_mse": final_stage["best_trajectory_mse"],
        "best_final_stage_coefficient_mse": final_stage["best_coefficient_mse"],
        "recovered_successfully": _success(
            float(final_stage["best_trajectory_mse"]),
            float(final_stage["best_omega_hat"]),
        ),
    }
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": rows})
    _write_json(output_dir / "metrics.json", run_metrics)
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    _write_run_index(output_dir, setup, rows)
    return rows


def _write_run_index(run_dir: Path, setup: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    columns = [
        "stage_index",
        "horizon_periods",
        "best_omega_hat",
        "best_trajectory_mse",
        "best_coefficient_mse",
        "residual_rms_test",
        "linear_weight_mse",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
    ]
    setup_rows = "".join(
        "<tr>"
        f"<th>{escape(str(key))}</th>"
        f"<td><pre>{escape(json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value))}</pre></td>"
        "</tr>"
        for key, value in setup.items()
    )
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Residual Curriculum LR Sweep Run</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Stage 1 Residual Curriculum LR Sweep Run</h1>
  <p>Each curriculum stage restores the best model and Adam optimizer state before continuing to the next horizon.</p>
  <h2>Setup</h2>
  <table>{setup_rows}</table>
  <h2>Stage Summary</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")


def _final_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if int(row["stage_index"]) == len(CURRICULUM_HORIZONS)]


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    final_rows = _final_stage_rows(rows)
    columns = [
        "model",
        "omega_init",
        "learning_rate",
        "final_omega_hat",
        "best_final_stage_omega_hat",
        "final_trajectory_mse",
        "best_final_stage_trajectory_mse",
        "final_coefficient_mse",
        "best_final_stage_coefficient_mse",
        "mean_abs_predicted_a_test",
        "mean_abs_predicted_b_test",
        "mean_abs_true_a_test",
        "mean_abs_true_b_test",
        "residual_rms_test",
        "linear_weight_mse",
        "epochs_completed",
        "best_epoch",
        "stopped_because",
        "recovered_successfully",
        "output_dir",
    ]
    summary_rows = []
    for row in final_rows:
        run_name = (
            f"{RUN_DATE}_{row['model']}_omega_init_{_omega_label(float(row['omega_init']))}"
            f"_lr_{_lr_label(float(row['learning_rate']))}"
        )
        summary_rows.append(
            {
                "model": row["model"],
                "omega_init": row["omega_init"],
                "learning_rate": row["learning_rate"],
                "final_omega_hat": row["final_omega_hat_before_restore"],
                "best_final_stage_omega_hat": row["best_omega_hat"],
                "final_trajectory_mse": row["final_trajectory_mse"],
                "best_final_stage_trajectory_mse": row["best_trajectory_mse"],
                "final_coefficient_mse": row["final_coefficient_mse"],
                "best_final_stage_coefficient_mse": row["best_coefficient_mse"],
                "mean_abs_predicted_a_test": row["mean_abs_predicted_a_test"],
                "mean_abs_predicted_b_test": row["mean_abs_predicted_b_test"],
                "mean_abs_true_a_test": row["mean_abs_true_a_test"],
                "mean_abs_true_b_test": row["mean_abs_true_b_test"],
                "residual_rms_test": row["residual_rms_test"],
                "linear_weight_mse": row["linear_weight_mse"],
                "epochs_completed": row["epochs_completed"],
                "best_epoch": row["best_epoch"],
                "stopped_because": row["stopped_because"],
                "recovered_successfully": _success(
                    float(row["best_trajectory_mse"]),
                    float(row["best_omega_hat"]),
                ),
                "output_dir": str(output_dir / run_name),
            },
        )

    with (output_dir / "residual_curriculum_lr_sweep_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{_format_html_cell(column, row.get(column, ''))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in summary_rows
    )
    images = [
        ("Best Loss by Model and LR", "best_loss_by_model_and_lr.png"),
        ("Best Loss by Init, Model, and LR", "best_loss_by_init_model_lr.png"),
        ("Omega Error by Model and LR", "omega_error_by_model_and_lr.png"),
        ("Residual RMS by Model and LR", "residual_rms_by_model_and_lr.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(filename)}" alt="{escape(title)}"></section>'
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Residual Curriculum LR Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 12px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 1080px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 1 Residual Curriculum Learning-Rate Sweep</h1>
  <p>Each run restores the best model and Adam optimizer state at every curriculum stage before moving to the next horizon.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {image_html}
</body>
</html>
"""
    (output_dir / "residual_curriculum_lr_sweep_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def _format_html_cell(column: str, value: Any) -> str:
    if column == "output_dir" and value:
        path = Path(str(value))
        return f'<a href="{escape(path.name)}/index.html">{escape(path.name)}</a>'
    return escape(str(value))


def _plot_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    final_rows = _final_stage_rows(rows)
    model_names = sorted({row["model"] for row in final_rows})

    plt.figure(figsize=(9, 5.5))
    for model_name in model_names:
        xs = []
        ys = []
        for lr in LEARNING_RATES:
            selected = [
                row
                for row in final_rows
                if row["model"] == model_name and float(row["learning_rate"]) == lr
            ]
            if selected:
                xs.append(lr)
                ys.append(min(float(row["best_trajectory_mse"]) for row in selected))
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("best final-stage trajectory MSE")
    plt.title("Stage 1 Residual Curriculum LR Sweep: Best Loss by Model and LR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "best_loss_by_model_and_lr.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for model_name in model_names:
        for omega_init in OMEGA_INITIALIZATIONS:
            selected = [
                row
                for row in final_rows
                if row["model"] == model_name and float(row["omega_init"]) == omega_init
            ]
            plt.plot(
                [float(row["learning_rate"]) for row in selected],
                [float(row["best_trajectory_mse"]) for row in selected],
                marker="o",
                label=f"{model_name} init {omega_init:g}",
                alpha=0.75,
            )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("best final-stage trajectory MSE")
    plt.title("Stage 1 Residual Curriculum LR Sweep: Best Loss by Init/Model/LR")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "best_loss_by_init_model_lr.png")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for model_name in model_names:
        xs = []
        ys = []
        for lr in LEARNING_RATES:
            selected = [
                row
                for row in final_rows
                if row["model"] == model_name and float(row["learning_rate"]) == lr
            ]
            if selected:
                xs.append(lr)
                ys.append(
                    max(
                        min(
                            abs(float(row["best_omega_hat"]) - TRUE_OMEGA)
                            for row in selected
                        ),
                        1e-16,
                    ),
                )
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("|best final-stage omega_hat - true omega|")
    plt.title("Stage 1 Residual Curriculum LR Sweep: Omega Error by Model and LR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_error_by_model_and_lr.png")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for model_name in model_names:
        xs = []
        ys = []
        for lr in LEARNING_RATES:
            selected = [
                row
                for row in final_rows
                if row["model"] == model_name and float(row["learning_rate"]) == lr
            ]
            if selected:
                xs.append(lr)
                ys.append(min(float(row["residual_rms_test"]) for row in selected))
        plt.plot(xs, ys, marker="o", label=model_name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("residual RMS on test set")
    plt.title("Stage 1 Residual Curriculum LR Sweep: Residual RMS by Model and LR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_rms_by_model_and_lr.png")
    plt.close()


def _load_existing_rows(run_dir: Path) -> list[dict[str, Any]]:
    return _read_json(run_dir / "curriculum_stage_summary.json")["stages"]


def run_residual_curriculum_lr_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    variants = _residual_variants()
    jobs = []
    for variant in variants:
        for omega_init in OMEGA_INITIALIZATIONS:
            for learning_rate in LEARNING_RATES:
                run_name = (
                    f"{variant['model_name']}_omega_init_{_omega_label(omega_init)}"
                    f"_lr_{_lr_label(learning_rate)}"
                )
                run_dir = OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
                print(
                    (
                        f"queueing residual curriculum LR sweep: "
                        f"{variant['model_name']} omega_init={omega_init:g} "
                        f"lr={learning_rate:g}"
                    ),
                    flush=True,
                )
                if (run_dir / "curriculum_stage_summary.json").exists():
                    rows.extend(_load_existing_rows(run_dir))
                    continue
                jobs.append((variant, omega_init, learning_rate, run_dir))

    max_workers = min(8, os.cpu_count() or 1, max(1, len(jobs)))
    print(
        f"running {len(jobs)} remaining curricula with {max_workers} worker(s)",
        flush=True,
    )
    if jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_curriculum_job,
                    variant,
                    omega_init,
                    learning_rate,
                    run_dir,
                ): (variant["model_name"], omega_init, learning_rate)
                for variant, omega_init, learning_rate, run_dir in jobs
            }
            for future in as_completed(futures):
                model_name, omega_init, learning_rate = futures[future]
                result_rows = future.result()
                rows.extend(result_rows)
                final_stage = result_rows[-1]
                print(
                    (
                        f"completed {model_name} init={omega_init:g} "
                        f"lr={learning_rate:g}: best_final_stage_mse="
                        f"{final_stage['best_trajectory_mse']:.6e} "
                        f"best_omega={final_stage['best_omega_hat']:.9g}"
                    ),
                    flush=True,
                )
    _plot_summary(rows, OUTPUT_DIR)
    _write_summary(rows, OUTPUT_DIR)
    return rows


def _run_one_curriculum_job(
    variant: dict[str, Any],
    omega_init: float,
    learning_rate: float,
    run_dir: Path,
) -> list[dict[str, Any]]:
    return run_one_curriculum(
        variant=variant,
        omega_init=omega_init,
        learning_rate=learning_rate,
        output_dir=run_dir,
    )


def main() -> None:
    rows = run_residual_curriculum_lr_sweep()
    final_rows = _final_stage_rows(rows)
    best_by_model = {}
    for model_name in sorted({row["model"] for row in final_rows}):
        selected = [row for row in final_rows if row["model"] == model_name]
        best_by_model[model_name] = min(
            selected,
            key=lambda row: float(row["best_trajectory_mse"]),
        )
    print(json.dumps(best_by_model, indent=2))


if __name__ == "__main__":
    main()
