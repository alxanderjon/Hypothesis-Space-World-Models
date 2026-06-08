"""Run a Stage 1 time-horizon curriculum for learned global omega."""

from __future__ import annotations

import csv
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

from experiments.sho.stage0_exact_solution_decoder import _target_linear_weight_matrix
from hypothesis_space_world_models.evaluation.metrics import mean_squared_error
from hypothesis_space_world_models.models import SHOLearnedGlobalOmegaSolutionModel
from hypothesis_space_world_models.systems import generate_sho_data
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1/time_horizon_curriculum")
TRUE_OMEGA = 2.0
OMEGA_INITIALIZATIONS = [0.5, 1.0, 1.5, 2.5, 3.0]
CURRICULUM_HORIZONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _omega_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _evaluate_stage(model: torch.nn.Module, data: Any) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred_coefficients, pred_trajectories = model(
            data.initial_conditions,
            data.times,
        )
    coefficient_mse = mean_squared_error(pred_coefficients, data.coefficients)
    trajectory_mse = mean_squared_error(pred_trajectories, data.trajectories)
    return {
        "coefficient_mse": coefficient_mse,
        "trajectory_mse": trajectory_mse,
        "mean_abs_predicted_a_test": pred_coefficients[:, 0].abs().mean().item(),
        "mean_abs_predicted_b_test": pred_coefficients[:, 1].abs().mean().item(),
        "mean_abs_true_a_test": data.coefficients[:, 0].abs().mean().item(),
        "mean_abs_true_b_test": data.coefficients[:, 1].abs().mean().item(),
    }


def _linear_weight_mse(model: SHOLearnedGlobalOmegaSolutionModel) -> float:
    target = _target_linear_weight_matrix(TRUE_OMEGA)
    learned = model.encoder.weight.detach().cpu()
    return torch.mean((learned - target) ** 2).item()


def _train_curriculum_stage(
    *,
    model: SHOLearnedGlobalOmegaSolutionModel,
    optimizer: torch.optim.Optimizer,
    train_data: Any,
    test_data: Any,
    stage_index: int,
    horizon_periods: float,
    max_epochs_per_stage: int,
    batch_size: int,
    stopping_min_delta: float,
    stopping_patience: int,
    log_every: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    num_train = train_data.initial_conditions.shape[0]
    omega_start = float(model.omega_hat.detach().cpu())
    omega_error_start = omega_start - TRUE_OMEGA
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    best_significant_test_loss = float("inf")
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    stage_history: list[dict[str, Any]] = []

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

        train_loss = epoch_loss_sum / epoch_example_count
        model.eval()
        with torch.no_grad():
            _, test_pred_trajectories = model(
                test_data.initial_conditions,
                test_data.times,
            )
            test_loss = mse_loss(test_pred_trajectories, test_data.trajectories).item()

        best_train_loss = min(best_train_loss, train_loss)
        best_test_loss = min(best_test_loss, test_loss)
        omega_hat = float(model.omega_hat.detach().cpu())
        stage_history.append(
            {
                "stage_index": stage_index,
                "horizon_periods": horizon_periods,
                "epoch_within_stage": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "best_train_loss_so_far_in_stage": best_train_loss,
                "best_test_loss_so_far_in_stage": best_test_loss,
                "omega_hat": omega_hat,
                "omega_error": omega_hat - TRUE_OMEGA,
                "relative_omega_error": (omega_hat - TRUE_OMEGA) / TRUE_OMEGA,
                "raw_omega": float(model.raw_omega.detach().cpu()),
            }
        )

        if test_loss < best_significant_test_loss - stopping_min_delta:
            best_significant_test_loss = test_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if log_every > 0 and (
            epoch == 1 or epoch % log_every == 0 or epoch == max_epochs_per_stage
        ):
            print(
                (
                    f"stage {stage_index} horizon {horizon_periods:g} "
                    f"epoch {epoch:05d} train_loss: {train_loss:.6e} "
                    f"test_loss: {test_loss:.6e} omega_hat: {omega_hat:.8g}"
                ),
                flush=True,
            )

        if epochs_since_improvement >= stopping_patience:
            stopped_because = "patience_exhausted"
            break

    omega_end = float(model.omega_hat.detach().cpu())
    stage_metrics = _evaluate_stage(model, test_data)
    stage_summary = {
        "stage_index": stage_index,
        "horizon_periods": horizon_periods,
        "omega_hat_start": omega_start,
        "omega_hat_end": omega_end,
        "omega_error_start": omega_error_start,
        "omega_error_end": omega_end - TRUE_OMEGA,
        "relative_omega_error_end": (omega_end - TRUE_OMEGA) / TRUE_OMEGA,
        "omega_abs_error_moved_closer": abs(omega_end - TRUE_OMEGA)
        < abs(omega_error_start),
        "train_loss": stage_history[-1]["train_loss"] if stage_history else float("inf"),
        "test_loss": stage_history[-1]["test_loss"] if stage_history else float("inf"),
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        "coefficient_mse": stage_metrics["coefficient_mse"],
        "trajectory_mse": stage_metrics["trajectory_mse"],
        "mean_abs_predicted_a_test": stage_metrics["mean_abs_predicted_a_test"],
        "mean_abs_predicted_b_test": stage_metrics["mean_abs_predicted_b_test"],
        "mean_abs_true_a_test": stage_metrics["mean_abs_true_a_test"],
        "mean_abs_true_b_test": stage_metrics["mean_abs_true_b_test"],
        "linear_weight_mse": _linear_weight_mse(model),
        "epochs_completed": len(stage_history),
        "stopped_because": stopped_because,
    }
    return stage_summary, stage_history


def _write_init_index(run_dir: Path, setup: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    def table(data: dict[str, Any]) -> str:
        return "<table>" + "\n".join(
            "<tr>"
            f"<th>{escape(str(key))}</th>"
            f"<td><pre>{escape(json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value))}</pre></td>"
            "</tr>"
            for key, value in data.items()
        ) + "</table>"

    columns = [
        "stage_index",
        "horizon_periods",
        "omega_hat_start",
        "omega_hat_end",
        "omega_error_start",
        "omega_error_end",
        "test_loss",
        "coefficient_mse",
        "trajectory_mse",
        "epochs_completed",
        "stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Time-Horizon Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    pre {{ margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Stage 1 Time-Horizon Curriculum</h1>
  <h2>Setup</h2>
  {table(setup)}
  <h2>Stages</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")


def run_one_curriculum(omega_init: float, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = 0
    num_train = 512
    num_test = 128
    num_time_steps = 100
    batch_size = 512
    learning_rate = 1e-2
    max_epochs_per_stage = 100000
    stopping_min_delta = 1e-13
    stopping_patience = 10000
    log_every = 5000

    set_seed(seed)
    model = SHOLearnedGlobalOmegaSolutionModel(initial_omega_hat=omega_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    setup = {
        "comparison_mode": "stage1_time_horizon_curriculum",
        "model_type": "sho_linear_encoder_learned_global_omega",
        "encoder_type": "linear",
        "optimizer": "Adam",
        "optimizer_reset_between_stages": False,
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
    all_rows = []
    all_history = []
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
        stage_summary, stage_history = _train_curriculum_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            max_epochs_per_stage=max_epochs_per_stage,
            batch_size=batch_size,
            stopping_min_delta=stopping_min_delta,
            stopping_patience=stopping_patience,
            log_every=log_every,
        )
        stage_summary["omega_init"] = omega_init
        stage_summary["run_name"] = f"omega_init_{_omega_label(omega_init)}"
        all_rows.append(stage_summary)
        all_history.extend(stage_history)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": stage_summary,
            },
            output_dir / f"checkpoint_after_stage_{stage_index:02d}_{_omega_label(horizon_periods)}_periods.pt",
        )

    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "curriculum_history.json", {"history": all_history})
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": all_rows})
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    _write_init_index(output_dir, setup, all_rows)
    return all_rows


def _write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    columns = [
        "run_name",
        "omega_init",
        "stage_index",
        "horizon_periods",
        "omega_hat_start",
        "omega_hat_end",
        "omega_error_start",
        "omega_error_end",
        "omega_abs_error_moved_closer",
        "train_loss",
        "test_loss",
        "coefficient_mse",
        "trajectory_mse",
        "mean_abs_predicted_a_test",
        "mean_abs_predicted_b_test",
        "mean_abs_true_a_test",
        "mean_abs_true_b_test",
        "linear_weight_mse",
        "epochs_completed",
        "stopped_because",
    ]
    with (output_dir / "curriculum_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    images = [
        ("Omega vs Curriculum Stage", "omega_vs_curriculum_stage.png"),
        ("Test Loss vs Curriculum Stage", "test_loss_vs_curriculum_stage.png"),
        ("Omega Convergence by Stage", "omega_convergence_by_stage.png"),
    ]
    image_html = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(filename)}" alt="{escape(title)}"></section>'
        for title, filename in images
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1 Time-Horizon Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 32px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 980px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 1 Time-Horizon Curriculum</h1>
  <p>Linear encoder plus learned global omega. Adam state is preserved across curriculum stages.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {image_html}
</body>
</html>
"""
    (output_dir / "curriculum_summary.html").write_text(html, encoding="utf-8")


def _plot_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    plt.figure(figsize=(7.5, 5))
    for omega_init in OMEGA_INITIALIZATIONS:
        init_rows = [row for row in rows if float(row["omega_init"]) == omega_init]
        plt.plot(
            [row["horizon_periods"] for row in init_rows],
            [row["omega_hat_end"] for row in init_rows],
            marker="o",
            label=f"init {omega_init:g}",
        )
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xscale("log")
    plt.xlabel("curriculum horizon in periods")
    plt.ylabel("omega_hat at end of stage")
    plt.title("Stage 1 Curriculum: Omega vs Curriculum Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_vs_curriculum_stage.png")
    plt.close()

    plt.figure(figsize=(7.5, 5))
    for omega_init in OMEGA_INITIALIZATIONS:
        init_rows = [row for row in rows if float(row["omega_init"]) == omega_init]
        plt.plot(
            [row["horizon_periods"] for row in init_rows],
            [row["test_loss"] for row in init_rows],
            marker="o",
            label=f"init {omega_init:g}",
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("curriculum horizon in periods")
    plt.ylabel("test loss at end of stage")
    plt.title("Stage 1 Curriculum: Test Loss vs Curriculum Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_loss_vs_curriculum_stage.png")
    plt.close()

    plt.figure(figsize=(9, 5.5))
    for omega_init in OMEGA_INITIALIZATIONS:
        run_dir = output_dir / f"omega_init_{_omega_label(omega_init)}"
        history = _read_json(run_dir / "curriculum_history.json")["history"]
        global_steps = []
        omega_hats = []
        step_offset = 0
        previous_stage = None
        for entry in history:
            if previous_stage is not None and entry["stage_index"] != previous_stage:
                step_offset = global_steps[-1]
            previous_stage = entry["stage_index"]
            global_steps.append(step_offset + entry["epoch_within_stage"])
            omega_hats.append(entry["omega_hat"])
        plt.plot(global_steps, omega_hats, label=f"init {omega_init:g}", alpha=0.85)
    plt.axhline(TRUE_OMEGA, linestyle="--", color="black", label="true omega")
    plt.xlabel("cumulative curriculum epoch")
    plt.ylabel("omega_hat")
    plt.title("Stage 1 Curriculum: Omega Convergence by Stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "omega_convergence_by_stage.png")
    plt.close()


def run_curriculum_sweep() -> list[dict[str, Any]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for omega_init in OMEGA_INITIALIZATIONS:
        run_dir = OUTPUT_DIR / f"omega_init_{_omega_label(omega_init)}"
        print(f"running time-horizon curriculum omega_init={omega_init:g}", flush=True)
        if not (run_dir / "curriculum_stage_summary.json").exists():
            rows.extend(run_one_curriculum(omega_init, run_dir))
        else:
            rows.extend(_read_json(run_dir / "curriculum_stage_summary.json")["stages"])
    _plot_summary(rows, OUTPUT_DIR)
    _write_summary(rows, OUTPUT_DIR)
    return rows


def main() -> None:
    rows = run_curriculum_sweep()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
