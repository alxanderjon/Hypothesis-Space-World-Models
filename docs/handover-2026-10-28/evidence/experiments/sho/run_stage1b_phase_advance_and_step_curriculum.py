"""Stage 1B phase-advance and step-count curriculum diagnostics.

Both experiments keep reconstruction loss only with the exact SHO decoder.
No omega labels, coefficient labels, auxiliary losses, or regularizers are used.
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

from experiments.sho.run_stage1b_capacity_and_window_sweeps import (
    CURRICULUM_HORIZONS,
    LOG_EVERY,
    WindowSHOData,
    _build_window_data_from_base,
    _evaluate_model,
    _save_omega_plot,
    _save_stage_metric_plots,
    _save_trajectory_examples,
    _write_csv,
    _write_json,
    _write_run_html,
)
from experiments.sho.run_stage1b_two_state_omega import (
    BATCH_SIZE,
    DELTA_T,
    NUM_TEST,
    NUM_TIME_STEPS,
    NUM_TRAIN,
    OMEGA_RANGE,
    RUN_DATE,
    SEED,
    _make_base_conditions,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/phase_advance_and_step_curriculum")
HIDDEN_SIZE = 32
WINDOW_SIZE = 20
LEARNING_RATE = 3e-3
MAX_EPOCHS_PER_STAGE = 50_000
STOPPING_PATIENCE = 10_000
STOPPING_MIN_DELTA = 0.0
SUCCESS_TRAJECTORY_MSE = 1e-15
STEP_COUNT_STAGES = [1, 2, 4, 8, 16, 32, 64, "full"]


class SHOPhaseAdvanceGRUWindowModel(nn.Module):
    """GRU window encoder that predicts local phase advance, then converts to omega."""

    def __init__(self, hidden_size: int, *, delta_t: float, omega_range: tuple[float, float]) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.delta_t = float(delta_t)
        self.theta_min = float(omega_range[0]) * self.delta_t
        self.theta_max = float(omega_range[1]) * self.delta_t
        self.gru = nn.GRU(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.output_head = nn.Linear(hidden_size, 3)

    def forward(
        self,
        observation_windows: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, hidden = self.gru(observation_windows)
        raw_outputs = self.output_head(hidden[-1])
        coefficients = raw_outputs[:, :2]
        theta_hat = self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(
            raw_outputs[:, 2],
        )
        omega_hat = theta_hat / self.delta_t
        a = coefficients[:, 0].unsqueeze(-1)
        b = coefficients[:, 1].unsqueeze(-1)
        phases = omega_hat.unsqueeze(-1) * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories, omega_hat


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _truncate_data(data: WindowSHOData, step_count: int | str) -> WindowSHOData:
    if step_count == "full":
        return data
    count = int(step_count)
    return WindowSHOData(
        observation_windows=data.observation_windows,
        times=data.times[:, :count],
        coefficients=data.coefficients,
        trajectories=data.trajectories[:, :count],
        omegas=data.omegas,
        periods=data.periods,
        delta_t=data.delta_t,
        num_periods=data.num_periods,
        window_size=data.window_size,
    )


def _train_stage(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: WindowSHOData,
    test_data: WindowSHOData,
    stage_index: int,
    stage_name: str,
    log_label: str,
) -> dict[str, Any]:
    num_train = train_data.observation_windows.shape[0]
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
                train_data.observation_windows[batch_indices],
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
            _, test_trajectories, _ = model(test_data.observation_windows, test_data.times)
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
                    f"{log_label} stage {stage_index:02d} {stage_name} "
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
        "stage_name": stage_name,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "stopped_because": stopped_because,
        "final_train_loss_before_restore": final_train_loss,
        "final_test_loss_before_restore": final_test_loss,
        "best_train_loss": best_train_loss,
        "best_test_loss": best_test_loss,
        **metrics,
    }


def _base_setup(run_name: str, experiment: str, curriculum: list[Any]) -> dict[str, Any]:
    theta_range = (OMEGA_RANGE[0] * DELTA_T, OMEGA_RANGE[1] * DELTA_T)
    return {
        "stage": "stage1B",
        "experiment": experiment,
        "run_name": run_name,
        "input": f"first {WINDOW_SIZE} observed states",
        "output": "(a, b, theta_hat_i) converted to omega_hat_i = theta_hat_i / delta_t",
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "theta_range": theta_range,
        "model": "GRU window encoder -> linear head -> (a, b, raw_theta)",
        "gru_hidden_size": HIDDEN_SIZE,
        "window_size": WINDOW_SIZE,
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
        "curriculum": curriculum,
        "random_seed": SEED,
        "run_date": RUN_DATE,
    }


def _run_period_curriculum(
    *,
    output_dir: Path,
    train_initial_conditions: torch.Tensor,
    train_omegas: torch.Tensor,
    test_initial_conditions: torch.Tensor,
    test_omegas: torch.Tensor,
) -> dict[str, Any]:
    run_name = "phase_advance_period_curriculum"
    if (output_dir / "metrics.json").exists():
        print(f"skipping completed {run_name}", flush=True)
        return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = SHOPhaseAdvanceGRUWindowModel(
        HIDDEN_SIZE,
        delta_t=DELTA_T,
        omega_range=OMEGA_RANGE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = _base_setup(run_name, "phase_advance_period_curriculum", CURRICULUM_HORIZONS)
    stage_rows = []
    for stage_index, horizon_periods in enumerate(CURRICULUM_HORIZONS, start=1):
        train_data = _build_window_data_from_base(
            train_initial_conditions,
            train_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
            window_size=WINDOW_SIZE,
        )
        test_data = _build_window_data_from_base(
            test_initial_conditions,
            test_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
            window_size=WINDOW_SIZE,
        )
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            stage_name=f"horizon_periods={horizon_periods:g}",
            log_label=run_name,
        )
        row["horizon_periods"] = horizon_periods
        stage_rows.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
            },
            output_dir / f"checkpoint_after_stage_{stage_index:02d}.pt",
        )

    final_data = _build_window_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=10.0,
        window_size=WINDOW_SIZE,
    )
    final_metrics, _, _, omega_hat = _evaluate_model(model, final_data)
    final_metrics.update(
        {
            "run_name": run_name,
            "experiment": "phase_advance_period_curriculum",
            "epochs_completed_total": sum(int(row["epochs_completed"]) for row in stage_rows),
            "final_stage_best_epoch": stage_rows[-1]["best_epoch"],
            "final_stage_stopped_because": stage_rows[-1]["stopped_because"],
            "final_stage_best_test_loss": stage_rows[-1]["best_test_loss"],
        },
    )
    _save_outputs(output_dir, run_name, setup, stage_rows, final_metrics, model, final_data, omega_hat)
    return final_metrics


def _run_step_count_curriculum(
    *,
    output_dir: Path,
    train_initial_conditions: torch.Tensor,
    train_omegas: torch.Tensor,
    test_initial_conditions: torch.Tensor,
    test_omegas: torch.Tensor,
) -> dict[str, Any]:
    run_name = "phase_advance_step_count_curriculum"
    if (output_dir / "metrics.json").exists():
        print(f"skipping completed {run_name}", flush=True)
        return json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = SHOPhaseAdvanceGRUWindowModel(
        HIDDEN_SIZE,
        delta_t=DELTA_T,
        omega_range=OMEGA_RANGE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = _base_setup(run_name, "phase_advance_step_count_curriculum", STEP_COUNT_STAGES)
    train_full = _build_window_data_from_base(
        train_initial_conditions,
        train_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=10.0,
        window_size=WINDOW_SIZE,
    )
    test_full = _build_window_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=10.0,
        window_size=WINDOW_SIZE,
    )
    stage_rows = []
    for stage_index, step_count in enumerate(STEP_COUNT_STAGES, start=1):
        train_data = _truncate_data(train_full, step_count)
        test_data = _truncate_data(test_full, step_count)
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            stage_name=f"step_count={step_count}",
            log_label=run_name,
        )
        row["step_count"] = step_count
        row["horizon_periods"] = (
            10.0 if step_count == "full" else float(step_count) / float(NUM_TIME_STEPS - 1) * 10.0
        )
        stage_rows.append(row)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stage_summary": row,
            },
            output_dir / f"checkpoint_after_stage_{stage_index:02d}.pt",
        )

    final_metrics, _, _, omega_hat = _evaluate_model(model, test_full)
    final_metrics.update(
        {
            "run_name": run_name,
            "experiment": "phase_advance_step_count_curriculum",
            "epochs_completed_total": sum(int(row["epochs_completed"]) for row in stage_rows),
            "final_stage_best_epoch": stage_rows[-1]["best_epoch"],
            "final_stage_stopped_because": stage_rows[-1]["stopped_because"],
            "final_stage_best_test_loss": stage_rows[-1]["best_test_loss"],
        },
    )
    _save_outputs(output_dir, run_name, setup, stage_rows, final_metrics, model, test_full, omega_hat)
    return final_metrics


def _save_outputs(
    output_dir: Path,
    run_name: str,
    setup: dict[str, Any],
    stage_rows: list[dict[str, Any]],
    final_metrics: dict[str, Any],
    model: nn.Module,
    final_data: WindowSHOData,
    omega_hat: torch.Tensor,
) -> None:
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", final_metrics)
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": stage_rows})
    _write_csv(output_dir / "curriculum_stage_summary.csv", stage_rows)
    _save_stage_metric_plots(stage_rows, output_dir, f"Stage 1B {run_name}")
    _save_omega_plot(
        output_dir / "final_predicted_vs_true_omega.png",
        omega_hat,
        final_data.omegas,
        f"Stage 1B {run_name}: Predicted vs True Omega",
    )
    _save_trajectory_examples(
        output_dir / "final_trajectory_examples.png",
        model,
        final_data,
        f"Stage 1B {run_name}",
    )
    _write_run_html(
        output_dir=output_dir,
        title=f"Stage 1B {run_name}",
        setup=setup,
        rows=stage_rows,
        final_metrics=final_metrics,
    )


def _best_previous_gru_window_reference() -> dict[str, Any]:
    candidates = []
    for path in Path("outputs/sho/stage1B").glob("**/metrics.json"):
        path_text = str(path)
        if "gru" not in path_text:
            continue
        if "gru32_fast_window_diagnostic" in path_text:
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "trajectory_mse" not in metrics:
            continue
        if metrics.get("completed_final_horizon") is False:
            continue
        candidates.append({"run_path": str(path.parent), **metrics})
    if not candidates:
        return {"run_name": "previous_best_gru_window", "trajectory_mse": float("nan")}
    best = min(candidates, key=lambda row: float(row["trajectory_mse"]))
    return {
        "run_name": "previous_best_gru_window",
        "run_path": best["run_path"],
        "trajectory_mse": best.get("trajectory_mse"),
        "coefficient_mse": best.get("coefficient_mse"),
        "omega_mse": best.get("omega_mse"),
        "omega_correlation": best.get("omega_correlation"),
        "mean_abs_omega_error": best.get("mean_abs_omega_error"),
        "max_omega_error": best.get("max_omega_error"),
    }


def _save_comparison_plots(rows: list[dict[str, Any]]) -> None:
    stage_rows = []
    for run_dir in [
        OUTPUT_DIR / "phase_advance_period_curriculum",
        OUTPUT_DIR / "phase_advance_step_count_curriculum",
    ]:
        stage_path = run_dir / "curriculum_stage_summary.json"
        if not stage_path.exists():
            continue
        stages = json.loads(stage_path.read_text(encoding="utf-8"))["stages"]
        for row in stages:
            stage_rows.append({"run_name": run_dir.name, **row})

    for filename, metric_key, ylabel in [
        ("trajectory_mse_vs_stage.png", "trajectory_mse", "trajectory MSE"),
        ("omega_mse_vs_stage.png", "omega_mse", "omega MSE"),
        ("omega_correlation_vs_stage.png", "omega_correlation", "omega correlation"),
    ]:
        plt.figure(figsize=(7, 4.5))
        for run_name in sorted({row["run_name"] for row in stage_rows}):
            run_rows = [row for row in stage_rows if row["run_name"] == run_name]
            plt.plot(
                [int(row["stage_index"]) for row in run_rows],
                [float(row[metric_key]) for row in run_rows],
                marker="o",
                label=run_name,
            )
        previous = next((row for row in rows if row["run_name"] == "previous_best_gru_window"), None)
        if previous and previous.get(metric_key) is not None:
            try:
                plt.axhline(float(previous[metric_key]), color="k", linestyle="--", label="previous best")
            except (TypeError, ValueError):
                pass
        plt.xlabel("curriculum stage")
        plt.ylabel(ylabel)
        if metric_key != "omega_correlation":
            plt.yscale("log")
        plt.title(f"Stage 1B Phase/Step Diagnostic: {ylabel}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "omega_correlation",
        "mean_abs_omega_error",
        "max_omega_error",
        "final_stage_best_epoch",
        "final_stage_stopped_because",
        "run_path",
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
            "trajectory_mse_vs_stage.png",
            "omega_mse_vs_stage.png",
            "omega_correlation_vs_stage.png",
        ]
    )
    links = "".join(
        f'<li><a href="{escape(row["run_name"])}/index.html">{escape(row["run_name"])}</a></li>'
        for row in rows
        if row["run_name"] != "previous_best_gru_window"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Phase Advance and Step Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Phase Advance and Step Curriculum</h1>
  <p>Reconstruction loss only. No omega or coefficient supervision.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
  <h2>Run Directories</h2>
  <ul>{links}</ul>
</body>
</html>
"""
    (OUTPUT_DIR / "comparison_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)

    rows = [
        _run_period_curriculum(
            output_dir=OUTPUT_DIR / "phase_advance_period_curriculum",
            train_initial_conditions=train_initial_conditions,
            train_omegas=train_omegas,
            test_initial_conditions=test_initial_conditions,
            test_omegas=test_omegas,
        ),
        _run_step_count_curriculum(
            output_dir=OUTPUT_DIR / "phase_advance_step_count_curriculum",
            train_initial_conditions=train_initial_conditions,
            train_omegas=train_omegas,
            test_initial_conditions=test_initial_conditions,
            test_omegas=test_omegas,
        ),
        _best_previous_gru_window_reference(),
    ]
    _write_csv(OUTPUT_DIR / "comparison_summary.csv", rows)
    _write_json(OUTPUT_DIR / "comparison_summary.json", {"runs": rows})
    _save_comparison_plots(rows)
    _write_summary_html(rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
