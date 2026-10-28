"""Stage 1B GRU capacity and observation-window diagnostics.

Both diagnostics keep the exact SHO decoder and reconstruction-only objective.
No omega labels, coefficient labels, auxiliary losses, or regularizers are used.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
import math
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from experiments.sho.run_stage1b_gru_two_state_curriculum import (
    CURRICULUM_HORIZONS,
    LEARNING_RATE,
    LOG_EVERY,
    MAX_EPOCHS_PER_STAGE,
    STOPPING_MIN_DELTA,
    STOPPING_PATIENCE,
    SUCCESS_TRAJECTORY_MSE,
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
    _metrics,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


GRU_SWEEP_OUTPUT_DIR = Path("outputs/sho/stage1B/gru_hidden_size_sweep")
WINDOW_SWEEP_OUTPUT_DIR = Path("outputs/sho/stage1B/window_size_sweep")
GRU_HIDDEN_SIZES = [16, 32, 64, 128]
WINDOW_SIZES = [2, 3, 4, 5, 6, 8, 10]


@dataclass(frozen=True)
class WindowSHOData:
    """Noise-free variable-omega SHO trajectories plus an observation window."""

    observation_windows: torch.Tensor
    times: torch.Tensor
    coefficients: torch.Tensor
    trajectories: torch.Tensor
    omegas: torch.Tensor
    periods: torch.Tensor
    delta_t: float
    num_periods: float
    window_size: int


class SHOWindowGRUModel(nn.Module):
    """Encode an N-state phase-space observation window with a GRU."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
        omega_hat = F.softplus(raw_outputs[:, 2]) + 1e-8
        a = coefficients[:, 0].unsqueeze(-1)
        b = coefficients[:, 1].unsqueeze(-1)
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


def _build_window_data_from_base(
    initial_conditions: torch.Tensor,
    omegas: torch.Tensor,
    *,
    num_time_steps: int,
    num_periods: float,
    window_size: int,
    delta_t: float = DELTA_T,
) -> WindowSHOData:
    periods = math.tau / omegas
    time_fractions = torch.linspace(
        0.0,
        num_periods,
        num_time_steps,
        device=initial_conditions.device,
    )
    times = time_fractions.unsqueeze(0) * periods.unsqueeze(-1)
    trajectories, _ = _state_at_time(initial_conditions, omegas, times)

    observation_times = (
        torch.arange(
            window_size,
            device=initial_conditions.device,
            dtype=initial_conditions.dtype,
        )
        * float(delta_t)
    )
    observation_times = observation_times.unsqueeze(0).expand(initial_conditions.shape[0], -1)
    positions, velocities = _state_at_time(initial_conditions, omegas, observation_times)
    observation_windows = torch.stack((positions, velocities), dim=-1)
    coefficients = torch.stack(
        (initial_conditions[:, 0], initial_conditions[:, 1] / omegas),
        dim=-1,
    )
    return WindowSHOData(
        observation_windows=observation_windows,
        times=times,
        coefficients=coefficients,
        trajectories=trajectories,
        omegas=omegas,
        periods=periods,
        delta_t=delta_t,
        num_periods=num_periods,
        window_size=window_size,
    )


def _evaluate_model(
    model: nn.Module,
    data: WindowSHOData,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        coefficients, trajectories, omega_hat = model(data.observation_windows, data.times)
    return (
        _metrics(
            coefficients=coefficients,
            trajectories=trajectories,
            omega_hat=omega_hat,
            data=data,
        ),
        coefficients,
        trajectories,
        omega_hat,
    )


def _train_stage(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: WindowSHOData,
    test_data: WindowSHOData,
    stage_index: int,
    horizon_periods: float,
    log_label: str,
    max_epochs_per_stage: int = MAX_EPOCHS_PER_STAGE,
    stopping_min_delta: float = STOPPING_MIN_DELTA,
    stopping_patience: int = STOPPING_PATIENCE,
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

    for epoch in range(1, max_epochs_per_stage + 1):
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
            _, test_trajectories, _ = model(
                test_data.observation_windows,
                test_data.times,
            )
            final_test_loss = mse_loss(test_trajectories, test_data.trajectories).item()

        if final_test_loss < best_test_loss - stopping_min_delta:
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
                    f"{log_label} stage {stage_index:02d} horizon {horizon_periods:g} "
                    f"epoch {epoch:06d} train_loss={final_train_loss:.6e} "
                    f"test_loss={final_test_loss:.6e} best_test={best_test_loss:.6e}"
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

    metrics, _, _, _ = _evaluate_model(model, test_data)
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
        **metrics,
    }


def _save_omega_plot(
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
    model: nn.Module,
    test_data: WindowSHOData,
    title: str,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        _, trajectories, _ = model(test_data.observation_windows, test_data.times)
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


def _save_stage_metric_plots(rows: list[dict[str, Any]], output_dir: Path, title: str) -> None:
    horizons = [float(row["horizon_periods"]) for row in rows]
    for filename, key, ylabel in [
        ("trajectory_mse_vs_horizon.png", "trajectory_mse", "trajectory MSE"),
        ("coefficient_mse_vs_horizon.png", "coefficient_mse", "coefficient MSE"),
        ("omega_mse_vs_horizon.png", "omega_mse", "omega MSE"),
        ("omega_correlation_vs_horizon.png", "omega_correlation", "omega correlation"),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(horizons, [float(row[key]) for row in rows], marker="o")
        plt.xlabel("horizon in periods")
        plt.ylabel(ylabel)
        if key != "omega_correlation":
            plt.yscale("log")
        plt.title(f"{title}: {ylabel}")
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def _write_run_html(
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
        "best_epoch",
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
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in [
            "trajectory_mse_vs_horizon.png",
            "coefficient_mse_vs_horizon.png",
            "omega_mse_vs_horizon.png",
            "omega_correlation_vs_horizon.png",
            "final_predicted_vs_true_omega.png",
            "final_trajectory_examples.png",
        ]
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


def _write_summary_html(
    *,
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    plot_files: list[str],
    links: list[tuple[str, str]],
) -> None:
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in plot_files
    )
    link_html = "".join(
        f'<li><a href="{escape(href)}">{escape(label)}</a></li>' for label, href in links
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
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plot_html}
  <h2>Run Directories</h2>
  <ul>{link_html}</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _run_curriculum(
    *,
    output_dir: Path,
    hidden_size: int,
    window_size: int,
    label: str,
    experiment_name: str,
    learning_rate: float = LEARNING_RATE,
    max_epochs_per_stage: int = MAX_EPOCHS_PER_STAGE,
    stopping_min_delta: float = STOPPING_MIN_DELTA,
    stopping_patience: int = STOPPING_PATIENCE,
    curriculum_early_stopping: bool = False,
) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"skipping completed Stage 1B run: {label}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    model = SHOWindowGRUModel(hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    setup = {
        "stage": "stage1B",
        "experiment": experiment_name,
        "run_name": label,
        "input": f"first {window_size} observed states [(x0,v0), ..., (x{window_size - 1},v{window_size - 1})]",
        "window_size": window_size,
        "output": "(a, b, omega_hat_i)",
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "model": "GRU window encoder -> linear head -> (a, b, raw_omega)",
        "gru_hidden_size": hidden_size,
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "loss": "trajectory reconstruction only",
        "omega_supervision": False,
        "coefficient_supervision": False,
        "auxiliary_losses": False,
        "regularization": False,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "batch_size": BATCH_SIZE,
        "max_epochs_per_stage": max_epochs_per_stage,
        "stopping_min_delta": stopping_min_delta,
        "stopping_patience": stopping_patience,
        "curriculum_early_stopping": curriculum_early_stopping,
        "curriculum_worsen_omega_factor": 5.0 if curriculum_early_stopping else "N/A",
        "curriculum_worsen_trajectory_factor": 10.0 if curriculum_early_stopping else "N/A",
        "curriculum_correlation_drop_rule": (
            "stop if omega correlation drops below 0.95 after exceeding 0.99"
            if curriculum_early_stopping
            else "N/A"
        ),
        "curriculum_plateau_rule": (
            "stop after 3 stages without trajectory-MSE or omega-MSE improvement"
            if curriculum_early_stopping
            else "N/A"
        ),
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "curriculum_horizons_periods": CURRICULUM_HORIZONS,
        "random_seed": SEED,
        "run_date": RUN_DATE,
    }
    stage_rows = []
    curriculum_stopped_because = "completed_all_horizons"
    best_trajectory_mse = float("inf")
    best_omega_mse = float("inf")
    no_trajectory_improvement_stages = 0
    no_omega_improvement_stages = 0
    omega_correlation_previously_high = False
    for stage_index, horizon_periods in enumerate(CURRICULUM_HORIZONS, start=1):
        train_data = _build_window_data_from_base(
            train_initial_conditions,
            train_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
            window_size=window_size,
        )
        test_data = _build_window_data_from_base(
            test_initial_conditions,
            test_omegas,
            num_time_steps=NUM_TIME_STEPS,
            num_periods=horizon_periods,
            window_size=window_size,
        )
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            log_label=label,
            max_epochs_per_stage=max_epochs_per_stage,
            stopping_min_delta=stopping_min_delta,
            stopping_patience=stopping_patience,
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
        if curriculum_early_stopping:
            trajectory_mse = float(row["trajectory_mse"])
            omega_mse = float(row["omega_mse"])
            omega_correlation = float(row["omega_correlation"])
            had_previous_stage = stage_index > 1
            if had_previous_stage and omega_mse > 5.0 * best_omega_mse:
                curriculum_stopped_because = "curriculum_worsened_omega"
                break
            if had_previous_stage and trajectory_mse > 10.0 * best_trajectory_mse:
                curriculum_stopped_because = "curriculum_worsened_trajectory"
                break
            if omega_correlation_previously_high and omega_correlation < 0.95:
                curriculum_stopped_because = "curriculum_worsened_correlation"
                break

            if trajectory_mse < best_trajectory_mse:
                best_trajectory_mse = trajectory_mse
                no_trajectory_improvement_stages = 0
            else:
                no_trajectory_improvement_stages += 1
            if omega_mse < best_omega_mse:
                best_omega_mse = omega_mse
                no_omega_improvement_stages = 0
            else:
                no_omega_improvement_stages += 1
            if omega_correlation > 0.99:
                omega_correlation_previously_high = True
            if no_trajectory_improvement_stages >= 3 and no_omega_improvement_stages >= 3:
                curriculum_stopped_because = "curriculum_plateaued"
                break

    final_test_data = _build_window_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=float(stage_rows[-1]["horizon_periods"]),
        window_size=window_size,
    )
    final_metrics, _, _, omega_hat = _evaluate_model(model, final_test_data)
    final_metrics.update(
        {
            "run_name": label,
            "hidden_size": hidden_size,
            "window_size": window_size,
            "epochs_completed_total": sum(int(row["epochs_completed"]) for row in stage_rows),
            "horizons_completed": len(stage_rows),
            "final_horizon_periods": float(stage_rows[-1]["horizon_periods"]),
            "completed_final_horizon": float(stage_rows[-1]["horizon_periods"]) == CURRICULUM_HORIZONS[-1],
            "curriculum_stopped_because": curriculum_stopped_because,
            "final_stage_best_epoch": stage_rows[-1]["best_epoch"],
            "final_stage_stopped_because": stage_rows[-1]["stopped_because"],
            "final_stage_best_test_loss": stage_rows[-1]["best_test_loss"],
            "recovered_successfully": (
                final_metrics["trajectory_mse"] < SUCCESS_TRAJECTORY_MSE
                and final_metrics["max_omega_error"] < 1e-6
            ),
        },
    )
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", final_metrics)
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": stage_rows})
    _write_csv(output_dir / "curriculum_stage_summary.csv", stage_rows)
    _save_stage_metric_plots(stage_rows, output_dir, f"Stage 1B {label}")
    _save_omega_plot(
        output_dir / "final_predicted_vs_true_omega.png",
        omega_hat,
        final_test_data.omegas,
        f"Stage 1B {label}: Predicted vs True Omega",
    )
    _save_trajectory_examples(
        output_dir / "final_trajectory_examples.png",
        model,
        final_test_data,
        f"Stage 1B {label}",
    )
    _write_run_html(
        output_dir=output_dir,
        title=f"Stage 1B {label}",
        setup=setup,
        rows=stage_rows,
        final_metrics=final_metrics,
    )
    return final_metrics


def _save_bar_summary_plots(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    title_prefix: str,
) -> None:
    x_values = [row[x_key] for row in rows]
    for filename, metric_key, ylabel, log_scale in [
        ("trajectory_mse_vs_hidden_size.png" if x_key == "hidden_size" else "trajectory_mse_vs_window_size.png", "trajectory_mse", "trajectory MSE", True),
        ("omega_mse_vs_hidden_size.png" if x_key == "hidden_size" else "omega_mse_vs_window_size.png", "omega_mse", "omega MSE", True),
        ("omega_corr_vs_hidden_size.png" if x_key == "hidden_size" else "omega_corr_vs_window_size.png", "omega_correlation", "omega correlation", False),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(x_values, [float(row[metric_key]) for row in rows], marker="o")
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale("log")
        plt.title(f"{title_prefix}: {ylabel}")
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def run_gru_hidden_size_sweep() -> tuple[list[dict[str, Any]], int]:
    GRU_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for hidden_size in GRU_HIDDEN_SIZES:
        label = f"gru_hidden_{hidden_size}"
        print(f"running Stage 1B GRU hidden-size sweep: {label}", flush=True)
        rows.append(
            _run_curriculum(
                output_dir=GRU_SWEEP_OUTPUT_DIR / label,
                hidden_size=hidden_size,
                window_size=2,
                label=label,
                experiment_name="gru_hidden_size_sweep",
            ),
        )
    _write_csv(GRU_SWEEP_OUTPUT_DIR / "gru_hidden_size_sweep_summary.csv", rows)
    _write_json(GRU_SWEEP_OUTPUT_DIR / "gru_hidden_size_sweep_summary.json", {"runs": rows})
    _save_bar_summary_plots(
        output_dir=GRU_SWEEP_OUTPUT_DIR,
        rows=rows,
        x_key="hidden_size",
        x_label="GRU hidden size",
        title_prefix="Stage 1B GRU Hidden-Size Sweep",
    )
    _write_summary_html(
        path=GRU_SWEEP_OUTPUT_DIR / "gru_hidden_size_sweep_summary.html",
        title="Stage 1B GRU Hidden-Size Sweep",
        rows=rows,
        columns=[
            "run_name",
            "hidden_size",
            "trajectory_mse",
            "coefficient_mse",
            "omega_mse",
            "omega_correlation",
            "mean_abs_omega_error",
            "epochs_completed_total",
            "final_stage_best_epoch",
            "final_stage_stopped_because",
        ],
        plot_files=[
            "trajectory_mse_vs_hidden_size.png",
            "omega_mse_vs_hidden_size.png",
            "omega_corr_vs_hidden_size.png",
        ],
        links=[(f"hidden size {row['hidden_size']}", f"{row['run_name']}/index.html") for row in rows],
    )
    best_row = min(rows, key=lambda row: float(row["trajectory_mse"]))
    return rows, int(best_row["hidden_size"])


def run_window_size_sweep(hidden_size: int) -> list[dict[str, Any]]:
    WINDOW_SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for window_size in WINDOW_SIZES:
        label = f"gru_hidden_{hidden_size}_window_{window_size}"
        print(f"running Stage 1B window-size sweep: {label}", flush=True)
        rows.append(
            _run_curriculum(
                output_dir=WINDOW_SWEEP_OUTPUT_DIR / label,
                hidden_size=hidden_size,
                window_size=window_size,
                label=label,
                experiment_name="window_size_sweep",
            ),
        )
    _write_csv(WINDOW_SWEEP_OUTPUT_DIR / "window_size_sweep_summary.csv", rows)
    _write_json(WINDOW_SWEEP_OUTPUT_DIR / "window_size_sweep_summary.json", {"runs": rows})
    _save_bar_summary_plots(
        output_dir=WINDOW_SWEEP_OUTPUT_DIR,
        rows=rows,
        x_key="window_size",
        x_label="observation window size in states",
        title_prefix="Stage 1B Window-Size Sweep",
    )
    _write_summary_html(
        path=WINDOW_SWEEP_OUTPUT_DIR / "window_size_sweep_summary.html",
        title=f"Stage 1B Window-Size Sweep with GRU Hidden Size {hidden_size}",
        rows=rows,
        columns=[
            "run_name",
            "hidden_size",
            "window_size",
            "trajectory_mse",
            "coefficient_mse",
            "omega_mse",
            "omega_correlation",
            "mean_abs_omega_error",
            "epochs_completed_total",
            "final_stage_best_epoch",
            "final_stage_stopped_because",
        ],
        plot_files=[
            "trajectory_mse_vs_window_size.png",
            "omega_mse_vs_window_size.png",
            "omega_corr_vs_window_size.png",
        ],
        links=[(f"window size {row['window_size']}", f"{row['run_name']}/index.html") for row in rows],
    )
    return rows


def main() -> None:
    hidden_rows, best_hidden_size = run_gru_hidden_size_sweep()
    window_rows = run_window_size_sweep(best_hidden_size)
    print(
        json.dumps(
            {
                "gru_hidden_size_sweep": hidden_rows,
                "best_hidden_size_by_trajectory_mse": best_hidden_size,
                "window_size_sweep": window_rows,
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
