"""Stage 1B: recover trajectory-specific omega from two phase-space states."""

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

from hypothesis_space_world_models.evaluation.metrics import mean_squared_error
from hypothesis_space_world_models.models import SHOTwoStateTrajectoryOmegaLinearModel
from hypothesis_space_world_models.training import mse_loss, set_seed


ANALYTIC_OUTPUT_DIR = Path("outputs/sho/stage1B/analytic_two_state_omega")
LINEAR_OUTPUT_DIR = Path("outputs/sho/stage1B/linear_two_state_omega_curriculum")
RESIDUAL_OUTPUT_DIR = Path("outputs/sho/stage1B/residual_two_state_omega_curriculum")
RUN_DATE = "2026-06-08"
OMEGA_RANGE = (1.5, 2.5)
DELTA_T = 0.25
CURRICULUM_HORIZONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
SUCCESS_TRAJECTORY_MSE = 1e-15
SUCCESS_OMEGA_TOLERANCE = 1e-6
STOPPING_MIN_DELTA = 0.0
STOPPING_PATIENCE = 5000
MAX_EPOCHS_PER_STAGE = 50000
BATCH_SIZE = 512
LEARNING_RATE = 1e-2
NUM_TRAIN = 512
NUM_TEST = 128
NUM_TIME_STEPS = 100
SEED = 0


def _inverse_softplus(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value))))


def _make_residual_branch(hidden_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    input_dim = 4
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 3))
    return nn.Sequential(*layers)


class SHOTwoStateTrajectoryOmegaLinearResidualModel(nn.Module):
    """Learn (a, b, omega_i) with a linear path plus a nonlinear residual."""

    def __init__(
        self,
        hidden_dims: list[int],
        initial_omega_hat: float = 2.0,
    ) -> None:
        super().__init__()
        if initial_omega_hat <= 0:
            raise ValueError("initial_omega_hat must be positive")
        self.hidden_dims = list(hidden_dims)
        self.linear_encoder = nn.Linear(4, 3)
        self.residual_encoder = _make_residual_branch(self.hidden_dims)
        self._initialize_omega_bias(initial_omega_hat)

    def _initialize_omega_bias(self, initial_omega_hat: float) -> None:
        with torch.no_grad():
            self.linear_encoder.bias[2].fill_(
                _inverse_softplus(initial_omega_hat - 1e-8).item(),
            )

    def forward(
        self,
        two_state_inputs: torch.Tensor,
        times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_outputs = self.linear_encoder(two_state_inputs) + self.residual_encoder(
            two_state_inputs,
        )
        coefficients = raw_outputs[..., :2]
        omega_hat = F.softplus(raw_outputs[..., 2]) + 1e-8
        a = coefficients[..., 0].unsqueeze(-1)
        b = coefficients[..., 1].unsqueeze(-1)
        phases = omega_hat.unsqueeze(-1) * times
        trajectories = a * torch.cos(phases) + b * torch.sin(phases)
        return coefficients, trajectories, omega_hat


def _residual_variants() -> list[dict[str, Any]]:
    return [
        {"run_name": "linear_tiny_residual", "hidden_dims": [4], "learning_rate": 1e-2},
        {
            "run_name": "linear_moderate_residual",
            "hidden_dims": [8, 8],
            "learning_rate": 1e-2,
        },
        {
            "run_name": "linear_large_residual",
            "hidden_dims": [32, 32],
            "learning_rate": 3e-2,
        },
    ]


@dataclass(frozen=True)
class TwoStateSHOData:
    """Noise-free SHO trajectories with per-trajectory omega."""

    two_state_inputs: torch.Tensor
    initial_conditions: torch.Tensor
    state1: torch.Tensor
    times: torch.Tensor
    coefficients: torch.Tensor
    trajectories: torch.Tensor
    velocities: torch.Tensor
    omegas: torch.Tensor
    periods: torch.Tensor
    delta_t: float
    num_periods: float


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _state_at_time(
    initial_conditions: torch.Tensor,
    omegas: torch.Tensor,
    times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = initial_conditions[:, 0].unsqueeze(-1)
    v0 = initial_conditions[:, 1].unsqueeze(-1)
    omega = omegas.unsqueeze(-1)
    a = x0
    b = v0 / omega
    phases = omega * times
    positions = a * torch.cos(phases) + b * torch.sin(phases)
    velocities = -a * omega * torch.sin(phases) + b * omega * torch.cos(phases)
    return positions, velocities


def _make_base_conditions(
    num_trajectories: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    initial_conditions = 2.0 * torch.rand(
        num_trajectories,
        2,
        generator=generator,
        device=device,
        dtype=dtype,
    ) - 1.0
    omega_min, omega_max = OMEGA_RANGE
    omegas = omega_min + (omega_max - omega_min) * torch.rand(
        num_trajectories,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return initial_conditions, omegas


def _build_data_from_base(
    initial_conditions: torch.Tensor,
    omegas: torch.Tensor,
    *,
    num_time_steps: int,
    num_periods: float,
    delta_t: float = DELTA_T,
) -> TwoStateSHOData:
    periods = math.tau / omegas
    time_fractions = torch.linspace(
        0.0,
        num_periods,
        num_time_steps,
        device=initial_conditions.device,
    )
    times = time_fractions.unsqueeze(0) * periods.unsqueeze(-1)
    trajectories, velocities = _state_at_time(initial_conditions, omegas, times)
    delta_times = torch.full(
        (initial_conditions.shape[0], 1),
        float(delta_t),
        device=initial_conditions.device,
    )
    x1, v1 = _state_at_time(initial_conditions, omegas, delta_times)
    state1 = torch.cat((x1, v1), dim=-1)
    two_state_inputs = torch.cat((initial_conditions, state1), dim=-1)
    coefficients = torch.stack(
        (initial_conditions[:, 0], initial_conditions[:, 1] / omegas),
        dim=-1,
    )
    return TwoStateSHOData(
        two_state_inputs=two_state_inputs,
        initial_conditions=initial_conditions,
        state1=state1,
        times=times,
        coefficients=coefficients,
        trajectories=trajectories,
        velocities=velocities,
        omegas=omegas,
        periods=periods,
        delta_t=delta_t,
        num_periods=num_periods,
    )


def generate_two_state_sho_data(
    num_trajectories: int,
    *,
    num_time_steps: int,
    num_periods: float,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> TwoStateSHOData:
    initial_conditions, omegas = _make_base_conditions(
        num_trajectories,
        seed=seed,
        dtype=dtype,
    )
    return _build_data_from_base(
        initial_conditions,
        omegas,
        num_time_steps=num_time_steps,
        num_periods=num_periods,
    )


def analytic_two_state_omega(
    two_state_inputs: torch.Tensor,
    *,
    denominator_eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = two_state_inputs[:, 0]
    v0 = two_state_inputs[:, 1]
    x1 = two_state_inputs[:, 2]
    v1 = two_state_inputs[:, 3]
    denominator = x0.square() - x1.square()
    ratio = (v1.square() - v0.square()) / denominator
    valid = (
        torch.isfinite(ratio)
        & (denominator.abs() > denominator_eps)
        & (ratio > 0.0)
    )
    omega_hat = torch.sqrt(torch.clamp(ratio, min=0.0))
    return omega_hat, valid


def decode_with_trajectory_omegas(
    coefficients: torch.Tensor,
    times: torch.Tensor,
    omegas: torch.Tensor,
) -> torch.Tensor:
    a = coefficients[:, 0].unsqueeze(-1)
    b = coefficients[:, 1].unsqueeze(-1)
    phases = omegas.unsqueeze(-1) * times
    return a * torch.cos(phases) + b * torch.sin(phases)


def _omega_correlation(omega_hat: torch.Tensor, omegas: torch.Tensor) -> float:
    pred = omega_hat.detach().cpu()
    true = omegas.detach().cpu()
    if pred.numel() < 2:
        return float("nan")
    pred_centered = pred - pred.mean()
    true_centered = true - true.mean()
    denominator = torch.sqrt(
        pred_centered.square().sum() * true_centered.square().sum(),
    )
    if denominator <= 0:
        return float("nan")
    return (pred_centered * true_centered).sum().div(denominator).item()


def _metrics(
    *,
    coefficients: torch.Tensor,
    trajectories: torch.Tensor,
    omega_hat: torch.Tensor,
    data: TwoStateSHOData,
) -> dict[str, float]:
    omega_error = omega_hat - data.omegas
    return {
        "trajectory_mse": mean_squared_error(trajectories, data.trajectories),
        "coefficient_mse": mean_squared_error(coefficients, data.coefficients),
        "omega_mse": mean_squared_error(omega_hat, data.omegas),
        "mean_omega_error": omega_error.mean().item(),
        "mean_abs_omega_error": omega_error.abs().mean().item(),
        "max_omega_error": omega_error.abs().max().item(),
        "omega_correlation": _omega_correlation(omega_hat, data.omegas),
    }


def _write_metric_table_html(
    output_dir: Path,
    title: str,
    setup: dict[str, Any],
    metrics: dict[str, Any],
    plots: list[str],
) -> None:
    setup_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td><pre>{escape(str(value))}</pre></td></tr>"
        for key, value in setup.items()
    )
    metric_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    plot_html = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in plots
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
  <h2>Metrics</h2>
  <table>{metric_rows}</table>
  {plot_html}
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _save_omega_plots(
    output_dir: Path,
    omega_hat: torch.Tensor,
    true_omega: torch.Tensor,
    title_prefix: str,
) -> None:
    omega_hat_np = omega_hat.detach().cpu()
    true_omega_np = true_omega.detach().cpu()
    error_np = omega_hat_np - true_omega_np

    plt.figure(figsize=(6, 6))
    plt.scatter(true_omega_np, omega_hat_np, s=16, alpha=0.75)
    min_value = min(true_omega_np.min().item(), omega_hat_np.min().item())
    max_value = max(true_omega_np.max().item(), omega_hat_np.max().item())
    plt.plot([min_value, max_value], [min_value, max_value], "k--", label="y = x")
    plt.xlabel("true omega")
    plt.ylabel("predicted omega")
    plt.title(f"{title_prefix}: Predicted vs True Omega")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_true_omega.png")
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(error_np, bins=40)
    plt.xlabel("omega_hat - omega_true")
    plt.ylabel("count")
    plt.title(f"{title_prefix}: Omega Error Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "omega_error_histogram.png")
    plt.close()


def run_analytic_sanity_check() -> dict[str, Any]:
    output_dir = ANALYTIC_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_two_state_sho_data(
        1024,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=10.0,
        seed=SEED,
        dtype=torch.float64,
    )
    omega_hat_all, valid = analytic_two_state_omega(data.two_state_inputs)
    filtered_data = TwoStateSHOData(
        two_state_inputs=data.two_state_inputs[valid],
        initial_conditions=data.initial_conditions[valid],
        state1=data.state1[valid],
        times=data.times[valid],
        coefficients=data.coefficients[valid],
        trajectories=data.trajectories[valid],
        velocities=data.velocities[valid],
        omegas=data.omegas[valid],
        periods=data.periods[valid],
        delta_t=data.delta_t,
        num_periods=data.num_periods,
    )
    omega_hat = omega_hat_all[valid]
    coefficients = torch.stack(
        (
            filtered_data.initial_conditions[:, 0],
            filtered_data.initial_conditions[:, 1] / omega_hat,
        ),
        dim=-1,
    )
    trajectories = decode_with_trajectory_omegas(
        coefficients,
        filtered_data.times,
        omega_hat,
    )
    metrics = _metrics(
        coefficients=coefficients,
        trajectories=trajectories,
        omega_hat=omega_hat,
        data=filtered_data,
    )
    metrics.update(
        {
            "num_generated_trajectories": int(data.omegas.numel()),
            "num_valid_trajectories": int(valid.sum().item()),
            "num_excluded_trajectories": int((~valid).sum().item()),
            "valid_fraction": valid.float().mean().item(),
        },
    )
    setup = {
        "stage": "stage1B",
        "experiment": "analytic_two_state_omega",
        "input": "(x0, v0, x1, v1)",
        "delta_t": DELTA_T,
        "omega_range": OMEGA_RANGE,
        "omega_formula": "sqrt((v1^2 - v0^2) / (x0^2 - x1^2))",
        "denominator_eps": 1e-10,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
    }
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", metrics)
    _save_omega_plots(output_dir, omega_hat, filtered_data.omegas, "Stage 1B Analytic")
    _write_metric_table_html(
        output_dir,
        "Stage 1B Analytic Two-State Omega",
        setup,
        metrics,
        ["predicted_vs_true_omega.png", "omega_error_histogram.png"],
    )
    return metrics


def _clone_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _evaluate_model(
    model: nn.Module,
    data: TwoStateSHOData,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        coefficients, trajectories, omega_hat = model(data.two_state_inputs, data.times)
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
    train_data: TwoStateSHOData,
    test_data: TwoStateSHOData,
    stage_index: int,
    horizon_periods: float,
    log_label: str = "",
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

        if epoch == 1 or epoch % 10000 == 0:
            print(
                (
                    f"{log_label}stage {stage_index} horizon {horizon_periods:g} "
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

    metrics, _, _, omega_hat = _evaluate_model(model, test_data)
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
        "best_omega_mean": omega_hat.mean().item(),
        **metrics,
    }


def run_linear_curriculum() -> dict[str, Any]:
    output_dir = LINEAR_OUTPUT_DIR / f"{RUN_DATE}_linear_two_state_omega"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    model = SHOTwoStateTrajectoryOmegaLinearModel(initial_omega_hat=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    setup = {
        "stage": "stage1B",
        "experiment": "linear_two_state_omega_curriculum",
        "input": "(x0, v0, x1, v1)",
        "delta_t": DELTA_T,
        "omega_range": OMEGA_RANGE,
        "model": "(x0, v0, x1, v1) -> linear encoder -> (a, b, omega_hat_i)",
        "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
        "stopping_monitor": "test_loss",
        "stopping_min_delta": STOPPING_MIN_DELTA,
        "stopping_patience": STOPPING_PATIENCE,
        "curriculum_horizons_periods": CURRICULUM_HORIZONS,
        "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
        "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
        "random_seed": SEED,
    }
    stage_rows = []
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
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            log_label="linear ",
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

    final_test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    )
    final_metrics, _, _, omega_hat = _evaluate_model(model, final_test_data)
    final_metrics["recovered_successfully"] = (
        final_metrics["trajectory_mse"] < SUCCESS_TRAJECTORY_MSE
        and final_metrics["max_omega_error"] < SUCCESS_OMEGA_TOLERANCE
    )
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", final_metrics)
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": stage_rows})
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _save_omega_plots(output_dir, omega_hat, final_test_data.omegas, "Stage 1B Linear")
    _write_metric_table_html(
        output_dir,
        "Stage 1B Linear Two-State Omega Curriculum",
        setup,
        final_metrics,
        ["predicted_vs_true_omega.png", "omega_error_histogram.png"],
    )
    _write_summary_csv(output_dir, stage_rows)
    return final_metrics


def _run_curriculum_for_model(
    *,
    model: nn.Module,
    output_dir: Path,
    setup: dict[str, Any],
    learning_rate: float,
    title: str,
    plot_title_prefix: str,
    log_label: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_initial_conditions, train_omegas = _make_base_conditions(NUM_TRAIN, seed=SEED)
    test_initial_conditions, test_omegas = _make_base_conditions(NUM_TEST, seed=SEED + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    stage_rows = []
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
        row = _train_stage(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            test_data=test_data,
            stage_index=stage_index,
            horizon_periods=horizon_periods,
            log_label=log_label,
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

    final_test_data = _build_data_from_base(
        test_initial_conditions,
        test_omegas,
        num_time_steps=NUM_TIME_STEPS,
        num_periods=CURRICULUM_HORIZONS[-1],
    )
    final_metrics, _, _, omega_hat = _evaluate_model(model, final_test_data)
    final_metrics["recovered_successfully"] = (
        final_metrics["trajectory_mse"] < SUCCESS_TRAJECTORY_MSE
        and final_metrics["max_omega_error"] < SUCCESS_OMEGA_TOLERANCE
    )
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", final_metrics)
    _write_json(output_dir / "curriculum_stage_summary.json", {"stages": stage_rows})
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _save_omega_plots(output_dir, omega_hat, final_test_data.omegas, plot_title_prefix)
    _write_metric_table_html(
        output_dir,
        title,
        setup,
        final_metrics,
        ["predicted_vs_true_omega.png", "omega_error_histogram.png"],
    )
    _write_summary_csv(output_dir, stage_rows)
    return final_metrics


def run_residual_curricula() -> list[dict[str, Any]]:
    RESIDUAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in _residual_variants():
        run_name = variant["run_name"]
        learning_rate = float(variant["learning_rate"])
        output_dir = RESIDUAL_OUTPUT_DIR / f"{RUN_DATE}_{run_name}"
        model = SHOTwoStateTrajectoryOmegaLinearResidualModel(
            hidden_dims=list(variant["hidden_dims"]),
            initial_omega_hat=2.0,
        )
        setup = {
            "stage": "stage1B",
            "experiment": "residual_two_state_omega_curriculum",
            "run_name": run_name,
            "input": "(x0, v0, x1, v1)",
            "delta_t": DELTA_T,
            "omega_range": OMEGA_RANGE,
            "model": "(x0, v0, x1, v1) -> linear encoder + residual encoder -> (a, b, omega_hat_i)",
            "residual_hidden_dims": list(variant["hidden_dims"]),
            "activation": "SiLU",
            "decoder": "x_hat_i(t) = a_i cos(omega_hat_i t) + b_i sin(omega_hat_i t)",
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "learning_rate_source": "best Stage 1A residual curriculum LR sweep",
            "batch_size": BATCH_SIZE,
            "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
            "stopping_monitor": "test_loss",
            "stopping_min_delta": STOPPING_MIN_DELTA,
            "stopping_patience": STOPPING_PATIENCE,
            "curriculum_horizons_periods": CURRICULUM_HORIZONS,
            "success_trajectory_mse": SUCCESS_TRAJECTORY_MSE,
            "success_omega_tolerance": SUCCESS_OMEGA_TOLERANCE,
            "random_seed": SEED,
        }
        print(f"running Stage 1B residual curriculum: {run_name} lr={learning_rate:g}")
        metrics = _run_curriculum_for_model(
            model=model,
            output_dir=output_dir,
            setup=setup,
            learning_rate=learning_rate,
            title=f"Stage 1B {run_name} Two-State Omega Curriculum",
            plot_title_prefix=f"Stage 1B {run_name}",
            log_label=f"{run_name} ",
        )
        rows.append({"run_name": run_name, **metrics, "output_dir": str(output_dir)})
    _write_residual_summary(rows)
    return rows


def _write_residual_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = [
        "run_name",
        "trajectory_mse",
        "coefficient_mse",
        "omega_mse",
        "mean_omega_error",
        "mean_abs_omega_error",
        "max_omega_error",
        "omega_correlation",
        "recovered_successfully",
        "output_dir",
    ]
    with (RESIDUAL_OUTPUT_DIR / "residual_two_state_omega_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{_format_summary_cell(column, row.get(column, ''))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Residual Two-State Omega Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
  </style>
</head>
<body>
  <h1>Stage 1B Residual Two-State Omega Curriculum</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
</body>
</html>
"""
    (RESIDUAL_OUTPUT_DIR / "residual_two_state_omega_summary.html").write_text(
        html,
        encoding="utf-8",
    )


def _format_summary_cell(column: str, value: Any) -> str:
    if column == "output_dir" and value:
        path = Path(str(value))
        return f'<a href="{escape(path.name)}/index.html">{escape(path.name)}</a>'
    return escape(str(value))


def _write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with (output_dir / "curriculum_stage_summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def main() -> None:
    residual_metrics = run_residual_curricula()
    print("residual curricula")
    print(json.dumps(residual_metrics, indent=2))


if __name__ == "__main__":
    main()
