"""Stage 1B learned-state least-squares operator curriculum diagnostic."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    DIAGNOSTIC_NUM_TEST,
    DIAGNOSTIC_NUM_TRAIN,
    TOTAL_SEQUENCE_STEPS,
    _fixed_dynamics_windows,
    _parameter_count,
)
from experiments.sho.run_stage1b_direct_state_koopman_ablation import (
    DirectStateKoopman,
    NUM_DYNAMICS_WINDOWS,
)
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import (
    CHECKPOINT as NEURAL_DIRECT_STATE_CHECKPOINT,
    _angle_error,
    _eig_angle,
    _true_k,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_state_ls_operator_curriculum")
SEED = 0
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1000
OBS_PREFIX_TRANSITIONS = 168
ROLLOUT_PERIODS = 10.0
CURRICULUM_STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, 128, "full_10_period"]
H_DIMS = [2, 4]
HIDDEN_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS_PER_STAGE = 50_000
PATIENCE = 10_000
STOPPING_MIN_DELTA = 0.0
EVAL_EVERY = 1_000
RIDGE = 1e-8


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


def _make_prefix_future_data(num_trajectories: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed, dtype=torch.float64)
    omega_min = float(OMEGA_RANGE[0])
    max_rollout_steps = int(math.ceil(ROLLOUT_PERIODS * math.tau / (omega_min * float(DELTA_T)))) + 2
    total_steps = OBS_PREFIX_TRANSITIONS + max_rollout_steps + 2
    times = torch.arange(total_steps, dtype=torch.float64) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions, omegas, initial_conditions


def _state_windows(positions: torch.Tensor, start: int, num_transitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_prev = positions[:, start : start + num_transitions]
    x_now = positions[:, start + 1 : start + num_transitions + 1]
    x_next = positions[:, start + 2 : start + num_transitions + 2]
    h_inputs = torch.stack((x_prev, x_now), dim=-1)
    h_targets = torch.stack((x_now, x_next), dim=-1)
    return h_inputs, h_targets


def _rollout_truth(positions: torch.Tensor, rollout_steps: int | str) -> torch.Tensor:
    start = OBS_PREFIX_TRANSITIONS
    if rollout_steps == "full_10_period":
        length = int(math.ceil(ROLLOUT_PERIODS * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T)))) + 1
    else:
        length = int(rollout_steps) + 1
    x_prev = positions[:, start : start + length]
    x_now = positions[:, start + 1 : start + length + 1]
    return torch.stack((x_prev, x_now), dim=-1)


def _fit_dense_k_qr(states_in: torch.Tensor, states_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = []
    conditions = []
    for a_matrix, b_matrix in zip(states_in, states_out):
        q_matrix, r_matrix = torch.linalg.qr(a_matrix, mode="reduced")
        solution = torch.linalg.solve_triangular(r_matrix, q_matrix.mT @ b_matrix, upper=True)
        matrices.append(solution.mT)
        conditions.append(torch.linalg.cond(a_matrix))
    return torch.stack(matrices), torch.stack(conditions)


def _fit_latent_k_ridge(h_in: torch.Tensor, h_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_dim = h_in.shape[-1]
    eye = torch.eye(h_dim, dtype=h_in.dtype, device=h_in.device).unsqueeze(0)
    gram = torch.bmm(h_in.transpose(1, 2), h_in)
    rhs = torch.bmm(h_in.transpose(1, 2), h_out)
    solution = torch.linalg.solve(gram + RIDGE * eye, rhs)
    residual = torch.mean((torch.bmm(h_in, solution) - h_out) ** 2)
    conditions = torch.linalg.cond(h_in)
    return solution.transpose(1, 2), residual, conditions


def _apply_k(k_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return torch.bmm(k_matrix, states.unsqueeze(-1)).squeeze(-1)


def _rollout_matrix(k_matrix: torch.Tensor, state0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [state0]
    state = state0
    for _ in range(steps):
        state = _apply_k(k_matrix, state)
        states.append(state)
    return torch.stack(states, dim=1)


class LearnedStateLSKoopman(nn.Module):
    def __init__(self, h_dim: int) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.encoder = nn.Sequential(
            nn.Linear(2, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, h_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(HIDDEN_SIZE, 2),
        )

    def encode_state(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode_state(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_states)

    def identify_k(self, prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_in = self.encode_state(prefix_in)
        h_out = self.encode_state(prefix_out)
        return _fit_latent_k_ridge(h_in, h_out)

    def rollout_decoded(self, s0: torch.Tensor, k_matrix: torch.Tensor, steps: int) -> torch.Tensor:
        h0 = self.encode_state(s0)
        latent_rollout = _rollout_matrix(k_matrix, h0, steps)
        return self.decode_state(latent_rollout)


def _trajectory_errors(predicted_states: torch.Tensor, true_states: torch.Tensor) -> dict[str, Any]:
    pos_error = predicted_states[..., 1] - true_states[..., 1]
    return {
        "rollout_mse": float(torch.mean(pos_error[:, 1:] ** 2).item()),
        "one_step_mse": float(torch.mean(pos_error[:, 1] ** 2).item()),
        "two_step_mse": float(torch.mean(pos_error[:, 2] ** 2).item()) if pos_error.shape[1] > 2 else float("nan"),
        "max_abs_trajectory_error": float(torch.max(torch.abs(pos_error[:, 1:])).item()),
        "final_time_abs_trajectory_error": float(torch.mean(torch.abs(pos_error[:, -1])).item()),
    }


def _growth_label(predicted_states: torch.Tensor, true_states: torch.Tensor) -> str:
    y = torch.mean(torch.abs(predicted_states[..., 1] - true_states[..., 1]), dim=0).detach().cpu().numpy()[1:]
    y = np.maximum(y, 1e-300)
    x = np.arange(1, y.shape[0] + 1, dtype=np.float64)
    if float(np.max(y)) < 1e-12:
        return "near_floor"
    if y.shape[0] < 8:
        return "short_horizon"
    log_y = np.log(y)
    scores = {
        "linear": _r2(y, np.polyval(np.polyfit(x, y, 1), x)),
        "quadratic": _r2(y, np.polyval(np.polyfit(x, y, 2), x)),
        "exponential": _r2(log_y, np.polyval(np.polyfit(x, log_y, 1), x)),
    }
    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] >= 0.5 else "bounded_or_oscillatory"


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / denom


def _k_spectral(k_matrix: torch.Tensor) -> dict[str, Any]:
    k_np = k_matrix.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(k_np)
    is_complex = np.any(np.abs(np.imag(eigvals)) > 1e-9, axis=1)
    return {
        "k_spectral_radius_mean": float(np.mean(np.abs(eigvals).max(axis=1))),
        "k_determinant_mean": float(np.mean(np.linalg.det(k_np))),
        "k_fraction_complex_eigenvalues": float(np.mean(is_complex)),
    }


def _raw_k_diagnostics(k_matrix: torch.Tensor, omegas: torch.Tensor) -> dict[str, Any]:
    k_true = _true_k(omegas)
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    det = np.linalg.det(k_np)
    true_det = np.linalg.det(true_np)
    return {
        "k_frobenius_error_mean": float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2)))),
        "trace_c_abs_error_mean": float(np.mean(np.abs(trace - true_trace))),
        "determinant_abs_error_mean": float(np.mean(np.abs(det - true_det))),
        "eigen_angle_abs_error_mean": float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np)))),
    }


def _evaluate_raw_ls(positions: torch.Tensor, omegas: torch.Tensor) -> dict[str, Any]:
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    k_matrix, conditions = _fit_dense_k_qr(prefix_in, prefix_out)
    true_states = _rollout_truth(positions, "full_10_period")
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return {
        "model": "raw_state_ls_baseline",
        "h_dim": "",
        "stage": "full_10_period",
        "best_epoch": 0,
        "parameter_count": 0,
        "state_autoencoder_mse": float("nan"),
        "latent_transition_residual": float("nan"),
        "ls_condition_number_mean": float(conditions.mean().item()),
        "ls_condition_number_max": float(conditions.max().item()),
        "error_growth_label": _growth_label(predicted, true_states),
        **_trajectory_errors(predicted, true_states),
        **_raw_k_diagnostics(k_matrix, omegas),
    }


def _load_neural_direct_state_model() -> DirectStateKoopman:
    model = DirectStateKoopman().to(dtype=torch.float64)
    state = torch.load(NEURAL_DIRECT_STATE_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate_neural_generated_k(positions: torch.Tensor) -> dict[str, Any]:
    model = _load_neural_direct_state_model()
    with torch.no_grad():
        windows = _fixed_dynamics_windows(positions[:, :TOTAL_SEQUENCE_STEPS].to(torch.float64), NUM_DYNAMICS_WINDOWS)
        d = model.encode_dynamics(windows)
        k_matrix = model.koopman(d)
        true_states = _rollout_truth(positions, "full_10_period")
        predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return {
        "model": "neural_generated_direct_state_k_baseline",
        "h_dim": "",
        "stage": "full_10_period",
        "best_epoch": 0,
        "parameter_count": _parameter_count(model),
        "state_autoencoder_mse": float("nan"),
        "latent_transition_residual": float("nan"),
        "ls_condition_number_mean": float("nan"),
        "ls_condition_number_max": float("nan"),
        "error_growth_label": _growth_label(predicted, true_states),
        **_trajectory_errors(predicted, true_states),
        **_k_spectral(k_matrix),
    }


def _loss_and_parts(
    model: LearnedStateLSKoopman,
    positions: torch.Tensor,
    stage: int | str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    k_matrix, residual, conditions = model.identify_k(prefix_in, prefix_out)
    true_states = _rollout_truth(positions, stage)
    predicted = model.rollout_decoded(true_states[:, 0], k_matrix, true_states.shape[1] - 1)
    loss = torch.mean((predicted - true_states) ** 2)
    with torch.no_grad():
        encoded = model.encode_state(true_states)
        decoded = model.decode_state(encoded)
        ae_mse = torch.mean((decoded - true_states) ** 2)
    parts = {
        **_trajectory_errors(predicted, true_states),
        "state_autoencoder_mse": float(ae_mse.item()),
        "latent_transition_residual": float(residual.item()),
        "ls_condition_number_mean": float(conditions.mean().item()),
        "ls_condition_number_max": float(conditions.max().item()),
        "error_growth_label": _growth_label(predicted, true_states),
        **_k_spectral(k_matrix),
    }
    return loss, parts, predicted, true_states


def _evaluate_learned_state_model(
    model: LearnedStateLSKoopman,
    positions: torch.Tensor,
    stage: int | str,
    *,
    best_epoch: int,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, _, _ = _loss_and_parts(model, positions, stage)
    return {
        "model": "learned_state_ls_operator",
        "h_dim": model.h_dim,
        "stage": stage,
        "best_epoch": best_epoch,
        "parameter_count": _parameter_count(model),
        **parts,
    }


def _train_h_dim(h_dim: int, train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], LearnedStateLSKoopman]:
    run_dir = OUTPUT_DIR / f"learned_state_h{h_dim}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        model = LearnedStateLSKoopman(h_dim).to(dtype=torch.float64)
        model.load_state_dict(torch.load(run_dir / "learned_parameters.pt", map_location="cpu"))
        return json.loads(metrics_path.read_text(encoding="utf-8"))["history"], model

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = LearnedStateLSKoopman(h_dim).to(dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_epoch_global = 0

    _write_json(
        run_dir / "setup.json",
        {
            "h_dim": h_dim,
            "ridge": RIDGE,
            "curriculum_stages": CURRICULUM_STAGES,
            "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
            "patience": PATIENCE,
            "stopping_min_delta": STOPPING_MIN_DELTA,
            "uses_velocity": False,
            "uses_omega_supervision": False,
            "uses_coefficient_supervision": False,
            "uses_analytic_decoder": False,
            "uses_hardcoded_sho_recurrence": False,
        },
    )

    global_epoch = 0
    for stage in CURRICULUM_STAGES:
        best_stage_metric = float("inf")
        best_stage_state = copy.deepcopy(model.state_dict())
        best_stage_epoch = 0
        epochs_since_improvement = 0
        stopped_because = "max_epochs_reached"
        for epoch in range(1, MAX_EPOCHS_PER_STAGE + 1):
            global_epoch += 1
            model.train()
            optimizer.zero_grad()
            loss, _, _, _ = _loss_and_parts(model, train_positions, stage)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            if epoch % EVAL_EVERY == 0 or epoch == 1:
                eval_row = _evaluate_learned_state_model(model, test_positions, stage, best_epoch=global_epoch)
                eval_row.update(
                    {
                        "curriculum_stage": stage,
                        "stage_epoch": epoch,
                        "global_epoch": global_epoch,
                        "train_loss": float(loss.item()),
                    },
                )
                history.append(eval_row)
                _write_json(run_dir / "training_history.json", {"history": history})
                _write_csv(run_dir / "history.csv", history)
                metric = eval_row["rollout_mse"]
                if metric < best_stage_metric - STOPPING_MIN_DELTA:
                    best_stage_metric = metric
                    best_stage_state = copy.deepcopy(model.state_dict())
                    best_stage_epoch = epoch
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += EVAL_EVERY
                if epochs_since_improvement >= PATIENCE:
                    stopped_because = "patience_exhausted"
                    break
        model.load_state_dict(best_stage_state)
        best_state = copy.deepcopy(model.state_dict())
        best_epoch_global = global_epoch - (epoch - best_stage_epoch)
        stage_summary = _evaluate_learned_state_model(model, test_positions, stage, best_epoch=best_epoch_global)
        stage_summary.update(
            {
                "curriculum_stage": stage,
                "stage_epoch": best_stage_epoch,
                "global_epoch": best_epoch_global,
                "stopped_because": stopped_because,
            },
        )
        history.append(stage_summary)
        _write_json(run_dir / "training_history.json", {"history": history})
        _write_csv(run_dir / "history.csv", history)

    model.load_state_dict(best_state)
    final_metrics = {"history": history, "final": _evaluate_learned_state_model(model, test_positions, "full_10_period", best_epoch=best_epoch_global)}
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _write_json(metrics_path, final_metrics)
    return history, model


def _plot_history(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str) -> None:
    learned = [row for row in rows if row["model"] == "learned_state_ls_operator" and "curriculum_stage" in row]
    if not learned:
        return
    plt.figure(figsize=(8.0, 5.0))
    for h_dim in sorted({row["h_dim"] for row in learned}):
        selected = [row for row in learned if row["h_dim"] == h_dim]
        plt.plot(
            [row["global_epoch"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=1.2,
            label=f"h_dim={h_dim}",
        )
    plt.yscale("log")
    plt.xlabel("global epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs curriculum progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _plot_examples(model: LearnedStateLSKoopman, positions: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        _, _, predicted, true_states = _loss_and_parts(model, positions[:4], "full_10_period")
    steps = np.arange(true_states.shape[1])
    plt.figure(figsize=(9.0, 5.2))
    for index in range(true_states.shape[0]):
        plt.plot(steps, true_states[index, :, 1].detach().cpu().numpy(), linestyle="--", alpha=0.7)
        plt.plot(steps, predicted[index, :, 1].detach().cpu().numpy(), alpha=0.9, label=f"traj {index}")
    plt.xlabel("rollout step after prefix")
    plt.ylabel("position")
    plt.title("learned-state LS rollout examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_examples.png")
    plt.close()

    plt.figure(figsize=(9.0, 5.2))
    for index in range(true_states.shape[0]):
        error = torch.abs(predicted[index, :, 1] - true_states[index, :, 1]).detach().cpu().numpy()
        plt.plot(steps, error, label=f"traj {index}")
    plt.yscale("log")
    plt.xlabel("rollout step after prefix")
    plt.ylabel("|position error|")
    plt.title("learned-state LS error growth examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_growth_examples.png")
    plt.close()


def _write_html(rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return escape(str(value))

    final_rows = [row for row in rows if row.get("stage") == "full_10_period"]
    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{fmt(row.get(column, ''))}</td>"
            for column in [
                "model",
                "h_dim",
                "rollout_mse",
                "one_step_mse",
                "two_step_mse",
                "max_abs_trajectory_error",
                "final_time_abs_trajectory_error",
                "best_epoch",
                "parameter_count",
                "state_autoencoder_mse",
                "latent_transition_residual",
                "ls_condition_number_mean",
                "error_growth_label",
            ]
        )
        + "</tr>"
        for row in final_rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Learned-State LS Operator Curriculum</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>Learned-State Least-Squares Operator Curriculum</h1>
  <p>This diagnostic learns <code>H(s)</code> and <code>D(h)</code>, but identifies each trajectory operator by differentiable least squares in latent state space. The observed prefix is used for operator identification; future rollout states are used only for reconstruction loss and evaluation.</p>
  <p>Latent LS uses a ridge term of <code>{RIDGE:g}</code> for numerical stability.</p>
  <h2>Final Comparison</h2>
  <table>
    <thead><tr><th>model</th><th>h_dim</th><th>rollout MSE</th><th>one-step</th><th>two-step</th><th>max abs err</th><th>final abs err</th><th>best epoch</th><th>params</th><th>AE MSE</th><th>latent residual</th><th>cond</th><th>growth</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Plots</h2>
  <div class="grid">
    <img src="rollout_mse_vs_curriculum_stage.png" alt="rollout MSE">
    <img src="one_step_mse_vs_curriculum_stage.png" alt="one-step MSE">
    <img src="max_error_vs_curriculum_stage.png" alt="max error">
    <img src="rollout_examples.png" alt="rollout examples">
    <img src="error_growth_examples.png" alt="error growth examples">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_positions, test_omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(_evaluate_raw_ls(test_positions, test_omegas))
    summary_rows.append(_evaluate_neural_generated_k(test_positions))

    last_model: LearnedStateLSKoopman | None = None
    for h_dim in H_DIMS:
        history, model = _train_h_dim(h_dim, train_positions, test_positions)
        last_model = model
        summary_rows.extend(history)
        final_row = _evaluate_learned_state_model(model, test_positions, "full_10_period", best_epoch=history[-1].get("best_epoch", 0))
        summary_rows.append(final_row)

    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_json(
        OUTPUT_DIR / "summary.json",
        {
            "output_dir": str(OUTPUT_DIR),
            "ridge": RIDGE,
            "obs_prefix_transitions": OBS_PREFIX_TRANSITIONS,
            "curriculum_stages": CURRICULUM_STAGES,
            "max_epochs_per_stage": MAX_EPOCHS_PER_STAGE,
            "patience": PATIENCE,
            "rows": summary_rows,
        },
    )
    _write_html(summary_rows)
    _plot_history(summary_rows, "rollout_mse", "rollout_mse_vs_curriculum_stage.png", "rollout MSE")
    _plot_history(summary_rows, "one_step_mse", "one_step_mse_vs_curriculum_stage.png", "one-step MSE")
    _plot_history(summary_rows, "max_abs_trajectory_error", "max_error_vs_curriculum_stage.png", "max absolute error")
    if last_model is not None:
        _plot_examples(last_model, test_positions)


if __name__ == "__main__":
    main()
