"""Direct observed-state Koopman ablation for Stage 1B SHO."""

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
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import torch
from torch import nn

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    BATCH_SIZE,
    DIAGNOSTIC_NUM_TEST,
    DIAGNOSTIC_NUM_TRAIN,
    DYNAMICS_LATENT_DIM,
    GENERATOR_HIDDEN_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    ROLLOUT_STEPS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    AttentionDynamicsEncoder,
    _attention_stats,
    _fixed_dynamics_windows,
    _fixed_rollout_chunks,
    _make_position_data,
    _parameter_count,
    _positions_to_states,
    _sample_training_batch,
    _save_attention_heatmap,
    _save_latent_pca,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/direct_state_koopman_ablation")
NUM_DYNAMICS_WINDOWS = 32
EVAL_EVERY = 1_000
FULL_MODEL_METRICS = Path("outputs/sho/stage1B/attention_large_window_sweep/attention_m32/metrics.json")


class DirectStateKoopman(nn.Module):
    """Generate a trajectory-specific dense 2x2 K and evolve observed states directly."""

    def __init__(self) -> None:
        super().__init__()
        self.dynamics_encoder = AttentionDynamicsEncoder()
        self.koopman_generator = nn.Sequential(
            nn.Linear(DYNAMICS_LATENT_DIM, GENERATOR_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(GENERATOR_HIDDEN_SIZE, 4),
        )

    def encode_dynamics(self, windows: torch.Tensor) -> torch.Tensor:
        return self.dynamics_encoder(windows)

    def dynamics_attention_weights(self, windows: torch.Tensor) -> torch.Tensor:
        return self.dynamics_encoder.attention_weights(windows)

    def koopman(self, dynamics_latent: torch.Tensor) -> torch.Tensor:
        return self.koopman_generator(dynamics_latent).reshape(-1, 2, 2)

    def apply_koopman(self, states: torch.Tensor, k_matrix: torch.Tensor) -> torch.Tensor:
        return torch.bmm(k_matrix, states.unsqueeze(-1)).squeeze(-1)

    def rollout(self, s0: torch.Tensor, k_matrix: torch.Tensor, steps: int) -> torch.Tensor:
        states = [s0]
        state = s0
        for _ in range(steps):
            state = self.apply_koopman(state, k_matrix)
            states.append(state)
        return torch.stack(states, dim=1)


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


def _linear_regression_prediction_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    return float(model.score(x, y)), float(np.corrcoef(prediction.reshape(-1), y.reshape(-1))[0, 1])


def _multioutput_r2(features: torch.Tensor, targets: torch.Tensor) -> float:
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    return float(model.score(x, y))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _losses(
    model: DirectStateKoopman,
    dynamics_windows: torch.Tensor,
    rollout_positions: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    d = model.encode_dynamics(dynamics_windows)
    k_matrix = model.koopman(d)
    true_states = _positions_to_states(rollout_positions)
    predicted_states = model.rollout(true_states[:, 0], k_matrix, ROLLOUT_STEPS)

    one_step_state_mse = mse_loss(predicted_states[:, 1], true_states[:, 1])
    two_step_state_mse = mse_loss(predicted_states[:, 2], true_states[:, 2])
    rollout_state_mse = mse_loss(predicted_states[:, 1:], true_states[:, 1:])
    total_loss = one_step_state_mse + two_step_state_mse + rollout_state_mse

    return total_loss, {
        "one_step_mse": mse_loss(predicted_states[:, 1, 1], true_states[:, 1, 1]),
        "two_step_mse": mse_loss(predicted_states[:, 2, 1], true_states[:, 2, 1]),
        "rollout_mse": mse_loss(predicted_states[:, 1:, 1], true_states[:, 1:, 1]),
        "one_step_state_mse": one_step_state_mse,
        "two_step_state_mse": two_step_state_mse,
        "rollout_state_mse": rollout_state_mse,
        "d_variance": d.var(dim=0, unbiased=False).mean(),
    }, d, k_matrix, predicted_states


def _k_spectral_diagnostics(k_matrix: torch.Tensor, per_trajectory_rollout_mse: torch.Tensor) -> dict[str, Any]:
    k_np = k_matrix.detach().cpu().numpy()
    rollout_np = per_trajectory_rollout_mse.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(k_np)
    abs_eig = np.abs(eigvals)
    spectral_radius = abs_eig.max(axis=1)
    determinant = np.linalg.det(k_np)
    trace = np.trace(k_np, axis1=1, axis2=2)
    condition_number = np.linalg.cond(k_np)
    is_complex = np.any(np.abs(np.imag(eigvals)) > 1e-8, axis=1)
    radius_error = np.abs(spectral_radius - 1.0)
    det_error = np.abs(determinant - 1.0)
    return {
        "spectral_radius_mean": float(np.mean(spectral_radius)),
        "spectral_radius_std": float(np.std(spectral_radius, ddof=0)),
        "spectral_radius_min": float(np.min(spectral_radius)),
        "spectral_radius_max": float(np.max(spectral_radius)),
        "spectral_radius_error_from_1_mean": float(np.mean(radius_error)),
        "determinant_mean": float(np.mean(determinant)),
        "determinant_std": float(np.std(determinant, ddof=0)),
        "determinant_error_from_1_mean": float(np.mean(det_error)),
        "trace_mean": float(np.mean(trace)),
        "trace_std": float(np.std(trace, ddof=0)),
        "condition_number_mean": float(np.mean(condition_number)),
        "condition_number_std": float(np.std(condition_number, ddof=0)),
        "fraction_complex_eigenvalues": float(np.mean(is_complex)),
        "corr_spectral_radius_with_rollout": _safe_corr(spectral_radius, rollout_np),
        "corr_spectral_radius_error_with_rollout": _safe_corr(radius_error, rollout_np),
        "corr_determinant_with_rollout": _safe_corr(determinant, rollout_np),
        "corr_determinant_error_with_rollout": _safe_corr(det_error, rollout_np),
        "corr_trace_with_rollout": _safe_corr(trace, rollout_np),
        "corr_condition_number_with_rollout": _safe_corr(condition_number, rollout_np),
        "eigenvalues_real_mean": float(np.mean(np.real(eigvals))),
        "eigenvalues_imag_abs_mean": float(np.mean(np.abs(np.imag(eigvals)))),
    }


def _evaluate(
    model: DirectStateKoopman,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, NUM_DYNAMICS_WINDOWS)
        rollout_positions = _fixed_rollout_chunks(sequences)
        _, parts, d, k_matrix, predicted_states = _losses(model, dynamics_windows, rollout_positions)
        true_states = _positions_to_states(rollout_positions)
        attention_weights = model.dynamics_attention_weights(dynamics_windows)
        per_trajectory_rollout_mse = torch.mean((predicted_states[:, 1:, 1] - true_states[:, 1:, 1]) ** 2, dim=1)

    omega_r2, omega_corr = _linear_regression_prediction_stats(d, omegas)
    metrics: dict[str, Any] = {
        "rollout_mse": parts["rollout_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "rollout_state_mse": parts["rollout_state_mse"].item(),
        "one_step_state_mse": parts["one_step_state_mse"].item(),
        "two_step_state_mse": parts["two_step_state_mse"].item(),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
        **_k_spectral_diagnostics(k_matrix, per_trajectory_rollout_mse),
        **_attention_stats(attention_weights),
    }
    return metrics, d, attention_weights, k_matrix


def _train_direct_state_model(
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = OUTPUT_DIR / "direct_state_attention_m32"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"
    if metrics_path.exists() and history_path.exists():
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(history_path.read_text(encoding="utf-8"))["history"],
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "setup.json",
        {
            "run_name": "direct_state_attention_m32",
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "seed": SEED,
            "train_data_seed": TRAIN_DATA_SEED,
            "test_data_seed": TEST_DATA_SEED,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_only": True,
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
            "contrastive_loss": False,
            "state_encoder": False,
            "state_decoder": False,
            "structured_koopman": False,
        },
    )

    set_seed(SEED)
    model = DirectStateKoopman()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    history: list[dict[str, Any]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        dynamics_windows, rollout_positions = _sample_training_batch(train_sequences, NUM_DYNAMICS_WINDOWS)
        optimizer.zero_grad()
        loss, train_parts, _, _, _ = _losses(model, dynamics_windows, rollout_positions)
        if not torch.isfinite(loss):
            stopped_because = "nan_or_inf_loss"
            break
        loss.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, _, _, _ = _evaluate(model, test_sequences, test_omegas, test_labels)
            if eval_metrics["rollout_mse"] < best_rollout:
                best_rollout = eval_metrics["rollout_mse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "train_one_step_state_mse": train_parts["one_step_state_mse"].item(),
                    "train_two_step_state_mse": train_parts["two_step_state_mse"].item(),
                    "train_rollout_state_mse": train_parts["rollout_state_mse"].item(),
                    **eval_metrics,
                },
            )
            _write_json(history_path, {"history": history})
            _write_csv(run_dir / "history.csv", history)
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, d, attention_weights, k_matrix = _evaluate(model, test_sequences, test_omegas, test_labels)
    metrics.update(
        {
            "run_name": "direct_state_attention_m32",
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "parameter_count": _parameter_count(model),
        },
    )
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    torch.save({"d": d, "attention_weights": attention_weights, "k_matrix": k_matrix}, run_dir / "latents_attention_k.pt")
    _write_json(metrics_path, metrics)
    _write_json(history_path, {"history": history})
    _write_csv(run_dir / "history.csv", history)
    _save_latent_pca(run_dir / "d_latent_pca_colored_by_omega.png", d, test_omegas, "direct-state d_i PCA")
    _save_attention_heatmap(run_dir / "attention_heatmap.png", attention_weights, "direct-state attention heatmap")
    return metrics, history


def _plot_history(history: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log_y: bool = True) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.plot([row["epoch"] for row in history], [row[metric] for row in history], marker="o", markersize=3)
    if log_y:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _plot_comparison(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log_y: bool = True) -> None:
    plt.figure(figsize=(7.0, 4.6))
    labels = [row["model"] for row in rows]
    values = [row[metric] for row in rows]
    plt.bar(labels, values)
    if log_y:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _summary_html(rows: list[dict[str, Any]], direct_metrics: dict[str, Any], full_metrics: dict[str, Any]) -> str:
    selected_columns = [
        "model",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "omega_r2_from_d",
        "omega_correlation_from_d",
        "abomega_multioutput_r2_from_d",
        "parameter_count",
        "best_epoch",
        "spectral_radius_mean",
        "determinant_mean",
        "trace_mean",
        "fraction_complex_eigenvalues",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in selected_columns)
    body = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{row.get(column, ''):.6g}</td>" if isinstance(row.get(column), float) else f"<td>{escape(str(row.get(column, '')))}</td>"
            for column in selected_columns
        )
        + "</tr>"
        for row in rows
    )
    direct_rollout = float(direct_metrics["rollout_mse"])
    full_rollout = float(full_metrics["rollout_mse"])
    if direct_rollout < full_rollout:
        answer = (
            "Removing H and D improves the rollout error, which indicates that state encoder/decoder alignment "
            "was a material weak link in the full separated model."
        )
    else:
        answer = (
            "Removing H and D does not improve the rollout error, which points away from H/D alignment as the only "
            "bottleneck and toward the dynamics-latent-to-K mapping or the unconstrained dense K itself."
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Direct-State Koopman Ablation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 2px 5px; border-radius: 4px; }}
    .note {{ background: #f0f4f8; border-left: 4px solid #486581; padding: 12px 16px; margin: 16px 0; }}
    img {{ max-width: 760px; width: 100%; display: block; margin: 14px 0 28px; }}
  </style>
</head>
<body>
  <h1>Direct-State Koopman Ablation</h1>
  <p>This diagnostic removes the learned state encoder <code>H</code> and decoder <code>D</code>. The model keeps the m=32 attention dynamics encoder and learns a dense trajectory-conditioned 2x2 matrix <code>K_i = G(d_i)</code>, then evolves the observed state <code>s_t=(x_{{t-1}},x_t)</code> directly.</p>
  <div class="note">{escape(answer)}</div>
  <h2>Model Comparison</h2>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Training Curves</h2>
  <img src="rollout_mse_vs_epoch.png" alt="rollout MSE vs epoch">
  <img src="one_step_mse_vs_epoch.png" alt="one-step MSE vs epoch">
  <img src="two_step_mse_vs_epoch.png" alt="two-step MSE vs epoch">
  <h2>Comparison Plots</h2>
  <img src="rollout_mse_comparison.png" alt="rollout MSE comparison">
  <img src="omega_r2_comparison.png" alt="omega R2 comparison">
  <h2>K Diagnostics</h2>
  <p>The direct-state matrix has mean spectral radius {direct_metrics['spectral_radius_mean']:.6g}, determinant {direct_metrics['determinant_mean']:.6g}, trace {direct_metrics['trace_mean']:.6g}, and complex eigenvalues for {direct_metrics['fraction_complex_eigenvalues']:.3g} of test trajectories.</p>
  <p>Correlation between spectral-radius error and per-trajectory rollout error: {direct_metrics['corr_spectral_radius_error_with_rollout']:.6g}.</p>
  <h2>Latent and Attention</h2>
  <img src="direct_state_attention_m32/d_latent_pca_colored_by_omega.png" alt="latent PCA colored by omega">
  <img src="direct_state_attention_m32/attention_heatmap.png" alt="attention heatmap">
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    direct_metrics, history = _train_direct_state_model(train_sequences, test_sequences, test_omegas, test_labels)

    full_metrics = json.loads(FULL_MODEL_METRICS.read_text(encoding="utf-8"))
    full_row = {
        "model": "full_attention_m32_HKD",
        "rollout_mse": full_metrics["rollout_mse"],
        "one_step_mse": full_metrics["one_step_mse"],
        "two_step_mse": full_metrics["two_step_mse"],
        "omega_r2_from_d": full_metrics["omega_r2_from_d"],
        "omega_correlation_from_d": full_metrics["omega_correlation_from_d"],
        "abomega_multioutput_r2_from_d": full_metrics["abomega_multioutput_r2_from_d"],
        "parameter_count": full_metrics["parameter_count"],
        "best_epoch": full_metrics["best_epoch"],
        "spectral_radius_mean": "",
        "determinant_mean": "",
        "trace_mean": "",
        "fraction_complex_eigenvalues": "",
    }
    direct_row = {"model": "direct_state_attention_m32", **direct_metrics}
    rows = [full_row, direct_row]

    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_json(OUTPUT_DIR / "summary.json", {"full_model": full_metrics, "direct_state": direct_metrics})
    _plot_history(history, "rollout_mse", "rollout_mse_vs_epoch.png", "rollout MSE")
    _plot_history(history, "one_step_mse", "one_step_mse_vs_epoch.png", "one-step MSE")
    _plot_history(history, "two_step_mse", "two_step_mse_vs_epoch.png", "two-step MSE")
    _plot_comparison(rows, "rollout_mse", "rollout_mse_comparison.png", "rollout MSE")
    _plot_comparison(rows, "omega_r2_from_d", "omega_r2_comparison.png", "omega R2 from d", log_y=False)
    (OUTPUT_DIR / "summary.html").write_text(_summary_html(rows, direct_metrics, full_metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
