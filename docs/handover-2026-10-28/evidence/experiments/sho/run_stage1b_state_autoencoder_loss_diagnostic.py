"""State autoencoder loss diagnostic for attention separated Koopman."""

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
from sklearn.linear_model import LinearRegression
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    BATCH_SIZE,
    DIAGNOSTIC_NUM_TEST,
    DIAGNOSTIC_NUM_TRAIN,
    DYNAMICS_LATENT_DIM,
    EVAL_EVERY,
    LAMBDA_LATENT,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    POSITION_WINDOW_SIZE,
    ROLLOUT_STEPS,
    SEED,
    STATE_LATENT_DIM,
    TOTAL_SEQUENCE_STEPS,
    AttentionDynamicsEncoder,
    SeparatedKoopmanWithDynamicsEncoder,
    _attention_stats,
    _error_decomposition,
    _fixed_dynamics_windows,
    _fixed_rollout_chunks,
    _make_position_data,
    _multioutput_r2,
    _parameter_count,
    _plot_attention_metric,
    _positions_to_states,
    _sample_dynamics_windows,
    _sample_windows,
    _save_attention_heatmap,
    _save_latent_pca,
    _write_csv,
    _write_json,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/state_autoencoder_loss_diagnostic")
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1000
NUM_DYNAMICS_WINDOWS = 32
LAMBDA_AE_VALUES = [1e-3, 1e-2, 1e-1, 1.0]


def _linear_regression_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    return float(model.score(x, y)), float(np.corrcoef(prediction.reshape(-1), y.reshape(-1))[0, 1])


def _sample_training_batch(
    sequences: torch.Tensor,
    num_windows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_trajectories = sequences.shape[0]
    indices = torch.randperm(num_trajectories)[:BATCH_SIZE]
    max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    max_rollout_start = TOTAL_SEQUENCE_STEPS - ROLLOUT_STEPS - 2
    starts = torch.randint(0, max_window_start + 1, (indices.shape[0], num_windows))
    rollout_starts = torch.randint(0, max_rollout_start + 1, (indices.shape[0],))
    return (
        _sample_dynamics_windows(sequences, indices, starts),
        _sample_windows(sequences, indices, rollout_starts, ROLLOUT_STEPS + 2),
        sequences[indices],
    )


def _state_autoencoder_mse(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
) -> torch.Tensor:
    all_states = _positions_to_states(sequences)
    h = model.encode_state(all_states.reshape(-1, 2))
    reconstructed = model.decode_state(h).reshape_as(all_states)
    return mse_loss(reconstructed, all_states)


def _losses(
    model: SeparatedKoopmanWithDynamicsEncoder,
    dynamics_windows: torch.Tensor,
    rollout_positions: torch.Tensor,
    ae_sequences: torch.Tensor,
    *,
    lambda_ae: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    d = model.encode_dynamics(dynamics_windows)
    k_matrix = model.koopman(d)
    true_states = _positions_to_states(rollout_positions)
    h0 = model.encode_state(true_states[:, 0])
    h_rollout = model.rollout_latent(h0, k_matrix, ROLLOUT_STEPS)
    decoded_states = model.decode_state(h_rollout.reshape(-1, STATE_LATENT_DIM)).reshape(
        true_states.shape[0],
        ROLLOUT_STEPS + 1,
        2,
    )
    rollout_state_mse = mse_loss(decoded_states, true_states)
    one_step_mse = mse_loss(decoded_states[:, 1, 1], true_states[:, 1, 1])
    two_step_mse = mse_loss(decoded_states[:, 2, 1], true_states[:, 2, 1])
    rollout_mse = mse_loss(decoded_states[:, 1:, 1], true_states[:, 1:, 1])
    h_true = model.encode_state(true_states.reshape(-1, 2)).reshape(
        true_states.shape[0],
        ROLLOUT_STEPS + 1,
        STATE_LATENT_DIM,
    )
    k_expanded = k_matrix.unsqueeze(1).expand(-1, ROLLOUT_STEPS, -1, -1).reshape(-1, STATE_LATENT_DIM, STATE_LATENT_DIM)
    h_pred_next = torch.bmm(k_expanded, h_true[:, :-1].reshape(-1, STATE_LATENT_DIM).unsqueeze(-1)).squeeze(-1)
    latent_transition_mse = mse_loss(h_pred_next, h_true[:, 1:].reshape(-1, STATE_LATENT_DIM))
    state_autoencoder_mse = _state_autoencoder_mse(model, ae_sequences)
    total_loss = rollout_state_mse + LAMBDA_LATENT * latent_transition_mse + lambda_ae * state_autoencoder_mse
    return total_loss, {
        "rollout_state_mse": rollout_state_mse,
        "rollout_mse": rollout_mse,
        "one_step_mse": one_step_mse,
        "two_step_mse": two_step_mse,
        "latent_transition_mse": latent_transition_mse,
        "state_autoencoder_mse": state_autoencoder_mse,
        "d_variance": d.var(dim=0, unbiased=False).mean(),
        "h_variance": h_true.var(dim=(0, 1), unbiased=False).mean(),
    }, d


def _evaluate(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_ae: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, NUM_DYNAMICS_WINDOWS)
        rollout_positions = _fixed_rollout_chunks(sequences)
        _, parts, d = _losses(model, dynamics_windows, rollout_positions, sequences, lambda_ae=lambda_ae)
        attention_weights = model.dynamics_attention_weights(dynamics_windows)
    omega_r2, omega_corr = _linear_regression_stats(d, omegas)
    return {
        "rollout_mse": parts["rollout_mse"].item(),
        "rollout_state_mse": parts["rollout_state_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "state_autoencoder_mse": parts["state_autoencoder_mse"].item(),
        "latent_transition_mse": parts["latent_transition_mse"].item(),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
        "h_variance": parts["h_variance"].item(),
    }, d, attention_weights


def _train_variant(
    lambda_ae: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = f"lambda_ae_{lambda_ae:g}".replace(".", "p").replace("-", "m")
    run_dir = OUTPUT_DIR / run_name
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"
    if metrics_path.exists() and history_path.exists():
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(history_path.read_text(encoding="utf-8"))["history"],
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = SeparatedKoopmanWithDynamicsEncoder(encoder_type="attention")
    if not isinstance(model.dynamics_encoder, AttentionDynamicsEncoder):
        raise TypeError("expected attention dynamics encoder")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    history: list[dict[str, Any]] = []
    _write_json(
        run_dir / "setup.json",
        {
            "run_name": run_name,
            "lambda_ae": lambda_ae,
            "lambda_latent": LAMBDA_LATENT,
            "lambda_dyn": 0.0,
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "seed": SEED,
            "train_data_seed": TRAIN_DATA_SEED,
            "test_data_seed": TEST_DATA_SEED,
            "dynamics_latent_dim": DYNAMICS_LATENT_DIM,
            "state_latent_dim": STATE_LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "loss": "L_rollout + lambda_latent * L_latent_transition + lambda_AE * L_AE",
            "state_autoencoder_loss": "mean over every local state s_t=(x_{t-1},x_t) in each selected full training sequence",
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
        },
    )
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        dynamics_windows, rollout_positions, ae_sequences = _sample_training_batch(train_sequences, NUM_DYNAMICS_WINDOWS)
        optimizer.zero_grad()
        loss, train_parts, _ = _losses(
            model,
            dynamics_windows,
            rollout_positions,
            ae_sequences,
            lambda_ae=lambda_ae,
        )
        if not torch.isfinite(loss):
            stopped_because = "nan_or_inf_loss"
            break
        loss.backward()
        optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, _, _ = _evaluate(
                model,
                test_sequences,
                test_omegas,
                test_labels,
                lambda_ae=lambda_ae,
            )
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
                    "train_rollout_state_mse": train_parts["rollout_state_mse"].item(),
                    "train_state_autoencoder_mse": train_parts["state_autoencoder_mse"].item(),
                    "train_latent_transition_mse": train_parts["latent_transition_mse"].item(),
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
    metrics, d, attention_weights = _evaluate(
        model,
        test_sequences,
        test_omegas,
        test_labels,
        lambda_ae=lambda_ae,
    )
    metrics.update(_error_decomposition(model, test_sequences, test_omegas, num_windows=NUM_DYNAMICS_WINDOWS))
    metrics.update(
        {
            "run_name": run_name,
            "lambda_ae": lambda_ae,
            "lambda_latent": LAMBDA_LATENT,
            "lambda_dyn": 0.0,
            "encoder_type": "attention",
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "dynamics_latent_dim": DYNAMICS_LATENT_DIM,
            "state_latent_dim": STATE_LATENT_DIM,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "parameter_count": _parameter_count(model),
            **_attention_stats(attention_weights),
        },
    )
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    torch.save({"d": d, "attention_weights": attention_weights}, run_dir / "latents_and_attention.pt")
    _write_json(metrics_path, metrics)
    _write_json(history_path, {"history": history})
    _write_csv(run_dir / "history.csv", history)
    _save_latent_pca(run_dir / "d_latent_pca_colored_by_omega.png", d, test_omegas, f"d_i PCA {run_name}")
    _save_attention_heatmap(run_dir / "attention_heatmap.png", attention_weights, f"attention heatmap {run_name}")
    return metrics, history


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_html(rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "run_name",
        "lambda_ae",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "state_autoencoder_mse",
        "latent_transition_mse",
        "omega_r2_from_d",
        "abomega_multioutput_r2_from_d",
        "decomp_state_autoencode_position_mse",
        "decomp_rollout_minus_autoencode_mse",
        "n_eff_mean",
        "parameter_count",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    interpretation_html = "".join(f"<p>{escape(line)}</p>" for line in interpretation)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>State Autoencoder Loss Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>State Autoencoder Loss Diagnostic</h1>
  <p>Attention separated dynamics/state Koopman, m=32, position-only, lambda_latent={LAMBDA_LATENT:g}, lambda_dyn=0. The only added objective is L_AE over every local state in each selected full trajectory.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>Rollout MSE</h2><img src="rollout_mse_vs_lambda_ae.png">
  <h2>State Autoencoder MSE</h2><img src="state_autoencoder_mse_vs_lambda_ae.png">
  <h2>Latent Transition MSE</h2><img src="latent_transition_mse_vs_lambda_ae.png">
  <h2>Omega R2</h2><img src="omega_r2_vs_lambda_ae.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _plot_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log_y: bool = False) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["lambda_ae"])
    plt.figure(figsize=(7.2, 4.6))
    plt.plot([row["lambda_ae"] for row in sorted_rows], [row[metric] for row in sorted_rows], marker="o")
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("lambda_AE")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    best_rollout = min(rows, key=lambda row: row["rollout_mse"])
    best_ae = min(rows, key=lambda row: row["state_autoencoder_mse"])
    return [
        f"Best rollout is {best_rollout['run_name']} with rollout MSE {best_rollout['rollout_mse']:.4e}.",
        f"Best state autoencoder MSE is {best_ae['run_name']} with MSE {best_ae['state_autoencoder_mse']:.4e}.",
        "Compare whether the best state autoencoder run is also the best rollout run to determine whether state reconstruction is the active bottleneck.",
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    rows: list[dict[str, Any]] = []
    for lambda_ae in LAMBDA_AE_VALUES:
        metrics, _ = _train_variant(lambda_ae, train_sequences, test_sequences, test_omegas, test_labels)
        rows.append(metrics)
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)
    interpretation = _interpret(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "interpretation": interpretation})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _plot_metric(rows, "rollout_mse", "rollout_mse_vs_lambda_ae.png", "rollout MSE", log_y=True)
    _plot_metric(rows, "state_autoencoder_mse", "state_autoencoder_mse_vs_lambda_ae.png", "state autoencoder MSE", log_y=True)
    _plot_metric(rows, "latent_transition_mse", "latent_transition_mse_vs_lambda_ae.png", "latent transition MSE", log_y=True)
    _plot_metric(rows, "omega_r2_from_d", "omega_r2_vs_lambda_ae.png", "omega R2 from d")
    _write_html(rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
