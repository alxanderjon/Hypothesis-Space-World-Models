"""Multi-step latent Koopman consistency diagnostic."""

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
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    POSITION_WINDOW_SIZE,
    ROLLOUT_STEPS,
    SEED,
    STATE_LATENT_DIM,
    TEST_DATA_SEED,
    TOTAL_SEQUENCE_STEPS,
    TRAIN_DATA_SEED,
    SeparatedKoopmanWithDynamicsEncoder,
    _attention_stats,
    _fixed_dynamics_windows,
    _fixed_rollout_chunks,
    _make_position_data,
    _multioutput_r2,
    _parameter_count,
    _positions_to_states,
    _sample_dynamics_windows,
    _sample_windows,
    _save_attention_heatmap,
    _save_latent_pca,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/multistep_latent_koopman_consistency")
NUM_DYNAMICS_WINDOWS = 32
LATENT_LOSS_K = [1, 2, 4, 8, 16]
EVAL_K = [1, 2, 4, 8, 16, 32]
LAMBDA_LATENT_MULTI_VALUES = [1e-3, 1e-2, 1e-1]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _linear_regression_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    return float(model.score(x, y)), float(np.corrcoef(prediction.reshape(-1), y.reshape(-1))[0, 1])


def _sample_training_batch(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_trajectories = sequences.shape[0]
    indices = torch.randperm(num_trajectories)[:BATCH_SIZE]
    max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    max_rollout_start = TOTAL_SEQUENCE_STEPS - ROLLOUT_STEPS - 2
    starts = torch.randint(0, max_window_start + 1, (indices.shape[0], NUM_DYNAMICS_WINDOWS))
    rollout_starts = torch.randint(0, max_rollout_start + 1, (indices.shape[0],))
    return (
        _sample_dynamics_windows(sequences, indices, starts),
        _sample_windows(sequences, indices, rollout_starts, ROLLOUT_STEPS + 2),
    )


def _latent_k_step_errors(
    model: SeparatedKoopmanWithDynamicsEncoder,
    h_states: torch.Tensor,
    k_matrix: torch.Tensor,
    k_values: list[int],
) -> dict[int, torch.Tensor]:
    errors = {}
    for k in k_values:
        k_power = torch.linalg.matrix_power(k_matrix, k)
        h_pred = torch.einsum("nij,ntj->nti", k_power, h_states[:, :-k])
        errors[k] = mse_loss(h_pred, h_states[:, k:])
    return errors


def _losses(
    model: SeparatedKoopmanWithDynamicsEncoder,
    dynamics_windows: torch.Tensor,
    rollout_positions: torch.Tensor,
    *,
    lambda_latent_multi: float,
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
    latent_errors = _latent_k_step_errors(model, h_true, k_matrix, LATENT_LOSS_K)
    latent_multi_mse = sum(latent_errors.values())
    total_loss = rollout_state_mse + lambda_latent_multi * latent_multi_mse
    parts = {
        "rollout_state_mse": rollout_state_mse,
        "rollout_mse": rollout_mse,
        "one_step_mse": one_step_mse,
        "two_step_mse": two_step_mse,
        "latent_multi_mse": latent_multi_mse,
        "d_variance": d.var(dim=0, unbiased=False).mean(),
        "h_variance": h_true.var(dim=(0, 1), unbiased=False).mean(),
    }
    for k, value in latent_errors.items():
        parts[f"latent_k{k}_mse"] = value
    return total_loss, parts, d


def _spectral_summary(k_matrix: torch.Tensor, decoded_k16_error: torch.Tensor) -> dict[str, float]:
    eigvals = torch.linalg.eigvals(k_matrix).detach().cpu().numpy()
    k_np = k_matrix.detach().cpu().numpy()
    spectral_radius = np.max(np.abs(eigvals), axis=1)
    determinant = np.linalg.det(k_np)
    trace = np.trace(k_np, axis1=1, axis2=2)
    condition_number = np.linalg.cond(k_np)
    complex_pair = np.any(np.abs(np.imag(eigvals)) > 1e-7, axis=1)
    rollout_np = decoded_k16_error.detach().cpu().numpy()
    radius_error = np.abs(spectral_radius - 1.0)
    corr = float(np.corrcoef(radius_error, rollout_np)[0, 1]) if np.std(radius_error) > 0 and np.std(rollout_np) > 0 else float("nan")
    return {
        "spectral_radius_mean": float(np.mean(spectral_radius)),
        "spectral_radius_std": float(np.std(spectral_radius)),
        "spectral_radius_error_from_1_mean": float(np.mean(radius_error)),
        "determinant_mean": float(np.mean(determinant)),
        "trace_mean": float(np.mean(trace)),
        "condition_number_mean": float(np.mean(condition_number)),
        "fraction_complex_eigenvalues": float(np.mean(complex_pair)),
        "corr_spectral_radius_error_with_k16_rollout": corr,
    }


def _evaluate(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_latent_multi: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, NUM_DYNAMICS_WINDOWS)
        rollout_positions = _fixed_rollout_chunks(sequences)
        _, parts, d = _losses(model, dynamics_windows, rollout_positions, lambda_latent_multi=lambda_latent_multi)
        k_matrix = model.koopman(d)
        states = _positions_to_states(sequences)
        h_states = model.encode_state(states.reshape(-1, 2)).reshape(states.shape[0], states.shape[1], STATE_LATENT_DIM)
        k_metrics: dict[str, float] = {}
        decoded_k16_error = None
        for k in EVAL_K:
            k_power = torch.linalg.matrix_power(k_matrix, k)
            h_pred = torch.einsum("nij,ntj->nti", k_power, h_states[:, :-k])
            h_target = h_states[:, k:]
            decoded = model.decode_state(h_pred.reshape(-1, STATE_LATENT_DIM)).reshape(h_pred.shape[0], h_pred.shape[1], 2)
            state_target = states[:, k:]
            latent_by_traj = (h_pred - h_target).square().mean(dim=(1, 2))
            decoded_by_traj = (decoded - state_target).square().mean(dim=(1, 2))
            k_metrics[f"eval_latent_k{k}_mse"] = latent_by_traj.mean().item()
            k_metrics[f"eval_decoded_k{k}_mse"] = decoded_by_traj.mean().item()
            if k == 16:
                decoded_k16_error = (decoded[..., 1] - state_target[..., 1]).square().mean(dim=1)
        if decoded_k16_error is None:
            raise RuntimeError("missing k=16 decoded error")
        attention_weights = model.dynamics_attention_weights(dynamics_windows)
    omega_r2, omega_corr = _linear_regression_stats(d, omegas)
    metrics = {
        "rollout_mse": parts["rollout_mse"].item(),
        "rollout_state_mse": parts["rollout_state_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "latent_multi_mse": parts["latent_multi_mse"].item(),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
        "h_variance": parts["h_variance"].item(),
        **{f"train_{key}": value.item() for key, value in parts.items() if key.startswith("latent_k")},
        **k_metrics,
        **_spectral_summary(k_matrix, decoded_k16_error),
    }
    return metrics, d, attention_weights


def _train_variant(
    lambda_latent_multi: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = f"lambda_latent_multi_{lambda_latent_multi:g}".replace(".", "p").replace("-", "m")
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
            "lambda_latent_multi": lambda_latent_multi,
            "latent_loss_k": LATENT_LOSS_K,
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "seed": SEED,
            "train_data_seed": TRAIN_DATA_SEED,
            "test_data_seed": TEST_DATA_SEED,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "loss": "L_rollout + lambda_latent_multi * sum_k ||K^k H(s_t)-H(s_{t+k})||^2",
            "state_autoencoder_loss": False,
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
        dynamics_windows, rollout_positions = _sample_training_batch(train_sequences)
        optimizer.zero_grad()
        loss, train_parts, _ = _losses(
            model,
            dynamics_windows,
            rollout_positions,
            lambda_latent_multi=lambda_latent_multi,
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
                lambda_latent_multi=lambda_latent_multi,
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
                    "train_latent_multi_mse": train_parts["latent_multi_mse"].item(),
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
        lambda_latent_multi=lambda_latent_multi,
    )
    metrics.update(
        {
            "run_name": run_name,
            "lambda_latent_multi": lambda_latent_multi,
            "encoder_type": "attention",
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
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


def _plot_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log_y: bool = False) -> None:
    sorted_rows = sorted(rows, key=lambda row: row["lambda_latent_multi"])
    plt.figure(figsize=(7.2, 4.6))
    plt.plot([row["lambda_latent_multi"] for row in sorted_rows], [row[metric] for row in sorted_rows], marker="o")
    plt.xscale("log")
    if log_y:
        plt.yscale("log")
    plt.xlabel("lambda_latent_multi")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _write_html(rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "run_name",
        "lambda_latent_multi",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "eval_latent_k1_mse",
        "eval_latent_k2_mse",
        "eval_latent_k4_mse",
        "eval_latent_k8_mse",
        "eval_latent_k16_mse",
        "eval_latent_k32_mse",
        "omega_r2_from_d",
        "abomega_multioutput_r2_from_d",
        "spectral_radius_error_from_1_mean",
        "corr_spectral_radius_error_with_k16_rollout",
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
  <title>Multi-step Latent Koopman Consistency</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Multi-step Latent Koopman Consistency</h1>
  <p>Attention separated Koopman, m=32, position-only. No AE loss and no spectral loss.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>Rollout MSE</h2><img src="rollout_mse_vs_lambda.png">
  <h2>Latent k16 MSE</h2><img src="latent_k16_mse_vs_lambda.png">
  <h2>Omega R2</h2><img src="omega_r2_vs_lambda.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    best = min(rows, key=lambda row: row["rollout_mse"])
    best_latent = min(rows, key=lambda row: row["eval_latent_k16_mse"])
    return [
        f"Best rollout is {best['run_name']} with rollout MSE {best['rollout_mse']:.4e}.",
        f"Best k=16 latent consistency is {best_latent['run_name']} with MSE {best_latent['eval_latent_k16_mse']:.4e}.",
        "Compare this against the no-AE m=32 baseline rollout MSE 6.0707e-07 and the best AE diagnostic rollout MSE 9.9920e-07.",
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    rows: list[dict[str, Any]] = []
    for lambda_latent_multi in LAMBDA_LATENT_MULTI_VALUES:
        metrics, _ = _train_variant(lambda_latent_multi, train_sequences, test_sequences, test_omegas, test_labels)
        rows.append(metrics)
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)
    interpretation = _interpret(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "interpretation": interpretation})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _plot_metric(rows, "rollout_mse", "rollout_mse_vs_lambda.png", "rollout MSE", log_y=True)
    _plot_metric(rows, "eval_latent_k16_mse", "latent_k16_mse_vs_lambda.png", "latent k=16 MSE", log_y=True)
    _plot_metric(rows, "omega_r2_from_d", "omega_r2_vs_lambda.png", "omega R2 from d")
    _write_html(rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
