"""Branch B dense Koopman diagnostic with latent and transition consistency."""

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

from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    SEED,
    _make_base_conditions,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/dense_koopman_latent_consistency")
BASELINE_DIR = Path("outputs/sho/stage1B/composition_long_training_comparison/branch_b_dense_koopman")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 4
ENCODER_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50_000
PATIENCE = 10_000
EVAL_EVERY = 1_000
LOCAL_WINDOWS_PER_TRAJECTORY = 4
LAMBDA_SAME = 1e-2
LAMBDA_DIFF = 1e-2
CONTRASTIVE_MARGIN = 0.5

ElementwiseLinear = nn.Linear


class DenseKoopmanModel(nn.Module):
    """Existing dense trajectory-conditioned 2x2 Koopman architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, ENCODER_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(ENCODER_HIDDEN_SIZE, LATENT_DIM),
        )
        self.generator = nn.Sequential(
            ElementwiseLinear(LATENT_DIM, GENERATOR_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(GENERATOR_HIDDEN_SIZE, 4),
        )

    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def operator_params(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        _, _, c, d = params.unbind(dim=-1)
        return c * x_prev + d * x_curr

    def rollout(self, x_prev: torch.Tensor, x_curr: torch.Tensor, params: torch.Tensor, steps: int) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, params)
            predictions.append(next_x)
            prev = curr
            curr = next_x
        return torch.stack(predictions, dim=-1)


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


def _make_position_data(num_trajectories: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    times = torch.arange(TOTAL_SEQUENCE_STEPS, dtype=initial_conditions.dtype) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    labels = torch.stack((initial_conditions[:, 0], initial_conditions[:, 1] / omegas, omegas), dim=-1)
    return positions, omegas, labels


def _sample_windows(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    offsets = torch.arange(length, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_windows(sequences, indices, starts, POSITION_WINDOW_SIZE + ROLLOUT_STEPS)


def _reconstruction_from_windows(
    model: DenseKoopmanModel,
    windows: torch.Tensor,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    z = model.encode(windows)
    params = model.operator_params(z)
    params_one = params.unsqueeze(1).expand(-1, chunks.shape[1] - 2, -1)
    one_step = model.step(chunks[:, :-2], chunks[:, 1:-1], params_one)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])

    params_two = params.unsqueeze(1).expand(-1, chunks.shape[1] - 3, -1)
    first = model.step(chunks[:, :-3], chunks[:, 1:-2], params_two)
    second = model.step(chunks[:, 1:-2], first, params_two)
    two_step_loss = mse_loss(second, chunks[:, 3:])

    rollout = model.rollout(chunks[:, POSITION_WINDOW_SIZE - 2], chunks[:, POSITION_WINDOW_SIZE - 1], params, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])
    return one_step_loss + two_step_loss + rollout_loss, {
        "one_step_mse": one_step_loss,
        "two_step_mse": two_step_loss,
        "rollout_mse": rollout_loss,
    }, z


def _sample_training_batch(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_trajectories = sequences.shape[0]
    indices = torch.randperm(num_trajectories)[:BATCH_SIZE]
    max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    max_chunk_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    starts_a = torch.randint(0, max_window_start + 1, (indices.shape[0],))
    starts_b = torch.randint(0, max_window_start + 1, (indices.shape[0],))
    diff_indices = (indices + torch.randint(1, num_trajectories, (indices.shape[0],))) % num_trajectories
    starts_c = torch.randint(0, max_window_start + 1, (indices.shape[0],))
    local_starts = torch.randint(0, max_chunk_start + 1, (indices.shape[0], LOCAL_WINDOWS_PER_TRAJECTORY))

    window_a = _sample_windows(sequences, indices, starts_a, POSITION_WINDOW_SIZE)
    window_b = _sample_windows(sequences, indices, starts_b, POSITION_WINDOW_SIZE)
    window_c = _sample_windows(sequences, diff_indices, starts_c, POSITION_WINDOW_SIZE)

    expanded_indices = indices.unsqueeze(1).expand(-1, LOCAL_WINDOWS_PER_TRAJECTORY).reshape(-1)
    expanded_starts = local_starts.reshape(-1)
    chunks = _sample_windows(sequences, expanded_indices, expanded_starts, POSITION_WINDOW_SIZE + ROLLOUT_STEPS)
    return window_a, window_b, window_c, chunks


def _consistency_losses(
    model: DenseKoopmanModel,
    window_a: torch.Tensor,
    window_b: torch.Tensor,
    window_c: torch.Tensor,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    z_a = model.encode(window_a)
    z_b = model.encode(window_b)
    z_c = model.encode(window_c)

    chunks_grouped = chunks.reshape(window_a.shape[0], LOCAL_WINDOWS_PER_TRAJECTORY, POSITION_WINDOW_SIZE + ROLLOUT_STEPS)
    flat_chunks = chunks_grouped.reshape(-1, POSITION_WINDOW_SIZE + ROLLOUT_STEPS)
    flat_windows = window_a.unsqueeze(1).expand(-1, LOCAL_WINDOWS_PER_TRAJECTORY, -1).reshape(-1, POSITION_WINDOW_SIZE)
    reconstruction_loss, reconstruction_parts, _ = _reconstruction_from_windows(model, flat_windows, flat_chunks)

    same_dist = torch.linalg.norm(z_a - z_b, dim=-1)
    diff_dist = torch.linalg.norm(z_a - z_c, dim=-1)
    latent_same_loss = same_dist.square().mean()
    latent_diff_loss = torch.relu(CONTRASTIVE_MARGIN - diff_dist).square().mean()
    total_loss = reconstruction_loss + LAMBDA_SAME * latent_same_loss + LAMBDA_DIFF * latent_diff_loss
    return total_loss, {
        **reconstruction_parts,
        "reconstruction_loss": reconstruction_loss,
        "latent_same_loss": latent_same_loss,
        "latent_diff_loss": latent_diff_loss,
        "same_latent_distance": same_dist.mean(),
        "different_latent_distance": diff_dist.mean(),
        "contrastive_margin_satisfaction_rate": (diff_dist > CONTRASTIVE_MARGIN).float().mean(),
    }, z_a


def _linear_regression_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
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


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate(
    model: DenseKoopmanModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        _, losses, z = _reconstruction_from_windows(model, windows, chunks)

        max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
        indices = torch.arange(sequences.shape[0])
        starts_a = torch.zeros(sequences.shape[0], dtype=torch.long)
        starts_b = torch.full((sequences.shape[0],), max_window_start, dtype=torch.long)
        diff_indices = (indices + 1) % sequences.shape[0]
        window_a = _sample_windows(sequences, indices, starts_a, POSITION_WINDOW_SIZE)
        window_b = _sample_windows(sequences, indices, starts_b, POSITION_WINDOW_SIZE)
        window_c = _sample_windows(sequences, diff_indices, starts_a, POSITION_WINDOW_SIZE)
        z_a = model.encode(window_a)
        z_b = model.encode(window_b)
        z_c = model.encode(window_c)
        same_dist = torch.linalg.norm(z_a - z_b, dim=-1)
        diff_dist = torch.linalg.norm(z_a - z_c, dim=-1)

    omega_r2, omega_corr = _linear_regression_stats(z, omegas)
    return {
        "one_step_mse": losses["one_step_mse"].item(),
        "two_step_mse": losses["two_step_mse"].item(),
        "rollout_mse": losses["rollout_mse"].item(),
        "omega_r2_from_latent": omega_r2,
        "omega_correlation_from_latent": omega_corr,
        "abomega_multioutput_r2_from_latent": _multioutput_r2(z, labels),
        "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
        "same_latent_distance": same_dist.mean().item(),
        "different_latent_distance": diff_dist.mean().item(),
        "contrastive_margin_satisfaction_rate": (diff_dist > CONTRASTIVE_MARGIN).float().mean().item(),
    }, z


def _save_latent_pca(path: Path, z: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
    z_np = z.detach().cpu().numpy()
    if z_np.shape[1] >= 2:
        xy = PCA(n_components=2).fit_transform(z_np)
    else:
        xy = np.column_stack((z_np[:, 0], np.zeros(z_np.shape[0])))
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=omegas.cpu().numpy(), cmap="viridis", s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_metric_plot(history: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row[metric] for row in history], marker="o", label="latent consistency")
    baseline_history_path = BASELINE_DIR / "history.csv"
    if baseline_history_path.exists():
        baseline_rows = _read_csv_dicts(baseline_history_path)
        if metric in baseline_rows[0]:
            plt.plot(
                [float(row["epoch"]) for row in baseline_rows],
                [float(row[metric]) for row in baseline_rows],
                marker="o",
                label="dense Koopman baseline",
            )
    if log_y:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _save_distance_plot(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row["same_latent_distance"] for row in history], marker="o", label="same trajectory")
    plt.plot(
        [row["epoch"] for row in history],
        [row["different_latent_distance"] for row in history],
        marker="o",
        label="different trajectory",
    )
    plt.axhline(CONTRASTIVE_MARGIN, color="gray", linestyle="--", linewidth=1, label="margin")
    plt.xlabel("epoch")
    plt.ylabel("latent distance")
    plt.title("Same/Different Latent Distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "same_different_latent_distance_vs_epoch.png")
    plt.close()


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _load_baseline_metrics() -> dict[str, Any] | None:
    path = BASELINE_DIR / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_html(metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> None:
    columns = [
        "run_name",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "omega_r2_from_latent",
        "abomega_multioutput_r2_from_latent",
        "same_latent_distance",
        "different_latent_distance",
        "contrastive_margin_satisfaction_rate",
        "latent_variance",
        "parameter_count",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    rows = [{**metrics, "run_name": "dense_koopman_latent_consistency"}]
    if baseline_metrics is not None:
        rows.append({**baseline_metrics, "run_name": "dense_koopman_baseline_50k"})
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dense Koopman Latent Consistency Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 920px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Dense Koopman Latent Consistency Diagnostic</h1>
  <p>Branch B only. Position-only, no omega labels, no coefficient labels, no analytic SHO decoder, no spectral loss. Contrastive margin: {CONTRASTIVE_MARGIN}. lambda_same={LAMBDA_SAME}, lambda_diff={LAMBDA_DIFF}. Reconstruction averages {LOCAL_WINDOWS_PER_TRAJECTORY} local windows per trajectory.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Rollout MSE vs Epoch</h2><img src="rollout_mse_vs_epoch.png">
  <h2>Omega R2 vs Epoch</h2><img src="omega_r2_vs_epoch.png">
  <h2>Same/Different Latent Distance</h2><img src="same_different_latent_distance_vs_epoch.png">
  <h2>Latent PCA Colored by Omega</h2><img src="latent_pca_colored_by_omega.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    model = DenseKoopmanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    history: list[dict[str, Any]] = []

    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "branch": "branch_b_dense_koopman",
            "latent_dim": LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "local_windows_per_trajectory": LOCAL_WINDOWS_PER_TRAJECTORY,
            "lambda_same": LAMBDA_SAME,
            "lambda_diff": LAMBDA_DIFF,
            "contrastive_margin": CONTRASTIVE_MARGIN,
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
            "losses": "L_reconstruction + lambda_same * L_latent_same + lambda_diff * L_latent_diff",
        },
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        window_a, window_b, window_c, chunks = _sample_training_batch(train_sequences)
        optimizer.zero_grad()
        loss, train_parts, _ = _consistency_losses(model, window_a, window_b, window_c, chunks)
        if not torch.isfinite(loss):
            stopped_because = "nan_or_inf_loss"
            break
        loss.backward()
        optimizer.step()

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, z = _evaluate(model, test_sequences, test_omegas, test_labels)
            rollout_mse = eval_metrics["rollout_mse"]
            if rollout_mse < best_rollout:
                best_rollout = rollout_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            row = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_reconstruction_loss": train_parts["reconstruction_loss"].item(),
                "train_latent_same_loss": train_parts["latent_same_loss"].item(),
                "train_latent_diff_loss": train_parts["latent_diff_loss"].item(),
                **eval_metrics,
            }
            history.append(row)
            _write_json(OUTPUT_DIR / "training_history.json", {"history": history})
            _write_csv(OUTPUT_DIR / "history.csv", history)
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, z = _evaluate(model, test_sequences, test_omegas, test_labels)
    metrics.update(
        {
            "run_name": "dense_koopman_latent_consistency",
            "latent_dim": LATENT_DIM,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "lambda_same": LAMBDA_SAME,
            "lambda_diff": LAMBDA_DIFF,
            "contrastive_margin": CONTRASTIVE_MARGIN,
            "local_windows_per_trajectory": LOCAL_WINDOWS_PER_TRAJECTORY,
            "parameter_count": _parameter_count(model),
        },
    )
    torch.save(model.state_dict(), OUTPUT_DIR / "learned_parameters.pt")
    _write_json(OUTPUT_DIR / "metrics.json", metrics)
    _write_json(OUTPUT_DIR / "training_history.json", {"history": history})
    _write_csv(OUTPUT_DIR / "history.csv", history)
    _save_metric_plot(history, "rollout_mse", "rollout_mse_vs_epoch.png", "rollout MSE", log_y=True)
    _save_metric_plot(history, "omega_r2_from_latent", "omega_r2_vs_epoch.png", "omega R2 from latent")
    _save_distance_plot(history)
    _save_latent_pca(OUTPUT_DIR / "latent_pca_colored_by_omega.png", z, test_omegas, "Dense Koopman latent consistency")
    baseline_metrics = _load_baseline_metrics()
    _write_json(OUTPUT_DIR / "summary.json", {"run": metrics, "baseline": baseline_metrics})
    _write_csv(OUTPUT_DIR / "summary.csv", [metrics])
    _write_html(metrics, baseline_metrics)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "run": metrics, "baseline": baseline_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
