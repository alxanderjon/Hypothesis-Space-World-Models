"""Seed stability check for Branch B dense Koopman with latent_dim=16."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/dense_koopman_dim16_seed_stability")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 16
ENCODER_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200_000
PATIENCE = 25_000
EVAL_EVERY = 10_000
TRAINING_SEEDS = [0, 1, 2]

ElementwiseLinear = nn.Linear


class BranchBDenseKoopmanModel(nn.Module):
    """Dense trajectory-conditioned 2x2 Koopman architecture."""

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


def _sample_chunks(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(POSITION_WINDOW_SIZE + ROLLOUT_STEPS, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_chunks(sequences, indices, starts)


def _losses(model: BranchBDenseKoopmanModel, chunks: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z = model.encode(windows)
    params = model.operator_params(z)
    params_one = params.unsqueeze(1).expand(-1, chunks.shape[1] - 2, -1)
    one_step = model.step(chunks[:, :-2], chunks[:, 1:-1], params_one)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])
    params_two = params.unsqueeze(1).expand(-1, chunks.shape[1] - 3, -1)
    first = model.step(chunks[:, :-3], chunks[:, 1:-2], params_two)
    second = model.step(chunks[:, 1:-2], first, params_two)
    two_step_loss = mse_loss(second, chunks[:, 3:])
    rollout = model.rollout(windows[:, -2], windows[:, -1], params, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])
    return one_step_loss + two_step_loss + rollout_loss, {
        "one_step_mse": one_step_loss,
        "two_step_mse": two_step_loss,
        "rollout_mse": rollout_loss,
    }, z


def _regression_r2(features: np.ndarray, targets: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(features, targets)
    return float(model.score(features, targets))


def _omega_stats(features: np.ndarray, omegas: np.ndarray) -> tuple[float, float]:
    model = LinearRegression()
    model.fit(features, omegas)
    prediction = model.predict(features)
    return float(model.score(features, omegas)), float(np.corrcoef(prediction.reshape(-1), omegas.reshape(-1))[0, 1])


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate(
    model: BranchBDenseKoopmanModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z = _losses(model, chunks)
    z_np = z.detach().cpu().numpy()
    omega_np = omegas.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    omega_r2, omega_corr = _omega_stats(z_np, omega_np)
    return {
        "one_step_mse": losses["one_step_mse"].item(),
        "two_step_mse": losses["two_step_mse"].item(),
        "rollout_mse": losses["rollout_mse"].item(),
        "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
        "omega_r2_from_latent": omega_r2,
        "omega_correlation_from_latent": omega_corr,
        "abomega_multioutput_r2_from_latent": _regression_r2(z_np, labels_np),
    }, z


def _pca_stats(z: torch.Tensor, omegas: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    z_np = z.detach().cpu().numpy()
    omega_np = omegas.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    pca = PCA(n_components=LATENT_DIM)
    scores = pca.fit_transform(z_np)
    explained = pca.explained_variance_ratio_
    rows: list[dict[str, Any]] = []
    for count in range(1, LATENT_DIM + 1):
        subset = scores[:, :count]
        omega_r2, omega_corr = _omega_stats(subset, omega_np)
        rows.append(
            {
                "num_pcs": count,
                "cumulative_explained_variance": float(np.sum(explained[:count])),
                "omega_r2_from_pcs": omega_r2,
                "omega_correlation_from_pcs": omega_corr,
                "abomega_r2_from_pcs": _regression_r2(subset, labels_np),
            },
        )
    summary = {
        "pca_explained_variance": [float(value) for value in explained],
        "used_variance_directions_gt_1pct": int(np.sum(explained > 0.01)),
        "directions_for_95pct_variance": int(np.searchsorted(np.cumsum(explained), 0.95) + 1),
        "directions_for_99pct_variance": int(np.searchsorted(np.cumsum(explained), 0.99) + 1),
    }
    return summary, rows


def _save_latent_scatter(path: Path, z: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
    z_np = z.detach().cpu().numpy()
    xy = PCA(n_components=2).fit_transform(z_np)
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=omegas.cpu().numpy(), cmap="viridis", s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _train_one_seed(
    training_seed: int,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict[str, Any]:
    run_dir = OUTPUT_DIR / f"seed_{training_seed:03d}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(training_seed)
    model = BranchBDenseKoopmanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    history: list[dict[str, Any]] = []

    _write_json(
        run_dir / "setup.json",
        {
            "training_seed": training_seed,
            "data_train_seed": SEED,
            "data_test_seed": SEED + 1,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "latent_dim": LATENT_DIM,
            "branch": "branch_b_dense_koopman",
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "spectral_loss": False,
            "contrastive_loss": False,
            "losses": "L_1step + L_2step + L_rollout",
        },
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_sequences.shape[0])
        train_loss_sum = 0.0
        train_count = 0
        for start_index in range(0, train_sequences.shape[0], BATCH_SIZE):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            starts = torch.randint(0, max_start + 1, (indices.shape[0],))
            chunks = _sample_chunks(train_sequences, indices, starts)
            optimizer.zero_grad()
            loss, _, _ = _losses(model, chunks)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * indices.shape[0]
            train_count += indices.shape[0]
        if stopped_because == "nan_or_inf_loss":
            break

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, _ = _evaluate(model, test_sequences, test_omegas, test_labels)
            rollout_mse = eval_metrics["rollout_mse"]
            if rollout_mse < best_rollout:
                best_rollout = rollout_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            history.append({"epoch": epoch, "train_loss": train_loss_sum / train_count, **eval_metrics})
            _write_json(run_dir / "training_history.json", {"history": history})
            _write_csv(run_dir / "history.csv", history)
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, z = _evaluate(model, test_sequences, test_omegas, test_labels)
    pca_summary, pca_rows = _pca_stats(z, test_omegas, test_labels)
    metrics.update(
        {
            "training_seed": training_seed,
            "latent_dim": LATENT_DIM,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "parameter_count": _parameter_count(model),
            **pca_summary,
        },
    )
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _write_json(metrics_path, metrics)
    _write_json(run_dir / "training_history.json", {"history": history})
    _write_csv(run_dir / "history.csv", history)
    _write_csv(run_dir / "pca_recoverability.csv", [{"training_seed": training_seed, **row} for row in pca_rows])
    _save_latent_scatter(run_dir / "latent_pca_colored_by_omega.png", z, test_omegas, f"Dense Koopman dim16 seed {training_seed}")
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "latent_variance",
        "used_variance_directions_gt_1pct",
        "directions_for_95pct_variance",
        "directions_for_99pct_variance",
    ]
    aggregate_rows = []
    for metric in metrics:
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        aggregate_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            },
        )
    return aggregate_rows


def _save_plot(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(6.5, 4.2))
    xs = [row["training_seed"] for row in rows]
    ys = [row[metric] for row in rows]
    plt.plot(xs, ys, marker="o")
    if log_y:
        plt.yscale("log")
    plt.xlabel("training seed")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_html(rows: list[dict[str, Any]], aggregate_rows: list[dict[str, Any]]) -> None:
    columns = [
        "training_seed",
        "rollout_mse",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "used_variance_directions_gt_1pct",
        "directions_for_95pct_variance",
        "directions_for_99pct_variance",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    aggregate_columns = ["metric", "mean", "std", "min", "max"]

    def table(table_columns: list[str], table_rows: list[dict[str, Any]]) -> str:
        header = "".join(f"<th>{escape(column)}</th>" for column in table_columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in table_columns) + "</tr>"
            for row in table_rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dense Koopman latent_dim=16 Seed Stability</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Dense Koopman latent_dim=16 Seed Stability</h1>
  <p>Branch B dense trajectory-conditioned Koopman. Same position-only data, losses, architecture, max_epochs=200000, patience=25000. Only training/model random seed changes.</p>
  <h2>Seed Results</h2>
  {table(columns, rows)}
  <h2>Aggregate</h2>
  {table(aggregate_columns, aggregate_rows)}
  <h2>Plots</h2>
  <h3>Rollout MSE</h3><img src="rollout_mse_by_seed.png">
  <h3>Omega R2</h3><img src="omega_r2_by_seed.png">
  <h3>(a,b,omega) R2</h3><img src="abomega_r2_by_seed.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    rows = [
        _train_one_seed(training_seed, train_sequences, test_sequences, test_omegas, test_labels)
        for training_seed in TRAINING_SEEDS
    ]
    aggregate_rows = _aggregate(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "aggregate": aggregate_rows})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "aggregate.csv", aggregate_rows)
    _save_plot(rows, "rollout_mse", "rollout_mse_by_seed.png", "rollout MSE", log_y=True)
    _save_plot(rows, "omega_r2_from_latent", "omega_r2_by_seed.png", "omega R2 from latent")
    _save_plot(rows, "abomega_multioutput_r2_from_latent", "abomega_r2_by_seed.png", "(a,b,omega) R2 from latent")
    _write_html(rows, aggregate_rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "aggregate": aggregate_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
