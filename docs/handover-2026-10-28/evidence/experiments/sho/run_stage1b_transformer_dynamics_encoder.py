"""Compare mean-pooled and attention-pooled dynamics encoders for separated Koopman."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/transformer_dynamics_encoder")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
DYNAMICS_LATENT_DIM = 16
STATE_LATENT_DIM = 2
DYNAMICS_HIDDEN_SIZE = 16
STATE_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
TRANSFORMER_FF_DIM = 32
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100_000
PATIENCE = 20_000
EVAL_EVERY = 1_000
LAMBDA_LATENT = 1e-2
LAMBDA_DYN = 0.0
VARIANTS = [
    {
        "run_name": "mean_pool_3_windows",
        "encoder_type": "mean_pool",
        "num_dynamics_windows": 3,
    },
    {
        "run_name": "transformer_attention_8_windows",
        "encoder_type": "transformer_attention",
        "num_dynamics_windows": 8,
    },
]


class MeanDynamicsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Linear(POSITION_WINDOW_SIZE, DYNAMICS_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_LATENT_DIM),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        embeddings = self.window_encoder(windows)
        return embeddings.mean(dim=1)


class TransformerAttentionDynamicsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Linear(POSITION_WINDOW_SIZE, DYNAMICS_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_LATENT_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=DYNAMICS_LATENT_DIM,
            nhead=4,
            dim_feedforward=TRANSFORMER_FF_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.attention_score = nn.Linear(DYNAMICS_LATENT_DIM, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        embeddings = self.window_encoder(windows)
        contextual = self.transformer(embeddings)
        weights = torch.softmax(self.attention_score(contextual), dim=1)
        return (weights * contextual).sum(dim=1)


class SeparatedKoopmanWithDynamicsEncoder(nn.Module):
    def __init__(self, *, encoder_type: str) -> None:
        super().__init__()
        if encoder_type == "mean_pool":
            self.dynamics_encoder = MeanDynamicsEncoder()
        elif encoder_type == "transformer_attention":
            self.dynamics_encoder = TransformerAttentionDynamicsEncoder()
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type}")
        self.koopman_generator = nn.Sequential(
            nn.Linear(DYNAMICS_LATENT_DIM, GENERATOR_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(GENERATOR_HIDDEN_SIZE, STATE_LATENT_DIM * STATE_LATENT_DIM),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(2, STATE_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(STATE_HIDDEN_SIZE, STATE_LATENT_DIM),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(STATE_LATENT_DIM, STATE_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(STATE_HIDDEN_SIZE, 2),
        )

    def encode_dynamics(self, windows: torch.Tensor) -> torch.Tensor:
        return self.dynamics_encoder(windows)

    def koopman(self, dynamics_latent: torch.Tensor) -> torch.Tensor:
        return self.koopman_generator(dynamics_latent).reshape(-1, STATE_LATENT_DIM, STATE_LATENT_DIM)

    def encode_state(self, states: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(states)

    def decode_state(self, latent_states: torch.Tensor) -> torch.Tensor:
        return self.state_decoder(latent_states)

    def apply_koopman(self, h: torch.Tensor, k_matrix: torch.Tensor) -> torch.Tensor:
        return torch.bmm(k_matrix, h.unsqueeze(-1)).squeeze(-1)

    def rollout_latent(self, h0: torch.Tensor, k_matrix: torch.Tensor, steps: int) -> torch.Tensor:
        states = [h0]
        h = h0
        for _ in range(steps):
            h = self.apply_koopman(h, k_matrix)
            states.append(h)
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


def _sample_dynamics_windows(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    pieces = [
        _sample_windows(sequences, indices, starts[:, window_index], POSITION_WINDOW_SIZE)
        for window_index in range(starts.shape[1])
    ]
    return torch.stack(pieces, dim=1)


def _positions_to_states(positions: torch.Tensor) -> torch.Tensor:
    return torch.stack((positions[:, :-1], positions[:, 1:]), dim=-1)


def _fixed_dynamics_windows(sequences: torch.Tensor, num_windows: int) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    starts = torch.linspace(0, max_start, num_windows, device=sequences.device).round().long()
    starts = starts.unsqueeze(0).expand(sequences.shape[0], -1)
    return _sample_dynamics_windows(sequences, indices, starts)


def _fixed_rollout_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_windows(sequences, indices, starts, ROLLOUT_STEPS + 2)


def _sample_training_batch(sequences: torch.Tensor, num_windows: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_trajectories = sequences.shape[0]
    indices = torch.randperm(num_trajectories)[:BATCH_SIZE]
    max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    max_rollout_start = TOTAL_SEQUENCE_STEPS - ROLLOUT_STEPS - 2
    starts = torch.randint(0, max_window_start + 1, (indices.shape[0], num_windows))
    rollout_starts = torch.randint(0, max_rollout_start + 1, (indices.shape[0],))
    return (
        _sample_dynamics_windows(sequences, indices, starts),
        _sample_windows(sequences, indices, rollout_starts, ROLLOUT_STEPS + 2),
    )


def _losses(
    model: SeparatedKoopmanWithDynamicsEncoder,
    dynamics_windows: torch.Tensor,
    rollout_positions: torch.Tensor,
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
    total_loss = rollout_state_mse + LAMBDA_LATENT * latent_transition_mse
    return total_loss, {
        "rollout_state_mse": rollout_state_mse,
        "rollout_mse": rollout_mse,
        "one_step_mse": one_step_mse,
        "two_step_mse": two_step_mse,
        "latent_transition_mse": latent_transition_mse,
        "d_variance": d.var(dim=0, unbiased=False).mean(),
        "h_variance": h_true.var(dim=(0, 1), unbiased=False).mean(),
    }, d


def _linear_regression_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    return float(model.score(x, y)), float(np.corrcoef(prediction.reshape(-1), y.reshape(-1))[0, 1])


def _multioutput_r2(features: torch.Tensor, targets: torch.Tensor) -> float:
    model = LinearRegression()
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    model.fit(x, y)
    return float(model.score(x, y))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_windows: int,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, num_windows)
        rollout_positions = _fixed_rollout_chunks(sequences)
        _, parts, d = _losses(model, dynamics_windows, rollout_positions)
    omega_r2, omega_corr = _linear_regression_stats(d, omegas)
    return {
        "rollout_mse": parts["rollout_mse"].item(),
        "rollout_state_mse": parts["rollout_state_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "latent_transition_mse": parts["latent_transition_mse"].item(),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
        "h_variance": parts["h_variance"].item(),
    }, d


def _train_variant(
    variant: dict[str, Any],
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = str(variant["run_name"])
    encoder_type = str(variant["encoder_type"])
    num_windows = int(variant["num_dynamics_windows"])
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
    model = SeparatedKoopmanWithDynamicsEncoder(encoder_type=encoder_type)
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
            **variant,
            "lambda_latent": LAMBDA_LATENT,
            "lambda_dyn": LAMBDA_DYN,
            "dynamics_latent_dim": DYNAMICS_LATENT_DIM,
            "state_latent_dim": STATE_LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
            "explicit_dynamics_consistency": False,
        },
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        dynamics_windows, rollout_positions = _sample_training_batch(train_sequences, num_windows)
        optimizer.zero_grad()
        loss, train_parts, _ = _losses(model, dynamics_windows, rollout_positions)
        if not torch.isfinite(loss):
            stopped_because = "nan_or_inf_loss"
            break
        loss.backward()
        optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, d = _evaluate(
                model,
                test_sequences,
                test_omegas,
                test_labels,
                num_windows=num_windows,
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
    metrics, d = _evaluate(
        model,
        test_sequences,
        test_omegas,
        test_labels,
        num_windows=num_windows,
    )
    metrics.update(
        {
            "run_name": run_name,
            "encoder_type": encoder_type,
            "num_dynamics_windows": num_windows,
            "lambda_latent": LAMBDA_LATENT,
            "lambda_dyn": LAMBDA_DYN,
            "dynamics_latent_dim": DYNAMICS_LATENT_DIM,
            "state_latent_dim": STATE_LATENT_DIM,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "parameter_count": _parameter_count(model),
        },
    )
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _write_json(metrics_path, metrics)
    _write_json(history_path, {"history": history})
    _write_csv(run_dir / "history.csv", history)
    _save_latent_pca(run_dir / "d_latent_pca_colored_by_omega.png", d, test_omegas, f"d_i PCA {run_name}")
    return metrics, history


def _save_latent_pca(path: Path, d: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
    xy = PCA(n_components=2).fit_transform(d.detach().cpu().numpy())
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=omegas.cpu().numpy(), cmap="viridis", s=30)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_summary_plot(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(7.0, 4.4))
    labels = [row["encoder_type"].replace("_", "\n") for row in rows]
    values = [row[metric] for row in rows]
    plt.bar(range(len(rows)), values)
    plt.xticks(range(len(rows)), labels)
    if log_y:
        plt.yscale("log")
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


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    mean_row = next(row for row in rows if row["encoder_type"] == "mean_pool")
    attention_row = next(row for row in rows if row["encoder_type"] == "transformer_attention")
    lines = [
        f"Best rollout is {min(rows, key=lambda row: row['rollout_mse'])['encoder_type']}.",
        f"Best omega R2 is {max(rows, key=lambda row: row['omega_r2_from_d'])['encoder_type']}.",
    ]
    if attention_row["rollout_mse"] < mean_row["rollout_mse"]:
        lines.append("Attention over multiple windows improves rollout relative to the current mean-pooled dynamics encoder.")
    else:
        lines.append("Attention over multiple windows does not improve rollout relative to the current mean-pooled dynamics encoder.")
    if attention_row["omega_r2_from_d"] > mean_row["omega_r2_from_d"]:
        lines.append("The attention dynamics latent is more predictive of omega.")
    else:
        lines.append("The attention dynamics latent is not more predictive of omega.")
    return lines


def _write_html(rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "encoder_type",
        "num_dynamics_windows",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "latent_transition_mse",
        "omega_r2_from_d",
        "omega_correlation_from_d",
        "abomega_multioutput_r2_from_d",
        "d_variance",
        "h_variance",
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
  <title>Transformer Dynamics Encoder Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Transformer Dynamics Encoder Diagnostic</h1>
  <p>Separated dynamics/state Koopman setup unchanged: position-only, learned state latent, dense generated Koopman, latent transition consistency, no labels, no analytic decoder, no spectral loss, lambda_latent={LAMBDA_LATENT:g}, lambda_dyn=0.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>Rollout MSE</h2><img src="rollout_mse_by_encoder.png">
  <h2>Omega R2</h2><img src="omega_r2_by_encoder.png">
  <h2>Latent Transition MSE</h2><img src="latent_transition_by_encoder.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        metrics, history = _train_variant(
            variant,
            train_sequences,
            test_sequences,
            test_omegas,
            test_labels,
        )
        rows.append(metrics)
        histories[metrics["run_name"]] = history
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)
    interpretation = _interpret(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "interpretation": interpretation})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _save_summary_plot(rows, "rollout_mse", "rollout_mse_by_encoder.png", "rollout MSE", log_y=True)
    _save_summary_plot(rows, "omega_r2_from_d", "omega_r2_by_encoder.png", "omega R2 from d")
    _save_summary_plot(rows, "latent_transition_mse", "latent_transition_by_encoder.png", "latent transition MSE", log_y=True)
    _write_html(rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
