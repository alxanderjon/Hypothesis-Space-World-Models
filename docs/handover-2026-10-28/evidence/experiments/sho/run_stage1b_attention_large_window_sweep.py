"""Large-window sweep for attention dynamics encoder in separated Koopman."""

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
    _make_base_conditions,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/attention_large_window_sweep")
SEED = 0
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1000
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
    {"run_name": "attention_m32", "encoder_type": "attention", "num_dynamics_windows": 32},
    {"run_name": "attention_m64", "encoder_type": "attention", "num_dynamics_windows": 64},
    {"run_name": "attention_m128", "encoder_type": "attention", "num_dynamics_windows": 128},
]


class SingleWindowDynamicsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Linear(POSITION_WINDOW_SIZE, DYNAMICS_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_LATENT_DIM),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.window_encoder(windows[:, 0])

    def attention_weights(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.ones((windows.shape[0], windows.shape[1]), dtype=windows.dtype, device=windows.device)


class MeanDynamicsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Linear(POSITION_WINDOW_SIZE, DYNAMICS_HIDDEN_SIZE),
            nn.SiLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_LATENT_DIM),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.window_encoder(windows).mean(dim=1)

    def attention_weights(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (windows.shape[0], windows.shape[1]),
            1.0 / windows.shape[1],
            dtype=windows.dtype,
            device=windows.device,
        )


class AttentionDynamicsEncoder(nn.Module):
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

    def _contextual_and_weights(self, windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.window_encoder(windows)
        contextual = self.transformer(embeddings)
        weights = torch.softmax(self.attention_score(contextual), dim=1).squeeze(-1)
        return contextual, weights

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        contextual, weights = self._contextual_and_weights(windows)
        return (weights.unsqueeze(-1) * contextual).sum(dim=1)

    def attention_weights(self, windows: torch.Tensor) -> torch.Tensor:
        _, weights = self._contextual_and_weights(windows)
        return weights


class SeparatedKoopmanWithDynamicsEncoder(nn.Module):
    def __init__(self, *, encoder_type: str) -> None:
        super().__init__()
        if encoder_type == "single_window":
            self.dynamics_encoder = SingleWindowDynamicsEncoder()
        elif encoder_type == "mean_pool":
            self.dynamics_encoder = MeanDynamicsEncoder()
        elif encoder_type == "attention":
            self.dynamics_encoder = AttentionDynamicsEncoder()
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

    def dynamics_attention_weights(self, windows: torch.Tensor) -> torch.Tensor:
        return self.dynamics_encoder.attention_weights(windows)

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


def _linear_regression_prediction_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    rmse = float(np.sqrt(np.mean(np.square(prediction.reshape(-1) - y.reshape(-1)))))
    corr = float(np.corrcoef(prediction.reshape(-1), y.reshape(-1))[0, 1])
    return float(model.score(x, y)), corr, rmse


def _multioutput_r2(features: torch.Tensor, targets: torch.Tensor) -> float:
    model = LinearRegression()
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    model.fit(x, y)
    return float(model.score(x, y))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _attention_stats(weights: torch.Tensor) -> dict[str, Any]:
    weights_np = weights.detach().cpu().numpy()
    clipped = np.clip(weights_np, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    normalized_entropy = entropy / math.log(weights_np.shape[1]) if weights_np.shape[1] > 1 else np.ones_like(entropy)
    n_eff = 1.0 / np.square(weights_np).sum(axis=1)
    max_weight = weights_np.max(axis=1)
    return {
        "attention_entropy_mean": float(entropy.mean()),
        "attention_entropy_std": float(entropy.std(ddof=0)),
        "attention_normalized_entropy_mean": float(normalized_entropy.mean()),
        "attention_normalized_entropy_std": float(normalized_entropy.std(ddof=0)),
        "n_eff_mean": float(n_eff.mean()),
        "n_eff_std": float(n_eff.std(ddof=0)),
        "n_eff_min": float(n_eff.min()),
        "n_eff_max": float(n_eff.max()),
        "attention_max_weight_mean": float(max_weight.mean()),
        "attention_max_weight_std": float(max_weight.std(ddof=0)),
        "attention_mean_by_window": [float(value) for value in weights_np.mean(axis=0)],
        "attention_std_by_window": [float(value) for value in weights_np.std(axis=0, ddof=0)],
    }


def _oracle_observed_state_rollout(rollout_positions: torch.Tensor, omegas: torch.Tensor) -> torch.Tensor:
    true_states = _positions_to_states(rollout_positions)
    s_prev = true_states[:, 0, 0]
    s_curr = true_states[:, 0, 1]
    c = 2.0 * torch.cos(omegas * float(DELTA_T))
    predicted = [true_states[:, 0]]
    for _ in range(ROLLOUT_STEPS):
        s_next = -s_prev + c * s_curr
        predicted.append(torch.stack((s_curr, s_next), dim=-1))
        s_prev, s_curr = s_curr, s_next
    return torch.stack(predicted, dim=1)


def _error_decomposition(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    *,
    num_windows: int,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, num_windows)
        rollout_positions = _fixed_rollout_chunks(sequences)
        d = model.encode_dynamics(dynamics_windows)
        k_matrix = model.koopman(d)
        true_states = _positions_to_states(rollout_positions)

        h_true = model.encode_state(true_states.reshape(-1, 2)).reshape(
            true_states.shape[0],
            ROLLOUT_STEPS + 1,
            STATE_LATENT_DIM,
        )
        decoded_true_h = model.decode_state(h_true.reshape(-1, STATE_LATENT_DIM)).reshape_as(true_states)
        h_rollout = model.rollout_latent(h_true[:, 0], k_matrix, ROLLOUT_STEPS)
        decoded_rollout = model.decode_state(h_rollout.reshape(-1, STATE_LATENT_DIM)).reshape_as(true_states)

        k_expanded = k_matrix.unsqueeze(1).expand(-1, ROLLOUT_STEPS, -1, -1).reshape(
            -1,
            STATE_LATENT_DIM,
            STATE_LATENT_DIM,
        )
        h_pred_next = torch.bmm(k_expanded, h_true[:, :-1].reshape(-1, STATE_LATENT_DIM).unsqueeze(-1)).squeeze(-1)
        latent_transition_mse = mse_loss(h_pred_next, h_true[:, 1:].reshape(-1, STATE_LATENT_DIM))

        oracle_states = _oracle_observed_state_rollout(rollout_positions, omegas)
        learned_position_rollout_mse = mse_loss(decoded_rollout[:, 1:, 1], true_states[:, 1:, 1])
        autoencode_position_rollout_mse = mse_loss(decoded_true_h[:, 1:, 1], true_states[:, 1:, 1])
        autoencode_state_mse = mse_loss(decoded_true_h, true_states)
        oracle_position_rollout_mse = mse_loss(oracle_states[:, 1:, 1], true_states[:, 1:, 1])

    learned_value = learned_position_rollout_mse.item()
    autoencode_value = autoencode_position_rollout_mse.item()
    return {
        "decomp_learned_rollout_position_mse": learned_value,
        "decomp_state_autoencode_position_mse": autoencode_value,
        "decomp_state_autoencode_full_state_mse": autoencode_state_mse.item(),
        "decomp_latent_transition_mse": latent_transition_mse.item(),
        "decomp_oracle_observed_sho_rollout_mse": oracle_position_rollout_mse.item(),
        "decomp_rollout_minus_autoencode_mse": max(0.0, learned_value - autoencode_value),
        "decomp_autoencode_fraction_of_rollout": autoencode_value / learned_value if learned_value > 0 else float("nan"),
    }


def _evaluate(
    model: SeparatedKoopmanWithDynamicsEncoder,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_windows: int,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, num_windows)
        rollout_positions = _fixed_rollout_chunks(sequences)
        _, parts, d = _losses(model, dynamics_windows, rollout_positions)
        attention_weights = model.dynamics_attention_weights(dynamics_windows)
    omega_r2, omega_corr, omega_rmse = _linear_regression_prediction_stats(d, omegas)
    return {
        "rollout_mse": parts["rollout_mse"].item(),
        "rollout_state_mse": parts["rollout_state_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "latent_transition_mse": parts["latent_transition_mse"].item(),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "omega_rmse_from_d": omega_rmse,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
        "h_variance": parts["h_variance"].item(),
    }, d, attention_weights


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
            "seed": SEED,
            "train_data_seed": TRAIN_DATA_SEED,
            "test_data_seed": TEST_DATA_SEED,
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
            eval_metrics, _, _ = _evaluate(model, test_sequences, test_omegas, test_labels, num_windows=num_windows)
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
    metrics, d, attention_weights = _evaluate(model, test_sequences, test_omegas, test_labels, num_windows=num_windows)
    metrics.update(_error_decomposition(model, test_sequences, test_omegas, num_windows=num_windows))
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


def _save_attention_heatmap(path: Path, weights: torch.Tensor, title: str) -> None:
    weights_np = weights.detach().cpu().numpy()
    plt.figure(figsize=(8.0, 5.8))
    plt.imshow(weights_np, aspect="auto", interpolation="nearest", cmap="magma")
    plt.xlabel("window index")
    plt.ylabel("test trajectory")
    plt.title(title)
    plt.colorbar(label="attention weight")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_attention_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log_y: bool = False) -> None:
    plt.figure(figsize=(7.2, 4.6))
    attention_rows = [row for row in rows if row["encoder_type"] == "attention"]
    attention_rows = sorted(attention_rows, key=lambda row: row["num_dynamics_windows"])
    plt.plot(
        [row["num_dynamics_windows"] for row in attention_rows],
        [row[metric] for row in attention_rows],
        marker="o",
        label="attention",
    )
    mean_rows = sorted([row for row in rows if row["encoder_type"] == "mean_pool"], key=lambda row: row["num_dynamics_windows"])
    if mean_rows:
        plt.scatter(
            [row["num_dynamics_windows"] for row in mean_rows],
            [row[metric] for row in mean_rows],
            marker="s",
            s=70,
            label="mean pool",
        )
    plt.xscale("log", base=2)
    if log_y:
        plt.yscale("log")
    plt.xticks([32, 64, 128], ["32", "64", "128"])
    plt.xlabel("number of trajectory windows")
    plt.ylabel(ylabel)
    plt.legend()
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
    if isinstance(value, list):
        return json.dumps(value)
    return str(value)


def _interpret(rows: list[dict[str, Any]]) -> list[str]:
    best = min(rows, key=lambda row: row["rollout_mse"])
    attention_rows = sorted([row for row in rows if row["encoder_type"] == "attention"], key=lambda row: row["num_dynamics_windows"])
    lines = [
        f"Best rollout is {best['run_name']} with rollout MSE {best['rollout_mse']:.4e}.",
        f"Best omega R2 is {max(rows, key=lambda row: row['omega_r2_from_d'])['run_name']}.",
    ]
    if len(attention_rows) >= 2:
        lines.append(
            f"Attention rollout changes from {attention_rows[0]['rollout_mse']:.4e} at m={attention_rows[0]['num_dynamics_windows']} to {attention_rows[-1]['rollout_mse']:.4e} at m={attention_rows[-1]['num_dynamics_windows']}."
        )
        first = attention_rows[0]
        last = attention_rows[-1]
        ratio = first["rollout_mse"] / last["rollout_mse"] if last["rollout_mse"] > 0 else float("inf")
        lines.append(f"The m={last['num_dynamics_windows']} run improves over m={first['num_dynamics_windows']} by {ratio:.2f}x in rollout MSE.")
    return lines


def _write_html(rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "run_name",
        "encoder_type",
        "num_dynamics_windows",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "latent_transition_mse",
        "omega_r2_from_d",
        "omega_correlation_from_d",
        "omega_rmse_from_d",
        "abomega_multioutput_r2_from_d",
        "attention_entropy_mean",
        "n_eff_mean",
        "attention_max_weight_mean",
        "decomp_state_autoencode_position_mse",
        "decomp_latent_transition_mse",
        "decomp_oracle_observed_sho_rollout_mse",
        "decomp_rollout_minus_autoencode_mse",
        "decomp_autoencode_fraction_of_rollout",
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
  <title>Attention Large Window Sweep</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Attention Large Window Sweep</h1>
  <p>Separated dynamics/state Koopman setup: position-only, dense K_i=G(d_i), learned state latent, lambda_latent={LAMBDA_LATENT:g}, lambda_dyn=0, no labels, no analytic decoder, no spectral loss.</p>
  <p>Error decomposition is diagnostic only. The oracle observed-state recurrence uses true omega only after training to estimate the numerical recurrence floor; it is not used for optimization.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>Rollout MSE</h2><img src="rollout_mse_vs_window_count.png">
  <h2>Omega R2</h2><img src="omega_r2_vs_window_count.png">
  <h2>Effective Windows</h2><img src="neff_vs_window_count.png">
  <h2>Attention Entropy</h2><img src="attention_entropy_vs_window_count.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        metrics, _ = _train_variant(variant, train_sequences, test_sequences, test_omegas, test_labels)
        rows.append(metrics)
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)
    interpretation = _interpret(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "interpretation": interpretation})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _plot_attention_metric(rows, "rollout_mse", "rollout_mse_vs_window_count.png", "rollout MSE", log_y=True)
    _plot_attention_metric(rows, "omega_r2_from_d", "omega_r2_vs_window_count.png", "omega R2 from d")
    _plot_attention_metric(rows, "n_eff_mean", "neff_vs_window_count.png", "effective attended windows")
    _plot_attention_metric(rows, "attention_entropy_mean", "attention_entropy_vs_window_count.png", "attention entropy")
    _write_html(rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
