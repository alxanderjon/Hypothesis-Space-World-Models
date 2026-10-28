"""Latent representation analysis for Stage 1B composition branches."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import shutil
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

OUTPUT_DIR = Path("outputs/sho/stage1B/latent_representation_analysis")
PRIOR_200K_DIR = Path("outputs/sho/stage1B/composition_long_training_200k")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIMS = [1, 2, 4, 8, 16]
ENCODER_HIDDEN_SIZE = 16
OPERATOR_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200_000
PATIENCE = 25_000
EVAL_EVERY = 10_000

ElementwiseLinear = nn.Linear


class BranchALatentOperatorModel(nn.Module):
    """Latent-operator-composition architecture with configurable latent size."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, ENCODER_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(ENCODER_HIDDEN_SIZE, latent_dim),
        )
        self.operator = nn.Sequential(
            ElementwiseLinear(latent_dim + 2, OPERATOR_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(OPERATOR_HIDDEN_SIZE, OPERATOR_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(OPERATOR_HIDDEN_SIZE, 1),
        )

    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), z), dim=-1)
        return self.operator(inputs).squeeze(-1)

    def rollout(self, x_prev: torch.Tensor, x_curr: torch.Tensor, z: torch.Tensor, steps: int) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, z)
            predictions.append(next_x)
            prev = curr
            curr = next_x
        return torch.stack(predictions, dim=-1)


class BranchBDenseKoopmanModel(nn.Module):
    """Dense trajectory-conditioned 2x2 Koopman architecture."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, ENCODER_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(ENCODER_HIDDEN_SIZE, latent_dim),
        )
        self.generator = nn.Sequential(
            ElementwiseLinear(latent_dim, GENERATOR_HIDDEN_SIZE),
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


def _make_position_data(num_trajectories: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    times = torch.arange(TOTAL_SEQUENCE_STEPS, dtype=initial_conditions.dtype) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    a = initial_conditions[:, 0]
    b = initial_conditions[:, 1] / omegas
    amplitude = torch.sqrt(a.square() + b.square())
    phase = torch.atan2(b, a)
    labels = torch.stack((a, b, omegas), dim=-1)
    interpreted_labels = torch.stack((a, b, amplitude, phase, omegas), dim=-1)
    return positions, omegas, labels, interpreted_labels


def _sample_chunks(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(POSITION_WINDOW_SIZE + ROLLOUT_STEPS, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_chunks(sequences, indices, starts)


def _branch_a_losses(
    model: BranchALatentOperatorModel,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z = model.encode(windows)
    one_step_z = z.unsqueeze(1).expand(-1, chunks.shape[1] - 2, -1)
    one_step = model.step(chunks[:, :-2], chunks[:, 1:-1], one_step_z)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])
    two_step_z = z.unsqueeze(1).expand(-1, chunks.shape[1] - 3, -1)
    first = model.step(chunks[:, :-3], chunks[:, 1:-2], two_step_z)
    second = model.step(chunks[:, 1:-2], first, two_step_z)
    two_step_loss = mse_loss(second, chunks[:, 3:])
    rollout = model.rollout(windows[:, -2], windows[:, -1], z, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])
    return one_step_loss + two_step_loss + rollout_loss, {
        "one_step_mse": one_step_loss,
        "two_step_mse": two_step_loss,
        "rollout_mse": rollout_loss,
    }, z, rollout


def _branch_b_losses(
    model: BranchBDenseKoopmanModel,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
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
    }, z, rollout


def _linear_regression_stats(features: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = target.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    prediction = model.predict(x)
    r2 = float(model.score(x, y))
    corr = float(np.corrcoef(np.asarray(prediction).reshape(-1), y.reshape(-1))[0, 1])
    return r2, corr


def _multioutput_r2(features: torch.Tensor, targets: torch.Tensor) -> float:
    model = LinearRegression()
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    model.fit(x, y)
    return float(model.score(x, y))


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _evaluate(
    branch_name: str,
    model: nn.Module,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        if branch_name == "branch_a_latent_operator":
            _, losses, z, _ = _branch_a_losses(model, chunks)  # type: ignore[arg-type]
        else:
            _, losses, z, _ = _branch_b_losses(model, chunks)  # type: ignore[arg-type]
    omega_r2, omega_corr = _linear_regression_stats(z, omegas)
    return {
        "one_step_mse": losses["one_step_mse"].item(),
        "two_step_mse": losses["two_step_mse"].item(),
        "rollout_mse": losses["rollout_mse"].item(),
        "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
        "omega_r2_from_latent": omega_r2,
        "omega_correlation_from_latent": omega_corr,
        "abomega_multioutput_r2_from_latent": _multioutput_r2(z, labels),
    }, z


def _latent_xy(z: torch.Tensor) -> tuple[np.ndarray, np.ndarray, str, str]:
    z_np = z.detach().cpu().numpy()
    if z_np.shape[1] == 1:
        return z_np[:, 0], np.zeros(z_np.shape[0]), "z[0]", "0"
    if z_np.shape[1] == 2:
        return z_np[:, 0], z_np[:, 1], "z[0]", "z[1]"
    pca = PCA(n_components=2)
    xy = pca.fit_transform(z_np)
    return xy[:, 0], xy[:, 1], "PC1", "PC2"


def _save_latent_scatter(path: Path, z: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
    x, y, xlabel, ylabel = _latent_xy(z)
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(x, y, c=omegas.cpu().numpy(), cmap="viridis", s=30)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_rollout_examples(path: Path, branch_name: str, model: nn.Module, sequences: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        if branch_name == "branch_a_latent_operator":
            z = model.encode(windows)  # type: ignore[attr-defined]
            rollout = model.rollout(windows[:, -2], windows[:, -1], z, ROLLOUT_STEPS)  # type: ignore[attr-defined]
        else:
            z = model.encode(windows)  # type: ignore[attr-defined]
            params = model.operator_params(z)  # type: ignore[attr-defined]
            rollout = model.rollout(windows[:, -2], windows[:, -1], params, ROLLOUT_STEPS)  # type: ignore[attr-defined]
        targets = chunks[:, POSITION_WINDOW_SIZE:]
    fig, axes = plt.subplots(4, 1, figsize=(8, 9.5))
    observed_x = torch.arange(POSITION_WINDOW_SIZE).cpu()
    rollout_x = torch.arange(POSITION_WINDOW_SIZE, POSITION_WINDOW_SIZE + ROLLOUT_STEPS).cpu()
    for index, ax in enumerate(axes):
        ax.plot(observed_x, windows[index].cpu(), label="observed window")
        ax.plot(rollout_x, targets[index].cpu(), label="target rollout")
        ax.plot(rollout_x, rollout[index].cpu(), "--", label="predicted rollout")
        ax.set_title(f"{branch_name} latent_dim={model.latent_dim} rollout example {index}")  # type: ignore[attr-defined]
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _prior_branch_dir(branch_name: str) -> Path:
    return PRIOR_200K_DIR / branch_name


def _copy_prior_dim4_if_available(run_dir: Path, branch_name: str, latent_dim: int) -> bool:
    if latent_dim != 4:
        return False
    prior_dir = _prior_branch_dir(branch_name)
    if not (prior_dir / "metrics.json").exists() or not (prior_dir / "learned_parameters.pt").exists():
        return False
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.json", "training_history.json", "history.csv", "learned_parameters.pt", "setup.json", "rollout_examples.png"]:
        source = prior_dir / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)
    if (prior_dir / "latent_final.png").exists():
        shutil.copy2(prior_dir / "latent_final.png", run_dir / "latent_final.png")
    return True


def _load_prior_or_train(
    branch_name: str,
    latent_dim: int,
    model: nn.Module,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = f"{branch_name}_latent_dim_{latent_dim}"
    run_dir = OUTPUT_DIR / run_name
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"
    if metrics_path.exists() and history_path.exists():
        model_path = run_dir / "learned_parameters.pt"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(history_path.read_text(encoding="utf-8"))["history"],
        )
    if _copy_prior_dim4_if_available(run_dir, branch_name, latent_dim):
        model.load_state_dict(torch.load(run_dir / "learned_parameters.pt", map_location="cpu"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))["history"]
        metrics = {**metrics, "run_name": run_name, "branch_name": branch_name, "latent_dim": latent_dim, "reused_from": str(_prior_branch_dir(branch_name))}
        _write_json(metrics_path, metrics)
        _write_json(history_path, {"history": history})
        return metrics, history

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latent_snapshots").mkdir(exist_ok=True)
    set_seed(SEED + latent_dim)
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
            "run_name": run_name,
            "branch_name": branch_name,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "latent_dim": latent_dim,
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
            if branch_name == "branch_a_latent_operator":
                loss, _, _, _ = _branch_a_losses(model, chunks)  # type: ignore[arg-type]
            else:
                loss, _, _, _ = _branch_b_losses(model, chunks)  # type: ignore[arg-type]
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
            eval_metrics, z = _evaluate(branch_name, model, test_sequences, test_omegas, test_labels)
            rollout_mse = eval_metrics["rollout_mse"]
            if rollout_mse < best_rollout:
                best_rollout = rollout_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            history.append({"epoch": epoch, "train_loss": train_loss_sum / train_count, **eval_metrics})
            _write_json(history_path, {"history": history})
            _write_csv(run_dir / "history.csv", history)
            _save_latent_scatter(
                run_dir / "latent_snapshots" / f"latent_epoch_{epoch:06d}.png",
                z,
                test_omegas,
                f"{run_name} latent at epoch {epoch}",
            )
            print(f"{run_name} epoch {epoch:06d} rollout_mse={rollout_mse:.6e} best={best_rollout:.6e}", flush=True)
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, z = _evaluate(branch_name, model, test_sequences, test_omegas, test_labels)
    metrics.update(
        {
            "run_name": run_name,
            "branch_name": branch_name,
            "latent_dim": latent_dim,
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
    _save_latent_scatter(run_dir / "latent_final.png", z, test_omegas, f"{run_name} final latent")
    _save_rollout_examples(run_dir / "rollout_examples.png", branch_name, model, test_sequences)
    return metrics, history


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _single_feature_r2(feature: np.ndarray, target: np.ndarray) -> float:
    model = LinearRegression()
    x = feature.reshape(-1, 1)
    model.fit(x, target)
    return float(model.score(x, target))


def _regression_r2(features: np.ndarray, targets: np.ndarray) -> float:
    model = LinearRegression()
    model.fit(features, targets)
    return float(model.score(features, targets))


def _analyze_latent(
    branch_name: str,
    latent_dim: int,
    model: nn.Module,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    interpreted_labels: torch.Tensor,
) -> dict[str, Any]:
    run_name = f"{branch_name}_latent_dim_{latent_dim}"
    run_dir = OUTPUT_DIR / run_name
    analysis_dir = run_dir / "latent_interpretation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(torch.load(run_dir / "learned_parameters.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        windows = _fixed_eval_chunks(test_sequences)[:, :POSITION_WINDOW_SIZE]
        z = model.encode(windows)  # type: ignore[attr-defined]
    z_np = z.detach().cpu().numpy()
    labels_np = interpreted_labels.detach().cpu().numpy()
    names = ["a", "b", "A", "phi", "omega"]
    target_by_name = {name: labels_np[:, index] for index, name in enumerate(names)}
    per_dim_rows: list[dict[str, Any]] = []
    for dim in range(latent_dim):
        row: dict[str, Any] = {"branch_name": branch_name, "latent_dim": latent_dim, "z_dim": dim}
        for name in names:
            row[f"corr_z_{name}"] = _corr(z_np[:, dim], target_by_name[name])
            row[f"r2_{name}_from_z"] = _single_feature_r2(z_np[:, dim], target_by_name[name])
        per_dim_rows.append(row)
    _write_csv(analysis_dir / "per_dimension_stats.csv", per_dim_rows)

    full_stats = {
        "branch_name": branch_name,
        "latent_dim": latent_dim,
        "omega_r2_from_z": _regression_r2(z_np, target_by_name["omega"]),
        "ab_r2_from_z": _regression_r2(z_np, labels_np[:, 0:2]),
        "A_phi_omega_r2_from_z": _regression_r2(z_np, labels_np[:, [2, 3, 4]]),
    }

    pca_components = min(latent_dim, z_np.shape[0])
    pca = PCA(n_components=pca_components)
    pc_scores = pca.fit_transform(z_np)
    pca_rows: list[dict[str, Any]] = []
    for count in range(1, pca_components + 1):
        pc_subset = pc_scores[:, :count]
        pca_rows.append(
            {
                "branch_name": branch_name,
                "latent_dim": latent_dim,
                "num_pcs": count,
                "cumulative_explained_variance": float(np.sum(pca.explained_variance_ratio_[:count])),
                "omega_r2_from_pcs": _regression_r2(pc_subset, target_by_name["omega"]),
                "ab_r2_from_pcs": _regression_r2(pc_subset, labels_np[:, 0:2]),
                "A_phi_omega_r2_from_pcs": _regression_r2(pc_subset, labels_np[:, [2, 3, 4]]),
            },
        )
    _write_csv(analysis_dir / "pca_recoverability.csv", pca_rows)

    used_variance_directions = int(np.sum(pca.explained_variance_ratio_ > 0.01))
    directions_for_95pct = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    omega_abs_corrs = [abs(float(row["corr_z_omega"])) for row in per_dim_rows]
    amplitude_abs_corrs = [abs(float(row["corr_z_A"])) for row in per_dim_rows]
    phase_abs_corrs = [abs(float(row["corr_z_phi"])) for row in per_dim_rows]
    interpretation = {
        **full_stats,
        "pca_explained_variance": [float(value) for value in pca.explained_variance_ratio_],
        "used_variance_directions_gt_1pct": used_variance_directions,
        "directions_for_95pct_variance": directions_for_95pct,
        "strongest_omega_dim": int(np.nanargmax(omega_abs_corrs)),
        "strongest_omega_abs_corr": float(np.nanmax(omega_abs_corrs)),
        "strongest_A_dim": int(np.nanargmax(amplitude_abs_corrs)),
        "strongest_A_abs_corr": float(np.nanmax(amplitude_abs_corrs)),
        "strongest_phi_dim": int(np.nanargmax(phase_abs_corrs)),
        "strongest_phi_abs_corr": float(np.nanmax(phase_abs_corrs)),
    }
    _write_json(analysis_dir / "interpretation_summary.json", interpretation)
    _save_latent_scatter(analysis_dir / "latent_colored_by_omega.png", z, test_omegas, f"{run_name} latent colored by omega")
    return interpretation


def _save_sweep_plot(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(7, 4.5))
    for branch_name in ["branch_a_latent_operator", "branch_b_dense_koopman"]:
        branch_rows = sorted([row for row in rows if row["branch_name"] == branch_name], key=lambda row: row["latent_dim"])
        plt.plot(
            [row["latent_dim"] for row in branch_rows],
            [row[metric] for row in branch_rows],
            marker="o",
            label=branch_name,
        )
    if log_y:
        plt.yscale("log")
    plt.xlabel("latent dimension")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xticks(LATENT_DIMS)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_html(summary_rows: list[dict[str, Any]], interpretation_rows: list[dict[str, Any]]) -> None:
    summary_columns = [
        "branch_name",
        "latent_dim",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "latent_variance",
        "parameter_count",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    interpretation_columns = [
        "branch_name",
        "latent_dim",
        "omega_r2_from_z",
        "ab_r2_from_z",
        "A_phi_omega_r2_from_z",
        "used_variance_directions_gt_1pct",
        "directions_for_95pct_variance",
        "strongest_omega_dim",
        "strongest_omega_abs_corr",
        "strongest_A_dim",
        "strongest_A_abs_corr",
        "strongest_phi_dim",
        "strongest_phi_abs_corr",
    ]

    def table(columns: list[str], rows: list[dict[str, Any]]) -> str:
        header = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(_format_number(row.get(column, '')))}</td>" for column in columns) + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Latent Representation Analysis</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 940px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Stage 1B Latent Representation Analysis</h1>
  <p>Position-only data. Architectures and losses are unchanged from the 200k composition comparison: L_1step + L_2step + L_rollout, with no omega labels, coefficient labels, spectral loss, contrastive loss, or added latent objectives.</p>
  <h2>Latent Dimension Sweep</h2>
  {table(summary_columns, summary_rows)}
  <h2>Best-Dimension Latent Interpretation</h2>
  {table(interpretation_columns, interpretation_rows)}
  <h2>Sweep Plots</h2>
  <h3>Rollout MSE vs Latent Dimension</h3><img src="rollout_mse_vs_latent_dim.png">
  <h3>Omega R2 vs Latent Dimension</h3><img src="omega_r2_vs_latent_dim.png">
  <h3>(a,b,omega) R2 vs Latent Dimension</h3><img src="abomega_r2_vs_latent_dim.png">
  <h3>Latent Variance vs Latent Dimension</h3><img src="latent_variance_vs_latent_dim.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels, interpreted_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    summary_rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}

    for branch_name, model_class in [
        ("branch_a_latent_operator", BranchALatentOperatorModel),
        ("branch_b_dense_koopman", BranchBDenseKoopmanModel),
    ]:
        for latent_dim in LATENT_DIMS:
            model = model_class(latent_dim)
            metrics, history = _load_prior_or_train(
                branch_name,
                latent_dim,
                model,
                train_sequences,
                test_sequences,
                test_omegas,
                test_labels,
            )
            metrics = {**metrics, "branch_name": branch_name, "latent_dim": latent_dim}
            summary_rows.append(metrics)
            histories[f"{branch_name}_latent_dim_{latent_dim}"] = history
            _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": summary_rows})
            _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)

    interpretation_rows: list[dict[str, Any]] = []
    for branch_name, model_class in [
        ("branch_a_latent_operator", BranchALatentOperatorModel),
        ("branch_b_dense_koopman", BranchBDenseKoopmanModel),
    ]:
        branch_rows = [row for row in summary_rows if row["branch_name"] == branch_name]
        best = min(branch_rows, key=lambda row: row["rollout_mse"])
        latent_dim = int(best["latent_dim"])
        model = model_class(latent_dim)
        interpretation_rows.append(
            _analyze_latent(branch_name, latent_dim, model, test_sequences, test_omegas, interpreted_labels),
        )

    _write_json(OUTPUT_DIR / "summary.json", {"runs": summary_rows, "interpretation": interpretation_rows})
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "latent_interpretation_summary.csv", interpretation_rows)
    _save_sweep_plot(summary_rows, "rollout_mse", "rollout_mse_vs_latent_dim.png", "rollout MSE", log_y=True)
    _save_sweep_plot(summary_rows, "omega_r2_from_latent", "omega_r2_vs_latent_dim.png", "omega R2 from latent")
    _save_sweep_plot(
        summary_rows,
        "abomega_multioutput_r2_from_latent",
        "abomega_r2_vs_latent_dim.png",
        "(a,b,omega) R2 from latent",
    )
    _save_sweep_plot(summary_rows, "latent_variance", "latent_variance_vs_latent_dim.png", "latent variance")
    _write_html(summary_rows, interpretation_rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": summary_rows, "interpretation": interpretation_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
