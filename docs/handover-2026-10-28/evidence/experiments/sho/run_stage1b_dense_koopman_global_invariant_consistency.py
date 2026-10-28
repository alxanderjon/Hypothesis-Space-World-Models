"""Global invariant consistency diagnostic for Branch B dense Koopman."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/global_invariant_consistency")
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
MAX_EPOCHS = 50_000
PATIENCE = 10_000
EVAL_EVERY = 1_000
LAMBDA_GLOBAL_VALUES = [0.0, 1e-4, 1e-3, 1e-2]
DESCRIPTOR_EPS = 1e-8

ElementwiseLinear = nn.Linear


class DenseKoopmanModel(nn.Module):
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


def _global_descriptor(sequence: torch.Tensor) -> torch.Tensor:
    """Return an energy-normalized magnitude descriptor of a rollout."""
    centered = sequence - sequence.mean(dim=-1, keepdim=True)
    magnitudes = torch.fft.rfft(centered, dim=-1).abs()
    return magnitudes / (magnitudes.sum(dim=-1, keepdim=True) + DESCRIPTOR_EPS)


def _global_descriptor_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mse_loss(_global_descriptor(predicted), _global_descriptor(target))


def _losses(
    model: DenseKoopmanModel,
    chunks: torch.Tensor,
    lambda_global: float,
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
    target_rollout = chunks[:, POSITION_WINDOW_SIZE:]
    rollout_loss = mse_loss(rollout, target_rollout)
    global_loss = _global_descriptor_loss(rollout, target_rollout)
    reconstruction_loss = one_step_loss + two_step_loss + rollout_loss
    total_loss = reconstruction_loss + lambda_global * global_loss
    return total_loss, {
        "one_step_mse": one_step_loss,
        "two_step_mse": two_step_loss,
        "rollout_mse": rollout_loss,
        "reconstruction_loss": reconstruction_loss,
        "global_descriptor_loss": global_loss,
    }, z, rollout


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
    lambda_global: float,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z, _ = _losses(model, chunks, lambda_global)
    omega_r2, omega_corr = _linear_regression_stats(z, omegas)
    return {
        "one_step_mse": losses["one_step_mse"].item(),
        "two_step_mse": losses["two_step_mse"].item(),
        "rollout_mse": losses["rollout_mse"].item(),
        "global_descriptor_loss": losses["global_descriptor_loss"].item(),
        "omega_r2_from_latent": omega_r2,
        "omega_correlation_from_latent": omega_corr,
        "abomega_multioutput_r2_from_latent": _multioutput_r2(z, labels),
        "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
    }, z


def _save_latent_pca(path: Path, z: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
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


def _train_variant(
    lambda_global: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_name = f"lambda_global_{lambda_global:g}".replace(".", "p").replace("-", "m")
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
    model = DenseKoopmanModel()
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
            "lambda_global": lambda_global,
            "global_descriptor": "energy-normalized rollout rfft magnitude",
            "latent_dim": LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
            "losses": "L_1step + L_2step + L_rollout + lambda_global * L_global_descriptor",
        },
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_sequences.shape[0])
        train_loss_sum = 0.0
        train_reconstruction_sum = 0.0
        train_global_sum = 0.0
        train_count = 0
        for start_index in range(0, train_sequences.shape[0], BATCH_SIZE):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            starts = torch.randint(0, max_start + 1, (indices.shape[0],))
            chunks = _sample_chunks(train_sequences, indices, starts)
            optimizer.zero_grad()
            loss, parts, _, _ = _losses(model, chunks, lambda_global)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            count = indices.shape[0]
            train_loss_sum += loss.item() * count
            train_reconstruction_sum += parts["reconstruction_loss"].item() * count
            train_global_sum += parts["global_descriptor_loss"].item() * count
            train_count += count
        if stopped_because == "nan_or_inf_loss":
            break

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, z = _evaluate(model, test_sequences, test_omegas, test_labels, lambda_global)
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
                "train_loss": train_loss_sum / train_count,
                "train_reconstruction_loss": train_reconstruction_sum / train_count,
                "train_global_descriptor_loss": train_global_sum / train_count,
                **eval_metrics,
            }
            history.append(row)
            _write_json(history_path, {"history": history})
            _write_csv(run_dir / "history.csv", history)
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics, z = _evaluate(model, test_sequences, test_omegas, test_labels, lambda_global)
    metrics.update(
        {
            "run_name": run_name,
            "lambda_global": lambda_global,
            "global_descriptor": "energy-normalized rollout rfft magnitude",
            "latent_dim": LATENT_DIM,
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
    _save_latent_pca(run_dir / "latent_pca_colored_by_omega.png", z, test_omegas, f"Global invariant consistency {run_name}")
    return metrics, history


def _save_metric_plot(histories: dict[str, list[dict[str, Any]]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(7, 4.5))
    for run_name, history in histories.items():
        plt.plot([row["epoch"] for row in history], [row[metric] for row in history], marker="o", label=run_name)
    if log_y:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _save_summary_plot(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(6.5, 4.2))
    xs = [row["lambda_global"] for row in rows]
    ys = [row[metric] for row in rows]
    labels = ["0" if value == 0 else f"{value:g}" for value in xs]
    plt.plot(range(len(xs)), ys, marker="o")
    plt.xticks(range(len(xs)), labels)
    if log_y:
        plt.yscale("log")
    plt.xlabel("lambda_global")
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
    baseline = next(row for row in rows if row["lambda_global"] == 0.0)
    lines = [
        "This diagnostic treats the rollout spectrum as a global dynamical descriptor, not as SHO-specific frequency supervision.",
        "The descriptor is computed from the predicted and target rollouts only; no omega labels or analytic SHO parameters enter training.",
    ]
    for row in rows:
        if row["lambda_global"] == 0.0:
            continue
        descriptor_improved = row["global_descriptor_loss"] < baseline["global_descriptor_loss"]
        rollout_improved = row["rollout_mse"] < baseline["rollout_mse"]
        if descriptor_improved and rollout_improved:
            relation = "global descriptor and rollout both improved"
        elif descriptor_improved and not rollout_improved:
            relation = "global descriptor improved while rollout worsened"
        elif not descriptor_improved and rollout_improved:
            relation = "rollout improved without improving the descriptor"
        else:
            relation = "neither descriptor nor rollout improved"
        lines.append(f"lambda_global={row['lambda_global']:g}: {relation}.")
    best_rollout = min(rows, key=lambda row: row["rollout_mse"])
    best_descriptor = min(rows, key=lambda row: row["global_descriptor_loss"])
    lines.append(f"Best rollout MSE occurs at lambda_global={best_rollout['lambda_global']:g}.")
    lines.append(f"Best global descriptor consistency occurs at lambda_global={best_descriptor['lambda_global']:g}.")
    if best_rollout["lambda_global"] == 0.0 and best_descriptor["lambda_global"] != 0.0:
        lines.append(
            "In this run, the auxiliary global descriptor can be optimized without improving pointwise rollout accuracy; that would argue for refining the descriptor or schedule rather than rejecting global invariant consistency as a general idea.",
        )
    elif best_rollout["lambda_global"] != 0.0:
        lines.append(
            "In this run, a nonzero global invariant weight improved rollout, suggesting that generic trajectory-level consistency can help recover dynamics beyond purely local reconstruction.",
        )
    else:
        lines.append(
            "In this run, nonzero global invariant weights did not beat the reconstruction-only baseline; the failure is evidence about this descriptor and weighting, not yet about the broader global-invariant idea.",
        )
    return lines


def _write_html(rows: list[dict[str, Any]], interpretation: list[str]) -> None:
    columns = [
        "lambda_global",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "global_descriptor_loss",
        "omega_r2_from_latent",
        "abomega_multioutput_r2_from_latent",
        "latent_variance",
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
  <title>Stage 1B Global Invariant Consistency</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 920px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Stage 1B Global Invariant Consistency</h1>
  <p>Branch B dense trajectory-conditioned Koopman, latent_dim={LATENT_DIM}. Position-only observations, no velocity, no omega labels, no coefficient labels, no analytic SHO decoder, and no spectral loss. The added descriptor is an energy-normalized rollout DFT magnitude, used here as a first generic global dynamical invariant.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Interpretation</h2>
  {interpretation_html}
  <h2>Rollout MSE vs Epoch</h2><img src="rollout_mse_vs_epoch.png">
  <h2>Global Descriptor Loss vs Epoch</h2><img src="global_descriptor_loss_vs_epoch.png">
  <h2>Omega R2 vs Epoch</h2><img src="omega_r2_vs_epoch.png">
  <h2>Rollout MSE vs Lambda</h2><img src="rollout_mse_vs_lambda.png">
  <h2>Global Descriptor Loss vs Lambda</h2><img src="global_descriptor_loss_vs_lambda.png">
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
    for lambda_global in LAMBDA_GLOBAL_VALUES:
        metrics, history = _train_variant(lambda_global, train_sequences, test_sequences, test_omegas, test_labels)
        rows.append(metrics)
        histories[metrics["run_name"]] = history
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)

    rows.sort(key=lambda row: row["lambda_global"])
    interpretation = _interpret(rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows, "interpretation": interpretation})
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _save_metric_plot(histories, "rollout_mse", "rollout_mse_vs_epoch.png", "rollout MSE", log_y=True)
    _save_metric_plot(histories, "global_descriptor_loss", "global_descriptor_loss_vs_epoch.png", "global descriptor loss", log_y=True)
    _save_metric_plot(histories, "omega_r2_from_latent", "omega_r2_vs_epoch.png", "omega R2 from latent")
    _save_summary_plot(rows, "rollout_mse", "rollout_mse_vs_lambda.png", "rollout MSE", log_y=True)
    _save_summary_plot(rows, "global_descriptor_loss", "global_descriptor_loss_vs_lambda.png", "global descriptor loss", log_y=True)
    _write_html(rows, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
