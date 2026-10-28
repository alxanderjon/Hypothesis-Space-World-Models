"""Stage 1B position-only latent operator composition diagnostic.

This diagnostic separates trajectory-level dynamics inference from local state
evolution. It infers one latent z per trajectory window, keeps z fixed, and
reuses a single operator F_theta(x_{t-1}, x_t, z) -> x_{t+1}.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from torch import nn

from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    OMEGA_RANGE,
    SEED,
    _make_base_conditions,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/latent_operator_composition")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 4
ENCODER_HIDDEN_SIZE = 16
OPERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10_000
DIAGNOSTIC_EPOCH_CUTOFF = 1_000
PATIENCE = 2_000
EVAL_EVERY = 50
LOG_EVERY = 250


class ElementwiseLinear(nn.Module):
    """Linear layer implemented without torch matmul for this small diagnostic."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        bound = input_dim**-0.5
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(output_dim).uniform_(-bound, bound))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.unsqueeze(-2) * self.weight).sum(dim=-1) + self.bias


class LatentOperatorCompositionModel(nn.Module):
    """Encode a position window into z and reuse one local evolution operator."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        encoder_hidden_size: int = ENCODER_HIDDEN_SIZE,
        operator_hidden_size: int = OPERATOR_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, encoder_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(encoder_hidden_size, latent_dim),
        )
        self.operator = nn.Sequential(
            ElementwiseLinear(latent_dim + 2, operator_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(operator_hidden_size, operator_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(operator_hidden_size, 1),
        )

    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), z), dim=-1)
        return self.operator(inputs).squeeze(-1)

    def rollout(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        z: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, z)
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


def _make_position_sequences(
    num_trajectories: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    times = (
        torch.arange(TOTAL_SEQUENCE_STEPS, dtype=initial_conditions.dtype)
        * float(DELTA_T)
    )
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions, omegas


def _sample_chunks(
    sequences: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.arange(POSITION_WINDOW_SIZE + ROLLOUT_STEPS, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_chunks(sequences, indices, starts)


def _losses(
    model: LatentOperatorCompositionModel,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z = model.encode(windows)

    one_step_z = z.repeat_interleave(chunks.shape[1] - 2, dim=0)
    one_step = model.step(
        chunks[:, :-2].reshape(-1),
        chunks[:, 1:-1].reshape(-1),
        one_step_z,
    ).reshape(chunks.shape[0], chunks.shape[1] - 2)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])

    two_step_z = z.repeat_interleave(chunks.shape[1] - 3, dim=0)
    first = model.step(
        chunks[:, :-3].reshape(-1),
        chunks[:, 1:-2].reshape(-1),
        two_step_z,
    )
    second = model.step(
        chunks[:, 1:-2].reshape(-1),
        first,
        two_step_z,
    ).reshape(chunks.shape[0], chunks.shape[1] - 3)
    two_step_loss = mse_loss(second, chunks[:, 3:])

    rollout = model.rollout(
        windows[:, -2],
        windows[:, -1],
        z,
        ROLLOUT_STEPS,
    )
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])

    loss = one_step_loss + two_step_loss + rollout_loss
    return (
        loss,
        {
            "one_step_loss": one_step_loss,
            "two_step_loss": two_step_loss,
            "rollout_loss": rollout_loss,
        },
        z,
        rollout,
    )


def _max_abs_latent_omega_correlation(z: torch.Tensor, omegas: torch.Tensor) -> float:
    centered_z = z - z.mean(dim=0, keepdim=True)
    centered_omega = omegas - omegas.mean()
    numerator = (centered_z * centered_omega.unsqueeze(-1)).mean(dim=0)
    denominator = centered_z.std(dim=0, unbiased=False) * centered_omega.std(unbiased=False)
    correlations = numerator / (denominator + 1e-8)
    return correlations.abs().max().item()


def _evaluate(
    model: LatentOperatorCompositionModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z, _ = _losses(model, chunks)
        latent_variance = z.var(dim=0, unbiased=False).mean()
        latent_omega_correlation = _max_abs_latent_omega_correlation(z, omegas)
    return {
        "one_step_mse": losses["one_step_loss"].item(),
        "two_step_mse": losses["two_step_loss"].item(),
        "rollout_mse": losses["rollout_loss"].item(),
        "latent_variance": latent_variance.item(),
        "max_abs_latent_omega_correlation": latent_omega_correlation,
    }


def _save_training_loss(path: Path, history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    plt.plot([row["epoch"] for row in history], [row["test_rollout_mse"] for row in history], label="test rollout")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Stage 1B Latent Operator Composition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_rollout_examples(
    path: Path,
    model: LatentOperatorCompositionModel,
    sequences: torch.Tensor,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        z = model.encode(windows)
        rollout = model.rollout(windows[:, -2], windows[:, -1], z, ROLLOUT_STEPS)
        targets = chunks[:, POSITION_WINDOW_SIZE:]

    count = min(num_examples, sequences.shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count))
    if count == 1:
        axes = [axes]
    observed_x = torch.arange(POSITION_WINDOW_SIZE).cpu()
    rollout_x = torch.arange(POSITION_WINDOW_SIZE, POSITION_WINDOW_SIZE + ROLLOUT_STEPS).cpu()
    for index in range(count):
        axes[index].plot(observed_x, windows[index].cpu(), label="observed window")
        axes[index].plot(rollout_x, targets[index].cpu(), label="target rollout")
        axes[index].plot(rollout_x, rollout[index].cpu(), "--", label="predicted rollout")
        axes[index].set_title(f"rollout example {index}")
        axes[index].set_xlabel("step")
        axes[index].set_ylabel("x")
        axes[index].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_latent_plot(
    path: Path,
    model: LatentOperatorCompositionModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        z = model.encode(chunks[:, :POSITION_WINDOW_SIZE])
    if z.shape[1] < 2:
        return
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(z[:, 0].cpu(), z[:, 1].cpu(), c=omegas.cpu(), cmap="viridis", s=28)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Latent space colored by held-out omega")
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_html(setup: dict[str, Any], metrics: dict[str, Any]) -> None:
    def rows(data: dict[str, Any]) -> str:
        return "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in data.items()
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Latent Operator Composition</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Latent Operator Composition</h1>
  <p>Position-only observation window, one trajectory-level latent, one reused local operator, no omega labels, no coefficient labels, no analytic SHO decoder.</p>
  <h2>Setup</h2>
  <table>{rows(setup)}</table>
  <h2>Metrics</h2>
  <table>{rows(metrics)}</table>
  <section><h2>Training Loss</h2><img src="training_loss.png" alt="Training Loss"></section>
  <section><h2>Rollout Examples</h2><img src="rollout_examples.png" alt="Rollout Examples"></section>
  <section><h2>Latent Visualization</h2><img src="latent_omega_scatter.png" alt="Latent Visualization"></section>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_sequences, _ = _make_position_sequences(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas = _make_position_sequences(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    model = LatentOperatorCompositionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    history: list[dict[str, Any]] = []
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"

    for epoch in range(1, min(MAX_EPOCHS, DIAGNOSTIC_EPOCH_CUTOFF) + 1):
        model.train()
        permutation = torch.randperm(train_sequences.shape[0])
        train_loss_sum = 0.0
        train_count = 0
        for start_index in range(0, train_sequences.shape[0], BATCH_SIZE):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            starts = torch.randint(0, max_start + 1, (indices.shape[0],))
            chunks = _sample_chunks(train_sequences, indices, starts)
            optimizer.zero_grad()
            loss, _, _, _ = _losses(model, chunks)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * indices.shape[0]
            train_count += indices.shape[0]
        if stopped_because == "nan_or_inf_loss":
            break

        train_loss = train_loss_sum / train_count
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            metrics = _evaluate(model, test_sequences, test_omegas)
            rollout_mse = metrics["rollout_mse"]
            if rollout_mse < best_rollout:
                best_rollout = rollout_mse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_rollout_mse": best_rollout if epoch % EVAL_EVERY else rollout_mse,
            },
        )

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                (
                    f"latent_operator_composition epoch {epoch:05d} "
                    f"train_loss={train_loss:.6e} best_rollout_mse={best_rollout:.6e}"
                ),
                flush=True,
            )
        if epochs_since_improvement >= PATIENCE:
            stopped_because = "patience_exhausted"
            break
    else:
        stopped_because = (
            "diagnostic_epoch_cutoff"
            if DIAGNOSTIC_EPOCH_CUTOFF < MAX_EPOCHS
            else "max_epochs_reached"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = _evaluate(model, test_sequences, test_omegas)
    metrics.update(
        {
            "run_name": "latent_operator_composition",
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "diagnostic_epoch_cutoff": DIAGNOSTIC_EPOCH_CUTOFF,
            "patience": PATIENCE,
        },
    )
    setup = {
        "stage": "stage1B",
        "experiment": "latent_operator_composition",
        "input": "position-only observation window",
        "state": "s_t = (x_{t-1}, x_t)",
        "latent": "z_i = Encoder(window_i), fixed during rollout",
        "operator": "F_theta(x_{t-1}, x_t, z_i) -> x_{t+1}",
        "loss": "L_1step + L_2step + L_rollout",
        "velocity_input": False,
        "omega_supervision": False,
        "coefficient_supervision": False,
        "analytic_decoder": False,
        "spectral_loss": False,
        "latent_contrastive_loss": False,
        "position_window_size": POSITION_WINDOW_SIZE,
        "rollout_steps": ROLLOUT_STEPS,
        "total_sequence_steps": TOTAL_SEQUENCE_STEPS,
        "latent_dim": LATENT_DIM,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "operator_hidden_size": OPERATOR_HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "diagnostic_epoch_cutoff": DIAGNOSTIC_EPOCH_CUTOFF,
        "patience": PATIENCE,
        "eval_every": EVAL_EVERY,
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "num_train": DIAGNOSTIC_NUM_TRAIN,
        "num_test": DIAGNOSTIC_NUM_TEST,
        "seed": SEED,
    }
    torch.save(model.state_dict(), OUTPUT_DIR / "learned_parameters.pt")
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", metrics)
    _write_json(OUTPUT_DIR / "training_history.json", {"history": history})
    _write_csv(OUTPUT_DIR / "metrics.csv", [metrics])
    _save_training_loss(OUTPUT_DIR / "training_loss.png", history)
    _save_rollout_examples(OUTPUT_DIR / "rollout_examples.png", model, test_sequences)
    _save_latent_plot(OUTPUT_DIR / "latent_omega_scatter.png", model, test_sequences, test_omegas)
    _write_html(setup, metrics)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
