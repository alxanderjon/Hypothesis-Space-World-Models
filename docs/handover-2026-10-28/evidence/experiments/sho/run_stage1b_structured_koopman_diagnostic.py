"""Stage 1B minimal structured trajectory-conditioned Koopman diagnostic."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/structured_koopman_diagnostic")
BRANCH_A_METRICS_PATH = Path("outputs/sho/stage1B/latent_operator_composition/metrics.json")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 8
ENCODER_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10_000
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


class StructuredKoopmanModel(nn.Module):
    """Infer a trajectory latent and structured second-order Koopman recurrence."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        encoder_hidden_size: int = ENCODER_HIDDEN_SIZE,
        generator_hidden_size: int = GENERATOR_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, encoder_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(encoder_hidden_size, latent_dim),
        )
        self.c_generator = nn.Sequential(
            ElementwiseLinear(latent_dim, generator_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(generator_hidden_size, 1),
        )

    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def coefficient(self, z: torch.Tensor) -> torch.Tensor:
        raw_c = self.c_generator(z).squeeze(-1)
        return 2.0 * torch.tanh(0.5 * raw_c)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c * x_curr - x_prev

    def rollout(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        c: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, c)
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


def _make_position_data(
    num_trajectories: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    times = (
        torch.arange(TOTAL_SEQUENCE_STEPS, dtype=initial_conditions.dtype)
        * float(DELTA_T)
    )
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    true_c = 2.0 * torch.cos(omegas * float(DELTA_T))
    return positions, omegas, true_c


def _sample_chunks(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(POSITION_WINDOW_SIZE + ROLLOUT_STEPS, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_chunks(sequences, indices, starts)


def _losses(
    model: StructuredKoopmanModel,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z = model.encode(windows)
    c = model.coefficient(z)

    c_one = c.repeat_interleave(chunks.shape[1] - 2)
    one_step = model.step(
        chunks[:, :-2].reshape(-1),
        chunks[:, 1:-1].reshape(-1),
        c_one,
    ).reshape(chunks.shape[0], chunks.shape[1] - 2)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])

    c_two = c.repeat_interleave(chunks.shape[1] - 3)
    first = model.step(
        chunks[:, :-3].reshape(-1),
        chunks[:, 1:-2].reshape(-1),
        c_two,
    )
    second = model.step(chunks[:, 1:-2].reshape(-1), first, c_two).reshape(
        chunks.shape[0],
        chunks.shape[1] - 3,
    )
    two_step_loss = mse_loss(second, chunks[:, 3:])

    rollout = model.rollout(windows[:, -2], windows[:, -1], c, ROLLOUT_STEPS)
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
        c,
        rollout,
    )


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return (
        (x_centered * y_centered).mean()
        / (x_centered.std(unbiased=False) * y_centered.std(unbiased=False) + 1e-8)
    ).item()


def _r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum()
    return (1.0 - residual / (total + 1e-8)).item()


def _evaluate(
    model: StructuredKoopmanModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    true_c: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z, learned_c, _ = _losses(model, chunks)
        latent_variance = z.var(dim=0, unbiased=False).mean()
        valid = learned_c.abs() <= 2.0
        omega_hat = torch.acos(torch.clamp(learned_c / 2.0, min=-1.0, max=1.0)) / float(DELTA_T)
        omega_r2 = _r2_score(omega_hat[valid], omegas[valid]) if valid.any() else float("nan")
        omega_corr = _pearson_correlation(omega_hat[valid], omegas[valid]) if valid.sum() > 1 else float("nan")
    return {
        "one_step_mse": losses["one_step_loss"].item(),
        "two_step_mse": losses["two_step_loss"].item(),
        "rollout_mse": losses["rollout_loss"].item(),
        "latent_variance": latent_variance.item(),
        "learned_c_true_c_correlation": _pearson_correlation(learned_c, true_c),
        "learned_c_true_c_mse": mse_loss(learned_c, true_c).item(),
        "omega_recovery_valid_fraction": valid.float().mean().item(),
        "omega_recovery_r2_from_c": omega_r2,
        "omega_recovery_correlation_from_c": omega_corr,
        "learned_c_mean": learned_c.mean().item(),
        "learned_c_std": learned_c.std(unbiased=False).item(),
        "true_c_mean": true_c.mean().item(),
        "true_c_std": true_c.std(unbiased=False).item(),
    }


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _save_training_loss(path: Path, history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    plt.plot([row["epoch"] for row in history], [row["test_rollout_mse"] for row in history], label="test rollout")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Stage 1B Structured Koopman Diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_rollout_examples(
    path: Path,
    model: StructuredKoopmanModel,
    sequences: torch.Tensor,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        z = model.encode(windows)
        c = model.coefficient(z)
        rollout = model.rollout(windows[:, -2], windows[:, -1], c, ROLLOUT_STEPS)
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
        axes[index].set_title(f"rollout example {index}, c={c[index].item():.4f}")
        axes[index].set_xlabel("step")
        axes[index].set_ylabel("x")
        axes[index].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_c_scatter(path: Path, learned_c: torch.Tensor, true_c: torch.Tensor) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(true_c.cpu(), learned_c.cpu(), s=28, alpha=0.85)
    min_value = min(true_c.min().item(), learned_c.min().item())
    max_value = max(true_c.max().item(), learned_c.max().item())
    plt.plot([min_value, max_value], [min_value, max_value], color="black", linestyle="--", linewidth=1)
    plt.xlabel("true 2 cos(omega dt)")
    plt.ylabel("learned c")
    plt.title("Structured Koopman coefficient recovery")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_html(setup: dict[str, Any], metrics: dict[str, Any], branch_a: dict[str, Any]) -> None:
    def rows(data: dict[str, Any]) -> str:
        return "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in data.items()
        )

    comparison = {
        "branch_a_one_step_mse": branch_a.get("one_step_mse", "missing"),
        "branch_b_one_step_mse": metrics["one_step_mse"],
        "branch_a_two_step_mse": branch_a.get("two_step_mse", "missing"),
        "branch_b_two_step_mse": metrics["two_step_mse"],
        "branch_a_rollout_mse": branch_a.get("rollout_mse", "missing"),
        "branch_b_rollout_mse": metrics["rollout_mse"],
        "branch_a_latent_variance": branch_a.get("latent_variance", "missing"),
        "branch_b_latent_variance": metrics["latent_variance"],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Structured Koopman Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Structured Koopman Diagnostic</h1>
  <p>Position-only trajectory-conditioned structured recurrence: K_i = [[0, 1], [-1, c_i]]. No velocity input, omega supervision, coefficient supervision, analytic SHO decoder, spectral loss, or contrastive loss.</p>
  <h2>Setup</h2>
  <table>{rows(setup)}</table>
  <h2>Branch B Metrics</h2>
  <table>{rows(metrics)}</table>
  <h2>Branch A vs Branch B</h2>
  <table>{rows(comparison)}</table>
  <section><h2>Training Loss</h2><img src="training_loss.png" alt="Training Loss"></section>
  <section><h2>Coefficient Recovery</h2><img src="c_recovery_scatter.png" alt="Coefficient Recovery"></section>
  <section><h2>Rollout Examples</h2><img src="rollout_examples.png" alt="Rollout Examples"></section>
</body>
</html>
"""
    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_true_c = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    model = StructuredKoopmanModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    history: list[dict[str, Any]] = []
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"

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
            loss, _, _, _, _ = _losses(model, chunks)
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
            metrics = _evaluate(model, test_sequences, test_omegas, test_true_c)
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
                    f"structured_koopman epoch {epoch:05d} "
                    f"train_loss={train_loss:.6e} best_rollout_mse={best_rollout:.6e}"
                ),
                flush=True,
            )
        if epochs_since_improvement >= PATIENCE:
            stopped_because = "patience_exhausted"
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = _evaluate(model, test_sequences, test_omegas, test_true_c)
    metrics.update(
        {
            "run_name": "structured_koopman_diagnostic",
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "koopman_latent_state_dim": 2,
            "koopman_matrix_dim": "2x2",
            "generated_parameters_per_trajectory": 1,
            "parameter_count": _parameter_count(model),
        },
    )
    setup = {
        "stage": "stage1B",
        "experiment": "structured_koopman_diagnostic",
        "branch": "B",
        "input": "position-only observation window",
        "state": "s_t = (x_{t-1}, x_t)",
        "latent": "z_i = Encoder(window_i), fixed during rollout",
        "koopman_operator": "K_i = [[0, 1], [-1, c_i]]",
        "coefficient": "c_i = 2 * tanh(G_theta(z_i) / 2)",
        "evolution": "s_{t+n} = K_i^n s_t",
        "prediction": "x_hat_{t+n} is second component of s_hat_{t+n}",
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
        "trajectory_latent_dim": LATENT_DIM,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "generator_hidden_size": GENERATOR_HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "eval_every": EVAL_EVERY,
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "num_train": DIAGNOSTIC_NUM_TRAIN,
        "num_test": DIAGNOSTIC_NUM_TEST,
        "seed": SEED,
    }

    branch_a = {}
    if BRANCH_A_METRICS_PATH.exists():
        branch_a = json.loads(BRANCH_A_METRICS_PATH.read_text(encoding="utf-8"))
    branch_comparison = {
        "branch_a": branch_a,
        "branch_b": metrics,
    }

    torch.save(model.state_dict(), OUTPUT_DIR / "learned_parameters.pt")
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", metrics)
    _write_json(OUTPUT_DIR / "branch_comparison.json", branch_comparison)
    _write_json(OUTPUT_DIR / "training_history.json", {"history": history})
    _write_csv(OUTPUT_DIR / "metrics.csv", [metrics])
    _save_training_loss(OUTPUT_DIR / "training_loss.png", history)
    with torch.no_grad():
        chunks = _fixed_eval_chunks(test_sequences)
        z = model.encode(chunks[:, :POSITION_WINDOW_SIZE])
        learned_c = model.coefficient(z)
    _save_c_scatter(OUTPUT_DIR / "c_recovery_scatter.png", learned_c, test_true_c)
    _save_rollout_examples(OUTPUT_DIR / "rollout_examples.png", model, test_sequences)
    _write_html(setup, metrics, branch_a)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
