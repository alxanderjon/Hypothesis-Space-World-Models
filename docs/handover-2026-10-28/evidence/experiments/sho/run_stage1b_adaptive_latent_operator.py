"""Stage 1B adaptive latent bottleneck operator-composition diagnostic."""

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
os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")

import matplotlib.pyplot as plt
import torch
from torch import nn

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover - sklearn is available in NNLearning.
    PCA = None
    LinearRegression = None

from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    OMEGA_RANGE,
    SEED,
    _make_base_conditions,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/adaptive_latent_operator")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 8
ENCODER_HIDDEN_SIZE = 16
OPERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10_000
DIAGNOSTIC_EPOCH_CUTOFF = 1_000
PATIENCE = 2_000
EVAL_EVERY = 50
LOG_EVERY = 250
GATE_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2]
GATE_THRESHOLD = 0.1


class ElementwiseLinear(nn.Module):
    """Linear layer implemented without torch matmul for this small diagnostic."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        bound = input_dim**-0.5
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(output_dim).uniform_(-bound, bound))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.unsqueeze(-2) * self.weight).sum(dim=-1) + self.bias


class AdaptiveLatentOperatorModel(nn.Module):
    """Infer z, gate it, and reuse one local operator for position evolution."""

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
        self.gate_logits = nn.Parameter(torch.zeros(latent_dim))
        self.operator = nn.Sequential(
            ElementwiseLinear(latent_dim + 2, operator_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(operator_hidden_size, operator_hidden_size),
            nn.SiLU(),
            ElementwiseLinear(operator_hidden_size, 1),
        )

    @property
    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def encode(self, position_windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(position_windows)
        return z, self.gates * z

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x_prev.unsqueeze(-1), x_curr.unsqueeze(-1), z_tilde), dim=-1)
        return self.operator(inputs).squeeze(-1)

    def rollout(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        z_tilde: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, z_tilde)
            predictions.append(next_x)
            prev = curr
            curr = next_x
        return torch.stack(predictions, dim=-1)


def _run_name(lambda_gate: float) -> str:
    return f"lambda_gate_{lambda_gate:.0e}".replace("+", "")


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
    labels = torch.stack(
        (
            initial_conditions[:, 0],
            initial_conditions[:, 1] / omegas,
            omegas,
        ),
        dim=-1,
    )
    return positions, omegas, labels


def _sample_chunks(sequences: torch.Tensor, indices: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(POSITION_WINDOW_SIZE + ROLLOUT_STEPS, device=sequences.device)
    return sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]


def _fixed_eval_chunks(sequences: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_chunks(sequences, indices, starts)


def _losses(
    model: AdaptiveLatentOperatorModel,
    chunks: torch.Tensor,
    *,
    lambda_gate: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z, z_tilde = model.encode(windows)

    one_step_z = z_tilde.repeat_interleave(chunks.shape[1] - 2, dim=0)
    one_step = model.step(
        chunks[:, :-2].reshape(-1),
        chunks[:, 1:-1].reshape(-1),
        one_step_z,
    ).reshape(chunks.shape[0], chunks.shape[1] - 2)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])

    two_step_z = z_tilde.repeat_interleave(chunks.shape[1] - 3, dim=0)
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

    rollout = model.rollout(windows[:, -2], windows[:, -1], z_tilde, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])
    gate_loss = lambda_gate * model.gates.sum()
    loss = one_step_loss + two_step_loss + rollout_loss + gate_loss
    return (
        loss,
        {
            "one_step_loss": one_step_loss,
            "two_step_loss": two_step_loss,
            "rollout_loss": rollout_loss,
            "gate_loss": gate_loss,
        },
        z,
        z_tilde,
        rollout,
    )


def _r2_score(features: torch.Tensor, targets: torch.Tensor) -> float:
    if LinearRegression is None:
        return float("nan")
    x = features.detach().cpu().numpy()
    y = targets.detach().cpu().numpy()
    model = LinearRegression()
    model.fit(x, y)
    return float(model.score(x, y))


def _evaluate(
    model: AdaptiveLatentOperatorModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    coefficient_omega_labels: torch.Tensor,
    *,
    lambda_gate: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z, z_tilde, _ = _losses(model, chunks, lambda_gate=lambda_gate)
        gates = model.gates.detach().cpu()
        effective_dim = int((gates > GATE_THRESHOLD).sum().item())
        latent_variance = z_tilde.var(dim=0, unbiased=False).mean()
        omega_r2 = _r2_score(z_tilde, omegas)
        coefficient_omega_r2 = _r2_score(z_tilde, coefficient_omega_labels)
    return {
        "one_step_mse": losses["one_step_loss"].item(),
        "two_step_mse": losses["two_step_loss"].item(),
        "rollout_mse": losses["rollout_loss"].item(),
        "latent_variance": latent_variance.item(),
        "gate_values": gates.tolist(),
        "effective_latent_dim": effective_dim,
        "omega_r2_from_gated_latent": omega_r2,
        "abomega_multioutput_r2_from_gated_latent": coefficient_omega_r2,
    }


def _save_training_loss(path: Path, history: list[dict[str, Any]], title: str) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    plt.plot([row["epoch"] for row in history], [row["test_rollout_mse"] for row in history], label="test rollout")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_rollout_examples(
    path: Path,
    model: AdaptiveLatentOperatorModel,
    sequences: torch.Tensor,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        _, z_tilde = model.encode(windows)
        rollout = model.rollout(windows[:, -2], windows[:, -1], z_tilde, ROLLOUT_STEPS)
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


def _save_gate_plot(path: Path, gate_values: list[float], title: str) -> None:
    plt.figure(figsize=(7, 4.2))
    plt.bar([f"z{i}" for i in range(len(gate_values))], gate_values)
    plt.axhline(GATE_THRESHOLD, color="black", linestyle="--", linewidth=1, label="effective dim threshold")
    plt.ylim(0.0, 1.0)
    plt.ylabel("sigmoid(g)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_latent_plot(
    path: Path,
    model: AdaptiveLatentOperatorModel,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, z_tilde = model.encode(chunks[:, :POSITION_WINDOW_SIZE])
    embedding = z_tilde[:, :2].detach().cpu()
    x_label = "gated z[0]"
    y_label = "gated z[1]"
    if PCA is not None and z_tilde.shape[1] > 2:
        embedding = torch.tensor(PCA(n_components=2).fit_transform(z_tilde.detach().cpu().numpy()))
        x_label = "gated latent PC1"
        y_label = "gated latent PC2"
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=omegas.cpu(), cmap="viridis", s=30)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Gated latent colored by held-out omega")
    plt.colorbar(scatter, label="omega")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_run_html(
    output_dir: Path,
    title: str,
    setup: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    def rows(data: dict[str, Any]) -> str:
        return "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in data.items()
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <h2>Setup</h2>
  <table>{rows(setup)}</table>
  <h2>Metrics</h2>
  <table>{rows(metrics)}</table>
  <section><h2>Training Loss</h2><img src="training_loss.png" alt="Training Loss"></section>
  <section><h2>Gate Values</h2><img src="gate_values.png" alt="Gate Values"></section>
  <section><h2>Latent Visualization</h2><img src="gated_latent_omega.png" alt="Gated Latent Visualization"></section>
  <section><h2>Rollout Examples</h2><img src="rollout_examples.png" alt="Rollout Examples"></section>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _run_variant(
    lambda_gate: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict[str, Any]:
    run_name = _run_name(lambda_gate)
    output_dir = OUTPUT_DIR / run_name
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"skipping completed adaptive latent diagnostic: {run_name}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = AdaptiveLatentOperatorModel()
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
            loss, _, _, _, _ = _losses(model, chunks, lambda_gate=lambda_gate)
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
            metrics = _evaluate(
                model,
                test_sequences,
                test_omegas,
                test_labels,
                lambda_gate=lambda_gate,
            )
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
                    f"{run_name} epoch {epoch:05d} train_loss={train_loss:.6e} "
                    f"best_rollout_mse={best_rollout:.6e}"
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

    metrics = _evaluate(
        model,
        test_sequences,
        test_omegas,
        test_labels,
        lambda_gate=lambda_gate,
    )
    metrics.update(
        {
            "run_name": run_name,
            "lambda_gate": lambda_gate,
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
        "experiment": "adaptive_latent_operator",
        "input": "position-only observation window",
        "state": "s_t = (x_{t-1}, x_t)",
        "latent": "z_i = Encoder(window_i), fixed during rollout",
        "gating": "z_tilde = sigmoid(g) * z",
        "operator": "F_theta(x_{t-1}, x_t, z_tilde_i) -> x_{t+1}",
        "loss": "L_1step + L_2step + L_rollout + lambda_gate * sum(sigmoid(g))",
        "lambda_gate": lambda_gate,
        "velocity_input": False,
        "omega_supervision": False,
        "coefficient_supervision": False,
        "analytic_decoder": False,
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
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "training_history.json", {"history": history})
    _write_csv(output_dir / "metrics.csv", [metrics])
    _save_training_loss(output_dir / "training_loss.png", history, f"Adaptive latent {run_name}")
    _save_gate_plot(output_dir / "gate_values.png", metrics["gate_values"], f"Gate values: {run_name}")
    _save_latent_plot(output_dir / "gated_latent_omega.png", model, test_sequences, test_omegas)
    _save_rollout_examples(output_dir / "rollout_examples.png", model, test_sequences)
    _write_run_html(output_dir, f"Stage 1B Adaptive Latent Operator: {run_name}", setup, metrics)
    return metrics


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "lambda_gate",
        "one_step_mse",
        "two_step_mse",
        "rollout_mse",
        "latent_variance",
        "effective_latent_dim",
        "omega_r2_from_gated_latent",
        "abomega_multioutput_r2_from_gated_latent",
        "gate_values",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    links = "".join(
        f'<li><a href="{escape(row["run_name"])}/index.html">{escape(row["run_name"])}</a></li>'
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Adaptive Latent Operator Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
  </style>
</head>
<body>
  <h1>Stage 1B Adaptive Latent Operator Diagnostics</h1>
  <p>Position-only local operator with an 8D learned gated trajectory latent. Omega and coefficient labels are used only for post-training diagnostics.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <h2>Run Directories</h2>
  <ul>{links}</ul>
</body>
</html>
"""
    (OUTPUT_DIR / "adaptive_latent_operator_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    rows = [
        _run_variant(lambda_gate, train_sequences, test_sequences, test_omegas, test_labels)
        for lambda_gate in GATE_LAMBDAS
    ]
    _write_json(OUTPUT_DIR / "adaptive_latent_operator_summary.json", {"runs": rows})
    _write_csv(OUTPUT_DIR / "adaptive_latent_operator_summary.csv", rows)
    _write_summary_html(rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
