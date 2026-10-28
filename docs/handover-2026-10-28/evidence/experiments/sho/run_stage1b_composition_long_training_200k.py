"""200k comparison of composition-based Stage 1B branches."""

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
from sklearn.linear_model import LinearRegression
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

OUTPUT_DIR = Path("outputs/sho/stage1B/composition_long_training_200k")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 4
ENCODER_HIDDEN_SIZE = 16
OPERATOR_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 200_000
PATIENCE = 25_000
EVAL_EVERY = 10_000
LOG_EVERY = 10_000


ElementwiseLinear = nn.Linear


class BranchALatentOperatorModel(nn.Module):
    """Existing latent-operator-composition architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ElementwiseLinear(POSITION_WINDOW_SIZE, ENCODER_HIDDEN_SIZE),
            nn.SiLU(),
            ElementwiseLinear(ENCODER_HIDDEN_SIZE, LATENT_DIM),
        )
        self.operator = nn.Sequential(
            ElementwiseLinear(LATENT_DIM + 2, OPERATOR_HIDDEN_SIZE),
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
    if prediction.ndim > 1:
        prediction = prediction.reshape(-1)
    corr = float(torch.corrcoef(torch.tensor([prediction, y.reshape(-1)]))[0, 1])
    return r2, corr


def _multioutput_r2(features: torch.Tensor, targets: torch.Tensor) -> float:
    model = LinearRegression()
    model.fit(features.detach().cpu().numpy(), targets.detach().cpu().numpy())
    return float(model.score(features.detach().cpu().numpy(), targets.detach().cpu().numpy()))


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


def _save_latent_scatter(path: Path, z: torch.Tensor, omegas: torch.Tensor, title: str) -> None:
    z_cpu = z.detach().cpu()
    plt.figure(figsize=(6.5, 5.2))
    scatter = plt.scatter(z_cpu[:, 0], z_cpu[:, 1], c=omegas.cpu(), cmap="viridis", s=30)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
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
        ax.set_title(f"{branch_name} rollout example {index}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _train_branch(
    branch_name: str,
    model: nn.Module,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir = OUTPUT_DIR / branch_name
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "training_history.json"
    if metrics_path.exists() and history_path.exists():
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(history_path.read_text(encoding="utf-8"))["history"],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latent_snapshots").mkdir(exist_ok=True)
    set_seed(SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    best_rollout = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    stopped_because = "max_epochs_reached"
    history: list[dict[str, Any]] = []
    _write_json(
        output_dir / "setup.json",
        {
            "run_name": branch_name,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "latent_dim": LATENT_DIM,
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
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_sum / train_count,
                    **eval_metrics,
                },
            )
            _write_json(output_dir / "training_history.json", {"history": history})
            _write_csv(output_dir / "history.csv", history)
            _save_latent_scatter(
                output_dir / "latent_snapshots" / f"latent_epoch_{epoch:06d}.png",
                z,
                test_omegas,
                f"{branch_name} latent at epoch {epoch}",
            )
            if epoch == 1 or epoch % LOG_EVERY == 0:
                print(
                    f"{branch_name} epoch {epoch:06d} rollout_mse={rollout_mse:.6e} best={best_rollout:.6e}",
                    flush=True,
                )
            if epochs_since_improvement >= PATIENCE:
                stopped_because = "patience_exhausted"
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics, z = _evaluate(branch_name, model, test_sequences, test_omegas, test_labels)
    metrics.update(
        {
            "run_name": branch_name,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "parameter_count": _parameter_count(model),
        },
    )
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "training_history.json", {"history": history})
    _write_csv(output_dir / "history.csv", history)
    _save_latent_scatter(output_dir / "latent_final.png", z, test_omegas, f"{branch_name} final latent")
    _save_rollout_examples(output_dir / "rollout_examples.png", branch_name, model, test_sequences)
    return metrics, history


def _save_metric_plot(histories: dict[str, list[dict[str, Any]]], metric: str, filename: str, ylabel: str, log_y: bool = True) -> None:
    plt.figure(figsize=(7, 4.5))
    for branch_name, history in histories.items():
        plt.plot([row["epoch"] for row in history], [row[metric] for row in history], marker="o", label=branch_name)
    if log_y:
        plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _write_html(metrics: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "latent_variance",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "parameter_count",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>"
        for row in metrics
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Composition Long Training 200k</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Composition Long Training 200k</h1>
  <p>Position-only, no omega labels, no coefficient labels, no spectral loss, no contrastive losses. Losses are exactly L_1step + L_2step + L_rollout.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <section><h2>Rollout MSE vs Epoch</h2><img src="rollout_mse_vs_epoch.png"></section>
  <section><h2>One-Step MSE vs Epoch</h2><img src="one_step_mse_vs_epoch.png"></section>
  <section><h2>Two-Step MSE vs Epoch</h2><img src="two_step_mse_vs_epoch.png"></section>
  <section><h2>Omega R2 vs Epoch</h2><img src="omega_r2_vs_epoch.png"></section>
  <section><h2>(a,b,omega) R2 vs Epoch</h2><img src="abomega_r2_vs_epoch.png"></section>
  <section><h2>Latent Variance vs Epoch</h2><img src="latent_variance_vs_epoch.png"></section>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    branch_specs = [
        ("branch_a_latent_operator", BranchALatentOperatorModel()),
        ("branch_b_dense_koopman", BranchBDenseKoopmanModel()),
    ]
    metrics: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for branch_name, model in branch_specs:
        branch_metrics, history = _train_branch(branch_name, model, train_sequences, test_sequences, test_omegas, test_labels)
        metrics.append(branch_metrics)
        histories[branch_name] = history

    _write_json(OUTPUT_DIR / "summary.json", {"runs": metrics})
    _write_csv(OUTPUT_DIR / "summary.csv", metrics)
    _save_metric_plot(histories, "rollout_mse", "rollout_mse_vs_epoch.png", "rollout MSE")
    _save_metric_plot(histories, "one_step_mse", "one_step_mse_vs_epoch.png", "one-step MSE")
    _save_metric_plot(histories, "two_step_mse", "two_step_mse_vs_epoch.png", "two-step MSE")
    _save_metric_plot(histories, "omega_r2_from_latent", "omega_r2_vs_epoch.png", "omega R2 from latent", log_y=False)
    _save_metric_plot(histories, "abomega_multioutput_r2_from_latent", "abomega_r2_vs_epoch.png", "(a,b,omega) R2 from latent", log_y=False)
    _save_metric_plot(histories, "latent_variance", "latent_variance_vs_epoch.png", "latent variance", log_y=False)
    _write_html(metrics)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
