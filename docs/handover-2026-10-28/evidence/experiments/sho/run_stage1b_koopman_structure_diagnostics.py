"""Stage 1B Koopman structure diagnostics."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/koopman_structure_diagnostics")
DIAGNOSTIC_NUM_TRAIN = 128
DIAGNOSTIC_NUM_TEST = 64
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
ENCODER_HIDDEN_SIZE = 16
GENERATOR_HIDDEN_SIZE = 16
BATCH_SIZE = DIAGNOSTIC_NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10_000
PATIENCE = 2_000
EVAL_EVERY = 50
LOG_EVERY = 250
STRUCTURED_LATENT_DIMS = [1, 2, 4]
DENSE_LATENT_DIM = 4


class ElementwiseLinear(nn.Module):
    """Linear layer implemented without torch matmul for this small diagnostic."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        bound = input_dim**-0.5
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(output_dim).uniform_(-bound, bound))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.unsqueeze(-2) * self.weight).sum(dim=-1) + self.bias


class BaseKoopmanModel(nn.Module):
    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def rollout(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        operator_params: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        predictions = []
        prev = x_prev
        curr = x_curr
        for _ in range(steps):
            next_x = self.step(prev, curr, operator_params)
            predictions.append(next_x)
            prev = curr
            curr = next_x
        return torch.stack(predictions, dim=-1)


class StructuredKoopmanModel(BaseKoopmanModel):
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
            ElementwiseLinear(GENERATOR_HIDDEN_SIZE, 1),
        )

    def operator_params(self, z: torch.Tensor) -> torch.Tensor:
        raw_c = self.generator(z).squeeze(-1)
        return 2.0 * torch.tanh(0.5 * raw_c)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return c * x_curr - x_prev


class DenseKoopmanModel(BaseKoopmanModel):
    def __init__(self, latent_dim: int = DENSE_LATENT_DIM) -> None:
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

    def operator_params(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def step(self, x_prev: torch.Tensor, x_curr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        a, b, c, d = params.unbind(dim=-1)
        return c * x_prev + d * x_curr


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
    model: BaseKoopmanModel,
    chunks: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    z = model.encode(windows)
    params = model.operator_params(z)

    params_one = params.repeat_interleave(chunks.shape[1] - 2, dim=0)
    one_step = model.step(
        chunks[:, :-2].reshape(-1),
        chunks[:, 1:-1].reshape(-1),
        params_one,
    ).reshape(chunks.shape[0], chunks.shape[1] - 2)
    one_step_loss = mse_loss(one_step, chunks[:, 2:])

    params_two = params.repeat_interleave(chunks.shape[1] - 3, dim=0)
    first = model.step(chunks[:, :-3].reshape(-1), chunks[:, 1:-2].reshape(-1), params_two)
    second = model.step(chunks[:, 1:-2].reshape(-1), first, params_two).reshape(
        chunks.shape[0],
        chunks.shape[1] - 3,
    )
    two_step_loss = mse_loss(second, chunks[:, 3:])

    rollout = model.rollout(windows[:, -2], windows[:, -1], params, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, chunks[:, POSITION_WINDOW_SIZE:])
    loss = one_step_loss + two_step_loss + rollout_loss
    return loss, {"one_step_loss": one_step_loss, "two_step_loss": two_step_loss, "rollout_loss": rollout_loss}, z, params, rollout


def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return ((x_centered * y_centered).mean() / (x_centered.std(unbiased=False) * y_centered.std(unbiased=False) + 1e-8)).item()


def _r2_score(prediction: torch.Tensor, target: torch.Tensor) -> float:
    residual = (target - prediction).square().sum()
    total = (target - target.mean()).square().sum()
    return (1.0 - residual / (total + 1e-8)).item()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _structured_param_metrics(params: torch.Tensor, omegas: torch.Tensor, true_c: torch.Tensor) -> dict[str, float]:
    valid = params.abs() <= 2.0
    omega_hat = torch.acos(torch.clamp(params / 2.0, min=-1.0, max=1.0)) / float(DELTA_T)
    return {
        "learned_c_true_c_correlation": _pearson_correlation(params, true_c),
        "learned_c_true_c_mse": mse_loss(params, true_c).item(),
        "omega_recovery_r2_from_c": _r2_score(omega_hat[valid], omegas[valid]) if valid.any() else float("nan"),
        "omega_recovery_correlation_from_c": _pearson_correlation(omega_hat[valid], omegas[valid]) if valid.sum() > 1 else float("nan"),
    }


def _evaluate(model: BaseKoopmanModel, sequences: torch.Tensor, omegas: torch.Tensor, true_c: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, losses, z, params, _ = _losses(model, chunks)
        metrics: dict[str, Any] = {
            "one_step_mse": losses["one_step_loss"].item(),
            "two_step_mse": losses["two_step_loss"].item(),
            "rollout_mse": losses["rollout_loss"].item(),
            "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
        }
        if isinstance(model, StructuredKoopmanModel):
            metrics.update(_structured_param_metrics(params, omegas, true_c))
        else:
            metrics.update(
                {
                    "learned_c_true_c_correlation": float("nan"),
                    "learned_c_true_c_mse": float("nan"),
                    "omega_recovery_r2_from_c": float("nan"),
                    "omega_recovery_correlation_from_c": float("nan"),
                },
            )
    return metrics


def _save_rollout_examples(path: Path, model: BaseKoopmanModel, sequences: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        windows = chunks[:, :POSITION_WINDOW_SIZE]
        params = model.operator_params(model.encode(windows))
        rollout = model.rollout(windows[:, -2], windows[:, -1], params, ROLLOUT_STEPS)
        targets = chunks[:, POSITION_WINDOW_SIZE:]
    fig, axes = plt.subplots(4, 1, figsize=(8, 9.5))
    observed_x = torch.arange(POSITION_WINDOW_SIZE).cpu()
    rollout_x = torch.arange(POSITION_WINDOW_SIZE, POSITION_WINDOW_SIZE + ROLLOUT_STEPS).cpu()
    for index, ax in enumerate(axes):
        ax.plot(observed_x, windows[index].cpu(), label="observed window")
        ax.plot(rollout_x, targets[index].cpu(), label="target rollout")
        ax.plot(rollout_x, rollout[index].cpu(), "--", label="predicted rollout")
        ax.set_title(f"rollout example {index}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run_variant(
    run_name: str,
    model: BaseKoopmanModel,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_true_c: torch.Tensor,
) -> dict[str, Any]:
    output_dir = OUTPUT_DIR / run_name
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
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
        history.append({"epoch": epoch, "train_loss": train_loss, "test_rollout_mse": best_rollout})
        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(f"{run_name} epoch {epoch:05d} train_loss={train_loss:.6e} best_rollout_mse={best_rollout:.6e}", flush=True)
        if epochs_since_improvement >= PATIENCE:
            stopped_because = "patience_exhausted"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _evaluate(model, test_sequences, test_omegas, test_true_c)
    variant_type = "structured" if isinstance(model, StructuredKoopmanModel) else "dense"
    generated_params = 1 if variant_type == "structured" else 4
    metrics.update(
        {
            "run_name": run_name,
            "variant_type": variant_type,
            "latent_dim": model.latent_dim,
            "generated_parameters_per_trajectory": generated_params,
            "parameter_count": _parameter_count(model),
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
    )
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "training_history.json", {"history": history})
    _save_rollout_examples(output_dir / "rollout_examples.png", model, test_sequences)
    return metrics


def _save_summary_plots(rows: list[dict[str, Any]]) -> None:
    structured = [row for row in rows if row["variant_type"] == "structured"]
    structured = sorted(structured, key=lambda row: row["latent_dim"])
    dims = [row["latent_dim"] for row in structured]
    plt.figure(figsize=(6, 4))
    plt.plot(dims, [row["rollout_mse"] for row in structured], marker="o")
    plt.yscale("log")
    plt.xlabel("structured Koopman latent_dim")
    plt.ylabel("rollout MSE")
    plt.title("Rollout MSE vs latent dimension")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_vs_latent_dim.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(dims, [row["learned_c_true_c_correlation"] for row in structured], marker="o", label="corr")
    plt.plot(dims, [row["learned_c_true_c_mse"] for row in structured], marker="o", label="MSE")
    plt.xlabel("structured Koopman latent_dim")
    plt.title("c recovery vs latent dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "c_recovery_vs_latent_dim.png")
    plt.close()

    selected = [row for row in rows if row["run_name"] in {"structured_latent_dim_4", "dense_koopman_latent_dim_4"}]
    plt.figure(figsize=(7, 4))
    labels = [row["run_name"].replace("_latent_dim_4", "") for row in selected]
    plt.bar(labels, [row["rollout_mse"] for row in selected])
    plt.yscale("log")
    plt.ylabel("rollout MSE")
    plt.title("Structured vs dense trajectory-conditioned Koopman")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "structured_vs_dense_koopman.png")
    plt.close()


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "variant_type",
        "latent_dim",
        "one_step_mse",
        "two_step_mse",
        "rollout_mse",
        "latent_variance",
        "learned_c_true_c_correlation",
        "learned_c_true_c_mse",
        "omega_recovery_r2_from_c",
        "omega_recovery_correlation_from_c",
        "generated_parameters_per_trajectory",
        "parameter_count",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>" for row in rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Koopman Structure Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Koopman Structure Diagnostics</h1>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  <section><h2>Rollout MSE vs Latent Dimension</h2><img src="rollout_mse_vs_latent_dim.png"></section>
  <section><h2>c Recovery vs Latent Dimension</h2><img src="c_recovery_vs_latent_dim.png"></section>
  <section><h2>Structured vs Dense Koopman</h2><img src="structured_vs_dense_koopman.png"></section>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=SEED)
    test_sequences, test_omegas, test_true_c = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=SEED + 1)
    rows: list[dict[str, Any]] = []
    rows.append(_run_variant("dense_koopman_latent_dim_4", DenseKoopmanModel(DENSE_LATENT_DIM), train_sequences, test_sequences, test_omegas, test_true_c))
    for latent_dim in STRUCTURED_LATENT_DIMS:
        rows.append(_run_variant(f"structured_latent_dim_{latent_dim}", StructuredKoopmanModel(latent_dim), train_sequences, test_sequences, test_omegas, test_true_c))
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_json(OUTPUT_DIR / "summary.json", {"runs": rows})
    _save_summary_plots(rows)
    _write_summary_html(rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
