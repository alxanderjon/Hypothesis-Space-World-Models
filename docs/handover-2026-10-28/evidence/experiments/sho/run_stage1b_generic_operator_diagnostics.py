"""Stage 1B generic position-only operator diagnostics.

These diagnostics deliberately avoid the SHO analytic decoder and do not expose
velocity, omega labels, coefficient labels, auxiliary physics labels, or
regularizers. The only observed input is a window of positions.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    NUM_TEST,
    NUM_TRAIN,
    OMEGA_RANGE,
    SEED,
    _make_base_conditions,
    _state_at_time,
)
from hypothesis_space_world_models.training import mse_loss, set_seed


OUTPUT_DIR = Path("outputs/sho/stage1B/generic_operator_diagnostics")
POSITION_WINDOW_SIZE = 20
ROLLOUT_STEPS = 16
TOTAL_SEQUENCE_STEPS = 64
LATENT_DIM = 16
HIDDEN_SIZE = 32
BATCH_SIZE = NUM_TRAIN
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10_000
DIAGNOSTIC_EPOCH_CUTOFF = 1_000
PATIENCE = 2_000
EVAL_EVERY = 50
LOG_EVERY = 250
CONTRASTIVE_MARGIN = 1.0


class PositionOperatorModel(nn.Module):
    """Encode a position window into z, then roll out x_{t+1} = F_z(x_t)."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(POSITION_WINDOW_SIZE, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim),
        )
        self.operator = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def encode(self, position_windows: torch.Tensor) -> torch.Tensor:
        return self.encoder(position_windows)

    def step(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.operator(torch.cat((x.unsqueeze(-1), z), dim=-1)).squeeze(-1)

    def rollout(self, x0: torch.Tensor, z: torch.Tensor, steps: int) -> torch.Tensor:
        values = []
        x = x0
        for _ in range(steps):
            x = self.step(x, z)
            values.append(x)
        return torch.stack(values, dim=-1)


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


def _make_position_sequences(num_trajectories: int, *, seed: int) -> torch.Tensor:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    times = (
        torch.arange(TOTAL_SEQUENCE_STEPS, dtype=initial_conditions.dtype)
        * float(DELTA_T)
    )
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions


def _sample_windows(
    sequences: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(
        POSITION_WINDOW_SIZE + ROLLOUT_STEPS,
        device=sequences.device,
    )
    chunks = sequences[indices.unsqueeze(-1), starts.unsqueeze(-1) + offsets]
    windows = chunks[:, :POSITION_WINDOW_SIZE]
    targets = chunks[:, POSITION_WINDOW_SIZE:]
    return windows, targets


def _fixed_eval_windows(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.arange(sequences.shape[0], device=sequences.device)
    starts = torch.zeros(sequences.shape[0], dtype=torch.long, device=sequences.device)
    return _sample_windows(sequences, indices, starts)


def _operator_losses(
    model: PositionOperatorModel,
    windows: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    z = model.encode(windows)
    one_step_pred = model.step(windows[:, :-1].reshape(-1), z.repeat_interleave(POSITION_WINDOW_SIZE - 1, dim=0))
    one_step_pred = one_step_pred.reshape(windows.shape[0], POSITION_WINDOW_SIZE - 1)
    one_step_loss = mse_loss(one_step_pred, windows[:, 1:])

    two_step_start = windows[:, :-2].reshape(-1)
    z_two = z.repeat_interleave(POSITION_WINDOW_SIZE - 2, dim=0)
    two_step = model.step(model.step(two_step_start, z_two), z_two)
    two_step = two_step.reshape(windows.shape[0], POSITION_WINDOW_SIZE - 2)
    composition_loss = mse_loss(two_step, windows[:, 2:])

    rollout = model.rollout(windows[:, -1], z, ROLLOUT_STEPS)
    rollout_loss = mse_loss(rollout, targets)
    base_loss = one_step_loss + rollout_loss + 0.5 * composition_loss
    return (
        base_loss,
        {
            "one_step_loss": one_step_loss,
            "composition_loss": composition_loss,
            "rollout_loss": rollout_loss,
        },
        z,
        rollout,
    )


def _spectral_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mag = torch.fft.rfft(predicted, dim=-1).abs()
    target_mag = torch.fft.rfft(target, dim=-1).abs()
    pred_mag = pred_mag / (pred_mag.norm(dim=-1, keepdim=True) + 1e-8)
    target_mag = target_mag / (target_mag.norm(dim=-1, keepdim=True) + 1e-8)
    return mse_loss(pred_mag, target_mag)


def _evaluate(
    model: PositionOperatorModel,
    sequences: torch.Tensor,
    *,
    variant: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        windows, targets = _fixed_eval_windows(sequences)
        _, losses, z, rollout = _operator_losses(model, windows, targets)
        spectral = _spectral_loss(rollout, targets)
        latent_variance = z.var(dim=0, unbiased=False).mean()
        metrics = {
            "one_step_mse": losses["one_step_loss"].item(),
            "composition_mse": losses["composition_loss"].item(),
            "multi_step_rollout_mse": losses["rollout_loss"].item(),
            "spectral_mse": spectral.item(),
            "latent_variance": latent_variance.item(),
        }
        if variant == "latent_same_different_consistency":
            same_windows = sequences[:, 4 : 4 + POSITION_WINDOW_SIZE]
            z_same = model.encode(same_windows)
            z_different = z_same.roll(1, dims=0)
            same_dist = (z - z_same).norm(dim=-1)
            different_dist = (z - z_different).norm(dim=-1)
            metrics.update(
                {
                    "same_trajectory_latent_distance": same_dist.mean().item(),
                    "different_trajectory_latent_distance": different_dist.mean().item(),
                    "latent_distance_ratio": (
                        same_dist.mean() / (different_dist.mean() + 1e-8)
                    ).item(),
                },
            )
        else:
            metrics.update(
                {
                    "same_trajectory_latent_distance": float("nan"),
                    "different_trajectory_latent_distance": float("nan"),
                    "latent_distance_ratio": float("nan"),
                },
            )
    return metrics


def _training_loss_for_variant(
    model: PositionOperatorModel,
    windows: torch.Tensor,
    targets: torch.Tensor,
    sequences: torch.Tensor,
    indices: torch.Tensor,
    starts: torch.Tensor,
    *,
    variant: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, losses, z, rollout = _operator_losses(model, windows, targets)
    if variant == "latent_same_different_consistency":
        max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
        same_starts = torch.clamp(starts + 4, max=max_start)
        same_windows, _ = _sample_windows(sequences, indices, same_starts)
        z_same = model.encode(same_windows)
        z_different = z_same.roll(1, dims=0)
        same_dist = (z - z_same).norm(dim=-1)
        different_dist = (z - z_different).norm(dim=-1)
        same_loss = same_dist.square().mean()
        separation_loss = F.relu(CONTRASTIVE_MARGIN - different_dist).square().mean()
        loss = loss + 0.1 * same_loss + 0.1 * separation_loss
        losses = {**losses, "same_loss": same_loss, "separation_loss": separation_loss}
    elif variant == "spectral_consistency":
        spectral = _spectral_loss(rollout, targets)
        loss = loss + spectral
        losses = {**losses, "spectral_loss": spectral}
    return loss, losses


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
    model: PositionOperatorModel,
    sequences: torch.Tensor,
    title: str,
    num_examples: int = 4,
) -> None:
    model.eval()
    with torch.no_grad():
        windows, targets = _fixed_eval_windows(sequences)
        z = model.encode(windows)
        rollout = model.rollout(windows[:, -1], z, ROLLOUT_STEPS)
    count = min(num_examples, sequences.shape[0])
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.4 * count))
    if count == 1:
        axes = [axes]
    x_axis = torch.arange(ROLLOUT_STEPS).cpu()
    for index in range(count):
        axes[index].plot(x_axis, targets[index].cpu(), label="target")
        axes[index].plot(x_axis, rollout[index].cpu(), "--", label="prediction")
        axes[index].set_title(f"{title}: rollout example {index}")
        axes[index].set_xlabel("rollout step")
        axes[index].set_ylabel("x")
        axes[index].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_run_html(output_dir: Path, title: str, setup: dict[str, Any], metrics: dict[str, Any]) -> None:
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
  <section><h2>Rollout Examples</h2><img src="rollout_examples.png" alt="Rollout Examples"></section>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _run_variant(variant: str, train_sequences: torch.Tensor, test_sequences: torch.Tensor) -> dict[str, Any]:
    output_dir = OUTPUT_DIR / variant
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        print(f"skipping completed generic operator diagnostic: {variant}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = PositionOperatorModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    max_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE - ROLLOUT_STEPS
    history: list[dict[str, Any]] = []
    best_test = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    epochs_since_improvement = 0
    last_eval_metrics: dict[str, float] | None = None
    stopped_because = "max_epochs_reached"

    for epoch in range(1, min(MAX_EPOCHS, DIAGNOSTIC_EPOCH_CUTOFF) + 1):
        model.train()
        permutation = torch.randperm(train_sequences.shape[0])
        train_loss_sum = 0.0
        train_count = 0
        for start_index in range(0, train_sequences.shape[0], BATCH_SIZE):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            starts = torch.randint(0, max_start + 1, (indices.shape[0],))
            windows, targets = _sample_windows(train_sequences, indices, starts)
            optimizer.zero_grad()
            loss, _ = _training_loss_for_variant(
                model,
                windows,
                targets,
                train_sequences,
                indices,
                starts,
                variant=variant,
            )
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
            test_metrics = _evaluate(model, test_sequences, variant=variant)
            last_eval_metrics = test_metrics
        else:
            test_metrics = last_eval_metrics or _evaluate(model, test_sequences, variant=variant)
        test_rollout = test_metrics["multi_step_rollout_mse"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_rollout_mse": test_rollout,
                **test_metrics,
            },
        )
        if epoch == 1 or epoch % EVAL_EVERY == 0:
            if test_rollout < best_test:
                best_test = test_rollout
                best_epoch = epoch
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
        else:
            epochs_since_improvement += 1

        if epoch == 1 or epoch % LOG_EVERY == 0:
            print(
                (
                    f"{variant} epoch {epoch:05d} train_loss={train_loss:.6e} "
                    f"test_rollout_mse={test_rollout:.6e} best={best_test:.6e}"
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

    metrics = _evaluate(model, test_sequences, variant=variant)
    metrics.update(
        {
            "run_name": variant,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "max_epochs": MAX_EPOCHS,
            "diagnostic_epoch_cutoff": DIAGNOSTIC_EPOCH_CUTOFF,
            "patience": PATIENCE,
            "input": "position-only window [x_t, ..., x_{t+k}]",
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
        },
    )
    setup = {
        "stage": "stage1B",
        "experiment": "generic_operator_diagnostics",
        "run_name": variant,
        "encoder_type": "flattened_position_window_mlp",
        "position_window_size": POSITION_WINDOW_SIZE,
        "rollout_steps": ROLLOUT_STEPS,
        "total_sequence_steps": TOTAL_SEQUENCE_STEPS,
        "latent_dim": LATENT_DIM,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "diagnostic_epoch_cutoff": DIAGNOSTIC_EPOCH_CUTOFF,
        "patience": PATIENCE,
        "eval_every": EVAL_EVERY,
        "omega_range": OMEGA_RANGE,
        "delta_t": DELTA_T,
        "loss_description": _loss_description(variant),
    }
    torch.save(model.state_dict(), output_dir / "learned_parameters.pt")
    _write_json(output_dir / "setup.json", setup)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "training_history.json", {"history": history})
    _save_training_loss(output_dir / "training_loss.png", history, f"Stage 1B {variant}")
    _save_rollout_examples(output_dir / "rollout_examples.png", model, test_sequences, f"Stage 1B {variant}")
    _write_run_html(output_dir, f"Stage 1B {variant}", setup, metrics)
    return metrics


def _loss_description(variant: str) -> str:
    if variant == "operator_consistency_only":
        return "one-step reconstruction + multi-step rollout + two-step operator composition"
    if variant == "latent_same_different_consistency":
        return "operator consistency plus same-trajectory latent closeness and different-trajectory contrastive separation"
    if variant == "spectral_consistency":
        return "operator consistency plus FFT magnitude consistency on rollout windows"
    raise ValueError(f"unknown variant: {variant}")


def _save_summary_plots(rows: list[dict[str, Any]]) -> None:
    labels = [row["run_name"] for row in rows]
    for filename, metric_key, ylabel, log_y in [
        ("rollout_mse_by_constraint.png", "multi_step_rollout_mse", "multi-step rollout MSE", True),
        ("one_step_mse_by_constraint.png", "one_step_mse", "one-step MSE", True),
        ("latent_variance_by_constraint.png", "latent_variance", "latent variance", True),
        ("spectral_mse_by_constraint.png", "spectral_mse", "spectral MSE", True),
    ]:
        plt.figure(figsize=(8, 4.5))
        plt.bar(labels, [float(row[metric_key]) for row in rows])
        plt.ylabel(ylabel)
        if log_y:
            plt.yscale("log")
        plt.title(f"Stage 1B Generic Operator Diagnostics: {ylabel}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_summary_html(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "one_step_mse",
        "multi_step_rollout_mse",
        "composition_mse",
        "spectral_mse",
        "latent_variance",
        "same_trajectory_latent_distance",
        "different_trajectory_latent_distance",
        "latent_distance_ratio",
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
    plots = "\n".join(
        f'<section><h2>{escape(Path(plot).stem)}</h2><img src="{escape(plot)}" alt="{escape(plot)}"></section>'
        for plot in [
            "rollout_mse_by_constraint.png",
            "one_step_mse_by_constraint.png",
            "latent_variance_by_constraint.png",
            "spectral_mse_by_constraint.png",
        ]
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Generic Operator Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 960px; width: 100%; height: auto; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  <h1>Stage 1B Generic Operator Diagnostics</h1>
  <p>Position-only input, no SHO decoder, no velocity input, no omega labels, no coefficient labels.</p>
  <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
  {plots}
  <h2>Run Directories</h2>
  <ul>{links}</ul>
</body>
</html>
"""
    (OUTPUT_DIR / "generic_operator_diagnostics_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_sequences = _make_position_sequences(NUM_TRAIN, seed=SEED)
    test_sequences = _make_position_sequences(NUM_TEST, seed=SEED + 1)
    rows = [
        _run_variant("operator_consistency_only", train_sequences, test_sequences),
        _run_variant("latent_same_different_consistency", train_sequences, test_sequences),
        _run_variant("spectral_consistency", train_sequences, test_sequences),
    ]
    _write_csv(OUTPUT_DIR / "generic_operator_diagnostics_summary.csv", rows)
    _write_json(OUTPUT_DIR / "generic_operator_diagnostics_summary.json", {"runs": rows})
    _save_summary_plots(rows)
    _write_summary_html(rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
