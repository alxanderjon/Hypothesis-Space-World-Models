"""Seed stability for Branch B dense Koopman with global invariant consistency."""

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

OUTPUT_DIR = Path("outputs/sho/stage1B/global_invariant_seed_stability")
BASELINE_SOURCE_DIR = Path("outputs/sho/stage1B/dense_koopman_dim16_seed_stability")
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
MAX_EPOCHS = 200_000
PATIENCE = 25_000
EVAL_EVERY = 10_000
TRAINING_SEEDS = [0, 1, 2]
LAMBDA_GLOBAL = 1e-2
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
    centered = sequence - sequence.mean(dim=-1, keepdim=True)
    magnitudes = torch.fft.rfft(centered, dim=-1).abs()
    return magnitudes / (magnitudes.sum(dim=-1, keepdim=True) + DESCRIPTOR_EPS)


def _global_descriptor_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mse_loss(_global_descriptor(predicted), _global_descriptor(target))


def _losses(
    model: DenseKoopmanModel,
    chunks: torch.Tensor,
    *,
    lambda_global: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
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
        "global_descriptor_loss": global_loss,
        "reconstruction_loss": reconstruction_loss,
    }, z


def _omega_stats(features: torch.Tensor, omegas: torch.Tensor) -> tuple[float, float]:
    x = features.detach().cpu().numpy()
    y = omegas.detach().cpu().numpy()
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
    *,
    lambda_global: float,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    with torch.no_grad():
        chunks = _fixed_eval_chunks(sequences)
        _, parts, z = _losses(model, chunks, lambda_global=lambda_global)
    omega_r2, omega_corr = _omega_stats(z, omegas)
    return {
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "rollout_mse": parts["rollout_mse"].item(),
        "global_descriptor_loss": parts["global_descriptor_loss"].item(),
        "omega_r2_from_latent": omega_r2,
        "omega_correlation_from_latent": omega_corr,
        "abomega_multioutput_r2_from_latent": _multioutput_r2(z, labels),
        "latent_variance": z.var(dim=0, unbiased=False).mean().item(),
    }, z


def _copy_existing_baseline(seed: int, run_dir: Path) -> dict[str, Any] | None:
    source_dir = BASELINE_SOURCE_DIR / f"seed_{seed:03d}"
    source_metrics = source_dir / "metrics.json"
    if not source_metrics.exists():
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metrics.json", "history.csv", "training_history.json", "setup.json", "learned_parameters.pt"]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)
    metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
    metrics.update(
        {
            "branch": "baseline",
            "run_name": f"baseline_seed_{seed}",
            "lambda_global": 0.0,
            "training_seed": seed,
            "reused_from": str(source_dir),
        },
    )
    _write_json(run_dir / "metrics.json", metrics)
    return metrics


def _fill_missing_baseline_descriptor_metrics(
    rows: list[dict[str, Any]],
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> None:
    for row in rows:
        if row["branch"] != "baseline" or "global_descriptor_loss" in row:
            continue
        run_dir = OUTPUT_DIR / "baseline" / f"seed_{int(row['training_seed']):03d}"
        checkpoint_path = run_dir / "learned_parameters.pt"
        if not checkpoint_path.exists():
            continue
        model = DenseKoopmanModel()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        recomputed, _ = _evaluate(
            model,
            test_sequences,
            test_omegas,
            test_labels,
            lambda_global=0.0,
        )
        row["global_descriptor_loss"] = recomputed["global_descriptor_loss"]
        metric_path = run_dir / "metrics.json"
        if metric_path.exists():
            stored = json.loads(metric_path.read_text(encoding="utf-8"))
            stored["global_descriptor_loss"] = recomputed["global_descriptor_loss"]
            _write_json(metric_path, stored)


def _train_run(
    *,
    branch: str,
    training_seed: int,
    lambda_global: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = OUTPUT_DIR / branch / f"seed_{training_seed:03d}"
    metrics_path = run_dir / "metrics.json"
    history_path = run_dir / "training_history.json"
    if metrics_path.exists() and history_path.exists():
        return (
            json.loads(metrics_path.read_text(encoding="utf-8")),
            json.loads(history_path.read_text(encoding="utf-8")).get("history", []),
        )
    if branch == "baseline":
        copied = _copy_existing_baseline(training_seed, run_dir)
        if copied is not None:
            history = []
            history_path = run_dir / "training_history.json"
            if history_path.exists():
                history = json.loads(history_path.read_text(encoding="utf-8")).get("history", [])
            return copied, history

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(training_seed)
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
            "branch": branch,
            "training_seed": training_seed,
            "lambda_global": lambda_global,
            "latent_dim": LATENT_DIM,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "eval_every": EVAL_EVERY,
            "position_window_size": POSITION_WINDOW_SIZE,
            "rollout_steps": ROLLOUT_STEPS,
            "global_descriptor": "energy-normalized rollout rfft magnitude",
            "velocity_input": False,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "losses": "L_1step + L_2step + L_rollout + lambda_global * L_global_descriptor",
        },
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_sequences.shape[0])
        train_loss_sum = 0.0
        train_global_sum = 0.0
        train_count = 0
        for start_index in range(0, train_sequences.shape[0], BATCH_SIZE):
            indices = permutation[start_index : start_index + BATCH_SIZE]
            starts = torch.randint(0, max_start + 1, (indices.shape[0],))
            chunks = _sample_chunks(train_sequences, indices, starts)
            optimizer.zero_grad()
            loss, parts, _ = _losses(model, chunks, lambda_global=lambda_global)
            if not torch.isfinite(loss):
                stopped_because = "nan_or_inf_loss"
                break
            loss.backward()
            optimizer.step()
            count = indices.shape[0]
            train_loss_sum += loss.item() * count
            train_global_sum += parts["global_descriptor_loss"].item() * count
            train_count += count
        if stopped_because == "nan_or_inf_loss":
            break

        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics, _ = _evaluate(
                model,
                test_sequences,
                test_omegas,
                test_labels,
                lambda_global=lambda_global,
            )
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
                    "train_global_descriptor_loss": train_global_sum / train_count,
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
    metrics, _ = _evaluate(model, test_sequences, test_omegas, test_labels, lambda_global=lambda_global)
    metrics.update(
        {
            "branch": branch,
            "run_name": f"{branch}_seed_{training_seed}",
            "training_seed": training_seed,
            "lambda_global": lambda_global,
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
    return metrics, history


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "global_descriptor_loss",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "latent_variance",
        "best_epoch",
    ]
    aggregate_rows = []
    for branch in ["baseline", "global_invariant"]:
        branch_rows = [row for row in rows if row["branch"] == branch]
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in branch_rows], dtype=float)
            aggregate_rows.append(
                {
                    "branch": branch,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=0)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                },
            )
    return aggregate_rows


def _paired_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas = []
    for seed in TRAINING_SEEDS:
        baseline = next(row for row in rows if row["branch"] == "baseline" and row["training_seed"] == seed)
        global_run = next(row for row in rows if row["branch"] == "global_invariant" and row["training_seed"] == seed)
        deltas.append(
            {
                "training_seed": seed,
                "delta_rollout_mse_global_minus_baseline": global_run["rollout_mse"] - baseline["rollout_mse"],
                "delta_omega_r2_global_minus_baseline": global_run["omega_r2_from_latent"] - baseline["omega_r2_from_latent"],
                "delta_abomega_r2_global_minus_baseline": (
                    global_run["abomega_multioutput_r2_from_latent"]
                    - baseline["abomega_multioutput_r2_from_latent"]
                ),
                "delta_global_descriptor_loss_global_minus_baseline": (
                    global_run["global_descriptor_loss"] - baseline["global_descriptor_loss"]
                ),
            },
        )
    return deltas


def _save_seed_plot(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, log_y: bool = False) -> None:
    plt.figure(figsize=(6.8, 4.4))
    for branch in ["baseline", "global_invariant"]:
        branch_rows = sorted([row for row in rows if row["branch"] == branch], key=lambda row: row["training_seed"])
        plt.plot(
            [row["training_seed"] for row in branch_rows],
            [row[metric] for row in branch_rows],
            marker="o",
            label=branch,
        )
    if log_y:
        plt.yscale("log")
    plt.xlabel("training seed")
    plt.ylabel(ylabel)
    plt.title(ylabel)
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
    return str(value)


def _analysis(rows: list[dict[str, Any]], deltas: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> list[str]:
    rollout_deltas = [row["delta_rollout_mse_global_minus_baseline"] for row in deltas]
    omega_deltas = [row["delta_omega_r2_global_minus_baseline"] for row in deltas]
    abomega_deltas = [row["delta_abomega_r2_global_minus_baseline"] for row in deltas]
    descriptor_deltas = [row["delta_global_descriptor_loss_global_minus_baseline"] for row in deltas]

    def agg(branch: str, metric: str, stat: str) -> float:
        return float(next(row for row in aggregate if row["branch"] == branch and row["metric"] == metric)[stat])

    lines = [
        "This is a seed-stability test of a generic global dynamical invariant objective, not an SHO-specific frequency-supervision experiment.",
        f"Rollout improved on {sum(delta < 0 for delta in rollout_deltas)}/{len(rollout_deltas)} seeds.",
        f"Omega R2 improved on {sum(delta > 0 for delta in omega_deltas)}/{len(omega_deltas)} seeds.",
        f"(a,b,omega) R2 improved on {sum(delta > 0 for delta in abomega_deltas)}/{len(abomega_deltas)} seeds.",
        f"Global descriptor loss improved on {sum(delta < 0 for delta in descriptor_deltas)}/{len(descriptor_deltas)} seeds.",
        (
            "Rollout std baseline/global: "
            f"{agg('baseline', 'rollout_mse', 'std'):.4e} / {agg('global_invariant', 'rollout_mse', 'std'):.4e}."
        ),
        (
            "Omega R2 std baseline/global: "
            f"{agg('baseline', 'omega_r2_from_latent', 'std'):.4e} / "
            f"{agg('global_invariant', 'omega_r2_from_latent', 'std'):.4e}."
        ),
    ]
    if all(delta < 0 for delta in rollout_deltas):
        lines.append("The global invariant loss consistently improved rollout quality across seeds.")
    else:
        lines.append("The global invariant loss did not consistently improve rollout quality across seeds.")
    if all(delta > 0 for delta in omega_deltas) and all(delta > 0 for delta in abomega_deltas):
        lines.append("It consistently improved latent recovery by the linear probes.")
    else:
        lines.append("It did not consistently improve latent recovery by the linear probes.")
    if agg("global_invariant", "rollout_mse", "std") < agg("baseline", "rollout_mse", "std"):
        lines.append("It reduced rollout variance across seeds.")
    else:
        lines.append("It did not reduce rollout variance across seeds.")
    lines.append(
        "If rollout/descriptor improve without corresponding latent-probe gains, that points more toward optimization/trajectory regularization than a genuinely better linearly readable representation.",
    )
    return lines


def _write_html(
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    analysis: list[str],
) -> None:
    run_columns = [
        "branch",
        "training_seed",
        "rollout_mse",
        "one_step_mse",
        "two_step_mse",
        "global_descriptor_loss",
        "omega_r2_from_latent",
        "omega_correlation_from_latent",
        "abomega_multioutput_r2_from_latent",
        "latent_variance",
        "best_epoch",
        "epochs_completed",
        "stopped_because",
    ]
    aggregate_columns = ["branch", "metric", "mean", "std", "min", "max"]
    delta_columns = list(delta_rows[0].keys())

    def table(columns: list[str], table_rows: list[dict[str, Any]]) -> str:
        header = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in columns) + "</tr>"
            for row in table_rows
        )
        return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    analysis_html = "".join(f"<p>{escape(line)}</p>" for line in analysis)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Global Invariant Seed Stability</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 920px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Global Invariant Seed Stability</h1>
  <p>Branch B dense Koopman, latent_dim={LATENT_DIM}, position-only, no velocity, no omega or coefficient supervision, no analytic SHO decoder. Branch 2 uses lambda_global={LAMBDA_GLOBAL:g}; Branch 1 uses lambda_global=0.</p>
  <h2>Runs</h2>
  {table(run_columns, rows)}
  <h2>Paired Deltas</h2>
  {table(delta_columns, delta_rows)}
  <h2>Aggregate</h2>
  {table(aggregate_columns, aggregate_rows)}
  <h2>Analysis</h2>
  {analysis_html}
  <h2>Rollout MSE</h2><img src="rollout_mse_seed_comparison.png">
  <h2>Omega R2</h2><img src="omega_r2_seed_comparison.png">
  <h2>(a,b,omega) R2</h2><img src="abomega_r2_seed_comparison.png">
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
    for seed in TRAINING_SEEDS:
        baseline_metrics, baseline_history = _train_run(
            branch="baseline",
            training_seed=seed,
            lambda_global=0.0,
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            test_omegas=test_omegas,
            test_labels=test_labels,
        )
        rows.append(baseline_metrics)
        histories[baseline_metrics["run_name"]] = baseline_history
        global_metrics, global_history = _train_run(
            branch="global_invariant",
            training_seed=seed,
            lambda_global=LAMBDA_GLOBAL,
            train_sequences=train_sequences,
            test_sequences=test_sequences,
            test_omegas=test_omegas,
            test_labels=test_labels,
        )
        rows.append(global_metrics)
        histories[global_metrics["run_name"]] = global_history
        _write_json(OUTPUT_DIR / "partial_summary.json", {"runs": rows})
        _write_csv(OUTPUT_DIR / "summary.csv", rows)

    rows.sort(key=lambda row: (row["training_seed"], row["branch"]))
    _fill_missing_baseline_descriptor_metrics(rows, test_sequences, test_omegas, test_labels)
    aggregate_rows = _aggregate(rows)
    delta_rows = _paired_deltas(rows)
    analysis = _analysis(rows, delta_rows, aggregate_rows)
    _write_json(
        OUTPUT_DIR / "summary.json",
        {"runs": rows, "aggregate": aggregate_rows, "paired_deltas": delta_rows, "analysis": analysis},
    )
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "aggregate.csv", aggregate_rows)
    _write_csv(OUTPUT_DIR / "paired_deltas.csv", delta_rows)
    _save_seed_plot(rows, "rollout_mse", "rollout_mse_seed_comparison.png", "rollout MSE", log_y=True)
    _save_seed_plot(rows, "omega_r2_from_latent", "omega_r2_seed_comparison.png", "omega R2 from latent")
    _save_seed_plot(
        rows,
        "abomega_multioutput_r2_from_latent",
        "abomega_r2_seed_comparison.png",
        "(a,b,omega) R2 from latent",
    )
    _write_html(rows, aggregate_rows, delta_rows, analysis)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "runs": rows, "aggregate": aggregate_rows, "paired_deltas": delta_rows, "analysis": analysis}, indent=2), flush=True)


if __name__ == "__main__":
    main()
