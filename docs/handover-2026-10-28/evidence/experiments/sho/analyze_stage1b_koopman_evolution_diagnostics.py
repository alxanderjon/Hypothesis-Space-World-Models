"""Koopman evolution diagnostics for separated dynamics/state attention models."""

from __future__ import annotations

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
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    DIAGNOSTIC_NUM_TEST,
    STATE_LATENT_DIM,
    TEST_DATA_SEED,
    SeparatedKoopmanWithDynamicsEncoder,
    _fixed_dynamics_windows,
    _make_position_data,
    _positions_to_states,
)


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/koopman_evolution_diagnostics")
K_VALUES = [1, 2, 4, 8, 16, 32]
NUM_DYNAMICS_WINDOWS = 32
CHECKPOINTS = [
    {
        "model_name": "no_ae_attention_m32",
        "checkpoint": Path("outputs/sho/stage1B/attention_large_window_sweep/attention_m32/learned_parameters.pt"),
        "metrics": Path("outputs/sho/stage1B/attention_large_window_sweep/attention_m32/metrics.json"),
    },
    {
        "model_name": "best_ae_lambda_0p01",
        "checkpoint": Path("outputs/sho/stage1B/state_autoencoder_loss_diagnostic/lambda_ae_0p01/learned_parameters.pt"),
        "metrics": Path("outputs/sho/stage1B/state_autoencoder_loss_diagnostic/lambda_ae_0p01/metrics.json"),
    },
]


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _batched_matrix_power(k_matrix: torch.Tensor, power: int) -> torch.Tensor:
    return torch.linalg.matrix_power(k_matrix, power)


def _evaluate_model(
    model_name: str,
    checkpoint: Path,
    source_metrics: Path,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    model = SeparatedKoopmanWithDynamicsEncoder(encoder_type="attention")
    model.load_state_dict(_load_state_dict(checkpoint))
    model.eval()
    with torch.no_grad():
        dynamics_windows = _fixed_dynamics_windows(sequences, NUM_DYNAMICS_WINDOWS)
        d = model.encode_dynamics(dynamics_windows)
        k_matrix = model.koopman(d)
        states = _positions_to_states(sequences)
        h_states = model.encode_state(states.reshape(-1, 2)).reshape(
            states.shape[0],
            states.shape[1],
            STATE_LATENT_DIM,
        )
        decoded_true_h = model.decode_state(h_states.reshape(-1, STATE_LATENT_DIM)).reshape_as(states)

    step_rows: list[dict[str, Any]] = []
    per_trajectory_rollout_errors = []
    for k in K_VALUES:
        with torch.no_grad():
            k_power = _batched_matrix_power(k_matrix, k)
            h_start = h_states[:, :-k]
            h_target = h_states[:, k:]
            h_pred = torch.einsum("nij,ntj->nti", k_power, h_start)
            decoded_pred = model.decode_state(h_pred.reshape(-1, STATE_LATENT_DIM)).reshape(
                h_pred.shape[0],
                h_pred.shape[1],
                2,
            )
            state_target = states[:, k:]
            latent_error_by_traj = (h_pred - h_target).square().mean(dim=(1, 2))
            decoded_error_by_traj = (decoded_pred - state_target).square().mean(dim=(1, 2))
            decoded_position_error_by_traj = (decoded_pred[..., 1] - state_target[..., 1]).square().mean(dim=1)
        if k == 16:
            per_trajectory_rollout_errors = decoded_position_error_by_traj.detach().cpu().numpy().tolist()
        step_rows.append(
            {
                "model_name": model_name,
                "k": k,
                "latent_transition_mse": latent_error_by_traj.mean().item(),
                "latent_transition_mse_std": latent_error_by_traj.std(unbiased=False).item(),
                "decoded_state_mse": decoded_error_by_traj.mean().item(),
                "decoded_state_mse_std": decoded_error_by_traj.std(unbiased=False).item(),
                "decoded_position_mse": decoded_position_error_by_traj.mean().item(),
                "decoded_position_mse_std": decoded_position_error_by_traj.std(unbiased=False).item(),
                "state_autoencode_mse_at_target": (decoded_true_h[:, k:] - state_target).square().mean().item(),
            },
        )

    eigvals = torch.linalg.eigvals(k_matrix).detach().cpu().numpy()
    k_np = k_matrix.detach().cpu().numpy()
    spectral_radius = np.max(np.abs(eigvals), axis=1)
    determinant = np.linalg.det(k_np)
    trace = np.trace(k_np, axis1=1, axis2=2)
    condition_number = np.linalg.cond(k_np)
    complex_pair = np.any(np.abs(np.imag(eigvals)) > 1e-7, axis=1)
    rollout_error_np = np.asarray(per_trajectory_rollout_errors, dtype=float)
    spectral_rows: list[dict[str, Any]] = []
    for index in range(k_np.shape[0]):
        spectral_rows.append(
            {
                "model_name": model_name,
                "trajectory_index": index,
                "omega": float(omegas[index].item()),
                "eig_0_real": float(np.real(eigvals[index, 0])),
                "eig_0_imag": float(np.imag(eigvals[index, 0])),
                "eig_1_real": float(np.real(eigvals[index, 1])),
                "eig_1_imag": float(np.imag(eigvals[index, 1])),
                "spectral_radius": float(spectral_radius[index]),
                "spectral_radius_error_from_1": float(abs(spectral_radius[index] - 1.0)),
                "determinant": float(determinant[index]),
                "trace": float(trace[index]),
                "condition_number": float(condition_number[index]),
                "has_complex_eigenvalues": bool(complex_pair[index]),
                "decoded_position_mse_k16": float(rollout_error_np[index]),
            },
        )

    spectral_summary = {
        "model_name": model_name,
        "source_checkpoint": str(checkpoint),
        "source_metrics": json.loads(source_metrics.read_text(encoding="utf-8")) if source_metrics.exists() else {},
        "spectral_radius_mean": float(np.mean(spectral_radius)),
        "spectral_radius_std": float(np.std(spectral_radius)),
        "spectral_radius_error_from_1_mean": float(np.mean(np.abs(spectral_radius - 1.0))),
        "determinant_mean": float(np.mean(determinant)),
        "determinant_std": float(np.std(determinant)),
        "trace_mean": float(np.mean(trace)),
        "trace_std": float(np.std(trace)),
        "condition_number_mean": float(np.mean(condition_number)),
        "condition_number_std": float(np.std(condition_number)),
        "fraction_complex_eigenvalues": float(np.mean(complex_pair)),
        "corr_spectral_radius_error_with_k16_rollout": _corr(np.abs(spectral_radius - 1.0), rollout_error_np),
        "corr_condition_number_with_k16_rollout": _corr(condition_number, rollout_error_np),
    }
    return step_rows, spectral_rows, spectral_summary


def _plot_k_errors(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for model_name in sorted({row["model_name"] for row in rows}):
        model_rows = sorted([row for row in rows if row["model_name"] == model_name], key=lambda row: row["k"])
        plt.plot([row["k"] for row in model_rows], [row[metric] for row in model_rows], marker="o", label=model_name)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(K_VALUES, [str(k) for k in K_VALUES])
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _plot_spectral_scatter(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for model_name in sorted({row["model_name"] for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name]
        plt.scatter(
            [row["spectral_radius_error_from_1"] for row in model_rows],
            [row["decoded_position_mse_k16"] for row in model_rows],
            s=28,
            alpha=0.75,
            label=model_name,
        )
    plt.yscale("log")
    plt.xlabel("|spectral radius - 1|")
    plt.ylabel("decoded position MSE at k=16")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "spectral_radius_error_vs_k16_rollout.png")
    plt.close()


def _format(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def _write_html(step_rows: list[dict[str, Any]], spectral_summaries: list[dict[str, Any]], interpretation: list[str]) -> None:
    step_columns = ["model_name", "k", "latent_transition_mse", "decoded_state_mse", "decoded_position_mse", "state_autoencode_mse_at_target"]
    spectral_columns = [
        "model_name",
        "spectral_radius_mean",
        "spectral_radius_error_from_1_mean",
        "determinant_mean",
        "trace_mean",
        "condition_number_mean",
        "fraction_complex_eigenvalues",
        "corr_spectral_radius_error_with_k16_rollout",
    ]
    step_header = "".join(f"<th>{escape(column)}</th>" for column in step_columns)
    step_body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in step_columns) + "</tr>"
        for row in step_rows
    )
    spectral_header = "".join(f"<th>{escape(column)}</th>" for column in spectral_columns)
    spectral_body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format(row.get(column, '')))}</td>" for column in spectral_columns) + "</tr>"
        for row in spectral_summaries
    )
    interpretation_html = "".join(f"<p>{escape(line)}</p>" for line in interpretation)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Koopman Evolution Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f4f7fb; }}
    img {{ max-width: 900px; width: 100%; height: auto; border: 1px solid #d9e2ec; margin-bottom: 28px; }}
  </style>
</head>
<body>
  <h1>Koopman Evolution Diagnostics</h1>
  <p>Checkpoint-only analysis. No retraining. k-step errors are averaged over trajectories and all valid time indices.</p>
  <h2>Interpretation</h2>{interpretation_html}
  <h2>k-step Errors</h2><table><thead><tr>{step_header}</tr></thead><tbody>{step_body}</tbody></table>
  <h2>Spectral Summary</h2><table><thead><tr>{spectral_header}</tr></thead><tbody>{spectral_body}</tbody></table>
  <h2>Latent k-step Error</h2><img src="latent_transition_mse_vs_k.png">
  <h2>Decoded State k-step Error</h2><img src="decoded_state_mse_vs_k.png">
  <h2>Decoded Position k-step Error</h2><img src="decoded_position_mse_vs_k.png">
  <h2>Spectral Radius Error vs Rollout</h2><img src="spectral_radius_error_vs_k16_rollout.png">
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _interpret(step_rows: list[dict[str, Any]], spectral_summaries: list[dict[str, Any]]) -> list[str]:
    lines = []
    for model_name in sorted({row["model_name"] for row in step_rows}):
        rows = sorted([row for row in step_rows if row["model_name"] == model_name], key=lambda row: row["k"])
        latent_growth = rows[-1]["latent_transition_mse"] / rows[0]["latent_transition_mse"]
        decoded_growth = rows[-1]["decoded_position_mse"] / rows[0]["decoded_position_mse"]
        lines.append(
            f"{model_name}: latent k-step error grows {latent_growth:.2f}x from k=1 to k=32; decoded position error grows {decoded_growth:.2f}x."
        )
    for summary in spectral_summaries:
        lines.append(
            f"{summary['model_name']}: mean spectral radius {summary['spectral_radius_mean']:.4f}, mean |rho-1| {summary['spectral_radius_error_from_1_mean']:.4e}, corr(|rho-1|, k16 rollout) {summary['corr_spectral_radius_error_with_k16_rollout']:.4f}."
        )
    if any(
        row["k"] == 32 and row["latent_transition_mse"] > 2.0 * next(r for r in step_rows if r["model_name"] == row["model_name"] and r["k"] == 1)["latent_transition_mse"]
        for row in step_rows
    ):
        lines.append("Latent k-step errors grow materially with k, so the multi-step latent consistency diagnostic is warranted.")
    return lines


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sequences, omegas, _ = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    all_step_rows: list[dict[str, Any]] = []
    all_spectral_rows: list[dict[str, Any]] = []
    spectral_summaries: list[dict[str, Any]] = []
    for config in CHECKPOINTS:
        step_rows, spectral_rows, spectral_summary = _evaluate_model(
            str(config["model_name"]),
            Path(config["checkpoint"]),
            Path(config["metrics"]),
            sequences,
            omegas,
        )
        all_step_rows.extend(step_rows)
        all_spectral_rows.extend(spectral_rows)
        spectral_summaries.append(spectral_summary)
    interpretation = _interpret(all_step_rows, spectral_summaries)
    _write_csv(OUTPUT_DIR / "k_step_errors.csv", all_step_rows)
    _write_csv(OUTPUT_DIR / "spectral_diagnostics.csv", all_spectral_rows)
    _write_json(
        OUTPUT_DIR / "summary.json",
        {
            "k_step_errors": all_step_rows,
            "spectral_summaries": spectral_summaries,
            "interpretation": interpretation,
        },
    )
    _plot_k_errors(all_step_rows, "latent_transition_mse", "latent_transition_mse_vs_k.png", "latent transition MSE")
    _plot_k_errors(all_step_rows, "decoded_state_mse", "decoded_state_mse_vs_k.png", "decoded state MSE")
    _plot_k_errors(all_step_rows, "decoded_position_mse", "decoded_position_mse_vs_k.png", "decoded position MSE")
    _plot_spectral_scatter(all_spectral_rows)
    _write_html(all_step_rows, spectral_summaries, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "interpretation": interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
