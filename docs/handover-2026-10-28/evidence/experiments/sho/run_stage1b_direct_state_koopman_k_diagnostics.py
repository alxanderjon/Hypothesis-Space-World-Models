"""K-error and long-power diagnostics for direct-state Koopman Stage 1B."""

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
from sklearn.linear_model import LinearRegression
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    BATCH_SIZE,
    DIAGNOSTIC_NUM_TEST,
    DIAGNOSTIC_NUM_TRAIN,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    POSITION_WINDOW_SIZE,
    ROLLOUT_STEPS,
    SEED,
    TEST_DATA_SEED,
    TOTAL_SEQUENCE_STEPS,
    TRAIN_DATA_SEED,
    _fixed_dynamics_windows,
    _make_position_data,
    _parameter_count,
    _positions_to_states,
    _sample_dynamics_windows,
    _sample_windows,
)
from experiments.sho.run_stage1b_direct_state_koopman_ablation import (
    DirectStateKoopman,
    NUM_DYNAMICS_WINDOWS,
    _linear_regression_prediction_stats,
    _multioutput_r2,
    _safe_corr,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from hypothesis_space_world_models.training import mse_loss, set_seed


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/direct_state_koopman_k_diagnostics")
CHECKPOINT = Path("outputs/sho/stage1B/direct_state_koopman_ablation/direct_state_attention_m32/learned_parameters.pt")
BASELINE_METRICS = Path("outputs/sho/stage1B/direct_state_koopman_ablation/direct_state_attention_m32/metrics.json")
EVAL_EVERY = 1_000
LONG_K_VALUES = [1, 2, 4, 8, 16, 32]
LAMBDA_LONG_VALUES = [1e-3, 1e-2, 1e-1]
TEN_PERIODS = 10.0


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


def _load_direct_model() -> DirectStateKoopman:
    model = DirectStateKoopman()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def _make_long_position_data(num_trajectories: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed)
    omega_min = float(OMEGA_RANGE[0])
    steps = int(math.ceil(TEN_PERIODS * math.tau / (omega_min * float(DELTA_T)))) + 2
    times = torch.arange(steps, dtype=initial_conditions.dtype) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    labels = torch.stack((initial_conditions[:, 0], initial_conditions[:, 1] / omegas, omegas), dim=-1)
    return positions, omegas, labels


def _sample_training_batch(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_trajectories = sequences.shape[0]
    indices = torch.randperm(num_trajectories)[:BATCH_SIZE]
    max_window_start = TOTAL_SEQUENCE_STEPS - POSITION_WINDOW_SIZE
    max_rollout_start = TOTAL_SEQUENCE_STEPS - ROLLOUT_STEPS - 2
    starts = torch.randint(0, max_window_start + 1, (indices.shape[0], NUM_DYNAMICS_WINDOWS))
    rollout_starts = torch.randint(0, max_rollout_start + 1, (indices.shape[0],))
    return (
        _sample_dynamics_windows(sequences, indices, starts),
        _sample_windows(sequences, indices, rollout_starts, ROLLOUT_STEPS + 2),
    )


def _true_k(omegas: torch.Tensor) -> torch.Tensor:
    c = 2.0 * torch.cos(omegas * float(DELTA_T))
    k = torch.zeros((omegas.shape[0], 2, 2), dtype=omegas.dtype, device=omegas.device)
    k[:, 0, 1] = 1.0
    k[:, 1, 0] = -1.0
    k[:, 1, 1] = c
    return k


def _matrix_powers_rollout(k_matrix: torch.Tensor, s0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [s0]
    state = s0
    for _ in range(steps):
        state = torch.bmm(k_matrix, state.unsqueeze(-1)).squeeze(-1)
        states.append(state)
    return torch.stack(states, dim=1)


def _per_trajectory_rollout_mse(k_matrix: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    true_states = _positions_to_states(positions)
    pred = _matrix_powers_rollout(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return torch.mean((pred[:, 1:, 1] - true_states[:, 1:, 1]) ** 2, dim=1)


def _eig_angle(k_matrix: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvals(k_matrix)
    selected = []
    for pair in eigvals:
        if np.any(np.imag(pair) > 0):
            value = pair[np.argmax(np.imag(pair))]
        else:
            value = pair[np.argmax(np.abs(pair))]
        selected.append(abs(np.angle(value)))
    return np.asarray(selected)


def _angle_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, 2.0 * np.pi - diff)


def _project_trace(k_matrix: torch.Tensor, true_trace: torch.Tensor) -> torch.Tensor:
    current_trace = k_matrix[:, 0, 0] + k_matrix[:, 1, 1]
    delta = (true_trace - current_trace) / 2.0
    projected = k_matrix.clone()
    projected[:, 0, 0] += delta
    projected[:, 1, 1] += delta
    return projected


def _normalize_determinant(k_matrix: torch.Tensor) -> torch.Tensor:
    det = torch.linalg.det(k_matrix)
    scale = torch.rsqrt(torch.clamp(torch.abs(det), min=1e-12))
    return k_matrix * scale.reshape(-1, 1, 1)


def _oscillator_projection_from_trace(k_matrix: torch.Tensor) -> torch.Tensor:
    c = k_matrix[:, 0, 0] + k_matrix[:, 1, 1]
    projected = torch.zeros_like(k_matrix)
    projected[:, 0, 1] = 1.0
    projected[:, 1, 0] = -1.0
    projected[:, 1, 1] = c
    return projected


def _c_to_angle(c: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(c / 2.0, -1.0, 1.0))


def _plot_scatter(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, s=28, alpha=0.82)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_identity(path: Path, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(x, y, s=28, alpha=0.82)
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    plt.plot([low, high], [low, high], color="black", linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _learned_k_for_sequences(model: DirectStateKoopman, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        windows = _fixed_dynamics_windows(sequences[:, :TOTAL_SEQUENCE_STEPS], NUM_DYNAMICS_WINDOWS)
        d = model.encode_dynamics(windows)
        k_matrix = model.koopman(d)
    return d, k_matrix


def _part1_part2_diagnostics() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_direct_model()
    sequences, omegas, labels = _make_long_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    d, k_learned = _learned_k_for_sequences(model, sequences)
    k_true = _true_k(omegas)
    true_trace_t = k_true[:, 0, 0] + k_true[:, 1, 1]

    learned_rollout = _per_trajectory_rollout_mse(k_learned, sequences)
    true_rollout = _per_trajectory_rollout_mse(k_true, sequences)
    k_trace_corrected = _project_trace(k_learned, true_trace_t)
    k_det_normalized = _normalize_determinant(k_learned)
    k_osc_trace = _oscillator_projection_from_trace(k_learned)
    k_osc_oracle = k_true

    variants = {
        "learned_K": k_learned,
        "true_K_oracle": k_true,
        "trace_corrected_oracle_trace": k_trace_corrected,
        "determinant_normalized": k_det_normalized,
        "oscillator_projected_learned_trace": k_osc_trace,
        "oscillator_projected_oracle_c": k_osc_oracle,
    }
    projection_rows = []
    for name, matrix in variants.items():
        per_traj = _per_trajectory_rollout_mse(matrix, sequences)
        projection_rows.append(
            {
                "variant": name,
                "ten_period_rollout_mse": float(per_traj.mean().item()),
                "ten_period_rollout_mse_std": float(per_traj.std(unbiased=False).item()),
                "oracle": "oracle" in name,
            }
        )

    k_np = k_learned.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    rollout_np = learned_rollout.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    det = np.linalg.det(k_np)
    true_det = np.linalg.det(true_np)
    eig_angle = _eig_angle(k_np)
    true_angle = _eig_angle(true_np)
    eig_angle_err = _angle_error(eig_angle, true_angle)
    eigvals = np.linalg.eigvals(k_np)
    radius = np.abs(eigvals).max(axis=1)
    true_radius = np.abs(np.linalg.eigvals(true_np)).max(axis=1)
    frob = np.linalg.norm(k_np - true_np, axis=(1, 2))
    entry_errors = k_np - true_np
    c_learned = trace
    c_true = true_trace
    c_error = c_learned - c_true

    per_traj_rows = []
    for index in range(k_np.shape[0]):
        per_traj_rows.append(
            {
                "trajectory": index,
                "omega": float(omegas[index].item()),
                "rollout_mse": float(rollout_np[index]),
                "k_frobenius_error": float(frob[index]),
                "k00_error": float(entry_errors[index, 0, 0]),
                "k01_error": float(entry_errors[index, 0, 1]),
                "k10_error": float(entry_errors[index, 1, 0]),
                "k11_error": float(entry_errors[index, 1, 1]),
                "trace_learned": float(trace[index]),
                "trace_true": float(true_trace[index]),
                "trace_error": float(trace[index] - true_trace[index]),
                "determinant_learned": float(det[index]),
                "determinant_true": float(true_det[index]),
                "determinant_error": float(det[index] - true_det[index]),
                "spectral_radius_learned": float(radius[index]),
                "spectral_radius_true": float(true_radius[index]),
                "spectral_radius_error": float(radius[index] - true_radius[index]),
                "eigen_angle_learned": float(eig_angle[index]),
                "eigen_angle_true": float(true_angle[index]),
                "eigen_angle_error": float(eig_angle_err[index]),
                "c_learned": float(c_learned[index]),
                "c_true": float(c_true[index]),
                "c_error": float(c_error[index]),
            }
        )

    error_arrays = {
        "k_frobenius_error": frob,
        "k00_abs_error": np.abs(entry_errors[:, 0, 0]),
        "k01_abs_error": np.abs(entry_errors[:, 0, 1]),
        "k10_abs_error": np.abs(entry_errors[:, 1, 0]),
        "k11_abs_error": np.abs(entry_errors[:, 1, 1]),
        "trace_abs_error": np.abs(trace - true_trace),
        "determinant_abs_error": np.abs(det - true_det),
        "spectral_radius_abs_error": np.abs(radius - true_radius),
        "eigen_angle_abs_error": eig_angle_err,
        "c_abs_error": np.abs(c_error),
    }
    correlation_rows = [
        {"error_metric": name, "correlation_with_rollout_mse": _safe_corr(values, rollout_np)}
        for name, values in error_arrays.items()
    ]

    summary = {
        "checkpoint": str(CHECKPOINT),
        "dt": float(DELTA_T),
        "num_test_trajectories": DIAGNOSTIC_NUM_TEST,
        "ten_period_sequence_steps": int(sequences.shape[1]),
        "learned_ten_period_rollout_mse": float(learned_rollout.mean().item()),
        "true_oracle_ten_period_rollout_mse": float(true_rollout.mean().item()),
        "k_frobenius_error_mean": float(np.mean(frob)),
        "k_frobenius_error_std": float(np.std(frob, ddof=0)),
        "trace_abs_error_mean": float(np.mean(np.abs(trace - true_trace))),
        "determinant_abs_error_mean": float(np.mean(np.abs(det - true_det))),
        "spectral_radius_abs_error_mean": float(np.mean(np.abs(radius - true_radius))),
        "eigen_angle_abs_error_mean": float(np.mean(eig_angle_err)),
        "c_abs_error_mean": float(np.mean(np.abs(c_error))),
        "omega_r2_from_d": _linear_regression_prediction_stats(d, omegas)[0],
        "omega_correlation_from_d": _linear_regression_prediction_stats(d, omegas)[1],
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "correlations": correlation_rows,
        "projection_rollouts": projection_rows,
    }

    _write_csv(OUTPUT_DIR / "k_per_trajectory.csv", per_traj_rows)
    _write_csv(OUTPUT_DIR / "k_error_correlations.csv", correlation_rows)
    _write_csv(OUTPUT_DIR / "projection_rollouts.csv", projection_rows)
    _write_json(OUTPUT_DIR / "k_diagnostics_summary.json", summary)

    _plot_identity(OUTPUT_DIR / "learned_vs_true_trace.png", true_trace, trace, "true trace", "learned trace", "learned vs true trace")
    _plot_identity(OUTPUT_DIR / "learned_vs_true_determinant.png", true_det, det, "true determinant", "learned determinant", "learned vs true determinant")
    _plot_identity(OUTPUT_DIR / "learned_vs_true_eigen_angle.png", true_angle, eig_angle, "true eigen-angle", "learned eigen-angle", "learned vs true eigen-angle")
    _plot_identity(OUTPUT_DIR / "learned_c_vs_true_c.png", c_true, c_learned, "true c", "learned c = trace(K)", "learned c vs true c")
    _plot_scatter(OUTPUT_DIR / "rollout_mse_vs_frobenius_k_error.png", frob, rollout_np, "K Frobenius error", "10-period rollout MSE", "rollout vs K error")
    _plot_scatter(OUTPUT_DIR / "rollout_mse_vs_eigen_angle_error.png", eig_angle_err, rollout_np, "eigen-angle error", "10-period rollout MSE", "rollout vs phase error")
    _plot_scatter(OUTPUT_DIR / "rollout_mse_vs_determinant_error.png", np.abs(det - true_det), rollout_np, "determinant abs error", "10-period rollout MSE", "rollout vs determinant error")

    return summary


def _losses(
    model: DirectStateKoopman,
    dynamics_windows: torch.Tensor,
    rollout_positions: torch.Tensor,
    *,
    lambda_long: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    d = model.encode_dynamics(dynamics_windows)
    k_matrix = model.koopman(d)
    true_states = _positions_to_states(rollout_positions)
    predicted_states = _matrix_powers_rollout(k_matrix, true_states[:, 0], ROLLOUT_STEPS)
    one_step_state_mse = mse_loss(predicted_states[:, 1], true_states[:, 1])
    two_step_state_mse = mse_loss(predicted_states[:, 2], true_states[:, 2])
    rollout_state_mse = mse_loss(predicted_states[:, 1:], true_states[:, 1:])
    long_terms = []
    for k_value in LONG_K_VALUES:
        if k_value <= ROLLOUT_STEPS:
            long_terms.append(mse_loss(predicted_states[:, k_value], true_states[:, k_value]))
    long_mse = torch.stack(long_terms).mean()
    total = one_step_state_mse + two_step_state_mse + rollout_state_mse + lambda_long * long_mse
    return total, {
        "one_step_mse": mse_loss(predicted_states[:, 1, 1], true_states[:, 1, 1]),
        "two_step_mse": mse_loss(predicted_states[:, 2, 1], true_states[:, 2, 1]),
        "rollout_mse": mse_loss(predicted_states[:, 1:, 1], true_states[:, 1:, 1]),
        "one_step_state_mse": one_step_state_mse,
        "two_step_state_mse": two_step_state_mse,
        "rollout_state_mse": rollout_state_mse,
        "long_power_state_mse": long_mse,
        "d_variance": d.var(dim=0, unbiased=False).mean(),
    }, d, k_matrix


def _evaluate_training_run(
    model: DirectStateKoopman,
    sequences: torch.Tensor,
    omegas: torch.Tensor,
    labels: torch.Tensor,
    *,
    lambda_long: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        windows = _fixed_dynamics_windows(sequences, NUM_DYNAMICS_WINDOWS)
        rollout_positions = sequences[:, : ROLLOUT_STEPS + 2]
        _, parts, d, k_matrix = _losses(model, windows, rollout_positions, lambda_long=lambda_long)
        k_true = _true_k(omegas)
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    eig_angle_err = _angle_error(_eig_angle(k_np), _eig_angle(true_np))
    trace_error = np.abs(np.trace(k_np, axis1=1, axis2=2) - np.trace(true_np, axis1=1, axis2=2))
    det_error = np.abs(np.linalg.det(k_np) - np.linalg.det(true_np))
    frob = np.linalg.norm(k_np - true_np, axis=(1, 2))
    omega_r2, omega_corr = _linear_regression_prediction_stats(d, omegas)
    return {
        "rollout_mse": parts["rollout_mse"].item(),
        "one_step_mse": parts["one_step_mse"].item(),
        "two_step_mse": parts["two_step_mse"].item(),
        "long_power_state_mse": parts["long_power_state_mse"].item(),
        "k_frobenius_error_mean": float(np.mean(frob)),
        "trace_abs_error_mean": float(np.mean(trace_error)),
        "determinant_abs_error_mean": float(np.mean(det_error)),
        "eigen_angle_abs_error_mean": float(np.mean(eig_angle_err)),
        "omega_r2_from_d": omega_r2,
        "omega_correlation_from_d": omega_corr,
        "abomega_multioutput_r2_from_d": _multioutput_r2(d, labels),
        "d_variance": parts["d_variance"].item(),
    }


def _train_long_power_variant(
    run_name: str,
    lambda_long: float,
    train_sequences: torch.Tensor,
    test_sequences: torch.Tensor,
    test_omegas: torch.Tensor,
    test_labels: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    model = DirectStateKoopman()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
            "lambda_long": lambda_long,
            "long_k_values": LONG_K_VALUES,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "num_dynamics_windows": NUM_DYNAMICS_WINDOWS,
            "omega_supervision": False,
            "coefficient_supervision": False,
            "analytic_decoder": False,
            "spectral_loss": False,
            "structured_koopman": False,
        },
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        dynamics_windows, rollout_positions = _sample_training_batch(train_sequences)
        optimizer.zero_grad()
        loss, train_parts, _, _ = _losses(model, dynamics_windows, rollout_positions, lambda_long=lambda_long)
        if not torch.isfinite(loss):
            stopped_because = "nan_or_inf_loss"
            break
        loss.backward()
        optimizer.step()
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            eval_metrics = _evaluate_training_run(
                model,
                test_sequences,
                test_omegas,
                test_labels,
                lambda_long=lambda_long,
            )
            if eval_metrics["rollout_mse"] < best_rollout:
                best_rollout = eval_metrics["rollout_mse"]
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += EVAL_EVERY
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "train_long_power_state_mse": train_parts["long_power_state_mse"].item(),
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
    metrics = _evaluate_training_run(model, test_sequences, test_omegas, test_labels, lambda_long=lambda_long)
    metrics.update(
        {
            "run_name": run_name,
            "lambda_long": lambda_long,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "stopped_because": stopped_because,
            "parameter_count": _parameter_count(model),
        },
    )
    torch.save(model.state_dict(), run_dir / "learned_parameters.pt")
    _write_json(metrics_path, metrics)
    _write_json(history_path, {"history": history})
    _write_csv(run_dir / "history.csv", history)
    return metrics, history


def _plot_training_summary(rows: list[dict[str, Any]]) -> None:
    metrics = [
        ("rollout_mse", "rollout_mse_vs_lambda_long.png", "rollout MSE", True),
        ("k_frobenius_error_mean", "k_frobenius_error_vs_lambda_long.png", "K Frobenius error", True),
        ("eigen_angle_abs_error_mean", "eigen_angle_error_vs_lambda_long.png", "eigen-angle error", True),
        ("omega_r2_from_d", "omega_r2_vs_lambda_long.png", "omega R2 from d", False),
    ]
    x = [row["lambda_long"] for row in rows]
    labels = [str(value) for value in x]
    for metric, filename, ylabel, log_y in metrics:
        plt.figure(figsize=(7.0, 4.6))
        plt.plot(range(len(rows)), [row[metric] for row in rows], marker="o")
        plt.xticks(range(len(rows)), labels)
        if log_y:
            plt.yscale("log")
        plt.xlabel("lambda_long")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _summary_html(k_summary: dict[str, Any], training_rows: list[dict[str, Any]] | None) -> str:
    projection_rows = k_summary["projection_rollouts"]
    proj_header = "".join(f"<th>{name}</th>" for name in projection_rows[0].keys())
    proj_body = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{value:.6g}</td>" if isinstance(value, float) else f"<td>{escape(str(value))}</td>"
            for value in row.values()
        )
        + "</tr>"
        for row in projection_rows
    )
    corr_rows = k_summary["correlations"]
    corr_body = "\n".join(
        f"<tr><td>{escape(row['error_metric'])}</td><td>{row['correlation_with_rollout_mse']:.6g}</td></tr>"
        for row in corr_rows
    )
    train_section = "<p>No long-power training was run.</p>"
    if training_rows:
        train_header = "".join(f"<th>{name}</th>" for name in training_rows[0].keys())
        train_body = "\n".join(
            "<tr>"
            + "".join(
                f"<td>{value:.6g}</td>" if isinstance(value, float) else f"<td>{escape(str(value))}</td>"
                for value in row.values()
            )
            + "</tr>"
            for row in training_rows
        )
        train_section = f"<table><thead><tr>{train_header}</tr></thead><tbody>{train_body}</tbody></table>"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Direct-State Koopman K Diagnostics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.45; color: #1f2933; }}
    h1, h2 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 2px 5px; border-radius: 4px; }}
    img {{ max-width: 760px; width: 100%; display: block; margin: 14px 0 28px; }}
  </style>
</head>
<body>
  <h1>Direct-State Koopman K Diagnostics</h1>
  <p>Loaded the completed direct-state m=32 attention checkpoint and compared the learned dense <code>K_i</code> with the true observed-state recurrence <code>[[0,1],[-1,2 cos(omega_i dt)]]</code>. Oracle variants are diagnostics only.</p>
  <h2>Part 1 Summary</h2>
  <p>Mean K Frobenius error: {k_summary['k_frobenius_error_mean']:.6g}. Mean trace abs error: {k_summary['trace_abs_error_mean']:.6g}. Mean determinant abs error: {k_summary['determinant_abs_error_mean']:.6g}. Mean eigen-angle abs error: {k_summary['eigen_angle_abs_error_mean']:.6g}.</p>
  <h2>Error Correlations With Rollout MSE</h2>
  <table><thead><tr><th>error metric</th><th>correlation</th></tr></thead><tbody>{corr_body}</tbody></table>
  <h2>Projection Rollouts</h2>
  <table><thead><tr>{proj_header}</tr></thead><tbody>{proj_body}</tbody></table>
  <h2>Plots</h2>
  <img src="learned_vs_true_trace.png">
  <img src="learned_vs_true_determinant.png">
  <img src="learned_vs_true_eigen_angle.png">
  <img src="learned_c_vs_true_c.png">
  <img src="rollout_mse_vs_frobenius_k_error.png">
  <img src="rollout_mse_vs_eigen_angle_error.png">
  <img src="rollout_mse_vs_determinant_error.png">
  <h2>Part 3 Long-Power Training</h2>
  {train_section}
</body>
</html>
"""


def main() -> None:
    k_summary = _part1_part2_diagnostics()
    learned = next(row for row in k_summary["projection_rollouts"] if row["variant"] == "learned_K")
    true_oracle = next(row for row in k_summary["projection_rollouts"] if row["variant"] == "true_K_oracle")
    should_train = (
        learned["ten_period_rollout_mse"] > 100.0 * true_oracle["ten_period_rollout_mse"]
        and (
            k_summary["eigen_angle_abs_error_mean"] > 1e-5
            or k_summary["determinant_abs_error_mean"] > 1e-5
            or k_summary["k_frobenius_error_mean"] > 1e-5
        )
    )
    training_rows: list[dict[str, Any]] | None = None
    if should_train:
        train_sequences, _, _ = _make_position_data(DIAGNOSTIC_NUM_TRAIN, seed=TRAIN_DATA_SEED)
        test_sequences, test_omegas, test_labels = _make_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
        baseline_metrics = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
        training_rows = [
            {
                "run_name": "baseline_existing",
                "lambda_long": 0.0,
                "rollout_mse": baseline_metrics["rollout_mse"],
                "one_step_mse": baseline_metrics["one_step_mse"],
                "two_step_mse": baseline_metrics["two_step_mse"],
                "k_frobenius_error_mean": k_summary["k_frobenius_error_mean"],
                "trace_abs_error_mean": k_summary["trace_abs_error_mean"],
                "determinant_abs_error_mean": k_summary["determinant_abs_error_mean"],
                "eigen_angle_abs_error_mean": k_summary["eigen_angle_abs_error_mean"],
                "omega_r2_from_d": baseline_metrics["omega_r2_from_d"],
                "best_epoch": baseline_metrics["best_epoch"],
            }
        ]
        for value in LAMBDA_LONG_VALUES:
            run_name = f"lambda_long_{value:g}".replace(".", "p").replace("-", "m")
            metrics, _ = _train_long_power_variant(
                run_name,
                value,
                train_sequences,
                test_sequences,
                test_omegas,
                test_labels,
            )
            training_rows.append(metrics)
        _write_csv(OUTPUT_DIR / "long_power_training_summary.csv", training_rows)
        _plot_training_summary(training_rows)
    (OUTPUT_DIR / "summary.html").write_text(_summary_html(k_summary, training_rows), encoding="utf-8")
    _write_json(
        OUTPUT_DIR / "summary.json",
        {
            "part1_part2": k_summary,
            "part3_ran": should_train,
            "part3_training": training_rows,
        },
    )


if __name__ == "__main__":
    main()
