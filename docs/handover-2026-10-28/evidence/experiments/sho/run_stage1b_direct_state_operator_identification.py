"""Direct observed-state operator identification diagnostic for Stage 1B SHO."""

from __future__ import annotations

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
import numpy as np
import torch

from experiments.sho.run_stage1b_attention_large_window_sweep import (
    DIAGNOSTIC_NUM_TEST,
    TOTAL_SEQUENCE_STEPS,
    _fixed_dynamics_windows,
    _positions_to_states,
)
from experiments.sho.run_stage1b_direct_state_koopman_ablation import (
    DirectStateKoopman,
    NUM_DYNAMICS_WINDOWS,
    _safe_corr,
)
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import (
    CHECKPOINT,
    _angle_error,
    _eig_angle,
    _make_long_position_data,
    _matrix_powers_rollout,
    _per_trajectory_rollout_mse,
    _true_k,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/direct_state_operator_identification")
TEST_DATA_SEED = 1000


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


def _load_neural_direct_state_model() -> DirectStateKoopman:
    model = DirectStateKoopman()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()
    return model


def _learned_neural_k(model: DirectStateKoopman, sequences: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        windows = _fixed_dynamics_windows(sequences[:, :TOTAL_SEQUENCE_STEPS], NUM_DYNAMICS_WINDOWS)
        d = model.encode_dynamics(windows)
        return model.koopman(d)


def _least_squares_k_from_windows(sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one dense K per trajectory from transitions inside fixed dynamics windows."""
    windows = _fixed_dynamics_windows(sequences[:, :TOTAL_SEQUENCE_STEPS], NUM_DYNAMICS_WINDOWS)
    matrices = []
    condition_numbers = []
    for trajectory_windows in windows:
        inputs = torch.stack(
            (trajectory_windows[:, :-2].reshape(-1), trajectory_windows[:, 1:-1].reshape(-1)),
            dim=-1,
        )
        targets = torch.stack(
            (trajectory_windows[:, 1:-1].reshape(-1), trajectory_windows[:, 2:].reshape(-1)),
            dim=-1,
        )
        solution = torch.linalg.lstsq(inputs, targets).solution
        matrices.append(solution.T)
        condition_numbers.append(torch.linalg.cond(inputs))
    return torch.stack(matrices), torch.stack(condition_numbers)


def _position_prediction_metrics(k_matrix: torch.Tensor, positions: torch.Tensor) -> dict[str, Any]:
    true_states = _positions_to_states(positions)
    predicted = _matrix_powers_rollout(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return {
        "one_step_mse": float(torch.mean((predicted[:, 1, 1] - true_states[:, 1, 1]) ** 2).item()),
        "two_step_mse": float(torch.mean((predicted[:, 2, 1] - true_states[:, 2, 1]) ** 2).item()),
        "ten_period_rollout_mse": float(torch.mean((predicted[:, 1:, 1] - true_states[:, 1:, 1]) ** 2).item()),
        "ten_period_rollout_mse_std": float(_per_trajectory_rollout_mse(k_matrix, positions).std(unbiased=False).item()),
    }


def _matrix_error_arrays(k_matrix: torch.Tensor, k_true: torch.Tensor) -> dict[str, np.ndarray]:
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    determinant = np.linalg.det(k_np)
    true_determinant = np.linalg.det(true_np)
    eigvals = np.linalg.eigvals(k_np)
    true_eigvals = np.linalg.eigvals(true_np)
    spectral_radius = np.abs(eigvals).max(axis=1)
    true_spectral_radius = np.abs(true_eigvals).max(axis=1)
    return {
        "k_frobenius_error": np.linalg.norm(k_np - true_np, axis=(1, 2)),
        "trace_abs_error": np.abs(trace - true_trace),
        "c_abs_error": np.abs(trace - true_trace),
        "determinant_abs_error": np.abs(determinant - true_determinant),
        "eigen_angle_abs_error": _angle_error(_eig_angle(k_np), _eig_angle(true_np)),
        "spectral_radius_abs_error": np.abs(spectral_radius - true_spectral_radius),
        "k00_abs_error": np.abs(k_np[:, 0, 0] - true_np[:, 0, 0]),
        "k01_abs_error": np.abs(k_np[:, 0, 1] - true_np[:, 0, 1]),
        "k10_abs_error": np.abs(k_np[:, 1, 0] - true_np[:, 1, 0]),
        "k11_abs_error": np.abs(k_np[:, 1, 1] - true_np[:, 1, 1]),
        "companion_k00_abs": np.abs(k_np[:, 0, 0]),
        "companion_k01_minus_1_abs": np.abs(k_np[:, 0, 1] - 1.0),
        "companion_k10_plus_1_abs": np.abs(k_np[:, 1, 0] + 1.0),
        "companion_trace_minus_k11_abs": np.abs(trace - k_np[:, 1, 1]),
    }


def _summarize_variant(
    name: str,
    k_matrix: torch.Tensor,
    k_true: torch.Tensor,
    positions: torch.Tensor,
    *,
    condition_numbers: torch.Tensor | None = None,
) -> dict[str, Any]:
    rollout_per_trajectory = _per_trajectory_rollout_mse(k_matrix, positions).detach().cpu().numpy()
    errors = _matrix_error_arrays(k_matrix, k_true)
    row: dict[str, Any] = {
        "variant": name,
        **_position_prediction_metrics(k_matrix, positions),
    }
    for key, values in errors.items():
        row[f"{key}_mean"] = float(np.mean(values))
        row[f"{key}_max"] = float(np.max(values))
        row[f"{key}_corr_with_rollout"] = _safe_corr(values, rollout_per_trajectory)
    if condition_numbers is not None:
        row["least_squares_design_condition_number_mean"] = float(condition_numbers.mean().item())
        row["least_squares_design_condition_number_max"] = float(condition_numbers.max().item())
    return row


def _per_trajectory_rows(
    k_by_variant: dict[str, torch.Tensor],
    k_true: torch.Tensor,
    positions: torch.Tensor,
    omegas: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for variant, k_matrix in k_by_variant.items():
        k_np = k_matrix.detach().cpu().numpy()
        true_np = k_true.detach().cpu().numpy()
        errors = _matrix_error_arrays(k_matrix, k_true)
        rollout = _per_trajectory_rollout_mse(k_matrix, positions).detach().cpu().numpy()
        trace = np.trace(k_np, axis1=1, axis2=2)
        true_trace = np.trace(true_np, axis1=1, axis2=2)
        determinant = np.linalg.det(k_np)
        true_determinant = np.linalg.det(true_np)
        eig_angle = _eig_angle(k_np)
        true_eig_angle = _eig_angle(true_np)
        radius = np.abs(np.linalg.eigvals(k_np)).max(axis=1)
        true_radius = np.abs(np.linalg.eigvals(true_np)).max(axis=1)
        for index in range(k_matrix.shape[0]):
            rows.append(
                {
                    "variant": variant,
                    "trajectory": index,
                    "omega": float(omegas[index].item()),
                    "rollout_mse": float(rollout[index]),
                    "k_frobenius_error": float(errors["k_frobenius_error"][index]),
                    "trace_learned": float(trace[index]),
                    "trace_true": float(true_trace[index]),
                    "trace_error": float(trace[index] - true_trace[index]),
                    "determinant_learned": float(determinant[index]),
                    "determinant_true": float(true_determinant[index]),
                    "determinant_error": float(determinant[index] - true_determinant[index]),
                    "spectral_radius_learned": float(radius[index]),
                    "spectral_radius_true": float(true_radius[index]),
                    "spectral_radius_error": float(radius[index] - true_radius[index]),
                    "eigen_angle_learned": float(eig_angle[index]),
                    "eigen_angle_true": float(true_eig_angle[index]),
                    "eigen_angle_error": float(errors["eigen_angle_abs_error"][index]),
                    "k00": float(k_np[index, 0, 0]),
                    "k01": float(k_np[index, 0, 1]),
                    "k10": float(k_np[index, 1, 0]),
                    "k11": float(k_np[index, 1, 1]),
                    "k00_true": float(true_np[index, 0, 0]),
                    "k01_true": float(true_np[index, 0, 1]),
                    "k10_true": float(true_np[index, 1, 0]),
                    "k11_true": float(true_np[index, 1, 1]),
                }
            )
    return rows


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


def _plot_bar(path: Path, rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, *, log: bool = True) -> None:
    names = [str(row["variant"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(names, values)
    if log:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _write_html(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return escape(str(value))

    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{fmt(row[column])}</td>"
            for column in [
                "variant",
                "one_step_mse",
                "two_step_mse",
                "ten_period_rollout_mse",
                "k_frobenius_error_mean",
                "trace_abs_error_mean",
                "determinant_abs_error_mean",
                "eigen_angle_abs_error_mean",
                "spectral_radius_abs_error_mean",
            ]
        )
        + "</tr>"
        for row in rows
    )
    companion = summary["least_squares_companion_form"]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Direct-State Operator Identification</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #18212f; line-height: 1.45; }}
    h1, h2 {{ margin-bottom: 0.35rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 8px 10px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>Direct-State Operator Identification Diagnostic</h1>
  <p>This diagnostic fits one dense <code>2x2</code> operator per trajectory directly from position-only local transitions <code>s_t=(x_{{t-1}},x_t)</code>. It uses no velocity, omega labels, coefficient labels, analytic decoder, or hard-coded SHO recurrence during fitting.</p>

  <h2>Summary</h2>
  <table>
    <thead><tr><th>variant</th><th>one-step MSE</th><th>two-step MSE</th><th>10-period rollout MSE</th><th>Frob K err</th><th>trace/c err</th><th>det err</th><th>angle err</th><th>radius err</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>

  <h2>Least-Squares Companion Form</h2>
  <p>The least-squares dense matrix is naturally close to the companion form <code>[[0,1],[-1,c_i]]</code>:</p>
  <ul>
    <li>mean |K00|: {fmt(companion["mean_abs_k00"])}</li>
    <li>mean |K01 - 1|: {fmt(companion["mean_abs_k01_minus_1"])}</li>
    <li>mean |K10 + 1|: {fmt(companion["mean_abs_k10_plus_1"])}</li>
    <li>mean |trace(K) - K11|: {fmt(companion["mean_abs_trace_minus_k11"])}</li>
  </ul>

  <h2>Interpretation</h2>
  <p>{escape(summary["interpretation"])}</p>

  <h2>Plots</h2>
  <div class="grid">
    <img src="rollout_mse_by_variant.png" alt="rollout MSE by variant">
    <img src="k_frobenius_error_by_variant.png" alt="K Frobenius error by variant">
    <img src="least_squares_trace_vs_true.png" alt="least-squares trace vs true trace">
    <img src="least_squares_determinant_vs_true.png" alt="least-squares determinant vs true determinant">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    positions, omegas, _ = _make_long_position_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    model = _load_neural_direct_state_model()
    k_neural = _learned_neural_k(model, positions)
    k_ls, condition_numbers = _least_squares_k_from_windows(positions)
    k_true = _true_k(omegas)

    variants = {
        "neural_direct_state_K": k_neural,
        "least_squares_identified_K": k_ls,
        "true_oracle_K": k_true,
    }
    rows = [
        _summarize_variant("neural_direct_state_K", k_neural, k_true, positions),
        _summarize_variant(
            "least_squares_identified_K",
            k_ls,
            k_true,
            positions,
            condition_numbers=condition_numbers,
        ),
        _summarize_variant("true_oracle_K", k_true, k_true, positions),
    ]
    per_trajectory = _per_trajectory_rows(variants, k_true, positions, omegas)

    k_ls_np = k_ls.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    ls_trace = np.trace(k_ls_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    ls_det = np.linalg.det(k_ls_np)
    true_det = np.linalg.det(true_np)

    companion_summary = {
        "mean_abs_k00": float(np.mean(np.abs(k_ls_np[:, 0, 0]))),
        "mean_abs_k01_minus_1": float(np.mean(np.abs(k_ls_np[:, 0, 1] - 1.0))),
        "mean_abs_k10_plus_1": float(np.mean(np.abs(k_ls_np[:, 1, 0] + 1.0))),
        "mean_abs_trace_minus_k11": float(np.mean(np.abs(ls_trace - k_ls_np[:, 1, 1]))),
        "max_abs_k00": float(np.max(np.abs(k_ls_np[:, 0, 0]))),
        "max_abs_k01_minus_1": float(np.max(np.abs(k_ls_np[:, 0, 1] - 1.0))),
        "max_abs_k10_plus_1": float(np.max(np.abs(k_ls_np[:, 1, 0] + 1.0))),
    }

    ls_row = next(row for row in rows if row["variant"] == "least_squares_identified_K")
    neural_row = next(row for row in rows if row["variant"] == "neural_direct_state_K")
    if ls_row["ten_period_rollout_mse"] < neural_row["ten_period_rollout_mse"]:
        interpretation = (
            "Direct least-squares identification substantially reduces the long-rollout error relative to the neural "
            "d -> K generator. This indicates that the remaining Stage 1B error is not caused by position-only state "
            "being insufficient; it comes from the neural map failing to emit a power-accurate operator."
        )
    else:
        interpretation = (
            "Direct least-squares identification did not improve the long-rollout error, so the bottleneck is not only "
            "the neural d -> K generator."
        )

    summary = {
        "checkpoint": str(CHECKPOINT),
        "output_dir": str(OUTPUT_DIR),
        "num_test_trajectories": DIAGNOSTIC_NUM_TEST,
        "dt": float(DELTA_T),
        "state": "s_t = (x_{t-1}, x_t)",
        "least_squares_fit": "dense 2x2 K_i = argmin_K sum ||K s_t - s_{t+1}||^2 using transitions inside the fixed 32 dynamics windows",
        "uses_velocity": False,
        "uses_omega_labels_for_fit": False,
        "uses_coefficient_labels_for_fit": False,
        "uses_analytic_decoder": False,
        "hard_codes_sho_recurrence_for_fit": False,
        "rows": rows,
        "least_squares_companion_form": companion_summary,
        "interpretation": interpretation,
    }

    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "per_trajectory.csv", per_trajectory)
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_html(summary, rows)

    _plot_bar(
        OUTPUT_DIR / "rollout_mse_by_variant.png",
        rows,
        "ten_period_rollout_mse",
        "10-period rollout MSE",
        "10-period rollout MSE by operator source",
    )
    _plot_bar(
        OUTPUT_DIR / "k_frobenius_error_by_variant.png",
        rows,
        "k_frobenius_error_mean",
        "mean ||K - K_true||_F",
        "K error by operator source",
    )
    _plot_identity(
        OUTPUT_DIR / "least_squares_trace_vs_true.png",
        true_trace,
        ls_trace,
        "true trace",
        "least-squares trace",
        "least-squares trace vs true trace",
    )
    _plot_identity(
        OUTPUT_DIR / "least_squares_determinant_vs_true.png",
        true_det,
        ls_det,
        "true determinant",
        "least-squares determinant",
        "least-squares determinant vs true determinant",
    )


if __name__ == "__main__":
    main()
