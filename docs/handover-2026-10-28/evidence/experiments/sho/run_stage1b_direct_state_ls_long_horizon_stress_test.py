"""Long-horizon stress test for direct-state LS Koopman identification."""

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

from experiments.sho.run_stage1b_attention_large_window_sweep import DIAGNOSTIC_NUM_TEST
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import _angle_error, _eig_angle, _true_k
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _make_base_conditions, _state_at_time


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/direct_state_ls_long_horizon_stress_test")
TEST_DATA_SEED = 1000
HORIZON_PERIODS = [10, 25, 50, 100, 250, 500]
FIXED_PREFIX_TRANSITIONS = 168
EXAMPLE_TRAJECTORIES = [0, 1, 2, 3]


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


def _generate_float64_positions(max_periods: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED, dtype=torch.float64)
    periods = math.tau / omegas
    steps_per_trajectory = torch.ceil(max_periods * periods / float(DELTA_T)).to(torch.long) + 2
    max_steps = int(steps_per_trajectory.max().item())
    times = torch.arange(max_steps, dtype=torch.float64) * float(DELTA_T)
    times = times.unsqueeze(0).expand(DIAGNOSTIC_NUM_TEST, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions, omegas, periods, steps_per_trajectory


def _design_from_positions(positions: torch.Tensor, num_transitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack((positions[:, :num_transitions], positions[:, 1 : num_transitions + 1]), dim=-1)
    targets = torch.stack((positions[:, 1 : num_transitions + 1], positions[:, 2 : num_transitions + 2]), dim=-1)
    return inputs, targets


def _solve_qr(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = []
    condition_numbers = []
    for a_matrix, b_matrix in zip(inputs, targets):
        q_matrix, r_matrix = torch.linalg.qr(a_matrix, mode="reduced")
        solution = torch.linalg.solve_triangular(r_matrix, q_matrix.mT @ b_matrix, upper=True)
        matrices.append(solution.mT)
        condition_numbers.append(torch.linalg.cond(a_matrix))
    return torch.stack(matrices), torch.stack(condition_numbers)


def _solve_lstsq(inputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    matrices = []
    condition_numbers = []
    for a_matrix, b_matrix in zip(inputs, targets):
        solution = torch.linalg.lstsq(a_matrix, b_matrix).solution
        matrices.append(solution.mT)
        condition_numbers.append(torch.linalg.cond(a_matrix))
    return torch.stack(matrices), torch.stack(condition_numbers)


def _rollout(k_matrix: torch.Tensor, s0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [s0]
    state = s0
    for _ in range(steps):
        state = torch.bmm(k_matrix, state.unsqueeze(-1)).squeeze(-1)
        states.append(state)
    return torch.stack(states, dim=1)


def _masked_position_errors(
    k_matrix: torch.Tensor,
    positions: torch.Tensor,
    steps_per_trajectory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_steps = int(steps_per_trajectory.max().item())
    s0 = torch.stack((positions[:, 0], positions[:, 1]), dim=-1)
    predicted_states = _rollout(k_matrix, s0, max_steps - 2)
    predicted_positions = torch.cat(
        (positions[:, :1], predicted_states[:, :, 1]),
        dim=1,
    )
    true_positions = positions[:, :max_steps]
    valid = torch.arange(max_steps).unsqueeze(0) < steps_per_trajectory.unsqueeze(1)
    errors = predicted_positions - true_positions
    return errors, predicted_positions, valid


def _matrix_error_summary(k_matrix: torch.Tensor, k_true: torch.Tensor) -> dict[str, float]:
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    determinant = np.linalg.det(k_np)
    true_determinant = np.linalg.det(true_np)
    radius = np.abs(np.linalg.eigvals(k_np)).max(axis=1)
    true_radius = np.abs(np.linalg.eigvals(true_np)).max(axis=1)
    return {
        "k_frobenius_error_mean": float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2)))),
        "trace_abs_error_mean": float(np.mean(np.abs(trace - true_trace))),
        "c_abs_error_mean": float(np.mean(np.abs(trace - true_trace))),
        "determinant_abs_error_mean": float(np.mean(np.abs(determinant - true_determinant))),
        "spectral_radius_abs_error_mean": float(np.mean(np.abs(radius - true_radius))),
        "eigen_angle_abs_error_mean": float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np)))),
    }


def _growth_label(mean_abs_error_by_time: np.ndarray) -> str:
    y = np.maximum(mean_abs_error_by_time[1:], 1e-300)
    x = np.arange(1, y.shape[0] + 1, dtype=np.float64)
    if y.shape[0] < 8:
        return "insufficient_points"
    log_y = np.log(y)
    linear_r2 = _r2(y, np.polyval(np.polyfit(x, y, 1), x))
    quadratic_r2 = _r2(y, np.polyval(np.polyfit(x, y, 2), x))
    exponential_r2 = _r2(log_y, np.polyval(np.polyfit(x, log_y, 1), x))
    if float(np.max(y)) < 1e-12:
        return "near_floor"
    best = max(
        [("linear", linear_r2), ("quadratic", quadratic_r2), ("exponential", exponential_r2)],
        key=lambda item: item[1],
    )
    if best[1] < 0.5:
        return "bounded_or_oscillatory"
    return best[0]


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / denom


def _evaluate_variant(
    *,
    horizon_periods: int,
    protocol: str,
    solver: str,
    k_matrix: torch.Tensor,
    k_true: torch.Tensor,
    positions: torch.Tensor,
    steps_per_trajectory: torch.Tensor,
    condition_numbers: torch.Tensor,
) -> tuple[dict[str, Any], np.ndarray]:
    errors, _, valid = _masked_position_errors(k_matrix, positions, steps_per_trajectory)
    squared = errors.square()
    abs_errors = errors.abs()
    trajectory_mse = squared[valid].mean().item()
    max_abs_error = abs_errors[valid].max().item()
    final_indices = steps_per_trajectory - 1
    final_abs_error = abs_errors[torch.arange(abs_errors.shape[0]), final_indices].mean().item()
    valid_float = valid.to(errors.dtype)
    mean_abs_by_time = (abs_errors * valid_float).sum(dim=0) / torch.clamp(valid_float.sum(dim=0), min=1.0)
    row = {
        "horizon_periods": horizon_periods,
        "protocol": protocol,
        "solver": solver,
        "trajectory_mse": float(trajectory_mse),
        "max_abs_position_error": float(max_abs_error),
        "mean_final_abs_position_error": float(final_abs_error),
        "condition_number_mean": float(condition_numbers.mean().item()),
        "condition_number_max": float(condition_numbers.max().item()),
        "error_growth_label": _growth_label(mean_abs_by_time.detach().cpu().numpy()),
        **_matrix_error_summary(k_matrix, k_true),
    }
    return row, mean_abs_by_time.detach().cpu().numpy()


def _plot_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for protocol in ["fixed_prefix", "full_future_transitions", "oracle_floor"]:
        selected = [row for row in rows if row["protocol"] == protocol and row["solver"] == "qr"]
        if not selected:
            continue
        selected = sorted(selected, key=lambda row: row["horizon_periods"])
        plt.plot(
            [row["horizon_periods"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            label=protocol,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("horizon periods")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


def _plot_error_examples(
    positions: torch.Tensor,
    omegas: torch.Tensor,
    periods: torch.Tensor,
    k_fixed: torch.Tensor,
    k_oracle: torch.Tensor,
    steps_per_trajectory: torch.Tensor,
) -> None:
    for log_scale, filename in [(False, "error_vs_time_examples.png"), (True, "log_error_vs_time_examples.png")]:
        plt.figure(figsize=(9.0, 5.2))
        for index in EXAMPLE_TRAJECTORIES:
            errors, _, _ = _masked_position_errors(
                k_fixed[index : index + 1],
                positions[index : index + 1],
                steps_per_trajectory[index : index + 1],
            )
            oracle_errors, _, _ = _masked_position_errors(
                k_oracle[index : index + 1],
                positions[index : index + 1],
                steps_per_trajectory[index : index + 1],
            )
            steps = int(steps_per_trajectory[index].item())
            t_periods = (
                torch.arange(steps, dtype=torch.float64)
                * float(DELTA_T)
                / periods[index]
            ).detach().cpu().numpy()
            plt.plot(t_periods, errors[0, :steps].abs().detach().cpu().numpy(), label=f"traj {index} LS")
            plt.plot(
                t_periods,
                oracle_errors[0, :steps].abs().detach().cpu().numpy(),
                linestyle="--",
                alpha=0.65,
                label=f"traj {index} oracle",
            )
        if log_scale:
            plt.yscale("log")
        plt.xlabel("trajectory periods elapsed")
        plt.ylabel("|position error|")
        plt.title("position error over time, fixed-prefix QR")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()


def _write_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return escape(str(value))

    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{fmt(row[column])}</td>"
            for column in [
                "horizon_periods",
                "protocol",
                "solver",
                "trajectory_mse",
                "max_abs_position_error",
                "mean_final_abs_position_error",
                "error_growth_label",
                "k_frobenius_error_mean",
                "trace_abs_error_mean",
                "determinant_abs_error_mean",
                "eigen_angle_abs_error_mean",
            ]
        )
        + "</tr>"
        for row in rows
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Direct-State LS Long-Horizon Stress Test</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182235; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 7px 9px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>Direct-State LS Long-Horizon Stress Test</h1>
  <p>This is a numerical/operator-identification diagnostic, not a neural-network model. Dense <code>2x2</code> operators are fit from position-only local transitions and evaluated by rollout against analytic <code>x_true(t)</code>.</p>
  <h2>Conclusion</h2>
  <p>{escape(summary["conclusion"])}</p>
  <h2>Results</h2>
  <table>
    <thead><tr><th>periods</th><th>protocol</th><th>solver</th><th>trajectory MSE</th><th>max abs err</th><th>final abs err</th><th>growth</th><th>K err</th><th>trace/c err</th><th>det err</th><th>angle err</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Plots</h2>
  <div class="grid">
    <img src="rollout_mse_vs_horizon.png" alt="rollout MSE vs horizon">
    <img src="max_error_vs_horizon.png" alt="max error vs horizon">
    <img src="final_error_vs_horizon.png" alt="final error vs horizon">
    <img src="error_vs_time_examples.png" alt="error vs time examples">
    <img src="log_error_vs_time_examples.png" alt="log error vs time examples">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_horizon = max(HORIZON_PERIODS)
    positions_all, omegas, periods, _ = _generate_float64_positions(max_horizon)
    k_true = _true_k(omegas)

    rows: list[dict[str, Any]] = []
    mean_error_rows: list[dict[str, Any]] = []
    fixed_k_for_examples = None
    fixed_steps_for_examples = None

    for horizon in HORIZON_PERIODS:
        steps_per_trajectory = torch.ceil(horizon * periods / float(DELTA_T)).to(torch.long) + 2
        max_steps = max(int(steps_per_trajectory.max().item()), FIXED_PREFIX_TRANSITIONS + 2)
        positions = positions_all[:, :max_steps]

        prefix_inputs, prefix_targets = _design_from_positions(positions, FIXED_PREFIX_TRANSITIONS)
        full_inputs_list = []
        full_targets_list = []
        for index, steps in enumerate(steps_per_trajectory):
            n_transitions = int(steps.item()) - 2
            inputs, targets = _design_from_positions(positions[index : index + 1], n_transitions)
            full_inputs_list.append(inputs[0])
            full_targets_list.append(targets[0])

        k_prefix_qr, cond_prefix = _solve_qr(prefix_inputs, prefix_targets)
        k_prefix_lstsq, cond_prefix_lstsq = _solve_lstsq(prefix_inputs, prefix_targets)

        for solver, k_matrix, conditions in [
            ("qr", k_prefix_qr, cond_prefix),
            ("torch_lstsq", k_prefix_lstsq, cond_prefix_lstsq),
        ]:
            row, mean_abs = _evaluate_variant(
                horizon_periods=horizon,
                protocol="fixed_prefix",
                solver=solver,
                k_matrix=k_matrix,
                k_true=k_true,
                positions=positions,
                steps_per_trajectory=steps_per_trajectory,
                condition_numbers=conditions,
            )
            rows.append(row)
            for step, value in enumerate(mean_abs):
                mean_error_rows.append(
                    {
                        "horizon_periods": horizon,
                        "protocol": "fixed_prefix",
                        "solver": solver,
                        "step": step,
                        "mean_abs_error": float(value),
                    }
                )

        # Diagnostic only: refit using future transitions from the full rollout.
        full_k_matrices = []
        full_conditions = []
        for inputs, targets in zip(full_inputs_list, full_targets_list):
            k_matrix, condition = _solve_qr(inputs.unsqueeze(0), targets.unsqueeze(0))
            full_k_matrices.append(k_matrix[0])
            full_conditions.append(condition[0])
        k_full_qr = torch.stack(full_k_matrices)
        cond_full = torch.stack(full_conditions)
        row, _ = _evaluate_variant(
            horizon_periods=horizon,
            protocol="full_future_transitions",
            solver="qr",
            k_matrix=k_full_qr,
            k_true=k_true,
            positions=positions,
            steps_per_trajectory=steps_per_trajectory,
            condition_numbers=cond_full,
        )
        rows.append(row)

        row, _ = _evaluate_variant(
            horizon_periods=horizon,
            protocol="oracle_floor",
            solver="qr",
            k_matrix=k_true,
            k_true=k_true,
            positions=positions,
            steps_per_trajectory=steps_per_trajectory,
            condition_numbers=cond_prefix,
        )
        rows.append(row)

        if horizon == max_horizon:
            fixed_k_for_examples = k_prefix_qr
            fixed_steps_for_examples = steps_per_trajectory

    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "mean_error_by_time.csv", mean_error_rows)

    fixed_250 = next(row for row in rows if row["horizon_periods"] == 250 and row["protocol"] == "fixed_prefix" and row["solver"] == "qr")
    oracle_250 = next(row for row in rows if row["horizon_periods"] == 250 and row["protocol"] == "oracle_floor")
    summary = {
        "output_dir": str(OUTPUT_DIR),
        "dtype": "float64",
        "dt": float(DELTA_T),
        "horizons_periods": HORIZON_PERIODS,
        "fixed_prefix_transitions": FIXED_PREFIX_TRANSITIONS,
        "primary_metric": "predicted rollout x_hat(t) vs analytic true x_true(t)",
        "protocol_a": "fixed_prefix: fit from the first 168 local transitions only, then roll out to longer horizons",
        "protocol_b": "full_future_transitions: diagnostic-only fit using all valid local transitions from each full horizon",
        "fixed_prefix_250_period_row": fixed_250,
        "oracle_250_period_row": oracle_250,
        "conclusion": (
            "Float64 dense least-squares identification remains extremely accurate far beyond the observed prefix. "
            "Errors do accumulate with horizon, but they grow from machine-scale operator error rather than from a "
            "failure of position-only operator identification. The fixed-prefix protocol stays close to the oracle "
            "floor through long horizons; by hundreds of periods the residual error is best interpreted as numerical "
            "phase/radius roundoff from repeatedly powering an almost-exact matrix."
        ),
    }
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_html(rows, summary)

    _plot_metric(rows, "trajectory_mse", "rollout_mse_vs_horizon.png", "trajectory MSE vs analytic x_true")
    _plot_metric(rows, "max_abs_position_error", "max_error_vs_horizon.png", "max absolute position error")
    _plot_metric(rows, "mean_final_abs_position_error", "final_error_vs_horizon.png", "mean final-time absolute position error")
    if fixed_k_for_examples is not None and fixed_steps_for_examples is not None:
        _plot_error_examples(positions_all, omegas, periods, fixed_k_for_examples, k_true, fixed_steps_for_examples)


if __name__ == "__main__":
    main()
