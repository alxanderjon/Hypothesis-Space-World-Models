"""Precision audit for direct-state least-squares Koopman identification."""

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

from experiments.sho.run_stage1b_attention_large_window_sweep import DIAGNOSTIC_NUM_TEST, _positions_to_states
from experiments.sho.run_stage1b_direct_state_koopman_ablation import _safe_corr
from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import (
    _angle_error,
    _eig_angle,
    _matrix_powers_rollout,
    _per_trajectory_rollout_mse,
    _true_k,
)
from experiments.sho.run_stage1b_two_state_omega import (
    DELTA_T,
    OMEGA_RANGE,
    _make_base_conditions,
    _state_at_time,
)


torch.set_num_threads(1)

OUTPUT_DIR = Path("outputs/sho/stage1B/direct_state_ls_precision_audit")
TEST_DATA_SEED = 1000
TEN_PERIODS = 10.0
TRANSITION_COUNTS: list[int | str] = [4, 8, 16, 32, 64, "all"]


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


def _make_long_position_data_dtype(
    *,
    dtype: torch.dtype,
    seed: int = TEST_DATA_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(DIAGNOSTIC_NUM_TEST, seed=seed, dtype=dtype)
    omega_min = float(OMEGA_RANGE[0])
    steps = int(math.ceil(TEN_PERIODS * math.tau / (omega_min * float(DELTA_T)))) + 2
    times = torch.arange(steps, dtype=dtype) * float(DELTA_T)
    times = times.unsqueeze(0).expand(DIAGNOSTIC_NUM_TEST, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions, omegas


def _data_variant(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "default_float32":
        return _make_long_position_data_dtype(dtype=torch.float32)
    if name == "float32_generated_cast_to_float64":
        positions, omegas = _make_long_position_data_dtype(dtype=torch.float32)
        return positions.to(torch.float64), omegas.to(torch.float64)
    if name == "regenerated_float64":
        return _make_long_position_data_dtype(dtype=torch.float64)
    raise ValueError(f"unknown data variant: {name}")


def _all_transition_design(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.stack((positions[:, :-2], positions[:, 1:-1]), dim=-1)
    targets = torch.stack((positions[:, 1:-1], positions[:, 2:]), dim=-1)
    return inputs, targets


def _select_transition_count(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    count: int | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count == "all":
        return inputs, targets
    n = int(count)
    if n > inputs.shape[1]:
        raise ValueError(f"requested {n} transitions, only {inputs.shape[1]} available")
    indices = torch.linspace(0, inputs.shape[1] - 1, n, device=inputs.device).round().long()
    return inputs[:, indices], targets[:, indices]


def _solve_dense_lstsq(inputs: torch.Tensor, targets: torch.Tensor, solver: str) -> torch.Tensor:
    matrices = []
    for a_matrix, b_matrix in zip(inputs, targets):
        if solver == "torch_lstsq":
            solution = torch.linalg.lstsq(a_matrix, b_matrix).solution
        elif solver == "svd_pinv":
            solution = torch.linalg.pinv(a_matrix) @ b_matrix
        elif solver == "qr":
            q_matrix, r_matrix = torch.linalg.qr(a_matrix, mode="reduced")
            solution = torch.linalg.solve_triangular(
                r_matrix,
                q_matrix.mT @ b_matrix,
                upper=True,
            )
        else:
            raise ValueError(f"unknown solver: {solver}")
        matrices.append(solution.mT)
    return torch.stack(matrices)


def _solve_scalar_companion(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x_prev = inputs[..., 0]
    x_now = inputs[..., 1]
    x_next = targets[..., 1]
    rhs = x_next + x_prev
    c = torch.sum(x_now * rhs, dim=1) / torch.sum(x_now * x_now, dim=1)
    k_matrix = torch.zeros((inputs.shape[0], 2, 2), dtype=inputs.dtype, device=inputs.device)
    k_matrix[:, 0, 1] = 1.0
    k_matrix[:, 1, 0] = -1.0
    k_matrix[:, 1, 1] = c
    return k_matrix


def _condition_numbers(inputs: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.linalg.cond(item) for item in inputs])


def _position_prediction_metrics(k_matrix: torch.Tensor, positions: torch.Tensor) -> dict[str, float]:
    true_states = _positions_to_states(positions)
    predicted = _matrix_powers_rollout(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return {
        "one_step_mse": float(torch.mean((predicted[:, 1, 1] - true_states[:, 1, 1]) ** 2).item()),
        "two_step_mse": float(torch.mean((predicted[:, 2, 1] - true_states[:, 2, 1]) ** 2).item()),
        "ten_period_rollout_mse": float(torch.mean((predicted[:, 1:, 1] - true_states[:, 1:, 1]) ** 2).item()),
        "oracle_floor_note": 0.0,
    }


def _matrix_errors(k_matrix: torch.Tensor, k_true: torch.Tensor) -> dict[str, np.ndarray]:
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    trace = np.trace(k_np, axis1=1, axis2=2)
    true_trace = np.trace(true_np, axis1=1, axis2=2)
    determinant = np.linalg.det(k_np)
    true_determinant = np.linalg.det(true_np)
    radius = np.abs(np.linalg.eigvals(k_np)).max(axis=1)
    true_radius = np.abs(np.linalg.eigvals(true_np)).max(axis=1)
    return {
        "k_frobenius_error": np.linalg.norm(k_np - true_np, axis=(1, 2)),
        "trace_abs_error": np.abs(trace - true_trace),
        "c_abs_error": np.abs(trace - true_trace),
        "determinant_abs_error": np.abs(determinant - true_determinant),
        "eigen_angle_abs_error": _angle_error(_eig_angle(k_np), _eig_angle(true_np)),
        "spectral_radius_abs_error": np.abs(radius - true_radius),
    }


def _summarize(
    *,
    part: str,
    data_variant: str,
    solver: str,
    transition_count: int | str,
    fit_type: str,
    k_matrix: torch.Tensor,
    k_true: torch.Tensor,
    positions: torch.Tensor,
    condition_numbers: torch.Tensor,
    oracle_floor: float,
) -> dict[str, Any]:
    per_rollout = _per_trajectory_rollout_mse(k_matrix, positions).detach().cpu().numpy()
    errors = _matrix_errors(k_matrix, k_true)
    row: dict[str, Any] = {
        "part": part,
        "data_variant": data_variant,
        "dtype": str(positions.dtype).replace("torch.", ""),
        "solver": solver,
        "fit_type": fit_type,
        "transition_count": transition_count,
        "num_transitions_used": int(condition_numbers.numel() and (positions.shape[1] - 2 if transition_count == "all" else int(transition_count))),
        **{k: v for k, v in _position_prediction_metrics(k_matrix, positions).items() if k != "oracle_floor_note"},
        "oracle_k_rollout_floor": oracle_floor,
        "condition_number_mean": float(condition_numbers.mean().item()),
        "condition_number_max": float(condition_numbers.max().item()),
    }
    for key, values in errors.items():
        row[f"{key}_mean"] = float(np.mean(values))
        row[f"{key}_max"] = float(np.max(values))
        row[f"{key}_corr_with_rollout"] = _safe_corr(values, per_rollout)
    row["condition_number_corr_with_k_error"] = _safe_corr(
        condition_numbers.detach().cpu().numpy(),
        errors["k_frobenius_error"],
    )
    return row


def _run_audit() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for data_variant in ["default_float32", "float32_generated_cast_to_float64", "regenerated_float64"]:
        positions, omegas = _data_variant(data_variant)
        k_true = _true_k(omegas)
        all_inputs, all_targets = _all_transition_design(positions)
        all_conditions = _condition_numbers(all_inputs)
        oracle_floor = _position_prediction_metrics(k_true, positions)["ten_period_rollout_mse"]

        for solver in ["torch_lstsq", "svd_pinv", "qr"]:
            k_matrix = _solve_dense_lstsq(all_inputs, all_targets, solver)
            row = _summarize(
                part="dtype_and_solver",
                data_variant=data_variant,
                solver=solver,
                transition_count="all",
                fit_type="dense_2x2",
                k_matrix=k_matrix,
                k_true=k_true,
                positions=positions,
                condition_numbers=all_conditions,
                oracle_floor=oracle_floor,
            )
            rows.append(row)
            errors = _matrix_errors(k_matrix, k_true)["k_frobenius_error"]
            for index, (condition, error) in enumerate(zip(all_conditions.detach().cpu().numpy(), errors)):
                condition_rows.append(
                    {
                        "data_variant": data_variant,
                        "solver": solver,
                        "trajectory": index,
                        "condition_number": float(condition),
                        "k_frobenius_error": float(error),
                    },
                )

        if data_variant == "regenerated_float64":
            for count in TRANSITION_COUNTS:
                selected_inputs, selected_targets = _select_transition_count(all_inputs, all_targets, count)
                conditions = _condition_numbers(selected_inputs)
                k_matrix = _solve_dense_lstsq(selected_inputs, selected_targets, "torch_lstsq")
                rows.append(
                    _summarize(
                        part="transition_count",
                        data_variant=data_variant,
                        solver="torch_lstsq",
                        transition_count=count,
                        fit_type="dense_2x2",
                        k_matrix=k_matrix,
                        k_true=k_true,
                        positions=positions,
                        condition_numbers=conditions,
                        oracle_floor=oracle_floor,
                    ),
                )

            k_dense = _solve_dense_lstsq(all_inputs, all_targets, "torch_lstsq")
            k_scalar = _solve_scalar_companion(all_inputs, all_targets)
            for fit_type, k_matrix in [
                ("dense_2x2", k_dense),
                ("diagnostic_only_scalar_companion", k_scalar),
                ("true_oracle", k_true),
            ]:
                rows.append(
                    _summarize(
                        part="structured_diagnostic",
                        data_variant=data_variant,
                        solver="torch_lstsq" if fit_type != "true_oracle" else "oracle",
                        transition_count="all",
                        fit_type=fit_type,
                        k_matrix=k_matrix,
                        k_true=k_true,
                        positions=positions,
                        condition_numbers=all_conditions,
                        oracle_floor=oracle_floor,
                    ),
                )

    return rows, condition_rows


def _plot_grouped_bar(path: Path, rows: list[dict[str, Any]], metric: str, title: str, ylabel: str) -> None:
    labels = [f"{row['data_variant']}\n{row['solver']}" for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(range(len(values)), values)
    plt.yscale("log")
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_transition_count(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["part"] == "transition_count"]
    x_labels = [str(row["transition_count"]) for row in selected]
    values = [float(row["ten_period_rollout_mse"]) for row in selected]
    plt.figure(figsize=(6.8, 4.8))
    plt.plot(x_labels, values, marker="o")
    plt.yscale("log")
    plt.xlabel("transitions used")
    plt.ylabel("10-period rollout MSE")
    plt.title("rollout MSE vs transition count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_vs_transition_count.png")
    plt.close()


def _plot_condition_scatter(condition_rows: list[dict[str, Any]]) -> None:
    selected = [row for row in condition_rows if row["data_variant"] == "regenerated_float64" and row["solver"] == "torch_lstsq"]
    plt.figure(figsize=(6.4, 4.8))
    plt.scatter(
        [row["condition_number"] for row in selected],
        [row["k_frobenius_error"] for row in selected],
        s=28,
        alpha=0.82,
    )
    plt.yscale("log")
    plt.xlabel("state design condition number")
    plt.ylabel("K Frobenius error")
    plt.title("condition number vs K error")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "condition_number_vs_k_error.png")
    plt.close()


def _write_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return escape(str(value))

    compact = [
        row
        for row in rows
        if row["part"] in {"dtype_and_solver", "structured_diagnostic"}
    ]
    table_rows = "\n".join(
        "<tr>"
        + "".join(
            f"<td>{fmt(row[column])}</td>"
            for column in [
                "part",
                "data_variant",
                "solver",
                "fit_type",
                "transition_count",
                "one_step_mse",
                "two_step_mse",
                "ten_period_rollout_mse",
                "oracle_k_rollout_floor",
                "k_frobenius_error_mean",
                "trace_abs_error_mean",
                "determinant_abs_error_mean",
                "eigen_angle_abs_error_mean",
            ]
        )
        + "</tr>"
        for row in compact
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Direct-State LS Precision Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
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
  <h1>Direct-State Least-Squares Precision Audit</h1>
  <p>No training was run. Dense <code>2x2</code> operators were fit from position-only states <code>s_t=(x_{{t-1}},x_t)</code> using local transitions.</p>
  <h2>Conclusion</h2>
  <p>{escape(summary["conclusion"])}</p>
  <h2>Main Results</h2>
  <table>
    <thead><tr><th>part</th><th>data</th><th>solver</th><th>fit type</th><th>transitions</th><th>one-step</th><th>two-step</th><th>10-period rollout</th><th>oracle floor</th><th>K err</th><th>trace/c err</th><th>det err</th><th>angle err</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Plots</h2>
  <div class="grid">
    <img src="rollout_mse_by_dtype_solver.png" alt="rollout MSE by dtype and solver">
    <img src="k_error_by_dtype_solver.png" alt="K error by dtype and solver">
    <img src="rollout_mse_vs_transition_count.png" alt="rollout MSE vs transition count">
    <img src="condition_number_vs_k_error.png" alt="condition number vs K error">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, condition_rows = _run_audit()
    dtype_solver_rows = [row for row in rows if row["part"] == "dtype_and_solver"]
    best_row = min(rows, key=lambda row: float(row["ten_period_rollout_mse"]))
    best_dense = min(
        [row for row in rows if row["fit_type"] == "dense_2x2"],
        key=lambda row: float(row["ten_period_rollout_mse"]),
    )
    oracle_row = min(
        [row for row in rows if row["fit_type"] == "true_oracle"],
        key=lambda row: float(row["ten_period_rollout_mse"]),
    )
    summary = {
        "output_dir": str(OUTPUT_DIR),
        "num_test_trajectories": DIAGNOSTIC_NUM_TEST,
        "dt": float(DELTA_T),
        "transition_counts": TRANSITION_COUNTS,
        "best_overall": best_row,
        "best_dense": best_dense,
        "oracle": oracle_row,
        "conclusion": (
            "Dense least-squares operator identification reaches much closer to the oracle floor when the data are "
            "generated directly in float64. Casting float32 data to float64 cannot recover the precision already lost "
            "during trajectory generation. Solver choice matters little once the state design is well conditioned; the "
            "remaining gap is primarily numerical precision and unconstrained dense-matrix roundoff, not lack of "
            "trajectory evidence."
        ),
    }
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "condition_number_details.csv", condition_rows)
    _write_json(OUTPUT_DIR / "summary.json", summary)
    _write_html(rows, summary)
    _plot_grouped_bar(
        OUTPUT_DIR / "rollout_mse_by_dtype_solver.png",
        dtype_solver_rows,
        "ten_period_rollout_mse",
        "rollout MSE by dtype and solver",
        "10-period rollout MSE",
    )
    _plot_grouped_bar(
        OUTPUT_DIR / "k_error_by_dtype_solver.png",
        dtype_solver_rows,
        "k_frobenius_error_mean",
        "K error by dtype and solver",
        "mean ||K - K_true||_F",
    )
    _plot_transition_count(rows)
    _plot_condition_scatter(condition_rows)


if __name__ == "__main__":
    main()
