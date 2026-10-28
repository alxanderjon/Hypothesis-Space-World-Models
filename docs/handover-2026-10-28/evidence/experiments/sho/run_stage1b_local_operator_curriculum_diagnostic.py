"""Stage 1B local-operator curriculum diagnostic.

This trains trajectory-specific K directly. It is not a neural architecture.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
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
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import (
    OBS_PREFIX_TRANSITIONS,
    TEST_DATA_SEED,
    _fit_dense_k_qr,
    _make_prefix_future_data,
    _rollout_truth,
    _state_windows,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/local_operator_curriculum_diagnostic")
DTYPE = torch.float64
SEED = 0
LEARNING_RATE = 1e-2
COV_PRETRAIN_STEPS = 1000
STAGE_STEPS = 2000
HORIZON_PERIODS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DIRECT_STEPS = STAGE_STEPS * len(HORIZON_PERIODS)
LAMBDA_COV_VALUES = [1e-3, 1e-2, 1e-1]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _apply_k(k_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    return torch.bmm(k_matrix, states.unsqueeze(-1)).squeeze(-1)


def _rollout_matrix(k_matrix: torch.Tensor, state0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [state0]
    state = state0
    for _ in range(steps):
        state = _apply_k(k_matrix, state)
        states.append(state)
    return torch.stack(states, dim=1)


def _period_steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _states_for_steps(positions: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.stack((positions[:, : steps + 1], positions[:, 1 : steps + 2]), dim=-1)


def _representative_medium(omegas: torch.Tensor) -> int:
    values = omegas.detach().cpu().numpy()
    order = np.argsort(values)
    return int(order[len(order) // 2])


def _make_initializations(k_true: torch.Tensor, theta_true: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(SEED)
    for sigma in [1e-3, 1e-2, 1e-1]:
        noise = torch.randn(k_true.shape, dtype=k_true.dtype, generator=generator)
        rows.append(
            {
                "init_kind": "dense_noise",
                "init_detail": f"sigma={sigma:g}",
                "k_init": k_true + sigma * noise,
            }
        )
    for r_delta in [-0.05, -0.03, 0.0, 0.03, 0.05]:
        for theta_delta in [-0.05, -0.03, 0.0, 0.03, 0.05]:
            radius = 1.0 + r_delta
            theta = theta_true + theta_delta
            k = torch.zeros_like(k_true)
            k[:, 0, 1] = 1.0
            k[:, 1, 0] = -(radius**2)
            k[:, 1, 1] = 2.0 * radius * torch.cos(theta)
            rows.append(
                {
                    "init_kind": "phase_radius_grid",
                    "init_detail": f"r_delta={r_delta:g},theta_delta={theta_delta:g}",
                    "k_init": k,
                }
            )
    for sample in range(10):
        radius = torch.empty(1, dtype=DTYPE).uniform_(0.85, 1.05, generator=generator)
        theta = theta_true + torch.empty(1, dtype=DTYPE).uniform_(-0.1, 0.1, generator=generator)
        k = torch.zeros_like(k_true)
        k[:, 0, 1] = 1.0
        k[:, 1, 0] = -(radius**2)
        k[:, 1, 1] = 2.0 * radius * torch.cos(theta)
        rows.append(
            {
                "init_kind": "random_companion",
                "init_detail": f"sample={sample},r={float(radius.item()):.8g},theta={float(theta.item()):.8g}",
                "k_init": k,
            }
        )
    return rows


def _rollout_loss(k_matrix: torch.Tensor, true_states: torch.Tensor) -> torch.Tensor:
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return torch.mean((predicted[:, 1:, 1] - true_states[:, 1:, 1]) ** 2)


def _cov_loss(k_matrix: torch.Tensor, prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> torch.Tensor:
    x = prefix_in.transpose(1, 2)
    y = prefix_out.transpose(1, 2)
    c00 = torch.bmm(x, x.transpose(1, 2)) / prefix_in.shape[1]
    c10 = torch.bmm(y, x.transpose(1, 2)) / prefix_in.shape[1]
    return torch.mean((torch.bmm(k_matrix, c00) - c10) ** 2)


def _metrics(k_matrix: torch.Tensor, k_true: torch.Tensor, true_states: torch.Tensor) -> dict[str, Any]:
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    error = predicted[:, 1:, 1] - true_states[:, 1:, 1]
    k_np = k_matrix.detach().cpu().numpy()
    true_np = k_true.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(k_np)
    return {
        "rollout_mse": float(torch.mean(error**2).item()),
        "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
        "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()),
        "max_abs_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_abs_error": float(torch.mean(torch.abs(error[:, -1])).item()),
        "k_frobenius_error": float(np.mean(np.linalg.norm(k_np - true_np, axis=(1, 2)))),
        "trace_c_error": float(np.mean(np.abs(np.trace(k_np, axis1=1, axis2=2) - np.trace(true_np, axis1=1, axis2=2)))),
        "determinant_error": float(np.mean(np.abs(np.linalg.det(k_np) - np.linalg.det(true_np)))),
        "spectral_radius_error": float(np.mean(np.abs(np.abs(eigvals).max(axis=1) - 1.0))),
        "eigen_angle_error": float(np.mean(_angle_error(_eig_angle(k_np), _eig_angle(true_np)))),
    }


def _success_label(rollout_mse: float) -> str:
    if rollout_mse < 1e-12:
        return "exact"
    if rollout_mse < 1e-9:
        return "near"
    if rollout_mse < 1e-6:
        return "useful"
    return "failure"


def _train_batch(
    protocol: str,
    k_init: torch.Tensor,
    prefix_in: torch.Tensor,
    prefix_out: torch.Tensor,
    states_by_period: dict[float, torch.Tensor],
    *,
    lambda_cov: float = 0.0,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    parameter = torch.nn.Parameter(k_init.clone())
    optimizer = torch.optim.Adam([parameter], lr=LEARNING_RATE)
    curves: list[dict[str, Any]] = []

    def train_steps(num_steps: int, stage_name: str, states: torch.Tensor | None, use_cov: bool, use_rollout: bool) -> None:
        for step in range(1, num_steps + 1):
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=DTYPE)
            if use_rollout:
                assert states is not None
                loss = loss + _rollout_loss(parameter, states)
            if use_cov:
                weight = 1.0 if not use_rollout else lambda_cov
                loss = loss + weight * _cov_loss(parameter, prefix_in, prefix_out)
            loss.backward()
            optimizer.step()
            if step == 1 or step % 500 == 0 or step == num_steps:
                with torch.no_grad():
                    full_loss = _rollout_loss(parameter, states_by_period[10.0])
                    curves.append(
                        {
                            "protocol": protocol,
                            "lambda_cov": lambda_cov,
                            "stage": stage_name,
                            "stage_step": step,
                            "global_curve_step": len(curves),
                            "full_10_period_rollout_mse": float(full_loss.item()),
                            "stage_loss": float(loss.detach().item()),
                        }
                    )

    if protocol == "direct_10_period_rollout":
        train_steps(DIRECT_STEPS, "10_period_direct", states_by_period[10.0], use_cov=False, use_rollout=True)
    elif protocol == "horizon_curriculum_only":
        for periods in HORIZON_PERIODS:
            train_steps(STAGE_STEPS, f"{periods:g}_periods", states_by_period[periods], use_cov=False, use_rollout=True)
    elif protocol == "covariance_pretrain_then_horizon_curriculum":
        train_steps(COV_PRETRAIN_STEPS, "covariance_pretrain", None, use_cov=True, use_rollout=False)
        for periods in HORIZON_PERIODS:
            train_steps(STAGE_STEPS, f"{periods:g}_periods", states_by_period[periods], use_cov=False, use_rollout=True)
    elif protocol == "covariance_plus_horizon_curriculum":
        for periods in HORIZON_PERIODS:
            train_steps(STAGE_STEPS, f"{periods:g}_periods", states_by_period[periods], use_cov=True, use_rollout=True)
    else:
        raise ValueError(protocol)
    return parameter.detach(), curves


def _closed_form_ls(prefix_in: torch.Tensor, prefix_out: torch.Tensor) -> torch.Tensor:
    k_matrix, _ = _fit_dense_k_qr(prefix_in, prefix_out)
    return k_matrix


def _plot_bars(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, *, log: bool = True) -> None:
    labels = sorted({row["protocol_label"] for row in rows})
    values = []
    for label in labels:
        selected = [row for row in rows if row["protocol_label"] == label]
        values.append(float(np.median([row[metric] for row in selected])))
    plt.figure(figsize=(10.0, 5.4))
    plt.bar(labels, values)
    if log:
        plt.yscale("log")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_success(rows: list[dict[str, Any]]) -> None:
    labels = sorted({row["protocol_label"] for row in rows})
    useful = []
    near = []
    exact = []
    for label in labels:
        selected = [row for row in rows if row["protocol_label"] == label]
        useful.append(np.mean([row["success_label"] in {"useful", "near", "exact"} for row in selected]))
        near.append(np.mean([row["success_label"] in {"near", "exact"} for row in selected]))
        exact.append(np.mean([row["success_label"] == "exact" for row in selected]))
    x = np.arange(len(labels))
    plt.figure(figsize=(10.0, 5.4))
    plt.bar(x - 0.25, useful, width=0.25, label="useful <1e-6")
    plt.bar(x, near, width=0.25, label="near <1e-9")
    plt.bar(x + 0.25, exact, width=0.25, label="exact <1e-12")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("success rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basin_success_by_protocol.png", dpi=160)
    plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10.0, 5.4))
    for label in sorted({row["protocol_label"] for row in curves}):
        selected = [row for row in curves if row["protocol_label"] == label]
        xs = np.arange(len(selected))
        ys = [row["median_full_10_period_rollout_mse"] for row in selected]
        plt.plot(xs, ys, label=label)
    plt.yscale("log")
    plt.xlabel("recorded curve point")
    plt.ylabel("median full 10-period rollout MSE")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160)
    plt.close()


def _write_html(rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    columns = ["protocol_label", "num_runs", "success_useful_rate", "success_near_rate", "success_exact_rate", "median_final_rollout_mse", "best_final_rollout_mse", "median_k_frobenius_error"]
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in summary_rows
    )
    best = min(summary_rows, key=lambda row: row["median_final_rollout_mse"])
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Local Operator Curriculum Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Stage 1B Local Operator Curriculum Diagnostic</h1>
  <p>Direct K optimization on one medium-omega representative trajectory, using position-only state <code>s_t=(x_{{t-1}},x_t)</code>. No neural network, no velocity, no labels for training. True K is used only for diagnostics.</p>
  <p>Best median protocol: <code>{escape(best["protocol_label"])}</code>.</p>
  <table><thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in columns)}</tr></thead><tbody>{table_rows}</tbody></table>
  <div class="grid">
    <img src="basin_success_by_protocol.png">
    <img src="final_rollout_mse_by_protocol.png">
    <img src="k_error_by_protocol.png">
    <img src="curriculum_stage_curves.png">
  </div>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def main() -> None:
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_steps = _period_steps(10.0)
    positions, omegas, _ = _make_prefix_future_data(DIAGNOSTIC_NUM_TEST, seed=TEST_DATA_SEED)
    positions = positions[:, : max_steps + 2].to(dtype=DTYPE)
    omegas = omegas.to(dtype=DTYPE)
    index = _representative_medium(omegas)
    positions = positions[index : index + 1]
    omegas = omegas[index : index + 1]
    k_true = _true_k(omegas)
    theta_true = omegas * float(DELTA_T)
    prefix_in, prefix_out = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    states_by_period = {periods: _states_for_steps(positions, _period_steps(periods)) for periods in HORIZON_PERIODS}
    full_states = states_by_period[10.0]

    setup = {
        "diagnostic": "stage1B_local_operator_curriculum_diagnostic",
        "runtime_scope": "medium_omega_representative_only_per_runtime_fallback",
        "representative_index": index,
        "omega": float(omegas.item()),
        "theta_true": float(theta_true.item()),
        "dtype": str(DTYPE),
        "position_only_state": "s_t=(x_{t-1},x_t)",
        "uses_neural_network": False,
        "uses_velocity": False,
        "uses_omega_labels_for_training": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder_in_model": False,
        "learning_rate": LEARNING_RATE,
        "cov_pretrain_steps": COV_PRETRAIN_STEPS,
        "stage_steps": STAGE_STEPS,
        "direct_steps": DIRECT_STEPS,
        "horizon_periods": HORIZON_PERIODS,
        "lambda_cov_values": LAMBDA_COV_VALUES,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    init_rows = _make_initializations(k_true, theta_true)
    k_init_batch = torch.cat([row["k_init"] for row in init_rows], dim=0)
    initial_metrics = [_metrics(row["k_init"], k_true, full_states) for row in init_rows]

    protocols: list[tuple[str, float]] = [
        ("direct_10_period_rollout", 0.0),
        ("horizon_curriculum_only", 0.0),
        ("covariance_pretrain_then_horizon_curriculum", 0.0),
    ] + [("covariance_plus_horizon_curriculum", value) for value in LAMBDA_COV_VALUES]

    per_run_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    def write_partial() -> None:
        _write_csv(OUTPUT_DIR / "per_run_results.csv", per_run_rows)
        _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curve_rows)

    for protocol, lambda_cov in protocols:
        protocol_label = protocol if lambda_cov == 0.0 else f"{protocol}_lambda_{lambda_cov:g}"
        print(f"training protocol {protocol_label}", flush=True)
        final_k, curves = _train_batch(protocol, k_init_batch, prefix_in.expand(k_init_batch.shape[0], -1, -1), prefix_out.expand(k_init_batch.shape[0], -1, -1), {key: value.expand(k_init_batch.shape[0], -1, -1) for key, value in states_by_period.items()}, lambda_cov=lambda_cov)
        for curve in curves:
            curve["protocol_label"] = protocol_label
        curve_by_point: dict[tuple[str, str, int], list[float]] = {}
        for curve in curves:
            key = (curve["protocol_label"], curve["stage"], int(curve["stage_step"]))
            curve_by_point.setdefault(key, []).append(curve["full_10_period_rollout_mse"])
        for (label, stage, stage_step), values in curve_by_point.items():
            curve_rows.append(
                {
                    "protocol_label": label,
                    "stage": stage,
                    "stage_step": stage_step,
                    "median_full_10_period_rollout_mse": float(np.median(values)),
                }
            )
        for run_index, init_row in enumerate(init_rows):
            final = final_k[run_index : run_index + 1]
            final_metrics = _metrics(final, k_true, full_states)
            label = _success_label(final_metrics["rollout_mse"])
            per_run_rows.append(
                {
                    "protocol": protocol,
                    "protocol_label": protocol_label,
                    "lambda_cov": lambda_cov,
                    "init_kind": init_row["init_kind"],
                    "init_detail": init_row["init_detail"],
                    "learning_rate": LEARNING_RATE,
                    "initial_rollout_mse": initial_metrics[run_index]["rollout_mse"],
                    "final_rollout_mse": final_metrics["rollout_mse"],
                    "one_step_mse": final_metrics["one_step_mse"],
                    "two_step_mse": final_metrics["two_step_mse"],
                    "max_abs_trajectory_error": final_metrics["max_abs_trajectory_error"],
                    "final_time_abs_error": final_metrics["final_time_abs_error"],
                    "initial_K": json.dumps(init_row["k_init"].detach().cpu().tolist()),
                    "final_K": json.dumps(final.detach().cpu().tolist()),
                    "true_K": json.dumps(k_true.detach().cpu().tolist()),
                    "k_frobenius_error": final_metrics["k_frobenius_error"],
                    "trace_c_error": final_metrics["trace_c_error"],
                    "determinant_error": final_metrics["determinant_error"],
                    "spectral_radius_error": final_metrics["spectral_radius_error"],
                    "eigen_angle_error": final_metrics["eigen_angle_error"],
                    "success_label": label,
                    "success_flag": label in {"exact", "near", "useful"},
                }
            )
        write_partial()

    k_ls = _closed_form_ls(prefix_in, prefix_out)
    ls_metrics = _metrics(k_ls, k_true, full_states)
    per_run_rows.append(
        {
            "protocol": "closed_form_LS_baseline",
            "protocol_label": "closed_form_LS_baseline",
            "lambda_cov": "",
            "init_kind": "closed_form",
            "init_detail": "QR least squares from prefix transitions",
            "learning_rate": "",
            "initial_rollout_mse": "",
            "final_rollout_mse": ls_metrics["rollout_mse"],
            "one_step_mse": ls_metrics["one_step_mse"],
            "two_step_mse": ls_metrics["two_step_mse"],
            "max_abs_trajectory_error": ls_metrics["max_abs_trajectory_error"],
            "final_time_abs_error": ls_metrics["final_time_abs_error"],
            "initial_K": "",
            "final_K": json.dumps(k_ls.detach().cpu().tolist()),
            "true_K": json.dumps(k_true.detach().cpu().tolist()),
            "k_frobenius_error": ls_metrics["k_frobenius_error"],
            "trace_c_error": ls_metrics["trace_c_error"],
            "determinant_error": ls_metrics["determinant_error"],
            "spectral_radius_error": ls_metrics["spectral_radius_error"],
            "eigen_angle_error": ls_metrics["eigen_angle_error"],
            "success_label": _success_label(ls_metrics["rollout_mse"]),
            "success_flag": True,
        }
    )

    summary_rows: list[dict[str, Any]] = []
    for label in sorted({row["protocol_label"] for row in per_run_rows}):
        selected = [row for row in per_run_rows if row["protocol_label"] == label]
        final_values = [float(row["final_rollout_mse"]) for row in selected]
        summary_rows.append(
            {
                "protocol_label": label,
                "num_runs": len(selected),
                "success_useful_rate": float(np.mean([row["success_label"] in {"useful", "near", "exact"} for row in selected])),
                "success_near_rate": float(np.mean([row["success_label"] in {"near", "exact"} for row in selected])),
                "success_exact_rate": float(np.mean([row["success_label"] == "exact" for row in selected])),
                "median_final_rollout_mse": float(np.median(final_values)),
                "best_final_rollout_mse": float(np.min(final_values)),
                "median_k_frobenius_error": float(np.median([float(row["k_frobenius_error"]) for row in selected])),
            }
        )

    setup["wall_clock_sec"] = time.perf_counter() - start_time
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_csv(OUTPUT_DIR / "per_run_results.csv", per_run_rows)
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curve_rows)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows})
    _plot_success(per_run_rows)
    _plot_bars(per_run_rows, "final_rollout_mse", "final_rollout_mse_by_protocol.png", "median final rollout MSE")
    _plot_bars(per_run_rows, "k_frobenius_error", "k_error_by_protocol.png", "median K Frobenius error")
    _plot_curves(curve_rows)
    _write_html(per_run_rows, summary_rows)


if __name__ == "__main__":
    main()
