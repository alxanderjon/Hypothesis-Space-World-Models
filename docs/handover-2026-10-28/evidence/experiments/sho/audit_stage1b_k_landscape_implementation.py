"""Create an implementation audit for the Stage 1B K-landscape diagnostic.

This script reads saved outputs and reconstructs only deterministic metadata.
It does not rerun K optimization or neural training.
"""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.sho.run_stage1b_direct_state_koopman_k_diagnostics import _true_k
from experiments.sho.run_stage1b_k_landscape_and_basin_analysis import (
    BASIN_THRESHOLDS,
    COV_PRETRAIN_STEPS,
    CURRICULUM_STAGES,
    DENSE_DELTA_GRID,
    DTYPE,
    OUTPUT_DIR,
    R_GRID,
    SEED,
    THETA_DELTA_GRID,
    TRAIN_LR,
    TRAIN_STEPS,
    _make_initial_k,
    _rollout_matrix,
)
from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import (
    OBS_PREFIX_TRANSITIONS,
    TEST_DATA_SEED,
    _make_prefix_future_data,
    _rollout_truth,
    _state_windows,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T


AUDIT_HTML = OUTPUT_DIR / "implementation_audit.html"
CONFIG_JSON = OUTPUT_DIR / "implementation_config_summary.json"
CONFIG_CSV = OUTPUT_DIR / "implementation_config_summary.csv"
APPENDIX_CSV = OUTPUT_DIR / "basin_sweep_run_appendix.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _random_plane_bases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    rows = []
    for plane_index in range(2):
        a = rng.normal(size=(2, 2))
        b = rng.normal(size=(2, 2))
        a = a / np.linalg.norm(a)
        b = b - np.sum(a * b) * a
        b = b / np.linalg.norm(b)
        rows.append(
            {
                "plane": f"random_plane_{plane_index}",
                "basis_a": a.tolist(),
                "basis_b": b.tolist(),
                "basis_a_frobenius_norm": float(np.linalg.norm(a)),
                "basis_b_frobenius_norm": float(np.linalg.norm(b)),
                "frobenius_inner_product": float(np.sum(a * b)),
            }
        )
    return rows


def _rollout_mse(k_matrix: torch.Tensor, true_states: torch.Tensor) -> float:
    predicted = _rollout_matrix(k_matrix, true_states[:, 0], true_states.shape[1] - 1)
    return float(torch.mean((predicted[:, 1:, 1] - true_states[:, 1:, 1]) ** 2).item())


def _reconstruct_appendix_rows(saved_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    positions, omegas, _ = _make_prefix_future_data(64, seed=TEST_DATA_SEED)
    positions = positions.to(dtype=DTYPE)
    omegas = omegas.to(dtype=DTYPE)
    true_states = _rollout_truth(positions, "full_10_period")
    prefix_in, _, = _state_windows(positions, 0, OBS_PREFIX_TRANSITIONS)
    subset = torch.linspace(0, prefix_in.shape[0] - 1, 16).round().to(torch.long)
    true_states = true_states.index_select(0, subset)
    omegas = omegas.index_select(0, subset)
    k_true = _true_k(omegas)

    appendix_rows = []
    for row_index, row in enumerate(saved_rows):
        init_kind = row["init_kind"]
        value_a = float(row["init_value_a"])
        value_b = float(row["init_value_b"])
        seed = SEED + int(abs(value_a) * 1e6) + int(value_b * 1000) + row_index
        rng = torch.Generator().manual_seed(seed)
        k_init = _make_initial_k(k_true, init_kind, value_a, value_b, rng)
        initial_mse = _rollout_mse(k_init, true_states)
        success = row["classification"] in {"exact_recovery", "near_recovery"}
        appendix_rows.append(
            {
                "run_index": row_index,
                "protocol": row["protocol"],
                "init_kind": init_kind,
                "init_value_a": value_a,
                "init_value_b": value_b,
                "rng_seed": seed,
                "num_trajectories": row["num_trajectories"],
                "initial_K": json.dumps(k_init.detach().cpu().tolist()),
                "true_K": json.dumps(k_true.detach().cpu().tolist()),
                "final_K": "NOT_SAVED_IN_ORIGINAL_OUTPUTS",
                "initial_rollout_mse": initial_mse,
                "final_rollout_mse": row["rollout_mse"],
                "k_frobenius_error": row["k_frobenius_error"],
                "trace_error": row["trace_c_error"],
                "determinant_error": row["determinant_error"],
                "eigen_angle_error": row["eigen_angle_error"],
                "success_flag": success,
                "classification": row["classification"],
                "final_k_reconstructable_without_rerun": False,
            }
        )
    return appendix_rows


def _config_summary() -> dict[str, Any]:
    return {
        "diagnostic": "stage1B_k_landscape_and_basin_analysis",
        "script": "experiments/sho/run_stage1b_k_landscape_and_basin_analysis.py",
        "output_dir": str(OUTPUT_DIR),
        "dtype": str(DTYPE),
        "delta_t": float(DELTA_T),
        "num_test_trajectories_fixed_landscapes": 64,
        "num_trajectories_basin_sweep": 16,
        "obs_prefix_transitions": OBS_PREFIX_TRANSITIONS,
        "fixed_landscape_rollout_periods": 10.0,
        "theta_delta_grid": [float(THETA_DELTA_GRID[0]), float(THETA_DELTA_GRID[-1]), len(THETA_DELTA_GRID)],
        "r_grid": [float(R_GRID[0]), float(R_GRID[-1]), len(R_GRID)],
        "dense_delta_grid": [float(DENSE_DELTA_GRID[0]), float(DENSE_DELTA_GRID[-1]), len(DENSE_DELTA_GRID)],
        "random_plane_delta_grid": [-0.015, 0.015, 41],
        "basin_thresholds": BASIN_THRESHOLDS,
        "optimizer": "Adam",
        "learning_rate": TRAIN_LR,
        "train_steps": TRAIN_STEPS,
        "cov_pretrain_steps": COV_PRETRAIN_STEPS,
        "curriculum_stages": CURRICULUM_STAGES,
        "stopping_rule": "fixed step count only; no early stopping",
        "closed_form_ls_used_in_covariance_only": False,
        "neural_networks_trained": False,
    }


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{''.join(f'<th>{escape(column)}</th>' for column in columns)}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _write_html(
    config: dict[str, Any],
    basin_rows: list[dict[str, str]],
    dense_summary: list[dict[str, str]],
    appendix_rows: list[dict[str, Any]],
    random_planes: list[dict[str, Any]],
) -> None:
    init_counts: dict[str, int] = {}
    for row in basin_rows:
        init_counts[row["init_kind"]] = init_counts.get(row["init_kind"], 0) + 1

    preview_cols = [
        "run_index",
        "protocol",
        "init_kind",
        "init_value_a",
        "init_value_b",
        "initial_rollout_mse",
        "final_rollout_mse",
        "k_frobenius_error",
        "trace_error",
        "determinant_error",
        "eigen_angle_error",
        "success_flag",
        "classification",
        "final_K",
    ]
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B K-Landscape Implementation Audit</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; vertical-align: top; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
    .note {{ background: #f7f9fc; border-left: 4px solid #5577aa; padding: 12px 14px; }}
    pre {{ white-space: pre-wrap; background: #f7f9fc; padding: 12px; border: 1px solid #d8dee8; }}
  </style>
</head>
<body>
  <h1>Stage 1B K-Landscape Implementation Audit</h1>
  <div class="note">This audit reads the script and saved outputs only. No neural networks were trained and no K optimization was rerun. Initial K and initial rollout MSE were reconstructed deterministically from saved init specs; final K matrices were not saved in the original outputs and cannot be reconstructed without rerunning optimization.</div>

  <h2>Experiment Configuration</h2>
  <pre>{escape(json.dumps(config, indent=2))}</pre>

  <h2>1. Initialization Types</h2>
  <h3>dense_noise</h3>
  <p><code>K_init = K_true + sigma * noise</code>, where <code>noise</code> is an iid standard normal tensor with the same shape as <code>K_true</code>. This is centered on <code>K_true</code>. All four entries are perturbed; no entries are kept exactly true except by chance. Sigma values were <code>1e-3</code> and <code>1e-2</code>, each with two seed indices, giving {init_counts.get("dense_noise", 0)} total basin-sweep runs across protocols. K is dense and stability is not enforced.</p>

  <h3>phase_radius_grid</h3>
  <p><code>theta = theta_true + delta_theta</code> and <code>r = 1 + delta_r</code>, with <code>delta_theta, delta_r in {{-0.03, 0, 0.03}}</code>. The initial operator is constrained to <code>[[0,1],[-r^2,2r cos(theta)]]</code>. It is centered at <code>theta_true</code> and <code>r=1</code>. This is not dense; it is constrained companion form. Stability is controlled by the chosen radius but no rejection/rescaling is applied.</p>

  <h3>random_stable</h3>
  <p>Despite the name, this does not sample arbitrary dense matrices. It samples companion-form matrices with <code>angle ~ Uniform(0.25,0.75)</code> and <code>radius ~ Uniform(0.85,1.05)</code>: <code>K = [[0,1],[-radius^2,2 radius cos(angle)]]</code>. Spectral radius is approximately the sampled radius, so values up to 1.05 are allowed. Matrices are not rescaled or rejected. Random seeds are deterministic from the global seed and run index.</p>

  <h2>2. Dense Perturbation Sensitivity</h2>
  <p>Entrywise slices used <code>delta in [-0.02, 0.02]</code> with 121 points. For <code>K00</code>, <code>K01</code>, <code>K10</code>, and <code>K11</code>, only that single entry changes and all other entries stay at <code>K_true</code>.</p>
  <p><code>trace_c</code>: adds <code>[[0,0],[0,delta]]</code>, so only <code>K11</code> changes. <code>radius_determinant</code>: replaces <code>K_true</code> with <code>K(1+delta, theta_true)</code>, so <code>K10=-(1+delta)^2</code> and <code>K11=2(1+delta)cos(theta_true)</code> both change, while <code>K00=0</code> and <code>K01=1</code> remain fixed. Its determinant is <code>(1+delta)^2</code>, and its trace is <code>2(1+delta)cos(theta_true)</code>.</p>
  <p><code>random_plane_0</code> and <code>random_plane_1</code>: two affine planes <code>K = K_true + delta_a A + delta_b B</code>. <code>A</code> is sampled from a standard normal 2x2 matrix and Frobenius-normalized. <code>B</code> is sampled independently, orthogonalized against <code>A</code>, and Frobenius-normalized. <code>delta_a, delta_b in [-0.015,0.015]</code> with 41 points per axis.</p>
  {_table(dense_summary, ["slice", "median_mse", "p90_mse", "max_mse", "median_eigen_angle_error", "median_determinant_error", "median_trace_error"])}

  <h3>Random Plane Bases</h3>
  <pre>{escape(json.dumps(random_planes, indent=2))}</pre>

  <h2>3. Covariance Protocols</h2>
  <p><code>direct_rollout</code>: optimize K with Adam on full 10-period rollout MSE only.</p>
  <p><code>curriculum</code>: optimize K with Adam through step-count stages <code>{escape(str(CURRICULUM_STAGES))}</code>. Non-full stages are numbers of local operator steps, not period-based horizons.</p>
  <p><code>covariance_only</code>: not closed-form LS. K is optimized by gradient descent on <code>L_cov</code> for {TRAIN_STEPS} Adam steps.</p>
  <p><code>cov_pretrain_then_curriculum</code>: first {COV_PRETRAIN_STEPS} Adam steps on <code>L_cov</code>, then rollout curriculum.</p>
  <p><code>covariance_plus_rollout</code>: optimize <code>L_rollout + 10 * L_cov</code> at the full 10-period stage.</p>
  <p>Covariance objective: for prefix transitions <code>X=[s_0,...,s_(m-1)]</code>, <code>Y=[s_1,...,s_m]</code>, the implementation uses <code>C00 = X X^T / m</code>, <code>C10 = Y X^T / m</code>, and <code>L_cov = mean((K C00 - C10)^2)</code> across trajectories and entries.</p>

  <h2>4. Horizon Details</h2>
  <p>In this script, fixed-K single-trajectory heatmaps and aggregate phase/radius heatmaps were 10-period only. Shorter fixed-K horizons were not evaluated in this script; they were added later in <code>outputs/sho/stage1B/k_landscape_horizon_sweep/</code>.</p>
  <p>For the basin sweep, <code>direct_rollout</code>, <code>covariance_only</code>, and <code>covariance_plus_rollout</code> used the full 10-period rollout target. Curriculum stages were step-count-based: <code>{escape(str(CURRICULUM_STAGES))}</code>. The final stage was full 10-period.</p>

  <h2>5. Success Criteria</h2>
  <p><code>exact_recovery</code>: rollout MSE &lt; 1e-12 and mean Frobenius K error &lt; 1e-8. <code>near_recovery</code>: rollout MSE &lt; 1e-8. <code>success_rate</code>: fraction of runs classified as exact or near recovery.</p>
  <p><code>unstable_divergent</code>: nonfinite rollout MSE or mean spectral radius above 1.2. <code>phase_error_failure</code>: eigen-angle error larger than both trace and determinant errors. <code>determinant_radius_failure</code>: determinant error larger than trace error. <code>wrong_stable_operator</code>: spectral radius below 1.05 and rollout MSE above 1e-6. Remaining failures are <code>general_dense_failure</code>.</p>

  <h2>6. Basin-Sweep Run Appendix</h2>
  <p>Full per-run appendix CSV: <code>basin_sweep_run_appendix.csv</code>. It includes reconstructed <code>initial_K</code>, reconstructed <code>true_K</code>, reconstructed initial rollout MSE, saved final rollout MSE and saved K-error metrics. <code>final_K</code> is marked <code>NOT_SAVED_IN_ORIGINAL_OUTPUTS</code> because the original diagnostic did not persist final K matrices.</p>
  {_table(appendix_rows[:12], preview_cols)}

  <h2>Plain-English Summary</h2>
  <ol>
    <li><code>dense_noise</code> means dense iid Gaussian perturbations of every K entry around <code>K_true</code>. <code>phase_radius_grid</code> means companion-form perturbations around true phase and radius one. <code>random_stable</code> means random companion-form phase/radius operators, not arbitrary dense stable matrices.</li>
    <li><code>covariance_only</code> was gradient optimization of the covariance normal-equation loss, not closed-form LS.</li>
    <li>The original fixed-K landscape in this output directory was 10-period only. The separate horizon sweep is the place where shorter horizons were evaluated.</li>
    <li>The fixed-K landscape and dense perturbation slices are trustworthy for qualitative sensitivity. The basin-sweep optimization results should be treated as preliminary because final K matrices were not saved, optimization used a short fixed-step Adam budget, and covariance-only did not use the closed-form LS solve that motivated the diagnostic.</li>
  </ol>
</body>
</html>
"""
    AUDIT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    basin_rows = _read_csv(OUTPUT_DIR / "basin_sweep_runs.csv")
    dense_summary = _read_csv(OUTPUT_DIR / "dense_perturbation_slice_summary.csv")
    config = _config_summary()
    random_planes = _random_plane_bases()
    appendix_rows = _reconstruct_appendix_rows(basin_rows)

    CONFIG_JSON.write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_csv(CONFIG_CSV, [config])
    _write_csv(APPENDIX_CSV, appendix_rows)
    _write_html(config, basin_rows, dense_summary, appendix_rows, random_planes)


if __name__ == "__main__":
    main()
