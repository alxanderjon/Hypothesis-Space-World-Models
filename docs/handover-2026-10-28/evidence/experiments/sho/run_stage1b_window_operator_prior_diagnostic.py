"""Stage 1B window-prior operator-identification diagnostic.

K_i is always identified by least squares from transitions. No network
generates K_i.
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
from torch import nn

from experiments.sho.run_stage1b_learned_state_ls_operator_curriculum import OBS_PREFIX_TRANSITIONS
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/window_operator_prior_diagnostic")
DTYPE = torch.float64
SEED = 0
TRAIN_DATA_SEED = 0
TEST_DATA_SEED = 1000

WINDOW_SIZES = [2, 8, 16, 32]
H_DIMS = [2, 4]
VARIANTS = {
    "rollout_only": {"lambda_recon": 0.0, "lambda_shift": 0.0, "lambda_split": 0.0},
    "rollout_recon": {"lambda_recon": 1e-1, "lambda_shift": 0.0, "lambda_split": 0.0},
    "rollout_recon_shift": {"lambda_recon": 1e-1, "lambda_shift": 1e-1, "lambda_split": 0.0},
    "split_consistency": {"lambda_recon": 1e-1, "lambda_shift": 1e-1, "lambda_split": 1e-1},
}
CURRICULUM_STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
TRAIN_NUM = 24
TEST_NUM = 16
MAX_WINDOW = max(WINDOW_SIZES)
ROLLOUT_PERIODS = 10.0
STAGE_STEPS = 50
STAGE_STEPS_BY_STAGE: dict[int | str, int] = {
    1: 50,
    2: 50,
    4: 50,
    8: 50,
    16: 40,
    32: 25,
    64: 15,
    "full_10_period": 3,
}
LEARNING_RATE = 1e-3
RIDGE = 1e-8
HIDDEN_WIDTH = 32


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


def _full_rollout_steps() -> int:
    return int(math.ceil(ROLLOUT_PERIODS * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_data(num_trajectories: int, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed, dtype=DTYPE)
    total_steps = OBS_PREFIX_TRANSITIONS + MAX_WINDOW + _full_rollout_steps() + 4
    times = torch.arange(total_steps, dtype=DTYPE) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions, omegas, initial_conditions


def _window_sequence(positions: torch.Tensor, m: int, start_end: int, num_windows: int) -> torch.Tensor:
    windows = [positions[:, end - m + 1 : end + 1] for end in range(start_end, start_end + num_windows)]
    return torch.stack(windows, dim=1)


def _prefix_pairs(positions: torch.Tensor, m: int, *, start_end: int | None = None, num_transitions: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    start = m - 1 if start_end is None else start_end
    count = OBS_PREFIX_TRANSITIONS if num_transitions is None else num_transitions
    inputs = _window_sequence(positions, m, start, count)
    targets = _window_sequence(positions, m, start + 1, count)
    return inputs, targets


def _rollout_truth(positions: torch.Tensor, m: int, stage: int | str) -> torch.Tensor:
    steps = _full_rollout_steps() if stage == "full_10_period" else int(stage)
    start_end = OBS_PREFIX_TRANSITIONS + m - 1
    return _window_sequence(positions, m, start_end, steps + 1)


def _apply_k(k_matrix: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    if states.ndim == 3:
        return torch.einsum("bij,btj->bti", k_matrix, states)
    return torch.bmm(k_matrix, states.unsqueeze(-1)).squeeze(-1)


def _rollout_matrix(k_matrix: torch.Tensor, state0: torch.Tensor, steps: int) -> torch.Tensor:
    states = [state0]
    state = state0
    for _ in range(steps):
        state = _apply_k(k_matrix, state)
        states.append(state)
    return torch.stack(states, dim=1)


def _fit_k_ridge(inputs: torch.Tensor, targets: torch.Tensor, ridge: float = RIDGE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dim = inputs.shape[-1]
    eye = torch.eye(dim, dtype=inputs.dtype, device=inputs.device).unsqueeze(0)
    gram = torch.bmm(inputs.transpose(1, 2), inputs)
    rhs = torch.bmm(inputs.transpose(1, 2), targets)
    solution = torch.linalg.solve(gram + ridge * eye, rhs)
    residual = torch.mean((torch.bmm(inputs, solution) - targets) ** 2)
    conditions = torch.linalg.cond(gram + ridge * eye)
    return solution.transpose(1, 2), residual, conditions


def _metrics_from_rollout(predicted: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    error = predicted[:, 1:, -1] - truth[:, 1:, -1]
    return {
        "rollout_mse": float(torch.mean(error**2).item()),
        "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
        "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()) if error.shape[1] > 1 else float("nan"),
        "max_abs_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_abs_error": float(torch.mean(torch.abs(error[:, -1])).item()),
    }


def _spectral_radius(k_matrix: torch.Tensor) -> float:
    eigvals = np.linalg.eigvals(k_matrix.detach().cpu().numpy())
    return float(np.mean(np.abs(eigvals).max(axis=1)))


class WindowLSModel(nn.Module):
    def __init__(self, m: int, h_dim: int) -> None:
        super().__init__()
        self.m = m
        self.h_dim = h_dim
        self.encoder = nn.Sequential(nn.Linear(m, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, h_dim))
        self.decoder = nn.Sequential(nn.Linear(h_dim, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, m))

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return self.encoder(q)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def identify_k(self, q_in: torch.Tensor, q_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fit_k_ridge(self.encode(q_in), self.encode(q_out))


def _evaluate_raw_window_ls(positions: torch.Tensor, m: int) -> dict[str, Any]:
    q_in, q_out = _prefix_pairs(positions, m)
    k_matrix, residual, conditions = _fit_k_ridge(q_in, q_out, ridge=1e-12)
    truth = _rollout_truth(positions, m, "full_10_period")
    predicted = _rollout_matrix(k_matrix, truth[:, 0], truth.shape[1] - 1)
    row = {
        "model_type": "raw_window_ls",
        "m": m,
        "h_dim": "",
        "variant": "raw_closed_form_ls",
        "parameter_count": 0,
        "latent_transition_mse": float(residual.item()),
        "ls_condition_number": float(conditions.mean().item()),
        "spectral_radius": _spectral_radius(k_matrix),
        "final_k_matrices": json.dumps(k_matrix.detach().cpu().tolist()),
    }
    row.update(_metrics_from_rollout(predicted, truth))
    return row


def _loss_parts(model: WindowLSModel, positions: torch.Tensor, stage: int | str, variant_cfg: dict[str, float]) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_matrix, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    h0 = model.encode(truth[:, 0])
    h_rollout = _rollout_matrix(k_matrix, h0, truth.shape[1] - 1)
    decoded = model.decode(h_rollout)
    rollout_loss = torch.mean((decoded[:, 1:, -1] - truth[:, 1:, -1]) ** 2)

    recon = model.decode(model.encode(q_in))
    recon_loss = torch.mean((recon - q_in) ** 2)
    shifted = model.decode(_apply_k(k_matrix, model.encode(q_in)))
    shift_loss = torch.mean((shifted - q_out) ** 2)
    split_loss = torch.zeros((), dtype=DTYPE)
    if variant_cfg["lambda_split"] > 0:
        first_in, first_out = _prefix_pairs(positions, model.m, start_end=model.m - 1, num_transitions=OBS_PREFIX_TRANSITIONS // 2)
        second_in, second_out = _prefix_pairs(
            positions,
            model.m,
            start_end=model.m - 1 + OBS_PREFIX_TRANSITIONS // 2,
            num_transitions=OBS_PREFIX_TRANSITIONS // 2,
        )
        split_k, _, _ = model.identify_k(first_in, first_out)
        split_pred = model.decode(_apply_k(split_k, model.encode(second_in)))
        split_loss = torch.mean((split_pred - second_out) ** 2)

    loss = (
        rollout_loss
        + variant_cfg["lambda_recon"] * recon_loss
        + variant_cfg["lambda_shift"] * shift_loss
        + variant_cfg["lambda_split"] * split_loss
    )
    parts = {
        "window_reconstruction_mse": float(recon_loss.detach().item()),
        "shift_consistency_mse": float(shift_loss.detach().item()),
        "split_consistency_mse": float(split_loss.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }
    return loss, parts, k_matrix


def _linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    pred = x_aug @ coef
    denom = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    if denom == 0.0:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def _decoder_amplification(model: WindowLSModel, q: torch.Tensor) -> float:
    with torch.no_grad():
        h = model.encode(q).reshape(-1, model.h_dim)
        if h.shape[0] > 512:
            h = h[:512]
        generator = torch.Generator().manual_seed(SEED)
        eps = 1e-4 * torch.randn(h.shape, dtype=h.dtype, generator=generator)
        numerator = torch.linalg.norm(model.decode(h + eps) - model.decode(h), dim=1)
        denominator = torch.linalg.norm(eps, dim=1).clamp_min(1e-12)
        return float(torch.mean(numerator / denominator).item())


def _evaluate_learned(
    model: WindowLSModel,
    positions: torch.Tensor,
    variant: str,
    raw_mse: float,
    best_mse: float,
) -> dict[str, Any]:
    model.eval()
    cfg = VARIANTS[variant]
    with torch.no_grad():
        _, parts, k_matrix = _loss_parts(model, positions, "full_10_period", cfg)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        h0 = model.encode(truth[:, 0])
        decoded = model.decode(_rollout_matrix(k_matrix, h0, truth.shape[1] - 1))
        metrics = _metrics_from_rollout(decoded, truth)
        q_in, q_out = _prefix_pairs(positions, model.m)
        h_in = model.encode(q_in).reshape(-1, model.h_dim).detach().cpu().numpy()
        q_flat = q_in.reshape(-1, model.m).detach().cpu().numpy()
        two_state = q_in[..., -2:].reshape(-1, 2).detach().cpu().numpy() if model.m >= 2 else q_flat[:, -1:]
        probe_h_to_q = _linear_r2(h_in, q_flat)
        probe_h_to_two_state = _linear_r2(h_in, two_state)
        probe_q_to_h = _linear_r2(q_flat, h_in)
        amplification = _decoder_amplification(model, q_in)
    return {
        "model_type": "learned_window_ls",
        "m": model.m,
        "h_dim": model.h_dim,
        "variant": variant,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "best_10_period_rollout_mse": best_mse,
        "raw_window_ls_rollout_mse": raw_mse,
        "learned_minus_raw_gap": metrics["rollout_mse"] - raw_mse,
        "spectral_radius": _spectral_radius(k_matrix),
        "final_k_matrices": json.dumps(k_matrix.detach().cpu().tolist()),
        "probe_h_to_q_r2": probe_h_to_q,
        "probe_h_to_two_state_r2": probe_h_to_two_state,
        "probe_q_to_h_r2": probe_q_to_h,
        "decoder_amplification": amplification,
        **parts,
        **metrics,
    }


def _train_learned_config(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    m: int,
    h_dim: int,
    variant: str,
    raw_mse: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = WindowLSModel(m, h_dim).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    cfg = VARIANTS[variant]
    curves: list[dict[str, Any]] = []
    best_mse = float("inf")
    global_step = 0
    for stage in CURRICULUM_STAGES:
        for _ in range(STAGE_STEPS_BY_STAGE.get(stage, STAGE_STEPS)):
            global_step += 1
            optimizer.zero_grad()
            loss, _, _ = _loss_parts(model, train_positions, stage, cfg)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
        eval_row = _evaluate_learned(model, test_positions, variant, raw_mse, best_mse)
        best_mse = min(best_mse, eval_row["rollout_mse"])
        eval_row["best_10_period_rollout_mse"] = best_mse
        curves.append(
            {
                "m": m,
                "h_dim": h_dim,
                "variant": variant,
                "stage": stage,
                "global_step": global_step,
                "rollout_mse": eval_row["rollout_mse"],
                "best_10_period_rollout_mse": best_mse,
            }
        )
    final_row = _evaluate_learned(model, test_positions, variant, raw_mse, best_mse)
    return final_row, curves


def _plot_metric(rows: list[dict[str, Any]], metric: str, filename: str, ylabel: str, group: str) -> None:
    plt.figure(figsize=(9.0, 5.2))
    if group == "m":
        for model_type in sorted({row["model_type"] for row in rows}):
            selected = [row for row in rows if row["model_type"] == model_type]
            by_m = []
            for m in WINDOW_SIZES:
                vals = [float(row[metric]) for row in selected if int(row["m"]) == m]
                by_m.append(float(np.median(vals)) if vals else np.nan)
            plt.plot(WINDOW_SIZES, by_m, marker="o", label=model_type)
    else:
        for variant in sorted({row["variant"] for row in rows if row["model_type"] == "learned_window_ls"}):
            selected = [row for row in rows if row["variant"] == variant]
            by_m = []
            for m in WINDOW_SIZES:
                vals = [float(row[metric]) for row in selected if int(row["m"]) == m]
                by_m.append(float(np.median(vals)) if vals else np.nan)
            plt.plot(WINDOW_SIZES, by_m, marker="o", label=variant)
    plt.yscale("log")
    plt.xlabel("window size m")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_probe(rows: list[dict[str, Any]]) -> None:
    labels = ["h_to_q", "h_to_two_state", "q_to_h"]
    keys = ["probe_h_to_q_r2", "probe_h_to_two_state_r2", "probe_q_to_h_r2"]
    vals = [np.nanmedian([float(row.get(key, "nan")) for row in rows if row["model_type"] == "learned_window_ls"]) for key in keys]
    plt.figure(figsize=(7.0, 4.5))
    plt.bar(labels, vals)
    plt.ylabel("linear probe R2")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latent_probe_scores.png", dpi=160)
    plt.close()


def _plot_decoder_amp(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["model_type"] == "learned_window_ls"]
    labels = [f"m{row['m']}-h{row['h_dim']}-{row['variant']}" for row in selected]
    vals = [float(row["decoder_amplification"]) for row in selected]
    plt.figure(figsize=(12.0, 5.0))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=6)
    plt.ylabel("decoder amplification")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decoder_amplification.png", dpi=160)
    plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10.0, 5.4))
    for variant in sorted({row["variant"] for row in curves}):
        selected = [row for row in curves if row["variant"] == variant and row["m"] == 8 and row["h_dim"] == 4]
        if selected:
            plt.plot(range(len(selected)), [row["rollout_mse"] for row in selected], marker="o", label=f"{variant} m8 h4")
    plt.yscale("log")
    plt.xlabel("curriculum stage index")
    plt.ylabel("10-period rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160)
    plt.close()


def _write_html(rows: list[dict[str, Any]]) -> None:
    summary_cols = [
        "model_type",
        "m",
        "h_dim",
        "variant",
        "rollout_mse",
        "best_10_period_rollout_mse",
        "raw_window_ls_rollout_mse",
        "learned_minus_raw_gap",
        "one_step_mse",
        "two_step_mse",
        "window_reconstruction_mse",
        "shift_consistency_mse",
        "latent_transition_mse",
        "ls_condition_number",
        "decoder_amplification",
    ]
    top = sorted(rows, key=lambda row: float(row["rollout_mse"]))[:20]
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_fmt(row.get(col, ''))}</td>" for col in summary_cols) + "</tr>"
        for row in top
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Window Operator Prior Diagnostic</title>
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
  <h1>Stage 1B Window Operator Prior Diagnostic</h1>
  <p>Runtime-adjusted diagnostic: window sizes {escape(str(WINDOW_SIZES))}, h_dims {escape(str(H_DIMS))}, {STAGE_STEPS} steps per curriculum stage. K is always identified by LS; no MLP generates K.</p>
  <h2>Top Rows</h2>
  <table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in summary_cols)}</tr></thead><tbody>{body}</tbody></table>
  <div class="grid">
    <img src="raw_window_ls_by_window_size.png">
    <img src="learned_vs_raw_window_ls.png">
    <img src="rollout_mse_by_window_size.png">
    <img src="rollout_mse_by_prior_variant.png">
    <img src="latent_probe_scores.png">
    <img src="decoder_amplification.png">
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
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    setup = {
        "diagnostic": "stage1B_window_operator_prior_diagnostic",
        "runtime_adjusted": True,
        "requested_window_sizes": [2, 4, 8, 16, 32],
        "used_window_sizes": WINDOW_SIZES,
        "requested_h_dims": [2, 4, 8],
        "used_h_dims": H_DIMS,
        "stage_steps": STAGE_STEPS,
        "stage_steps_by_stage": {str(key): value for key, value in STAGE_STEPS_BY_STAGE.items()},
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels_for_training": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "k_generated_by_mlp": False,
        "ridge": RIDGE,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "curriculum_stages": CURRICULUM_STAGES,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    for m in WINDOW_SIZES:
        raw_row = _evaluate_raw_window_ls(test_positions, m)
        raw_by_m[m] = raw_row["rollout_mse"]
        rows.append(raw_row)

    for m in WINDOW_SIZES:
        for h_dim in H_DIMS:
            for variant in VARIANTS:
                print(f"training learned_window_ls m={m} h_dim={h_dim} variant={variant}", flush=True)
                final_row, curve_rows = _train_learned_config(train_positions, test_positions, m, h_dim, variant, raw_by_m[m])
                rows.append(final_row)
                curves.extend(curve_rows)
                _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
                _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curves)

    compact_rows = [{key: value for key, value in row.items() if key != "final_k_matrices"} for row in rows]
    _write_csv(OUTPUT_DIR / "summary.csv", compact_rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curves)
    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": compact_rows})

    _plot_metric(rows, "rollout_mse", "raw_window_ls_by_window_size.png", "rollout MSE", "m")
    _plot_metric(rows, "rollout_mse", "learned_vs_raw_window_ls.png", "rollout MSE", "m")
    _plot_metric(rows, "rollout_mse", "rollout_mse_by_window_size.png", "rollout MSE", "m")
    _plot_metric(rows, "rollout_mse", "rollout_mse_by_prior_variant.png", "rollout MSE", "variant")
    _plot_probe(rows)
    _plot_decoder_amp(rows)
    _plot_curves(curves)
    _write_html(compact_rows)


if __name__ == "__main__":
    main()
