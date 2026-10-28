"""Stage 1B biased-regularized window-capacity diagnostic.

K_h is always identified by least squares from latent transitions. The
experiment tests whether capacity and stage-to-stage parameter bias improve
learned-window LS without using labels, velocity, analytic SHO structure, or an
MLP-generated operator.
"""

from __future__ import annotations

import csv
import json
import os
import time
from copy import deepcopy
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

from experiments.sho.run_stage1b_ls_teacher_conjugacy_diagnostic import (
    TEACHER_STEPS,
    _apply_power,
    _fit_raw_teacher,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    CURRICULUM_STAGES,
    DTYPE,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    RIDGE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _apply_k,
    _decoder_amplification,
    _evaluate_raw_window_ls,
    _fit_k_ridge,
    _linear_r2,
    _make_data,
    _metrics_from_rollout,
    _prefix_pairs,
    _rollout_matrix,
    _rollout_truth,
    _spectral_radius,
    _write_csv,
    _write_json,
)
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/biased_regularized_window_capacity_diagnostic")
WINDOW_SIZES = [2, 8, 32]
MU_VALUES = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
VARIANTS = {
    "rollout_only": {"lambda_recon": 0.0, "lambda_conj": 0.0},
    "rollout_recon": {"lambda_recon": 1e-1, "lambda_conj": 0.0},
}
LAMBDA_RECON = 1e-1
LAMBDA_CONJ = 1e-1
TRAIN_NUM = 12
TEST_NUM = 8
STAGE_STEPS = {
    1: 12,
    2: 12,
    4: 12,
    8: 12,
    16: 8,
    32: 5,
    64: 3,
    "full_10_period": 1,
}


class WindowCapacityModel(nn.Module):
    def __init__(self, m: int, h_dim: int, encoder_type: str) -> None:
        super().__init__()
        self.m = m
        self.h_dim = h_dim
        self.encoder_type = encoder_type
        if encoder_type == "linear_H_linear_D":
            self.encoder = nn.Linear(m, h_dim, bias=False)
            self.decoder = nn.Linear(h_dim, m, bias=False)
            if h_dim == m:
                with torch.no_grad():
                    self.encoder.weight.copy_(torch.eye(m, dtype=DTYPE))
                    self.decoder.weight.copy_(torch.eye(m, dtype=DTYPE))
        elif encoder_type == "MLP_H_MLP_D":
            self.encoder = nn.Sequential(nn.Linear(m, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, h_dim))
            self.decoder = nn.Sequential(nn.Linear(h_dim, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, m))
        else:
            raise ValueError(f"unknown encoder_type={encoder_type}")

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return self.encoder(q)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h)

    def identify_k(self, q_in: torch.Tensor, q_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fit_k_ridge(self.encode(q_in), self.encode(q_out), ridge=RIDGE)


def _param_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [param.detach().clone() for param in model.parameters()]


def _param_distance(model: nn.Module, snapshot: list[torch.Tensor] | None) -> torch.Tensor:
    if snapshot is None:
        return torch.zeros((), dtype=DTYPE)
    total = torch.zeros((), dtype=DTYPE)
    for param, old in zip(model.parameters(), snapshot):
        total = total + torch.sum((param - old.to(param.device)) ** 2)
    return total


def _linear_identity_diagnostics(model: WindowCapacityModel) -> dict[str, Any]:
    if model.encoder_type != "linear_H_linear_D" or model.h_dim != model.m:
        return {
            "decoder_encoder_identity_error": "",
            "encoder_decoder_identity_error": "",
        }
    with torch.no_grad():
        w_h = model.encoder.weight.detach()
        w_d = model.decoder.weight.detach()
        eye = torch.eye(model.m, dtype=w_h.dtype)
        return {
            "decoder_encoder_identity_error": float(torch.linalg.norm(w_d @ w_h - eye).item()),
            "encoder_decoder_identity_error": float(torch.linalg.norm(w_h @ w_d - eye).item()),
        }


def _conjugacy_loss(model: WindowCapacityModel, q_in: torch.Tensor, k_h: torch.Tensor, k_q: torch.Tensor) -> torch.Tensor:
    h_in = model.encode(q_in)
    pieces = []
    for step in TEACHER_STEPS:
        student = _apply_power(k_h, h_in, step)
        teacher_encoded = model.encode(_apply_power(k_q, q_in, step))
        pieces.append(torch.mean((student - teacher_encoded) ** 2))
    return torch.stack(pieces).mean()


def _loss_parts(
    model: WindowCapacityModel,
    positions: torch.Tensor,
    stage: int | str,
    variant: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, latent_residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    decoded = model.decode(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
    rollout_loss = torch.mean((decoded[:, 1:, -1] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode(model.encode(q_in)) - q_in) ** 2)
    conj_loss = torch.zeros((), dtype=DTYPE)
    if VARIANTS[variant]["lambda_conj"] > 0:
        k_q, _, _ = _fit_raw_teacher(positions, model.m)
        conj_loss = _conjugacy_loss(model, q_in, k_h, k_q)
    loss = (
        rollout_loss
        + VARIANTS[variant]["lambda_recon"] * recon_loss
        + VARIANTS[variant]["lambda_conj"] * conj_loss
    )
    return loss, {
        "window_reconstruction_mse": float(recon_loss.detach().item()),
        "conjugacy_loss": float(conj_loss.detach().item()),
        "latent_transition_mse": float(latent_residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }, k_h


def _evaluate(
    model: WindowCapacityModel,
    positions: torch.Tensor,
    variant: str,
    mu: float,
    raw_mse: float,
    best_mse: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, k_h = _loss_parts(model, positions, "full_10_period", variant)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        decoded = model.decode(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _metrics_from_rollout(decoded, truth)
        q_in, _ = _prefix_pairs(positions, model.m)
        h_flat = model.encode(q_in).reshape(-1, model.h_dim).detach().cpu().numpy()
        q_flat = q_in.reshape(-1, model.m).detach().cpu().numpy()
        two_state = q_in[..., -2:].reshape(-1, 2).detach().cpu().numpy()
    row = {
        "model_type": "learned_window_ls_biased_regularized",
        "m": model.m,
        "h_dim": model.h_dim,
        "encoder_type": model.encoder_type,
        "variant": variant,
        "mu": mu,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "raw_window_ls_rollout_mse": raw_mse,
        "gap_to_raw_window_ls_floor": metrics["rollout_mse"] - raw_mse,
        "best_rollout_mse": best_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "decoder_amplification": _decoder_amplification(model, q_in),
        "probe_h_to_q_r2": _linear_r2(h_flat, q_flat),
        "probe_h_to_two_state_r2": _linear_r2(h_flat, two_state),
        **_linear_identity_diagnostics(model),
        **parts,
        **metrics,
    }
    return row


def _train_one(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    m: int,
    h_dim: int,
    encoder_type: str,
    variant: str,
    mu: float,
    raw_mse: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = WindowCapacityModel(m, h_dim, encoder_type).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    prev_snapshot: list[torch.Tensor] | None = None
    curves: list[dict[str, Any]] = []
    best_global = float("inf")
    global_step = 0

    for stage in CURRICULUM_STAGES:
        best_stage_loss = float("inf")
        best_stage_snapshot = _param_snapshot(model)
        for _ in range(STAGE_STEPS.get(stage, 1)):
            global_step += 1
            optimizer.zero_grad()
            stage_loss, _, _ = _loss_parts(model, train_positions, stage, variant)
            distance = _param_distance(model, prev_snapshot)
            loss = stage_loss + mu * distance
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            loss_value = float(stage_loss.detach().item())
            if loss_value < best_stage_loss:
                best_stage_loss = loss_value
                best_stage_snapshot = _param_snapshot(model)

        with torch.no_grad():
            stage_distance = float(_param_distance(model, prev_snapshot).item())
            stage_penalty = float((mu * _param_distance(model, prev_snapshot)).item())
        prev_snapshot = best_stage_snapshot
        eval_row = _evaluate(model, test_positions, variant, mu, raw_mse, best_global)
        best_global = min(best_global, eval_row["rollout_mse"])
        curves.append({
            "m": m,
            "h_dim": h_dim,
            "encoder_type": encoder_type,
            "variant": variant,
            "mu": mu,
            "stage": stage,
            "global_step": global_step,
            "rollout_mse": eval_row["rollout_mse"],
            "best_rollout_mse": best_global,
            "stage_train_loss": best_stage_loss,
            "biased_regularization_penalty": stage_penalty,
            "parameter_distance_from_previous_stage": stage_distance,
        })

    return _evaluate(model, test_positions, variant, mu, raw_mse, best_global), curves


def _plot_capacity(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    plt.figure(figsize=(9, 5))
    for encoder_type in sorted({row["encoder_type"] for row in learned}):
        selected = [row for row in learned if row["encoder_type"] == encoder_type and row["mu"] == 0.0]
        labels = []
        vals = []
        for row in selected:
            labels.append(f"m{row['m']}-h{row['h_dim']}-{row['variant']}")
            vals.append(float(row["gap_to_raw_window_ls_floor"]))
        if vals:
            plt.scatter(range(len(vals)), vals, label=encoder_type)
    plt.yscale("log")
    plt.ylabel("gap to raw LS floor")
    plt.xticks([])
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "capacity_gap_by_hdim.png", dpi=160)
    plt.close()


def _plot_mu(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    plt.figure(figsize=(8, 5))
    for variant in VARIANTS:
        medians = []
        for mu in MU_VALUES:
            vals = [float(row["rollout_mse"]) for row in learned if row["variant"] == variant and float(row["mu"]) == mu]
            medians.append(float(np.median(vals)) if vals else np.nan)
        plt.plot(MU_VALUES, medians, marker="o", label=variant)
    plt.xscale("symlog", linthresh=1e-4)
    plt.yscale("log")
    plt.xlabel("mu")
    plt.ylabel("median rollout MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "biased_regularization_by_mu.png", dpi=160)
    plt.close()


def _plot_stage(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for mu in [0.0, 1e-2, 1.0]:
        selected = [row for row in curves if row["m"] == 8 and row["h_dim"] == 4 and row["variant"] == "rollout_recon" and float(row["mu"]) == mu]
        if selected:
            plt.plot(range(len(selected)), [row["rollout_mse"] for row in selected], marker="o", label=f"mu={mu:g}")
    plt.yscale("log")
    plt.xlabel("curriculum stage")
    plt.ylabel("rollout MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_stage.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for mu in [0.0, 1e-2, 1.0]:
        selected = [row for row in curves if row["m"] == 8 and row["h_dim"] == 4 and row["variant"] == "rollout_recon" and float(row["mu"]) == mu]
        if selected:
            plt.plot(range(len(selected)), [max(float(row["parameter_distance_from_previous_stage"]), 1e-30) for row in selected], marker="o", label=f"mu={mu:g}")
    plt.yscale("log")
    plt.xlabel("curriculum stage")
    plt.ylabel("parameter distance from previous stage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_drift_by_stage.png", dpi=160)
    plt.close()


def _plot_raw_vs_learned(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for m in WINDOW_SIZES:
        raw = [float(row["rollout_mse"]) for row in rows if row["model_type"] == "raw_window_ls" and int(row["m"]) == m][0]
        best = min(float(row["rollout_mse"]) for row in rows if row["model_type"] != "raw_window_ls" and int(row["m"]) == m)
        plt.scatter([m], [raw], marker="x", s=80, color="black")
        plt.scatter([m], [best], marker="o", s=70, color="tab:blue")
    plt.yscale("log")
    plt.xlabel("window size m")
    plt.ylabel("10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "raw_vs_learned_ls_floor.png", dpi=160)
    plt.close()


def _plot_linear_diag(rows: list[dict[str, Any]]) -> None:
    linear = [row for row in rows if row.get("encoder_type") == "linear_H_linear_D"]
    labels = [f"m{row['m']}-{row['variant']}-mu{float(row['mu']):g}" for row in linear]
    vals = [float(row["decoder_encoder_identity_error"]) for row in linear]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=80, ha="right", fontsize=5)
    plt.ylabel("||W_D W_H - I||")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_hdim_m_diagnostics.png", dpi=160)
    plt.close()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    cols = [
        "model_type",
        "m",
        "h_dim",
        "encoder_type",
        "variant",
        "mu",
        "rollout_mse",
        "best_rollout_mse",
        "raw_window_ls_rollout_mse",
        "gap_to_raw_window_ls_floor",
        "one_step_mse",
        "two_step_mse",
        "window_reconstruction_mse",
        "latent_transition_mse",
        "ls_condition_number",
        "decoder_amplification",
        "probe_h_to_q_r2",
        "decoder_encoder_identity_error",
    ]
    top = sorted(rows, key=lambda row: float(row["rollout_mse"]))[:24]
    body = "\n".join("<tr>" + "".join(f"<td>{_fmt(row.get(col, ''))}</td>" for col in cols) + "</tr>" for row in top)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Biased-Regularized Window Capacity Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
  <h1>Stage 1B Biased-Regularized Window Capacity Diagnostic</h1>
  <p>Runtime-adjusted screen. K_h is identified by latent least squares for every learned model; no MLP-generated K, no labels, no velocity, no analytic recurrence, and no analytic decoder are used.</p>
  <p>Biased regularization penalizes trainable encoder/decoder parameter drift from the previous curriculum stage snapshot. K_h is not regularized because it is solved by LS.</p>
  <h2>Top Rows</h2>
  <table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
  <div class="grid">
    <img src="capacity_gap_by_hdim.png">
    <img src="biased_regularization_by_mu.png">
    <img src="rollout_mse_by_stage.png">
    <img src="parameter_drift_by_stage.png">
    <img src="raw_vs_learned_ls_floor.png">
    <img src="linear_hdim_m_diagnostics.png">
  </div>
  <h2>Configuration</h2>
  <pre>{escape(json.dumps(setup, indent=2))}</pre>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _configs() -> list[tuple[int, int, str]]:
    configs = []
    for m in WINDOW_SIZES:
        configs.append((m, 2, "MLP_H_MLP_D"))
        configs.append((m, 4, "MLP_H_MLP_D"))
        configs.append((m, m, "linear_H_linear_D"))
    return configs


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    setup = {
        "diagnostic": "stage1B_biased_regularized_window_capacity_diagnostic",
        "runtime_adjusted": True,
        "requested_window_sizes": [2, 8, 16, 32],
        "used_window_sizes": WINDOW_SIZES,
        "requested_h_dims": "2, 4, m",
        "used_configs": [{"m": m, "h_dim": h_dim, "encoder_type": encoder_type} for m, h_dim, encoder_type in _configs()],
        "mu_values": MU_VALUES,
        "variants": VARIANTS,
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels_for_training": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence_for_training": False,
        "k_generated_by_mlp": False,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "stage_steps": {str(key): value for key, value in STAGE_STEPS.items()},
        "lambda_recon": LAMBDA_RECON,
        "lambda_conj": LAMBDA_CONJ,
        "rollout_periods": ROLLOUT_PERIODS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    for m in WINDOW_SIZES:
        raw = _evaluate_raw_window_ls(test_positions, m)
        raw_by_m[m] = raw["rollout_mse"]
        raw["encoder_type"] = ""
        raw["mu"] = ""
        raw["best_rollout_mse"] = raw["rollout_mse"]
        raw["raw_window_ls_rollout_mse"] = raw["rollout_mse"]
        raw["gap_to_raw_window_ls_floor"] = 0.0
        rows.append(raw)

    for m, h_dim, encoder_type in _configs():
        for variant in VARIANTS:
            for mu in MU_VALUES:
                print(f"training biased_capacity m={m} h_dim={h_dim} encoder={encoder_type} variant={variant} mu={mu:g}", flush=True)
                final_row, curve_rows = _train_one(train_positions, test_positions, m, h_dim, encoder_type, variant, mu, raw_by_m[m])
                rows.append(final_row)
                curves.extend(curve_rows)
                _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
                _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)

    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": rows, "curves": curves})

    _plot_capacity(rows)
    _plot_mu(rows)
    _plot_stage(curves)
    _plot_raw_vs_learned(rows)
    _plot_linear_diag(rows)
    _write_html(rows, setup)


if __name__ == "__main__":
    main()
