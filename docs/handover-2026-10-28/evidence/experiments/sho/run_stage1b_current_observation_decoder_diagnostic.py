"""Stage 1B current-observation decoder diagnostic.

The encoder sees a position-only window q_t, but the decoder predicts only the
current position x_t. K_h is always identified by least squares from latent
transitions; no network generates K_h.
"""

from __future__ import annotations

import csv
import json
import os
import sys
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
    _evaluate_raw_window_ls,
    _fit_k_ridge,
    _linear_r2,
    _make_data,
    _prefix_pairs,
    _rollout_matrix,
    _rollout_truth,
    _spectral_radius,
    _window_sequence,
    _write_csv,
    _write_json,
)
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/current_observation_decoder_diagnostic")
WINDOW_SIZES = [2, 8, 32]
MU_VALUES = [0.0, 1e-1, 1.0]
TRAIN_NUM = 12
TEST_NUM = 8
LAMBDA_RECON = 1e-1
LAMBDA_SEGMENT = 1e-1
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
VARIANTS = {
    "current_rollout_only": {"lambda_recon": 0.0, "lambda_segment": 0.0},
    "current_rollout_recon": {"lambda_recon": LAMBDA_RECON, "lambda_segment": 0.0},
    "current_rollout_segment": {"lambda_recon": 0.0, "lambda_segment": LAMBDA_SEGMENT},
}


class CurrentDecoderModel(nn.Module):
    def __init__(self, m: int, h_dim: int, encoder_type: str) -> None:
        super().__init__()
        self.m = m
        self.h_dim = h_dim
        self.encoder_type = encoder_type
        if encoder_type == "linear_H_linear_d":
            self.encoder = nn.Linear(m, h_dim, bias=False)
            self.decoder = nn.Linear(h_dim, 1, bias=False)
            if h_dim == m:
                with torch.no_grad():
                    self.encoder.weight.copy_(torch.eye(m, dtype=DTYPE))
                    self.decoder.weight.zero_()
                    self.decoder.weight[0, -1] = 1.0
        elif encoder_type == "MLP_H_MLP_d":
            self.encoder = nn.Sequential(nn.Linear(m, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, h_dim))
            self.decoder = nn.Sequential(nn.Linear(h_dim, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, 1))
        else:
            raise ValueError(f"unknown encoder_type={encoder_type}")

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return self.encoder(q)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).squeeze(-1)

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


def _forward_segments(positions: torch.Tensor, m: int) -> torch.Tensor:
    start = m - 1
    segments = [positions[:, end : end + m] for end in range(start, start + 168)]
    return torch.stack(segments, dim=1)


def _rollout_x_metrics(predicted_x: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    true_x = truth[..., -1]
    error = predicted_x[:, 1:] - true_x[:, 1:]
    return {
        "rollout_mse": float(torch.mean(error**2).item()),
        "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
        "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()) if error.shape[1] > 1 else float("nan"),
        "max_abs_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_abs_error": float(torch.mean(torch.abs(error[:, -1])).item()),
    }


def _decoder_amplification_scalar(model: CurrentDecoderModel, q: torch.Tensor) -> float:
    with torch.no_grad():
        h = model.encode(q).reshape(-1, model.h_dim)
        if h.shape[0] > 512:
            h = h[:512]
        generator = torch.Generator().manual_seed(SEED)
        eps = 1e-4 * torch.randn(h.shape, dtype=h.dtype, generator=generator)
        numerator = torch.abs(model.decode_x(h + eps) - model.decode_x(h))
        denominator = torch.linalg.norm(eps, dim=1).clamp_min(1e-12)
        return float(torch.mean(numerator / denominator).item())


def _current_decoder_identity_diagnostics(model: CurrentDecoderModel) -> dict[str, Any]:
    if model.encoder_type != "linear_H_linear_d" or model.h_dim != model.m:
        return {
            "encoder_identity_error": "",
            "decoder_last_coordinate_error": "",
        }
    with torch.no_grad():
        eye = torch.eye(model.m, dtype=DTYPE)
        target = torch.zeros(1, model.m, dtype=DTYPE)
        target[0, -1] = 1.0
        return {
            "encoder_identity_error": float(torch.linalg.norm(model.encoder.weight.detach() - eye).item()),
            "decoder_last_coordinate_error": float(torch.linalg.norm(model.decoder.weight.detach() - target).item()),
        }


def _segment_loss(model: CurrentDecoderModel, q_in: torch.Tensor, k_h: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    h = model.encode(q_in)
    preds = []
    state = h
    for step in range(model.m):
        if step > 0:
            state = _apply_k(k_h, state)
        preds.append(model.decode_x(state))
    predicted = torch.stack(preds, dim=-1)
    return torch.mean((predicted - segments) ** 2)


def _loss_parts(
    model: CurrentDecoderModel,
    positions: torch.Tensor,
    stage: int | str,
    variant: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, latent_residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    h_rollout = _rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode_x(model.encode(q_in)) - q_in[..., -1]) ** 2)
    segments = _forward_segments(positions, model.m)
    segment_loss = _segment_loss(model, q_in, k_h, segments)
    cfg = VARIANTS[variant]
    loss = rollout_loss + cfg["lambda_recon"] * recon_loss + cfg["lambda_segment"] * segment_loss
    return loss, {
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "segment_reconstruction_mse": float(segment_loss.detach().item()),
        "latent_transition_mse": float(latent_residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }, k_h


def _evaluate(
    model: CurrentDecoderModel,
    positions: torch.Tensor,
    variant: str,
    mu: float,
    raw_mse: float,
    full_window_baseline: str,
    best_mse: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, k_h = _loss_parts(model, positions, "full_10_period", variant)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _rollout_x_metrics(pred_x, truth)
        q_in, _ = _prefix_pairs(positions, model.m)
        h_flat = model.encode(q_in).reshape(-1, model.h_dim).detach().cpu().numpy()
        q_flat = q_in.reshape(-1, model.m).detach().cpu().numpy()
        two_state = q_in[..., -2:].reshape(-1, 2).detach().cpu().numpy()
        x_now = q_in[..., -1:].reshape(-1, 1).detach().cpu().numpy()
    return {
        "model_type": "current_observation_decoder",
        "m": model.m,
        "h_dim": model.h_dim,
        "encoder_type": model.encoder_type,
        "variant": variant,
        "mu": mu,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "raw_window_ls_rollout_mse": raw_mse,
        "previous_full_window_decoder_baseline_mse": full_window_baseline,
        "gap_to_raw_window_ls_floor": metrics["rollout_mse"] - raw_mse,
        "best_rollout_mse": best_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "decoder_amplification": _decoder_amplification_scalar(model, q_in),
        "probe_h_to_q_r2": _linear_r2(h_flat, q_flat),
        "probe_h_to_two_state_r2": _linear_r2(h_flat, two_state),
        "probe_h_to_x_t_r2": _linear_r2(h_flat, x_now),
        **_current_decoder_identity_diagnostics(model),
        **parts,
        **metrics,
    }


def _train_one(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    m: int,
    h_dim: int,
    encoder_type: str,
    variant: str,
    mu: float,
    raw_mse: float,
    full_window_baseline: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = CurrentDecoderModel(m, h_dim, encoder_type).to(dtype=DTYPE)
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
            drift = float(_param_distance(model, prev_snapshot).item())
            penalty = float((mu * _param_distance(model, prev_snapshot)).item())
        prev_snapshot = best_stage_snapshot
        eval_row = _evaluate(model, test_positions, variant, mu, raw_mse, full_window_baseline, best_global)
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
            "biased_regularization_penalty": penalty,
            "parameter_drift_from_previous_stage": drift,
            "current_reconstruction_mse": eval_row["current_reconstruction_mse"],
            "segment_reconstruction_mse": eval_row["segment_reconstruction_mse"],
        })

    return _evaluate(model, test_positions, variant, mu, raw_mse, full_window_baseline, best_global), curves


def _load_full_window_baselines() -> dict[tuple[int, int], float]:
    path = Path("outputs/sho/stage1B/biased_regularized_window_capacity_diagnostic/summary.csv")
    if not path.exists():
        return {}
    baselines: dict[tuple[int, int], float] = {}
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row.get("model_type") == "raw_window_ls":
            continue
        try:
            key = (int(row["m"]), int(row["h_dim"]))
            value = float(row["rollout_mse"])
        except (ValueError, KeyError):
            continue
        if key not in baselines or value < baselines[key]:
            baselines[key] = value
    return baselines


def _configs() -> list[tuple[int, int, str]]:
    configs = []
    for m in WINDOW_SIZES:
        configs.append((m, 2, "MLP_H_MLP_d"))
        configs.append((m, 4, "MLP_H_MLP_d"))
        configs.append((m, m, "linear_H_linear_d"))
    return configs


def _plot_by_m_h(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    plt.figure(figsize=(10, 5))
    for encoder_type in sorted({row["encoder_type"] for row in learned}):
        selected = [row for row in learned if row["encoder_type"] == encoder_type]
        xs = [int(row["m"]) + 0.05 * int(row["h_dim"]) for row in selected]
        ys = [float(row["rollout_mse"]) for row in selected]
        plt.scatter(xs, ys, s=28, label=encoder_type, alpha=0.75)
    plt.yscale("log")
    plt.xlabel("window size m, jittered by h_dim")
    plt.ylabel("rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_m_hdim.png", dpi=160)
    plt.close()


def _plot_variant(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    labels = list(VARIANTS)
    vals = []
    for variant in labels:
        these = [float(row["rollout_mse"]) for row in learned if row["variant"] == variant]
        vals.append(float(np.median(these)) if these else np.nan)
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.yscale("log")
    plt.ylabel("median rollout MSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_loss_variant.png", dpi=160)
    plt.close()


def _plot_mu(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    vals = []
    for mu in MU_VALUES:
        these = [float(row["rollout_mse"]) for row in learned if float(row["mu"]) == mu]
        vals.append(float(np.median(these)) if these else np.nan)
    plt.figure(figsize=(7, 5))
    plt.plot(MU_VALUES, vals, marker="o")
    plt.xscale("symlog", linthresh=1e-4)
    plt.yscale("log")
    plt.xlabel("mu")
    plt.ylabel("median rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "biased_regularization_by_mu.png", dpi=160)
    plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for variant in VARIANTS:
        selected = [row for row in curves if row["m"] == 8 and row["h_dim"] == 4 and row["variant"] == variant and float(row["mu"]) == 0.1]
        if selected:
            plt.plot(range(len(selected)), [row["rollout_mse"] for row in selected], marker="o", label=variant)
    plt.yscale("log")
    plt.xlabel("curriculum stage")
    plt.ylabel("rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160)
    plt.close()


def _plot_segment(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    x = [max(float(row["segment_reconstruction_mse"]), 1e-30) for row in learned]
    y = [float(row["rollout_mse"]) for row in learned]
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=25)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("K-conditioned segment reconstruction MSE")
    plt.ylabel("rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_reconstruction_vs_rollout.png", dpi=160)
    plt.close()


def _plot_probes(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    keys = ["probe_h_to_q_r2", "probe_h_to_two_state_r2", "probe_h_to_x_t_r2"]
    labels = ["h -> q", "h -> two-state", "h -> x_t"]
    vals = [np.median([float(row[key]) for row in learned]) for key in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("linear probe R2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latent_probe_scores.png", dpi=160)
    plt.close()


def _plot_amp(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] != "raw_window_ls"]
    vals = [float(row["decoder_amplification"]) for row in learned]
    labels = [f"m{row['m']}-h{row['h_dim']}-{row['variant']}-mu{float(row['mu']):g}" for row in learned]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=85, ha="right", fontsize=4)
    plt.ylabel("decoder amplification")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decoder_amplification.png", dpi=160)
    plt.close()


def _plot_current_vs_full(rows: list[dict[str, Any]]) -> None:
    learned = [
        row for row in rows
        if row["model_type"] != "raw_window_ls" and row.get("previous_full_window_decoder_baseline_mse")
    ]
    plt.figure(figsize=(6, 5))
    if learned:
        plt.scatter(
            [float(row["previous_full_window_decoder_baseline_mse"]) for row in learned],
            [float(row["rollout_mse"]) for row in learned],
            s=26,
        )
        lo = min(min(float(row["previous_full_window_decoder_baseline_mse"]) for row in learned), min(float(row["rollout_mse"]) for row in learned))
        hi = max(max(float(row["previous_full_window_decoder_baseline_mse"]) for row in learned), max(float(row["rollout_mse"]) for row in learned))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("previous full-window decoder best MSE")
    plt.ylabel("current-position decoder MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "current_decoder_vs_full_window_decoder.png", dpi=160)
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
        "previous_full_window_decoder_baseline_mse",
        "gap_to_raw_window_ls_floor",
        "one_step_mse",
        "two_step_mse",
        "current_reconstruction_mse",
        "segment_reconstruction_mse",
        "latent_transition_mse",
        "ls_condition_number",
        "decoder_amplification",
        "probe_h_to_x_t_r2",
    ]
    top = sorted(rows, key=lambda row: float(row["rollout_mse"]))[:24]
    body = "\n".join("<tr>" + "".join(f"<td>{_fmt(row.get(col, ''))}</td>" for col in cols) + "</tr>" for row in top)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Current-Observation Decoder Diagnostic</title>
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
  <h1>Stage 1B Current-Observation Decoder Diagnostic</h1>
  <p>Runtime-adjusted screen. H sees q_t, decoder predicts only x_t, and K_h is always identified by latent LS. K_q/omega/coefficient/velocity/analytic recurrence are not used for learned rollout.</p>
  <h2>Top Rows</h2>
  <table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
  <div class="grid">
    <img src="current_decoder_vs_full_window_decoder.png">
    <img src="rollout_mse_by_m_hdim.png">
    <img src="rollout_mse_by_loss_variant.png">
    <img src="biased_regularization_by_mu.png">
    <img src="curriculum_stage_curves.png">
    <img src="segment_reconstruction_vs_rollout.png">
    <img src="latent_probe_scores.png">
    <img src="decoder_amplification.png">
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
        configs.append((m, 2, "MLP_H_MLP_d"))
        configs.append((m, 4, "MLP_H_MLP_d"))
        configs.append((m, m, "linear_H_linear_d"))
    return configs


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    full_window_baselines = _load_full_window_baselines()
    setup = {
        "diagnostic": "stage1B_current_observation_decoder_diagnostic",
        "runtime_adjusted": True,
        "window_sizes": WINDOW_SIZES,
        "configs": [{"m": m, "h_dim": h_dim, "encoder_type": encoder_type} for m, h_dim, encoder_type in _configs()],
        "mu_values": MU_VALUES,
        "variants": VARIANTS,
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "k_generated_by_mlp": False,
        "learned_rollout_uses_latent_ls_k_h": True,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "stage_steps": {str(key): value for key, value in STAGE_STEPS.items()},
        "lambda_recon": LAMBDA_RECON,
        "lambda_segment": LAMBDA_SEGMENT,
        "rollout_periods": ROLLOUT_PERIODS,
        "full_window_baseline_source": "outputs/sho/stage1B/biased_regularized_window_capacity_diagnostic/summary.csv if present",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    for m in WINDOW_SIZES:
        raw = _evaluate_raw_window_ls(test_positions, m)
        raw["encoder_type"] = ""
        raw["mu"] = ""
        raw["best_rollout_mse"] = raw["rollout_mse"]
        raw["raw_window_ls_rollout_mse"] = raw["rollout_mse"]
        raw["gap_to_raw_window_ls_floor"] = 0.0
        raw_by_m[m] = raw["rollout_mse"]
        rows.append(raw)

    for m, h_dim, encoder_type in _configs():
        full_baseline = full_window_baselines.get((m, h_dim), "")
        for variant in VARIANTS:
            for mu in MU_VALUES:
                print(f"training current_decoder m={m} h_dim={h_dim} encoder={encoder_type} variant={variant} mu={mu:g}", flush=True)
                final_row, curve_rows = _train_one(
                    train_positions,
                    test_positions,
                    m,
                    h_dim,
                    encoder_type,
                    variant,
                    mu,
                    raw_by_m[m],
                    "" if full_baseline == "" else str(full_baseline),
                )
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
    _plot_current_vs_full(rows)
    _plot_by_m_h(rows)
    _plot_variant(rows)
    _plot_mu(rows)
    _plot_curves(curves)
    _plot_segment(rows)
    _plot_probes(rows)
    _plot_amp(rows)
    _write_html(rows, setup)


if __name__ == "__main__":
    main()
