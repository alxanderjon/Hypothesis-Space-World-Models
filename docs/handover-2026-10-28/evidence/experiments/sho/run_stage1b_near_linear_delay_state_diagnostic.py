"""Stage 1B near-linear delay-state LS diagnostic.

K_h is always identified by least squares from latent transitions. The point is
to test whether successful learned representations must stay close to linear
delay coordinates.
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
    _write_csv,
    _write_json,
)
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/near_linear_delay_state_diagnostic")
WINDOW_SIZES = [2, 8, 32]
MU_VALUES = [0.0, 1e-2, 1e-1, 1.0]
LINEAR_H_DIMS = [2, 4]
RESIDUAL_EPS = [1e-2, 1e-1]
LAMBDA_RES = 1e-2
LAMBDA_RECON = 1e-1
VARIANTS = {
    "rollout_only": {"lambda_recon": 0.0},
    "rollout_recon": {"lambda_recon": LAMBDA_RECON},
}
TRAIN_NUM = 12
TEST_NUM = 8
STAGE_STEPS = {
    1: 10,
    2: 10,
    4: 10,
    8: 10,
    16: 6,
    32: 4,
    64: 2,
    "full_10_period": 1,
}


def _selector_matrix(m: int, h_dim: int) -> torch.Tensor:
    weight = torch.zeros(h_dim, m, dtype=DTYPE)
    weight[0, -2] = 1.0
    weight[1, -1] = 1.0
    if h_dim > 2:
        generator = torch.Generator().manual_seed(SEED + m + h_dim)
        weight[2:] = 1e-3 * torch.randn(h_dim - 2, m, dtype=DTYPE, generator=generator)
    return weight


class DelayStateModel(nn.Module):
    def __init__(
        self,
        m: int,
        h_dim: int,
        model_type: str,
        init_kind: str,
        eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.m = m
        self.h_dim = h_dim
        self.model_type = model_type
        self.init_kind = init_kind
        self.eps = eps
        self.linear = nn.Linear(m, h_dim, bias=False)
        self.decoder = nn.Linear(h_dim, 1, bias=False)
        self.residual = nn.Sequential(nn.Linear(m, HIDDEN_WIDTH), nn.SiLU(), nn.Linear(HIDDEN_WIDTH, h_dim))
        self._initialize()

    def _initialize(self) -> None:
        with torch.no_grad():
            if self.init_kind == "delay_selector":
                self.linear.weight.copy_(_selector_matrix(self.m, self.h_dim))
                self.decoder.weight.zero_()
                self.decoder.weight[0, 1] = 1.0
            elif self.init_kind == "random":
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
                nn.init.normal_(self.decoder.weight, mean=0.0, std=0.05)
            else:
                raise ValueError(f"unknown init_kind={self.init_kind}")
            for module in self.residual.modules():
                if isinstance(module, nn.Linear):
                    module.weight.mul_(0.0)
                    if module.bias is not None:
                        module.bias.zero_()

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        base = self.linear(q)
        if self.model_type == "near_linear_residual":
            return base + self.eps * self.residual(q)
        return base

    def residual_component(self, q: torch.Tensor) -> torch.Tensor:
        if self.model_type != "near_linear_residual":
            return torch.zeros((*q.shape[:-1], self.h_dim), dtype=q.dtype, device=q.device)
        return self.eps * self.residual(q)

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


def _rollout_x_metrics(pred_x: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    error = pred_x[:, 1:] - truth[:, 1:, -1]
    return {
        "rollout_mse": float(torch.mean(error**2).item()),
        "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
        "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()) if error.shape[1] > 1 else float("nan"),
        "max_abs_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_abs_error": float(torch.mean(torch.abs(error[:, -1])).item()),
    }


def _fixed_delay_rollout(positions: torch.Tensor, m: int) -> dict[str, Any]:
    q_in, q_out = _prefix_pairs(positions, m)
    h_in = q_in[..., -2:]
    h_out = q_out[..., -2:]
    k_h, residual, conditions = _fit_k_ridge(h_in, h_out, ridge=1e-12)
    truth = _rollout_truth(positions, m, "full_10_period")
    h0 = truth[:, 0, -2:]
    rollout = _rollout_matrix(k_h, h0, truth.shape[1] - 1)
    pred_x = rollout[..., 1]
    row = {
        "model_type": "fixed_delay_selector",
        "m": m,
        "h_dim": 2,
        "init_kind": "fixed_delay_selector",
        "variant": "closed_form_ls",
        "mu": "",
        "eps": "",
        "lambda_res": "",
        "parameter_count": 0,
        "latent_transition_mse": float(residual.item()),
        "ls_condition_number": float(conditions.mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
        "fixed_delay_rollout_mse": "",
        "residual_norm_ratio": "",
        "nonlinear_residual_mse": "",
        "distance_from_selector_A": 0.0,
        "learned_A": "",
    }
    row.update(_rollout_x_metrics(pred_x, truth))
    row["fixed_delay_rollout_mse"] = row["rollout_mse"]
    return row


def _residual_stats(model: DelayStateModel, q_in: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        base = model.linear(q_in)
        residual = model.residual_component(q_in)
        ratio = torch.mean(torch.linalg.norm(residual, dim=-1)) / torch.mean(torch.linalg.norm(base, dim=-1)).clamp_min(1e-12)
        h = model.encode(q_in).reshape(-1, model.h_dim)
        q = q_in.reshape(-1, model.m)
        q_aug = torch.cat([q, torch.ones(q.shape[0], 1, dtype=q.dtype)], dim=1)
        coef = torch.linalg.lstsq(q_aug, h).solution
        pred = q_aug @ coef
        nonlinear_mse = torch.mean((h - pred) ** 2)
        selector = _selector_matrix(model.m, model.h_dim)
        selector_distance = torch.linalg.norm(model.linear.weight.detach() - selector)
        return {
            "residual_norm_ratio": float(ratio.item()),
            "nonlinear_residual_mse": float(nonlinear_mse.item()),
            "distance_from_selector_A": float(selector_distance.item()),
        }


def _loss_parts(
    model: DelayStateModel,
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
    residual = model.residual_component(q_in)
    residual_loss = torch.mean(residual**2)
    loss = rollout_loss + VARIANTS[variant]["lambda_recon"] * recon_loss
    if model.model_type == "near_linear_residual":
        loss = loss + LAMBDA_RES * residual_loss
    parts = {
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "latent_transition_mse": float(latent_residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
        "residual_penalty_mse": float(residual_loss.detach().item()),
    }
    return loss, parts, k_h


def _evaluate(
    model: DelayStateModel,
    positions: torch.Tensor,
    variant: str,
    mu: float,
    raw_mse: float,
    fixed_delay_mse: float,
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
    return {
        "model_type": model.model_type,
        "m": model.m,
        "h_dim": model.h_dim,
        "init_kind": model.init_kind,
        "variant": variant,
        "mu": mu,
        "eps": model.eps if model.model_type == "near_linear_residual" else "",
        "lambda_res": LAMBDA_RES if model.model_type == "near_linear_residual" else "",
        "parameter_count": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_delay_mse,
        "gap_to_raw_window_ls_floor": metrics["rollout_mse"] - raw_mse,
        "gap_to_fixed_delay_floor": metrics["rollout_mse"] - fixed_delay_mse,
        "best_rollout_mse": best_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "probe_h_to_two_state_r2": _linear_r2(h_flat, two_state),
        "probe_h_to_q_r2": _linear_r2(h_flat, q_flat),
        "probe_q_to_h_r2": _linear_r2(q_flat, h_flat),
        "learned_A": json.dumps(model.linear.weight.detach().cpu().tolist()),
        **_residual_stats(model, q_in),
        **parts,
        **metrics,
    }


def _train_one(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    m: int,
    h_dim: int,
    model_type: str,
    init_kind: str,
    eps: float,
    variant: str,
    mu: float,
    raw_mse: float,
    fixed_delay_mse: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = DelayStateModel(m, h_dim, model_type, init_kind, eps=eps).to(dtype=DTYPE)
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
            stage_value = float(stage_loss.detach().item())
            if stage_value < best_stage_loss:
                best_stage_loss = stage_value
                best_stage_snapshot = _param_snapshot(model)
        with torch.no_grad():
            drift = float(_param_distance(model, prev_snapshot).item())
            penalty = float((mu * _param_distance(model, prev_snapshot)).item())
        prev_snapshot = best_stage_snapshot
        eval_row = _evaluate(model, test_positions, variant, mu, raw_mse, fixed_delay_mse, best_global)
        best_global = min(best_global, eval_row["rollout_mse"])
        curves.append({
            "m": m,
            "h_dim": h_dim,
            "model_type": model_type,
            "init_kind": init_kind,
            "variant": variant,
            "mu": mu,
            "eps": eps if model_type == "near_linear_residual" else "",
            "stage": stage,
            "global_step": global_step,
            "rollout_mse": eval_row["rollout_mse"],
            "best_rollout_mse": best_global,
            "stage_train_loss": best_stage_loss,
            "biased_regularization_penalty": penalty,
            "parameter_drift_from_previous_stage": drift,
            "residual_norm_ratio": eval_row["residual_norm_ratio"],
            "distance_from_selector_A": eval_row["distance_from_selector_A"],
        })
    return _evaluate(model, test_positions, variant, mu, raw_mse, fixed_delay_mse, best_global), curves


def _configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for m in WINDOW_SIZES:
        for h_dim in LINEAR_H_DIMS:
            for init_kind in ["random", "delay_selector"]:
                configs.append({"m": m, "h_dim": h_dim, "model_type": "trainable_linear", "init_kind": init_kind, "eps": 0.0})
        for eps in RESIDUAL_EPS:
            configs.append({"m": m, "h_dim": 2, "model_type": "near_linear_residual", "init_kind": "delay_selector", "eps": eps})
    return configs


def _plot_raw_fixed(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 5))
    for m in WINDOW_SIZES:
        raw = [float(row["rollout_mse"]) for row in rows if row["model_type"] == "raw_window_ls" and int(row["m"]) == m][0]
        fixed = [float(row["rollout_mse"]) for row in rows if row["model_type"] == "fixed_delay_selector" and int(row["m"]) == m][0]
        plt.scatter([m], [raw], marker="x", color="black", s=80)
        plt.scatter([m], [fixed], marker="o", color="tab:blue", s=70)
    plt.yscale("log")
    plt.xlabel("window size m")
    plt.ylabel("rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "raw_vs_fixed_delay_floor.png", dpi=160)
    plt.close()


def _plot_model_type(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    labels = sorted({row["model_type"] for row in learned})
    vals = []
    for label in labels:
        xs = [float(row["rollout_mse"]) for row in learned if row["model_type"] == label]
        vals.append(float(np.median(xs)) if xs else np.nan)
    plt.figure(figsize=(7, 5))
    plt.bar(labels, vals)
    plt.yscale("log")
    plt.ylabel("median rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_model_type.png", dpi=160)
    plt.close()


def _plot_mu(rows: list[dict[str, Any]]) -> None:
    learned = [row for row in rows if row["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    vals = []
    for mu in MU_VALUES:
        xs = [float(row["rollout_mse"]) for row in learned if float(row["mu"]) == mu]
        vals.append(float(np.median(xs)) if xs else np.nan)
    plt.figure(figsize=(7, 5))
    plt.plot(MU_VALUES, vals, marker="o")
    plt.xscale("symlog", linthresh=1e-4)
    plt.yscale("log")
    plt.xlabel("mu")
    plt.ylabel("median rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_mu.png", dpi=160)
    plt.close()


def _plot_eps(rows: list[dict[str, Any]]) -> None:
    residual = [row for row in rows if row["model_type"] == "near_linear_residual"]
    vals = []
    labels = []
    for eps in RESIDUAL_EPS:
        xs = [float(row["rollout_mse"]) for row in residual if float(row["eps"]) == eps]
        vals.append(float(np.median(xs)) if xs else np.nan)
        labels.append(str(eps))
    plt.figure(figsize=(6, 5))
    plt.bar(labels, vals)
    plt.yscale("log")
    plt.xlabel("residual scale eps")
    plt.ylabel("median rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_residual_scale.png", dpi=160)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], x_key: str, filename: str, xlabel: str) -> None:
    learned = [row for row in rows if row["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    x = [max(float(row[x_key]), 1e-30) for row in learned]
    y = [float(row["rollout_mse"]) for row in learned]
    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=24)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    selectors = [
        ("trainable_linear", "delay_selector", 2, 0.1),
        ("trainable_linear", "random", 2, 0.1),
        ("near_linear_residual", "delay_selector", 2, 0.1),
    ]
    for model_type, init_kind, h_dim, mu in selectors:
        selected = [
            row for row in curves
            if row["m"] == 8 and row["h_dim"] == h_dim and row["model_type"] == model_type
            and row["init_kind"] == init_kind and float(row["mu"]) == mu and row["variant"] == "rollout_recon"
        ]
        if selected:
            plt.plot(range(len(selected)), [row["rollout_mse"] for row in selected], marker="o", label=f"{model_type}/{init_kind}")
    plt.yscale("log")
    plt.xlabel("curriculum stage")
    plt.ylabel("rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for model_type, init_kind, h_dim, mu in selectors:
        selected = [
            row for row in curves
            if row["m"] == 8 and row["h_dim"] == h_dim and row["model_type"] == model_type
            and row["init_kind"] == init_kind and float(row["mu"]) == mu and row["variant"] == "rollout_recon"
        ]
        if selected:
            plt.plot(range(len(selected)), [max(float(row["parameter_drift_from_previous_stage"]), 1e-30) for row in selected], marker="o", label=f"{model_type}/{init_kind}")
    plt.yscale("log")
    plt.xlabel("curriculum stage")
    plt.ylabel("parameter drift")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_drift_by_stage.png", dpi=160)
    plt.close()


def _plot_a_heatmaps(rows: list[dict[str, Any]]) -> None:
    learned = [
        row for row in rows
        if row["model_type"] in {"trainable_linear", "near_linear_residual"} and row["init_kind"] == "delay_selector"
    ]
    best_by_m: list[dict[str, Any]] = []
    for m in WINDOW_SIZES:
        candidates = [row for row in learned if int(row["m"]) == m and int(row["h_dim"]) == 2]
        if candidates:
            best_by_m.append(min(candidates, key=lambda row: float(row["rollout_mse"])))
    if not best_by_m:
        return
    fig, axes = plt.subplots(1, len(best_by_m), figsize=(4 * len(best_by_m), 3))
    if len(best_by_m) == 1:
        axes = [axes]
    for ax, row in zip(axes, best_by_m):
        matrix = np.array(json.loads(row["learned_A"]), dtype=float)
        im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_title(f"m={row['m']} {row['model_type']}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learned_A_heatmaps.png", dpi=160)
    plt.close()


def _plot_linear_vs_previous(rows: list[dict[str, Any]]) -> None:
    current_path = Path("outputs/sho/stage1B/current_observation_decoder_diagnostic/summary.csv")
    prev_points: list[tuple[int, int, float]] = []
    if current_path.exists():
        csv.field_size_limit(sys.maxsize)
        with current_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("model_type") == "current_observation_decoder" and row.get("encoder_type") == "MLP_H_MLP_d":
                    try:
                        prev_points.append((int(row["m"]), int(row["h_dim"]), float(row["rollout_mse"])))
                    except ValueError:
                        pass
    learned = [row for row in rows if row["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    plt.figure(figsize=(7, 5))
    if prev_points:
        plt.scatter([p[0] for p in prev_points], [p[2] for p in prev_points], s=20, alpha=0.4, label="previous MLP current decoder")
    linear = [row for row in learned if row["model_type"] == "trainable_linear"]
    plt.scatter([int(row["m"]) for row in linear], [float(row["rollout_mse"]) for row in linear], s=24, label="near-linear diagnostic linear")
    plt.yscale("log")
    plt.xlabel("m")
    plt.ylabel("rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_vs_mlp_previous_results.png", dpi=160)
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
        "init_kind",
        "variant",
        "mu",
        "eps",
        "rollout_mse",
        "raw_window_ls_rollout_mse",
        "fixed_delay_rollout_mse",
        "gap_to_fixed_delay_floor",
        "one_step_mse",
        "two_step_mse",
        "current_reconstruction_mse",
        "latent_transition_mse",
        "ls_condition_number",
        "residual_norm_ratio",
        "nonlinear_residual_mse",
        "distance_from_selector_A",
    ]
    top = sorted(rows, key=lambda row: float(row["rollout_mse"]))[:24]
    body = "\n".join("<tr>" + "".join(f"<td>{_fmt(row.get(col, ''))}</td>" for col in cols) + "</tr>" for row in top)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B Near-Linear Delay-State LS Diagnostic</title>
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
  <h1>Stage 1B Near-Linear Delay-State LS Diagnostic</h1>
  <p>Runtime-adjusted screen. Evaluation uses H, latent LS K_h, K_h rollout, and scalar decoder d only. No labels, velocity, analytic decoder, analytic recurrence, raw-window teacher, or MLP-generated K are used for learned models.</p>
  <h2>Top Rows</h2>
  <table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
  <div class="grid">
    <img src="raw_vs_fixed_delay_floor.png">
    <img src="linear_vs_mlp_previous_results.png">
    <img src="rollout_mse_by_model_type.png">
    <img src="rollout_mse_by_mu.png">
    <img src="rollout_mse_by_residual_scale.png">
    <img src="residual_norm_vs_rollout_mse.png">
    <img src="nonlinearity_vs_rollout_mse.png">
    <img src="curriculum_stage_curves.png">
    <img src="learned_A_heatmaps.png">
    <img src="parameter_drift_by_stage.png">
  </div>
  <h2>Configuration</h2>
  <pre>{escape(json.dumps(setup, indent=2))}</pre>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    setup = {
        "diagnostic": "stage1B_near_linear_delay_state_diagnostic",
        "runtime_adjusted": True,
        "window_sizes": WINDOW_SIZES,
        "linear_h_dims": LINEAR_H_DIMS,
        "linear_init_kinds": ["random", "delay_selector"],
        "residual_h_dim": 2,
        "residual_eps": RESIDUAL_EPS,
        "lambda_res": LAMBDA_RES,
        "mu_values": MU_VALUES,
        "variants": VARIANTS,
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence_in_training": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "stage_steps": {str(key): value for key, value in STAGE_STEPS.items()},
        "rollout_periods": ROLLOUT_PERIODS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    fixed_by_m: dict[int, float] = {}
    for m in WINDOW_SIZES:
        raw = _evaluate_raw_window_ls(test_positions, m)
        raw["model_type"] = "raw_window_ls"
        raw["fixed_delay_rollout_mse"] = ""
        raw["gap_to_raw_window_ls_floor"] = 0.0
        raw_by_m[m] = raw["rollout_mse"]
        rows.append(raw)
        fixed = _fixed_delay_rollout(test_positions, m)
        fixed["raw_window_ls_rollout_mse"] = raw_by_m[m]
        fixed["gap_to_raw_window_ls_floor"] = fixed["rollout_mse"] - raw_by_m[m]
        fixed_by_m[m] = fixed["rollout_mse"]
        rows.append(fixed)

    for cfg in _configs():
        for variant in VARIANTS:
            for mu in MU_VALUES:
                print(
                    "training near_linear "
                    f"m={cfg['m']} h_dim={cfg['h_dim']} model={cfg['model_type']} "
                    f"init={cfg['init_kind']} eps={cfg['eps']} variant={variant} mu={mu:g}",
                    flush=True,
                )
                row, curve_rows = _train_one(
                    train_positions,
                    test_positions,
                    int(cfg["m"]),
                    int(cfg["h_dim"]),
                    str(cfg["model_type"]),
                    str(cfg["init_kind"]),
                    float(cfg["eps"]),
                    variant,
                    mu,
                    raw_by_m[int(cfg["m"])],
                    fixed_by_m[int(cfg["m"])],
                )
                rows.append(row)
                curves.extend(curve_rows)
                _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
                _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)

    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": rows, "curves": curves})

    _plot_raw_fixed(rows)
    _plot_linear_vs_previous(rows)
    _plot_model_type(rows)
    _plot_mu(rows)
    _plot_eps(rows)
    _plot_scatter(rows, "residual_norm_ratio", "residual_norm_vs_rollout_mse.png", "residual norm ratio")
    _plot_scatter(rows, "nonlinear_residual_mse", "nonlinearity_vs_rollout_mse.png", "nonlinear residual MSE")
    _plot_curves(curves)
    _plot_a_heatmaps(rows)
    _write_html(rows, setup)


if __name__ == "__main__":
    main()
