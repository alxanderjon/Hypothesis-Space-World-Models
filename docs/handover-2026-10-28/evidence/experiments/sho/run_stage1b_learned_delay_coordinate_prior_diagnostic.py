"""Stage 1B learned delay-coordinate prior diagnostic.

Tests whether structured priors can discover delay-coordinate states from
position windows while keeping K_h identified by latent least squares.
"""

from __future__ import annotations

import csv
import json
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

from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    CURRICULUM_STAGES,
    DTYPE,
    RIDGE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_delay_coordinate_prior_diagnostic")
WINDOW_SIZES = [2, 8, 32]
TRAIN_NUM = 8
TEST_NUM = 6
MU_VALUES = [0.0, 1e-2, 1e-1, 1.0]
ENTROPY_LAMBDAS = [0.0, 1e-2, 1e-1]
SPARSE_LAMBDAS = [1e-3, 1e-2]
LAMBDA_RECON = 1e-1
LAMBDA_NORM = 1e-1
LAMBDA_OVERLAP = 1e-2
LAMBDA_ORTH = 1e-2
VARIANTS = {
    "rollout_only": {"lambda_recon": 0.0},
    "rollout_recon": {"lambda_recon": LAMBDA_RECON},
}
DISCOVERY_STAGES: list[int | str] = [1, 2, 4, 8]
CURRICULUM_HARD_STAGES: list[int | str] = [16, 32, 64, "full_10_period"]
STAGE_STEPS = {1: 6, 2: 6, 4: 6, 8: 6, 16: 4, 32: 3, 64: 2, "full_10_period": 1}
LEARNING_RATE = 1e-3


def _selector_matrix(m: int) -> torch.Tensor:
    a = torch.zeros(2, m, dtype=DTYPE)
    a[0, -2] = 1.0
    a[1, -1] = 1.0
    return a


class DelayPriorModel(nn.Module):
    def __init__(
        self,
        m: int,
        model_type: str,
        init_kind: str,
        lambda_entropy: float = 0.0,
        lambda_sparse: float = 0.0,
    ) -> None:
        super().__init__()
        self.m = m
        self.h_dim = 2
        self.model_type = model_type
        self.init_kind = init_kind
        self.lambda_entropy = lambda_entropy
        self.lambda_sparse = lambda_sparse
        self.decoder = nn.Linear(2, 1, bias=False)
        if model_type == "soft_delay_selector":
            self.alpha = nn.Parameter(torch.zeros(max(m - 1, 1), dtype=DTYPE))
            self.A = None
        elif model_type == "sparse_linear":
            self.A = nn.Parameter(torch.empty(2, m, dtype=DTYPE))
            self.alpha = None
        else:
            raise ValueError(f"unknown model_type={model_type}")
        self._initialize()

    def _initialize(self) -> None:
        generator = torch.Generator().manual_seed(SEED + self.m)
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.weight[0, 1] = 1.0
            if self.model_type == "soft_delay_selector":
                if self.init_kind == "random_alpha":
                    self.alpha.copy_(0.01 * torch.randn(self.alpha.shape, dtype=DTYPE, generator=generator))
                elif self.init_kind == "uniform_alpha":
                    self.alpha.zero_()
                else:
                    raise ValueError(f"bad soft init={self.init_kind}")
            else:
                if self.init_kind == "random":
                    self.A.copy_(0.05 * torch.randn(self.A.shape, dtype=DTYPE, generator=generator))
                elif self.init_kind == "noisy_delay":
                    self.A.copy_(_selector_matrix(self.m) + 0.01 * torch.randn(self.A.shape, dtype=DTYPE, generator=generator))
                else:
                    raise ValueError(f"bad sparse init={self.init_kind}")

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        if self.model_type == "soft_delay_selector":
            pi = torch.softmax(self.alpha, dim=0)
            past = q[..., :-1]
            delayed = torch.sum(past * pi, dim=-1)
            return torch.stack((delayed, q[..., -1]), dim=-1)
        return torch.einsum("ij,btj->bti", self.A, q) if q.ndim == 3 else q @ self.A.t()

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).squeeze(-1)

    def identify_k(self, q_in: torch.Tensor, q_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fit_k_ridge(self.encode(q_in), self.encode(q_out), ridge=RIDGE)

    def regularizer(self) -> tuple[torch.Tensor, dict[str, float]]:
        if self.model_type == "soft_delay_selector":
            pi = torch.softmax(self.alpha, dim=0)
            entropy = -torch.sum(pi * torch.log(pi.clamp_min(1e-30)))
            return self.lambda_entropy * entropy, {"entropy": float(entropy.detach().item())}
        assert self.A is not None
        row_norm = torch.sum((torch.linalg.norm(self.A, dim=1) - 1.0) ** 2)
        overlap = torch.sum(torch.abs(self.A[0]) * torch.abs(self.A[1]))
        orth = torch.sum(self.A[0] * self.A[1]) ** 2
        l1 = torch.sum(torch.abs(self.A))
        reg = self.lambda_sparse * l1 + LAMBDA_NORM * row_norm + LAMBDA_OVERLAP * overlap + LAMBDA_ORTH * orth
        return reg, {
            "sparse_l1": float(l1.detach().item()),
            "row_norm_penalty": float(row_norm.detach().item()),
            "row_overlap": float(overlap.detach().item()),
            "row_dot": float(torch.sum(self.A[0] * self.A[1]).detach().item()),
        }


def _param_snapshot(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _param_distance(model: nn.Module, snapshot: list[torch.Tensor] | None) -> torch.Tensor:
    if snapshot is None:
        return torch.zeros((), dtype=DTYPE)
    total = torch.zeros((), dtype=DTYPE)
    for p, old in zip(model.parameters(), snapshot):
        total = total + torch.sum((p - old.to(p.device)) ** 2)
    return total


def _fixed_delay_rollout(positions: torch.Tensor, m: int) -> dict[str, Any]:
    q_in, q_out = _prefix_pairs(positions, m)
    h_in, h_out = q_in[..., -2:], q_out[..., -2:]
    k_h, residual, conditions = _fit_k_ridge(h_in, h_out, ridge=1e-12)
    truth = _rollout_truth(positions, m, "full_10_period")
    rollout = _rollout_matrix(k_h, truth[:, 0, -2:], truth.shape[1] - 1)
    pred_x = rollout[..., 1]
    row = {
        "model_type": "fixed_delay_selector",
        "m": m,
        "h_dim": 2,
        "variant": "closed_form_ls",
        "init_kind": "fixed_delay_selector",
        "mu": "",
        "latent_transition_mse": float(residual.item()),
        "ls_condition_number": float(conditions.mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
    }
    row.update(_rollout_metrics(pred_x, truth))
    return row


def _rollout_metrics(pred_x: torch.Tensor, truth: torch.Tensor) -> dict[str, Any]:
    error = pred_x[:, 1:] - truth[:, 1:, -1]
    return {
        "rollout_mse": float(torch.mean(error**2).item()),
        "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
        "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()) if error.shape[1] > 1 else float("nan"),
        "max_abs_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_abs_error": float(torch.mean(torch.abs(error[:, -1])).item()),
    }


def _loss_parts(model: DelayPriorModel, positions: torch.Tensor, stage: int | str, variant: str) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode_x(model.encode(q_in)) - q_in[..., -1]) ** 2)
    reg, reg_parts = model.regularizer()
    loss = rollout_loss + VARIANTS[variant]["lambda_recon"] * recon_loss + reg
    parts = {
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
        "regularizer_value": float(reg.detach().item()),
        **reg_parts,
    }
    return loss, parts, k_h


def _diagnostics(model: DelayPriorModel, q_in: torch.Tensor) -> dict[str, Any]:
    h = model.encode(q_in).reshape(-1, 2).detach().cpu().numpy()
    q = q_in.reshape(-1, model.m).detach().cpu().numpy()
    two_state = q_in[..., -2:].reshape(-1, 2).detach().cpu().numpy()
    row: dict[str, Any] = {
        "probe_h_to_two_state_r2": _linear_r2(h, two_state),
        "probe_h_to_q_r2": _linear_r2(h, q),
        "probe_q_to_h_r2": _linear_r2(q, h),
    }
    if model.model_type == "soft_delay_selector":
        pi = torch.softmax(model.alpha.detach(), dim=0).cpu().numpy()
        selected = int(np.argmax(pi)) + 1
        entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-300))))
        row.update({
            "pi_values": json.dumps(pi.tolist()),
            "entropy": entropy,
            "selected_lag": selected,
            "max_pi": float(np.max(pi)),
            "selected_lag_is_1": selected == 1,
        })
    else:
        assert model.A is not None
        a = model.A.detach().cpu()
        row.update({
            "learned_A": json.dumps(a.tolist()),
            "row_argmax_coordinates": json.dumps(torch.argmax(torch.abs(a), dim=1).tolist()),
            "row_l1_norms": json.dumps(torch.sum(torch.abs(a), dim=1).tolist()),
            "sparse_l1": float(torch.sum(torch.abs(a)).item()),
            "row_overlap": float(torch.sum(torch.abs(a[0]) * torch.abs(a[1])).item()),
            "row_dot": float(torch.sum(a[0] * a[1]).item()),
            "distance_from_delay_selector_A": float(torch.linalg.norm(a - _selector_matrix(model.m)).item()),
        })
    return row


def _evaluate(model: DelayPriorModel, positions: torch.Tensor, variant: str, mu: float, raw_mse: float, fixed_mse: float, best_mse: float) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, k_h = _loss_parts(model, positions, "full_10_period", variant)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _rollout_metrics(pred_x, truth)
        q_in, _ = _prefix_pairs(positions, model.m)
    return {
        "model_type": model.model_type,
        "m": model.m,
        "h_dim": 2,
        "variant": variant,
        "init_kind": model.init_kind,
        "mu": mu,
        "lambda_entropy": model.lambda_entropy if model.model_type == "soft_delay_selector" else "",
        "lambda_sparse": model.lambda_sparse if model.model_type == "sparse_linear" else "",
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
        "gap_to_fixed_delay_floor": metrics["rollout_mse"] - fixed_mse,
        "best_rollout_mse": best_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        **parts,
        **_diagnostics(model, q_in),
        **metrics,
    }


def _train_one(train_positions: torch.Tensor, test_positions: torch.Tensor, model: DelayPriorModel, variant: str, mu: float, raw_mse: float, fixed_mse: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    prev_snapshot: list[torch.Tensor] | None = None
    curves: list[dict[str, Any]] = []
    best = float("inf")
    global_step = 0
    for stage in DISCOVERY_STAGES + CURRICULUM_HARD_STAGES:
        best_stage = float("inf")
        best_snapshot = _param_snapshot(model)
        for _ in range(STAGE_STEPS[stage]):
            global_step += 1
            optimizer.zero_grad()
            stage_loss, _, _ = _loss_parts(model, train_positions, stage, variant)
            bias = torch.zeros((), dtype=DTYPE) if stage in DISCOVERY_STAGES else mu * _param_distance(model, prev_snapshot)
            loss = stage_loss + bias
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            val = float(stage_loss.detach().item())
            if val < best_stage:
                best_stage = val
                best_snapshot = _param_snapshot(model)
        drift = float(_param_distance(model, prev_snapshot).detach().item())
        penalty = 0.0 if stage in DISCOVERY_STAGES else float((mu * _param_distance(model, prev_snapshot)).detach().item())
        prev_snapshot = best_snapshot
        row = _evaluate(model, test_positions, variant, mu, raw_mse, fixed_mse, best)
        best = min(best, row["rollout_mse"])
        curves.append({
            "m": model.m,
            "model_type": model.model_type,
            "variant": variant,
            "init_kind": model.init_kind,
            "mu": mu,
            "lambda_entropy": model.lambda_entropy if model.model_type == "soft_delay_selector" else "",
            "lambda_sparse": model.lambda_sparse if model.model_type == "sparse_linear" else "",
            "stage": stage,
            "phase": "discovery" if stage in DISCOVERY_STAGES else "biased_curriculum",
            "global_step": global_step,
            "rollout_mse": row["rollout_mse"],
            "best_rollout_mse": best,
            "stage_train_loss": best_stage,
            "biased_regularization_penalty": penalty,
            "parameter_drift_from_previous_stage": drift,
            "selected_lag": row.get("selected_lag", ""),
            "max_pi": row.get("max_pi", ""),
        })
    return _evaluate(model, test_positions, variant, mu, raw_mse, fixed_mse, best), curves


def _configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for m in WINDOW_SIZES:
        for init in ["uniform_alpha", "random_alpha"]:
            for lam in ENTROPY_LAMBDAS:
                configs.append({"m": m, "model_type": "soft_delay_selector", "init_kind": init, "lambda_entropy": lam, "lambda_sparse": 0.0})
        for init in ["random", "noisy_delay"]:
            for lam in SPARSE_LAMBDAS:
                configs.append({"m": m, "model_type": "sparse_linear", "init_kind": init, "lambda_entropy": 0.0, "lambda_sparse": lam})
    return configs


def _plot_raw_fixed(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 5))
    for m in WINDOW_SIZES:
        raw = [float(r["rollout_mse"]) for r in rows if r["model_type"] == "raw_window_ls" and int(r["m"]) == m][0]
        fixed = [float(r["rollout_mse"]) for r in rows if r["model_type"] == "fixed_delay_selector" and int(r["m"]) == m][0]
        plt.scatter([m], [raw], marker="x", s=80, color="black")
        plt.scatter([m], [fixed], marker="o", s=70, color="tab:blue")
    plt.yscale("log"); plt.xlabel("m"); plt.ylabel("rollout MSE"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "raw_vs_fixed_delay_floor.png", dpi=160); plt.close()


def _plot_soft_weights(rows: list[dict[str, Any]]) -> None:
    soft = [r for r in rows if r["model_type"] == "soft_delay_selector"]
    best = []
    for m in WINDOW_SIZES:
        c = [r for r in soft if int(r["m"]) == m]
        if c:
            best.append(min(c, key=lambda r: float(r["rollout_mse"])))
    if not best:
        return
    fig, axes = plt.subplots(len(best), 1, figsize=(8, 2.5 * len(best)))
    if len(best) == 1:
        axes = [axes]
    for ax, row in zip(axes, best):
        pi = np.array(json.loads(row["pi_values"]))
        ax.bar(np.arange(1, len(pi) + 1), pi)
        ax.set_title(f"m={row['m']} best lag={row['selected_lag']} MSE={float(row['rollout_mse']):.3g}")
        ax.set_xlabel("lag")
        ax.set_ylabel("pi")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "soft_selector_lag_weights.png", dpi=160); plt.close()


def _plot_sparse_heatmaps(rows: list[dict[str, Any]]) -> None:
    sparse = [r for r in rows if r["model_type"] == "sparse_linear"]
    best = []
    for m in WINDOW_SIZES:
        c = [r for r in sparse if int(r["m"]) == m]
        if c:
            best.append(min(c, key=lambda r: float(r["rollout_mse"])))
    if not best:
        return
    fig, axes = plt.subplots(1, len(best), figsize=(4 * len(best), 3))
    if len(best) == 1:
        axes = [axes]
    for ax, row in zip(axes, best):
        mat = np.array(json.loads(row["learned_A"]))
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
        ax.set_title(f"m={row['m']} {row['init_kind']}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "sparse_A_heatmaps.png", dpi=160); plt.close()


def _bar_by(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    learned = [r for r in rows if r["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    labels = sorted({str(r[key]) for r in learned if str(r.get(key, "")) != ""})
    vals = []
    for label in labels:
        xs = [float(r["rollout_mse"]) for r in learned if str(r.get(key, "")) == label]
        vals.append(float(np.median(xs)) if xs else np.nan)
    plt.figure(figsize=(7, 5)); plt.bar(labels, vals); plt.yscale("log"); plt.ylabel("median rollout MSE")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(OUTPUT_DIR / filename, dpi=160); plt.close()


def _scatter(rows: list[dict[str, Any]], xkey: str, filename: str, xlabel: str) -> None:
    learned = [r for r in rows if r["model_type"] not in {"raw_window_ls", "fixed_delay_selector"} and str(r.get(xkey, "")) != ""]
    x = [max(float(r[xkey]), 1e-30) for r in learned]
    y = [float(r["rollout_mse"]) for r in learned]
    plt.figure(figsize=(6, 5)); plt.scatter(x, y, s=24); plt.xscale("log"); plt.yscale("log")
    plt.xlabel(xlabel); plt.ylabel("rollout MSE"); plt.tight_layout(); plt.savefig(OUTPUT_DIR / filename, dpi=160); plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for model_type in ["soft_delay_selector", "sparse_linear"]:
        sel = [r for r in curves if int(r["m"]) == 8 and r["model_type"] == model_type and r["variant"] == "rollout_recon" and float(r["mu"]) == 0.1]
        if sel:
            # first matching config only
            first = sel[0]
            seq = [r for r in sel if r["init_kind"] == first["init_kind"] and r["lambda_entropy"] == first["lambda_entropy"] and r["lambda_sparse"] == first["lambda_sparse"]]
            plt.plot(range(len(seq)), [r["rollout_mse"] for r in seq], marker="o", label=model_type)
    plt.yscale("log"); plt.xlabel("stage"); plt.ylabel("rollout MSE"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160); plt.close()

    plt.figure(figsize=(9, 5))
    for model_type in ["soft_delay_selector", "sparse_linear"]:
        sel = [r for r in curves if int(r["m"]) == 8 and r["model_type"] == model_type and r["variant"] == "rollout_recon" and float(r["mu"]) == 0.1]
        if sel:
            first = sel[0]
            seq = [r for r in sel if r["init_kind"] == first["init_kind"] and r["lambda_entropy"] == first["lambda_entropy"] and r["lambda_sparse"] == first["lambda_sparse"]]
            plt.plot(range(len(seq)), [max(float(r["parameter_drift_from_previous_stage"]), 1e-30) for r in seq], marker="o", label=model_type)
    plt.yscale("log"); plt.xlabel("stage"); plt.ylabel("parameter drift"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "parameter_drift_by_stage.png", dpi=160); plt.close()


def _success_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    learned = [r for r in rows if r["model_type"] not in {"raw_window_ls", "fixed_delay_selector"}]
    for r in learned:
        out.append({
            "model_type": r["model_type"],
            "m": r["m"],
            "variant": r["variant"],
            "init_kind": r["init_kind"],
            "mu": r["mu"],
            "lambda_entropy": r.get("lambda_entropy", ""),
            "lambda_sparse": r.get("lambda_sparse", ""),
            "rollout_mse": r["rollout_mse"],
            "selected_lag": r.get("selected_lag", ""),
            "max_pi": r.get("max_pi", ""),
            "selected_lag_is_1": r.get("selected_lag_is_1", ""),
            "row_argmax_coordinates": r.get("row_argmax_coordinates", ""),
            "gap_to_fixed_delay_floor": r.get("gap_to_fixed_delay_floor", ""),
        })
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return escape(str(v))


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    cols = ["model_type", "m", "variant", "init_kind", "mu", "lambda_entropy", "lambda_sparse", "rollout_mse", "fixed_delay_rollout_mse", "selected_lag", "max_pi", "entropy", "row_argmax_coordinates", "sparse_l1", "row_overlap", "ls_condition_number"]
    top = sorted(rows, key=lambda r: float(r["rollout_mse"]))[:30]
    body = "\n".join("<tr>" + "".join(f"<td>{_fmt(r.get(c, ''))}</td>" for c in cols) + "</tr>" for r in top)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stage 1B Learned Delay Coordinate Prior</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}</style></head><body>
<h1>Stage 1B Learned Delay-Coordinate Prior Diagnostic</h1>
<p>Runtime-adjusted screen. Evaluation uses H, latent LS K_h, K_h rollout, and scalar decoder d only. No labels, velocity, analytic recurrence, analytic decoder, raw-window teacher, or MLP-generated K are used.</p>
<table><thead><tr>{''.join(f'<th>{escape(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="raw_vs_fixed_delay_floor.png"><img src="soft_selector_lag_weights.png"><img src="soft_selector_entropy_vs_rollout.png"><img src="sparse_A_heatmaps.png"><img src="sparse_A_selected_coordinates.png"><img src="rollout_mse_by_model_type.png"><img src="rollout_mse_by_mu.png"><img src="rollout_mse_by_sparsity_lambda.png"><img src="curriculum_stage_curves.png"><img src="parameter_drift_by_stage.png">
</div><h2>Configuration</h2><pre>{escape(json.dumps(setup, indent=2))}</pre></body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def _plot_sparse_coords(rows: list[dict[str, Any]]) -> None:
    sparse = [r for r in rows if r["model_type"] == "sparse_linear"]
    coords = []
    labels = []
    for r in sparse:
        coords.extend(json.loads(r["row_argmax_coordinates"]))
        labels.extend([int(r["m"])] * 2)
    plt.figure(figsize=(7, 5))
    if coords:
        plt.scatter(labels, coords, s=18, alpha=0.6)
    plt.xlabel("m"); plt.ylabel("selected coordinate index"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sparse_A_selected_coordinates.png", dpi=160); plt.close()


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    setup = {
        "diagnostic": "stage1B_learned_delay_coordinate_prior_diagnostic",
        "runtime_adjusted": True,
        "window_sizes": WINDOW_SIZES,
        "entropy_lambdas": ENTROPY_LAMBDAS,
        "sparse_lambdas": SPARSE_LAMBDAS,
        "mu_values": MU_VALUES,
        "variants": VARIANTS,
        "discovery_stages": DISCOVERY_STAGES,
        "biased_curriculum_stages": CURRICULUM_HARD_STAGES,
        "stage_steps": {str(k): v for k, v in STAGE_STEPS.items()},
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    fixed_by_m: dict[int, float] = {}
    for m in WINDOW_SIZES:
        raw = _evaluate_raw_window_ls(test_positions, m)
        raw["model_type"] = "raw_window_ls"
        raw_by_m[m] = raw["rollout_mse"]
        rows.append(raw)
        fixed = _fixed_delay_rollout(test_positions, m)
        fixed["raw_window_ls_rollout_mse"] = raw_by_m[m]
        fixed["fixed_delay_rollout_mse"] = fixed["rollout_mse"]
        fixed_by_m[m] = fixed["rollout_mse"]
        rows.append(fixed)

    for cfg in _configs():
        for variant in VARIANTS:
            for mu in MU_VALUES:
                print(f"training learned_delay m={cfg['m']} type={cfg['model_type']} init={cfg['init_kind']} ent={cfg['lambda_entropy']} sparse={cfg['lambda_sparse']} variant={variant} mu={mu:g}", flush=True)
                set_seed(SEED)
                model = DelayPriorModel(
                    int(cfg["m"]),
                    str(cfg["model_type"]),
                    str(cfg["init_kind"]),
                    lambda_entropy=float(cfg["lambda_entropy"]),
                    lambda_sparse=float(cfg["lambda_sparse"]),
                ).to(dtype=DTYPE)
                row, curve = _train_one(train_positions, test_positions, model, variant, mu, raw_by_m[int(cfg["m"])], fixed_by_m[int(cfg["m"])])
                rows.append(row)
                curves.extend(curve)
                _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
                _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)

    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_csv(OUTPUT_DIR / "summary.csv", rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", curves)
    _write_csv(OUTPUT_DIR / "delay_discovery_success_table.csv", _success_table(rows))
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": rows, "curves": curves})

    _plot_raw_fixed(rows)
    _plot_soft_weights(rows)
    _scatter(rows, "entropy", "soft_selector_entropy_vs_rollout.png", "soft-selector entropy")
    _plot_sparse_heatmaps(rows)
    _plot_sparse_coords(rows)
    _bar_by(rows, "model_type", "rollout_mse_by_model_type.png")
    _bar_by(rows, "mu", "rollout_mse_by_mu.png")
    _bar_by(rows, "lambda_sparse", "rollout_mse_by_sparsity_lambda.png")
    _plot_curves(curves)
    _write_html(rows, setup)


if __name__ == "__main__":
    main()
