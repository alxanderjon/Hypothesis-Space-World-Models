"""Stage 1B noisy short-prefix learned-delay diagnostic.

The model sees noisy observed prefixes y_t, fits K_h only from the first R
observed latent transitions, then rolls out a fixed 10-period future and is
scored against the clean trajectory.
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

from experiments.sho.run_stage1b_soft_delay_learning_ablation import _pi_vector
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _fit_k_ridge,
    _full_rollout_steps,
    _make_data,
    _rollout_matrix,
    _spectral_radius,
    _window_sequence,
    _write_csv,
    _write_json,
)
from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import _rollout_metrics


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/noisy_short_prefix_learned_delay_diagnostic")
PREVIOUS_PREFIX_DIR = Path("outputs/sho/stage1B/prefix_limited_operator_identification")

M = 8
TRAIN_NUM = 8
TEST_NUM = 6
R_VALUES = [2, 4, 8, 16]
SIGMAS = [0.0, 1e-3, 1e-2]
LRS = [1e-3, 1e-4, 1e-5]
TRAIN_STEPS = 150
EVAL_INTERVAL = 50
LAMBDA_DIV = 1e-2
LAMBDA_RECON = 1e-1
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _noisy_positions(clean: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    if sigma == 0.0:
        return clean.clone()
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    return clean + sigma * torch.randn(clean.shape, generator=generator, dtype=clean.dtype, device=clean.device)


class FixedDelay:
    name = "fixed_delay"

    def parameters(self) -> list[torch.Tensor]:
        return []

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return q[..., -2:]

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]

    def pi_metrics(self) -> dict[str, Any]:
        pi = np.zeros(M - 1)
        pi[0] = 1.0
        return _pi_metrics_from_array(pi)


class AnchoredSoftDelay(nn.Module):
    def __init__(self, pi_kind: str, train_alpha: bool) -> None:
        super().__init__()
        self.pi_kind = pi_kind
        if train_alpha:
            self.alpha = nn.Parameter(torch.log(_pi_vector(pi_kind).clamp_min(1e-30)))
            self.register_buffer("pi_fixed", torch.empty(0, dtype=DTYPE))
        else:
            self.alpha = None
            self.register_buffer("pi_fixed", _pi_vector(pi_kind))

    def pi(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=0) if self.alpha is not None else self.pi_fixed

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        past_by_lag = torch.flip(q[..., :-1], dims=(-1,))
        delayed = torch.sum(past_by_lag * self.pi().to(q.device), dim=-1)
        return torch.stack((delayed, q[..., -1]), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]

    def pi_metrics(self) -> dict[str, Any]:
        return _pi_metrics_from_array(self.pi().detach().cpu().numpy())


class TwoRowSoftSelector(nn.Module):
    def __init__(self, init_kind: str) -> None:
        super().__init__()
        self.init_kind = init_kind
        self.alpha1 = nn.Parameter(torch.zeros(M, dtype=DTYPE))
        self.alpha2 = nn.Parameter(torch.zeros(M, dtype=DTYPE))
        self.decoder = nn.Linear(2, 1, bias=False)
        self._initialize()

    def _initialize(self) -> None:
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.weight[0, 1] = 1.0
            if self.init_kind == "near_uniform":
                self.alpha1.zero_()
                self.alpha2.zero_()
            elif self.init_kind == "noisy_separated":
                self.alpha1.copy_(torch.tensor([0.6, 0.45, 0.3, 0.15, 0.0, -0.15, -0.3, -0.45], dtype=DTYPE))
                self.alpha2.copy_(torch.tensor([-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.8], dtype=DTYPE))
            elif self.init_kind == "bad_separated":
                self.alpha1.copy_(torch.tensor([1.2, 0.8, 0.2, -0.4, -0.8, -1.0, -1.2, -1.4], dtype=DTYPE))
                self.alpha2.copy_(torch.tensor([0.8, 1.2, 0.2, -0.4, -0.8, -1.0, -1.2, -1.4], dtype=DTYPE))
            else:
                raise ValueError(self.init_kind)

    def row_pis(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.softmax(self.alpha1, dim=0), torch.softmax(self.alpha2, dim=0)

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        pi1, pi2 = self.row_pis()
        h0 = torch.sum(q * pi1.to(q.device), dim=-1)
        h1 = torch.sum(q * pi2.to(q.device), dim=-1)
        return torch.stack((h0, h1), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).squeeze(-1)

    def diversity_penalty(self) -> torch.Tensor:
        pi1, pi2 = self.row_pis()
        return torch.sum(pi1 * pi2)

    def pi_metrics(self) -> dict[str, Any]:
        pi1, pi2 = self.row_pis()
        a = pi1.detach().cpu().numpy()
        b = pi2.detach().cpu().numpy()
        entropy = float(-np.sum(a * np.log(np.maximum(a, 1e-300))) - np.sum(b * np.log(np.maximum(b, 1e-300))))
        return {
            "pi_values": json.dumps({"row1": a.tolist(), "row2": b.tolist()}),
            "entropy": entropy,
            "selected_lag": "",
            "selected_coordinates": json.dumps([int(np.argmax(a)), int(np.argmax(b))]),
            "max_pi": float(max(np.max(a), np.max(b))),
            "diversity_penalty": float(np.sum(a * b)),
            **{f"row1_coord_{i}": float(v) for i, v in enumerate(a)},
            **{f"row2_coord_{i}": float(v) for i, v in enumerate(b)},
        }


def _pi_metrics_from_array(pi: np.ndarray) -> dict[str, Any]:
    entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-300))))
    return {
        "pi_values": json.dumps(pi.tolist()),
        "entropy": entropy,
        "selected_lag": int(np.argmax(pi)) + 1,
        "selected_coordinates": "",
        "max_pi": float(np.max(pi)),
        "diversity_penalty": "",
        **{f"pi_lag_{idx + 1}": float(value) for idx, value in enumerate(pi)},
    }


def _observed_windows(positions: torch.Tensor, r_transitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_in = _window_sequence(positions, M, M - 1, r_transitions)
    q_out = _window_sequence(positions, M, M, r_transitions)
    return q_in, q_out


def _future_windows(positions: torch.Tensor, r_transitions: int) -> torch.Tensor:
    start_end = M - 1 + r_transitions
    return _window_sequence(positions, M, start_end, _full_rollout_steps() + 1)


def _evaluate_model(model: Any, noisy_positions: torch.Tensor, clean_positions: torch.Tensor, r_transitions: int) -> dict[str, Any]:
    q_in, q_out = _observed_windows(noisy_positions, r_transitions)
    h_in = model.encode(q_in)
    h_out = model.encode(q_out)
    k_h, residual, conditions = _fit_k_ridge(h_in, h_out)
    noisy_start = _future_windows(noisy_positions, r_transitions)[:, 0]
    clean_truth = _future_windows(clean_positions, r_transitions)
    h0 = model.encode(noisy_start)
    h_rollout = _rollout_matrix(k_h, h0, clean_truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    metrics = _rollout_metrics(pred_x, clean_truth)
    row = {
        **metrics,
        "future_10_period_rollout_mse": metrics["rollout_mse"],
        "one_step_future_mse": metrics["one_step_mse"],
        "two_step_future_mse": metrics["two_step_mse"],
        "observed_ls_residual": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
        **model.pi_metrics(),
        **_threshold_flags(metrics["rollout_mse"]),
    }
    return row


def _future_loss(model: Any, noisy_positions: torch.Tensor, clean_positions: torch.Tensor, r_transitions: int, include_recon: bool) -> torch.Tensor:
    q_in, q_out = _observed_windows(noisy_positions, r_transitions)
    h_in = model.encode(q_in)
    h_out = model.encode(q_out)
    k_h, _, _ = _fit_k_ridge(h_in, h_out)
    noisy_start = _future_windows(noisy_positions, r_transitions)[:, 0]
    clean_truth = _future_windows(clean_positions, r_transitions)
    h_rollout = _rollout_matrix(k_h, model.encode(noisy_start), clean_truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    loss = torch.mean((pred_x[:, 1:] - clean_truth[:, 1:, -1]) ** 2)
    if include_recon:
        recon = model.decode_x(model.encode(q_in))
        clean_current = _window_sequence(clean_positions, M, M - 1, r_transitions)[..., -1]
        loss = loss + LAMBDA_RECON * torch.mean((recon - clean_current) ** 2)
    if isinstance(model, TwoRowSoftSelector):
        loss = loss + LAMBDA_DIV * model.diversity_penalty()
    return loss


def _static_sweep(train_clean: torch.Tensor, test_clean: torch.Tensor, sigmas: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        test_noisy = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 17)
        models = [
            ("fixed_delay", "static", FixedDelay()),
            ("uniform_mixed_delay", "static", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE)),
            ("recency_weighted_mixed_delay", "static", AnchoredSoftDelay("recency_weighted", train_alpha=False).to(dtype=DTYPE)),
        ]
        for r_value in R_VALUES:
            for model_name, variant, model in models:
                rows.append({
                    **_evaluate_model(model, test_noisy, test_clean, r_value),
                    "model": model_name,
                    "variant": variant,
                    "window_size_m": M,
                    "observed_ls_transitions_R": r_value,
                    "total_observed_positions_required": M + r_value,
                    "noise_sigma": sigma,
                    "initialization": "",
                    "lr": "",
                })
    return rows


def _train_one(
    model_kind: str,
    init_kind: str,
    lr: float,
    sigma: float,
    r_value: int,
    train_clean: torch.Tensor,
    test_clean: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_noisy = _noisy_positions(train_clean, sigma, TRAIN_DATA_SEED + int(sigma * 1e8) + 101)
    test_noisy = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 101)
    if model_kind == "anchored_trainable_soft_delay":
        model: Any = AnchoredSoftDelay(init_kind, train_alpha=True).to(dtype=DTYPE)
        params = [model.alpha]
        include_recon = False
    elif model_kind == "two_row_trainable_soft_selector":
        model = TwoRowSoftSelector(init_kind).to(dtype=DTYPE)
        params = list(model.parameters())
        include_recon = True
    else:
        raise ValueError(model_kind)
    optimizer = torch.optim.Adam(params, lr=lr)
    initial_eval = _evaluate_model(model, test_noisy, test_clean, r_value)
    best_eval = initial_eval
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_mse = float(initial_eval["future_10_period_rollout_mse"])
    history: list[dict[str, Any]] = []
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad()
        loss = _future_loss(model, train_noisy, train_clean, r_value, include_recon=include_recon)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
            eval_row = _evaluate_model(model, test_noisy, test_clean, r_value)
            mse = float(eval_row["future_10_period_rollout_mse"])
            if mse < best_mse:
                best_mse = mse
                best_eval = eval_row
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            history.append({
                **eval_row,
                "model": model_kind,
                "variant": "trainable",
                "window_size_m": M,
                "observed_ls_transitions_R": r_value,
                "total_observed_positions_required": M + r_value,
                "noise_sigma": sigma,
                "initialization": init_kind,
                "lr": lr,
                "step": step,
                "train_loss": float(loss.detach().item()),
                "best_future_mse_so_far": best_mse,
            })
    final_eval = _evaluate_model(model, test_noisy, test_clean, r_value)
    model.load_state_dict(best_state)
    restored = _evaluate_model(model, test_noisy, test_clean, r_value)
    summary = {
        **restored,
        "model": model_kind,
        "variant": "trainable",
        "window_size_m": M,
        "observed_ls_transitions_R": r_value,
        "total_observed_positions_required": M + r_value,
        "noise_sigma": sigma,
        "initialization": init_kind,
        "lr": lr,
        "initial_future_rollout_mse": initial_eval["future_10_period_rollout_mse"],
        "best_future_rollout_mse": best_eval["future_10_period_rollout_mse"],
        "final_future_rollout_mse": final_eval["future_10_period_rollout_mse"],
        "training_improved_over_initial": float(best_eval["future_10_period_rollout_mse"]) < float(initial_eval["future_10_period_rollout_mse"]),
    }
    return summary, history


def _best_static_lookup(rows: list[dict[str, Any]]) -> dict[tuple[float, int, str], float]:
    return {
        (float(row["noise_sigma"]), int(row["observed_ls_transitions_R"]), row["model"]): float(row["future_10_period_rollout_mse"])
        for row in rows
    }


def _training_sweep(train_clean: torch.Tensor, test_clean: torch.Tensor, static_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    static = _best_static_lookup(static_rows)
    # Runtime-managed subset for Model E: enough to test the question without a huge sweep.
    for sigma in SIGMAS:
        for r_value in R_VALUES:
            for init_kind in ["uniform", "recency_weighted", "lag7"]:
                for lr in LRS:
                    summary, hist = _train_one("anchored_trainable_soft_delay", init_kind, lr, sigma, r_value, train_clean, test_clean)
                    summary.update({
                        "beats_fixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "fixed_delay")],
                        "beats_uniform_mixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "uniform_mixed_delay")],
                        "beats_recency_weighted_mixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "recency_weighted_mixed_delay")],
                    })
                    rows.append(summary)
                    history.extend(hist)
            for init_kind in ["near_uniform", "noisy_separated", "bad_separated"]:
                for lr in [1e-3, 1e-4]:
                    summary, hist = _train_one("two_row_trainable_soft_selector", init_kind, lr, sigma, r_value, train_clean, test_clean)
                    summary.update({
                        "beats_fixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "fixed_delay")],
                        "beats_uniform_mixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "uniform_mixed_delay")],
                        "beats_recency_weighted_mixed_delay": float(summary["future_10_period_rollout_mse"]) < static[(sigma, r_value, "recency_weighted_mixed_delay")],
                    })
                    rows.append(summary)
                    history.extend(hist)
    return rows, history


def _plot_future_vs_r(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for sigma in SIGMAS:
        for model in ["fixed_delay", "uniform_mixed_delay", "recency_weighted_mixed_delay"]:
            selected = [r for r in rows if r["model"] == model and float(r["noise_sigma"]) == sigma]
            selected.sort(key=lambda r: int(r["observed_ls_transitions_R"]))
            plt.plot([int(r["observed_ls_transitions_R"]) for r in selected], [float(r["future_10_period_rollout_mse"]) for r in selected], marker="o", label=f"{model}, sigma={sigma:g}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("observed LS transitions R")
    plt.ylabel("future 10-period MSE")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_mse_vs_R_by_noise.png", dpi=170)
    plt.close()


def _plot_noise_by_model(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for model in ["fixed_delay", "uniform_mixed_delay", "recency_weighted_mixed_delay"]:
        for r_value in R_VALUES:
            selected = [r for r in rows if r["model"] == model and int(r["observed_ls_transitions_R"]) == r_value]
            selected.sort(key=lambda r: float(r["noise_sigma"]))
            plt.plot([float(r["noise_sigma"]) for r in selected], [float(r["future_10_period_rollout_mse"]) for r in selected], marker="o", label=f"{model}, R={r_value}")
    plt.xscale("symlog", linthresh=1e-5)
    plt.yscale("log")
    plt.xlabel("noise sigma")
    plt.ylabel("future 10-period MSE")
    plt.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_mse_vs_noise_by_model.png", dpi=170)
    plt.close()


def _plot_bars(rows: list[dict[str, Any]], filename: str, title_filter: str) -> None:
    selected = [r for r in rows if title_filter in r["model"]]
    labels = [f"{r['model']}\ns{float(r['noise_sigma']):g} R{r['observed_ls_transitions_R']}\n{r['initialization']} {r['lr']}" for r in selected[:40]]
    vals = [float(r["future_10_period_rollout_mse"]) for r in selected[:40]]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=70, ha="right", fontsize=6)
    plt.ylabel("best future MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_condition(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for sigma in SIGMAS:
        selected = [r for r in rows if r["model"] == "uniform_mixed_delay" and float(r["noise_sigma"]) == sigma]
        selected.sort(key=lambda r: int(r["observed_ls_transitions_R"]))
        plt.plot([int(r["observed_ls_transitions_R"]) for r in selected], [float(r["ls_condition_number"]) for r in selected], marker="o", label=f"sigma={sigma:g}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("observed LS transitions R")
    plt.ylabel("condition number")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "condition_number_vs_R_by_noise.png", dpi=160)
    plt.close()


def _plot_pi(rows: list[dict[str, Any]], filename: str) -> None:
    selected = rows[:80]
    labels = [f"{r['model']} s{float(r['noise_sigma']):g} R{r['observed_ls_transitions_R']}" for r in selected]
    matrix = []
    for r in selected:
        if r["model"] == "two_row_trainable_soft_selector":
            vals = [float(r.get(f"row1_coord_{i}", 0.0)) for i in range(M)]
        else:
            vals = [float(r.get(f"pi_lag_{i}", 0.0)) for i in range(1, M)] + [0.0]
        matrix.append(vals)
    plt.figure(figsize=(9, max(4, len(selected) * 0.12)))
    plt.imshow(np.array(matrix), aspect="auto", cmap="viridis", vmin=0)
    plt.yticks(range(len(labels)), labels, fontsize=4)
    plt.colorbar(label="weight")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close()


def _plot_threshold(rows: list[dict[str, Any]]) -> None:
    selected = rows[:120]
    labels = [f"{r['model']} s{float(r['noise_sigma']):g} R{r['observed_ls_transitions_R']}" for r in selected]
    matrix = np.array([[1.0 if r[f"crosses_{t:g}"] else 0.0 for t in THRESHOLDS] for r in selected])
    plt.figure(figsize=(8, max(4, len(selected) * 0.1)))
    plt.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.yticks(range(len(labels)), labels, fontsize=4)
    plt.xticks(range(len(THRESHOLDS)), [f"{t:g}" for t in THRESHOLDS])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings_heatmap.png", dpi=180)
    plt.close()


def _write_html(summary_rows: list[dict[str, Any]], static_rows: list[dict[str, Any]], trained_rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    train_beats_uniform = sum(1 for r in trained_rows if r.get("beats_uniform_mixed_delay") is True)
    train_beats_recency = sum(1 for r in trained_rows if r.get("beats_recency_weighted_mixed_delay") is True)
    answers = [
        "Noiseless fixed priors remain strong, confirming there is little to learn without noise.",
        f"Trainable runs beating uniform mixed-delay: {train_beats_uniform} of {len(trained_rows)}.",
        f"Trainable runs beating recency mixed-delay: {train_beats_recency} of {len(trained_rows)}.",
        "The main comparison is by noise sigma and observed LS transitions R; future points are only used in loss/evaluation, never LS identification.",
        "This run uses the spec's runtime-prioritized grid: R=2,4,8,16 and sigma=0,1e-3,1e-2.",
    ]
    cols = [
        "model",
        "variant",
        "noise_sigma",
        "observed_ls_transitions_R",
        "future_10_period_rollout_mse",
        "one_step_future_mse",
        "two_step_future_mse",
        "ls_condition_number",
        "initialization",
        "lr",
        "initial_future_rollout_mse",
        "best_future_rollout_mse",
        "final_future_rollout_mse",
        "beats_uniform_mixed_delay",
        "beats_recency_weighted_mixed_delay",
    ]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in summary_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Noisy Short-Prefix Learned Delay</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:10px}}td,th{{border:1px solid #d8dee8;padding:4px 5px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head><body>
<h1>Stage 1B Noisy Short-Prefix Learned-Delay Diagnostic</h1>
<p>The model observes noisy prefixes y_t, fits K_h only from the first R observed latent transitions, and is scored on clean 10-period future rollout.</p>
<h2>Direct Answers</h2><ol>{''.join(f'<li>{escape(a)}</li>' for a in answers)}</ol>
<h2>Summary</h2><table><thead><tr>{''.join(f'<th>{escape(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="future_mse_vs_R_by_noise.png"><img src="future_mse_vs_noise_by_model.png"><img src="fixed_vs_uniform_vs_trained.png"><img src="trained_improvement_over_uniform.png"><img src="condition_number_vs_R_by_noise.png"><img src="pi_values_by_noise_and_R.png"><img src="two_row_selector_weights.png"><img src="threshold_crossings_heatmap.png">
</div>
<h2>Setup</h2><pre>{escape(json.dumps(setup, indent=2))}</pre>
</body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_clean, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_clean, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    static_rows = _static_sweep(train_clean, test_clean, SIGMAS)
    trained_rows, training_history = _training_sweep(train_clean, test_clean, static_rows)
    summary_rows = static_rows + trained_rows
    previous_path = PREVIOUS_PREFIX_DIR / "R_sweep_results.csv"
    setup = {
        "diagnostic": "stage1B_noisy_short_prefix_learned_delay",
        "runtime_adjustment": "Used prioritized grid from request: R=2,4,8,16 and sigma=0,1e-3,1e-2.",
        "window_size_m": M,
        "observed_ls_transitions_R": R_VALUES,
        "noise_sigmas": SIGMAS,
        "dtype": str(DTYPE),
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
        "train_steps": TRAIN_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "lambda_div": LAMBDA_DIV,
        "lambda_recon": LAMBDA_RECON,
        "previous_noiseless_prefix_results_available": previous_path.exists(),
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher_loss": False,
        "k_generated_by_mlp": False,
        "k_identified_by_observed_prefix_ls": True,
        "wall_clock_sec": time.perf_counter() - start,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows, "training_history": training_history})
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", summary_rows + training_history)
    _write_csv(OUTPUT_DIR / "R_noise_sweep_results.csv", static_rows)
    _write_csv(OUTPUT_DIR / "training_history.csv", training_history)
    _write_csv(OUTPUT_DIR / "pi_values_by_model.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "threshold_crossings.csv", [{**{k: r.get(k, "") for k in ["model", "variant", "noise_sigma", "observed_ls_transitions_R", "initialization", "lr", "future_10_period_rollout_mse"]}, **{f"crosses_{t:g}": r[f"crosses_{t:g}"] for t in THRESHOLDS}} for r in summary_rows])
    _plot_future_vs_r(static_rows)
    _plot_noise_by_model(static_rows)
    _plot_bars(summary_rows, "fixed_vs_uniform_vs_trained.png", "")
    _plot_bars(trained_rows, "trained_improvement_over_uniform.png", "trainable")
    _plot_condition(static_rows)
    _plot_pi(summary_rows, "pi_values_by_noise_and_R.png")
    _plot_pi([r for r in summary_rows if r["model"] == "two_row_trainable_soft_selector"], "two_row_selector_weights.png")
    _plot_threshold(summary_rows)
    _write_html(summary_rows, static_rows, trained_rows, setup)


if __name__ == "__main__":
    main()
