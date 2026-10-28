"""Stage 1B recovery-limit and curriculum diagnostic.

Audits prefix-limited LS recovery floors, decomposes noisy-prefix error, and
tests whether short curricula help learned delay filters approach those floors.
"""

from __future__ import annotations

import csv
import json
import math
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

from experiments.sho.run_stage1b_noisy_short_prefix_learned_delay_diagnostic import (
    AnchoredSoftDelay,
    FixedDelay,
    TwoRowSoftSelector,
    _noisy_positions,
)
from experiments.sho.run_stage1b_soft_delay_learning_ablation import _pi_vector
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _apply_k,
    _fit_k_ridge,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/recovery_limit_and_curriculum_diagnostic")
PREVIOUS_NOISY_DIR = Path("outputs/sho/stage1B/noisy_short_prefix_learned_delay_diagnostic")

M = 8
TRAIN_NUM = 8
TEST_NUM = 6
ROLLOUT_PERIODS = 10.0
R_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
SIGMAS = [0.0, 1e-4, 1e-3, 1e-2]
MODELS = ["fixed_delay", "uniform_mixed_delay", "recency_weighted_mixed_delay"]
SOLVER_KINDS = ["normal", "qr", "svd", "ridge"]
RIDGES = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

CURRICULUM_STAGES = [
    {"stage": 1, "sigma": 0.0, "R": 128, "periods": 1.0},
    {"stage": 2, "sigma": 0.0, "R": 64, "periods": 2.0},
    {"stage": 3, "sigma": 1e-4, "R": 64, "periods": 5.0},
    {"stage": 4, "sigma": 1e-3, "R": 32, "periods": 10.0},
    {"stage": 5, "sigma": 1e-3, "R": 16, "periods": 10.0},
    {"stage": 6, "sigma": 1e-2, "R": 32, "periods": 10.0},
    {"stage": 7, "sigma": 1e-2, "R": 16, "periods": 10.0},
]
CURRICULUM_MUS = [0.0, 1e-2, 1e-1, 1.0]
TRAIN_LRS = [1e-3, 1e-4, 1e-5]
STAGE_STEPS = 80
DIRECT_STEPS = 160
EVAL_INTERVAL = 40
LAMBDA_DIV = 1e-2


def _full_rollout_steps(periods: float = ROLLOUT_PERIODS) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_long_data(num_trajectories: int, seed: int) -> torch.Tensor:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed, dtype=DTYPE)
    total_steps = max(R_VALUES) + M + _full_rollout_steps() + 8
    times = torch.arange(total_steps, dtype=DTYPE) * float(DELTA_T)
    times = times.unsqueeze(0).expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return positions


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _model_for_name(name: str) -> Any:
    if name == "fixed_delay":
        return FixedDelay()
    if name == "uniform_mixed_delay":
        return AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE)
    if name == "recency_weighted_mixed_delay":
        return AnchoredSoftDelay("recency_weighted", train_alpha=False).to(dtype=DTYPE)
    raise ValueError(name)


def _observed_windows(positions: torch.Tensor, r_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _window_sequence(positions, M, M - 1, r_value), _window_sequence(positions, M, M, r_value)


def _future_windows(positions: torch.Tensor, r_value: int, periods: float = ROLLOUT_PERIODS) -> torch.Tensor:
    return _window_sequence(positions, M, M - 1 + r_value, _full_rollout_steps(periods) + 1)


def _solve_k(h_in: torch.Tensor, h_out: torch.Tensor, solver: str, ridge: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if solver == "ridge":
        return _fit_k_ridge(h_in, h_out, ridge=ridge)
    if solver == "normal":
        return _fit_k_ridge(h_in, h_out, ridge=0.0)
    k_list = []
    residuals = []
    conds = []
    for x, y in zip(h_in, h_out):
        if solver == "qr":
            sol = torch.linalg.lstsq(x, y).solution
        elif solver == "svd":
            sol = torch.linalg.pinv(x) @ y
        else:
            raise ValueError(solver)
        k = sol.T
        k_list.append(k)
        residuals.append(torch.mean((x @ sol - y) ** 2))
        conds.append(torch.linalg.cond(x))
    return torch.stack(k_list), torch.stack(residuals).mean(), torch.stack(conds)


def _evaluate(
    model: Any,
    observed_positions: torch.Tensor,
    clean_positions: torch.Tensor,
    r_value: int,
    solver: str = "ridge",
    ridge: float = 1e-8,
    periods: float = ROLLOUT_PERIODS,
    k_override: torch.Tensor | None = None,
    start_positions: torch.Tensor | None = None,
) -> dict[str, Any]:
    q_in, q_out = _observed_windows(observed_positions, r_value)
    h_in = model.encode(q_in)
    h_out = model.encode(q_out)
    if k_override is None:
        k_h, residual, conds = _solve_k(h_in, h_out, solver, ridge)
    else:
        k_h = k_override
        residual = torch.mean((_apply_k(k_h, h_in) - h_out) ** 2)
        conds = torch.linalg.cond(torch.bmm(h_in.transpose(1, 2), h_in) + 1e-12 * torch.eye(h_in.shape[-1], dtype=DTYPE).unsqueeze(0))
    truth = _future_windows(clean_positions, r_value, periods)
    start_source = observed_positions if start_positions is None else start_positions
    q0 = _future_windows(start_source, r_value, periods)[:, 0]
    h0 = model.encode(q0)
    pred_x = model.decode_x(_rollout_matrix(k_h, h0, truth.shape[1] - 1))
    metrics = _rollout_metrics(pred_x, truth)
    return {
        **metrics,
        "future_10_period_mse": metrics["rollout_mse"],
        "future_rollout_horizon_periods": periods,
        "observed_ls_residual": float(residual.detach().item()),
        "ls_condition_number": float(conds.detach().mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
        **_threshold_flags(metrics["rollout_mse"]),
    }


def _ls_audit(train_clean: torch.Tensor, test_clean: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sigma in SIGMAS:
        test_obs = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 33)
        for model_name in MODELS:
            model = _model_for_name(model_name)
            for r_value in R_VALUES:
                for solver in ["normal", "qr", "svd"]:
                    try:
                        result = _evaluate(model, test_obs, test_clean, r_value, solver=solver, ridge=0.0)
                    except RuntimeError:
                        continue
                    rows.append({
                        **result,
                        "model_type": model_name,
                        "sigma": sigma,
                        "observed_ls_transitions_R": r_value,
                        "total_observed_positions_required": M + r_value,
                        "ls_solver": solver,
                        "ridge": 0.0,
                    })
                for ridge in RIDGES:
                    try:
                        result = _evaluate(model, test_obs, test_clean, r_value, solver="ridge", ridge=ridge)
                    except RuntimeError:
                        continue
                    rows.append({
                        **result,
                        "model_type": model_name,
                        "sigma": sigma,
                        "observed_ls_transitions_R": r_value,
                        "total_observed_positions_required": M + r_value,
                        "ls_solver": "ridge",
                        "ridge": ridge,
                    })
    return rows


def _best_floor_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, float, int], dict[str, Any]]:
    best: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["model_type"], float(row["sigma"]), int(row["observed_ls_transitions_R"]))
        if key not in best or float(row["future_10_period_mse"]) < float(best[key]["future_10_period_mse"]):
            best[key] = row
    return best


def _error_decomposition(test_clean: torch.Tensor, best_lookup: dict[tuple[str, float, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sigma in SIGMAS:
        noisy = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 33)
        for model_name in MODELS:
            model = _model_for_name(model_name)
            for r_value in [16, 64, 128, 256]:
                best = best_lookup[(model_name, sigma, r_value)]
                solver, ridge = best["ls_solver"], float(best["ridge"])
                clean_q_in, clean_q_out = _observed_windows(test_clean, r_value)
                noisy_q_in, noisy_q_out = _observed_windows(noisy, r_value)
                clean_k, _, _ = _solve_k(model.encode(clean_q_in), model.encode(clean_q_out), solver, ridge)
                noisy_k, _, _ = _solve_k(model.encode(noisy_q_in), model.encode(noisy_q_out), solver, ridge)
                cases = [
                    ("clean_prefix_identification", test_clean, clean_k, test_clean),
                    ("noisy_prefix_identification", noisy, noisy_k, noisy),
                    ("noisy_state_clean_K", noisy, clean_k, noisy),
                    ("clean_state_noisy_K", noisy, noisy_k, test_clean),
                ]
                for case, observed, k_override, start in cases:
                    rows.append({
                        **_evaluate(model, observed, test_clean, r_value, solver=solver, ridge=ridge, k_override=k_override, start_positions=start),
                        "case": case,
                        "model_type": model_name,
                        "sigma": sigma,
                        "observed_ls_transitions_R": r_value,
                        "ls_solver": solver,
                        "ridge": ridge,
                    })
    return rows


class DenoisedCurrentFilter(nn.Module):
    def __init__(self, init_kind: str) -> None:
        super().__init__()
        self.init_kind = init_kind
        self.alpha_past = nn.Parameter(torch.log(_pi_vector("uniform").clamp_min(1e-30)))
        self.alpha_current = nn.Parameter(torch.log(self._current_pi("uniform").clamp_min(1e-30)))
        self._initialize(init_kind)

    @staticmethod
    def _current_pi(init_kind: str) -> torch.Tensor:
        if init_kind == "uniform":
            return torch.ones(M, dtype=DTYPE) / M
        if init_kind == "recency_weighted":
            by_lag = torch.exp(-0.5 * torch.arange(M, dtype=DTYPE))
            by_lag = by_lag / by_lag.sum()
            return torch.flip(by_lag, dims=(-1,))
        if init_kind == "lag7":
            weights = torch.zeros(M, dtype=DTYPE)
            weights[0] = 1.0
            return weights
        raise ValueError(init_kind)

    def _initialize(self, init_kind: str) -> None:
        with torch.no_grad():
            if init_kind == "uniform":
                return
            if init_kind == "recency_weighted":
                self.alpha_past.copy_(torch.log(_pi_vector("recency_weighted").clamp_min(1e-30)))
                self.alpha_current.copy_(torch.log(self._current_pi("recency_weighted").clamp_min(1e-30)))
            elif init_kind == "lag7":
                self.alpha_past.copy_(torch.log(_pi_vector("lag7").clamp_min(1e-30)))
                self.alpha_current.copy_(torch.log(self._current_pi("lag7").clamp_min(1e-30)))
            else:
                raise ValueError(init_kind)

    def pis(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.softmax(self.alpha_past, dim=0), torch.softmax(self.alpha_current, dim=0)

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        past_by_lag = torch.flip(q[..., :-1], dims=(-1,))
        pi_past, pi_current = self.pis()
        h0 = torch.sum(past_by_lag * pi_past.to(q.device), dim=-1)
        h1 = torch.sum(q * pi_current.to(q.device), dim=-1)
        return torch.stack((h0, h1), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]

    def diversity_penalty(self) -> torch.Tensor:
        a, b = self.pis()
        current_by_lag = torch.flip(b[:-1], dims=(-1,))
        return torch.sum(a * current_by_lag)

    def filter_metrics(self) -> dict[str, Any]:
        a, b = self.pis()
        av, bv = a.detach().cpu().numpy(), b.detach().cpu().numpy()
        current_selected_q_index = int(np.argmax(bv))
        return {
            "filter_weights": json.dumps({"past": av.tolist(), "current": bv.tolist()}),
            "entropy": float(-np.sum(av * np.log(np.maximum(av, 1e-300))) - np.sum(bv * np.log(np.maximum(bv, 1e-300)))),
            "diversity_penalty": float(np.sum(av * np.flip(bv[:-1]))),
            "selected_lag": json.dumps([int(np.argmax(av)) + 1, M - 1 - current_selected_q_index]),
            "max_pi": float(max(np.max(av), np.max(bv))),
        }


def _trainable_model(kind: str, init: str) -> nn.Module:
    if kind == "anchored":
        return AnchoredSoftDelay(init, train_alpha=True).to(dtype=DTYPE)
    if kind == "denoised_current":
        return DenoisedCurrentFilter(init).to(dtype=DTYPE)
    raise ValueError(kind)


def _train_loss(model: nn.Module, noisy: torch.Tensor, clean: torch.Tensor, r_value: int, periods: float) -> torch.Tensor:
    q_in, q_out = _observed_windows(noisy, r_value)
    k_h, _, _ = _fit_k_ridge(model.encode(q_in), model.encode(q_out), ridge=1e-8)
    truth = _future_windows(clean, r_value, periods)
    q0 = _future_windows(noisy, r_value, periods)[:, 0]
    pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(q0), truth.shape[1] - 1))
    loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    if isinstance(model, DenoisedCurrentFilter):
        loss = loss + LAMBDA_DIV * model.diversity_penalty()
    return loss


def _train_eval(model: nn.Module, noisy: torch.Tensor, clean: torch.Tensor, r_value: int, periods: float = ROLLOUT_PERIODS) -> dict[str, Any]:
    result = _evaluate(model, noisy, clean, r_value, solver="ridge", ridge=1e-8, periods=periods)
    if hasattr(model, "filter_metrics"):
        result.update(model.filter_metrics())
    else:
        pi = model.pi().detach().cpu().numpy()
        result.update({
            "filter_weights": json.dumps(pi.tolist()),
            "entropy": float(-np.sum(pi * np.log(np.maximum(pi, 1e-300)))),
            "diversity_penalty": "",
            "selected_lag": int(np.argmax(pi)) + 1,
            "max_pi": float(np.max(pi)),
        })
    return result


def _run_train(
    train_clean: torch.Tensor,
    test_clean: torch.Tensor,
    kind: str,
    init: str,
    lr: float,
    stages: list[dict[str, Any]],
    mu: float,
    direct_name: str,
    steps_per_stage: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = _trainable_model(kind, init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accepted = [p.detach().clone() for p in model.parameters()]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_eval: dict[str, Any] | None = None
    best_mse = float("inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    for stage in stages:
        sigma, r_value, periods = float(stage["sigma"]), int(stage["R"]), float(stage["periods"])
        train_noisy = _noisy_positions(train_clean, sigma, TRAIN_DATA_SEED + int(sigma * 1e8) + 77)
        test_noisy = _noisy_positions(test_clean, sigma, TEST_DATA_SEED + int(sigma * 1e8) + 77)
        for local_step in range(1, steps_per_stage + 1):
            global_step += 1
            optimizer.zero_grad()
            loss = _train_loss(model, train_noisy, train_clean, r_value, periods)
            drift = torch.zeros((), dtype=DTYPE)
            for p, old in zip(model.parameters(), accepted):
                drift = drift + torch.sum((p - old.to(p.device)) ** 2)
            total = loss + mu * drift
            total.backward()
            optimizer.step()
            if local_step == 1 or local_step % EVAL_INTERVAL == 0 or local_step == steps_per_stage:
                eval_row = _train_eval(model, test_noisy, test_clean, r_value, periods=periods)
                checkpoint_eval = _train_eval(model, test_noisy, test_clean, r_value, periods=ROLLOUT_PERIODS)
                mse = float(checkpoint_eval["future_10_period_mse"])
                if mse < best_mse:
                    best_mse = mse
                    best_eval = checkpoint_eval
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                history.append({
                    **eval_row,
                    "run_type": direct_name,
                    "model_type": kind,
                    "init": init,
                    "lr": lr,
                    "mu": mu,
                    "stage": stage["stage"],
                    "sigma": sigma,
                    "observed_ls_transitions_R": r_value,
                    "global_step": global_step,
                    "local_step": local_step,
                    "train_loss": float(loss.detach().item()),
                    "biased_regularization_penalty": float((mu * drift).detach().item()),
                    "parameter_drift": float(drift.detach().item()),
                    "checkpoint_full_future_mse": mse,
                    "best_future_mse_so_far": best_mse,
                })
        accepted = [p.detach().clone() for p in model.parameters()]
    model.load_state_dict(best_state)
    final_stage = stages[-1]
    final_noisy = _noisy_positions(test_clean, float(final_stage["sigma"]), TEST_DATA_SEED + int(float(final_stage["sigma"]) * 1e8) + 77)
    final_eval = _train_eval(model, final_noisy, test_clean, int(final_stage["R"]), periods=float(final_stage["periods"]))
    summary = {
        **final_eval,
        "run_type": direct_name,
        "model_type": kind,
        "init": init,
        "lr": lr,
        "mu": mu,
        "sigma": final_stage["sigma"],
        "observed_ls_transitions_R": final_stage["R"],
        "best_future_mse": best_mse,
        "final_future_mse": final_eval["future_10_period_mse"],
        "curriculum_stages": len(stages),
    }
    return summary, history


def _training(train_clean: torch.Tensor, test_clean: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    curriculum_rows: list[dict[str, Any]] = []
    curriculum_history: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    direct_history: list[dict[str, Any]] = []
    for kind in ["anchored", "denoised_current"]:
        for init in ["uniform", "recency_weighted"]:
            for lr in TRAIN_LRS:
                for mu in CURRICULUM_MUS:
                    summary, hist = _run_train(train_clean, test_clean, kind, init, lr, CURRICULUM_STAGES, mu, "curriculum", STAGE_STEPS)
                    curriculum_rows.append(summary)
                    curriculum_history.extend(hist)
            for sigma in [1e-3, 1e-2]:
                stages = [{"stage": f"direct_sigma_{sigma:g}", "sigma": sigma, "R": 16, "periods": 10.0}]
                for lr in TRAIN_LRS:
                    summary, hist = _run_train(train_clean, test_clean, kind, init, lr, stages, 0.0, "direct", DIRECT_STEPS)
                    direct_rows.append(summary)
                    direct_history.extend(hist)
    return curriculum_rows, curriculum_history, direct_rows, direct_history


def _plot_lines(rows: list[dict[str, Any]], xkey: str, ykey: str, group: str, filename: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    for label in sorted({str(r[group]) for r in rows})[:20]:
        selected = [r for r in rows if str(r[group]) == label]
        selected.sort(key=lambda r: float(r[xkey]))
        plt.plot([float(r[xkey]) for r in selected], [float(r[ykey]) for r in selected], marker="o", label=label)
    plt.yscale("log")
    if "R" in xkey:
        plt.xscale("log", base=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_bars(rows: list[dict[str, Any]], filename: str) -> None:
    selected = rows[:60]
    labels = [f"{r.get('run_type', r.get('case', ''))}\n{r.get('model_type', '')}\ns{r.get('sigma', '')} R{r.get('observed_ls_transitions_R', '')}" for r in selected]
    vals = [float(r["future_10_period_mse"]) for r in selected]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=70, ha="right", fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_threshold(rows: list[dict[str, Any]]) -> None:
    selected = rows[:120]
    labels = [f"{r.get('model_type', '')} s{r.get('sigma', '')} R{r.get('observed_ls_transitions_R', '')}" for r in selected]
    matrix = np.array([[1.0 if r.get(f"crosses_{t:g}") else 0.0 for t in THRESHOLDS] for r in selected])
    plt.figure(figsize=(8, max(4, len(labels) * 0.1)))
    plt.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.yticks(range(len(labels)), labels, fontsize=4)
    plt.xticks(range(len(THRESHOLDS)), [f"{t:g}" for t in THRESHOLDS])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings_heatmap.png", dpi=180)
    plt.close()


def _write_html(setup: dict[str, Any], best_rows: list[dict[str, Any]], decomp: list[dict[str, Any]], curriculum: list[dict[str, Any]], direct: list[dict[str, Any]]) -> None:
    best_noiseless = min(float(r["future_10_period_mse"]) for r in best_rows if float(r["sigma"]) == 0.0)
    best_noisy = min(float(r["future_10_period_mse"]) for r in best_rows if float(r["sigma"]) > 0.0)
    best_curr = min(float(r["future_10_period_mse"]) for r in curriculum) if curriculum else float("nan")
    best_direct = min(float(r["future_10_period_mse"]) for r in direct) if direct else float("nan")
    answers = [
        f"Best noiseless prefix-limited LS floor in this audit: {best_noiseless:.3e}.",
        f"Best noisy prefix-limited LS floor in this audit: {best_noisy:.3e}.",
        "Noisy-prefix error separates into K error and noisy initial-state effects in error_decomposition.csv.",
        f"Best curriculum trained model: {best_curr:.3e}; best direct trained model: {best_direct:.3e}.",
        "Exact recovery is not realistic under noisy finite-prefix observations; the floor is noise/R limited.",
    ]
    cols = ["model_type", "sigma", "observed_ls_transitions_R", "future_10_period_mse", "ls_solver", "ridge", "ls_condition_number", "spectral_radius_k_h"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in best_rows[:120])
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Recovery Limit and Curriculum Diagnostic</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:11px}}td,th{{border:1px solid #d8dee8;padding:5px 6px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head><body>
<h1>Stage 1B Recovery-Limit and Curriculum Diagnostic</h1>
<h2>Direct Answers</h2><ol>{''.join(f'<li>{escape(a)}</li>' for a in answers)}</ol>
<h2>Best LS Floors</h2><table><thead><tr>{''.join(f'<th>{escape(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="future_mse_vs_R_by_solver.png"><img src="future_mse_vs_R_by_noise.png"><img src="future_mse_vs_noise_by_model.png"><img src="ridge_sweep_by_noise.png"><img src="error_source_decomposition.png"><img src="curriculum_vs_direct_training.png"><img src="learned_filter_weights.png"><img src="condition_number_vs_R.png"><img src="threshold_crossings_heatmap.png">
</div>
<h2>Setup</h2><pre>{escape(json.dumps(setup, indent=2))}</pre>
</body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_clean = _make_long_data(TRAIN_NUM, TRAIN_DATA_SEED)
    test_clean = _make_long_data(TEST_NUM, TEST_DATA_SEED)
    ls_rows = _ls_audit(train_clean, test_clean)
    best_lookup = _best_floor_lookup(ls_rows)
    best_rows = list(best_lookup.values())
    decomp_rows = _error_decomposition(test_clean, best_lookup)
    curriculum_rows, curriculum_history, direct_rows, direct_history = _training(train_clean, test_clean)
    summary_rows = best_rows + curriculum_rows + direct_rows
    setup = {
        "diagnostic": "stage1B_recovery_limit_and_curriculum",
        "runtime_note": "Full LS solver/ridge audit is run; trainable curricula use the requested lr/mu grid with compact optimizer steps.",
        "m_window_size": M,
        "observed_ls_transitions_R": R_VALUES,
        "sigmas": SIGMAS,
        "solvers": SOLVER_KINDS,
        "ridges": RIDGES,
        "curriculum_mus": CURRICULUM_MUS,
        "train_lrs": TRAIN_LRS,
        "stage_steps": STAGE_STEPS,
        "direct_steps": DIRECT_STEPS,
        "dtype": str(DTYPE),
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher_loss": False,
        "k_generated_by_mlp": False,
        "k_identified_by_ls_style_solver": True,
        "previous_noisy_short_prefix_best_available": (PREVIOUS_NOISY_DIR / "summary.csv").exists(),
        "wall_clock_sec": time.perf_counter() - start,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows, "ls_solver_audit": ls_rows, "error_decomposition": decomp_rows})
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", ls_rows + decomp_rows + curriculum_history + direct_history)
    _write_csv(OUTPUT_DIR / "ls_solver_audit.csv", ls_rows)
    _write_csv(OUTPUT_DIR / "error_decomposition.csv", decomp_rows)
    _write_csv(OUTPUT_DIR / "curriculum_results.csv", curriculum_rows)
    _write_csv(OUTPUT_DIR / "direct_training_results.csv", direct_rows)
    _write_csv(OUTPUT_DIR / "filter_weights.csv", curriculum_rows + direct_rows)
    _write_csv(OUTPUT_DIR / "threshold_crossings.csv", [{**{k: r.get(k, "") for k in ["model_type", "sigma", "observed_ls_transitions_R", "future_10_period_mse", "ls_solver", "ridge"]}, **{f"crosses_{t:g}": r.get(f"crosses_{t:g}", "") for t in THRESHOLDS}} for r in summary_rows])

    _plot_lines([r for r in ls_rows if r["ls_solver"] != "ridge" and r["model_type"] == "uniform_mixed_delay" and float(r["sigma"]) in [0.0, 1e-3]], "observed_ls_transitions_R", "future_10_period_mse", "ls_solver", "future_mse_vs_R_by_solver.png", "R", "future MSE")
    _plot_lines([r for r in best_rows if r["model_type"] == "uniform_mixed_delay"], "observed_ls_transitions_R", "future_10_period_mse", "sigma", "future_mse_vs_R_by_noise.png", "R", "best future MSE")
    _plot_lines([r for r in best_rows if int(r["observed_ls_transitions_R"]) == 16], "sigma", "future_10_period_mse", "model_type", "future_mse_vs_noise_by_model.png", "sigma", "best future MSE")
    _plot_lines([r for r in ls_rows if r["ls_solver"] == "ridge" and r["model_type"] == "uniform_mixed_delay" and int(r["observed_ls_transitions_R"]) == 16], "ridge", "future_10_period_mse", "sigma", "ridge_sweep_by_noise.png", "ridge", "future MSE")
    _plot_bars(decomp_rows, "error_source_decomposition.png")
    _plot_bars(curriculum_rows + direct_rows, "curriculum_vs_direct_training.png")
    _plot_bars(curriculum_rows + direct_rows, "learned_filter_weights.png")
    _plot_lines(best_rows, "observed_ls_transitions_R", "ls_condition_number", "model_type", "condition_number_vs_R.png", "R", "condition number")
    _plot_threshold(summary_rows)
    _write_html(setup, best_rows, decomp_rows, curriculum_rows, direct_rows)


if __name__ == "__main__":
    main()
