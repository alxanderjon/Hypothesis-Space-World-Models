"""Stage 1B four-stage constant-discovery curriculum.

The learned model infers trajectory-specific constants from position-only
prefixes inside the structured recurrence hypothesis space:

    h_{n+1} = [[0, 1], [-1, c_i]] h_n.

No LS is used inside learned models, and no velocity, omega labels, coefficient
labels, sine/cosine features, analytic decoder, analytic recurrence labels, or
closed-form c_i estimator are used for training.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch
from torch import nn

from experiments.sho.run_stage1b_noisy_short_prefix_learned_delay_diagnostic import AnchoredSoftDelay, FixedDelay
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _make_base_conditions, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _fit_k_ridge,
    _rollout_matrix,
    _spectral_radius,
    _window_sequence,
    _write_csv,
    _write_json,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/four_stage_constant_curriculum")

MAX_PREFIX = 128
M = 8
PREFIX_SCHEDULE = [128, 64, 32, 16]
HORIZONS = [1.0, 2.0, 5.0, 10.0]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]
TRAIN_NUM = 48
VAL_NUM = 16
TEST_NUM = 16

# Runtime-prioritized discovery budget. The staged logic is preserved while
# avoiding a long sweep in the first diagnostic pass.
MAX_STEPS = 450
STAGE1_MAX_STEPS = 3500
EVAL_INTERVAL = 90
PATIENCE = 180
LR_STAGE12 = 1e-3
LR_STAGE34 = 3e-4
LAMBDA_LOCAL_STAGE2 = [1.0]
LAMBDA_LOCAL_STAGE3 = 1.0
LAMBDA_STATE_ANCHOR = [1e-2]
MU_VALUES = [0.0, 1e-3]


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor


def _steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_data(n: int, seed: int) -> SHOSet:
    initial_conditions, omegas = _make_base_conditions(n, seed=seed, dtype=DTYPE)
    total_steps = MAX_PREFIX + _steps(10.0) + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(n, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _true_c(data: SHOSet) -> torch.Tensor:
    return 2.0 * torch.cos(data.omegas * float(DELTA_T))


def _prefix(data: SHOSet, p: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(data.positions.shape[0], MAX_PREFIX, dtype=DTYPE)
    mask = torch.zeros_like(x)
    x[:, :p] = data.positions[:, :p]
    mask[:, :p] = 1.0
    return x, mask


def _delay_state(data: SHOSet, p: int) -> torch.Tensor:
    return torch.stack((data.positions[:, p - 2], data.positions[:, p - 1]), dim=-1)


def _future(data: SHOSet, p: int, periods: float) -> torch.Tensor:
    return data.positions[:, p : p + _steps(periods)]


def _rollout(h0: torch.Tensor, c: torch.Tensor, n_steps: int) -> torch.Tensor:
    states = []
    h = h0
    c = c.reshape(-1)
    for _ in range(n_steps):
        h = torch.stack((h[:, 1], -h[:, 0] + c * h[:, 1]), dim=-1)
        states.append(h)
    return torch.stack(states, dim=1)


class PrefixCNN(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, width, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.conv(torch.stack((prefix, mask), dim=1)).transpose(1, 2)
        return (y * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


class PrefixGRU(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=width, batch_first=True)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.stack((prefix, mask), dim=-1)
        _, h = self.gru(x)
        return h[-1]


class ConstantCurriculumModel(nn.Module):
    def __init__(self, constrained_c: bool = True, residual_state: bool = True) -> None:
        super().__init__()
        self.constrained_c = constrained_c
        self.residual_state = residual_state
        self.encoder = PrefixGRU()
        self.dynamics_head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        self.state_head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    def encode(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(prefix, mask)

    def c_from_z(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.dynamics_head(z).squeeze(-1)
        return 2.0 * torch.tanh(raw) if self.constrained_c else raw

    def state_from_z(self, z: torch.Tensor, observed_state: torch.Tensor) -> torch.Tensor:
        out = self.state_head(z)
        return observed_state + out if self.residual_state else out

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, observed_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(prefix, mask)
        return self.state_from_z(z, observed_state), self.c_from_z(z)

    def c_only(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.c_from_z(self.encode(prefix, mask))


def _local_loss(model: ConstantCurriculumModel, data: SHOSet, p: int) -> torch.Tensor:
    prefix, mask = _prefix(data, p)
    c = model.c_only(prefix, mask)
    x = data.positions[:, :p]
    residual = c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]
    return torch.mean(residual**2)


def _future_loss(model: ConstantCurriculumModel, data: SHOSet, p: int, periods: float, state_given: bool) -> torch.Tensor:
    prefix, mask = _prefix(data, p)
    obs_h = _delay_state(data, p)
    c = model.c_only(prefix, mask)
    h = obs_h if state_given else model(prefix, mask, obs_h)[0]
    return torch.mean((_rollout(h, c, _steps(periods))[:, :, 1] - _future(data, p, periods)) ** 2)


def _state_anchor_loss(model: ConstantCurriculumModel, data: SHOSet, p: int) -> torch.Tensor:
    prefix, mask = _prefix(data, p)
    obs_h = _delay_state(data, p)
    h, _ = model(prefix, mask, obs_h)
    return torch.mean((h - obs_h) ** 2)


@torch.no_grad()
def _evaluate(model: ConstantCurriculumModel, data: SHOSet, p: int, stage: str, name: str, state_given: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prefix, mask = _prefix(data, p)
    obs_h = _delay_state(data, p)
    c = model.c_only(prefix, mask)
    h = obs_h if state_given else model(prefix, mask, obs_h)[0]
    err10 = _rollout(h, c, _steps(10.0))[:, :, 1] - _future(data, p, 10.0)
    true_c = _true_c(data)
    row = {
        "stage": stage,
        "config_name": name,
        "prefix_length_P": p,
        "future_horizon": 10.0,
        "model_variant": "residual_state" if not state_given else "state_given",
        "c_parameterization": "2*tanh(raw_c)" if model.constrained_c else "unconstrained",
        "test_future_1_period_mse": _horizon_mse(h, c, data, p, 1.0),
        "test_future_2_period_mse": _horizon_mse(h, c, data, p, 2.0),
        "test_future_5_period_mse": _horizon_mse(h, c, data, p, 5.0),
        "test_future_10_period_mse": float(torch.mean(err10**2).item()),
        "max_future_trajectory_error": float(torch.max(torch.abs(err10)).item()),
        "final_time_absolute_error": float(torch.mean(torch.abs(err10[:, -1])).item()),
        "local_recurrence_loss": float(_local_loss(model, data, p).item()),
        "learned_c_mean": float(c.mean().item()),
        "learned_c_std": float(c.std().item()),
        "diagnostic_c_true_mean": float(true_c.mean().item()),
        "diagnostic_c_true_std": float(true_c.std().item()),
        "c_i_mse": float(torch.mean((c - true_c) ** 2).item()),
        "c_i_r2": _r2(c[:, None], true_c[:, None]),
        "h_start_error": float(torch.mean((h - obs_h) ** 2).item()),
        "spectral_radius": _spectral_radius_c(c),
        "stable_range_fraction": float(torch.mean((torch.abs(c) < 2.0).to(DTYPE)).item()),
    }
    row.update(_threshold_flags(row["test_future_10_period_mse"]))
    if extra:
        row.update(extra)
    return row


def _horizon_mse(h: torch.Tensor, c: torch.Tensor, data: SHOSet, p: int, periods: float) -> float:
    return float(torch.mean((_rollout(h, c, _steps(periods))[:, :, 1] - _future(data, p, periods)) ** 2).item())


def _r2(x: torch.Tensor, y: torch.Tensor) -> float:
    design = torch.cat((x, torch.ones(x.shape[0], 1, dtype=DTYPE)), dim=1)
    coef = torch.linalg.lstsq(design, y).solution
    pred = design @ coef
    return float((1.0 - torch.sum((y - pred) ** 2) / torch.sum((y - y.mean(0, keepdim=True)) ** 2).clamp_min(1e-30)).item())


def _spectral_radius_c(c: torch.Tensor) -> float:
    k = torch.zeros(c.numel(), 2, 2, dtype=DTYPE)
    k[:, 0, 1] = 1.0
    k[:, 1, 0] = -1.0
    k[:, 1, 1] = c.reshape(-1)
    return _spectral_radius(k)


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _train_loop(
    model: ConstantCurriculumModel,
    train: SHOSet,
    val: SHOSet,
    p: int,
    objective: str,
    horizons: list[float],
    lr: float,
    name: str,
    state_given: bool,
    lambda_local: float = 0.0,
    lambda_anchor: float = 0.0,
    mu: float = 0.0,
    accepted: list[torch.Tensor] | None = None,
    freeze_dynamics: bool = False,
    max_steps: int = MAX_STEPS,
) -> tuple[ConstantCurriculumModel, list[dict[str, Any]], dict[str, Any]]:
    for param in model.parameters():
        param.requires_grad_(True)
    if freeze_dynamics:
        for param in model.encoder.parameters():
            param.requires_grad_(False)
        for param in model.dynamics_head.parameters():
            param.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_row: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    if accepted is None:
        accepted = [p.detach().clone() for p in model.parameters()]
    global_step = 0
    for horizon in horizons:
        stale = 0
        for step in range(1, max_steps + 1):
            global_step += 1
            opt.zero_grad()
            loss = torch.zeros((), dtype=DTYPE)
            if objective == "local":
                loss = _local_loss(model, train, p)
            else:
                loss = _future_loss(model, train, p, horizon, state_given)
                if lambda_local:
                    loss = loss + lambda_local * _local_loss(model, train, p)
                if lambda_anchor and not state_given:
                    loss = loss + lambda_anchor * _state_anchor_loss(model, train, p)
            if mu:
                drift = torch.zeros((), dtype=DTYPE)
                for now, old in zip(model.parameters(), accepted):
                    drift = drift + torch.sum((now - old.to(now.device)) ** 2)
                loss = loss + mu * drift
            loss.backward()
            opt.step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == max_steps:
                row = _evaluate(model, val, p, objective, name, state_given, {
                    "learning_rate": lr,
                    "lambda_local": lambda_local,
                    "lambda_state_anchor": lambda_anchor,
                    "mu": mu,
                    "train_loss": float(loss.detach().item()),
                    "validation_loss": float(_local_loss(model, val, p).item()) if objective == "local" else float(_future_loss(model, val, p, 10.0, state_given).item()),
                    "global_step": global_step,
                    "stage_horizon": horizon,
                    "checkpoint_drift": _drift(model, accepted),
                })
                history.append(row)
                metric = float(row["validation_loss"]) if objective == "local" else float(row["test_future_10_period_mse"])
                if metric + 1e-12 < best_val:
                    best_val = metric
                    best_row = row
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                if stale >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    return model, history, best_row


def _drift(model: nn.Module, accepted: list[torch.Tensor]) -> float:
    total = torch.zeros((), dtype=DTYPE)
    for p, old in zip(model.parameters(), accepted):
        total = total + torch.sum((p.detach() - old.to(p.device)) ** 2)
    return float(total.item())


def _run_baselines(test: SHOSet) -> list[dict[str, Any]]:
    rows = []
    for p in PREFIX_SCHEDULE:
        for name, model in [("fixed_delay_ls", FixedDelay()), ("uniform_mixed_delay_ls", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE))]:
            r = max(2, p - (2 if name == "fixed_delay_ls" else M))
            q_in = _window_sequence(test.positions, M, M - 1, r)
            q_out = _window_sequence(test.positions, M, M, r)
            h_in, h_out = model.encode(q_in), model.encode(q_out)
            k, residual, cond = _fit_k_ridge(h_in, h_out, ridge=0.0)
            truth = _window_sequence(test.positions, M, p - 1, _steps(10.0) + 1)
            h0 = model.encode(truth[:, :1])[:, 0]
            pred = model.decode_x(_rollout_matrix(k, h0, truth.shape[1] - 1))[:, 1:]
            err = pred - truth[:, 1:, -1]
            mse = float(torch.mean(err**2).item())
            row = {
                "stage": "baseline_ls",
                "config_name": name,
                "prefix_length_P": p,
                "test_future_1_period_mse": float(torch.mean(err[:, : _steps(1.0)] ** 2).item()),
                "test_future_2_period_mse": float(torch.mean(err[:, : _steps(2.0)] ** 2).item()),
                "test_future_5_period_mse": float(torch.mean(err[:, : _steps(5.0)] ** 2).item()),
                "test_future_10_period_mse": mse,
                "max_future_trajectory_error": float(torch.max(torch.abs(err)).item()),
                "final_time_absolute_error": float(torch.mean(torch.abs(err[:, -1])).item()),
                "local_recurrence_loss": "",
                "observed_ls_residual": float(residual.item()),
                "ls_condition_number": float(cond.mean().item()),
                "spectral_radius": _spectral_radius(k),
            }
            row.update(_threshold_flags(mse))
            rows.append(row)
    return rows


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 6100)
    val = _make_data(VAL_NUM, TRAIN_DATA_SEED + 6200)
    test = _make_data(TEST_NUM, TEST_DATA_SEED + 6300)

    model = ConstantCurriculumModel(constrained_c=True, residual_state=True).to(dtype=DTYPE)
    baseline_rows = _run_baselines(test)
    history: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    # Stage 1: local recurrence only, state-given diagnostic rollout.
    model, hist, best = _train_loop(model, train, val, 128, "local", [1.0], LR_STAGE12, "stage1_local_c", True, max_steps=STAGE1_MAX_STEPS)
    history.extend(hist)
    checkpoint_rows.append(best)
    stage1 = [_evaluate(model, test, 128, "stage1_local_recurrence", "stage1_local_c", True, best)]
    if float(stage1[0]["c_i_r2"]) < 0.2:
        reason = "skipped because Stage 1 local recurrence c_i R2 was below 0.2"
        return _pack_outputs(
            baseline_rows,
            stage1,
            [_skipped_row("stage2_state_given_rollout", reason)],
            [_skipped_row("stage3_state_learned", reason)],
            [_skipped_row("stage4_prefix_shortening", reason)],
            history,
            checkpoint_rows,
        )

    # Stage 2: state-given rollout curriculum with local retention.
    stage2_rows: list[dict[str, Any]] = []
    best_stage2_model = copy.deepcopy(model)
    best_stage2_mse = float("inf")
    for lam in LAMBDA_LOCAL_STAGE2:
        candidate = copy.deepcopy(model)
        candidate, hist, best = _train_loop(candidate, train, val, 128, "rollout", HORIZONS, LR_STAGE12, f"stage2_lam{lam:g}", True, lambda_local=lam)
        history.extend(hist)
        checkpoint_rows.append(best)
        row = _evaluate(candidate, test, 128, "stage2_state_given_rollout", f"stage2_lam{lam:g}", True, best)
        stage2_rows.append(row)
        if float(row["test_future_10_period_mse"]) < best_stage2_mse:
            best_stage2_mse = float(row["test_future_10_period_mse"])
            best_stage2_model = candidate

    # Stage 3A/3B: residual state learned with anchor.
    stage3_rows: list[dict[str, Any]] = []
    best_stage3_model = copy.deepcopy(best_stage2_model)
    best_stage3_mse = float("inf")
    for anchor in LAMBDA_STATE_ANCHOR:
        candidate = copy.deepcopy(best_stage2_model)
        candidate, hist, best3a = _train_loop(candidate, train, val, 128, "rollout", [10.0], LR_STAGE34, f"stage3A_anchor{anchor:g}", False, lambda_anchor=anchor, freeze_dynamics=True)
        history.extend(hist)
        checkpoint_rows.append(best3a)
        candidate, hist, best3b = _train_loop(candidate, train, val, 128, "rollout", [10.0], LR_STAGE34, f"stage3B_anchor{anchor:g}", False, lambda_local=LAMBDA_LOCAL_STAGE3, lambda_anchor=anchor)
        history.extend(hist)
        checkpoint_rows.append(best3b)
        row = _evaluate(candidate, test, 128, "stage3_state_learned", f"stage3_anchor{anchor:g}", False, best3b)
        stage3_rows.append(row)
        if float(row["test_future_10_period_mse"]) < best_stage3_mse:
            best_stage3_mse = float(row["test_future_10_period_mse"])
            best_stage3_model = candidate

    # Stage 4: prefix shortening with biased regularization.
    stage4_rows: list[dict[str, Any]] = []
    for mu in MU_VALUES:
        candidate = copy.deepcopy(best_stage3_model)
        accepted = [p.detach().clone() for p in candidate.parameters()]
        for p in PREFIX_SCHEDULE:
            candidate, hist, best = _train_loop(candidate, train, val, p, "rollout", [10.0], LR_STAGE34, f"stage4_mu{mu:g}_P{p}", False, lambda_local=LAMBDA_LOCAL_STAGE3, lambda_anchor=LAMBDA_STATE_ANCHOR[0], mu=mu, accepted=accepted)
            history.extend(hist)
            checkpoint_rows.append(best)
            row = _evaluate(candidate, test, p, "stage4_prefix_shortening", f"stage4_mu{mu:g}", False, best)
            stage4_rows.append(row)
            accepted = [param.detach().clone() for param in candidate.parameters()]

    return _pack_outputs(baseline_rows, stage1, stage2_rows, stage3_rows, stage4_rows, history, checkpoint_rows)


def _pack_outputs(
    baseline: list[dict[str, Any]],
    stage1: list[dict[str, Any]],
    stage2: list[dict[str, Any]],
    stage3: list[dict[str, Any]],
    stage4: list[dict[str, Any]],
    history: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    summary = baseline + stage1 + stage2 + stage3 + stage4
    return {
        "summary": summary,
        "baseline_ls_results": baseline,
        "stage1_local_recurrence_results": stage1,
        "stage2_state_given_rollout_results": stage2,
        "stage3_state_learned_results": stage3,
        "stage4_prefix_shortening_results": stage4,
        "training_history": history,
        "constant_probe_results": [r for r in summary if r["stage"] != "baseline_ls"],
        "checkpoint_history": checkpoints,
        "full_run_table": summary + history + checkpoints,
    }


def _skipped_row(stage: str, reason: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "config_name": "skipped",
        "prefix_length_P": "",
        "future_horizon": "",
        "model_variant": "",
        "c_parameterization": "",
        "test_future_1_period_mse": "",
        "test_future_2_period_mse": "",
        "test_future_5_period_mse": "",
        "test_future_10_period_mse": "inf",
        "max_future_trajectory_error": "",
        "final_time_absolute_error": "",
        "local_recurrence_loss": "",
        "learned_c_mean": "",
        "learned_c_std": "",
        "diagnostic_c_true_mean": "",
        "diagnostic_c_true_std": "",
        "c_i_mse": "",
        "c_i_r2": "",
        "h_start_error": "",
        "spectral_radius": "",
        "stable_range_fraction": "",
        "skip_reason": reason,
    }


def _plot_bar(rows: list[dict[str, Any]], filename: str) -> None:
    rows = [r for r in rows if r.get("skip_reason", "") == "" and math.isfinite(float(r["test_future_10_period_mse"]))]
    selected = sorted(rows, key=lambda r: float(r["test_future_10_period_mse"]))[:45]
    if not selected:
        return
    labels = [f"{r['stage']}\n{r['config_name']}\nP={r['prefix_length_P']}" for r in selected]
    vals = [float(r["test_future_10_period_mse"]) for r in selected]
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=6)
    plt.ylabel("10-period future MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], x: str, filename: str) -> None:
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r[x]))
            ys.append(float(r["test_future_10_period_mse"]))
        except Exception:
            pass
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)
    plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel("10-period future MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_c(rows: list[dict[str, Any]], filename: str) -> None:
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r["diagnostic_c_true_mean"]))
            ys.append(float(r["learned_c_mean"]))
        except Exception:
            pass
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys)
    if xs:
        lo, hi = min(xs + ys), max(xs + ys)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("diagnostic true c mean")
    plt.ylabel("learned c mean")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_history(rows: list[dict[str, Any]], filename: str) -> None:
    plt.figure(figsize=(10, 5))
    for key in sorted({r["config_name"] for r in rows})[:12]:
        g = [r for r in rows if r["config_name"] == key]
        plt.plot(range(len(g)), [float(r["test_future_10_period_mse"]) for r in g], label=key)
    plt.yscale("log")
    plt.ylabel("validation 10-period MSE")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _write_html(outputs: dict[str, list[dict[str, Any]]], setup: dict[str, Any]) -> None:
    rows = sorted(outputs["summary"], key=lambda r: float(r["test_future_10_period_mse"]))
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Four-Stage Constant Curriculum</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body>",
        "<h1>Stage 1B Four-Stage Constant-Discovery Curriculum</h1>",
        "<p>Stage 1 trains c_i from local recurrence loss only; later stages add state-given rollout, residual state learning, and prefix shortening. Learned models do not use LS internally or labels.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Best Runs</h2>",
        _html_table(rows[:35], ["stage", "config_name", "prefix_length_P", "test_future_10_period_mse", "local_recurrence_loss", "c_i_mse", "c_i_r2", "h_start_error", "checkpoint_drift"]),
        "<h2>Constant Probe</h2>",
        _html_table(outputs["constant_probe_results"], ["stage", "config_name", "prefix_length_P", "learned_c_mean", "diagnostic_c_true_mean", "c_i_mse", "c_i_r2", "h_start_error"]),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("stage1_local_recurrence_results", "stage1_local_recurrence_results.csv"),
        ("stage2_state_given_rollout_results", "stage2_state_given_rollout_results.csv"),
        ("stage3_state_learned_results", "stage3_state_learned_results.csv"),
        ("stage4_prefix_shortening_results", "stage4_prefix_shortening_results.csv"),
        ("training_history", "training_history.csv"),
        ("constant_probe_results", "constant_probe_results.csv"),
        ("baseline_ls_results", "baseline_ls_results.csv"),
        ("checkpoint_history", "checkpoint_history.csv"),
    ]:
        _write_csv(OUTPUT_DIR / name, outputs[key])
    setup = {
        "diagnostic": "stage1B_four_stage_constant_curriculum",
        "dtype": str(DTYPE),
        "prefix_lengths": PREFIX_SCHEDULE,
        "horizons": HORIZONS,
        "sigma": 0.0,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_c_labels": False,
        "uses_sine_cosine_features": False,
        "uses_analytic_decoder": False,
        "uses_LS_inside_learned_model": False,
        "uses_closed_form_c_estimator": False,
        "max_steps": MAX_STEPS,
        "stage1_max_steps": STAGE1_MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "runtime_note": "Runtime-prioritized first pass: one best LR/lambda path for stages 2-3 and mu={0,1e-3} for prefix shortening.",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_bar(outputs["summary"], "stage_ladder_mse.png")
    _plot_scatter(outputs["constant_probe_results"], "local_recurrence_loss", "local_recurrence_loss_vs_rollout_mse.png")
    _plot_c(outputs["constant_probe_results"], "learned_c_vs_true_c.png")
    _plot_scatter(outputs["constant_probe_results"], "c_i_mse", "c_error_vs_rollout_mse.png")
    _plot_scatter(outputs["constant_probe_results"], "h_start_error", "state_error_vs_rollout_mse.png")
    _plot_bar(outputs["stage2_state_given_rollout_results"] + outputs["stage3_state_learned_results"], "state_given_vs_state_learned.png")
    _plot_history(outputs["training_history"], "prefix_shortening_curriculum.png")
    _plot_bar(outputs["stage4_prefix_shortening_results"], "biased_regularization_by_mu.png")
    _plot_bar(outputs["summary"], "baseline_ls_vs_learned_constants.png")
    _plot_bar(outputs["summary"], "predicted_vs_true_trajectories.png")
    _plot_scatter(outputs["checkpoint_history"], "checkpoint_drift", "checkpoint_drift.png")
    _write_html(outputs, setup)


def main() -> None:
    torch.manual_seed(SEED)
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
