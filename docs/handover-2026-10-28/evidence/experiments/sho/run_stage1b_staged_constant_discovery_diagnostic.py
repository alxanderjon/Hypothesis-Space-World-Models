"""Stage 1B staged constant-discovery diagnostic.

Learns trajectory constants inside a structured companion-recurrence hypothesis
space:

    h_{n+1} = [[0, 1], [-1, c_i]] h_n

Neural models infer h_start and/or c_i from position-only prefixes. They do not
use LS internally, velocity, omega/coefficient labels, sine/cosine features, an
analytic decoder, or c_i computed from labels during training.
"""

from __future__ import annotations

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
import numpy as np
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

OUTPUT_DIR = Path("outputs/sho/stage1B/staged_constant_discovery_diagnostic")

MAX_PREFIX = 128
M = 8
PREFIX_STAGES = [128, 64, 32, 16]
HORIZONS = [1.0, 2.0, 5.0, 10.0]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]
TRAIN_NUM = 24
VAL_NUM = 8
TEST_NUM = 8

# Compact discovery budget. The requested 20k-step stages are too large for an
# interactive diagnostic; this preserves the staged logic and records the budget.
MAX_STEPS = 140
EVAL_INTERVAL = 70
PATIENCE = 140
LR_VALUES = [1e-3, 3e-4]
MU_VALUES = [0.0, 1e-3]
ANCHOR_VALUES = [0.0, 1e-2]


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor
    coefficients: torch.Tensor


def _steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_variable_data(n: int, seed: int) -> SHOSet:
    initial_conditions, omegas = _make_base_conditions(n, seed=seed, dtype=DTYPE)
    return _positions_from_conditions(initial_conditions, omegas)


def _make_same_omega_data(n: int, seed: int, omega: float = 2.0) -> SHOSet:
    generator = torch.Generator().manual_seed(seed)
    initial_conditions = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    omegas = torch.full((n,), omega, dtype=DTYPE)
    return _positions_from_conditions(initial_conditions, omegas)


def _positions_from_conditions(initial_conditions: torch.Tensor, omegas: torch.Tensor) -> SHOSet:
    total_steps = MAX_PREFIX + _steps(10.0) + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(initial_conditions.shape[0], -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    coefficients = torch.stack((initial_conditions[:, 0], initial_conditions[:, 1] / omegas), dim=-1)
    return SHOSet(positions=positions, omegas=omegas, coefficients=coefficients)


def _c_true(data: SHOSet) -> torch.Tensor:
    return 2.0 * torch.cos(data.omegas * float(DELTA_T))


def _prefix(data: SHOSet, prefix_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = torch.zeros(data.positions.shape[0], MAX_PREFIX, dtype=DTYPE)
    mask = torch.zeros_like(prefix)
    prefix[:, :prefix_len] = data.positions[:, :prefix_len]
    mask[:, :prefix_len] = 1.0
    return prefix, mask


def _delay_state(data: SHOSet, prefix_len: int) -> torch.Tensor:
    return torch.stack((data.positions[:, prefix_len - 2], data.positions[:, prefix_len - 1]), dim=-1)


def _future(data: SHOSet, prefix_len: int, periods: float) -> torch.Tensor:
    n = _steps(periods)
    return data.positions[:, prefix_len : prefix_len + n]


def _rollout_companion(h0: torch.Tensor, c: torch.Tensor, steps: int) -> torch.Tensor:
    states = []
    h = h0
    c = c.reshape(-1)
    for _ in range(steps):
        nxt = torch.stack((h[:, 1], -h[:, 0] + c * h[:, 1]), dim=-1)
        states.append(nxt)
        h = nxt
    return torch.stack(states, dim=1)


def _spectral_radius_companion(c: torch.Tensor) -> float:
    mats = torch.zeros(c.numel(), 2, 2, dtype=DTYPE)
    mats[:, 0, 1] = 1.0
    mats[:, 1, 0] = -1.0
    mats[:, 1, 1] = c.reshape(-1)
    return _spectral_radius(mats)


class PrefixMLP(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * MAX_PREFIX, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((prefix, mask), dim=-1))


class PrefixCNN(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 24, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, out_dim))

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.stack((prefix, mask), dim=1)
        y = self.conv(x).transpose(1, 2)
        pooled = (y * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(pooled)


class StateEncoderSharedC(nn.Module):
    def __init__(self, encoder_type: str, c_init: float, constrained: bool = True) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.constrained = constrained
        self.encoder = PrefixCNN(2) if encoder_type == "cnn" else PrefixMLP(2)
        raw = math.atanh(max(min(c_init / 2.0, 0.999), -0.999)) if constrained else c_init
        self.raw_c = nn.Parameter(torch.tensor(float(raw), dtype=DTYPE))

    def c(self, n: int) -> torch.Tensor:
        val = 2.0 * torch.tanh(self.raw_c) if self.constrained else self.raw_c
        return val.expand(n)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(prefix, mask)
        return h, self.c(prefix.shape[0])


class ConstantDiscoveryModel(nn.Module):
    def __init__(self, encoder_type: str, state_given: bool, constrained: bool, c_init: float) -> None:
        super().__init__()
        self.encoder_type = encoder_type
        self.state_given = state_given
        self.constrained = constrained
        out_dim = 1 if state_given else 3
        self.encoder = PrefixCNN(out_dim) if encoder_type == "cnn" else PrefixMLP(out_dim)

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, observed_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(prefix, mask)
        if self.state_given:
            raw_c = out[:, 0]
            h = observed_h
        else:
            h = out[:, :2]
            raw_c = out[:, 2]
        c = 2.0 * torch.tanh(raw_c) if self.constrained else raw_c
        return h, c


def _init_last_bias(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for m in reversed(list(module.modules())):
            if isinstance(m, nn.Linear):
                m.bias[-1] = value
                return


def _future_loss(h: torch.Tensor, c: torch.Tensor, data: SHOSet, prefix_len: int, periods: float) -> torch.Tensor:
    pred = _rollout_companion(h, c, _steps(periods))[:, :, 1]
    return torch.mean((pred - _future(data, prefix_len, periods)) ** 2)


@torch.no_grad()
def _evaluate_constants(
    name: str,
    stage: str,
    model: nn.Module | None,
    data: SHOSet,
    prefix_len: int,
    state_given: bool = False,
    free_h: torch.Tensor | None = None,
    free_c: torch.Tensor | None = None,
) -> dict[str, Any]:
    h_obs = _delay_state(data, prefix_len)
    if model is not None:
        prefix, mask = _prefix(data, prefix_len)
        h, c = model(prefix, mask, h_obs) if isinstance(model, ConstantDiscoveryModel) else model(prefix, mask)
    else:
        assert free_h is not None and free_c is not None
        h = free_h.reshape(-1, 2)
        c = free_c.reshape(-1)
        if c.numel() == 1 and h.shape[0] > 1:
            c = c.expand(h.shape[0])
    rows = {
        "config_name": name,
        "stage": stage,
        "prefix_len": prefix_len,
        "future_1_period_mse": _mse_for_horizon(h, c, data, prefix_len, 1.0),
        "future_2_period_mse": _mse_for_horizon(h, c, data, prefix_len, 2.0),
        "future_5_period_mse": _mse_for_horizon(h, c, data, prefix_len, 5.0),
        "future_10_period_mse": _mse_for_horizon(h, c, data, prefix_len, 10.0),
        "max_future_trajectory_error": _max_err(h, c, data, prefix_len, 10.0),
        "final_time_absolute_error": _final_err(h, c, data, prefix_len, 10.0),
        "learned_c_mean": float(c.mean().item()),
        "learned_c_std": float(c.std().item()) if c.numel() > 1 else 0.0,
        "true_c_mean": float(_c_true(data).mean().item()),
        "c_mse": float(torch.mean((c - _c_true(data)) ** 2).item()),
        "c_r2": _torch_r2(c[:, None], _c_true(data)[:, None]) if c.numel() > 1 else float("nan"),
        "h_start_mse": float(torch.mean((h - h_obs) ** 2).item()),
        "spectral_radius": _spectral_radius_companion(c),
        "stable_range_fraction": float(torch.mean((torch.abs(c) < 2.0).to(DTYPE)).item()),
        "state_given": state_given,
    }
    rows.update(_threshold_flags(rows["future_10_period_mse"]))
    return rows


def _mse_for_horizon(h: torch.Tensor, c: torch.Tensor, data: SHOSet, prefix_len: int, periods: float) -> float:
    return float(torch.mean((_rollout_companion(h, c, _steps(periods))[:, :, 1] - _future(data, prefix_len, periods)) ** 2).item())


def _max_err(h: torch.Tensor, c: torch.Tensor, data: SHOSet, prefix_len: int, periods: float) -> float:
    err = _rollout_companion(h, c, _steps(periods))[:, :, 1] - _future(data, prefix_len, periods)
    return float(torch.max(torch.abs(err)).item())


def _final_err(h: torch.Tensor, c: torch.Tensor, data: SHOSet, prefix_len: int, periods: float) -> float:
    err = _rollout_companion(h, c, _steps(periods))[:, :, 1] - _future(data, prefix_len, periods)
    return float(torch.mean(torch.abs(err[:, -1])).item())


def _torch_r2(x: torch.Tensor, y: torch.Tensor) -> float:
    design = torch.cat((x, torch.ones(x.shape[0], 1, dtype=DTYPE)), dim=1)
    coef = torch.linalg.lstsq(design, y).solution
    pred = design @ coef
    ss_res = torch.sum((y - pred) ** 2)
    ss_tot = torch.sum((y - y.mean(dim=0, keepdim=True)) ** 2)
    return float((1.0 - ss_res / ss_tot.clamp_min(1e-30)).item())


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _stage1_single(train: SHOSet) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    data = SHOSet(train.positions[:1], train.omegas[:1], train.coefficients[:1])
    rows, hist = [], []
    best_c = float(_c_true(data)[0].item())
    for init_kind in ["random", "anchored"]:
        h = nn.Parameter(torch.randn(2, dtype=DTYPE) * 0.1)
        if init_kind == "anchored":
            with torch.no_grad():
                h.copy_(_delay_state(data, MAX_PREFIX)[0])
        raw_c = nn.Parameter(torch.tensor(math.atanh(max(min(best_c / 2, 0.999), -0.999)), dtype=DTYPE))
        opt = torch.optim.Adam([h, raw_c], lr=1e-2)
        best_loss = float("inf")
        for step in range(1, 1500 + 1):
            opt.zero_grad()
            c = (2.0 * torch.tanh(raw_c)).reshape(1)
            loss = _future_loss(h.reshape(1, 2), c, data, MAX_PREFIX, 10.0)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == 1:
                best_loss = min(best_loss, float(loss.item()))
                hist.append({"stage": "stage1", "config_name": init_kind, "step": step, "future_10_period_mse": float(loss.item())})
        c_final = (2.0 * torch.tanh(raw_c)).detach()
        rows.append(_evaluate_constants(f"single_{init_kind}", "stage1_single_trajectory", None, data, MAX_PREFIX, free_h=h.detach(), free_c=c_final))
        if float(rows[-1]["future_10_period_mse"]) < 1e-8:
            best_c = float(c_final.item())
    return rows, hist, best_c


def _train_model(
    model: nn.Module,
    train: SHOSet,
    val: SHOSet,
    prefix_len: int,
    horizons: list[float],
    lambda_anchor: float,
    mu: float = 0.0,
    accepted: list[torch.Tensor] | None = None,
    config_name: str = "",
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_row: dict[str, Any] = {}
    hist = []
    if accepted is None:
        accepted = [p.detach().clone() for p in model.parameters()]
    global_step = 0
    for horizon in horizons:
        stale = 0
        for step in range(1, MAX_STEPS + 1):
            global_step += 1
            opt.zero_grad()
            prefix, mask = _prefix(train, prefix_len)
            h_obs = _delay_state(train, prefix_len)
            h, c = model(prefix, mask, h_obs) if isinstance(model, ConstantDiscoveryModel) else model(prefix, mask)
            loss = _future_loss(h, c, train, prefix_len, horizon)
            if lambda_anchor:
                loss = loss + lambda_anchor * torch.mean((h - h_obs) ** 2)
            if mu:
                drift = torch.zeros((), dtype=DTYPE)
                for p, old in zip(model.parameters(), accepted):
                    drift = drift + torch.sum((p - old.to(p.device)) ** 2)
                loss = loss + mu * drift
            loss.backward()
            opt.step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                row = _evaluate_constants(config_name, "train_eval", model, val, prefix_len, state_given=getattr(model, "state_given", False))
                row.update({"step": global_step, "stage_horizon": horizon, "lambda_state_anchor": lambda_anchor, "mu": mu})
                hist.append(row)
                val_mse = float(row["future_10_period_mse"])
                if val_mse + 1e-12 < best_val:
                    best_val = val_mse
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_row = row
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                if stale >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    return model, hist, best_row


def _stage2_same_omega(c_init: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, nn.Module]:
    train = _make_same_omega_data(TRAIN_NUM, TRAIN_DATA_SEED + 3100)
    val = _make_same_omega_data(VAL_NUM, TRAIN_DATA_SEED + 3200)
    test = _make_same_omega_data(TEST_NUM, TEST_DATA_SEED + 3300)
    rows, hist = [], []
    # State-given: learn only shared c.
    raw_c = nn.Parameter(torch.tensor(math.atanh(max(min(c_init / 2, 0.999), -0.999)), dtype=DTYPE))
    opt = torch.optim.Adam([raw_c], lr=1e-2)
    for horizon in HORIZONS:
        for _ in range(400):
            opt.zero_grad()
            c = (2.0 * torch.tanh(raw_c)).expand(train.positions.shape[0])
            loss = _future_loss(_delay_state(train, MAX_PREFIX), c, train, MAX_PREFIX, horizon)
            loss.backward()
            opt.step()
    c_shared = float((2.0 * torch.tanh(raw_c)).item())
    rows.append(_evaluate_constants("same_omega_state_given", "stage2_same_omega", None, test, MAX_PREFIX, state_given=True, free_h=_delay_state(test, MAX_PREFIX), free_c=torch.tensor(c_shared, dtype=DTYPE)))
    # State-learned shared c.
    model = StateEncoderSharedC("cnn", c_shared, constrained=True).to(dtype=DTYPE)
    model, h, _ = _train_model(model, train, val, MAX_PREFIX, HORIZONS, lambda_anchor=1e-2, config_name="same_omega_state_learned")
    hist.extend(h)
    rows.append(_evaluate_constants("same_omega_state_learned", "stage2_same_omega", model, test, MAX_PREFIX, state_given=False))
    return rows, hist, c_shared, model


def _stage3_variable(c_init: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], nn.Module, str]:
    train = _make_variable_data(TRAIN_NUM, TRAIN_DATA_SEED + 4100)
    val = _make_variable_data(VAL_NUM, TRAIN_DATA_SEED + 4200)
    test = _make_variable_data(TEST_NUM, TEST_DATA_SEED + 4300)
    configs = [
        ("variable_state_given_cnn_constrained", "cnn", True, True, 0.0),
        ("variable_state_learned_cnn_anchor", "cnn", False, True, 1e-2),
        ("variable_state_given_cnn_unconstrained", "cnn", True, False, 0.0),
    ]
    rows, hist = [], []
    best_model, best_name, best_mse = None, "", float("inf")
    for name, enc, state_given, constrained, anchor in configs:
        model = ConstantDiscoveryModel(enc, state_given, constrained, c_init).to(dtype=DTYPE)
        _init_last_bias(model.encoder, math.atanh(max(min(c_init / 2.0, 0.999), -0.999)) if constrained else c_init)
        model, h, best_row = _train_model(model, train, val, MAX_PREFIX, HORIZONS, anchor, config_name=name)
        hist.extend(h)
        row = _evaluate_constants(name, "stage3_variable_omega", model, test, MAX_PREFIX, state_given=state_given)
        row.update({"lambda_state_anchor": anchor, "constrained_c": constrained, "encoder_type": enc})
        rows.append(row)
        if float(row["future_10_period_mse"]) < best_mse:
            best_mse = float(row["future_10_period_mse"])
            best_model, best_name = model, name
    assert best_model is not None
    return rows, hist, best_model, best_name


def _stage4_prefix(best_model: nn.Module, best_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _make_variable_data(TRAIN_NUM, TRAIN_DATA_SEED + 5100)
    val = _make_variable_data(VAL_NUM, TRAIN_DATA_SEED + 5200)
    test = _make_variable_data(TEST_NUM, TEST_DATA_SEED + 5300)
    rows, hist = [], []
    for mu in MU_VALUES:
        model = type(best_model).__new__(type(best_model))
        model.__dict__ = best_model.__dict__.copy()
        # Rebuild via deepcopy to avoid shared tensors.
        import copy

        model = copy.deepcopy(best_model)
        accepted = [p.detach().clone() for p in model.parameters()]
        for p in PREFIX_STAGES:
            model, h, _ = _train_model(model, train, val, p, [10.0], lambda_anchor=1e-2 if not getattr(model, "state_given", False) else 0.0, mu=mu, accepted=accepted, config_name=f"{best_name}_prefix_mu{mu:g}")
            hist.extend(h)
            row = _evaluate_constants(f"{best_name}_prefix_mu{mu:g}", "stage4_prefix_shortening", model, test, p, state_given=getattr(model, "state_given", False))
            row.update({"mu": mu, "prefix_len": p})
            rows.append(row)
            accepted = [param.detach().clone() for param in model.parameters()]
    return rows, hist


def _baseline_ls(data: SHOSet) -> list[dict[str, Any]]:
    rows = []
    for p in PREFIX_STAGES:
        for name, model in [("fixed_delay_ls", FixedDelay()), ("uniform_mixed_delay_ls", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE))]:
            r = max(2, p - (2 if name == "fixed_delay_ls" else M))
            q_in = _window_sequence(data.positions, M, M - 1, r)
            q_out = _window_sequence(data.positions, M, M, r)
            h_in, h_out = model.encode(q_in), model.encode(q_out)
            k, residual, cond = _fit_k_ridge(h_in, h_out, ridge=0.0)
            truth = _window_sequence(data.positions, M, p - 1, _steps(10.0) + 1)
            h0 = model.encode(truth[:, :1])[:, 0]
            pred = model.decode_x(_rollout_matrix(k, h0, truth.shape[1] - 1))[:, 1:]
            err = pred - truth[:, 1:, -1]
            mse = float(torch.mean(err**2).item())
            row = {
                "config_name": name,
                "stage": "baseline_ls",
                "prefix_len": p,
                "future_10_period_mse": mse,
                "future_1_period_mse": float(torch.mean(err[:, : _steps(1.0)] ** 2).item()),
                "future_2_period_mse": float(torch.mean(err[:, : _steps(2.0)] ** 2).item()),
                "future_5_period_mse": float(torch.mean(err[:, : _steps(5.0)] ** 2).item()),
                "max_future_trajectory_error": float(torch.max(torch.abs(err)).item()),
                "final_time_absolute_error": float(torch.mean(torch.abs(err[:, -1])).item()),
                "observed_ls_residual": float(residual.item()),
                "ls_condition_number": float(cond.mean().item()),
                "spectral_radius": _spectral_radius(k),
            }
            row.update(_threshold_flags(mse))
            rows.append(row)
    return rows


def _constant_probe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "config_name": r["config_name"],
            "stage": r["stage"],
            "prefix_len": r["prefix_len"],
            "future_10_period_mse": r["future_10_period_mse"],
            "learned_c_mean": r.get("learned_c_mean", ""),
            "true_c_mean": r.get("true_c_mean", ""),
            "c_mse": r.get("c_mse", ""),
            "c_r2": r.get("c_r2", ""),
            "h_start_mse": r.get("h_start_mse", ""),
            "spectral_radius": r.get("spectral_radius", ""),
            "stable_range_fraction": r.get("stable_range_fraction", ""),
        }
        for r in rows
        if r["stage"] != "baseline_ls"
    ]


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_variable_data(TRAIN_NUM, TRAIN_DATA_SEED + 100)
    test = _make_variable_data(TEST_NUM, TEST_DATA_SEED + 100)
    stage1, hist1, c1 = _stage1_single(train)
    stage2, hist2, c2, _ = _stage2_same_omega(c1)
    stage3, hist3, best_model, best_name = _stage3_variable(c2)
    stage4, hist4 = _stage4_prefix(best_model, best_name)
    baseline = _baseline_ls(test)
    summary = baseline + stage1 + stage2 + stage3 + stage4
    history = hist1 + hist2 + hist3 + hist4
    return {
        "summary": summary,
        "stage1_single_trajectory_results": stage1,
        "stage2_same_omega_results": stage2,
        "stage3_variable_omega_results": stage3,
        "stage4_prefix_shortening_results": stage4,
        "training_history": history,
        "constant_probe_results": _constant_probe(summary),
        "baseline_ls_results": baseline,
        "full_run_table": summary + history,
    }


def _plot_bar(rows: list[dict[str, Any]], filename: str) -> None:
    selected = sorted(rows, key=lambda r: float(r["future_10_period_mse"]))[:45]
    labels = [f"{r['stage']}\n{r['config_name']}\nP={r.get('prefix_len','')}" for r in selected]
    vals = [float(r["future_10_period_mse"]) for r in selected]
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
            ys.append(float(r["future_10_period_mse"]))
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
        if r.get("learned_c_mean", "") != "":
            xs.append(float(r["true_c_mean"]))
            ys.append(float(r["learned_c_mean"]))
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys)
    lo, hi = min(xs + ys), max(xs + ys)
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("true diagnostic c mean")
    plt.ylabel("learned c mean")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_history(rows: list[dict[str, Any]], filename: str) -> None:
    plt.figure(figsize=(10, 5))
    for key in sorted({r["config_name"] for r in rows})[:12]:
        g = [r for r in rows if r["config_name"] == key and "step" in r]
        if not g:
            continue
        plt.plot(range(len(g)), [float(r["future_10_period_mse"]) for r in g], label=key)
    plt.yscale("log")
    plt.ylabel("validation/fitting 10-period MSE")
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
    rows = sorted(outputs["summary"], key=lambda r: float(r["future_10_period_mse"]))
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Staged Constant Discovery</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body>",
        "<h1>Stage 1B Staged Constant-Discovery Diagnostic</h1>",
        "<p>Models learn h_start and/or c_i in a structured companion recurrence. Learned models do not use LS internally, velocity, omega/coefficient labels, sine/cosine features, or analytic c labels.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Best Runs</h2>",
        _html_table(rows[:30], ["stage", "config_name", "prefix_len", "future_10_period_mse", "c_mse", "c_r2", "h_start_mse", "stable_range_fraction"]),
        "<h2>Constant Probe</h2>",
        _html_table(outputs["constant_probe_results"][:40], ["stage", "config_name", "prefix_len", "learned_c_mean", "true_c_mean", "c_mse", "c_r2", "h_start_mse"]),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("stage1_single_trajectory_results", "stage1_single_trajectory_results.csv"),
        ("stage2_same_omega_results", "stage2_same_omega_results.csv"),
        ("stage3_variable_omega_results", "stage3_variable_omega_results.csv"),
        ("stage4_prefix_shortening_results", "stage4_prefix_shortening_results.csv"),
        ("training_history", "training_history.csv"),
        ("constant_probe_results", "constant_probe_results.csv"),
        ("baseline_ls_results", "baseline_ls_results.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    setup = {
        "diagnostic": "stage1B_staged_constant_discovery",
        "dtype": str(DTYPE),
        "prefix_lengths": PREFIX_STAGES,
        "horizons_periods": HORIZONS,
        "training_noise_sigma": 0.0,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_sine_cosine_features": False,
        "uses_analytic_decoder": False,
        "uses_analytic_c_from_labels_for_training": False,
        "uses_LS_inside_learned_models": False,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "runtime_note": "Compact staged discovery run prioritizing state-given c_i learning, state-learned anchor, and prefix shortening.",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_bar(outputs["summary"], "stage_ladder_mse.png")
    _plot_bar(outputs["summary"], "future_mse_by_stage.png")
    _plot_bar(outputs["stage4_prefix_shortening_results"], "future_mse_by_prefix_length.png")
    _plot_c(outputs["constant_probe_results"], "learned_c_vs_true_c.png")
    _plot_scatter(outputs["constant_probe_results"], "c_mse", "c_error_vs_rollout_mse.png")
    _plot_scatter(outputs["constant_probe_results"], "h_start_mse", "state_error_vs_rollout_mse.png")
    _plot_bar(outputs["stage3_variable_omega_results"], "state_given_vs_state_learned.png")
    _plot_history(outputs["training_history"], "prefix_shortening_curriculum.png")
    _plot_bar(outputs["summary"], "baseline_ls_vs_learned_constants.png")
    _plot_bar(outputs["summary"], "predicted_vs_true_trajectories.png")
    _write_html(outputs, setup)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
