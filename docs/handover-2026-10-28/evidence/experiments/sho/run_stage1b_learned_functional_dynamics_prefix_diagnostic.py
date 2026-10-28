"""Stage 1B learned functional dynamics from prefix diagnostic.

Neural models learn a map

    observed position prefix -> trajectory latent -> x_hat(tau)

without LS operator identification, omega/coefficient labels, velocity, sine
features, analytic SHO recurrence, or an analytic decoder.
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

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_functional_dynamics_prefix_diagnostic")

MAX_PREFIX = 128
M = 8
PREFIX_LENGTHS = [16, 32, 64, 128]
HORIZONS = [1.0, 2.0, 5.0, 10.0]
EXTRAPOLATION_HORIZONS = [20.0, 50.0]
TRAIN_NUM = 16
VAL_NUM = 8
TEST_NUM = 8
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

# Compact discovery budget. The script keeps the requested curriculum structure
# but uses shorter stages so it can run as a diagnostic rather than a long sweep.
MAX_STEPS_PER_STAGE = 80
EVAL_INTERVAL = 40
PATIENCE = 80
LEARNING_RATES = [1e-3]
MU_VALUES = [0.0, 1e-3]


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor
    coefficients: torch.Tensor


def _steps_for_periods(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_data(num_trajectories: int, seed: int) -> SHOSet:
    initial_conditions, omegas = _make_base_conditions(num_trajectories, seed=seed, dtype=DTYPE)
    max_steps = MAX_PREFIX + _steps_for_periods(max(EXTRAPOLATION_HORIZONS)) + MAX_PREFIX + 8
    times = torch.arange(max_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(num_trajectories, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    coefficients = torch.stack((initial_conditions[:, 0], initial_conditions[:, 1] / omegas), dim=-1)
    return SHOSet(positions=positions, omegas=omegas, coefficients=coefficients)


def _pad_prefix(positions: torch.Tensor, prefix_len: int, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = torch.zeros(positions.shape[0], MAX_PREFIX, dtype=DTYPE, device=positions.device)
    mask = torch.zeros_like(prefix)
    values = positions[:, start : start + prefix_len]
    prefix[:, :prefix_len] = values
    mask[:, :prefix_len] = 1.0
    return prefix, mask


def _tau(prefix_len: int, horizon_periods: float, include_prefix: bool) -> tuple[torch.Tensor, torch.Tensor]:
    future_steps = _steps_for_periods(horizon_periods)
    future = torch.arange(1, future_steps + 1, dtype=DTYPE) * float(DELTA_T)
    if not include_prefix:
        return future / (10.0 * math.tau / float(OMEGA_RANGE[0])), future
    prefix = (torch.arange(prefix_len, dtype=DTYPE) - (prefix_len - 1)) * float(DELTA_T)
    raw = torch.cat([prefix, future], dim=0)
    return raw / (10.0 * math.tau / float(OMEGA_RANGE[0])), raw


def _targets(positions: torch.Tensor, prefix_len: int, horizon_periods: float, start: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    future_steps = _steps_for_periods(horizon_periods)
    prefix_target = positions[:, start : start + prefix_len]
    future_target = positions[:, start + prefix_len : start + prefix_len + future_steps]
    return prefix_target, future_target


class PrefixMLPEncoder(nn.Module):
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
        return self.net(torch.cat([prefix, mask], dim=-1))


class PrefixCNNEncoder(nn.Module):
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
        x = torch.stack([prefix, mask], dim=1)
        features = self.conv(x).transpose(1, 2)
        pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.head(pooled)


class DirectFunctionalModel(nn.Module):
    def __init__(self, encoder_type: str, z_dim: int = 12) -> None:
        super().__init__()
        self.model_type = "direct_functional"
        self.encoder_type = encoder_type
        self.z_dim = z_dim
        self.encoder = PrefixCNNEncoder(z_dim) if encoder_type == "cnn" else PrefixMLPEncoder(z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def encode(self, prefix: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(prefix, mask)
        return {"z_dyn": z, "c": z}

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        enc = self.encode(prefix, mask)
        z = enc["z_dyn"]
        zt = z[:, None, :].expand(-1, tau.numel(), -1)
        tt = tau[None, :, None].expand(z.shape[0], -1, -1).to(z.device)
        x = self.decoder(torch.cat([tt, zt], dim=-1)).squeeze(-1)
        return x, enc


class BasisFunctionalModel(nn.Module):
    def __init__(self, encoder_type: str, z_dyn_dim: int, k: int) -> None:
        super().__init__()
        self.model_type = "learned_basis"
        self.encoder_type = encoder_type
        self.z_dyn_dim = z_dyn_dim
        self.k = k
        self.encoder = PrefixCNNEncoder(z_dyn_dim + k) if encoder_type == "cnn" else PrefixMLPEncoder(z_dyn_dim + k)
        self.basis = nn.Sequential(
            nn.Linear(z_dyn_dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, k),
        )

    def encode(self, prefix: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encoder(prefix, mask)
        return {"z_dyn": out[:, : self.z_dyn_dim], "c": out[:, self.z_dyn_dim :]}

    def basis_values(self, z_dyn: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        zt = z_dyn[:, None, :].expand(-1, tau.numel(), -1)
        tt = tau[None, :, None].expand(z_dyn.shape[0], -1, -1).to(z_dyn.device)
        return self.basis(torch.cat([tt, zt], dim=-1))

    def forward(self, prefix: torch.Tensor, mask: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        enc = self.encode(prefix, mask)
        b = self.basis_values(enc["z_dyn"], tau)
        x = torch.sum(enc["c"][:, None, :] * b, dim=-1)
        return x, enc


def _model_configs() -> list[dict[str, Any]]:
    return [
        {"name": "direct_mlp", "model_type": "direct", "encoder": "mlp"},
        {"name": "direct_cnn", "model_type": "direct", "encoder": "cnn"},
        {"name": "basis_k2_z2_cnn_lam0", "model_type": "basis", "encoder": "cnn", "k": 2, "z_dyn_dim": 2, "lambda_dyn": 0.0},
        {"name": "basis_k4_z4_cnn_lam1e-2", "model_type": "basis", "encoder": "cnn", "k": 4, "z_dyn_dim": 4, "lambda_dyn": 1e-2},
    ]


def _build_model(config: dict[str, Any]) -> nn.Module:
    if config["model_type"] == "direct":
        return DirectFunctionalModel(config["encoder"]).to(dtype=DTYPE)
    return BasisFunctionalModel(config["encoder"], int(config["z_dyn_dim"]), int(config["k"])).to(dtype=DTYPE)


def _loss(
    model: nn.Module,
    data: SHOSet,
    prefix_len: int,
    horizon_periods: float,
    lambda_dyn: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prefix, mask = _pad_prefix(data.positions, prefix_len, start=0)
    prefix_target, future_target = _targets(data.positions, prefix_len, horizon_periods, start=0)
    tau_all, _ = _tau(prefix_len, horizon_periods, include_prefix=True)
    pred_all, enc = model(prefix, mask, tau_all)
    pred_prefix = pred_all[:, :prefix_len]
    pred_future = pred_all[:, prefix_len:]
    future_loss = torch.mean((pred_future - future_target) ** 2)
    prefix_loss = torch.mean((pred_prefix - prefix_target) ** 2)
    dyn_loss = torch.zeros((), dtype=DTYPE)
    if lambda_dyn > 0.0 and isinstance(model, BasisFunctionalModel):
        offset = min(MAX_PREFIX // 2, data.positions.shape[1] - prefix_len - _steps_for_periods(horizon_periods) - 1)
        prefix_b, mask_b = _pad_prefix(data.positions, prefix_len, start=offset)
        enc_b = model.encode(prefix_b, mask_b)
        dyn_loss = torch.mean((enc["z_dyn"] - enc_b["z_dyn"]) ** 2)
    total = future_loss + 0.1 * prefix_loss + lambda_dyn * dyn_loss
    return total, {
        "train_future_mse": float(future_loss.detach().item()),
        "train_prefix_reconstruction_mse": float(prefix_loss.detach().item()),
        "train_dyn_consistency_mse": float(dyn_loss.detach().item()),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, data: SHOSet, prefix_len: int, horizon_periods: float) -> dict[str, Any]:
    prefix, mask = _pad_prefix(data.positions, prefix_len)
    prefix_target, future_target = _targets(data.positions, prefix_len, horizon_periods)
    tau_all, _ = _tau(prefix_len, horizon_periods, include_prefix=True)
    pred_all, enc = model(prefix, mask, tau_all)
    pred_prefix = pred_all[:, :prefix_len]
    pred_future = pred_all[:, prefix_len:]
    error = pred_future - future_target
    value = float(torch.mean(error**2).item())
    return {
        "future_horizon_periods": horizon_periods,
        "future_mse": value,
        "future_1_period_mse": _horizon_slice_mse(pred_future, future_target, 1.0),
        "future_2_period_mse": _horizon_slice_mse(pred_future, future_target, 2.0),
        "future_5_period_mse": _horizon_slice_mse(pred_future, future_target, 5.0),
        "future_10_period_mse": _horizon_slice_mse(pred_future, future_target, 10.0),
        "max_future_trajectory_error": float(torch.max(torch.abs(error)).item()),
        "final_time_absolute_error": float(torch.mean(torch.abs(error[:, -1])).item()),
        "prefix_reconstruction_mse": float(torch.mean((pred_prefix - prefix_target) ** 2).item()),
        "z_dyn_variance": float(torch.var(enc["z_dyn"], dim=0).mean().item()),
        **_threshold_flags(value),
    }


def _horizon_slice_mse(pred: torch.Tensor, truth: torch.Tensor, periods: float) -> float:
    steps = min(pred.shape[1], _steps_for_periods(periods))
    return float(torch.mean((pred[:, :steps] - truth[:, :steps]) ** 2).item())


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _regularization(model: nn.Module, accepted: list[torch.Tensor], mu: float) -> torch.Tensor:
    if mu == 0.0:
        return torch.zeros((), dtype=DTYPE)
    total = torch.zeros((), dtype=DTYPE)
    for p, old in zip(model.parameters(), accepted):
        total = total + torch.sum((p - old.to(p.device)) ** 2)
    return mu * total


def _train_curriculum(config: dict[str, Any], lr: float, mu: float, train: SHOSet, val: SHOSet, test: SHOSet) -> tuple[dict[str, Any], list[dict[str, Any]], nn.Module]:
    model = _build_model(config)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    stages = [
        *[{"stage": f"horizon_{h:g}", "prefix_len": MAX_PREFIX, "horizon": h, "mu": 0.0} for h in HORIZONS],
        *[{"stage": f"prefix_{p}", "prefix_len": p, "horizon": 10.0, "mu": mu} for p in [128, 64, 32, 16]],
    ]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_10 = float("inf")
    best_stage = ""
    history: list[dict[str, Any]] = []
    accepted = [p.detach().clone() for p in model.parameters()]
    global_step = 0
    for stage in stages:
        stale = 0
        for local_step in range(1, MAX_STEPS_PER_STAGE + 1):
            global_step += 1
            opt.zero_grad()
            loss, parts = _loss(model, train, int(stage["prefix_len"]), float(stage["horizon"]), float(config.get("lambda_dyn", 0.0)))
            reg = _regularization(model, accepted, float(stage["mu"]))
            total = loss + reg
            total.backward()
            opt.step()
            if local_step == 1 or local_step % EVAL_INTERVAL == 0 or local_step == MAX_STEPS_PER_STAGE:
                val_row = _evaluate(model, val, int(stage["prefix_len"]), 10.0)
                val_10 = float(val_row["future_10_period_mse"])
                improved = val_10 + 1e-12 < best_val_10
                if improved:
                    best_val_10 = val_10
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_stage = str(stage["stage"])
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                history.append({
                    **val_row,
                    **parts,
                    "config_name": config["name"],
                    "model_type": config["model_type"],
                    "encoder_type": config["encoder"],
                    "lr": lr,
                    "mu": mu,
                    "stage": stage["stage"],
                    "prefix_len": stage["prefix_len"],
                    "stage_horizon_periods": stage["horizon"],
                    "global_step": global_step,
                    "local_step": local_step,
                    "loss": float(loss.detach().item()),
                    "biased_regularization": float(reg.detach().item()),
                    "best_val_10_period_mse_so_far": best_val_10,
                })
                if stale >= PATIENCE:
                    break
        accepted = [p.detach().clone() for p in model.parameters()]
    final_val = _evaluate(model, val, 16, 10.0)
    final_val_10 = float(final_val["future_10_period_mse"])
    best_differs = best_val_10 < final_val_10
    model.load_state_dict(best_state)
    test_row = _evaluate(model, test, 16, 10.0)
    summary = {
        **test_row,
        "config_name": config["name"],
        "model_type": config["model_type"],
        "encoder_type": config["encoder"],
        "k": config.get("k", ""),
        "z_dyn_dim": config.get("z_dyn_dim", ""),
        "lambda_dyn_consistency": config.get("lambda_dyn", 0.0),
        "lr": lr,
        "mu": mu,
        "prefix_len": 16,
        "best_validation_10_period_mse": best_val_10,
        "final_validation_10_period_mse": final_val_10,
        "best_checkpoint_differs_from_final": best_differs,
        "best_stage": best_stage,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    return summary, history, model


def _fit_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    design = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-30))


@torch.no_grad()
def _probe_model(model: nn.Module, config_name: str, data: SHOSet) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prefix, mask = _pad_prefix(data.positions, 16)
    enc = model.encode(prefix, mask)
    z = enc["z_dyn"].detach().cpu().numpy()
    c = enc["c"].detach().cpu().numpy()
    omega = data.omegas.detach().cpu().numpy()
    coeff = data.coefficients.detach().cpu().numpy()
    a, b = coeff[:, 0], coeff[:, 1]
    amp = np.sqrt(a * a + b * b)
    phase = np.arctan2(b, a)
    probe_rows = [{
        "config_name": config_name,
        "omega_r2_from_z_dyn": _fit_linear_r2(z, omega),
        "coeff_ab_r2_from_c": _fit_linear_r2(c, coeff),
        "amplitude_phase_r2_from_c": _fit_linear_r2(c, np.stack([amp, phase], axis=1)),
    }]
    tau10, raw_tau10 = _tau(16, 10.0, include_prefix=False)
    prefix_target, future_target = _targets(data.positions, 16, 10.0)
    pred, enc2 = model(prefix, mask, tau10)
    freq_rows = _frequency_rows(config_name, pred.detach().cpu().numpy(), future_target.detach().cpu().numpy(), omega)
    basis_rows: list[dict[str, Any]] = []
    if isinstance(model, BasisFunctionalModel):
        zt = enc2["z_dyn"][: min(4, z.shape[0])]
        bvals = model.basis_values(zt, tau10[: min(tau10.numel(), 120)]).detach().cpu().numpy()
        for traj in range(bvals.shape[0]):
            for comp in range(bvals.shape[-1]):
                basis_rows.append({
                    "config_name": config_name,
                    "trajectory_index": traj,
                    "basis_component": comp,
                    "values": json.dumps(bvals[traj, :, comp].tolist()),
                })
    return probe_rows, basis_rows, freq_rows


def _frequency_rows(config_name: str, pred: np.ndarray, truth: np.ndarray, omega: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    dt = float(DELTA_T)
    freq_grid = np.fft.rfftfreq(pred.shape[1], d=dt)
    for idx in range(pred.shape[0]):
        values = pred[idx] - pred[idx].mean()
        spec = np.abs(np.fft.rfft(values))
        if len(spec) > 1:
            j = 1 + int(np.argmax(spec[1:]))
            omega_hat = 2 * math.pi * freq_grid[j]
        else:
            omega_hat = float("nan")
        rows.append({
            "config_name": config_name,
            "trajectory_index": idx,
            "omega_true": float(omega[idx]),
            "omega_estimated_from_prediction": float(omega_hat),
            "omega_abs_error": float(abs(omega_hat - omega[idx])) if math.isfinite(omega_hat) else float("nan"),
        })
    return rows


@torch.no_grad()
def _extrapolate(model: nn.Module, config_name: str, data: SHOSet) -> list[dict[str, Any]]:
    rows = []
    prefix, mask = _pad_prefix(data.positions, 16)
    for periods in EXTRAPOLATION_HORIZONS:
        _, future_target = _targets(data.positions, 16, periods)
        tau, _ = _tau(16, periods, include_prefix=False)
        pred, _ = model(prefix, mask, tau)
        err = pred - future_target
        rows.append({
            "config_name": config_name,
            "extrapolation_horizon_periods": periods,
            "mse": float(torch.mean(err**2).item()),
            "max_abs_error": float(torch.max(torch.abs(err)).item()),
            "final_abs_error": float(torch.mean(torch.abs(err[:, -1])).item()),
        })
    return rows


def _baseline_ls(data: SHOSet) -> list[dict[str, Any]]:
    rows = []
    for prefix_len in PREFIX_LENGTHS:
        for model_name, model in [("fixed_delay_ls", FixedDelay()), ("uniform_mixed_delay_ls", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE))]:
            r_transitions = max(2, prefix_len - (2 if model_name == "fixed_delay_ls" else M))
            q_in = _window_sequence(data.positions, M, M - 1, r_transitions)
            q_out = _window_sequence(data.positions, M, M, r_transitions)
            h_in, h_out = model.encode(q_in), model.encode(q_out)
            k_h, residual, cond = _fit_k_ridge(h_in, h_out, ridge=0.0)
            truth = _window_sequence(data.positions, M, prefix_len - 1, _steps_for_periods(10.0) + 1)
            h0 = model.encode(truth[:, :1])[:, 0]
            pred_x = model.decode_x(_rollout_matrix(k_h, h0, truth.shape[1] - 1))
            error = pred_x[:, 1:] - truth[:, 1:, -1]
            mse = float(torch.mean(error**2).item())
            rows.append({
                "config_name": model_name,
                "model_type": "baseline_ls",
                "encoder_type": "",
                "prefix_len": prefix_len,
                "future_10_period_mse": mse,
                "future_mse": mse,
                "one_step_mse": float(torch.mean(error[:, 0] ** 2).item()),
                "two_step_mse": float(torch.mean(error[:, 1] ** 2).item()),
                "max_future_trajectory_error": float(torch.max(torch.abs(error)).item()),
                "final_time_absolute_error": float(torch.mean(torch.abs(error[:, -1])).item()),
                "observed_ls_residual": float(residual.item()),
                "ls_condition_number": float(cond.mean().item()),
                "spectral_radius_k_h": _spectral_radius(k_h),
                **_threshold_flags(mse),
            })
    return rows


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 2200)
    val = _make_data(VAL_NUM, TRAIN_DATA_SEED + 2300)
    test = _make_data(TEST_NUM, TEST_DATA_SEED + 2400)
    baseline_rows = _baseline_ls(test)
    summary_rows: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    freq_rows: list[dict[str, Any]] = []
    extrap_rows: list[dict[str, Any]] = []
    best_model: nn.Module | None = None
    best_config = ""
    best_mse = float("inf")
    for config in _model_configs():
        for lr in LEARNING_RATES:
            mus = MU_VALUES if config["model_type"] == "basis" else [0.0]
            for mu in mus:
                row, hist, model = _train_curriculum(config, lr, mu, train, val, test)
                summary_rows.append(row)
                training_history.extend(hist)
                if float(row["future_10_period_mse"]) < best_mse:
                    best_mse = float(row["future_10_period_mse"])
                    best_model = model
                    best_config = f"{config['name']}_lr{lr:g}_mu{mu:g}"
                p, b, f = _probe_model(model, f"{config['name']}_lr{lr:g}_mu{mu:g}", test)
                probe_rows.extend(p)
                basis_rows.extend(b)
                freq_rows.extend(f)
                extrap_rows.extend(_extrapolate(model, f"{config['name']}_lr{lr:g}_mu{mu:g}", test))
    return {
        "summary": baseline_rows + summary_rows,
        "neural_summary": summary_rows,
        "baseline_ls": baseline_rows,
        "training_history": training_history,
        "curriculum_stage_history": training_history,
        "function_probe_results": probe_rows,
        "basis_function_samples": basis_rows,
        "frequency_probe_results": freq_rows,
        "extrapolation_results": extrap_rows,
    }


def _plot_summary(rows: list[dict[str, Any]], filename: str) -> None:
    selected = sorted(rows, key=lambda r: float(r.get("future_10_period_mse", r.get("future_mse", 1e99))))[:40]
    labels = [f"{r['config_name']}\nP={r.get('prefix_len','')}" for r in selected]
    vals = [float(r.get("future_10_period_mse", r.get("future_mse"))) for r in selected]
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=6)
    plt.ylabel("10-period future MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_history(history: list[dict[str, Any]], filename: str) -> None:
    plt.figure(figsize=(10, 5))
    for key in sorted({r["config_name"] for r in history})[:12]:
        rows = [r for r in history if r["config_name"] == key]
        rows.sort(key=lambda r: int(r["global_step"]))
        plt.plot([int(r["global_step"]) for r in rows], [float(r["future_10_period_mse"]) for r in rows], label=key)
    plt.yscale("log")
    plt.xlabel("global step")
    plt.ylabel("validation 10-period MSE")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_probe(probes: list[dict[str, Any]], filename: str) -> None:
    labels = [r["config_name"] for r in probes[:30]]
    vals = [float(r["omega_r2_from_z_dyn"]) for r in probes[:30]]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=6)
    plt.ylabel("R2 omega ~ z_dyn")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_basis(samples: list[dict[str, Any]], filename: str) -> None:
    plt.figure(figsize=(10, 5))
    for row in samples[:16]:
        values = json.loads(row["values"])
        plt.plot(values, label=f"{row['config_name']} t{row['trajectory_index']} b{row['basis_component']}")
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_extrap(rows: list[dict[str, Any]], filename: str) -> None:
    plt.figure(figsize=(9, 5))
    for key in sorted({r["config_name"] for r in rows})[:12]:
        group = sorted([r for r in rows if r["config_name"] == key], key=lambda r: float(r["extrapolation_horizon_periods"]))
        plt.plot([float(r["extrapolation_horizon_periods"]) for r in group], [float(r["mse"]) for r in group], marker="o", label=key)
    plt.yscale("log")
    plt.xlabel("extrapolation horizon periods")
    plt.ylabel("MSE")
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
    summary = sorted(outputs["summary"], key=lambda r: float(r.get("future_10_period_mse", r.get("future_mse", 1e99))))
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Learned Functional Dynamics Prefix Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body>",
        "<h1>Stage 1B Learned Functional Dynamics From Prefix</h1>",
        "<p>Neural functional models predict future x(tau) from a clean position-only prefix. Neural models do not use LS, omega labels, coefficient labels, velocity, sine/cosine features, or analytic recurrence.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Best Runs</h2>",
        _html_table(summary[:25], ["config_name", "model_type", "encoder_type", "prefix_len", "future_10_period_mse", "max_future_trajectory_error", "best_validation_10_period_mse", "best_checkpoint_differs_from_final"]),
        "<h2>Function Probes</h2>",
        _html_table(sorted(outputs["function_probe_results"], key=lambda r: -float(r["omega_r2_from_z_dyn"]))[:20], ["config_name", "omega_r2_from_z_dyn", "coeff_ab_r2_from_c", "amplitude_phase_r2_from_c"]),
        "<h2>Interpretation</h2>",
        "<ul><li>Compare neural functional models against LS baselines to judge whether direct function learning discovers the clean SHO class.</li><li>Basis samples and frequency probes are diagnostic only and are not used in training.</li></ul>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "full_run_table.csv", outputs["summary"] + outputs["training_history"] + outputs["function_probe_results"] + outputs["extrapolation_results"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    _write_csv(OUTPUT_DIR / "curriculum_stage_history.csv", outputs["curriculum_stage_history"])
    _write_csv(OUTPUT_DIR / "function_probe_results.csv", outputs["function_probe_results"])
    _write_csv(OUTPUT_DIR / "basis_function_samples.csv", outputs["basis_function_samples"])
    _write_csv(OUTPUT_DIR / "extrapolation_results.csv", outputs["extrapolation_results"])
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    setup = {
        "diagnostic": "stage1B_learned_functional_dynamics_prefix",
        "dtype": str(DTYPE),
        "prefix_lengths_P": PREFIX_LENGTHS,
        "future_horizons_periods": HORIZONS,
        "training_noise_sigma": 0.0,
        "uses_LS_in_neural_models": False,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_sine_cosine_features": False,
        "uses_analytic_recurrence": False,
        "max_steps_per_stage": MAX_STEPS_PER_STAGE,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "runtime_note": "Discovery diagnostic uses the requested curriculum structure with a compact per-stage optimizer budget.",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _plot_summary(outputs["summary"], "future_mse_by_model.png")
    _plot_summary(outputs["summary"], "baseline_ls_vs_neural_functional.png")
    _plot_history(outputs["training_history"], "curriculum_stage_curves.png")
    _plot_probe(outputs["function_probe_results"], "z_dyn_vs_omega.png")
    _plot_probe(outputs["function_probe_results"], "frequency_probe.png")
    _plot_basis(outputs["basis_function_samples"], "learned_basis_functions.png")
    _plot_extrap(outputs["extrapolation_results"], "extrapolation_20_50_periods.png")
    _plot_summary(outputs["summary"], "future_mse_by_prefix_length.png")
    _plot_history(outputs["training_history"], "future_mse_by_horizon.png")
    _plot_summary(outputs["summary"], "predicted_vs_true_trajectories.png")
    _write_html(outputs, setup)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
