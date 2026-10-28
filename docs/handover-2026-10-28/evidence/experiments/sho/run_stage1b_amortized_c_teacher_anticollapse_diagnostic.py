"""Stage 1B amortized c-teacher anti-collapse diagnostic.

Focuses only on the bottleneck

    position prefix -> trajectory-specific recurrence constant c_i.

Teachers are created by self-supervised free-c optimization on each observed
prefix. Encoders are trained to predict standardized teacher constants with
anti-collapse losses. No velocity, omega labels, coefficient labels, simulator
c labels, LS inside learned encoders, closed-form c estimator, analytic decoder,
or future points are used for teacher construction or encoder training.
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

OUTPUT_DIR = Path("outputs/sho/stage1B/amortized_c_teacher_anticollapse_diagnostic")

P = 128
M = 8
TRAIN_NUM = 56
VAL_NUM = 20
TEST_NUM = 20
J_ALL = [1, 2, 4, 8, 16, 32, 64]
DRIFT_HEAVY = {1: 1.0, 2: 1.0, 4: 2.0, 8: 4.0, 16: 4.0, 32: 4.0, 64: 4.0}
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

# Runtime-prioritized first pass.
TEACHER_STEPS = 450
DISTILL_STEPS = 550
CURRICULUM_STEPS = 250
FINETUNE_STEPS = 200
EVAL_INTERVAL = 100
PATIENCE = 250
BATCH_SIZE = 48


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor


@dataclass(frozen=True)
class TeacherSet:
    data: SHOSet
    c_free: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor

    @property
    def z(self) -> torch.Tensor:
        return (self.c_free - self.mu) / self.sigma.clamp_min(1e-12)


def _steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_data(n: int, seed: int, omega_range: tuple[float, float] | None = None) -> SHOSet:
    if omega_range is None:
        initial_conditions, omegas = _make_base_conditions(n, seed=seed, dtype=DTYPE)
    else:
        gen = torch.Generator().manual_seed(seed)
        initial_conditions = 2.0 * torch.rand(n, 2, generator=gen, dtype=DTYPE) - 1.0
        lo, hi = omega_range
        omegas = lo + (hi - lo) * torch.rand(n, generator=gen, dtype=DTYPE)
    total_steps = P + _steps(10.0) + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(n, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _sim_c(data: SHOSet) -> torch.Tensor:
    return 2.0 * torch.cos(data.omegas * float(DELTA_T))


def _prefix_terms(c: torch.Tensor, positions: torch.Tensor) -> dict[int, torch.Tensor]:
    x = positions[:, :P]
    terms = {}
    for j in J_ALL:
        starts = torch.arange(1, P - j, device=x.device)
        a = x[:, starts - 1]
        b = x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        terms[j] = torch.mean((pred - truth) ** 2)
    return terms


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    terms = _prefix_terms(c, positions)
    return sum(DRIFT_HEAVY[j] * terms[j] for j in J_ALL)


def _future_rollout_mse(c: torch.Tensor, data: SHOSet, periods: float) -> tuple[float, float, float]:
    steps = _steps(periods)
    a = data.positions[:, P - 2]
    b = data.positions[:, P - 1]
    preds = []
    for _ in range(steps):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = data.positions[:, P : P + steps]
    err = pred - truth
    return (
        float(torch.mean(err**2).item()),
        float(torch.max(torch.abs(err)).item()),
        float(torch.mean(torch.abs(err[:, -1])).item()),
    )


def _free_teacher(data: SHOSet) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    init = torch.full((data.positions.shape[0],), float(_sim_c(data).mean().item()), dtype=DTYPE)
    raw = nn.Parameter(torch.atanh((init / 2.0).clamp(-0.999, 0.999)))
    opt = torch.optim.Adam([raw], lr=1e-2)
    best_raw = raw.detach().clone()
    best = float("inf")
    hist = []
    for step in range(1, TEACHER_STEPS + 1):
        opt.zero_grad()
        c = 2.0 * torch.tanh(raw)
        loss = _prefix_loss(c, data.positions)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == TEACHER_STEPS:
            c_now = (2.0 * torch.tanh(raw)).detach()
            row = _metrics(
                "free_c_teacher",
                "free_c_teacher",
                c_now,
                c_now,
                data,
                None,
                {"step": step, "prefix_loss": float(loss.item()), "prefix_multistep_loss": float(loss.item())},
            )
            hist.append(row)
            if float(row["prefix_multistep_loss"]) < best:
                best = float(row["prefix_multistep_loss"])
                best_raw = raw.detach().clone()
    with torch.no_grad():
        raw.copy_(best_raw)
    return (2.0 * torch.tanh(raw)).detach(), hist


class MLPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P, 160), nn.SiLU(), nn.Linear(160, 160), nn.SiLU(), nn.Linear(160, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x.unsqueeze(1)).mean(dim=-1)).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class TCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8]))
        self.head = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(torch.nn.functional.silu(self.inp(x.unsqueeze(1))))
        return self.head(y.mean(dim=-1)).squeeze(-1)


def _make_encoder(kind: str) -> nn.Module:
    if kind == "mlp":
        return MLPEncoder().to(dtype=DTYPE)
    if kind == "cnn":
        return CNNEncoder().to(dtype=DTYPE)
    if kind == "tcn":
        return TCNEncoder().to(dtype=DTYPE)
    raise ValueError(kind)


def _loss_parts(z_pred: torch.Tensor, z_true: torch.Tensor, variant: str) -> tuple[torch.Tensor, dict[str, float]]:
    mse = torch.mean((z_pred - z_true) ** 2)
    var = (torch.std(z_pred) - torch.std(z_true)) ** 2
    mean = (torch.mean(z_pred) - torch.mean(z_true)) ** 2
    corr_val = _corr_tensor(z_pred, z_true)
    corr_loss = 1.0 - corr_val
    diff_pred = z_pred[:, None] - z_pred[None, :]
    diff_true = z_true[:, None] - z_true[None, :]
    pair = torch.mean((diff_pred - diff_true) ** 2)
    if variant == "mse":
        total = mse
    elif variant == "variance":
        total = mse + var
    elif variant == "correlation":
        total = mse + corr_loss
    elif variant == "pairwise":
        total = mse + pair
    elif variant == "combined":
        total = mse + var + corr_loss + pair + 0.1 * mean
    else:
        raise ValueError(variant)
    return total, {
        "z_mse": float(mse.detach().item()),
        "batch_variance_match": float(var.detach().item()),
        "batch_correlation": float(corr_val.detach().item()),
        "pairwise_difference_error": float(pair.detach().item()),
        "batch_mean_match": float(mean.detach().item()),
    }


def _corr_tensor(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xc, yc = x - x.mean(), y - y.mean()
    return torch.sum(xc * yc) / torch.sqrt(torch.sum(xc**2) * torch.sum(yc**2)).clamp_min(1e-30)


def _train_encoder(train: TeacherSet, val: TeacherSet, test: TeacherSet, kind: str, variant: str, steps: int = DISTILL_STEPS, lr: float = 1e-3) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = _make_encoder(kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train.data.positions.shape[0]
    hist = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    stale = 0
    gen = torch.Generator().manual_seed(SEED + 991)
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        x = train.data.positions[idx, :P]
        z = train.z[idx]
        opt.zero_grad()
        pred = model(x)
        loss, parts = _loss_parts(pred, z, variant)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == steps:
            row = _eval_encoder(model, val, kind, variant, {"step": step, "train_loss": float(loss.item()), **parts})
            hist.append(row)
            score = float(row["validation_teacher_z_mse"])
            if score < best:
                best = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final = _eval_encoder(model, test, kind, variant, {"best_validation_teacher_z_mse": best})
    return model, hist, final


@torch.no_grad()
def _eval_encoder(model: nn.Module, dataset: TeacherSet, kind: str, variant: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    z_pred = model(dataset.data.positions[:, :P])
    c_pred = dataset.mu + dataset.sigma * z_pred
    row = _metrics("amortized_encoder", f"{kind}_{variant}", c_pred, dataset.c_free, dataset.data, dataset, {
        "encoder_type": kind,
        "loss_variant": variant,
        "validation_teacher_z_mse": float(torch.mean((z_pred - dataset.z) ** 2).item()),
        "test_teacher_c_mse": float(torch.mean((c_pred - dataset.c_free) ** 2).item()),
    })
    z_loss, parts = _loss_parts(z_pred, dataset.z, "combined")
    row.update(parts)
    if extra:
        row.update(extra)
    return row


def _metrics(stage: str, name: str, c_pred: torch.Tensor, c_teacher: torch.Tensor, data: SHOSet, teacher_set: TeacherSet | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    sim_c = _sim_c(data)
    mse10, maxerr, finalerr = _future_rollout_mse(c_pred, data, 10.0)
    row = {
        "stage": stage,
        "config_name": name,
        "train_c_mse_against_teacher": "",
        "validation_c_mse_against_teacher": "",
        "test_c_mse_against_teacher": float(torch.mean((c_pred - c_teacher) ** 2).item()),
        "diagnostic_c_r2": _r2(c_pred[:, None], sim_c[:, None]),
        "diagnostic_c_mse": float(torch.mean((c_pred - sim_c) ** 2).item()),
        "correlation": _corr(c_pred, sim_c),
        "collapse_ratio": float((torch.std(c_pred) / torch.std(c_teacher).clamp_min(1e-12)).item()),
        "pred_c_mean": float(c_pred.mean().item()),
        "pred_c_std": float(c_pred.std().item()),
        "teacher_c_mean": float(c_teacher.mean().item()),
        "teacher_c_std": float(c_teacher.std().item()),
        "state_given_1_period_mse": _future_rollout_mse(c_pred, data, 1.0)[0],
        "state_given_2_period_mse": _future_rollout_mse(c_pred, data, 2.0)[0],
        "state_given_5_period_mse": _future_rollout_mse(c_pred, data, 5.0)[0],
        "state_given_10_period_mse": mse10,
        "max_trajectory_error": maxerr,
        "final_time_absolute_error": finalerr,
    }
    row.update(_threshold_flags(mse10))
    if extra:
        row.update(extra)
    return row


def _r2(x: torch.Tensor, y: torch.Tensor) -> float:
    design = torch.cat((x, torch.ones(x.shape[0], 1, dtype=DTYPE)), dim=1)
    coef = torch.linalg.lstsq(design, y).solution
    pred = design @ coef
    return float((1.0 - torch.sum((y - pred) ** 2) / torch.sum((y - y.mean(0, keepdim=True)) ** 2).clamp_min(1e-30)).item())


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    return float((torch.sum(xc * yc) / torch.sqrt(torch.sum(xc**2) * torch.sum(yc**2)).clamp_min(1e-30)).item())


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _make_teacher_sets(train: SHOSet, val: SHOSet, test: SHOSet) -> tuple[TeacherSet, TeacherSet, TeacherSet, list[dict[str, Any]]]:
    rows = []
    print("building free-c teachers: train", flush=True)
    c_train, h = _free_teacher(train)
    rows.extend(h)
    print("building free-c teachers: val", flush=True)
    c_val, h = _free_teacher(val)
    rows.extend(h)
    print("building free-c teachers: test", flush=True)
    c_test, h = _free_teacher(test)
    rows.extend(h)
    mu, sigma = c_train.mean(), c_train.std().clamp_min(1e-12)
    return TeacherSet(train, c_train, mu, sigma), TeacherSet(val, c_val, mu, sigma), TeacherSet(test, c_test, mu, sigma), rows


def _wider_curriculum(best_kind: str, best_variant: str, test_target: TeacherSet) -> tuple[list[dict[str, Any]], nn.Module | None]:
    ranges = [("wide", (1.0, 3.0), TRAIN_DATA_SEED + 9000), ("medium", (1.25, 2.75), TRAIN_DATA_SEED + 9100), ("target", None, TRAIN_DATA_SEED + 9200)]
    model: nn.Module | None = None
    rows = []
    for stage, omega_range, seed in ranges:
        train = _make_data(TRAIN_NUM, seed, omega_range)
        val = _make_data(VAL_NUM, seed + 1, omega_range)
        tr, va, _, _ = _make_teacher_sets(train, val, test_target.data)
        if model is None:
            model, hist, row = _train_encoder(tr, va, test_target, best_kind, best_variant, steps=CURRICULUM_STEPS, lr=1e-3)
        else:
            # Continue training on new standardized teachers.
            opt = torch.optim.Adam(model.parameters(), lr=3e-4)
            for _ in range(CURRICULUM_STEPS):
                opt.zero_grad()
                loss, _ = _loss_parts(model(tr.data.positions[:, :P]), tr.z, best_variant)
                loss.backward()
                opt.step()
            row = _eval_encoder(model, test_target, best_kind, f"curriculum_{stage}_{best_variant}", {"dynamics_range_stage": stage})
        row["dynamics_range_stage"] = stage
        rows.append(row)
    return rows, model


def _finetune(model: nn.Module, train: TeacherSet, test: TeacherSet, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    hist = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    for step in range(1, FINETUNE_STEPS + 1):
        opt.zero_grad()
        c = train.mu + train.sigma * model(train.data.positions[:, :P])
        loss = _prefix_loss(c, train.data.positions)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == FINETUNE_STEPS:
            row = _eval_encoder(model, test, "finetuned", name, {"step": step, "prefix_finetune_loss": float(loss.item())})
            hist.append(row)
            if float(row["state_given_10_period_mse"]) < best:
                best = float(row["state_given_10_period_mse"])
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final = _eval_encoder(model, test, "finetuned", name, {"best_finetune_mse": best})
    return hist, final


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    total = torch.zeros((), dtype=DTYPE)
    for j, weight in DRIFT_HEAVY.items():
        starts = torch.arange(1, P - j)
        a, b = x[:, starts - 1], x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        total = total + weight * torch.mean((pred - truth) ** 2)
    return total


def _baselines(test: TeacherSet) -> list[dict[str, Any]]:
    rows = [
        {"stage": "baseline", "config_name": "shared_average_c", "diagnostic_c_r2": 0.0, "collapse_ratio": 0.0, "state_given_10_period_mse": _future_rollout_mse(test.c_free.mean().expand(test.c_free.shape[0]), test.data, 10.0)[0]},
        {"stage": "baseline", "config_name": "previous_drift_heavy_encoder", "diagnostic_c_r2": 0.551, "collapse_ratio": 0.148, "state_given_10_period_mse": 0.345},
        {"stage": "baseline", "config_name": "previous_amortized_encoder_best", "diagnostic_c_r2": 0.602, "collapse_ratio": 0.494, "state_given_10_period_mse": 0.270},
    ]
    for name, model in [("fixed_delay_ls", FixedDelay()), ("uniform_mixed_delay_ls", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE))]:
        r = max(2, P - (2 if name == "fixed_delay_ls" else M))
        q_in = _window_sequence(test.data.positions, M, M - 1, r)
        q_out = _window_sequence(test.data.positions, M, M, r)
        h_in, h_out = model.encode(q_in), model.encode(q_out)
        k, residual, cond = _fit_k_ridge(h_in, h_out, ridge=0.0)
        truth = _window_sequence(test.data.positions, M, P - 1, _steps(10.0) + 1)
        h0 = model.encode(truth[:, :1])[:, 0]
        pred = model.decode_x(_rollout_matrix(k, h0, truth.shape[1] - 1))[:, 1:]
        err = pred - truth[:, 1:, -1]
        rows.append({"stage": "baseline", "config_name": name, "state_given_10_period_mse": float(torch.mean(err**2).item()), "observed_ls_residual": float(residual.item()), "ls_condition_number": float(cond.mean().item()), "spectral_radius": _spectral_radius(k)})
    return rows


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 10100)
    val = _make_data(VAL_NUM, TRAIN_DATA_SEED + 10200)
    test = _make_data(TEST_NUM, TEST_DATA_SEED + 10300)
    train_t, val_t, test_t, teacher_rows = _make_teacher_sets(train, val, test)
    teacher_summary = [
        _metrics("free_c_teacher", "free_c_train", train_t.c_free, train_t.c_free, train, train_t),
        _metrics("free_c_teacher", "free_c_val", val_t.c_free, val_t.c_free, val, val_t),
        _metrics("free_c_teacher", "free_c_test", test_t.c_free, test_t.c_free, test, test_t),
    ]

    standardized_rows = []
    anti_rows = []
    hist_all = []
    configs = [("mlp", "mse"), ("cnn", "mse"), ("tcn", "mse"), ("cnn", "variance"), ("cnn", "correlation"), ("cnn", "pairwise"), ("cnn", "combined"), ("tcn", "combined")]
    best_model: nn.Module | None = None
    best_row: dict[str, Any] | None = None
    for kind, variant in configs:
        print(f"training encoder {kind}/{variant}", flush=True)
        model, hist, row = _train_encoder(train_t, val_t, test_t, kind, variant)
        hist_all.extend(hist)
        if variant == "mse":
            standardized_rows.append(row)
        else:
            anti_rows.append(row)
        if best_row is None or float(row["diagnostic_c_r2"]) > float(best_row["diagnostic_c_r2"]):
            best_row = row
            best_model = model

    curriculum_rows = []
    curriculum_model = None
    if best_row and (float(best_row["collapse_ratio"]) < 0.7 or float(best_row["diagnostic_c_r2"]) < 0.9):
        print("running wider dynamics curriculum gate follow-up", flush=True)
        curriculum_rows, curriculum_model = _wider_curriculum(str(best_row["encoder_type"]), str(best_row["loss_variant"]), test_t)

    finetune_rows = []
    finetune_hist = []
    candidate_model = curriculum_model if curriculum_model is not None else best_model
    candidate_row = max((standardized_rows + anti_rows + curriculum_rows), key=lambda r: float(r.get("diagnostic_c_r2", -1e9)))
    if candidate_model is not None and float(candidate_row["diagnostic_c_r2"]) > 0.8 and float(candidate_row["collapse_ratio"]) > 0.5:
        print("running prefix-loss fine-tune gate follow-up", flush=True)
        finetune_hist, final = _finetune(candidate_model, train_t, test_t, f"{candidate_row['config_name']}_prefix_finetune")
        finetune_rows.append(final)
    else:
        finetune_rows.append({"stage": "distill_finetune", "status": "skipped", "reason": "gate c_i R2 > 0.8 and collapse ratio > 0.5 not met"})

    baseline_rows = _baselines(test_t)
    encoder_rows = standardized_rows + anti_rows + curriculum_rows
    summary = teacher_summary + encoder_rows + finetune_rows + baseline_rows
    return {
        "summary": summary,
        "full_run_table": summary + teacher_rows + hist_all + finetune_hist,
        "free_c_teacher_results": teacher_summary,
        "standardized_distillation_results": standardized_rows,
        "anticollapse_loss_results": anti_rows,
        "encoder_comparison": encoder_rows,
        "wider_dynamics_curriculum_results": curriculum_rows or [{"stage": "wider_dynamics_curriculum", "status": "skipped", "reason": "not needed"}],
        "distill_finetune_results": finetune_rows,
        "collapse_diagnostics": [r for r in summary if "collapse_ratio" in r],
        "future_rollout_diagnostics": [r for r in summary if "state_given_10_period_mse" in r],
        "baseline_comparison": baseline_rows,
    }


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(r["config_name"])
        except Exception:
            pass
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in key or "loss" in key:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], x: str, y: str, filename: str) -> None:
    xs, ys = [], []
    for r in rows:
        try:
            xs.append(float(r[x]))
            ys.append(float(r[y]))
        except Exception:
            pass
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys)
    if "mse" in y:
        plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _html_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _write_html(outputs: dict[str, list[dict[str, Any]]], setup: dict[str, Any]) -> None:
    rows = outputs["summary"]
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Amortized c Teacher Anti-collapse</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B Amortized c Teacher Anti-collapse Diagnostic</h1>",
        "<p>Encoders predict standardized self-supervised free-c teachers. Simulator c is diagnostics only.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Summary</h2>",
        _html_table(rows, ["stage", "config_name", "encoder_type", "loss_variant", "diagnostic_c_r2", "correlation", "collapse_ratio", "test_c_mse_against_teacher", "state_given_10_period_mse", "status", "reason"]),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("free_c_teacher_results", "free_c_teacher_results.csv"),
        ("standardized_distillation_results", "standardized_distillation_results.csv"),
        ("anticollapse_loss_results", "anticollapse_loss_results.csv"),
        ("encoder_comparison", "encoder_comparison.csv"),
        ("wider_dynamics_curriculum_results", "wider_dynamics_curriculum_results.csv"),
        ("distill_finetune_results", "distill_finetune_results.csv"),
        ("collapse_diagnostics", "collapse_diagnostics.csv"),
        ("future_rollout_diagnostics", "future_rollout_diagnostics.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
    ]:
        _write_csv(OUTPUT_DIR / name, outputs[key])
    setup = {
        "diagnostic": "stage1B_amortized_c_teacher_anticollapse",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training_loss": False,
        "uses_LS_inside_learned_encoder": False,
        "uses_closed_form_c_estimator": False,
        "teacher_uses_future_points": False,
        "teacher_steps": TEACHER_STEPS,
        "distill_steps": DISTILL_STEPS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_bar(outputs["free_c_teacher_results"], "teacher_c_std", "teacher_c_distribution.png")
    _plot_scatter(outputs["encoder_comparison"], "teacher_c_mean", "pred_c_mean", "predicted_vs_teacher_c.png")
    _plot_scatter(outputs["encoder_comparison"], "teacher_c_mean", "pred_c_mean", "predicted_vs_simulator_c.png")
    _plot_bar(outputs["encoder_comparison"], "collapse_ratio", "collapse_ratio_by_loss.png")
    _plot_bar(outputs["encoder_comparison"], "diagnostic_c_r2", "c_r2_by_encoder.png")
    _plot_bar(outputs["encoder_comparison"], "state_given_10_period_mse", "future_rollout_mse_by_encoder_loss.png")
    _plot_bar(outputs["encoder_comparison"], "pairwise_difference_error", "pairwise_difference_scatter.png")
    _plot_bar(outputs["wider_dynamics_curriculum_results"], "diagnostic_c_r2", "wider_dynamics_curriculum_curves.png")
    _plot_bar(outputs["distill_finetune_results"], "state_given_10_period_mse", "distill_finetune_curves.png")
    _plot_bar(outputs["baseline_comparison"], "state_given_10_period_mse", "baseline_comparison.png")
    _write_html(outputs, setup)


def main() -> None:
    torch.manual_seed(SEED)
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
