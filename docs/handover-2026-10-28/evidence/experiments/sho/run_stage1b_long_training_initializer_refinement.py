"""Stage 1B long-training learned-initializer + efficient-refinement diagnostic.

The online/inference path is deliberately cheap:

    prefix -> TCN -> c_init -> K prefix-only refinement steps -> rollout.

Training may use prefix-only free-c teachers for the initializer and prefix
recurrence consistency for refinement. Future data is evaluation-only.
"""

from __future__ import annotations

import copy
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
import torch
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    DRIFT_HEAVY,
    J_ALL,
    P,
    TEST_NUM,
    TRAIN_NUM,
    VAL_NUM,
    TCNBlock,
    TeacherSet,
    _baselines,
    _corr,
    _future_rollout_mse,
    _make_data,
    _make_teacher_sets,
    _r2,
    _sim_c,
    _threshold_flags,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
    _write_json,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/long_training_initializer_refinement")

BATCH_SIZE = int(os.environ.get("LTIR_BATCH_SIZE", "16"))
INIT_MAX_STEPS = int(os.environ.get("LTIR_INIT_MAX_STEPS", "100000"))
REFINE_MAX_STEPS = int(os.environ.get("LTIR_REFINE_MAX_STEPS", "100000"))
EVAL_INTERVAL = int(os.environ.get("LTIR_EVAL_INTERVAL", "500"))
PATIENCE = int(os.environ.get("LTIR_PATIENCE", "15000"))
INIT_LAMBDA_ABS = float(os.environ.get("LTIR_INIT_LAMBDA_ABS", "1"))
FIXED_ETA = float(os.environ.get("LTIR_FIXED_ETA", "0.5"))
FIXED_DAMPING = float(os.environ.get("LTIR_FIXED_DAMPING", "1e-4"))
EXPLICIT_STEPS = [0, 1, 2, 5, 10, 20, 50]
INFERENCE_STEPS = [0, 1, 2, 5, 10]


class InitializerRefinementTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.shared = nn.Sequential(nn.Linear(32, 64), nn.SiLU())
        self.c_head = nn.Linear(64, 1)
        self.eta_head = nn.Linear(64, 1)
        self.damping_head = nn.Linear(64, 1)
        self.delta_head = nn.Sequential(nn.Linear(65, 64), nn.SiLU(), nn.Linear(64, 1))

    def features(self, prefix: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.inp(prefix.unsqueeze(1)))
        y = self.blocks(y).mean(dim=-1)
        return self.shared(y)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(prefix)
        z_pred = self.c_head(feat).squeeze(-1)
        eta = torch.sigmoid(self.eta_head(feat).squeeze(-1))
        damping = torch.nn.functional.softplus(self.damping_head(feat).squeeze(-1)) + 1e-8
        return {"features": feat, "z_pred": z_pred, "eta": eta, "damping": damping}

    def correction(self, features: torch.Tensor, c_init: torch.Tensor) -> torch.Tensor:
        raw = self.delta_head(torch.cat([features, c_init[:, None]], dim=-1)).squeeze(-1)
        return 0.1 * torch.tanh(raw)


def _param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _predict_c(outputs: dict[str, torch.Tensor], dataset: TeacherSet) -> torch.Tensor:
    return dataset.mu + dataset.sigma * outputs["z_pred"]


def _prefix_loss_and_derivatives(c: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-trajectory prefix loss, gradient, and Hessian diagonal wrt scalar c.

    This avoids autograd Hessian loops. For recurrence x' = c b - a:
        d x'/dc = b + c db/dc - da/dc
        d2 x'/dc2 = 2 db/dc + c d2b/dc2 - d2a/dc2.
    """

    x = positions[:, :P]
    losses = torch.zeros_like(c)
    grads = torch.zeros_like(c)
    hess = torch.zeros_like(c)
    for j in J_ALL:
        starts = torch.arange(1, P - j, device=x.device)
        a = x[:, starts - 1]
        b = x[:, starts]
        da = torch.zeros_like(a)
        db = torch.zeros_like(b)
        dda = torch.zeros_like(a)
        ddb = torch.zeros_like(b)
        pred_terms = []
        d_terms = []
        dd_terms = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            dnxt = b + c[:, None] * db - da
            ddnxt = 2.0 * db + c[:, None] * ddb - dda
            pred_terms.append(nxt)
            d_terms.append(dnxt)
            dd_terms.append(ddnxt)
            a, b = b, nxt
            da, db = db, dnxt
            dda, ddb = ddb, ddnxt
        pred = torch.stack(pred_terms, dim=-1)
        deriv = torch.stack(d_terms, dim=-1)
        deriv2 = torch.stack(dd_terms, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        err = pred - truth
        weight = DRIFT_HEAVY[j]
        losses = losses + weight * torch.mean(err**2, dim=(1, 2))
        grads = grads + weight * 2.0 * torch.mean(err * deriv, dim=(1, 2))
        hess = hess + weight * 2.0 * torch.mean(deriv**2 + err * deriv2, dim=(1, 2))
    return losses, grads, hess


def _prefix_loss_vector(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return _prefix_loss_and_derivatives(c, positions)[0]


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.mean(_prefix_loss_vector(c, positions))


def _newton_refine_efficient(
    c_init: torch.Tensor,
    positions: torch.Tensor,
    steps: int,
    *,
    eta: float | torch.Tensor = FIXED_ETA,
    damping: float | torch.Tensor = FIXED_DAMPING,
) -> torch.Tensor:
    c = c_init
    if not torch.is_tensor(eta):
        eta = torch.full_like(c, float(eta))
    if not torch.is_tensor(damping):
        damping = torch.full_like(c, float(damping))
    for _ in range(steps):
        _, grad, hess = _prefix_loss_and_derivatives(c, positions)
        denom = hess + damping
        denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
        c = (c - eta * grad / denom).clamp(-1.999999999, 1.999999999)
    return c


def _newton_refine_constrained_line_search(c_init: torch.Tensor, positions: torch.Tensor, steps: int) -> torch.Tensor:
    """Stable diagnostic-only Newton in raw atanh(c/2) coordinates.

    This reproduces the successful explicit prefix-only optimizer more closely
    than direct c-space Newton, while keeping the learned layer itself separate.
    """

    raw = torch.atanh((c_init / 2.0).clamp(-0.999999999, 0.999999999))
    best_raw = raw.detach().clone()
    best_loss = _prefix_loss_vector(2.0 * torch.tanh(raw), positions).detach()
    for _ in range(steps):
        c = 2.0 * torch.tanh(raw)
        losses, grad_c, hess_c = _prefix_loss_and_derivatives(c, positions)
        dc = 2.0 * (1.0 - torch.tanh(raw) ** 2)
        d2c = -4.0 * torch.tanh(raw) * (1.0 - torch.tanh(raw) ** 2)
        grad_raw = grad_c * dc
        hess_raw = hess_c * dc**2 + grad_c * d2c
        denom = hess_raw + FIXED_DAMPING
        denom = torch.where(torch.abs(denom) < 1e-12, torch.full_like(denom, 1e-12), denom)
        step_raw = FIXED_ETA * grad_raw / denom
        accepted = torch.zeros_like(raw, dtype=torch.bool)
        proposal = raw.detach().clone()
        eta_scale = torch.ones_like(raw)
        old_loss = losses.detach()
        for _attempt in range(10):
            cand = raw - eta_scale * step_raw
            cand_loss = _prefix_loss_vector(2.0 * torch.tanh(cand), positions).detach()
            mask = (~accepted) & (cand_loss <= old_loss)
            proposal[mask] = cand.detach()[mask]
            accepted = accepted | mask
            eta_scale = eta_scale * 0.5
            if bool(torch.all(accepted)):
                break
        raw = proposal.detach()
        current_loss = _prefix_loss_vector(2.0 * torch.tanh(raw), positions).detach()
        mask = current_loss < best_loss
        best_loss[mask] = current_loss[mask]
        best_raw[mask] = raw[mask]
    return 2.0 * torch.tanh(best_raw)


def _adam_refine_efficient(c_init: torch.Tensor, positions: torch.Tensor, steps: int, lr: float = 1e-2) -> torch.Tensor:
    c = c_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    best_c = c.detach().clone()
    best = torch.full_like(c, float("inf"))
    for _ in range(steps):
        loss_vec = _prefix_loss_vector(c, positions)
        loss = torch.mean(loss_vec)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            c.clamp_(-1.999999999, 1.999999999)
            current = _prefix_loss_vector(c, positions)
            mask = current < best
            best[mask] = current[mask]
            best_c[mask] = c.detach()[mask]
    return best_c


def _future_metrics(c: torch.Tensor, dataset: TeacherSet) -> dict[str, Any]:
    mse10, maxerr, finalerr = _future_rollout_mse(c, dataset.data, 10.0)
    row = {
        "future_1_period_mse": _future_rollout_mse(c, dataset.data, 1.0)[0],
        "future_2_period_mse": _future_rollout_mse(c, dataset.data, 2.0)[0],
        "future_5_period_mse": _future_rollout_mse(c, dataset.data, 5.0)[0],
        "future_10_period_mse": mse10,
        "state_given_10_period_mse": mse10,
        "max_trajectory_error": maxerr,
        "final_time_error": finalerr,
        "final_time_absolute_error": finalerr,
    }
    row.update(_threshold_flags(mse10))
    return row


def _c_metrics(stage: str, name: str, c: torch.Tensor, dataset: TeacherSet, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    teacher = dataset.c_free
    err = c - teacher
    row: dict[str, Any] = {
        "stage": stage,
        "config_name": name,
        "c_rmse": float(torch.sqrt(torch.mean(err**2)).item()),
        "c_mae": float(torch.mean(torch.abs(err)).item()),
        "c_max_abs_error": float(torch.max(torch.abs(err)).item()),
        "c_bias": float(torch.mean(err).item()),
        "c_r2_against_teacher": _r2(c[:, None], teacher[:, None]),
        "diagnostic_c_r2": _r2(c[:, None], _sim_c(dataset.data)[:, None]),
        "correlation": _corr(c, teacher),
        "collapse_ratio": float((torch.std(c) / torch.std(teacher).clamp_min(1e-12)).item()),
        "predicted_c_std": float(torch.std(c).item()),
        "teacher_c_std": float(torch.std(teacher).item()),
        "prefix_loss": float(_prefix_loss(c, dataset.data.positions).item()),
    }
    row.update(_future_metrics(c, dataset))
    if extra:
        row.update(extra)
    return row


def _phase0_explicit_refinement(train: TeacherSet, test: TeacherSet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    for init_name, init in [("average_c", avg), ("teacher_c", test.c_free)]:
        for method in ["newton", "newton_constrained_linesearch", "adam"]:
            for steps in EXPLICIT_STEPS:
                start = time.perf_counter()
                if steps == 0:
                    c = init
                elif method == "newton":
                    c = _newton_refine_efficient(init, test.data.positions, steps)
                elif method == "newton_constrained_linesearch":
                    c = _newton_refine_constrained_line_search(init, test.data.positions, steps)
                else:
                    c = _adam_refine_efficient(init, test.data.positions, steps)
                runtime = time.perf_counter() - start
                row = _c_metrics(
                    "phase0_explicit_refinement",
                    f"{init_name}_{method}_steps_{steps}",
                    c.detach(),
                    test,
                    {
                        "initialization": init_name,
                        "refinement_method": method,
                        "refinement_steps": steps,
                        "runtime_seconds": runtime,
                        "runtime_per_trajectory_seconds": runtime / max(1, test.data.positions.shape[0]),
                    },
                )
                rows.append(row)
    best = min(rows, key=lambda r: float(r["future_10_period_mse"]))
    if float(best["future_10_period_mse"]) > 1e-12:
        print("WARNING: Phase 0 did not reach the numerical floor; continuing for diagnostic output.", flush=True)
    return rows


def _eval_initializer(model: InitializerRefinementTCN, dataset: TeacherSet, split: str, config_name: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model(dataset.data.positions[:, :P]), dataset)
    row = _c_metrics("phase1_initializer", config_name, c, dataset, {"split": split})
    if extra:
        row.update(extra)
    return row


def _train_initializer(train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[InitializerRefinementTCN, list[dict[str, Any]], list[dict[str, Any]]]:
    model = InitializerRefinementTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("LTIR_INIT_LR", "3e-4")))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 71000)
    n = train.data.positions.shape[0]
    config_name = f"tcn_standardized_plus_raw_lambda_{INIT_LAMBDA_ABS:g}"
    for step in range(1, INIT_MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        z_true = train.z[idx]
        c_true = train.c_free[idx]
        outputs = model(pos[:, :P])
        c_pred = train.mu + train.sigma * outputs["z_pred"]
        loss = torch.mean((outputs["z_pred"] - z_true) ** 2) + INIT_LAMBDA_ABS * torch.mean((c_pred - c_true) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == INIT_MAX_STEPS:
            val_row = _eval_initializer(model, val, "val", config_name, {"step": step, "train_loss": float(loss.item())})
            train_row = _eval_initializer(model, train, "train", config_name, {"step": step, "train_loss": float(loss.item())})
            history.extend([train_row, val_row])
            score = float(val_row["c_rmse"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    finals = [
        _eval_initializer(model, train, "train", config_name, {"best_validation_c_rmse": best, "parameter_count": _param_count(model)}),
        _eval_initializer(model, val, "val", config_name, {"best_validation_c_rmse": best, "parameter_count": _param_count(model)}),
        _eval_initializer(model, test, "test", config_name, {"best_validation_c_rmse": best, "parameter_count": _param_count(model)}),
    ]
    return model, history, finals


def _refined_c_from_model(model: InitializerRefinementTCN, dataset: TeacherSet, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(dataset.data.positions[:, :P])
    c_init = _predict_c(outputs, dataset)
    method = str(config["method"])
    if method == "none":
        return c_init, c_init
    if method == "fixed_newton":
        return c_init, _newton_refine_efficient(c_init, dataset.data.positions, int(config["k_steps"]), eta=FIXED_ETA, damping=FIXED_DAMPING)
    if method == "learned_newton":
        return c_init, _newton_refine_efficient(c_init, dataset.data.positions, int(config["k_steps"]), eta=outputs["eta"], damping=outputs["damping"])
    if method == "learned_correction":
        c_corr = (c_init + model.correction(outputs["features"], c_init)).clamp(-1.999999999, 1.999999999)
        if int(config["k_steps"]) > 0:
            c_corr = _newton_refine_efficient(c_corr, dataset.data.positions, int(config["k_steps"]), eta=FIXED_ETA, damping=FIXED_DAMPING)
        return c_init, c_corr
    raise ValueError(method)


def _eval_refinement(model: InitializerRefinementTCN, dataset: TeacherSet, config: dict[str, Any], stage: str, split: str) -> dict[str, Any]:
    start = time.perf_counter()
    with torch.enable_grad():
        c_init, c_ref = _refined_c_from_model(model, dataset, config)
    runtime = time.perf_counter() - start
    row = _c_metrics(
        stage,
        str(config["config_name"]),
        c_ref.detach(),
        dataset,
        {
            "split": split,
            "method": config["method"],
            "refinement_steps": config["k_steps"],
            "beta_init": config.get("beta_init", ""),
            "c_init_rmse": float(torch.sqrt(torch.mean((c_init.detach() - dataset.c_free) ** 2)).item()),
            "runtime_seconds": runtime,
            "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0]),
            "parameter_count": _param_count(model),
        },
    )
    return row


def _refinement_configs() -> list[dict[str, Any]]:
    requested = os.environ.get("LTIR_REFINE_CONFIGS")
    names = [x.strip() for x in requested.split(",") if x.strip()] if requested else [
        "fixed_K1_beta0.01",
        "fixed_K2_beta0.01",
        "fixed_K5_beta0.01",
        "learned_K1_beta0.01",
        "learned_K2_beta0.01",
        "learned_K5_beta0.01",
        "correction_K0_beta0.01",
        "correction_K1_beta0.01",
    ]
    out: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        parts = name.split("_")
        if parts[0] == "fixed":
            method = "fixed_newton"
        elif parts[0] == "learned":
            method = "learned_newton"
        elif parts[0] == "correction":
            method = "learned_correction"
        else:
            raise ValueError(name)
        k = int(parts[1].replace("K", ""))
        beta = float(parts[2].replace("beta", ""))
        out.append({"config_index": idx, "config_name": name, "method": method, "k_steps": k, "beta_init": beta})
    return out


def _train_refinement(
    init_model: InitializerRefinementTCN,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    config: dict[str, Any],
) -> tuple[InitializerRefinementTCN, list[dict[str, Any]], dict[str, Any]]:
    model = copy.deepcopy(init_model)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("LTIR_REFINE_LR", "1e-4")))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 72000 + int(config["config_index"]))
    n = train.data.positions.shape[0]
    for step in range(1, REFINE_MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos_set = TeacherSet(
            data=type(train.data)(positions=train.data.positions[idx], omegas=train.data.omegas[idx]),
            c_free=train.c_free[idx],
            mu=train.mu,
            sigma=train.sigma,
        )
        c_init, c_ref = _refined_c_from_model(model, pos_set, config)
        loss_ref = _prefix_loss(c_ref, pos_set.data.positions)
        loss_init = _prefix_loss(c_init, pos_set.data.positions)
        loss = loss_ref + float(config["beta_init"]) * loss_init
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == REFINE_MAX_STEPS:
            val_row = _eval_refinement(model, val, config, "phase2_training_history", "val")
            val_row.update({"step": step, "train_total_loss": float(loss.item()), "train_refined_prefix_loss": float(loss_ref.item()), "train_init_prefix_loss": float(loss_init.item())})
            history.append(val_row)
            score = float(val_row["prefix_loss"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final = _eval_refinement(model, test, config, "phase2_refinement", "test")
    final["best_validation_prefix_loss"] = best
    return model, history, final


def _inference_cost_rows(model: InitializerRefinementTCN, train: TeacherSet, test: TeacherSet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [{"config_name": "pure_encoder_K0", "method": "none", "k_steps": 0}]
    configs.extend({"config_name": f"encoder_fixed_newton_K{k}", "method": "fixed_newton", "k_steps": k} for k in [1, 2, 5, 10])
    configs.extend({"config_name": f"encoder_learned_newton_K{k}", "method": "learned_newton", "k_steps": k} for k in [1, 2, 5])
    configs.extend({"config_name": f"encoder_correction_then_newton_K{k}", "method": "learned_correction", "k_steps": k} for k in [0, 1, 2, 5])
    for cfg in configs:
        row = _eval_refinement(model, test, cfg, "phase3_inference_cost", "test")
        rows.append(row)
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    for k in INFERENCE_STEPS:
        start = time.perf_counter()
        c = avg if k == 0 else _newton_refine_efficient(avg, test.data.positions, k)
        runtime = time.perf_counter() - start
        rows.append(
            _c_metrics(
                "phase3_inference_cost",
                f"average_c_fixed_newton_K{k}",
                c.detach(),
                test,
                {
                    "method": "average_c_fixed_newton",
                    "refinement_steps": k,
                    "runtime_seconds": runtime,
                    "runtime_per_trajectory_seconds": runtime / max(1, test.data.positions.shape[0]),
                    "parameter_count": 0,
                },
            )
        )
    return rows


def _baseline_rows(test: TeacherSet) -> list[dict[str, Any]]:
    rows = [_c_metrics("baseline", "free_c_teacher", test.c_free, test, {"method": "free_prefix_c_optimization", "refinement_steps": 0})]
    rows.extend(dict(r, stage="baseline") for r in _baselines(test))
    return rows


def _plot_by_steps(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    for method in ["fixed_newton", "learned_newton", "learned_correction", "average_c_fixed_newton"]:
        vals = [r for r in rows if r.get("method") == method]
        if not vals:
            continue
        by_k: dict[int, float] = {}
        for r in vals:
            k = int(r.get("refinement_steps", 0))
            by_k[k] = min(by_k.get(k, float("inf")), float(r.get(key, float("inf"))))
        xs = sorted(by_k)
        plt.plot(xs, [by_k[x] for x in xs], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("refinement steps K")
    plt.ylabel(key)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(str(r.get("config_name", "")))
        except Exception:
            pass
    if not vals:
        return
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in key or "rmse" in key:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=55, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_runtime(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r.get("stage") == "phase3_inference_cost"]
    plt.figure(figsize=(7, 5))
    for r in vals:
        try:
            plt.scatter(float(r["runtime_per_trajectory_seconds"]), float(r["future_10_period_mse"]), s=35)
        except Exception:
            pass
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("runtime per trajectory (s)")
    plt.ylabel("10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_vs_accuracy.png", dpi=170)
    plt.close()


def _interpret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    inference = [r for r in rows if r.get("stage") == "phase3_inference_cost"]
    phase1 = [r for r in rows if r.get("stage") == "phase1_initializer" and r.get("split") == "test"]
    best_inf = min(inference, key=lambda r: float(r.get("future_10_period_mse", float("inf"))), default={})
    learned = [r for r in inference if r.get("method") == "learned_newton"]
    fixed = [r for r in inference if r.get("method") == "fixed_newton"]
    avg = [r for r in inference if r.get("method") == "average_c_fixed_newton"]
    return {
        "can_long_training_produce_good_c_init": bool(phase1 and float(phase1[0]["c_rmse"]) < 1e-3),
        "pure_encoder_test_c_rmse": phase1[0].get("c_rmse", "") if phase1 else "",
        "pure_encoder_10_period_mse": phase1[0].get("future_10_period_mse", "") if phase1 else "",
        "learned_refinement_beats_fixed_newton": min((float(r["future_10_period_mse"]) for r in learned), default=float("inf")) < min((float(r["future_10_period_mse"]) for r in fixed), default=float("inf")),
        "encoder_init_reduces_steps_vs_average_c": min((float(r["future_10_period_mse"]) for r in fixed + learned), default=float("inf")) < min((float(r["future_10_period_mse"]) for r in avg), default=float("inf")),
        "best_inference_config": best_inf.get("config_name", ""),
        "best_inference_10_period_mse": best_inf.get("future_10_period_mse", ""),
        "useful_success_K_le_5": any(float(r.get("future_10_period_mse", float("inf"))) < 1e-6 and int(r.get("refinement_steps", 99)) <= 5 for r in inference),
        "strong_success_K_le_5": any(float(r.get("future_10_period_mse", float("inf"))) < 1e-9 and int(r.get("refinement_steps", 99)) <= 5 for r in inference),
        "framing": "encoder initialization + prefix-refinement layer" if best_inf else "undetermined",
    }


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "split", "config_name", "method", "refinement_steps", "c_rmse", "c_init_rmse", "future_1_period_mse", "future_2_period_mse", "future_5_period_mse", "future_10_period_mse", "runtime_per_trajectory_seconds"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Long Training Initializer Refinement</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Long-training Initializer + Efficient Prefix-refinement</h1>",
        "<p>Training uses clean position-only prefixes. Future rollout is evaluation-only. The efficient refinement layer uses analytic scalar prefix-loss derivatives, not autograd Hessian loops.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 81000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 81100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 81200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    print("phase 0 explicit refinement", flush=True)
    phase0 = _phase0_explicit_refinement(train, test)
    print("phase 1 initializer training", flush=True)
    model, init_history, init_results = _train_initializer(train, val, test)
    print("phase 2 refinement training", flush=True)
    refinement_results: list[dict[str, Any]] = []
    refinement_history: list[dict[str, Any]] = []
    trained_models: list[InitializerRefinementTCN] = [model]
    for cfg in _refinement_configs():
        print(f"training refinement {cfg['config_name']}", flush=True)
        trained, hist, row = _train_refinement(model, train, val, test, cfg)
        trained_models.append(trained)
        refinement_history.extend(hist)
        refinement_results.append(row)
    best_model = min(
        [(model, _eval_refinement(model, val, {"config_name": "initializer", "method": "none", "k_steps": 0}, "model_selection", "val"))]
        + [(m, _eval_refinement(m, val, {"config_name": f"candidate_{i}", "method": "fixed_newton", "k_steps": 2}, "model_selection", "val")) for i, m in enumerate(trained_models[1:])],
        key=lambda pair: float(pair[1]["future_10_period_mse"]),
    )[0]
    print("phase 3 inference-cost evaluation", flush=True)
    inference = _inference_cost_rows(best_model, train, test)
    baselines = _baseline_rows(test)
    summary = baselines + phase0 + init_results + refinement_results + inference
    return {
        "summary": summary,
        "full_run_table": summary + teacher_history + init_history + refinement_history,
        "training_history": init_history + refinement_history,
        "refinement_results": refinement_results,
        "inference_cost_results": inference,
        "phase0_explicit_refinement": phase0,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("training_history", "training_history.csv"),
        ("refinement_results", "refinement_results.csv"),
        ("inference_cost_results", "inference_cost_results.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs["summary"])
    setup = {
        "diagnostic": "stage1B_long_training_initializer_refinement",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "phase1_max_steps": INIT_MAX_STEPS,
        "phase2_max_steps": REFINE_MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "initializer_loss": "MSE(z_pred,z_teacher)+lambda_abs*MSE(c_pred,c_teacher)",
        "lambda_abs": INIT_LAMBDA_ABS,
        "refinement_loss": "L_prefix(c_refined)+beta_init*L_prefix(c_init)",
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_simulator_c_labels_in_training": False,
        "uses_future_data_during_refinement": False,
        "uses_LS_inside_encoder": False,
        "uses_closed_form_c_estimator": False,
        "efficient_refinement": "analytic prefix-loss gradient and Hessian for scalar c",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_by_steps(outputs["inference_cost_results"], "c_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_by_steps(outputs["inference_cost_results"], "future_10_period_mse", "rollout_mse_vs_refinement_steps.png")
    _plot_bar([r for r in outputs["inference_cost_results"] if r.get("method") in {"fixed_newton", "learned_newton"}], "future_10_period_mse", "learned_vs_fixed_refinement.png")
    _plot_bar(outputs["inference_cost_results"], "future_10_period_mse", "encoder_init_vs_average_init.png")
    _plot_runtime(outputs["inference_cost_results"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    interp = _interpret(outputs["summary"])
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interp}, indent=2), flush=True)


if __name__ == "__main__":
    main()
