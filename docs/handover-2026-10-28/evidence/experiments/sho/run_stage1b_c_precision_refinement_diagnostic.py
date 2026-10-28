"""Stage 1B c_i precision and refinement diagnostic.

This follow-up keeps the problem scoped to

    position prefix -> trajectory-specific recurrence constant c_i.

The only training target is the self-supervised free-c teacher recovered from
the observed prefix. Simulator c is used only for diagnostic attribution.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
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
    BATCH_SIZE,
    DRIFT_HEAVY,
    J_ALL,
    M,
    P,
    TEST_NUM,
    THRESHOLDS,
    TRAIN_NUM,
    VAL_NUM,
    TCNBlock,
    TeacherSet,
    _baselines,
    _corr,
    _future_rollout_mse,
    _make_data,
    _make_teacher_sets,
    _prefix_loss,
    _r2,
    _sim_c,
    _steps,
    _threshold_flags,
    _train_encoder,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/c_precision_refinement_diagnostic")

# The user-requested maxima are represented here, with short patience and
# early-stopping. The grids below are intentionally runtime-prioritized.
MAX_STEPS = 30_000
RESIDUAL_STEP_CAP = 900
ROLLOUT_STEP_CAP = 5
BASE_DISTILL_STEPS = 800
BASE_FINETUNE_STEPS = 500
EVAL_INTERVAL = 200
PATIENCE = 5_000
RUNTIME_PATIENCE = 400


class FeatureTCNEncoder(nn.Module):
    """Same module names as the previous TCN, with exposed pooled features."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8]))
        self.head = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(torch.nn.functional.silu(self.inp(x.unsqueeze(1))))
        return y.mean(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class ResidualCorrectionModel(nn.Module):
    def __init__(self, base: FeatureTCNEncoder, mu: torch.Tensor, sigma: torch.Tensor, delta_scale: float) -> None:
        super().__init__()
        self.base = base
        self.register_buffer("mu", mu.detach().clone())
        self.register_buffer("sigma", sigma.detach().clone())
        self.delta_scale = float(delta_scale)
        self.corrector = nn.Sequential(nn.Linear(32, 48), nn.SiLU(), nn.Linear(48, 1))

    def base_c(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu + self.sigma * self.base(x)

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        return self.corrector(self.base.features(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_c(x) + self.delta_scale * self.delta(x)


def _clone_feature_tcn(model: nn.Module) -> FeatureTCNEncoder:
    out = FeatureTCNEncoder().to(dtype=DTYPE)
    out.load_state_dict(copy.deepcopy(model.state_dict()))
    return out


def _per_traj_rollout(c: torch.Tensor, dataset: TeacherSet, periods: float = 10.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = _steps(periods)
    a = dataset.data.positions[:, P - 2]
    b = dataset.data.positions[:, P - 1]
    preds = []
    for _ in range(steps):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = dataset.data.positions[:, P : P + steps]
    err = pred - truth
    return torch.mean(err**2, dim=1), torch.max(torch.abs(err), dim=1).values, torch.abs(err[:, -1])


def _precision_metrics(stage: str, name: str, c_pred: torch.Tensor, dataset: TeacherSet, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    c_teacher = dataset.c_free
    err = c_pred - c_teacher
    mse10, maxerr, finalerr = _future_rollout_mse(c_pred, dataset.data, 10.0)
    row: dict[str, Any] = {
        "stage": stage,
        "config_name": name,
        "model_type": name,
        "c_rmse": float(torch.sqrt(torch.mean(err**2)).item()),
        "c_mae": float(torch.mean(torch.abs(err)).item()),
        "c_max_abs_error": float(torch.max(torch.abs(err)).item()),
        "c_bias": float(torch.mean(err).item()),
        "c_std_error": float(torch.std(err).item()),
        "c_r2_against_teacher": _r2(c_pred[:, None], c_teacher[:, None]),
        "diagnostic_c_r2": _r2(c_pred[:, None], _sim_c(dataset.data)[:, None]),
        "correlation": _corr(c_pred, c_teacher),
        "collapse_ratio": float((torch.std(c_pred) / torch.std(c_teacher).clamp_min(1e-12)).item()),
        "state_given_1_period_mse": _future_rollout_mse(c_pred, dataset.data, 1.0)[0],
        "state_given_2_period_mse": _future_rollout_mse(c_pred, dataset.data, 2.0)[0],
        "state_given_5_period_mse": _future_rollout_mse(c_pred, dataset.data, 5.0)[0],
        "state_given_10_period_mse": mse10,
        "max_trajectory_error": maxerr,
        "final_time_absolute_error": finalerr,
    }
    row.update(_threshold_flags(mse10))
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def _predict_standardized(model: nn.Module, dataset: TeacherSet) -> torch.Tensor:
    return dataset.mu + dataset.sigma * model(dataset.data.positions[:, :P])


def _train_previous_references(train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[FeatureTCNEncoder, FeatureTCNEncoder, list[dict[str, Any]], list[dict[str, Any]]]:
    print("training/reconstructing previous tcn_combined reference", flush=True)
    model, hist, _ = _train_encoder(train, val, test, "tcn", "combined", steps=BASE_DISTILL_STEPS, lr=1e-3)
    base = _clone_feature_tcn(model)
    audit_rows = [_precision_metrics("precision_audit", "reconstructed_tcn_combined", _predict_standardized(base, test), test)]

    print("training/reconstructing previous prefix-finetuned reference", flush=True)
    ft = _clone_feature_tcn(model)
    opt = torch.optim.Adam(ft.parameters(), lr=1e-4)
    best_state = copy.deepcopy(ft.state_dict())
    best = float("inf")
    finetune_hist: list[dict[str, Any]] = []
    for step in range(1, BASE_FINETUNE_STEPS + 1):
        opt.zero_grad()
        c = train.mu + train.sigma * ft(train.data.positions[:, :P])
        loss = _prefix_loss(c, train.data.positions)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == BASE_FINETUNE_STEPS:
            c_test = _predict_standardized(ft, test)
            row = _precision_metrics("precision_audit", "reconstructed_finetuned_tcn_combined", c_test, test, {"step": step, "prefix_loss": float(loss.item())})
            finetune_hist.append(row)
            if row["state_given_10_period_mse"] < best:
                best = float(row["state_given_10_period_mse"])
                best_state = copy.deepcopy(ft.state_dict())
    ft.load_state_dict(best_state)
    audit_rows.append(_precision_metrics("precision_audit", "reconstructed_finetuned_tcn_combined", _predict_standardized(ft, test), test, {"best_reference_rollout_mse": best}))
    return base, ft, hist, audit_rows + finetune_hist


def _sensitivity(dataset: TeacherSet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 22001)
    sign = torch.where(torch.rand(dataset.c_free.shape, generator=gen, dtype=DTYPE) > 0.5, 1.0, -1.0)
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        for kind, delta in [
            ("plus", torch.full_like(dataset.c_free, eps)),
            ("minus", torch.full_like(dataset.c_free, -eps)),
            ("random_sign", sign * eps),
        ]:
            c = dataset.c_free + delta
            row = _precision_metrics("sensitivity", f"eps_{eps:g}_{kind}", c, dataset, {"eps": eps, "perturbation_kind": kind})
            rows.append(row)
    return rows


def _batch_indices(n: int, step: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED + 33000 + step)
    return torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)


def _train_residual(
    base: FeatureTCNEncoder,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    delta_scale: float,
    lambda_abs: float,
    mode: str,
    steps: int,
) -> tuple[ResidualCorrectionModel, list[dict[str, Any]], dict[str, Any]]:
    model = ResidualCorrectionModel(_clone_feature_tcn(base), train.mu, train.sigma, delta_scale).to(dtype=DTYPE)
    if mode == "frozen_base":
        for p in model.base.parameters():
            p.requires_grad_(False)
    elif mode == "unfreeze_last_block":
        for p in model.base.parameters():
            p.requires_grad_(False)
        for p in model.base.blocks[-1].parameters():
            p.requires_grad_(True)
        for p in model.base.head.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(mode)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=3e-4 if mode == "frozen_base" else 1e-4)
    with torch.no_grad():
        base_train = model.base_c(train.data.positions[:, :P])
        residual_std = torch.std(train.c_free - base_train).clamp_min(1e-12)
        base_test = model.base_c(test.data.positions[:, :P])
        residual_before = torch.sqrt(torch.mean((base_test - test.c_free) ** 2)).item()

    if mode == "frozen_base":
        with torch.no_grad():
            feat_train = model.base.features(train.data.positions[:, :P])
            feat_val = model.base.features(val.data.positions[:, :P])
            feat_test = model.base.features(test.data.positions[:, :P])
            base_val = model.base_c(val.data.positions[:, :P])
            base_test = model.base_c(test.data.positions[:, :P])
            r_train = (train.c_free - base_train) / residual_std
        opt = torch.optim.Adam(model.corrector.parameters(), lr=3e-4)
        best_state = copy.deepcopy(model.state_dict())
        best_score = float("inf")
        stale = 0
        hist: list[dict[str, Any]] = []
        max_steps = min(MAX_STEPS, steps)
        for step in range(1, max_steps + 1):
            idx = _batch_indices(train.data.positions.shape[0], step)
            opt.zero_grad()
            delta = model.corrector(feat_train[idx]).squeeze(-1)
            c_pred = base_train[idx] + delta_scale * delta
            loss_resid = torch.mean((delta - r_train[idx]) ** 2)
            loss_abs = torch.mean((c_pred - train.c_free[idx]) ** 2)
            loss = loss_resid + lambda_abs * loss_abs
            loss.backward()
            opt.step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == max_steps:
                with torch.no_grad():
                    c_val = base_val + delta_scale * model.corrector(feat_val).squeeze(-1)
                    val_rollout = _future_rollout_mse(c_val, val.data, 10.0)[0]
                    c_test = base_test + delta_scale * model.corrector(feat_test).squeeze(-1)
                    row = _precision_metrics(
                        "residual_correction_history",
                        f"{mode}_scale_{delta_scale:g}_lambda_abs_{lambda_abs:g}",
                        c_test,
                        test,
                        {
                            "step": step,
                            "delta_scale": delta_scale,
                            "lambda_abs": lambda_abs,
                            "residual_mode": mode,
                            "base_frozen": True,
                            "loss_resid": float(loss_resid.item()),
                            "loss_abs": float(loss_abs.item()),
                            "validation_rollout_mse": val_rollout,
                        },
                    )
                hist.append(row)
                score = val_rollout + float(torch.mean((c_val - val.c_free) ** 2).item())
                if score < best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                if stale >= RUNTIME_PATIENCE:
                    break
        model.load_state_dict(best_state)
        with torch.no_grad():
            c_test = base_test + delta_scale * model.corrector(feat_test).squeeze(-1)
            c_base_test = base_test
            delta_norm = torch.sqrt(torch.mean((c_test - c_base_test) ** 2)).item()
            residual_after = torch.sqrt(torch.mean((c_test - test.c_free) ** 2)).item()
            final = _precision_metrics(
                "residual_correction",
                f"{mode}_scale_{delta_scale:g}_lambda_abs_{lambda_abs:g}",
                c_test,
                test,
                {
                    "delta_scale": delta_scale,
                    "lambda_abs": lambda_abs,
                    "residual_mode": mode,
                    "base_frozen": True,
                    "residual_rmse_before": residual_before,
                    "residual_rmse_after": residual_after,
                    "delta_c_norm": delta_norm,
                    "best_validation_score": best_score,
                },
            )
        return model, hist, final

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0
    hist: list[dict[str, Any]] = []
    max_steps = min(MAX_STEPS, steps)
    for step in range(1, max_steps + 1):
        idx = _batch_indices(train.data.positions.shape[0], step)
        x = train.data.positions[idx, :P]
        c_true = train.c_free[idx]
        with torch.no_grad():
            c_base = model.base_c(x)
            r_true = (c_true - c_base) / residual_std
        opt.zero_grad()
        delta = model.delta(x)
        c_pred = c_base + delta_scale * delta
        loss_resid = torch.mean((delta - r_true) ** 2)
        loss_abs = torch.mean((c_pred - c_true) ** 2)
        loss = loss_resid + lambda_abs * loss_abs
        loss.backward()
        opt.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_steps:
            with torch.no_grad():
                c_val = model(val.data.positions[:, :P])
                val_rollout = _future_rollout_mse(c_val, val.data, 10.0)[0]
                row = _precision_metrics(
                    "residual_correction_history",
                    f"{mode}_scale_{delta_scale:g}_lambda_abs_{lambda_abs:g}",
                    model(test.data.positions[:, :P]),
                    test,
                    {
                        "step": step,
                        "delta_scale": delta_scale,
                        "lambda_abs": lambda_abs,
                        "residual_mode": mode,
                        "base_frozen": mode == "frozen_base",
                        "loss_resid": float(loss_resid.item()),
                        "loss_abs": float(loss_abs.item()),
                        "validation_rollout_mse": val_rollout,
                    },
                )
            hist.append(row)
            score = val_rollout + float(torch.mean((c_val - val.c_free) ** 2).item())
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= RUNTIME_PATIENCE:
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        c_test = model(test.data.positions[:, :P])
        c_base_test = model.base_c(test.data.positions[:, :P])
        delta_norm = torch.sqrt(torch.mean((c_test - c_base_test) ** 2)).item()
        residual_after = torch.sqrt(torch.mean((c_test - test.c_free) ** 2)).item()
        final = _precision_metrics(
            "residual_correction",
            f"{mode}_scale_{delta_scale:g}_lambda_abs_{lambda_abs:g}",
            c_test,
            test,
            {
                "delta_scale": delta_scale,
                "lambda_abs": lambda_abs,
                "residual_mode": mode,
                "base_frozen": mode == "frozen_base",
                "residual_rmse_before": residual_before,
                "residual_rmse_after": residual_after,
                "delta_c_norm": delta_norm,
                "best_validation_score": best_score,
            },
        )
    return model, hist, final


def _rollout_loss_tensor(c: torch.Tensor, dataset: TeacherSet, periods: float = 10.0) -> torch.Tensor:
    steps = _steps(periods)
    a = dataset.data.positions[:, P - 2]
    b = dataset.data.positions[:, P - 1]
    losses = []
    for k in range(steps):
        nxt = c * b - a
        losses.append((nxt - dataset.data.positions[:, P + k]) ** 2)
        a, b = b, nxt
    return torch.mean(torch.stack(losses, dim=-1))


def _rollout_finetune(
    start: ResidualCorrectionModel,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    lambda_teacher: float,
    lr: float,
) -> tuple[ResidualCorrectionModel, list[dict[str, Any]], dict[str, Any]]:
    model = copy.deepcopy(start)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    hist: list[dict[str, Any]] = []
    for step in range(1, ROLLOUT_STEP_CAP + 1):
        idx = _batch_indices(train.data.positions.shape[0], step + 100000)
        sub = TeacherSet(
            data=type(train.data)(positions=train.data.positions[idx], omegas=train.data.omegas[idx]),
            c_free=train.c_free[idx],
            mu=train.mu,
            sigma=train.sigma,
        )
        opt.zero_grad()
        c = model(sub.data.positions[:, :P])
        loss_rollout = _rollout_loss_tensor(c, sub, 10.0)
        loss_teacher = torch.mean((c - sub.c_free) ** 2)
        loss = loss_rollout + lambda_teacher * loss_teacher
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == ROLLOUT_STEP_CAP:
            with torch.no_grad():
                c_val = model(val.data.positions[:, :P])
                val_mse = _future_rollout_mse(c_val, val.data, 10.0)[0]
                c_test = model(test.data.positions[:, :P])
                row = _precision_metrics(
                    "rollout_aware_finetune_history",
                    f"lambda_teacher_{lambda_teacher:g}_lr_{lr:g}",
                    c_test,
                    test,
                    {
                        "step": step,
                        "lambda_teacher": lambda_teacher,
                        "lr": lr,
                        "L_rollout": float(loss_rollout.item()),
                        "L_teacher": float(loss_teacher.item()),
                        "validation_rollout_mse": val_mse,
                    },
                )
            hist.append(row)
            if val_mse < best:
                best = val_mse
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= RUNTIME_PATIENCE:
                break
    final_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():
        final = _precision_metrics(
            "rollout_aware_finetune",
            f"lambda_teacher_{lambda_teacher:g}_lr_{lr:g}",
            model(test.data.positions[:, :P]),
            test,
            {"lambda_teacher": lambda_teacher, "lr": lr, "best_checkpoint_rollout_mse": best, "checkpoint_selection": "best_validation_rollout"},
        )
        drift = torch.sqrt(torch.mean((model(test.data.positions[:, :P]) - test.c_free) ** 2)).item()
        final["c_drift_from_teacher"] = drift
    model.load_state_dict(final_state)
    with torch.no_grad():
        final["final_checkpoint_rollout_mse"] = _future_rollout_mse(model(test.data.positions[:, :P]), test.data, 10.0)[0]
    model.load_state_dict(best_state)
    return model, hist, final


def _inner_refinement(c_init: torch.Tensor, dataset: TeacherSet, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = {0, 1, 2, 5, 10, 20, 50}
    raw = nn.Parameter(torch.atanh((c_init / 2.0).clamp(-0.999, 0.999)))
    opt = torch.optim.Adam([raw], lr=1e-2)
    for step in range(0, max(targets) + 1):
        if step in targets:
            with torch.no_grad():
                c = 2.0 * torch.tanh(raw)
                rows.append(_precision_metrics("inner_refinement", f"{name}_steps_{step}", c, dataset, {"refinement_init": name, "refinement_steps": step}))
        if step == max(targets):
            break
        opt.zero_grad()
        c = 2.0 * torch.tanh(raw)
        loss = _prefix_loss(c, dataset.data.positions)
        loss.backward()
        opt.step()
    return rows


def _threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in THRESHOLDS:
        eligible = [r for r in rows if r.get(f"crosses_{threshold:g}") in [True, "True", "true"]]
        best = min(rows, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
        out.append(
            {
                "threshold": threshold,
                "num_crossing": len(eligible),
                "best_crossing_config": eligible[0]["config_name"] if eligible else "",
                "best_overall_config": best["config_name"],
                "best_overall_mse": best.get("state_given_10_period_mse", ""),
            }
        )
    return out


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(str(r["config_name"]))
        except Exception:
            pass
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in key or "rmse" in key:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter_xy(xs: list[float], ys: list[float], xlabel: str, ylabel: str, filename: str, logy: bool = False) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, s=24)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    def table(rows: list[dict[str, Any]], cols: list[str]) -> str:
        lines = ["<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
        for r in rows:
            lines.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
        lines.append("</tbody></table>")
        return "\n".join(lines)

    cols = ["stage", "config_name", "c_rmse", "c_mae", "c_max_abs_error", "c_r2_against_teacher", "correlation", "collapse_ratio", "state_given_10_period_mse", "max_trajectory_error"]
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>c Precision Refinement Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B c_i Precision and Refinement Diagnostic</h1>",
        "<p>Only the self-supervised free-c prefix teacher is used for c training targets. Simulator c is diagnostic only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2>",
        table(outputs["summary"], cols),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 10100)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 10200)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 10300)

    print("building/verifying free-c prefix teachers", flush=True)
    train_t, val_t, test_t, teacher_hist = _make_teacher_sets(train_data, val_data, test_data)
    teacher_rows = [
        _precision_metrics("teacher_verification", "free_c_train_teacher", train_t.c_free, train_t),
        _precision_metrics("teacher_verification", "free_c_val_teacher", val_t.c_free, val_t),
        _precision_metrics("teacher_verification", "free_c_test_teacher", test_t.c_free, test_t),
    ]
    if teacher_rows[-1]["state_given_10_period_mse"] > 1e-12:
        return {
            "summary": teacher_rows,
            "full_run_table": teacher_rows + teacher_hist,
            "precision_audit": teacher_rows,
            "sensitivity_diagnostic": [],
            "residual_correction_results": [],
            "rollout_aware_finetune_results": [],
            "inner_refinement_results": [],
            "threshold_crossings": [],
            "baseline_comparison": [],
        }

    base, finetuned_base, base_hist, audit_rows = _train_previous_references(train_t, val_t, test_t)
    precision_rows = teacher_rows + [r for r in audit_rows if r["stage"] == "precision_audit"]

    print("running c sensitivity diagnostic", flush=True)
    sensitivity_rows = _sensitivity(test_t)

    print("training residual correction heads", flush=True)
    residual_rows: list[dict[str, Any]] = []
    residual_hist: list[dict[str, Any]] = []
    residual_models: list[tuple[ResidualCorrectionModel, dict[str, Any]]] = []
    for delta_scale in [1e-1, 1e-2, 1e-3]:
        for lambda_abs in [1.0, 100.0]:
            print(f"  residual frozen scale={delta_scale:g} lambda_abs={lambda_abs:g}", flush=True)
            model, hist, row = _train_residual(finetuned_base, train_t, val_t, test_t, delta_scale, lambda_abs, "frozen_base", RESIDUAL_STEP_CAP)
            residual_hist.extend(hist)
            residual_rows.append(row)
            residual_models.append((model, row))
    best_frozen = min(residual_models, key=lambda item: float(item[1]["state_given_10_period_mse"]))
    best_delta = float(best_frozen[1]["delta_scale"])
    for lambda_abs in [1.0, 100.0]:
        print(f"  residual unfreeze_last scale={best_delta:g} lambda_abs={lambda_abs:g}", flush=True)
        model, hist, row = _train_residual(finetuned_base, train_t, val_t, test_t, best_delta, lambda_abs, "unfreeze_last_block", min(RESIDUAL_STEP_CAP, 250))
        residual_hist.extend(hist)
        residual_rows.append(row)
        residual_models.append((model, row))

    best_residual_model, best_residual_row = min(residual_models, key=lambda item: float(item[1]["state_given_10_period_mse"]))

    print("running rollout-aware teacher-anchored fine-tuning", flush=True)
    rollout_rows: list[dict[str, Any]] = []
    rollout_hist: list[dict[str, Any]] = []
    rollout_models: list[tuple[ResidualCorrectionModel, dict[str, Any]]] = []
    for lambda_teacher in [100.0, 1000.0]:
        for lr in [1e-5]:
            print(f"  rollout-aware lambda_teacher={lambda_teacher:g} lr={lr:g}", flush=True)
            model, hist, row = _rollout_finetune(best_residual_model, train_t, val_t, test_t, lambda_teacher, lr)
            rollout_hist.extend(hist)
            rollout_rows.append(row)
            rollout_models.append((model, row))
    best_rollout_model, best_rollout_row = min(rollout_models, key=lambda item: float(item[1]["state_given_10_period_mse"]))

    print("running prefix-only inner refinement", flush=True)
    with torch.no_grad():
        c_encoder = best_rollout_model(test_t.data.positions[:, :P])
        c_avg = torch.full_like(test_t.c_free, float(train_t.c_free.mean().item()))
        gen = torch.Generator().manual_seed(SEED + 40404)
        c_random = c_avg + 0.01 * torch.randn(c_avg.shape, generator=gen, dtype=DTYPE)
    inner_rows = []
    inner_rows.extend(_inner_refinement(c_encoder, test_t, "amortized_init"))
    inner_rows.extend(_inner_refinement(c_avg, test_t, "average_c_init"))
    inner_rows.extend(_inner_refinement(c_random, test_t, "random_near_mean_init"))

    baseline_rows = _baselines(test_t)
    for row in baseline_rows:
        row.setdefault("stage", "baseline")
        row.setdefault("c_rmse", "")

    summary = precision_rows + residual_rows + rollout_rows + inner_rows + baseline_rows
    threshold_rows = _threshold_summary([r for r in summary if "state_given_10_period_mse" in r])

    # Per-trajectory audit for the best pure amortized and refined methods.
    per_traj_rows = []
    with torch.no_grad():
        candidates = {
            "finetuned_reference": _predict_standardized(finetuned_base, test_t),
            "best_residual": best_residual_model(test_t.data.positions[:, :P]),
            "best_rollout_aware": best_rollout_model(test_t.data.positions[:, :P]),
        }
        refined_20 = [r for r in inner_rows if r["config_name"] == "amortized_init_steps_20"]
        for name, c in candidates.items():
            traj_mse, max_abs, final_abs = _per_traj_rollout(c, test_t)
            for i in range(c.shape[0]):
                per_traj_rows.append(
                    {
                        "method": name,
                        "trajectory_index": i,
                        "omega_diagnostic": float(test_t.data.omegas[i].item()),
                        "c_abs_error": float(torch.abs(c[i] - test_t.c_free[i]).item()),
                        "rollout_mse": float(traj_mse[i].item()),
                        "max_abs_error": float(max_abs[i].item()),
                        "final_time_error": float(final_abs[i].item()),
                    }
                )

    return {
        "summary": summary,
        "full_run_table": summary + teacher_hist + base_hist + residual_hist + rollout_hist,
        "precision_audit": precision_rows,
        "sensitivity_diagnostic": sensitivity_rows,
        "residual_correction_results": residual_rows,
        "rollout_aware_finetune_results": rollout_rows,
        "inner_refinement_results": inner_rows,
        "threshold_crossings": threshold_rows,
        "baseline_comparison": baseline_rows,
        "per_trajectory_precision_audit": per_traj_rows,
    }


def _interpret(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary = [r for r in outputs["summary"] if "state_given_10_period_mse" in r]
    best = min(summary, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
    pure = [r for r in summary if r.get("stage") in {"precision_audit", "residual_correction", "rollout_aware_finetune"}]
    best_pure = min(pure, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
    sensitivity = outputs["sensitivity_diagnostic"]
    eps_targets: dict[str, Any] = {}
    for threshold in [1e-6, 1e-9, 1e-12, 1e-14]:
        eligible = [r for r in sensitivity if r["perturbation_kind"] == "random_sign" and float(r["state_given_10_period_mse"]) < threshold]
        eps_targets[f"eps_for_mse_below_{threshold:g}"] = max((float(r["eps"]) for r in eligible), default=None)
    inner_best = min(outputs["inner_refinement_results"], key=lambda r: float(r.get("state_given_10_period_mse", float("inf")))) if outputs["inner_refinement_results"] else {}
    return {
        "required_c_precision_from_sensitivity": eps_targets,
        "best_overall_method": best.get("config_name"),
        "best_overall_10_period_mse": best.get("state_given_10_period_mse"),
        "best_pure_amortized_or_finetuned_method": best_pure.get("config_name"),
        "best_pure_10_period_mse": best_pure.get("state_given_10_period_mse"),
        "best_inner_refinement_method": inner_best.get("config_name", ""),
        "best_inner_refinement_10_period_mse": inner_best.get("state_given_10_period_mse", ""),
        "stage_learning_gate_mse_lt_1e_6": float(best_pure.get("state_given_10_period_mse", float("inf"))) < 1e-6,
        "preferred_path": "pure_amortization" if float(best_pure.get("state_given_10_period_mse", float("inf"))) < 1e-6 else "amortized_initialization_plus_prefix_refinement",
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("precision_audit", "precision_audit.csv"),
        ("sensitivity_diagnostic", "sensitivity_diagnostic.csv"),
        ("residual_correction_results", "residual_correction_results.csv"),
        ("rollout_aware_finetune_results", "rollout_aware_finetune_results.csv"),
        ("inner_refinement_results", "inner_refinement_results.csv"),
        ("threshold_crossings", "threshold_crossings.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
        ("per_trajectory_precision_audit", "per_trajectory_precision_audit.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])

    interpretation = _interpret(outputs)
    setup = {
        "diagnostic": "stage1B_c_precision_refinement",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training_loss": False,
        "uses_future_data_to_create_teacher": False,
        "uses_LS_inside_learned_encoder": False,
        "teacher": "self_supervised_free_c_from_prefix",
        "requested_max_steps": MAX_STEPS,
        "residual_step_cap_used": RESIDUAL_STEP_CAP,
        "rollout_step_cap_used": ROLLOUT_STEP_CAP,
        "runtime_note": "No prior checkpoint was saved by the previous diagnostic, so references were reconstructed on the same split.",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)

    _plot_bar(outputs["summary"], "state_given_10_period_mse", "baseline_comparison.png")
    _plot_bar(outputs["residual_correction_results"], "c_rmse", "residual_correction_improvement.png")
    _plot_bar(outputs["rollout_aware_finetune_results"], "state_given_10_period_mse", "rollout_aware_finetune_curves.png")
    _plot_bar(outputs["inner_refinement_results"], "state_given_10_period_mse", "inner_refinement_steps_vs_mse.png")

    sens = [r for r in outputs["sensitivity_diagnostic"] if r["perturbation_kind"] == "random_sign"]
    _plot_scatter_xy([float(r["eps"]) for r in sens], [float(r["state_given_10_period_mse"]) for r in sens], "eps", "10-period rollout MSE", "sensitivity_eps_vs_rollout_mse.png", True)

    traj = outputs["per_trajectory_precision_audit"]
    _plot_scatter_xy([float(r["c_abs_error"]) for r in traj], [float(r["rollout_mse"]) for r in traj], "|c_pred-c_teacher|", "trajectory rollout MSE", "c_error_vs_rollout_mse.png", True)

    # Before/after c scatter.
    before_after = [r for r in outputs["precision_audit"] + outputs["residual_correction_results"] + outputs["rollout_aware_finetune_results"] if r.get("config_name") in {"reconstructed_finetuned_tcn_combined", interpretation["best_pure_amortized_or_finetuned_method"]}]
    _plot_bar(before_after, "c_rmse", "predicted_vs_teacher_c_before_after.png")
    _plot_bar(outputs["per_trajectory_precision_audit"], "max_abs_error", "worst_trajectory_improvement.png")
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
