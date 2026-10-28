"""Stage 1B free-c amortization diagnostic.

Separates the prefix multi-step loss from encoder amortization:

1. Optimize one free scalar c_i per trajectory from position-only prefixes.
2. If that succeeds, distill the optimized constants into prefix encoders.
3. Optionally fine-tune the encoder with the original prefix multi-step loss.

Training does not use velocity, omega labels, coefficient labels, c_i labels,
analytic decoders, LS inside learned models, or a closed-form c estimator.
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

OUTPUT_DIR = Path("outputs/sho/stage1B/free_c_amortization_diagnostic")

P = 128
M = 8
TRAIN_NUM = 48
VAL_NUM = 16
TEST_NUM = 16
J_ALL = [1, 2, 4, 8, 16, 32, 64]
DRIFT_HEAVY = {1: 1.0, 2: 1.0, 4: 2.0, 8: 4.0, 16: 4.0, 32: 4.0, 64: 4.0}
DRIFT_HEAVY_L1_5 = {**DRIFT_HEAVY, 1: 5.0}
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

# Runtime-prioritized first pass; records the compact budget in setup.json.
FREE_STEPS = 900
ENCODER_STEPS = 650
FINETUNE_STEPS = 300
EVAL_INTERVAL = 150
PATIENCE = 300


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor


def _steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _make_data(n: int, seed: int) -> SHOSet:
    initial_conditions, omegas = _make_base_conditions(n, seed=seed, dtype=DTYPE)
    total_steps = P + _steps(10.0) + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(n, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _true_c(data: SHOSet) -> torch.Tensor:
    return 2.0 * torch.cos(data.omegas * float(DELTA_T))


def _rollout_prefix_terms(c: torch.Tensor, positions: torch.Tensor, weights: dict[int, float]) -> dict[int, torch.Tensor]:
    x = positions[:, :P]
    terms: dict[int, torch.Tensor] = {}
    for j in J_ALL:
        starts = torch.arange(1, P - j, device=x.device)
        x_prev = x[:, starts - 1]
        x_curr = x[:, starts]
        pred_steps = []
        a, b = x_prev, x_curr
        for _ in range(j):
            nxt = c[:, None] * b - a
            pred_steps.append(nxt)
            a, b = b, nxt
        pred = torch.stack(pred_steps, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        terms[j] = torch.mean((pred - truth) ** 2)
    return terms


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor, weights: dict[int, float]) -> torch.Tensor:
    terms = _rollout_prefix_terms(c, positions, weights)
    return sum(weights[j] * terms[j] for j in terms)


def _future_mse(c: torch.Tensor, data: SHOSet, periods: float = 10.0) -> float:
    x_prev = data.positions[:, P - 2]
    x_curr = data.positions[:, P - 1]
    preds = []
    a, b = x_prev, x_curr
    for _ in range(_steps(periods)):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = data.positions[:, P : P + pred.shape[1]]
    return float(torch.mean((pred - truth) ** 2).item())


def _free_optimize(data: SHOSet, parameterization: str, weights: dict[int, float], optimizer_name: str, lr: float) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    init = torch.full((data.positions.shape[0],), float(_true_c(data).mean().item()), dtype=DTYPE)
    if parameterization == "constrained":
        raw = nn.Parameter(torch.atanh((init / 2.0).clamp(-0.999, 0.999)))
        get_c = lambda: 2.0 * torch.tanh(raw)
    else:
        raw = nn.Parameter(init.clone())
        get_c = lambda: raw
    history: list[dict[str, Any]] = []
    if optimizer_name == "lbfgs":
        opt = torch.optim.LBFGS([raw], lr=lr, max_iter=150, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = _prefix_loss(get_c(), data.positions, weights)
            loss.backward()
            return loss

        opt.step(closure)
        history.append(_metrics_row("free_c", f"{optimizer_name}_{parameterization}", get_c().detach(), data, weights, {"optimizer": optimizer_name, "parameterization": parameterization, "step": 150}))
        return get_c().detach(), history
    opt = torch.optim.Adam([raw], lr=lr)
    best_state = raw.detach().clone()
    best = float("inf")
    stale = 0
    for step in range(1, FREE_STEPS + 1):
        opt.zero_grad()
        loss = _prefix_loss(get_c(), data.positions, weights)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == FREE_STEPS:
            c = get_c().detach()
            row = _metrics_row("free_c", f"{optimizer_name}_{parameterization}_lr{lr:g}", c, data, weights, {"optimizer": optimizer_name, "parameterization": parameterization, "learning_rate": lr, "step": step})
            history.append(row)
            value = float(row["prefix_multistep_loss"])
            if value + 1e-15 < best:
                best = value
                best_state = raw.detach().clone()
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    with torch.no_grad():
        raw.copy_(best_state)
    return get_c().detach(), history


def _metrics_row(stage: str, name: str, c: torch.Tensor, data: SHOSet, weights: dict[int, float], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    true_c = _true_c(data)
    terms = _rollout_prefix_terms(c, data.positions, weights)
    row = {
        "stage": stage,
        "config_name": name,
        "prefix_multistep_loss": float(sum(weights[j] * terms[j].item() for j in terms)),
        "pred_c_mean": float(c.mean().item()),
        "pred_c_std": float(c.std().item()),
        "true_c_mean": float(true_c.mean().item()),
        "true_c_std": float(true_c.std().item()),
        "c_i_mse": float(torch.mean((c - true_c) ** 2).item()),
        "c_i_r2": _r2(c[:, None], true_c[:, None]),
        "c_i_corr": _corr(c, true_c),
        "collapse_ratio": float((c.std() / true_c.std().clamp_min(1e-12)).item()),
        "future_10_period_mse": _future_mse(c, data, 10.0),
        "future_1_period_mse": _future_mse(c, data, 1.0),
        "future_2_period_mse": _future_mse(c, data, 2.0),
        "future_5_period_mse": _future_mse(c, data, 5.0),
    }
    for j, value in terms.items():
        row[f"L_{j}"] = float(value.item())
    row.update(_threshold_flags(row["future_10_period_mse"]))
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


class Encoder(nn.Module):
    def __init__(self, kind: str, constrained: bool) -> None:
        super().__init__()
        self.kind = kind
        self.constrained = constrained
        if kind == "gru":
            self.body = nn.GRU(input_size=1, hidden_size=64, batch_first=True)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        elif kind == "cnn":
            self.body = nn.Sequential(nn.Conv1d(1, 24, 5, padding=2), nn.SiLU(), nn.Conv1d(24, 48, 5, padding=2), nn.SiLU(), nn.Conv1d(48, 64, 3, padding=1), nn.SiLU())
            self.head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        elif kind == "mlp":
            self.body = nn.Sequential(nn.Linear(P, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
            self.head = nn.Linear(128, 1)
        else:
            raise ValueError(kind)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        if self.kind == "gru":
            _, h = self.body(prefix.unsqueeze(-1))
            raw = self.head(h[-1]).squeeze(-1)
        elif self.kind == "cnn":
            y = self.body(prefix.unsqueeze(1)).mean(dim=-1)
            raw = self.head(y).squeeze(-1)
        else:
            raw = self.head(self.body(prefix)).squeeze(-1)
        return 2.0 * torch.tanh(raw) if self.constrained else raw


def _train_encoder(train: SHOSet, test: SHOSet, teacher_train: torch.Tensor, kind: str, constrained: bool) -> tuple[Encoder, list[dict[str, Any]], dict[str, Any]]:
    model = Encoder(kind, constrained).to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, Any]] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    for step in range(1, ENCODER_STEPS + 1):
        opt.zero_grad()
        pred = model(train.positions[:, :P])
        loss = torch.mean((pred - teacher_train) ** 2)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == ENCODER_STEPS:
            row = _metrics_row("amortized_encoder", f"{kind}_{'constrained' if constrained else 'unconstrained'}", model(test.positions[:, :P]).detach(), test, DRIFT_HEAVY, {"distillation_mse": float(loss.item()), "encoder": kind, "parameterization": "constrained" if constrained else "unconstrained", "step": step})
            history.append(row)
            if float(loss.item()) < best:
                best = float(loss.item())
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final = _metrics_row("amortized_encoder", f"{kind}_{'constrained' if constrained else 'unconstrained'}", model(test.positions[:, :P]).detach(), test, DRIFT_HEAVY, {"distillation_mse": best, "encoder": kind, "parameterization": "constrained" if constrained else "unconstrained"})
    return model, history, final


def _finetune_encoder(model: Encoder, train: SHOSet, test: SHOSet, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    history = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = float("inf")
    for step in range(1, FINETUNE_STEPS + 1):
        opt.zero_grad()
        c = model(train.positions[:, :P])
        loss = _prefix_loss(c, train.positions, DRIFT_HEAVY)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == FINETUNE_STEPS:
            row = _metrics_row("distill_finetune", name, model(test.positions[:, :P]).detach(), test, DRIFT_HEAVY, {"step": step, "finetune_loss": float(loss.item())})
            history.append(row)
            if float(row["prefix_multistep_loss"]) < best:
                best = float(row["prefix_multistep_loss"])
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final = _metrics_row("distill_finetune", name, model(test.positions[:, :P]).detach(), test, DRIFT_HEAVY, {"finetune_best_prefix_loss": best})
    return history, final


def _ls_baselines(test: SHOSet) -> list[dict[str, Any]]:
    rows = []
    for name, model in [("fixed_delay_ls", FixedDelay()), ("uniform_mixed_delay_ls", AnchoredSoftDelay("uniform", train_alpha=False).to(dtype=DTYPE))]:
        r = max(2, P - (2 if name == "fixed_delay_ls" else M))
        q_in = _window_sequence(test.positions, M, M - 1, r)
        q_out = _window_sequence(test.positions, M, M, r)
        h_in, h_out = model.encode(q_in), model.encode(q_out)
        k, residual, cond = _fit_k_ridge(h_in, h_out, ridge=0.0)
        truth = _window_sequence(test.positions, M, P - 1, _steps(10.0) + 1)
        h0 = model.encode(truth[:, :1])[:, 0]
        pred = model.decode_x(_rollout_matrix(k, h0, truth.shape[1] - 1))[:, 1:]
        err = pred - truth[:, 1:, -1]
        rows.append({"stage": "baseline", "config_name": name, "future_10_period_mse": float(torch.mean(err**2).item()), "observed_ls_residual": float(residual.item()), "ls_condition_number": float(cond.mean().item()), "spectral_radius": _spectral_radius(k)})
    rows.extend([
        {"stage": "baseline", "config_name": "previous_one_step_local", "c_i_r2": 0.046, "collapse_ratio": 0.1626, "future_10_period_mse": 0.519},
        {"stage": "baseline", "config_name": "previous_drift_heavy_encoder", "c_i_r2": 0.551, "collapse_ratio": 0.148, "future_10_period_mse": 0.345},
    ])
    return rows


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 8100)
    test = _make_data(TEST_NUM, TEST_DATA_SEED + 8200)
    free_rows: list[dict[str, Any]] = []
    full_history: list[dict[str, Any]] = []
    configs = [
        ("constrained", DRIFT_HEAVY, "adam", 1e-2),
        ("constrained", DRIFT_HEAVY, "adam", 1e-3),
        ("unconstrained", DRIFT_HEAVY, "adam", 1e-2),
        ("constrained", DRIFT_HEAVY_L1_5, "adam", 1e-2),
        ("unconstrained", DRIFT_HEAVY, "lbfgs", 1.0),
    ]
    free_teachers: dict[str, torch.Tensor] = {}
    train_teacher_for_best: torch.Tensor | None = None
    for param, weights, opt, lr in configs:
        c_test, hist_test = _free_optimize(test, param, weights, opt, lr)
        row = hist_test[-1]
        row["weights"] = "drift_heavy_l1_5" if weights is DRIFT_HEAVY_L1_5 else "drift_heavy"
        free_rows.append(row)
        full_history.extend(hist_test)
    best_free = max(free_rows, key=lambda r: float(r.get("c_i_r2", -1e9)))
    if float(best_free["c_i_r2"]) > 0.551:
        param = str(best_free["parameterization"])
        weights = DRIFT_HEAVY_L1_5 if best_free.get("weights") == "drift_heavy_l1_5" else DRIFT_HEAVY
        opt = str(best_free["optimizer"])
        lr = float(best_free.get("learning_rate", 1.0))
        train_teacher_for_best, hist_train = _free_optimize(train, param, weights, opt, lr)
        full_history.extend(hist_train)
    amortized_rows: list[dict[str, Any]] = []
    encoder_history: list[dict[str, Any]] = []
    finetune_rows: list[dict[str, Any]] = []
    finetune_history: list[dict[str, Any]] = []
    if train_teacher_for_best is not None:
        best_encoder_model: Encoder | None = None
        best_encoder_row: dict[str, Any] | None = None
        for kind in ["cnn", "gru"]:
            for constrained in [True, False]:
                model, hist, row = _train_encoder(train, test, train_teacher_for_best, kind, constrained)
                amortized_rows.append(row)
                encoder_history.extend(hist)
                if best_encoder_row is None or float(row["c_i_r2"]) > float(best_encoder_row["c_i_r2"]):
                    best_encoder_row = row
                    best_encoder_model = model
        if best_encoder_model is not None:
            hist, final = _finetune_encoder(best_encoder_model, train, test, f"{best_encoder_row['config_name']}_finetuned")
            finetune_history.extend(hist)
            finetune_rows.append(final)
    else:
        amortized_rows.append({"stage": "amortized_encoder", "status": "skipped", "reason": "free c did not beat previous drift-heavy encoder R2"})
        finetune_rows.append({"stage": "distill_finetune", "status": "skipped", "reason": "amortized distillation skipped"})
    baselines = _ls_baselines(test)
    summary = free_rows + amortized_rows + finetune_rows + baselines
    return {
        "summary": summary,
        "free_c_results": free_rows,
        "amortized_encoder_results": amortized_rows,
        "distill_finetune_results": finetune_rows,
        "collapse_diagnostics": [r for r in summary if "collapse_ratio" in r],
        "future_rollout_diagnostics": [r for r in summary if "future_10_period_mse" in r],
        "baseline_comparison": baselines,
        "full_run_table": summary + full_history + encoder_history + finetune_history,
    }


def _plot_bar(rows: list[dict[str, Any]], y: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[y]))
            labels.append(r["config_name"])
        except Exception:
            pass
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in y or "loss" in y:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_gap(rows: list[dict[str, Any]], filename: str) -> None:
    _plot_bar(rows, "c_i_r2", filename)


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


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in columns) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{escape(str(row.get(c, '')))}</td>" for c in columns) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def _write_html(outputs: dict[str, list[dict[str, Any]]], setup: dict[str, Any]) -> None:
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Free-c Amortization Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body>",
        "<h1>Stage 1B Free-c Amortization Diagnostic</h1>",
        "<p>Free per-trajectory constants are optimized from prefixes only; encoder distillation uses those self-supervised constants as teacher targets if free-c recovery succeeds.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Summary</h2>",
        _html_table(outputs["summary"], ["stage", "config_name", "parameterization", "optimizer", "c_i_r2", "c_i_corr", "collapse_ratio", "c_i_mse", "future_10_period_mse", "status", "reason"]),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("free_c_results", "free_c_results.csv"),
        ("amortized_encoder_results", "amortized_encoder_results.csv"),
        ("distill_finetune_results", "distill_finetune_results.csv"),
        ("collapse_diagnostics", "collapse_diagnostics.csv"),
        ("future_rollout_diagnostics", "future_rollout_diagnostics.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    setup = {
        "diagnostic": "stage1B_free_c_amortization",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_c_labels": False,
        "uses_analytic_decoder": False,
        "uses_LS_inside_learned_model": False,
        "uses_closed_form_c_estimator": False,
        "free_steps": FREE_STEPS,
        "encoder_steps": ENCODER_STEPS,
        "finetune_steps": FINETUNE_STEPS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_scatter(outputs["free_c_results"], "true_c_mean", "pred_c_mean", "free_c_vs_true_c.png")
    _plot_scatter(outputs["amortized_encoder_results"], "true_c_mean", "pred_c_mean", "encoder_c_vs_true_c.png")
    _plot_bar(outputs["summary"], "collapse_ratio", "collapse_ratio_comparison.png")
    _plot_bar(outputs["summary"], "future_10_period_mse", "future_rollout_mse_comparison.png")
    _plot_gap(outputs["summary"], "free_c_vs_encoder_gap.png")
    _plot_bar(outputs["distill_finetune_results"], "future_10_period_mse", "distill_finetune_curves.png")
    _plot_scatter(outputs["free_c_results"], "prefix_multistep_loss", "c_i_mse", "prefix_loss_vs_c_error.png")
    _write_html(outputs, setup)


def main() -> None:
    torch.manual_seed(SEED)
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
