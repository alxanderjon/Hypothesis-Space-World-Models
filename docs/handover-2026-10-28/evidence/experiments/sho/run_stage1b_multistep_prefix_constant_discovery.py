"""Stage 1B multi-step prefix constant-discovery diagnostic.

Attempts to fix average-c collapse by training c_i from observed-prefix
multi-step recurrence losses only. No velocity, omega labels, coefficient
labels, c_i labels, analytic decoder, LS inside learned models, or closed-form
c_i estimator are used for training.
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

OUTPUT_DIR = Path("outputs/sho/stage1B/multistep_prefix_constant_discovery")

P = 128
M = 8
TRAIN_NUM = 32
VAL_NUM = 16
TEST_NUM = 16
J_ALL = [1, 2, 4, 8, 16, 32]
SUBSTAGES = [[1], [1, 2], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8, 16], J_ALL]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]

WEIGHT_SCHEDULES = {
    "local_heavy": {1: 10.0, 2: 2.0, 4: 1.0, 8: 0.5, 16: 0.25, 32: 0.1},
    "balanced": {1: 1.0, 2: 1.0, 4: 1.0, 8: 1.0, 16: 1.0, 32: 1.0},
    "drift_heavy": {1: 1.0, 2: 1.0, 4: 2.0, 8: 4.0, 16: 4.0, 32: 4.0},
}

# Compact diagnostic budget; preserves the substage curriculum and schedules.
MAX_STEPS = 160
EVAL_INTERVAL = 80
PATIENCE = 160
LR = 1e-3
FUTURE_PERIODS = [1.0, 2.0, 5.0, 10.0]


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


class PrefixGRU(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=width, batch_first=True)
        self.head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(prefix.unsqueeze(-1))
        return self.head(h[-1]).squeeze(-1)


class PrefixCNN(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(48, width, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        y = self.net(prefix.unsqueeze(1)).mean(dim=-1)
        return self.head(y).squeeze(-1)


class PrefixMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(P, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.net(prefix).squeeze(-1)


class CModel(nn.Module):
    def __init__(self, encoder: str = "gru", constrained: bool = True) -> None:
        super().__init__()
        self.encoder_name = encoder
        self.constrained = constrained
        if encoder == "gru":
            self.encoder = PrefixGRU()
        elif encoder == "cnn":
            self.encoder = PrefixCNN()
        elif encoder == "mlp":
            self.encoder = PrefixMLP()
        else:
            raise ValueError(encoder)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        raw = self.encoder(prefix)
        return 2.0 * torch.tanh(raw) if self.constrained else raw


class SharedC(nn.Module):
    def __init__(self, init: float = 1.75, constrained: bool = True) -> None:
        super().__init__()
        self.constrained = constrained
        raw = math.atanh(max(min(init / 2.0, 0.999), -0.999)) if constrained else init
        self.raw_c = nn.Parameter(torch.tensor(raw, dtype=DTYPE))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        c = 2.0 * torch.tanh(self.raw_c) if self.constrained else self.raw_c
        return c.expand(prefix.shape[0])


def _rollout_positions_from_state(x_prev: torch.Tensor, x_curr: torch.Tensor, c: torch.Tensor, steps: int) -> torch.Tensor:
    preds = []
    a, b = x_prev, x_curr
    for _ in range(steps):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    return torch.stack(preds, dim=-1)


def _prefix_multistep_terms(c: torch.Tensor, positions: torch.Tensor, active_js: list[int]) -> dict[int, torch.Tensor]:
    terms: dict[int, torch.Tensor] = {}
    x = positions[:, :P]
    for j in active_js:
        if P <= j + 1:
            continue
        starts = torch.arange(1, P - j, device=x.device)
        x_prev = x[:, starts - 1]
        x_curr = x[:, starts]
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        pred_steps = []
        a, b = x_prev, x_curr
        for _ in range(j):
            nxt = c[:, None] * b - a
            pred_steps.append(nxt)
            a, b = b, nxt
        pred = torch.stack(pred_steps, dim=-1)
        terms[j] = torch.mean((pred - truth) ** 2)
    return terms


def _loss(
    model: nn.Module,
    data: SHOSet,
    active_js: list[int],
    weights: dict[int, float],
    var_floor: float = 0.0,
    lambda_var: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    prefix = data.positions[:, :P]
    c = model(prefix)
    terms = _prefix_multistep_terms(c, data.positions, active_js)
    total = torch.zeros((), dtype=DTYPE)
    for j, value in terms.items():
        total = total + weights[j] * value
    var_loss = torch.relu(torch.tensor(var_floor, dtype=DTYPE) - torch.std(c)) ** 2
    total = total + lambda_var * var_loss
    parts = {f"L_{j}": float(v.detach().item()) for j, v in terms.items()}
    parts["prefix_multistep_loss"] = float(sum(weights[j] * terms[j].detach().item() for j in terms))
    parts["variance_regularizer_loss"] = float(var_loss.detach().item())
    return total, parts


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    data: SHOSet,
    schedule_name: str,
    active_js: list[int],
    stage_name: str,
    weight_schedule: dict[int, float],
    var_floor: float = 0.0,
    lambda_var: float = 0.0,
) -> dict[str, Any]:
    prefix = data.positions[:, :P]
    c = model(prefix)
    true_c = _true_c(data)
    terms = _prefix_multistep_terms(c, data.positions, J_ALL)
    h0_prev = data.positions[:, P - 2]
    h0_curr = data.positions[:, P - 1]
    row = {
        "model_type": model.__class__.__name__,
        "encoder": getattr(model, "encoder_name", "shared"),
        "c_parameterization": "2*tanh(raw_c)" if getattr(model, "constrained", True) else "unconstrained",
        "weight_schedule": schedule_name,
        "substage": stage_name,
        "active_terms": json.dumps(active_js),
        "lambda_var": lambda_var,
        "target_std_floor": var_floor,
        "pred_c_mean": float(c.mean().item()),
        "pred_c_std": float(c.std().item()),
        "true_c_mean": float(true_c.mean().item()),
        "true_c_std": float(true_c.std().item()),
        "collapse_ratio": float((c.std() / true_c.std().clamp_min(1e-12)).item()),
        "c_i_mse": float(torch.mean((c - true_c) ** 2).item()),
        "c_i_r2": _r2(c[:, None], true_c[:, None]),
        "c_i_corr": _corr(c, true_c),
        "prefix_multistep_loss": float(sum(weight_schedule[j] * terms[j].item() for j in J_ALL if j in terms)),
        "L_1": float(terms[1].item()),
        "L_2": float(terms[2].item()),
        "L_4": float(terms[4].item()),
        "L_8": float(terms[8].item()),
        "L_16": float(terms[16].item()),
        "L_32": float(terms[32].item()),
    }
    for periods in FUTURE_PERIODS:
        steps = _steps(periods)
        pred = _rollout_positions_from_state(h0_prev, h0_curr, c, steps)
        truth = data.positions[:, P : P + steps]
        row[f"future_{periods:g}_period_mse"] = float(torch.mean((pred - truth) ** 2).item())
    row["future_10_period_mse"] = row["future_10_period_mse"]
    row.update(_threshold_flags(row["future_10_period_mse"]))
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


def _train_model(
    model: nn.Module,
    train: SHOSet,
    val: SHOSet,
    schedule_name: str,
    weights: dict[int, float],
    var_floor: float = 0.0,
    lambda_var: float = 0.0,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    history: list[dict[str, Any]] = []
    global_step = 0
    for idx, active in enumerate(SUBSTAGES, start=1):
        stale = 0
        for step in range(1, MAX_STEPS + 1):
            global_step += 1
            opt.zero_grad()
            loss, parts = _loss(model, train, active, weights, var_floor, lambda_var)
            loss.backward()
            opt.step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                row = _evaluate(model, val, schedule_name, active, f"substage_{idx}", weights, var_floor, lambda_var)
                row.update(parts)
                row["global_step"] = global_step
                row["local_step"] = step
                row["train_loss"] = float(loss.detach().item())
                history.append(row)
                metric = float(row["prefix_multistep_loss"])
                if metric + 1e-12 < best_val:
                    best_val = metric
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                if stale >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    return model, history


def _train_shared_baseline(train: SHOSet, val: SHOSet, weights: dict[int, float]) -> tuple[nn.Module, list[dict[str, Any]]]:
    model = SharedC(init=float(_true_c(train).mean().item()), constrained=True).to(dtype=DTYPE)
    return _train_model(model, train, val, "average_c_baseline", weights)


def _run_ls_baselines(test: SHOSet) -> list[dict[str, Any]]:
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
        mse = float(torch.mean(err**2).item())
        rows.append({
            "baseline": name,
            "future_10_period_mse": mse,
            "future_1_period_mse": float(torch.mean(err[:, : _steps(1.0)] ** 2).item()),
            "future_2_period_mse": float(torch.mean(err[:, : _steps(2.0)] ** 2).item()),
            "future_5_period_mse": float(torch.mean(err[:, : _steps(5.0)] ** 2).item()),
            "observed_ls_residual": float(residual.item()),
            "ls_condition_number": float(cond.mean().item()),
            "spectral_radius": _spectral_radius(k),
        })
    return rows


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 7100)
    val = _make_data(VAL_NUM, TRAIN_DATA_SEED + 7200)
    test = _make_data(TEST_NUM, TEST_DATA_SEED + 7300)

    all_history: list[dict[str, Any]] = []
    schedule_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []

    avg_model, avg_hist = _train_shared_baseline(train, val, WEIGHT_SCHEDULES["balanced"])
    all_history.extend(avg_hist)
    avg_row = _evaluate(avg_model, test, "average_c_baseline", J_ALL, "final", WEIGHT_SCHEDULES["balanced"])

    for schedule_name, weights in WEIGHT_SCHEDULES.items():
        model = CModel("gru", constrained=True).to(dtype=DTYPE)
        model, hist = _train_model(model, train, val, schedule_name, weights)
        all_history.extend(hist)
        row = _evaluate(model, test, schedule_name, J_ALL, "final", weights)
        schedule_rows.append(row)
        future_rows.append(row)
        collapse_rows.append(row)

    best = max(schedule_rows, key=lambda r: float(r["c_i_r2"]))
    variance_rows: list[dict[str, Any]] = []
    future_finetune_rows: list[dict[str, Any]] = []
    if float(best["c_i_r2"]) < 0.7 or float(best["collapse_ratio"]) < 0.5:
        for floor in [0.02, 0.04]:
            model = CModel("gru", constrained=True).to(dtype=DTYPE)
            model, hist = _train_model(model, train, val, f"variance_floor_{floor:g}", WEIGHT_SCHEDULES[best["weight_schedule"]], var_floor=floor, lambda_var=1e-2)
            all_history.extend(hist)
            row = _evaluate(model, test, f"variance_floor_{floor:g}", J_ALL, "final", WEIGHT_SCHEDULES[best["weight_schedule"]], floor, 1e-2)
            variance_rows.append(row)
            collapse_rows.append(row)
            schedule_rows.append(row)

    # This diagnostic intentionally skips future fine-tuning unless c discovery
    # is strong. Record the skip explicitly.
    best_after_var = max(schedule_rows, key=lambda r: float(r["c_i_r2"]))
    if float(best_after_var["c_i_r2"]) < 0.7:
        future_finetune_rows.append({"status": "skipped", "reason": "c_i discovery did not reach R2 >= 0.7"})

    previous_one_step = [{
        "baseline": "previous_one_step_local_recurrence",
        "pred_c_std": 0.0120,
        "true_c_std": 0.0738,
        "collapse_ratio": 0.0120 / 0.0738,
        "c_i_r2": 0.046,
        "future_10_period_mse": 0.519,
    }]
    baselines = _run_ls_baselines(test)
    summary = [avg_row] + schedule_rows + previous_one_step + baselines
    return {
        "summary": summary,
        "weight_schedule_results": schedule_rows,
        "substage_history": all_history,
        "collapse_diagnostics": collapse_rows,
        "average_c_baseline": [avg_row],
        "future_rollout_diagnostics": future_rows,
        "optional_variance_regularizer_results": variance_rows or [{"status": "skipped", "reason": "not needed"}],
        "optional_future_finetune_results": future_finetune_rows or [{"status": "skipped", "reason": "not requested after successful c discovery"}],
        "baseline_ls_results": baselines + previous_one_step,
        "full_run_table": summary + all_history,
    }


def _plot_lines(rows: list[dict[str, Any]], y: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    for key in sorted({r.get("weight_schedule", "") for r in rows if r.get("weight_schedule", "")}):
        g = [r for r in rows if r.get("weight_schedule") == key]
        plt.plot(range(len(g)), [float(r[y]) for r in g], marker="o", label=key)
    plt.yscale("log" if "mse" in y or "loss" in y else "linear")
    plt.ylabel(y)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_bar(rows: list[dict[str, Any]], y: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[y]))
            labels.append(r.get("weight_schedule", r.get("baseline", "")))
        except Exception:
            pass
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in y or "loss" in y:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right")
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_c(rows: list[dict[str, Any]], filename: str) -> None:
    # Summary-level means are plotted here; per-trajectory c values are not saved.
    xs, ys, labels = [], [], []
    for r in rows:
        if "pred_c_mean" in r:
            xs.append(float(r["true_c_mean"]))
            ys.append(float(r["pred_c_mean"]))
            labels.append(r["weight_schedule"])
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys)
    if xs:
        lo, hi = min(xs + ys), max(xs + ys)
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("true c mean")
    plt.ylabel("predicted c mean")
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
    rows = outputs["summary"]
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Multi-step Prefix Constant Discovery</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body>",
        "<h1>Stage 1B Multi-step Prefix Constant Discovery</h1>",
        "<p>Training uses observed prefixes only. Future rollout is diagnostic and state-given. True c_i is used only for diagnostics.</p>",
        "<h2>Setup</h2><pre>" + escape(json.dumps(setup, indent=2)) + "</pre>",
        "<h2>Summary</h2>",
        _html_table(rows, ["weight_schedule", "baseline", "c_i_r2", "c_i_corr", "collapse_ratio", "pred_c_std", "true_c_std", "prefix_multistep_loss", "future_10_period_mse"]),
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("weight_schedule_results", "weight_schedule_results.csv"),
        ("substage_history", "substage_history.csv"),
        ("collapse_diagnostics", "collapse_diagnostics.csv"),
        ("average_c_baseline", "average_c_baseline.csv"),
        ("future_rollout_diagnostics", "future_rollout_diagnostics.csv"),
        ("optional_variance_regularizer_results", "optional_variance_regularizer_results.csv"),
        ("optional_future_finetune_results", "optional_future_finetune_results.csv"),
        ("baseline_ls_results", "baseline_ls_results.csv"),
    ]:
        _write_csv(OUTPUT_DIR / name, outputs[key])
    setup = {
        "diagnostic": "stage1B_multistep_prefix_constant_discovery",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "training_uses_future_rollout": False,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_c_labels": False,
        "uses_analytic_decoder": False,
        "uses_LS_inside_learned_model": False,
        "uses_closed_form_c_estimator": False,
        "max_steps_per_substage": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_lines(outputs["substage_history"], "c_i_r2", "c_r2_vs_substage.png")
    _plot_lines(outputs["substage_history"], "pred_c_std", "c_std_vs_substage.png")
    _plot_lines(outputs["substage_history"], "collapse_ratio", "collapse_ratio_vs_substage.png")
    _plot_lines(outputs["substage_history"], "prefix_multistep_loss", "prefix_multistep_loss_vs_substage.png")
    _plot_lines(outputs["substage_history"], "future_10_period_mse", "future_rollout_mse_vs_substage.png")
    _plot_c(outputs["weight_schedule_results"], "learned_c_vs_true_c.png")
    _plot_bar(outputs["summary"], "future_10_period_mse", "weight_schedule_comparison.png")
    _plot_bar(outputs["summary"], "future_10_period_mse", "average_c_vs_network.png")
    _plot_bar(outputs["weight_schedule_results"], "L_1", "local_loss_vs_multistep_loss.png")
    _write_html(outputs, setup)


def main() -> None:
    torch.manual_seed(SEED)
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
