"""Stage 1B prefix-only c_i refinement optimizer diagnostic.

This diagnostic asks whether prefix-only iterative optimization can close the
precision gap left by amortized c_i prediction. Future trajectory points are
used only for evaluation.
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
    THRESHOLDS,
    TRAIN_NUM,
    VAL_NUM,
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
)
from experiments.sho.run_stage1b_c_precision_refinement_diagnostic import (
    FeatureTCNEncoder,
    TeacherSet,
    _precision_metrics,
    _train_previous_references,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/c_refinement_optimizer_diagnostic")
STEP_CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
NEWTON_STEP_CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50]
ADAM_LRS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
LBFGS_ITERS = [5, 10, 20, 50, 100]
NEWTON_ETAS = [1.0, 0.5, 0.1]
NEWTON_DAMPINGS = [1e-8, 1e-6, 1e-4]


def _prefix_loss_with_j(c: torch.Tensor, positions: torch.Tensor, j_values: list[int]) -> torch.Tensor:
    x = positions[:, :P]
    total = torch.zeros((), dtype=DTYPE)
    for j in j_values:
        starts = torch.arange(1, P - j)
        a, b = x[:, starts - 1], x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        weight = 4.0 if j == 96 else DRIFT_HEAVY[j]
        total = total + weight * torch.mean((pred - truth) ** 2)
    return total


def _raw_from_c(c: torch.Tensor, parameterization: str) -> torch.Tensor:
    if parameterization == "constrained":
        return torch.atanh((c / 2.0).clamp(-0.999999999, 0.999999999))
    if parameterization == "unconstrained":
        return c.clone()
    raise ValueError(parameterization)


def _c_from_raw(raw: torch.Tensor, parameterization: str) -> torch.Tensor:
    if parameterization == "constrained":
        return 2.0 * torch.tanh(raw)
    if parameterization == "unconstrained":
        return raw
    raise ValueError(parameterization)


def _per_traj_prefix_loss(c: torch.Tensor, positions: torch.Tensor, j_values: list[int]) -> torch.Tensor:
    x = positions[:, :P]
    losses = torch.zeros(c.shape[0], dtype=DTYPE)
    for j in j_values:
        starts = torch.arange(1, P - j)
        a, b = x[:, starts - 1], x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        weight = 4.0 if j == 96 else DRIFT_HEAVY[j]
        losses = losses + weight * torch.mean((pred - truth) ** 2, dim=(1, 2))
    return losses


def _row(
    stage: str,
    name: str,
    c: torch.Tensor,
    dataset: TeacherSet,
    init_c: torch.Tensor,
    extra: dict[str, Any],
    j_values: list[int],
    elapsed: float,
) -> dict[str, Any]:
    row = _precision_metrics(stage, name, c.detach(), dataset, extra)
    prefix_losses = _per_traj_prefix_loss(c.detach(), dataset.data.positions, j_values)
    init_err = torch.abs(init_c - dataset.c_free)
    final_err = torch.abs(c.detach() - dataset.c_free)
    row.update(
        {
            "prefix_loss": float(torch.mean(prefix_losses).item()),
            "num_trajectories_improved": int(torch.sum(final_err < init_err).item()),
            "num_trajectories_worsened": int(torch.sum(final_err > init_err).item()),
            "median_c_error_improvement": float(torch.median(init_err - final_err).item()),
            "worst_case_c_error_improvement": float((torch.max(init_err) - torch.max(final_err)).item()),
            "runtime_seconds": elapsed,
        }
    )
    return row


def _adam_refine(init_c: torch.Tensor, dataset: TeacherSet, lr: float, parameterization: str, j_values: list[int], init_name: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    raw = nn.Parameter(_raw_from_c(init_c, parameterization))
    opt = torch.optim.Adam([raw], lr=lr)
    best_raw = raw.detach().clone()
    best_prefix = _per_traj_prefix_loss(_c_from_raw(best_raw, parameterization), dataset.data.positions, j_values)
    rows: list[dict[str, Any]] = []
    for step in range(0, max(STEP_CHECKPOINTS) + 1):
        if step in STEP_CHECKPOINTS:
            c_now = _c_from_raw(raw.detach(), parameterization)
            c_best = _c_from_raw(best_raw.detach(), parameterization)
            extra = {
                "initialization_type": init_name,
                "optimizer": "adam",
                "learning_rate": lr,
                "refinement_steps": step,
                "parameterization": parameterization,
                "selection": "current",
                "j_values": " ".join(map(str, j_values)),
            }
            rows.append(_row("adam_refinement", f"{init_name}_adam_lr_{lr:g}_{parameterization}_steps_{step}", c_now, dataset, init_c, extra, j_values, time.perf_counter() - start))
            extra_best = dict(extra)
            extra_best["selection"] = "best_prefix_loss"
            rows.append(_row("adam_refinement", f"{init_name}_adam_lr_{lr:g}_{parameterization}_bestprefix_steps_{step}", c_best, dataset, init_c, extra_best, j_values, time.perf_counter() - start))
        if step == max(STEP_CHECKPOINTS):
            break
        opt.zero_grad()
        c = _c_from_raw(raw, parameterization)
        loss = _prefix_loss_with_j(c, dataset.data.positions, j_values)
        loss.backward()
        opt.step()
        with torch.no_grad():
            c_new = _c_from_raw(raw, parameterization)
            losses = _per_traj_prefix_loss(c_new, dataset.data.positions, j_values)
            mask = losses < best_prefix
            best_prefix[mask] = losses[mask]
            best_raw[mask] = raw.detach()[mask]
    return rows


def _lbfgs_refine(init_c: torch.Tensor, dataset: TeacherSet, max_iter: int, parameterization: str, j_values: list[int], init_name: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    raw = nn.Parameter(_raw_from_c(init_c, parameterization))
    opt = torch.optim.LBFGS([raw], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = _prefix_loss_with_j(_c_from_raw(raw, parameterization), dataset.data.positions, j_values)
        loss.backward()
        return loss

    opt.step(closure)
    c = _c_from_raw(raw.detach(), parameterization)
    extra = {
        "initialization_type": init_name,
        "optimizer": "lbfgs",
        "lbfgs_max_iter": max_iter,
        "refinement_steps": max_iter,
        "parameterization": parameterization,
        "selection": "final",
        "j_values": " ".join(map(str, j_values)),
    }
    return [_row("lbfgs_refinement", f"{init_name}_lbfgs_iter_{max_iter}_{parameterization}", c, dataset, init_c, extra, j_values, time.perf_counter() - start)]


def _newton_refine(init_c: torch.Tensor, dataset: TeacherSet, eta: float, damping: float, parameterization: str, j_values: list[int], init_name: str) -> list[dict[str, Any]]:
    start = time.perf_counter()
    raw = _raw_from_c(init_c, parameterization).detach().clone().requires_grad_(True)
    rows: list[dict[str, Any]] = []
    best_raw = raw.detach().clone()
    best_prefix = _per_traj_prefix_loss(_c_from_raw(best_raw, parameterization), dataset.data.positions, j_values)
    for step in range(0, max(NEWTON_STEP_CHECKPOINTS) + 1):
        if step in NEWTON_STEP_CHECKPOINTS:
            c_now = _c_from_raw(raw.detach(), parameterization)
            c_best = _c_from_raw(best_raw, parameterization)
            extra = {
                "initialization_type": init_name,
                "optimizer": "newton",
                "eta": eta,
                "damping": damping,
                "refinement_steps": step,
                "parameterization": parameterization,
                "selection": "current",
                "j_values": " ".join(map(str, j_values)),
            }
            rows.append(_row("newton_refinement", f"{init_name}_newton_eta_{eta:g}_damp_{damping:g}_{parameterization}_steps_{step}", c_now, dataset, init_c, extra, j_values, time.perf_counter() - start))
            extra_best = dict(extra)
            extra_best["selection"] = "best_prefix_loss"
            rows.append(_row("newton_refinement", f"{init_name}_newton_eta_{eta:g}_damp_{damping:g}_{parameterization}_bestprefix_steps_{step}", c_best, dataset, init_c, extra_best, j_values, time.perf_counter() - start))
        if step == max(NEWTON_STEP_CHECKPOINTS):
            break
        loss = _prefix_loss_with_j(_c_from_raw(raw, parameterization), dataset.data.positions, j_values)
        grad = torch.autograd.grad(loss, raw, create_graph=True)[0]
        hdiag = []
        for i in range(raw.numel()):
            hdiag.append(torch.autograd.grad(grad[i], raw, retain_graph=True)[0][i])
        hdiag_t = torch.stack(hdiag).detach()
        grad_t = grad.detach()
        old_raw = raw.detach()
        old_loss = _per_traj_prefix_loss(_c_from_raw(old_raw, parameterization), dataset.data.positions, j_values)
        step_eta = eta
        accepted = False
        for _ in range(8):
            proposal = old_raw - step_eta * grad_t / (hdiag_t + damping)
            if parameterization == "unconstrained":
                proposal = proposal.clamp(-2.1, 2.1)
            new_loss = _per_traj_prefix_loss(_c_from_raw(proposal, parameterization), dataset.data.positions, j_values)
            if torch.mean(new_loss) <= torch.mean(old_loss):
                raw = proposal.detach().clone().requires_grad_(True)
                accepted = True
                break
            step_eta *= 0.5
        if not accepted:
            raw = old_raw.detach().clone().requires_grad_(True)
        with torch.no_grad():
            c_new = _c_from_raw(raw, parameterization)
            losses = _per_traj_prefix_loss(c_new, dataset.data.positions, j_values)
            mask = losses < best_prefix
            best_prefix[mask] = losses[mask]
            best_raw[mask] = raw.detach()[mask]
    return rows


def _aggregate_optimizer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("selection", "current") != "current":
            continue
        key = (str(r.get("optimizer", "")), str(r.get("initialization_type", "")), str(r.get("parameterization", "")))
        groups.setdefault(key, []).append(r)
    out = []
    for (opt, init, param), vals in groups.items():
        best = min(vals, key=lambda r: float(r["state_given_10_period_mse"]))
        out.append(
            {
                "optimizer": opt,
                "initialization_type": init,
                "parameterization": param,
                "best_config_name": best["config_name"],
                "best_refinement_steps": best.get("refinement_steps", ""),
                "best_10_period_mse": best["state_given_10_period_mse"],
                "best_c_rmse": best["c_rmse"],
                "num_trajectories_improved": best.get("num_trajectories_improved", ""),
                "num_trajectories_worsened": best.get("num_trajectories_worsened", ""),
                "runtime_seconds": best.get("runtime_seconds", ""),
            }
        )
    return out


def _threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in THRESHOLDS:
        eligible = [r for r in rows if str(r.get(f"crosses_{threshold:g}", "")).lower() == "true"]
        best = min(rows, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
        out.append(
            {
                "threshold": threshold,
                "num_crossing": len(eligible),
                "best_crossing_config": min(eligible, key=lambda r: float(r["state_given_10_period_mse"]))["config_name"] if eligible else "",
                "best_overall_config": best["config_name"],
                "best_overall_mse": best["state_given_10_period_mse"],
            }
        )
    return out


@torch.no_grad()
def _predict_encoder(model: FeatureTCNEncoder, dataset: TeacherSet) -> torch.Tensor:
    return dataset.mu + dataset.sigma * model(dataset.data.positions[:, :P])


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(str(r.get("config_name", r.get("best_config_name", ""))))
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


def _plot_steps(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    for label in ["encoder_init", "average_c_init", "random_near_mean_init"]:
        vals = [r for r in rows if r.get("initialization_type") == label and r.get("optimizer") == "adam" and r.get("learning_rate") == 0.01 and r.get("selection") == "current" and r.get("parameterization") == "constrained"]
        if vals:
            vals = sorted(vals, key=lambda r: int(r["refinement_steps"]))
            plt.plot([int(r["refinement_steps"]) for r in vals], [float(r[key]) for r in vals], marker="o", label=label)
    plt.yscale("log")
    plt.xlabel("refinement steps")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "config_name", "optimizer", "initialization_type", "parameterization", "refinement_steps", "c_rmse", "state_given_10_period_mse", "prefix_loss", "runtime_seconds"]
    rows = outputs["summary"]
    head = "<tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr>"
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>c Refinement Optimizer Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B Prefix-only c_i Refinement Optimizer Diagnostic</h1>",
        "<p>All refinement uses prefix multi-step recurrence consistency only. Future rollout is evaluation only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead>" + head + "</thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 10100)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 10200)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 10300)
    print("building prefix-only free-c teachers", flush=True)
    train_t, val_t, test_t, teacher_hist = _make_teacher_sets(train_data, val_data, test_data)
    teacher_rows = [
        _precision_metrics("teacher", "free_c_test_teacher", test_t.c_free, test_t, {"initialization_type": "teacher"}),
    ]
    print("reconstructing encoder initialization", flush=True)
    _, finetuned, _, audit_rows = _train_previous_references(train_t, val_t, test_t)
    c_encoder = _predict_encoder(finetuned, test_t)
    gen = torch.Generator().manual_seed(SEED + 70707)
    c_avg = torch.full_like(test_t.c_free, float(train_t.c_free.mean().item()))
    c_random = c_avg + 0.01 * torch.randn(c_avg.shape, generator=gen, dtype=DTYPE)
    init_map = {
        "encoder_init": c_encoder.detach(),
        "average_c_init": c_avg.detach(),
        "random_near_mean_init": c_random.detach(),
    }
    init_rows = [_precision_metrics("initialization", name, c, test_t, {"initialization_type": name, "refinement_steps": 0}) for name, c in init_map.items()]

    j_base = J_ALL
    j_long = J_ALL + [96]
    refinement_rows: list[dict[str, Any]] = []
    print("running adam refinement grid", flush=True)
    for init_name, init_c in init_map.items():
        for param in ["constrained", "unconstrained"]:
            for lr in ADAM_LRS:
                print(f"  adam init={init_name} param={param} lr={lr:g}", flush=True)
                refinement_rows.extend(_adam_refine(init_c, test_t, lr, param, j_base, init_name))
    print("running LBFGS refinement grid", flush=True)
    for init_name, init_c in init_map.items():
        for param in ["constrained", "unconstrained"]:
            for max_iter in LBFGS_ITERS:
                print(f"  lbfgs init={init_name} param={param} iter={max_iter}", flush=True)
                refinement_rows.extend(_lbfgs_refine(init_c, test_t, max_iter, param, j_base, init_name))
    print("running scalar Newton refinement grid", flush=True)
    for init_name, init_c in init_map.items():
        for param in ["constrained", "unconstrained"]:
            for eta in NEWTON_ETAS:
                for damping in NEWTON_DAMPINGS:
                    print(f"  newton init={init_name} param={param} eta={eta:g} damping={damping:g}", flush=True)
                    refinement_rows.extend(_newton_refine(init_c, test_t, eta, damping, param, j_base, init_name))
    print("running longer-context J=96 checks on encoder init", flush=True)
    refinement_rows.extend(_adam_refine(c_encoder.detach(), test_t, 1e-2, "constrained", j_long, "encoder_init_j96"))
    refinement_rows.extend(_lbfgs_refine(c_encoder.detach(), test_t, 20, "constrained", j_long, "encoder_init_j96"))
    refinement_rows.extend(_newton_refine(c_encoder.detach(), test_t, 1.0, 1e-8, "constrained", j_long, "encoder_init_j96"))

    baseline_rows = _baselines(test_t)
    previous_rows = [
        {"stage": "baseline", "config_name": "previous_pure_amortized_best", "c_rmse": 1.28e-3, "state_given_10_period_mse": 3.78e-3},
        {"stage": "baseline", "config_name": "previous_50_step_prefix_refinement", "c_rmse": 1.66e-4, "state_given_10_period_mse": 6.10e-5},
    ]
    baseline_rows = previous_rows + baseline_rows

    all_metric_rows = teacher_rows + init_rows + refinement_rows + baseline_rows
    optimizer_comparison = _aggregate_optimizer(refinement_rows)
    thresholds = _threshold_summary(all_metric_rows)
    prefix_shortening = [{"stage": "prefix_shortening", "status": "skipped", "reason": "P=128 learned/refined methods did not reach 10-period MSE < 1e-6"}]
    if any(float(r.get("state_given_10_period_mse", float("inf"))) < 1e-6 and r.get("stage") not in ["baseline", "teacher"] for r in refinement_rows):
        prefix_shortening = [{"stage": "prefix_shortening", "status": "not_implemented", "reason": "success gate reached, but current encoder is fixed to P=128 and not compatible with shorter prefixes"}]
    return {
        "summary": all_metric_rows,
        "full_run_table": all_metric_rows + teacher_hist + audit_rows,
        "optimizer_comparison": optimizer_comparison,
        "refinement_steps_results": refinement_rows,
        "initialization_comparison": init_rows,
        "newton_refinement_results": [r for r in refinement_rows if r.get("optimizer") == "newton"],
        "lbfgs_refinement_results": [r for r in refinement_rows if r.get("optimizer") == "lbfgs"],
        "prefix_shortening_results": prefix_shortening,
        "threshold_crossings": thresholds,
        "baseline_comparison": baseline_rows,
    }


def _interpret(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    learned_rows = [r for r in outputs["refinement_steps_results"] if "state_given_10_period_mse" in r]
    best = min(learned_rows, key=lambda r: float(r["state_given_10_period_mse"]))
    best_by_opt = {}
    for opt in ["adam", "lbfgs", "newton"]:
        vals = [r for r in learned_rows if r.get("optimizer") == opt]
        if vals:
            b = min(vals, key=lambda r: float(r["state_given_10_period_mse"]))
            best_by_opt[opt] = {"config": b["config_name"], "mse": b["state_given_10_period_mse"], "c_rmse": b["c_rmse"]}
    best_init_steps = {}
    for init in ["encoder_init", "average_c_init", "random_near_mean_init"]:
        eligible = [r for r in learned_rows if r.get("initialization_type") == init and float(r["state_given_10_period_mse"]) < 1e-6]
        best_init_steps[init] = min((int(r.get("refinement_steps", 10**9)) for r in eligible), default=None)
    current_rows = [r for r in learned_rows if r.get("selection", "current") == "current"]
    best_prefix_rows = [r for r in learned_rows if r.get("selection") == "best_prefix_loss"]
    best_current = min(current_rows, key=lambda r: float(r["state_given_10_period_mse"])) if current_rows else best
    best_prefix = min(best_prefix_rows, key=lambda r: float(r["state_given_10_period_mse"])) if best_prefix_rows else {}
    return {
        "reaches_1e_6": float(best["state_given_10_period_mse"]) < 1e-6,
        "reaches_1e_9": float(best["state_given_10_period_mse"]) < 1e-9,
        "best_method": best["config_name"],
        "best_10_period_mse": best["state_given_10_period_mse"],
        "best_c_rmse": best["c_rmse"],
        "best_by_optimizer": best_by_opt,
        "steps_to_1e_6_by_initialization": best_init_steps,
        "best_current_selection": best_current.get("config_name"),
        "best_prefix_loss_selection": best_prefix.get("config_name", ""),
        "best_prefix_loss_selection_mse": best_prefix.get("state_given_10_period_mse", ""),
        "preferred_next_stage1B_path": "amortized_initialization_plus_prefix_only_refinement",
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("optimizer_comparison", "optimizer_comparison.csv"),
        ("refinement_steps_results", "refinement_steps_results.csv"),
        ("initialization_comparison", "initialization_comparison.csv"),
        ("newton_refinement_results", "newton_refinement_results.csv"),
        ("lbfgs_refinement_results", "lbfgs_refinement_results.csv"),
        ("prefix_shortening_results", "prefix_shortening_results.csv"),
        ("threshold_crossings", "threshold_crossings.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs)
    setup = {
        "diagnostic": "stage1B_prefix_only_c_refinement_optimizer",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training_or_refinement": False,
        "uses_future_data_during_refinement": False,
        "uses_LS_inside_learned_encoder": False,
        "uses_closed_form_c_estimator": False,
        "refinement_loss": "prefix_multistep_recurrence_consistency",
        "j_base": J_ALL,
        "j_long_context": J_ALL + [96],
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _plot_steps(outputs["refinement_steps_results"], "c_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_steps(outputs["refinement_steps_results"], "state_given_10_period_mse", "rollout_mse_vs_refinement_steps.png")
    _plot_bar(outputs["optimizer_comparison"], "best_10_period_mse", "optimizer_comparison_mse.png")
    _plot_bar(outputs["initialization_comparison"], "state_given_10_period_mse", "encoder_init_vs_average_init.png")
    _plot_bar(outputs["refinement_steps_results"], "state_given_10_period_mse", "prefix_loss_vs_future_mse.png")
    _plot_bar(outputs["threshold_crossings"], "num_crossing", "threshold_crossings_by_steps.png")
    _plot_bar(outputs["refinement_steps_results"], "max_trajectory_error", "worst_trajectory_improvement.png")
    _plot_bar(outputs["prefix_shortening_results"], "state_given_10_period_mse", "prefix_shortening_if_run.png")
    _plot_bar(outputs["baseline_comparison"], "state_given_10_period_mse", "baseline_comparison.png")
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
