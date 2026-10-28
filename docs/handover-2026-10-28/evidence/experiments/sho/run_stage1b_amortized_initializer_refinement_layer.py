"""Stage 1B amortized initializer + prefix-refinement layer diagnostic.

This diagnostic trains a prefix encoder as an initializer for a scalar
trajectory dynamics parameter c_i, then applies a small differentiable
prefix-only refinement layer. Future rollout is evaluation-only.
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
from experiments.sho.run_stage1b_c_refinement_optimizer_diagnostic import _newton_refine
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

OUTPUT_DIR = Path("outputs/sho/stage1B/amortized_initializer_refinement_layer")

# Runtime-prioritized defaults; override from the shell for a larger pass.
BATCH_SIZE = int(os.environ.get("AIRL_BATCH_SIZE", "8"))
MAX_STEPS = int(os.environ.get("AIRL_MAX_STEPS", "1200"))
EVAL_INTERVAL = int(os.environ.get("AIRL_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("AIRL_PATIENCE", "600"))
BETA_INIT = float(os.environ.get("AIRL_BETA_INIT", "0.01"))
FIXED_ETA = float(os.environ.get("AIRL_FIXED_ETA", "0.5"))
FIXED_DAMPING = float(os.environ.get("AIRL_FIXED_DAMPING", "1e-4"))


class RefinementTCN(nn.Module):
    """TCN prefix encoder with c, eta, and damping heads."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8]))
        self.shared = nn.Sequential(nn.Linear(32, 48), nn.SiLU())
        self.c_head = nn.Linear(48, 1)
        self.eta_head = nn.Linear(48, 1)
        self.damping_head = nn.Linear(48, 1)

    def features(self, prefix: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.inp(prefix.unsqueeze(1)))
        y = self.blocks(y).mean(dim=-1)
        return self.shared(y)

    def forward(self, prefix: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.features(prefix)
        raw_c = self.c_head(feat).squeeze(-1)
        return {
            "raw_c": raw_c,
            "c_init": _c_from_raw(raw_c),
            "eta": torch.sigmoid(self.eta_head(feat).squeeze(-1)),
            "damping": torch.nn.functional.softplus(self.damping_head(feat).squeeze(-1)) + 1e-8,
        }


def _c_from_raw(raw: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.tanh(raw)


def _raw_from_c(c: torch.Tensor) -> torch.Tensor:
    return torch.atanh((c / 2.0).clamp(-0.999999999, 0.999999999))


def _prefix_loss_vector(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    losses = torch.zeros(c.shape[0], dtype=DTYPE, device=c.device)
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
        losses = losses + DRIFT_HEAVY[j] * torch.mean((pred - truth) ** 2, dim=(1, 2))
    return losses


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.mean(_prefix_loss_vector(c, positions))


def _newton_refinement_layer(
    c_init: torch.Tensor,
    positions: torch.Tensor,
    k_steps: int,
    *,
    eta: float | torch.Tensor,
    damping: float | torch.Tensor,
    stop_gradient: bool,
    create_graph: bool,
) -> torch.Tensor:
    """Apply K Newton-style prefix refinement steps directly in c-space."""

    c = c_init
    if k_steps <= 0:
        return c
    if not torch.is_tensor(eta):
        eta = torch.full_like(c, float(eta))
    if not torch.is_tensor(damping):
        damping = torch.full_like(c, float(damping))
    for _ in range(k_steps):
        c_work = c.detach().requires_grad_(True) if stop_gradient else c
        losses = _prefix_loss_vector(c_work, positions)
        # A Hessian diagonal requires the first derivative itself to carry a
        # graph, even during evaluation where model parameters are detached.
        grad = torch.autograd.grad(losses.sum(), c_work, create_graph=True)[0]
        hdiag = []
        for i in range(c_work.numel()):
            h = torch.autograd.grad(
                grad[i],
                c_work,
                retain_graph=True,
                create_graph=create_graph and not stop_gradient,
            )[0][i]
            hdiag.append(h)
        hdiag_t = torch.stack(hdiag)
        denom = hdiag_t + damping
        denom = torch.where(torch.abs(denom) < 1e-12, torch.sign(denom).clamp_min(0.0) * 2e-12 - 1e-12, denom)
        c_next = (c_work - eta * grad / denom).clamp(-1.999999999, 1.999999999)
        c = c_next.detach() if stop_gradient else c_next
    return c


def _model_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _future_metrics(c: torch.Tensor, dataset: TeacherSet) -> dict[str, Any]:
    mse10, maxerr, finalerr = _future_rollout_mse(c, dataset.data, 10.0)
    out: dict[str, Any] = {
        "future_1_period_mse": _future_rollout_mse(c, dataset.data, 1.0)[0],
        "future_2_period_mse": _future_rollout_mse(c, dataset.data, 2.0)[0],
        "future_5_period_mse": _future_rollout_mse(c, dataset.data, 5.0)[0],
        "future_10_period_mse": mse10,
        "state_given_1_period_mse": _future_rollout_mse(c, dataset.data, 1.0)[0],
        "state_given_2_period_mse": _future_rollout_mse(c, dataset.data, 2.0)[0],
        "state_given_5_period_mse": _future_rollout_mse(c, dataset.data, 5.0)[0],
        "state_given_10_period_mse": mse10,
        "max_trajectory_error": maxerr,
        "final_time_error": finalerr,
        "final_time_absolute_error": finalerr,
    }
    out.update(_threshold_flags(mse10))
    return out


def _c_metrics(stage: str, name: str, c_init: torch.Tensor, c_refined: torch.Tensor, dataset: TeacherSet, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    teacher = dataset.c_free
    sim_c = _sim_c(dataset.data)
    init_err = c_init - teacher
    ref_err = c_refined - teacher
    row: dict[str, Any] = {
        "stage": stage,
        "config_name": name,
        "c_init_rmse": float(torch.sqrt(torch.mean(init_err**2)).item()),
        "c_refined_rmse": float(torch.sqrt(torch.mean(ref_err**2)).item()),
        "c_refined_mae": float(torch.mean(torch.abs(ref_err)).item()),
        "c_refined_max_abs_error": float(torch.max(torch.abs(ref_err)).item()),
        "c_refined_bias": float(torch.mean(ref_err).item()),
        "c_init_r2_against_teacher": _r2(c_init[:, None], teacher[:, None]),
        "c_refined_r2_against_teacher": _r2(c_refined[:, None], teacher[:, None]),
        "diagnostic_c_r2": _r2(c_refined[:, None], sim_c[:, None]),
        "correlation": _corr(c_refined, teacher),
        "collapse_ratio": float((torch.std(c_refined) / torch.std(teacher).clamp_min(1e-12)).item()),
        "predicted_c_std": float(torch.std(c_refined).item()),
        "teacher_c_std": float(torch.std(teacher).item()),
        "prefix_loss_init": float(_prefix_loss(c_init, dataset.data.positions).item()),
        "prefix_loss_refined": float(_prefix_loss(c_refined, dataset.data.positions).item()),
    }
    row.update(_future_metrics(c_refined, dataset))
    if extra:
        row.update(extra)
    return row


def _refine_from_outputs(outputs: dict[str, torch.Tensor], positions: torch.Tensor, config: dict[str, Any], *, create_graph: bool) -> torch.Tensor:
    method = str(config["method"])
    if method == "none":
        return outputs["c_init"]
    if method == "fixed_newton":
        return _newton_refinement_layer(
            outputs["c_init"],
            positions,
            int(config["k_steps"]),
            eta=float(config["eta"]),
            damping=float(config["damping"]),
            stop_gradient=bool(config["stop_gradient"]),
            create_graph=create_graph,
        )
    if method == "learned_newton":
        return _newton_refinement_layer(
            outputs["c_init"],
            positions,
            int(config["k_steps"]),
            eta=outputs["eta"],
            damping=outputs["damping"],
            stop_gradient=bool(config["stop_gradient"]),
            create_graph=create_graph,
        )
    raise ValueError(method)


def _eval_model(model: RefinementTCN, dataset: TeacherSet, config: dict[str, Any], stage: str) -> dict[str, Any]:
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(dataset.data.positions[:, :P])
        c_init = outputs["c_init"].detach()
    with torch.enable_grad():
        outputs_for_refine = {k: v.detach().clone().requires_grad_(k == "c_init") for k, v in outputs.items()}
        c_ref = _refine_from_outputs(outputs_for_refine, dataset.data.positions, config, create_graph=False).detach()
    runtime = time.perf_counter() - start
    extra = {
        "method": config["method"],
        "refinement_method": config["method"],
        "refinement_steps": config["k_steps"],
        "number_of_refinement_steps": config["k_steps"],
        "gradients_through_refinement": not bool(config["stop_gradient"]),
        "beta_init": config["beta_init"],
        "runtime_seconds": runtime,
        "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0]),
        "parameter_count": _model_parameter_count(model),
    }
    if str(config["method"]) == "learned_newton":
        with torch.no_grad():
            pred = model(dataset.data.positions[:, :P])
            extra.update(
                {
                    "learned_eta_mean": float(pred["eta"].mean().item()),
                    "learned_eta_std": float(pred["eta"].std().item()),
                    "learned_damping_mean": float(pred["damping"].mean().item()),
                    "learned_damping_std": float(pred["damping"].std().item()),
                }
            )
    return _c_metrics(stage, str(config["config_name"]), c_init, c_ref, dataset, extra)


def _train_config(train: TeacherSet, val: TeacherSet, test: TeacherSet, config: dict[str, Any]) -> tuple[RefinementTCN, list[dict[str, Any]], dict[str, Any]]:
    model = RefinementTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 13000 + int(config["config_index"]))
    n = train.data.positions.shape[0]
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        outputs = model(pos[:, :P])
        c_ref = _refine_from_outputs(outputs, pos, config, create_graph=True)
        loss_ref = _prefix_loss(c_ref, pos)
        loss_init = _prefix_loss(outputs["c_init"], pos)
        loss = loss_ref + float(config["beta_init"]) * loss_init
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            row = _eval_model(model, val, config, "training_history")
            row.update(
                {
                    "step": step,
                    "train_total_loss": float(loss.item()),
                    "train_refined_prefix_loss": float(loss_ref.item()),
                    "train_init_prefix_loss": float(loss_init.item()),
                }
            )
            history.append(row)
            score = float(row["prefix_loss_refined"])
            if math.isfinite(score) and score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final = _eval_model(model, test, config, "refinement_layer")
    final["best_validation_prefix_loss"] = best_score
    return model, history, final


def _previous_baselines(test: TeacherSet) -> list[dict[str, Any]]:
    teacher = _c_metrics("baseline", "free_prefix_c_teacher", test.c_free, test.c_free, test, {"baseline_type": "free_c_optimization_from_prefix"})
    rows: list[dict[str, Any]] = [
        teacher,
        {
            "stage": "baseline",
            "config_name": "previous_encoder_only_result",
            "baseline_type": "recorded_previous_result",
            "c_refined_rmse": 1.28e-3,
            "future_10_period_mse": 3.78e-3,
            "state_given_10_period_mse": 3.78e-3,
        },
        {
            "stage": "baseline",
            "config_name": "previous_encoder_plus_explicit_50_step_refinement",
            "baseline_type": "recorded_previous_result",
            "c_refined_rmse": 1.66e-4,
            "future_10_period_mse": 6.10e-5,
            "state_given_10_period_mse": 6.10e-5,
        },
    ]
    for row in _baselines(test):
        row = dict(row)
        row.setdefault("baseline_type", "fixed_or_mixed_delay_ls_reference")
        rows.append(row)
    return rows


def _average_init_refinement_baselines(train: TeacherSet, test: TeacherSet, k_values: list[int]) -> list[dict[str, Any]]:
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    rows: list[dict[str, Any]] = [_c_metrics("baseline", "average_c_init_no_refinement", avg, avg, test, {"baseline_type": "average_c_init"})]
    for k in k_values:
        refined_rows = _newton_refine(avg, test, FIXED_ETA, FIXED_DAMPING, "constrained", J_ALL, "average_c_init")
        matches = [r for r in refined_rows if r.get("selection") == "current" and int(r.get("refinement_steps", -1)) == k]
        if not matches:
            continue
        c_ref = test.c_free.new_tensor(float("nan")).expand_as(test.c_free)
        row = dict(matches[0])
        row.update(
            {
                "stage": "baseline",
                "config_name": f"average_c_init_fixed_newton_K{k}",
                "baseline_type": "average_c_init_same_refinement_layer",
                "method": "fixed_newton",
                "refinement_method": "fixed_newton",
                "refinement_steps": k,
                "number_of_refinement_steps": k,
                "c_init_rmse": float(torch.sqrt(torch.mean((avg - test.c_free) ** 2)).item()),
                "runtime_per_trajectory_seconds": float(row.get("runtime_seconds", 0.0)) / max(1, test.data.positions.shape[0]),
            }
        )
        rows.append(row)
    return rows


def _config_grid() -> list[dict[str, Any]]:
    requested = os.environ.get("AIRL_CONFIGS")
    names = [x.strip() for x in requested.split(",") if x.strip()] if requested else [
        "none_0",
        "fixed_1",
        "fixed_2",
        "fixed_5",
        "fixed_10",
        "fixed_stopgrad_2",
        "learned_1",
        "learned_2",
        "learned_5",
        "learned_stopgrad_2",
    ]
    configs: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        parts = name.split("_")
        method = parts[0]
        stop_gradient = "stopgrad" in parts
        k = int(parts[-1]) if parts[-1].isdigit() else 0
        if method == "none":
            cfg_method = "none"
        elif method == "fixed":
            cfg_method = "fixed_newton"
        elif method == "learned":
            cfg_method = "learned_newton"
        else:
            raise ValueError(f"unknown config {name}")
        configs.append(
            {
                "config_index": idx,
                "config_name": name,
                "method": cfg_method,
                "k_steps": k,
                "eta": FIXED_ETA,
                "damping": FIXED_DAMPING,
                "stop_gradient": stop_gradient,
                "beta_init": BETA_INIT,
                "lr": float(os.environ.get("AIRL_LR", "3e-4")),
            }
        )
    return configs


def _threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in [1e-6, 1e-9, 1e-12]:
        eligible = [r for r in rows if _as_float(r.get("future_10_period_mse", r.get("state_given_10_period_mse", float("inf")))) < threshold]
        best = min(rows, key=lambda r: _as_float(r.get("future_10_period_mse", r.get("state_given_10_period_mse", float("inf")))))
        out.append(
            {
                "threshold": threshold,
                "num_crossing": len(eligible),
                "best_crossing_config": min(eligible, key=lambda r: _as_float(r.get("future_10_period_mse", r.get("state_given_10_period_mse", float("inf")))))["config_name"] if eligible else "",
                "best_overall_config": best.get("config_name", ""),
                "best_overall_future_10_period_mse": best.get("future_10_period_mse", best.get("state_given_10_period_mse", "")),
            }
        )
    return out


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("inf")


def _plot_by_steps(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(7, 5))
    groups = [
        ("fixed_newton", "fixed Newton"),
        ("learned_newton", "learned Newton"),
        ("none", "no refinement"),
    ]
    for method, label in groups:
        vals = [r for r in rows if r.get("method") == method and str(r.get("stage")) == "refinement_layer" and r.get("gradients_through_refinement", True)]
        if not vals:
            continue
        by_k: dict[int, float] = {}
        for r in vals:
            k = int(r.get("refinement_steps", 0))
            by_k[k] = min(by_k.get(k, float("inf")), _as_float(r.get(key)))
        xs = sorted(by_k)
        plt.plot(xs, [by_k[x] for x in xs], marker="o", label=label)
    plt.yscale("log")
    plt.xlabel("refinement steps K")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str, *, stage_filter: str | None = None) -> None:
    vals, labels = [], []
    for row in rows:
        if stage_filter is not None and row.get("stage") != stage_filter:
            continue
        val = _as_float(row.get(key))
        if not math.isfinite(val):
            continue
        vals.append(val)
        labels.append(str(row.get("config_name", "")))
    if not vals:
        return
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in key or "rmse" in key:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=55, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_history(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for name in sorted({str(r.get("config_name", "")) for r in history}):
        vals = sorted([r for r in history if r.get("config_name") == name], key=lambda r: int(r.get("step", 0)))
        if vals:
            plt.plot([int(r["step"]) for r in vals], [_as_float(r["c_refined_rmse"]) for r in vals], label=name)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("validation c_refined_rmse")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_c_rmse_vs_steps.png", dpi=170)
    plt.close()


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    trained = [r for r in summary if r.get("stage") == "refinement_layer"]
    best = min(trained, key=lambda r: _as_float(r.get("future_10_period_mse")), default={})
    fixed = [r for r in trained if r.get("method") == "fixed_newton" and r.get("gradients_through_refinement", True)]
    learned = [r for r in trained if r.get("method") == "learned_newton" and r.get("gradients_through_refinement", True)]
    best_fixed = min(fixed, key=lambda r: _as_float(r.get("future_10_period_mse")), default={})
    best_learned = min(learned, key=lambda r: _as_float(r.get("future_10_period_mse")), default={})
    avg_rows = [r for r in summary if str(r.get("config_name", "")).startswith("average_c_init")]
    best_avg = min(avg_rows, key=lambda r: _as_float(r.get("state_given_10_period_mse", r.get("future_10_period_mse"))), default={})
    return {
        "reaches_1e_6_with_K_le_2": any(_as_float(r.get("future_10_period_mse")) < 1e-6 and int(r.get("refinement_steps", 99)) <= 2 for r in trained),
        "reaches_1e_9_with_K_le_5": any(_as_float(r.get("future_10_period_mse")) < 1e-9 and int(r.get("refinement_steps", 99)) <= 5 for r in trained),
        "learned_refinement_improves_over_fixed_newton": _as_float(best_learned.get("future_10_period_mse")) < _as_float(best_fixed.get("future_10_period_mse")),
        "encoder_initialization_beats_average_c_same_refinement": _as_float(best.get("future_10_period_mse")) < _as_float(best_avg.get("state_given_10_period_mse", best_avg.get("future_10_period_mse"))),
        "best_config": best.get("config_name", ""),
        "best_future_10_period_mse": best.get("future_10_period_mse", ""),
        "best_c_refined_rmse": best.get("c_refined_rmse", ""),
        "best_fixed_newton_config": best_fixed.get("config_name", ""),
        "best_learned_newton_config": best_learned.get("config_name", ""),
        "method_framing": "amortized initializer plus prefix-consistency refinement layer; pure encoder prediction is the K=0 ablation.",
        "current_sho_specific_part": "scalar c_i and companion recurrence K(c)=[[0,1],[-1,c]]",
        "general_method": "replace scalar c_i with a dynamics parameter vector, define a prefix-only consistency loss, and unroll a small differentiable refinement layer before rollout.",
    }


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = [
        "stage",
        "config_name",
        "method",
        "refinement_steps",
        "gradients_through_refinement",
        "c_init_rmse",
        "c_refined_rmse",
        "future_1_period_mse",
        "future_2_period_mse",
        "future_5_period_mse",
        "future_10_period_mse",
        "runtime_per_trajectory_seconds",
    ]
    rows = outputs["summary"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Amortized Initializer Refinement Layer</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}code,pre{background:#f7f7f7;padding:2px 4px;border-radius:4px}</style>",
        "</head><body><h1>Stage 1B Amortized Initializer + Prefix-refinement Layer</h1>",
        "<p>Training uses clean position-only prefixes, prefix recurrence consistency only, and torch.float64. Future rollout is evaluation-only.</p>",
        "<h2>Method</h2>",
        "<p>Current SHO-specific diagnostic predicts scalar <code>c_i</code> for <code>K(c)=[[0,1],[-1,c]]</code>. The general method is an amortized initializer followed by a small differentiable prefix-consistency refinement layer.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 52000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 52100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 52200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)

    baselines = _previous_baselines(test)
    avg_baselines = _average_init_refinement_baselines(train, test, [1, 2, 5, 10])
    configs = _config_grid()
    results: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for config in configs:
        print(f"training {config['config_name']}", flush=True)
        _, hist, row = _train_config(train, val, test, config)
        history.extend(hist)
        results.append(row)

    summary = baselines + avg_baselines + results
    return {
        "summary": summary,
        "full_run_table": summary + teacher_history + history,
        "refinement_layer_results": results,
        "training_history": history,
        "threshold_summary": _threshold_summary(summary),
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "full_run_table.csv", outputs["full_run_table"])
    _write_csv(OUTPUT_DIR / "refinement_layer_results.csv", outputs["refinement_layer_results"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    _write_csv(OUTPUT_DIR / "threshold_summary.csv", outputs["threshold_summary"])
    interpretation = _interpret(outputs["summary"])
    setup = {
        "diagnostic": "stage1B_amortized_initializer_refinement_layer",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "beta_init": BETA_INIT,
        "fixed_newton_eta": FIXED_ETA,
        "fixed_newton_damping": FIXED_DAMPING,
        "max_steps": MAX_STEPS,
        "patience": PATIENCE,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training": False,
        "uses_future_data_in_training_or_refinement": False,
        "uses_LS_inside_learned_encoder": False,
        "uses_closed_form_c_estimator": False,
        "training_loss": "L_prefix(c_refined)+beta_init*L_prefix(c_init)",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_by_steps(outputs["refinement_layer_results"], "future_10_period_mse", "future_mse_vs_refinement_steps.png")
    _plot_by_steps(outputs["refinement_layer_results"], "c_refined_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_bar(outputs["summary"], "future_10_period_mse", "encoder_init_vs_average_init.png")
    _plot_bar([r for r in outputs["refinement_layer_results"] if r.get("method") in {"fixed_newton", "learned_newton"}], "future_10_period_mse", "learned_vs_fixed_refinement.png")
    _plot_history(outputs["training_history"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    best = min(outputs["refinement_layer_results"], key=lambda r: _as_float(r.get("future_10_period_mse")), default={})
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "best_config": best.get("config_name", ""), "best_future_10_period_mse": best.get("future_10_period_mse", "")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
