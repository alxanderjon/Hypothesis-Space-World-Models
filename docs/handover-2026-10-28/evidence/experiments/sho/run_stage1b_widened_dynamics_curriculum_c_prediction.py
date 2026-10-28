"""Stage 1B widened-dynamics curriculum for amortized c_i prediction.

The encoder is trained on self-supervised free-c teachers from increasingly
narrow c_i ranges, ending on the target Stage 1B distribution. No velocity,
omega labels, coefficient labels, true c labels in the loss, LS inside the
encoder, or future points for teacher construction are used.
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    BATCH_SIZE,
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
    _threshold_flags,
)
from experiments.sho.run_stage1b_c_precision_refinement_diagnostic import _precision_metrics
from experiments.sho.run_stage1b_c_refinement_optimizer_diagnostic import _newton_refine
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
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

OUTPUT_DIR = Path("outputs/sho/stage1B/widened_dynamics_curriculum_c_prediction")


def _float_list_env(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [float(part) for part in raw.split(",") if part.strip()]


MAX_STEPS_PER_STAGE = int(os.environ.get("WIDENED_MAX_STEPS_PER_STAGE", "30000"))
EVAL_INTERVAL = int(os.environ.get("WIDENED_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("WIDENED_PATIENCE", "5000"))
REFINE_STEPS = [0, 1, 2, 5, 10, 20, 50]
TEACHER_STEPS = 700
LR_VALUES = _float_list_env("WIDENED_LR_VALUES", [1e-3, 3e-4, 1e-4])
MU_VALUES = _float_list_env("WIDENED_MU_VALUES", [0.0, 1e-4, 1e-3, 1e-2])


@dataclass(frozen=True)
class SHOSet:
    positions: torch.Tensor
    omegas: torch.Tensor


class TCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(torch.nn.functional.silu(self.inp(x.unsqueeze(1))))
        return self.head(y.mean(dim=-1)).squeeze(-1)


def _steps(periods: float, omega_min: float = 1.0) -> int:
    return int(math.ceil(periods * math.tau / (omega_min * float(DELTA_T))))


def _omega_from_c(c: torch.Tensor) -> torch.Tensor:
    return torch.arccos((c / 2.0).clamp(-0.999999999, 0.999999999)) / float(DELTA_T)


def _make_c_range_data(n: int, seed: int, c_range: tuple[float, float] | None) -> SHOSet:
    if c_range is None:
        base = _make_data(n, seed)
        return SHOSet(base.positions, base.omegas)
    gen = torch.Generator().manual_seed(seed)
    bins = min(8, n)
    per_bin = math.ceil(n / bins)
    cs = []
    lo, hi = c_range
    edges = torch.linspace(lo, hi, bins + 1, dtype=DTYPE)
    for b in range(bins):
        u = torch.rand(per_bin, generator=gen, dtype=DTYPE)
        cs.append(edges[b] + (edges[b + 1] - edges[b]) * u)
    c = torch.cat(cs)[:n]
    perm = torch.randperm(n, generator=gen)
    c = c[perm]
    omegas = _omega_from_c(c)
    initial_conditions = 2.0 * torch.rand(n, 2, generator=gen, dtype=DTYPE) - 1.0
    total_steps = P + _steps(10.0, omega_min=0.5) + 8
    times = torch.arange(total_steps, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, omegas, times.expand(n, -1))
    return SHOSet(positions, omegas)


def _loss_parts(z_pred: torch.Tensor, z_true: torch.Tensor, variant: str) -> tuple[torch.Tensor, dict[str, float]]:
    mse = torch.mean((z_pred - z_true) ** 2)
    var = (torch.std(z_pred) - torch.std(z_true)) ** 2
    mean = (torch.mean(z_pred) - torch.mean(z_true)) ** 2
    xc, yc = z_pred - z_pred.mean(), z_true - z_true.mean()
    corr = torch.sum(xc * yc) / torch.sqrt(torch.sum(xc**2) * torch.sum(yc**2)).clamp_min(1e-30)
    corr_loss = 1.0 - corr
    pair = torch.mean(((z_pred[:, None] - z_pred[None, :]) - (z_true[:, None] - z_true[None, :])) ** 2)
    if variant == "mse":
        total = mse
    elif variant == "combined":
        total = mse + var + corr_loss + pair + 0.1 * mean
    else:
        raise ValueError(variant)
    return total, {
        "z_mse": float(mse.detach().item()),
        "var_loss": float(var.detach().item()),
        "corr": float(corr.detach().item()),
        "pair_loss": float(pair.detach().item()),
        "mean_loss": float(mean.detach().item()),
    }


def _z(dataset: TeacherSet) -> torch.Tensor:
    return (dataset.c_free - dataset.mu) / dataset.sigma.clamp_min(1e-12)


def _per_traj_prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    losses = torch.zeros(c.shape[0], dtype=DTYPE)
    for j, weight in {1: 1.0, 2: 1.0, 4: 2.0, 8: 4.0, 16: 4.0, 32: 4.0, 64: 4.0}.items():
        starts = torch.arange(1, P - j)
        a, b = x[:, starts - 1], x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        losses = losses + weight * torch.mean((pred - truth) ** 2, dim=(1, 2))
    return losses


def _free_teacher_multistart(data: SHOSet) -> torch.Tensor:
    n = data.positions.shape[0]
    best_c = torch.zeros(n, dtype=DTYPE)
    best_loss = torch.full((n,), float("inf"), dtype=DTYPE)
    for init in [-1.5, -0.5, 0.5, 1.0, 1.5, 1.9]:
        raw = nn.Parameter(torch.full((n,), math.atanh(max(min(init / 2.0, 0.999), -0.999)), dtype=DTYPE))
        opt = torch.optim.Adam([raw], lr=1e-2)
        for _ in range(TEACHER_STEPS):
            opt.zero_grad()
            c = 2.0 * torch.tanh(raw)
            loss = torch.mean(_per_traj_prefix_loss(c, data.positions))
            loss.backward()
            opt.step()
        with torch.no_grad():
            c = 2.0 * torch.tanh(raw)
            losses = _per_traj_prefix_loss(c, data.positions)
            mask = losses < best_loss
            best_loss[mask] = losses[mask]
            best_c[mask] = c[mask]
    return best_c.detach()


@torch.no_grad()
def _eval(model: nn.Module, dataset: TeacherSet, stage: str, config_name: str, extra: dict[str, Any]) -> dict[str, Any]:
    z_pred = model(dataset.data.positions[:, :P])
    c_pred = dataset.mu + dataset.sigma * z_pred
    row = _precision_metrics(stage, config_name, c_pred, dataset, extra)
    row.update(
        {
            "predicted_c_mean": float(c_pred.mean().item()),
            "predicted_c_std": float(c_pred.std().item()),
            "teacher_c_mean": float(dataset.c_free.mean().item()),
            "teacher_c_std": float(dataset.c_free.std().item()),
            "test_c_rmse": row["c_rmse"],
            "test_c_mae": row["c_mae"],
            "test_c_max_abs_error": row["c_max_abs_error"],
            "test_c_bias": row["c_bias"],
        }
    )
    return row


def _param_drift(model: nn.Module, prev_state: dict[str, torch.Tensor] | None) -> torch.Tensor:
    if prev_state is None:
        return torch.zeros((), dtype=DTYPE)
    total = torch.zeros((), dtype=DTYPE)
    for name, p in model.state_dict().items():
        total = total + torch.sum((p - prev_state[name]) ** 2)
    return total


def _train_stage(
    model: nn.Module,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    stage_name: str,
    c_range_label: str,
    loss_variant: str,
    mu: float,
    lr: float,
    prev_state: dict[str, torch.Tensor] | None,
) -> tuple[nn.Module, dict[str, torch.Tensor], list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale = 0
    hist = []
    gen = torch.Generator().manual_seed(SEED + hash((stage_name, loss_variant, mu)) % 100000)
    n = train.data.positions.shape[0]
    prev_ref = copy.deepcopy(prev_state) if prev_state is not None else None
    for step in range(1, MAX_STEPS_PER_STAGE + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        x = train.data.positions[idx, :P]
        z_true = _z(train)[idx]
        opt.zero_grad()
        loss_stage, parts = _loss_parts(model(x), z_true, loss_variant)
        drift = _param_drift(model, prev_ref)
        loss = loss_stage + mu * drift
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS_PER_STAGE:
            row = _eval(
                model,
                val,
                "curriculum_history",
                f"{stage_name}_{loss_variant}_mu_{mu:g}_lr_{lr:g}",
                {
                    "curriculum_stage": stage_name,
                    "c_i_range": c_range_label,
                    "encoder_type": "tcn",
                    "loss_variant": loss_variant,
                    "mu": mu,
                    "learning_rate": lr,
                    "step": step,
                    "train_loss": float(loss.item()),
                    "biased_regularization_penalty": float((mu * drift).detach().item()),
                    **parts,
                },
            )
            hist.append(row)
            score = float(row["c_rmse"]) + float(row["state_given_10_period_mse"])
            if score < best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    accepted = copy.deepcopy(model.state_dict())
    final = _eval(
        model,
        test,
        "curriculum_training",
        f"{stage_name}_{loss_variant}_mu_{mu:g}_lr_{lr:g}",
        {
            "curriculum_stage": stage_name,
            "c_i_range": c_range_label,
            "encoder_type": "tcn",
            "loss_variant": loss_variant,
            "mu": mu,
            "learning_rate": lr,
            "parameter_drift_between_stages": float(_param_drift(model, prev_ref).item()),
            "teacher_c_mean": float(train.c_free.mean().item()),
            "teacher_c_std": float(train.c_free.std().item()),
        },
    )
    return model, accepted, hist, final


def _make_stage_teachers(stage_name: str, c_range: tuple[float, float] | None, seed: int) -> tuple[TeacherSet, TeacherSet, TeacherSet, list[dict[str, Any]]]:
    train = _make_c_range_data(TRAIN_NUM, seed, c_range)
    val = _make_c_range_data(VAL_NUM, seed + 100, c_range)
    test = _make_c_range_data(TEST_NUM, seed + 200, c_range)
    c_train = _free_teacher_multistart(train)
    c_val = _free_teacher_multistart(val)
    c_test = _free_teacher_multistart(test)
    mu, sigma = c_train.mean(), c_train.std().clamp_min(1e-12)
    tr, va, te = TeacherSet(train, c_train, mu, sigma), TeacherSet(val, c_val, mu, sigma), TeacherSet(test, c_test, mu, sigma)
    rows = []
    for split, ds in [("train", tr), ("val", va), ("test", te)]:
        rows.append(
            _precision_metrics(
                "teacher",
                f"{stage_name}_{split}_free_c_teacher",
                ds.c_free,
                ds,
                {
                    "curriculum_stage": stage_name,
                    "split": split,
                    "teacher_c_mean": float(ds.c_free.mean().item()),
                    "teacher_c_std": float(ds.c_free.std().item()),
                },
            )
        )
    return tr, va, te, rows


def _run_curriculum(mu: float, lr: float, stages: list[tuple[str, tuple[float, float] | None, str, tuple[TeacherSet, TeacherSet, TeacherSet]]]) -> tuple[nn.Module, list[dict[str, Any]], list[dict[str, Any]]]:
    model = TCNEncoder().to(dtype=DTYPE)
    prev_state = None
    rows = []
    hist_all = []
    for name, _range, label, datasets in stages:
        print(f"  curriculum mu={mu:g} lr={lr:g} stage={name}", flush=True)
        tr, va, te = datasets
        model, prev_state, hist, final = _train_stage(model, tr, va, te, name, label, "combined", mu, lr, prev_state)
        hist_all.extend(hist)
        rows.append(final)
    return model, rows, hist_all


def _target_only(target: tuple[TeacherSet, TeacherSet, TeacherSet], variant: str, lr: float) -> tuple[nn.Module, list[dict[str, Any]], list[dict[str, Any]]]:
    model = TCNEncoder().to(dtype=DTYPE)
    tr, va, te = target
    model, _, hist, final = _train_stage(model, tr, va, te, "target_only", "target", variant, 0.0, lr, None)
    return model, [final], hist


def _refinement_rows(best_c: torch.Tensor, prev_c: torch.Tensor, target_test: TeacherSet, target_train: TeacherSet) -> list[dict[str, Any]]:
    gen = torch.Generator().manual_seed(SEED + 50505)
    avg = torch.full_like(target_test.c_free, float(target_train.c_free.mean().item()))
    rand = avg + 0.01 * torch.randn(avg.shape, generator=gen, dtype=DTYPE)
    rows = []
    for name, c0 in [
        ("widened_curriculum_encoder_init", best_c),
        ("previous_best_encoder_init", prev_c),
        ("average_c_init", avg),
        ("random_near_mean_init", rand),
    ]:
        all_rows = _newton_refine(c0.detach(), target_test, 0.5, 1e-4, "constrained", [1, 2, 4, 8, 16, 32, 64], name)
        for r in all_rows:
            if r.get("selection", "current") == "current" and int(r.get("refinement_steps", -1)) in REFINE_STEPS:
                r["stage"] = "refinement_evaluation"
                r["initialization_type"] = name
                r["refinement_step_count"] = r.get("refinement_steps")
                rows.append(r)
    return rows


def _threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in THRESHOLDS:
        eligible = [r for r in rows if str(r.get(f"crosses_{threshold:g}", "")).lower() == "true"]
        best = min(rows, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
        out.append(
            {
                "threshold": threshold,
                "num_crossing": len(eligible),
                "best_crossing_config": min(eligible, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))["config_name"] if eligible else "",
                "best_overall_config": best["config_name"],
                "best_overall_mse": best["state_given_10_period_mse"],
            }
        )
    return out


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str, label_key: str = "config_name") -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(str(r.get(label_key, r.get("config_name", ""))))
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


def _plot_refinement(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for init in sorted({r.get("initialization_type", "") for r in rows}):
        vals = [r for r in rows if r.get("initialization_type") == init]
        vals = sorted(vals, key=lambda r: int(r.get("refinement_step_count", 0)))
        if vals:
            plt.plot([int(r["refinement_step_count"]) for r in vals], [float(r["state_given_10_period_mse"]) for r in vals], marker="o", label=init)
    plt.yscale("log")
    plt.xlabel("prefix-only Newton refinement steps")
    plt.ylabel("10-period rollout MSE")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "refinement_steps_vs_mse.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "config_name", "curriculum_stage", "loss_variant", "mu", "c_rmse", "diagnostic_c_r2", "correlation", "collapse_ratio", "state_given_10_period_mse", "refinement_step_count"]
    body = []
    for r in outputs["summary"]:
        body.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Widened Dynamics Curriculum c Prediction</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B Widened-Dynamics Curriculum for c_i Prediction</h1>",
        "<p>Teachers are self-supervised free-c prefix optimizations. True c is used only to generate diagnostic ranges and for attribution.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    stage_specs = [
        ("A_very_wide", (0.5, 1.95), "c in [0.5,1.95]", TRAIN_DATA_SEED + 30000),
        ("B_wide", (1.0, 1.95), "c in [1.0,1.95]", TRAIN_DATA_SEED + 31000),
        ("C_medium", (1.4, 1.95), "c in [1.4,1.95]", TRAIN_DATA_SEED + 32000),
        ("D_target", None, "target", TRAIN_DATA_SEED + 33000),
    ]
    stage_data = []
    teacher_rows = []
    for name, crange, label, seed in stage_specs:
        print(f"building teachers for {name}", flush=True)
        tr, va, te, rows = _make_stage_teachers(name, crange, seed)
        teacher_rows.extend(rows)
        stage_data.append((name, crange, label, (tr, va, te)))

    curriculum_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    models: list[tuple[float, float, nn.Module, dict[str, Any]]] = []
    for lr in LR_VALUES:
        for mu in MU_VALUES:
            model, rows, hist = _run_curriculum(mu, lr, stage_data)
            curriculum_rows.extend(rows)
            history_rows.extend(hist)
            models.append((mu, lr, model, rows[-1]))

    print("running target-only comparisons", flush=True)
    target = stage_data[-1][3]
    target_only_rows = []
    target_only_hist = []
    target_only_models: dict[tuple[str, float], nn.Module] = {}
    for lr in LR_VALUES:
        for variant in ["mse", "combined"]:
            target_model, rows, hist = _target_only(target, variant, lr)
            target_only_models[(variant, lr)] = target_model
            target_only_rows.extend(rows)
            target_only_hist.extend(hist)

    target_train, _, target_test = target
    best_mu, best_lr, best_model, best_row = min(models, key=lambda item: float(item[3]["state_given_10_period_mse"]))
    best_c = target_test.mu + target_test.sigma * best_model(target_test.data.positions[:, :P]).detach()
    prev_c = best_c.clone()
    # Use target-only combined as a previous-encoder proxy if available.
    combined_models = [(lr, model) for (variant, lr), model in target_only_models.items() if variant == "combined"]
    if combined_models:
        _, prev_model = min(
            combined_models,
            key=lambda item: float(
                next(
                    r["state_given_10_period_mse"]
                    for r in target_only_rows
                    if r["loss_variant"] == "combined" and float(r["learning_rate"]) == item[0]
                )
            ),
        )
        prev_c = target_test.mu + target_test.sigma * prev_model(target_test.data.positions[:, :P]).detach()
    print("running prefix-only refinement evaluation", flush=True)
    refinement_rows = _refinement_rows(best_c, prev_c, target_test, target_train)

    baseline_rows = [
        {"stage": "baseline", "config_name": "previous_tcn_combined", "diagnostic_c_r2": 0.9997, "state_given_10_period_mse": 0.0214},
        {"stage": "baseline", "config_name": "previous_finetuned_tcn", "state_given_10_period_mse": 0.00275},
        {"stage": "baseline", "config_name": "previous_residual_correction", "state_given_10_period_mse": 0.00378},
        {"stage": "baseline", "config_name": "previous_best_encoder_refinement", "state_given_10_period_mse": 2.0e-29},
    ] + _baselines(target_test)

    summary = curriculum_rows + target_only_rows + refinement_rows + baseline_rows
    return {
        "summary": summary,
        "full_run_table": summary + teacher_rows + history_rows + target_only_hist,
        "teacher_results_by_stage": teacher_rows,
        "curriculum_training_results": curriculum_rows,
        "target_only_comparison": target_only_rows,
        "refinement_step_results": refinement_rows,
        "collapse_diagnostics": [r for r in summary if "collapse_ratio" in r],
        "threshold_crossings": _threshold_summary(summary),
        "baseline_comparison": baseline_rows,
    }


def _interpret(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    curr = outputs["curriculum_training_results"]
    target_only = outputs["target_only_comparison"]
    refine = outputs["refinement_step_results"]
    best_curr = min(curr, key=lambda r: float(r["state_given_10_period_mse"]))
    best_target = min(target_only, key=lambda r: float(r["state_given_10_period_mse"])) if target_only else {}
    best_refine = min(refine, key=lambda r: float(r["state_given_10_period_mse"])) if refine else {}
    le2 = [r for r in refine if int(r.get("refinement_step_count", 999)) <= 2]
    best_le2 = min(le2, key=lambda r: float(r["state_given_10_period_mse"])) if le2 else {}
    return {
        "prevents_average_c_collapse": float(best_curr.get("collapse_ratio", 0.0)) > 0.7,
        "best_curriculum_config": best_curr["config_name"],
        "best_curriculum_10_period_mse": best_curr["state_given_10_period_mse"],
        "best_curriculum_c_rmse": best_curr["c_rmse"],
        "best_target_only_config": best_target.get("config_name", ""),
        "best_target_only_10_period_mse": best_target.get("state_given_10_period_mse", ""),
        "curriculum_beats_target_only": float(best_curr["state_given_10_period_mse"]) < float(best_target.get("state_given_10_period_mse", float("inf"))),
        "pure_amortized_reaches_1e_6": float(best_curr["state_given_10_period_mse"]) < 1e-6,
        "best_refinement_config": best_refine.get("config_name", ""),
        "best_refinement_10_period_mse": best_refine.get("state_given_10_period_mse", ""),
        "best_le2_refinement_config": best_le2.get("config_name", ""),
        "best_le2_refinement_mse": best_le2.get("state_given_10_period_mse", ""),
        "hybrid_le2_reaches_1e_6": float(best_le2.get("state_given_10_period_mse", float("inf"))) < 1e-6,
        "preferred_next_method": "widened_curriculum_amortized_initialization_plus_prefix_only_refinement",
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("teacher_results_by_stage", "teacher_results_by_stage.csv"),
        ("curriculum_training_results", "curriculum_training_results.csv"),
        ("target_only_comparison", "target_only_comparison.csv"),
        ("refinement_step_results", "refinement_step_results.csv"),
        ("collapse_diagnostics", "collapse_diagnostics.csv"),
        ("threshold_crossings", "threshold_crossings.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs)
    setup = {
        "diagnostic": "stage1B_widened_dynamics_curriculum_c_prediction",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training_loss": False,
        "teacher": "self_supervised_free_c_from_prefix",
        "max_steps_per_stage": MAX_STEPS_PER_STAGE,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "learning_rates": LR_VALUES,
        "mu_values": MU_VALUES,
        "stage_c_ranges": {
            "A_very_wide": [0.5, 1.95],
            "B_wide": [1.0, 1.95],
            "C_medium": [1.4, 1.95],
            "D_target": "target/practical distribution",
        },
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _plot_bar(outputs["teacher_results_by_stage"], "teacher_c_std", "c_distribution_by_curriculum_stage.png", "config_name")
    _plot_bar(outputs["curriculum_training_results"], "diagnostic_c_r2", "c_r2_by_stage.png")
    _plot_bar(outputs["curriculum_training_results"], "collapse_ratio", "collapse_ratio_by_stage.png")
    _plot_bar(outputs["curriculum_training_results"], "c_rmse", "c_rmse_by_stage.png")
    _plot_bar(outputs["curriculum_training_results"], "state_given_10_period_mse", "rollout_mse_by_stage.png")
    _plot_bar(outputs["target_only_comparison"] + outputs["curriculum_training_results"], "state_given_10_period_mse", "target_curriculum_vs_target_only.png")
    _plot_bar(outputs["curriculum_training_results"], "c_rmse", "predicted_vs_teacher_c_target.png")
    _plot_refinement(outputs["refinement_step_results"])
    _plot_bar(outputs["threshold_crossings"], "num_crossing", "threshold_crossings_by_refinement_steps.png")
    _plot_bar(outputs["baseline_comparison"], "state_given_10_period_mse", "baseline_comparison.png")
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
