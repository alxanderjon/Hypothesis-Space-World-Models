"""Stage 1B covariance-first, horizon-SuperLoss curriculum diagnostic.

Tests whether a prefix encoder can learn transition structure first, then use
SuperLoss across rollout horizons, then sharpen c_i prediction with a
prefix-derived free-c teacher.

No future samples are used in training losses. Full 10-period rollout is used
only for validation checkpointing and final evaluation.
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    P,
    TEST_NUM,
    TRAIN_NUM,
    VAL_NUM,
    TeacherSet,
    _make_data,
    _make_teacher_sets,
    _r2,
)
from experiments.sho.run_stage1b_covariance_superloss_encoder_diagnostic import (
    _covariance_loss_vector,
    _superloss_weights,
)
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import (
    CInitTCN,
    _metrics,
    _param_count,
    _predict_c,
    _refine_c,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE
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

OUTPUT_DIR = Path("outputs/sho/stage1B/covariance_first_horizon_superloss_curriculum")
BATCH_SIZE = int(os.environ.get("CFHS_BATCH_SIZE", "32"))
EVAL_INTERVAL = int(os.environ.get("CFHS_EVAL_INTERVAL", "1000"))
ALPHA_COV = float(os.environ.get("CFHS_ALPHA_COV", "0.1"))
SUPERLOSS_LAMBDA = float(os.environ.get("CFHS_SUPERLOSS_LAMBDA", "0.1"))
TAU_EMA = float(os.environ.get("CFHS_TAU_EMA", "0.98"))
SMOOTHMAX_TEMP = float(os.environ.get("CFHS_SMOOTHMAX_TEMP", "20"))
STAGE1_STEPS = int(os.environ.get("CFHS_STAGE1_STEPS", "30000"))
STAGE2_STEPS = int(os.environ.get("CFHS_STAGE2_STEPS", "30000"))
STAGE3_STEPS = int(os.environ.get("CFHS_STAGE3_STEPS", "50000"))
REFINE_STEPS = [0, 20, 50, 100]
REFINE_LR = 1e-3
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
LAMBDA_C_VALUES = [1.0, 10.0]
BETA_COV_VALUES = [0.0, 0.01]


def _steps(periods: float) -> int:
    return int(math.ceil(periods * math.tau / (float(OMEGA_RANGE[0]) * float(DELTA_T))))


def _batch_dataset(dataset: TeacherSet, idx: torch.Tensor) -> TeacherSet:
    data = type(dataset.data)(positions=dataset.data.positions[idx], omegas=dataset.data.omegas[idx])
    return TeacherSet(data, dataset.c_free[idx], dataset.mu, dataset.sigma)


def _smoothmax(values: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(SMOOTHMAX_TEMP * values, dim=0) / SMOOTHMAX_TEMP


def _prefix_horizon_loss_vector(c: torch.Tensor, positions: torch.Tensor, horizon: int) -> torch.Tensor:
    x = positions[:, :P]
    starts = torch.arange(1, P - horizon, device=x.device)
    a = x[:, starts - 1]
    b = x[:, starts]
    preds = []
    for _ in range(horizon):
        nxt = c[:, None] * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = torch.stack([x[:, starts + k] for k in range(1, horizon + 1)], dim=-1)
    return torch.mean((pred - truth) ** 2, dim=(1, 2))


def _horizon_smooth_losses(c: torch.Tensor, positions: torch.Tensor) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    per_traj = {h: _prefix_horizon_loss_vector(c, positions, h) for h in HORIZONS}
    smooth = {h: _smoothmax(v) for h, v in per_traj.items()}
    return smooth, per_traj


def _future_rollout_error_vector(c: torch.Tensor, dataset: TeacherSet, periods: float = 10.0) -> torch.Tensor:
    steps = _steps(periods)
    data = dataset.data
    a = data.positions[:, P - 2]
    b = data.positions[:, P - 1]
    preds = []
    for _ in range(steps):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = data.positions[:, P : P + steps]
    return torch.mean((pred - truth) ** 2, dim=1)


def _future_abs_error_stats(c: torch.Tensor, dataset: TeacherSet, periods: float = 10.0) -> tuple[float, float]:
    steps = _steps(periods)
    data = dataset.data
    a = data.positions[:, P - 2]
    b = data.positions[:, P - 1]
    preds = []
    for _ in range(steps):
        nxt = c * b - a
        preds.append(nxt)
        a, b = b, nxt
    pred = torch.stack(preds, dim=-1)
    truth = data.positions[:, P : P + steps]
    err = torch.abs(pred - truth)
    return float(err.max().item()), float(torch.mean(err[:, -1]).item())


def _eval_row(model: CInitTCN, dataset: TeacherSet, run_name: str, stage_name: str, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        future = _future_rollout_error_vector(c, dataset, 10.0)
        cov = _covariance_loss_vector(c, dataset.data.positions)
        horizon_smooth, horizon_vec = _horizon_smooth_losses(c, dataset.data.positions)
        max_abs_rollout, final_abs = _future_abs_error_stats(c, dataset)
        c_err = c - dataset.c_free
    row: dict[str, Any] = {
        "stage": "stage_eval",
        "config_name": run_name,
        "run_name": run_name,
        "stage_name": stage_name,
        "split": split,
        "c_rmse": float(torch.sqrt(torch.mean(c_err**2)).item()),
        "c_r2_against_teacher": _r2(c[:, None], dataset.c_free[:, None]),
        "collapse_ratio": float((torch.std(c) / torch.std(dataset.c_free).clamp_min(1e-12)).item()),
        "max_abs_c_error": float(torch.max(torch.abs(c_err)).item()),
        "c_bias": float(torch.mean(c_err).item()),
        "covariance_residual": float(cov.mean().item()),
        "covariance_residual_max": float(cov.max().item()),
        "future_10_period_mse_mean": float(future.mean().item()),
        "future_10_period_mse_median": float(future.median().item()),
        "future_10_period_mse_max": float(future.max().item()),
        "future_10_period_mse": float(future.mean().item()),
        "max_trajectory_error": max_abs_rollout,
        "final_time_absolute_error": final_abs,
        "parameter_count": _param_count(model),
    }
    for h in HORIZONS:
        row[f"horizon_{h}_smooth_loss"] = float(horizon_smooth[h].item())
        row[f"horizon_{h}_max_loss"] = float(horizon_vec[h].max().item())
        row[f"horizon_{h}_mean_loss"] = float(horizon_vec[h].mean().item())
    if extra:
        row.update(extra)
    return row


def _stage_loss(
    model: CInitTCN,
    dataset: TeacherSet,
    stage_kind: str,
    tau: dict[int, torch.Tensor],
    lambda_c: float = 1.0,
    beta_cov: float = 0.0,
    use_covariance: bool = True,
) -> tuple[torch.Tensor, dict[str, Any], dict[int, torch.Tensor]]:
    c = _predict_c(model, dataset)
    horizon_smooth, horizon_vec = _horizon_smooth_losses(c, dataset.data.positions)
    cov_vec = _covariance_loss_vector(c, dataset.data.positions)
    cov_max = _smoothmax(cov_vec)
    weights: dict[int, torch.Tensor] = {}
    if stage_kind == "transition":
        loss = horizon_smooth[1] + (ALPHA_COV * cov_max if use_covariance else 0.0)
    elif stage_kind == "horizon_superloss":
        terms = []
        for h in HORIZONS:
            lh = horizon_smooth[h]
            tau[h] = TAU_EMA * tau[h] + (1.0 - TAU_EMA) * lh.detach()
            w = _superloss_weights(lh.reshape(1), tau[h].reshape(()), SUPERLOSS_LAMBDA)[0]
            weights[h] = w
            terms.append(w * lh)
        loss = sum(terms) + (ALPHA_COV * cov_max if use_covariance else 0.0)
    elif stage_kind == "precision":
        c_err2 = (c - dataset.c_free) ** 2
        c_max = _smoothmax(c_err2)
        loss = lambda_c * c_max + beta_cov * cov_max
    else:
        raise ValueError(stage_kind)
    stats: dict[str, Any] = {
        "train_loss": float(loss.detach().item()),
        "train_covariance_smoothmax": float(cov_max.detach().item()),
        "stage_kind": stage_kind,
    }
    for h in HORIZONS:
        stats[f"train_horizon_{h}_smooth_loss"] = float(horizon_smooth[h].detach().item())
        stats[f"train_horizon_{h}_max_loss"] = float(horizon_vec[h].max().detach().item())
        stats[f"superloss_weight_horizon_{h}"] = float(weights[h].detach().item()) if h in weights else 1.0
        stats[f"superloss_tau_horizon_{h}"] = float(tau[h].detach().item())
    return loss, stats, tau


def _train_stage(
    model: CInitTCN,
    train: TeacherSet,
    val: TeacherSet,
    run_name: str,
    stage_name: str,
    stage_kind: str,
    steps: int,
    lr: float,
    seed_offset: int,
    lambda_c: float = 1.0,
    beta_cov: float = 0.0,
    use_covariance: bool = True,
) -> tuple[CInitTCN, list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 221000 + seed_offset)
    n = train.data.positions.shape[0]
    tau = {h: torch.tensor(1.0, dtype=DTYPE) for h in HORIZONS}
    start = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        batch = _batch_dataset(train, idx)
        loss, stats, tau = _stage_loss(model, batch, stage_kind, tau, lambda_c, beta_cov, use_covariance)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == steps:
            row = _eval_row(
                model,
                val,
                run_name,
                stage_name,
                "val",
                {
                    "step": step,
                    "stage_elapsed_seconds": time.perf_counter() - start,
                    "learning_rate": lr,
                    "lambda_c": lambda_c,
                    "beta_cov": beta_cov,
                    "use_covariance": use_covariance,
                    **stats,
                },
            )
            history.append(row)
            score = float(row["future_10_period_mse_max"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    best_row = _eval_row(
        model,
        val,
        run_name,
        stage_name,
        "val",
        {
            "best_validation_worst_10period_mse": best,
            "learning_rate": lr,
            "lambda_c": lambda_c,
            "beta_cov": beta_cov,
            "use_covariance": use_covariance,
            "stage_kind": stage_kind,
        },
    )
    return model, history, best_row


def _pretrain_base(run_name: str, train: TeacherSet, val: TeacherSet, index: int) -> tuple[CInitTCN, list[dict[str, Any]], list[dict[str, Any]]]:
    model = CInitTCN().to(dtype=DTYPE)
    history: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    if run_name == "cov_horizon_superloss_precision":
        plan = [
            ("stage1_transition_cov", "transition", STAGE1_STEPS, 1e-4, True),
            ("stage2_horizon_superloss_cov", "horizon_superloss", STAGE2_STEPS, 1e-4, True),
        ]
    elif run_name == "horizon_superloss_precision_no_cov":
        plan = [
            ("stage1_transition_no_cov", "transition", STAGE1_STEPS, 1e-4, False),
            ("stage2_horizon_superloss_no_cov", "horizon_superloss", STAGE2_STEPS, 1e-4, False),
        ]
    elif run_name == "cov_precision_no_superloss":
        plan = [("stage1_transition_cov", "transition", STAGE1_STEPS, 1e-4, True)]
    else:
        raise ValueError(run_name)
    for j, (stage_name, kind, steps, lr, use_cov) in enumerate(plan):
        print(f"  {run_name} / {stage_name}", flush=True)
        model, hist, row = _train_stage(model, train, val, run_name, stage_name, kind, steps, lr, index * 10000 + j * 1000, use_covariance=use_cov)
        history.extend(hist)
        stages.append(row)
    return model, history, stages


def _final_precision_variants(
    base_model: CInitTCN,
    run_name: str,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    for lambda_c in LAMBDA_C_VALUES:
        for beta_cov in BETA_COV_VALUES:
            final_name = f"{run_name}_lambda{lambda_c:g}_beta{beta_cov:g}"
            model = copy.deepcopy(base_model)
            print(f"  {final_name} / stage3_worst_case_precision", flush=True)
            model, hist, row = _train_stage(
                model,
                train,
                val,
                final_name,
                "stage3_worst_case_precision",
                "precision",
                STAGE3_STEPS,
                1e-5,
                index * 10000 + int(lambda_c * 100) + int(beta_cov * 100000),
                lambda_c=lambda_c,
                beta_cov=beta_cov,
                use_covariance=beta_cov > 0,
            )
            history.extend(hist)
            stages.append(row)
            finals.append(_eval_row(model, test, final_name, "final", "test", {"stage": "encoder_final", "lambda_c": lambda_c, "beta_cov": beta_cov}))
            refinement.extend(_refinement_rows(model, test, final_name))
    return history, stages, finals, refinement


def _refinement_rows(model: CInitTCN, dataset: TeacherSet, run_name: str) -> list[dict[str, Any]]:
    with torch.no_grad():
        init = _predict_c(model, dataset)
    rows = []
    for steps in REFINE_STEPS:
        _current, best, runtime = _refine_c(init, dataset, "adam", REFINE_LR, steps)
        row = _metrics(
            "secondary_adam_refinement",
            f"{run_name}_adam_refine_steps_{steps}",
            best,
            dataset,
            {
                "run_name": run_name,
                "refinement_steps": steps,
                "optimizer": "adam",
                "learning_rate": REFINE_LR,
                "runtime_seconds": runtime,
                "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0]),
            },
        )
        with torch.no_grad():
            future = _future_rollout_error_vector(best, dataset, 10.0)
            row["future_10_period_mse_mean"] = float(future.mean().item())
            row["future_10_period_mse_median"] = float(future.median().item())
            row["future_10_period_mse_max"] = float(future.max().item())
            row["max_abs_c_error"] = float(torch.max(torch.abs(best - dataset.c_free)).item())
            row["covariance_residual"] = float(_covariance_loss_vector(best, dataset.data.positions).mean().item())
        rows.append(row)
    return rows


def _threshold_steps(rows: list[dict[str, Any]], run_name: str, threshold: float) -> int | None:
    vals = [
        int(r["refinement_steps"])
        for r in rows
        if r.get("run_name") == run_name and float(r["future_10_period_mse_mean"]) < threshold
    ]
    return min(vals) if vals else None


def _interpret(finals: list[dict[str, Any]], refinement: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(finals, key=lambda r: float(r["future_10_period_mse_max"]))
    cov_super = min([r for r in finals if str(r["run_name"]).startswith("cov_horizon_superloss")], key=lambda r: float(r["future_10_period_mse_max"]))
    no_cov_super = min([r for r in finals if str(r["run_name"]).startswith("horizon_superloss")], key=lambda r: float(r["future_10_period_mse_max"]))
    cov_no_super = min([r for r in finals if str(r["run_name"]).startswith("cov_precision_no_superloss")], key=lambda r: float(r["future_10_period_mse_max"]))
    weight_start = [r for r in history if r.get("stage_kind") == "horizon_superloss" and int(r.get("step", 0)) <= EVAL_INTERVAL]
    weight_end = [r for r in history if r.get("stage_kind") == "horizon_superloss" and int(r.get("step", 0)) >= STAGE2_STEPS]
    return {
        "best_encoder_only_run": best["run_name"],
        "best_encoder_only_worst_10period_mse": best["future_10_period_mse_max"],
        "best_encoder_only_mean_10period_mse": best["future_10_period_mse_mean"],
        "best_encoder_only_c_rmse": best["c_rmse"],
        "covariance_plus_superloss_outperforms_no_cov_superloss": float(cov_super["future_10_period_mse_max"]) < float(no_cov_super["future_10_period_mse_max"]),
        "covariance_plus_superloss_outperforms_cov_no_superloss": float(cov_super["future_10_period_mse_max"]) < float(cov_no_super["future_10_period_mse_max"]),
        "min_refinement_steps_to_1e_6": {r["run_name"]: _threshold_steps(refinement, str(r["run_name"]), 1e-6) for r in finals},
        "min_refinement_steps_to_1e_9": {r["run_name"]: _threshold_steps(refinement, str(r["run_name"]), 1e-9) for r in finals},
        "min_refinement_steps_to_1e_12": {r["run_name"]: _threshold_steps(refinement, str(r["run_name"]), 1e-12) for r in finals},
        "horizon_superloss_recorded": bool(weight_start and weight_end),
        "future_data_training_note": "prefix horizons only are used in losses; full 10-period rollout is validation/evaluation only to respect no-future-training constraint",
        "precision_loss_note": "stage 3 uses positive lambda_c*smoothmax(c error^2)+beta_cov*cov; literal minus signs would be anti-learning",
    }


def _plot_horizon_weights(history: list[dict[str, Any]]) -> None:
    rows = [r for r in history if r.get("stage_kind") == "horizon_superloss"]
    plt.figure(figsize=(9, 5))
    for h in HORIZONS:
        vals = sorted(rows, key=lambda r: (str(r["run_name"]), int(r["step"])))
        x = list(range(len(vals)))
        plt.plot(x, [float(r[f"superloss_weight_horizon_{h}"]) for r in vals], label=f"H={h}", linewidth=1)
    plt.xlabel("horizon-SuperLoss eval index")
    plt.ylabel("SuperLoss weight")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "superloss_horizon_weights_vs_training.png", dpi=170)
    plt.close()


def _plot_stage_metric(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(10, 5))
    for name in sorted({str(r["run_name"]) for r in rows}):
        vals = [r for r in rows if str(r["run_name"]) == name]
        plt.plot(range(len(vals)), [float(r[key]) for r in vals], marker="o", label=name, linewidth=1)
    plt.yscale("log")
    plt.xlabel("stage index")
    plt.ylabel(key)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], xkey: str, ykey: str, filename: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows], s=24)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_refinement(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for name in sorted({str(r["run_name"]) for r in rows}):
        vals = sorted([r for r in rows if str(r["run_name"]) == name], key=lambda r: int(r["refinement_steps"]))
        plt.plot([int(r["refinement_steps"]) for r in vals], [float(r["future_10_period_mse_mean"]) for r in vals], marker="o", label=name, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Adam refinement steps")
    plt.ylabel("mean 10-period rollout MSE")
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "refinement_steps_vs_rollout_mse.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["run_name", "c_rmse", "c_r2_against_teacher", "collapse_ratio", "max_abs_c_error", "covariance_residual", "future_10_period_mse_max", "future_10_period_mse_mean"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Covariance-First Horizon-SuperLoss Curriculum</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Covariance-First, Horizon-SuperLoss Curriculum</h1>",
        "<p>TCN prefix encoder predicts c_i from clean position-only prefixes. Training losses use prefix data only; full 10-period rollout is evaluation/checkpointing.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Encoder-Only Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 191000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 191100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 191200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, _teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    history: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    run_names = [
        "cov_horizon_superloss_precision",
        "horizon_superloss_precision_no_cov",
        "cov_precision_no_superloss",
    ]
    for i, run_name in enumerate(run_names):
        print(f"training {run_name}", flush=True)
        base_model, hist, stages = _pretrain_base(run_name, train, val, i)
        history.extend(hist)
        stage_rows.extend(stages)
        hist, stages, final_rows, refine_rows = _final_precision_variants(base_model, run_name, train, val, test, i)
        history.extend(hist)
        stage_rows.extend(stages)
        finals.extend(final_rows)
        refinement.extend(refine_rows)
    return {
        "summary": finals,
        "stage_history": stage_rows,
        "refinement_results": refinement,
        "training_history": history,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "stage_history.csv", outputs["stage_history"])
    _write_csv(OUTPUT_DIR / "refinement_results.csv", outputs["refinement_results"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    interpretation = _interpret(outputs["summary"], outputs["refinement_results"], outputs["training_history"])
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_covariance_first_horizon_superloss_curriculum",
            "dtype": str(DTYPE),
            "prefix_length_P": P,
            "operator": "K(c)=[[0,1],[-1,c]]",
            "prefix_horizons": HORIZONS,
            "smoothmax_temperature": SMOOTHMAX_TEMP,
            "alpha_cov": ALPHA_COV,
            "superloss_lambda": SUPERLOSS_LAMBDA,
            "lambda_c_values": LAMBDA_C_VALUES,
            "beta_cov_values": BETA_COV_VALUES,
            "stage1_steps": STAGE1_STEPS,
            "stage2_steps": STAGE2_STEPS,
            "stage3_steps": STAGE3_STEPS,
            "uses_velocity": False,
            "uses_labels": False,
            "uses_LS_or_closed_form_estimator": False,
            "uses_future_data_in_training": False,
        },
    )
    _plot_horizon_weights(outputs["training_history"])
    _plot_stage_metric(outputs["stage_history"], "future_10_period_mse_max", "worst_trajectory_10period_mse_vs_stage.png")
    _plot_stage_metric(outputs["stage_history"], "max_abs_c_error", "maximum_c_error_vs_stage.png")
    _plot_scatter(outputs["summary"], "covariance_residual", "future_10_period_mse_max", "covariance_residual_vs_worst_rollout_loss.png")
    _plot_refinement(outputs["refinement_results"])
    _write_html(outputs, interpretation)
    return interpretation


def main() -> None:
    outputs = _run_all()
    interpretation = _write_outputs(outputs)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
