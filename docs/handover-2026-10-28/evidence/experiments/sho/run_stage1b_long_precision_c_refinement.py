"""Stage 1B long-precision c-only hybrid refinement diagnostic.

Tests how close the simplified hybrid method can get to exact recovery:

    prefix -> TCN encoder -> c_init
    c_init -> prefix-only Adam refinement of c
    c_refined -> companion recurrence rollout.

No future data, labels, closed-form solve, LS, Hessian, or Newton is used in
training/refinement.
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
    DRIFT_HEAVY,
    J_ALL,
    P,
    TEST_NUM,
    TRAIN_NUM,
    VAL_NUM,
    TeacherSet,
    _future_rollout_mse,
    _make_data,
    _make_teacher_sets,
    _steps,
)
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import (
    CInitTCN,
    _metrics,
    _param_count,
    _predict_c,
    _prefix_loss_vector,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/long_precision_c_refinement")
BATCH_SIZE = int(os.environ.get("LPC_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("LPC_MAX_STEPS", "500000"))
EVAL_INTERVAL = int(os.environ.get("LPC_EVAL_INTERVAL", "500"))
PATIENCE = int(os.environ.get("LPC_PATIENCE", "200000"))
ENCODER_LR = float(os.environ.get("LPC_ENCODER_LR", "1e-4"))
CHECKPOINT_STEPS = [int(x) for x in os.environ.get("LPC_CHECKPOINT_STEPS", "10000,25000,50000,100000,200000,500000,1000000").split(",")]
REFINE_LR = float(os.environ.get("LPC_REFINE_LR", "1e-3"))
REFINE_STEPS = [int(x) for x in os.environ.get("LPC_REFINE_STEPS", "0,50,100,200,500,1000,2000,5000,10000,20000").split(",")]
THRESHOLDS = [1e-12, 1e-14, 1e-15, 1e-16]
SCHEDULE = [(1e-3, 500), (1e-4, 5000), (1e-5, 20000)]


def _per_traj_future_metrics(c: torch.Tensor, dataset: TeacherSet, periods: float = 10.0) -> dict[str, torch.Tensor]:
    steps = _steps(periods)
    # Use project utility for aggregate consistency; local sequence below gives
    # per-trajectory audit values.
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
    return {
        "trajectory_mse": torch.mean(err**2, dim=1),
        "one_step_mse": err[:, 0] ** 2,
        "two_step_mse": err[:, 1] ** 2,
        "max_abs_error": torch.max(torch.abs(err), dim=1).values,
        "final_abs_error": torch.abs(err[:, -1]),
    }


def _aggregate_refinement_row(stage: str, name: str, c: torch.Tensor, dataset: TeacherSet, extra: dict[str, Any]) -> dict[str, Any]:
    row = _metrics(stage, name, c, dataset, extra)
    per = _per_traj_future_metrics(c, dataset)
    row.update(
        {
            "one_step_mse": float(torch.mean(per["one_step_mse"]).item()),
            "two_step_mse": float(torch.mean(per["two_step_mse"]).item()),
            "prefix_loss": float(torch.mean(_prefix_loss_vector(c, dataset.data.positions)).item()),
        }
    )
    for threshold in THRESHOLDS:
        row[f"percent_trajectories_below_{threshold:g}"] = float((per["trajectory_mse"] < threshold).to(DTYPE).mean().item() * 100.0)
        row[f"all_trajectories_below_{threshold:g}"] = bool(torch.all(per["trajectory_mse"] < threshold).item())
    return row


def _per_trajectory_rows(stage: str, checkpoint: str, c: torch.Tensor, dataset: TeacherSet, step: int) -> list[dict[str, Any]]:
    per = _per_traj_future_metrics(c, dataset)
    c_err = c - dataset.c_free
    prefix = _prefix_loss_vector(c, dataset.data.positions)
    rows = []
    for i in range(c.numel()):
        row = {
            "stage": stage,
            "checkpoint": checkpoint,
            "trajectory_index": i,
            "refinement_step": step,
            "c_pred": float(c[i].item()),
            "c_teacher": float(dataset.c_free[i].item()),
            "c_abs_error": float(torch.abs(c_err[i]).item()),
            "prefix_loss": float(prefix[i].item()),
            "trajectory_mse_10_period": float(per["trajectory_mse"][i].item()),
            "one_step_mse": float(per["one_step_mse"][i].item()),
            "two_step_mse": float(per["two_step_mse"][i].item()),
            "max_abs_error": float(per["max_abs_error"][i].item()),
            "final_abs_error": float(per["final_abs_error"][i].item()),
        }
        for threshold in THRESHOLDS:
            row[f"below_{threshold:g}"] = bool(per["trajectory_mse"][i] < threshold)
        rows.append(row)
    return rows


def _train_encoder(train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[CInitTCN, list[dict[str, Any]], list[dict[str, Any]]]:
    model = CInitTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=ENCODER_LR)
    best_rmse_state = copy.deepcopy(model.state_dict())
    best_rollout_state = copy.deepcopy(model.state_dict())
    best_rmse = float("inf")
    best_rollout = float("inf")
    best_score = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 131000)
    n = train.data.positions.shape[0]
    checkpoints = sorted({s for s in CHECKPOINT_STEPS if s <= MAX_STEPS} | {1, MAX_STEPS})
    next_checkpoint = 0
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        z_true = train.z[idx]
        c_true = train.c_free[idx]
        z_pred = model(pos[:, :P])
        c_pred = train.mu + train.sigma * z_pred
        loss = torch.mean((z_pred - z_true) ** 2) + torch.mean((c_pred - c_true) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        should_eval = step % EVAL_INTERVAL == 0 or step == checkpoints[next_checkpoint]
        if should_eval:
            with torch.no_grad():
                train_c = _predict_c(model, train)
                val_c = _predict_c(model, val)
                test_c = _predict_c(model, test)
                train_row = _aggregate_refinement_row("encoder_training", "c_init_encoder", train_c, train, {"split": "train", "step": step, "encoder_lr": ENCODER_LR, "elapsed_seconds": time.perf_counter() - start})
                val_row = _aggregate_refinement_row("encoder_training", "c_init_encoder", val_c, val, {"split": "val", "step": step, "encoder_lr": ENCODER_LR, "elapsed_seconds": time.perf_counter() - start})
                test_row = _aggregate_refinement_row("encoder_training", "c_init_encoder", test_c, test, {"split": "test", "step": step, "encoder_lr": ENCODER_LR, "elapsed_seconds": time.perf_counter() - start})
            history.extend([train_row, val_row, test_row])
            val_rmse = float(val_row["c_rmse"])
            val_rollout = float(val_row["future_10_period_mse"])
            improved = False
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_rmse_state = copy.deepcopy(model.state_dict())
                improved = True
            if val_rollout < best_rollout:
                best_rollout = val_rollout
                best_rollout_state = copy.deepcopy(model.state_dict())
                improved = True
            score = min(val_rmse / max(best_rmse, 1e-30), val_rollout / max(best_rollout, 1e-30))
            if improved or score < best_score:
                best_score = min(best_score, score)
                stale = 0
            else:
                stale += EVAL_INTERVAL
            while next_checkpoint + 1 < len(checkpoints) and step >= checkpoints[next_checkpoint]:
                next_checkpoint += 1
            if stale >= PATIENCE and step >= min(50000, MAX_STEPS):
                break
    model.load_state_dict(best_rmse_state)
    finals = []
    with torch.no_grad():
        for split, dataset in [("train", train), ("val", val), ("test", test)]:
            finals.append(
                _aggregate_refinement_row(
                    "encoder_final_best_val_c_rmse",
                    "c_init_encoder",
                    _predict_c(model, dataset),
                    dataset,
                    {"split": split, "encoder_lr": ENCODER_LR, "best_validation_c_rmse": best_rmse, "best_validation_10_period_mse": best_rollout, "parameter_count": _param_count(model)},
                )
            )
    rollout_model = copy.deepcopy(model)
    rollout_model.load_state_dict(best_rollout_state)
    with torch.no_grad():
        for split, dataset in [("train", train), ("val", val), ("test", test)]:
            finals.append(
                _aggregate_refinement_row(
                    "encoder_final_best_val_rollout",
                    "c_init_encoder",
                    _predict_c(rollout_model, dataset),
                    dataset,
                    {"split": split, "encoder_lr": ENCODER_LR, "best_validation_c_rmse": best_rmse, "best_validation_10_period_mse": best_rollout, "parameter_count": _param_count(rollout_model)},
                )
            )
    return model, history, finals


def _adam_refinement_curve(c_init: torch.Tensor, dataset: TeacherSet, checkpoints: list[int], lr: float, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], torch.Tensor]:
    max_steps = max(checkpoints)
    c = c_init.detach().clone().requires_grad_(True)
    best_c = c.detach().clone()
    best_prefix = _prefix_loss_vector(c, dataset.data.positions).detach()
    opt = torch.optim.Adam([c], lr=lr)
    rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    checkpoint_set = set(checkpoints)
    start = time.perf_counter()
    for step in range(0, max_steps + 1):
        if step in checkpoint_set:
            runtime = time.perf_counter() - start
            best = best_c.detach().clone()
            row = _aggregate_refinement_row(
                "constant_lr_adam_refinement",
                f"{label}_adam_lr_{lr:g}_steps_{step}_best_prefix_loss",
                best,
                dataset,
                {"refinement_schedule": "constant", "learning_rate": lr, "refinement_steps": step, "selection": "best_prefix_loss", "runtime_seconds": runtime, "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0])},
            )
            rows.append(row)
            per_rows.extend(_per_trajectory_rows("constant_lr_adam_refinement", row["config_name"], best, dataset, step))
        if step == max_steps:
            break
        loss = torch.mean(_prefix_loss_vector(c, dataset.data.positions))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            c.clamp_(-1.999999999, 1.999999999)
            current = _prefix_loss_vector(c, dataset.data.positions).detach()
            mask = current < best_prefix
            best_prefix[mask] = current[mask]
            best_c[mask] = c.detach()[mask]
    return rows, per_rows, best_c.detach()


def _schedule_refinement_curve(c_init: torch.Tensor, dataset: TeacherSet, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], torch.Tensor]:
    c = c_init.detach().clone().requires_grad_(True)
    best_c = c.detach().clone()
    best_prefix = _prefix_loss_vector(c, dataset.data.positions).detach()
    rows: list[dict[str, Any]] = []
    per_rows: list[dict[str, Any]] = []
    total_step = 0
    start = time.perf_counter()
    for lr, steps in SCHEDULE:
        opt = torch.optim.Adam([c], lr=lr)
        for _ in range(steps):
            loss = torch.mean(_prefix_loss_vector(c, dataset.data.positions))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_step += 1
            with torch.no_grad():
                c.clamp_(-1.999999999, 1.999999999)
                current = _prefix_loss_vector(c, dataset.data.positions).detach()
                mask = current < best_prefix
                best_prefix[mask] = current[mask]
                best_c[mask] = c.detach()[mask]
        runtime = time.perf_counter() - start
        name = f"{label}_staged_adam_through_{total_step}_best_prefix_loss"
        row = _aggregate_refinement_row(
            "staged_lr_adam_refinement",
            name,
            best_c.detach(),
            dataset,
            {"refinement_schedule": "1e-3x500_1e-4x5000_1e-5x20000", "learning_rate": lr, "refinement_steps": total_step, "selection": "best_prefix_loss", "runtime_seconds": runtime, "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0])},
        )
        rows.append(row)
        per_rows.extend(_per_trajectory_rows("staged_lr_adam_refinement", name, best_c.detach(), dataset, total_step))
    return rows, per_rows, best_c.detach()


def _threshold_rows(refinement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for threshold in THRESHOLDS:
        eligible = [r for r in refinement_rows if float(r["future_10_period_mse"]) < threshold]
        rows.append(
            {
                "threshold": threshold,
                "first_constant_lr_step": min([int(r["refinement_steps"]) for r in eligible if r.get("refinement_schedule") == "constant"], default=""),
                "first_any_step": min([int(r["refinement_steps"]) for r in eligible], default=""),
                "num_crossing_rows": len(eligible),
                "best_crossing_config": min(eligible, key=lambda r: float(r["future_10_period_mse"]))["config_name"] if eligible else "",
            }
        )
    return rows


def _precision_floor_audit(refinement_rows: list[dict[str, Any]], per_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best = min(refinement_rows, key=lambda r: float(r["future_10_period_mse"]))
    per_best = [r for r in per_rows if r["checkpoint"] == best["config_name"]]
    c_errors = torch.tensor([float(r["c_abs_error"]) for r in per_best], dtype=DTYPE)
    traj_mse = torch.tensor([float(r["trajectory_mse_10_period"]) for r in per_best], dtype=DTYPE)
    rows.append(
        {
            "audit": "best_observed",
            "config_name": best["config_name"],
            "best_c_rmse": best["c_rmse"],
            "best_10_period_mse": best["future_10_period_mse"],
            "smallest_abs_c_error": float(torch.min(c_errors).item()),
            "largest_abs_c_error": float(torch.max(c_errors).item()),
            "prefix_loss": best["prefix_loss"],
            "max_trajectory_error": best["max_trajectory_error"],
            "final_time_error": best["final_time_error"],
        }
    )
    # Fit log(MSE) = a + b log(c_abs_error) on nonzero finite per-trajectory values.
    mask = (c_errors > 0) & (traj_mse > 0) & torch.isfinite(c_errors) & torch.isfinite(traj_mse)
    if int(mask.sum().item()) >= 2:
        x = torch.log10(c_errors[mask])
        y = torch.log10(traj_mse[mask])
        design = torch.stack([torch.ones_like(x), x], dim=1)
        coef = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
        a, b = float(coef[0].item()), float(coef[1].item())
        for threshold in THRESHOLDS:
            est_log_c = (math.log10(threshold) - a) / b if abs(b) > 1e-12 else float("nan")
            rows.append(
                {
                    "audit": "estimated_c_rmse_needed",
                    "target_10_period_mse": threshold,
                    "fit_intercept_log10": a,
                    "fit_slope_log10": b,
                    "estimated_c_error_needed": 10.0**est_log_c if math.isfinite(est_log_c) else "",
                }
            )
    constant = sorted([r for r in refinement_rows if r.get("refinement_schedule") == "constant"], key=lambda r: int(r["refinement_steps"]))
    if len(constant) >= 2:
        rows.append(
            {
                "audit": "last_constant_lr_trend",
                "previous_step": constant[-2]["refinement_steps"],
                "last_step": constant[-1]["refinement_steps"],
                "previous_prefix_loss": constant[-2]["prefix_loss"],
                "last_prefix_loss": constant[-1]["prefix_loss"],
                "previous_10_period_mse": constant[-2]["future_10_period_mse"],
                "last_10_period_mse": constant[-1]["future_10_period_mse"],
                "prefix_loss_continues_decreasing": float(constant[-1]["prefix_loss"]) < float(constant[-2]["prefix_loss"]),
                "future_rollout_plateaued_or_worsened": float(constant[-1]["future_10_period_mse"]) >= 0.95 * float(constant[-2]["future_10_period_mse"]),
            }
        )
    return rows


def _plot_xy(rows: list[dict[str, Any]], xkey: str, ykey: str, filename: str, *, stage: str | None = None) -> None:
    vals = [r for r in rows if stage is None or r.get("stage") == stage]
    if not vals:
        return
    plt.figure(figsize=(7, 5))
    xs = [float(r[xkey]) for r in vals]
    ys = [float(r[ykey]) for r in vals]
    plt.plot(xs, ys, marker="o")
    if any(v > 0 for v in xs):
        plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_thresholds(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r.get("refinement_schedule") == "constant"]
    plt.figure(figsize=(7, 5))
    for threshold in THRESHOLDS:
        plt.plot([int(r["refinement_steps"]) for r in vals], [float(r[f"percent_trajectories_below_{threshold:g}"]) for r in vals], marker="o", label=f"< {threshold:g}")
    plt.xscale("symlog", linthresh=1)
    plt.xlabel("Adam refinement steps")
    plt.ylabel("% trajectories below threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings_vs_steps.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "config_name", "split", "step", "refinement_steps", "c_rmse", "prefix_loss", "one_step_mse", "two_step_mse", "future_10_period_mse", "max_trajectory_error", "final_time_error"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Long Precision c Refinement</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Long-precision c-only Hybrid Refinement</h1>",
        "<p>Encoder predicts trajectory-specific c from position-only prefix. Prefix-only Adam refines c. Future rollout is evaluation-only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _interpret(encoder_rows: list[dict[str, Any]], refinement_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    test_history = [r for r in encoder_rows if r.get("stage") == "encoder_training" and r.get("split") == "test"]
    best_encoder = min([r for r in encoder_rows if r.get("stage", "").startswith("encoder_final") and r.get("split") == "test"], key=lambda r: float(r["c_rmse"]))
    best_ref = min(refinement_rows, key=lambda r: float(r["future_10_period_mse"]))
    def step_to(threshold: float, schedule: str = "constant") -> int | None:
        vals = [int(r["refinement_steps"]) for r in refinement_rows if r.get("refinement_schedule") == schedule and float(r["future_10_period_mse"]) < threshold]
        return min(vals) if vals else None
    schedule_best = min([r for r in refinement_rows if r.get("refinement_schedule") != "constant"], key=lambda r: float(r["future_10_period_mse"]), default={})
    constant_best = min([r for r in refinement_rows if r.get("refinement_schedule") == "constant"], key=lambda r: float(r["future_10_period_mse"]), default={})
    return {
        "longer_encoder_training_reduced_c_init_vs_previous_0p001444": float(best_encoder["c_rmse"]) < 0.001444,
        "best_encoder_test_c_rmse": best_encoder["c_rmse"],
        "best_encoder_test_10_period_mse": best_encoder["future_10_period_mse"],
        "reaches_below_1e_12": any(float(r["future_10_period_mse"]) < 1e-12 for r in refinement_rows),
        "reaches_below_1e_14": any(float(r["future_10_period_mse"]) < 1e-14 for r in refinement_rows),
        "reaches_below_1e_15": any(float(r["future_10_period_mse"]) < 1e-15 for r in refinement_rows),
        "reaches_below_1e_16": any(float(r["future_10_period_mse"]) < 1e-16 for r in refinement_rows),
        "steps_to_1e_12_constant_lr": step_to(1e-12),
        "steps_to_1e_14_constant_lr": step_to(1e-14),
        "steps_to_1e_15_constant_lr": step_to(1e-15),
        "steps_to_1e_16_constant_lr": step_to(1e-16),
        "staged_lower_lr_improves_over_constant": float(schedule_best.get("future_10_period_mse", float("inf"))) < float(constant_best.get("future_10_period_mse", float("inf"))),
        "best_constant_lr_10_period_mse": constant_best.get("future_10_period_mse", ""),
        "best_staged_lr_10_period_mse": schedule_best.get("future_10_period_mse", ""),
        "best_overall_config": best_ref["config_name"],
        "best_overall_10_period_mse": best_ref["future_10_period_mse"],
        "best_overall_c_rmse": best_ref["c_rmse"],
        "likely_remaining_error_source": "precision/refinement numerical floor" if float(best_ref["future_10_period_mse"]) < 1e-14 else "refinement optimization or c precision",
        "simplified_c_only_hybrid_exact_stage1B_recovery": float(best_ref["future_10_period_mse"]) < 1e-12,
        "future_end_to_end_network_requirement": "It must learn to approximate the prefix-only optimizer/refinement map from prefix evidence to c with errors around the c precision estimated in precision_floor_audit.csv.",
    }


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 151000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 151100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 151200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    print("training long c_init encoder", flush=True)
    model, encoder_history, encoder_finals = _train_encoder(train, val, test)
    with torch.no_grad():
        c_init = _predict_c(model, test)
    print("constant-lr Adam c refinement", flush=True)
    refinement_rows, per_rows, _ = _adam_refinement_curve(c_init, test, REFINE_STEPS, REFINE_LR, "encoder_init")
    print("staged lower-lr Adam c refinement", flush=True)
    staged_rows, staged_per_rows, _ = _schedule_refinement_curve(c_init, test, "encoder_init")
    refinement_rows.extend(staged_rows)
    per_rows.extend(staged_per_rows)
    threshold_rows = _threshold_rows(refinement_rows)
    audit_rows = _precision_floor_audit(refinement_rows, per_rows)
    summary = encoder_finals + refinement_rows
    return {
        "summary": summary,
        "encoder_training_curve": encoder_history,
        "refinement_curve": refinement_rows,
        "per_trajectory_refinement": per_rows,
        "threshold_crossings": threshold_rows,
        "precision_floor_audit": audit_rows,
        "teacher_history": teacher_history,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("encoder_training_curve", "encoder_training_curve.csv"),
        ("refinement_curve", "refinement_curve.csv"),
        ("per_trajectory_refinement", "per_trajectory_refinement.csv"),
        ("threshold_crossings", "threshold_crossings.csv"),
        ("precision_floor_audit", "precision_floor_audit.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs["summary"] + outputs["encoder_training_curve"], outputs["refinement_curve"], outputs["precision_floor_audit"])
    setup = {
        "diagnostic": "stage1B_long_precision_c_refinement",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "operator": "K(c)=[[0,1],[-1,c]]",
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "encoder_lr": ENCODER_LR,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "refinement_optimizer": "adam",
        "constant_refinement_lr": REFINE_LR,
        "constant_refinement_steps": REFINE_STEPS,
        "staged_refinement_schedule": SCHEDULE,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_closed_form_c_solve": False,
        "uses_LS_inside_encoder": False,
        "uses_hessian_or_newton": False,
        "uses_future_data_during_refinement": False,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_xy([r for r in outputs["encoder_training_curve"] if r.get("split") == "test"], "step", "c_rmse", "encoder_c_rmse_vs_training_steps.png")
    _plot_xy([r for r in outputs["encoder_training_curve"] if r.get("split") == "test"], "step", "future_10_period_mse", "encoder_only_rollout_mse_vs_training_steps.png")
    _plot_xy([r for r in outputs["refinement_curve"] if r.get("refinement_schedule") == "constant"], "refinement_steps", "c_rmse", "refinement_c_rmse_vs_steps.png")
    _plot_xy([r for r in outputs["refinement_curve"] if r.get("refinement_schedule") == "constant"], "refinement_steps", "future_10_period_mse", "refinement_rollout_mse_vs_steps.png")
    _plot_xy(outputs["refinement_curve"], "prefix_loss", "future_10_period_mse", "prefix_loss_vs_future_rollout_mse.png")
    _plot_xy(outputs["refinement_curve"], "c_rmse", "future_10_period_mse", "c_error_vs_rollout_mse.png")
    _plot_thresholds(outputs["refinement_curve"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    interpretation = _interpret(outputs["summary"] + outputs["encoder_training_curve"], outputs["refinement_curve"], outputs["precision_floor_audit"])
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
