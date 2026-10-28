"""Stage 1B encoder-initialized gradient-refinement diagnostic.

Hybrid inference method:

    prefix -> TCN encoder -> c_init
    c_init -> small prefix-only GD/Adam refinement -> c_refined
    c_refined -> K(c_refined) rollout.

No future trajectory points are used for training the encoder or refining c.
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

OUTPUT_DIR = Path("outputs/sho/stage1B/encoder_initialized_gradient_refinement")
BATCH_SIZE = int(os.environ.get("EIGR_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("EIGR_MAX_STEPS", "100000"))
EVAL_INTERVAL = int(os.environ.get("EIGR_EVAL_INTERVAL", "500"))
PATIENCE = int(os.environ.get("EIGR_PATIENCE", "15000"))
LAMBDA_ABS = float(os.environ.get("EIGR_LAMBDA_ABS", "1"))
REFINE_STEPS = [0, 1, 2, 5, 10, 20, 50, 100]
REFINE_LRS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]


class CInitTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.inp(prefix.unsqueeze(1)))
        y = self.blocks(y).mean(dim=-1)
        return self.head(y).squeeze(-1)


def _param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


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


def _predict_c(model: CInitTCN, dataset: TeacherSet) -> torch.Tensor:
    return dataset.mu + dataset.sigma * model(dataset.data.positions[:, :P])


def _metrics(stage: str, name: str, c: torch.Tensor, dataset: TeacherSet, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    teacher = dataset.c_free
    err = c - teacher
    mse10, maxerr, finalerr = _future_rollout_mse(c, dataset.data, 10.0)
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
        "prefix_loss": float(_prefix_loss(c, dataset.data.positions).item()),
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
    if extra:
        row.update(extra)
    return row


def _train_encoder(train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[CInitTCN, list[dict[str, Any]], list[dict[str, Any]]]:
    model = CInitTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=float(os.environ.get("EIGR_ENCODER_LR", "3e-4")))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 91000)
    n = train.data.positions.shape[0]
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        z_true = train.z[idx]
        c_true = train.c_free[idx]
        z_pred = model(pos[:, :P])
        c_pred = train.mu + train.sigma * z_pred
        loss = torch.mean((z_pred - z_true) ** 2) + LAMBDA_ABS * torch.mean((c_pred - c_true) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            with torch.no_grad():
                train_row = _metrics("training_history", "c_init_encoder", _predict_c(model, train), train, {"split": "train", "step": step, "train_loss": float(loss.item())})
                val_row = _metrics("training_history", "c_init_encoder", _predict_c(model, val), val, {"split": "val", "step": step, "train_loss": float(loss.item())})
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
    finals = []
    with torch.no_grad():
        for split, dataset in [("train", train), ("val", val), ("test", test)]:
            finals.append(_metrics("encoder_final", "c_init_encoder", _predict_c(model, dataset), dataset, {"split": split, "best_validation_c_rmse": best, "parameter_count": _param_count(model)}))
    return model, history, finals


def _refine_c(c_init: torch.Tensor, dataset: TeacherSet, optimizer_name: str, lr: float, steps: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    if steps == 0:
        return c_init.detach(), c_init.detach(), 0.0
    start = time.perf_counter()
    c = c_init.detach().clone().requires_grad_(True)
    best_c = c.detach().clone()
    best_loss = _prefix_loss_vector(c, dataset.data.positions).detach()
    if optimizer_name == "adam":
        opt = torch.optim.Adam([c], lr=lr)
        for _ in range(steps):
            loss_vec = _prefix_loss_vector(c, dataset.data.positions)
            loss = torch.mean(loss_vec)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                c.clamp_(-1.999999999, 1.999999999)
                current = _prefix_loss_vector(c, dataset.data.positions).detach()
                mask = current < best_loss
                best_loss[mask] = current[mask]
                best_c[mask] = c.detach()[mask]
    elif optimizer_name == "gd":
        for _ in range(steps):
            loss_vec = _prefix_loss_vector(c, dataset.data.positions)
            loss = torch.mean(loss_vec)
            grad = torch.autograd.grad(loss, c)[0]
            with torch.no_grad():
                c -= lr * grad
                c.clamp_(-1.999999999, 1.999999999)
                current = _prefix_loss_vector(c, dataset.data.positions).detach()
                mask = current < best_loss
                best_loss[mask] = current[mask]
                best_c[mask] = c.detach()[mask]
            c.requires_grad_(True)
    else:
        raise ValueError(optimizer_name)
    return c.detach(), best_c.detach(), time.perf_counter() - start


def _refinement_grid(model: CInitTCN, train: TeacherSet, test: TeacherSet) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with torch.no_grad():
        encoder_init = _predict_c(model, test)
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    gen = torch.Generator().manual_seed(SEED + 92000)
    random_init = avg + 0.01 * torch.randn(avg.shape, generator=gen, dtype=DTYPE)
    inits = [("encoder_init", encoder_init), ("average_c", avg), ("random_near_mean", random_init)]
    rows: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for init_name, init_c in inits:
        baselines.append(_metrics("baseline_comparison", f"{init_name}_K0", init_c, test, {"initialization": init_name, "optimizer": "none", "refinement_steps": 0, "selection": "current"}))
        for opt_name in ["gd", "adam"]:
            for lr in REFINE_LRS:
                for steps in REFINE_STEPS:
                    current, best, runtime = _refine_c(init_c, test, opt_name, lr, steps)
                    for selection, c in [("current", current), ("best_prefix_loss", best)]:
                        row = _metrics(
                            "gradient_refinement",
                            f"{init_name}_{opt_name}_lr_{lr:g}_steps_{steps}_{selection}",
                            c,
                            test,
                            {
                                "initialization": init_name,
                                "optimizer": opt_name,
                                "learning_rate": lr,
                                "refinement_steps": steps,
                                "selection": selection,
                                "runtime_seconds": runtime,
                                "runtime_per_trajectory_seconds": runtime / max(1, test.data.positions.shape[0]),
                            },
                        )
                        rows.append(row)
    return rows, baselines


def _baseline_rows(test: TeacherSet) -> list[dict[str, Any]]:
    rows = [_metrics("baseline_comparison", "free_c_teacher", test.c_free, test, {"initialization": "free_c_teacher", "optimizer": "prefix_free_c_teacher", "refinement_steps": 0})]
    for row in _baselines(test):
        rows.append(dict(row, stage="baseline_comparison"))
    return rows


def _plot_steps(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    for init in ["encoder_init", "average_c", "random_near_mean"]:
        vals = [r for r in rows if r.get("initialization") == init and r.get("optimizer") == "adam" and float(r.get("learning_rate", 0)) == 0.01 and r.get("selection") == "best_prefix_loss"]
        if vals:
            vals = sorted(vals, key=lambda r: int(r["refinement_steps"]))
            plt.plot([int(r["refinement_steps"]) for r in vals], [float(r[key]) for r in vals], marker="o", label=init)
    plt.yscale("log")
    plt.xlabel("refinement steps")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_runtime(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 5))
    for r in rows:
        if r.get("selection") != "best_prefix_loss":
            continue
        try:
            plt.scatter(float(r["runtime_per_trajectory_seconds"]), float(r["future_10_period_mse"]), s=18)
        except Exception:
            pass
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("runtime per trajectory (s)")
    plt.ylabel("10-period MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "runtime_vs_accuracy.png", dpi=170)
    plt.close()


def _plot_encoder_vs_average(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r.get("selection") == "best_prefix_loss" and r.get("optimizer") == "adam"]
    labels, y = [], []
    for init in ["encoder_init", "average_c"]:
        subset = [r for r in vals if r.get("initialization") == init]
        if subset:
            best = min(subset, key=lambda r: float(r["future_10_period_mse"]))
            labels.append(init)
            y.append(float(best["future_10_period_mse"]))
    plt.figure(figsize=(5, 4))
    plt.bar(range(len(y)), y)
    plt.yscale("log")
    plt.xticks(range(len(y)), labels)
    plt.ylabel("best 10-period MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoder_init_vs_average_init.png", dpi=170)
    plt.close()


def _interpret(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ref = [r for r in rows if r.get("stage") == "gradient_refinement" and r.get("selection") == "best_prefix_loss"]
    enc = [r for r in ref if r.get("initialization") == "encoder_init"]
    avg = [r for r in ref if r.get("initialization") == "average_c"]
    best = min(ref, key=lambda r: float(r["future_10_period_mse"]), default={})
    best_enc = min(enc, key=lambda r: float(r["future_10_period_mse"]), default={})
    best_avg = min(avg, key=lambda r: float(r["future_10_period_mse"]), default={})
    def steps_to(subset: list[dict[str, Any]], threshold: float) -> int | None:
        xs = [int(r["refinement_steps"]) for r in subset if float(r["future_10_period_mse"]) < threshold]
        return min(xs) if xs else None
    adam_best = min([r for r in ref if r.get("optimizer") == "adam"], key=lambda r: float(r["future_10_period_mse"]), default={})
    gd_best = min([r for r in ref if r.get("optimizer") == "gd"], key=lambda r: float(r["future_10_period_mse"]), default={})
    return {
        "encoder_init_plus_refinement_reaches_1e_6": any(float(r["future_10_period_mse"]) < 1e-6 for r in enc),
        "encoder_init_plus_refinement_reaches_1e_9": any(float(r["future_10_period_mse"]) < 1e-9 for r in enc),
        "encoder_steps_needed_for_1e_6": steps_to(enc, 1e-6),
        "average_c_steps_needed_for_1e_6": steps_to(avg, 1e-6),
        "adam_or_gd_better": "adam" if float(adam_best.get("future_10_period_mse", float("inf"))) < float(gd_best.get("future_10_period_mse", float("inf"))) else "gd",
        "best_config": best.get("config_name", ""),
        "best_10_period_mse": best.get("future_10_period_mse", ""),
        "best_encoder_config": best_enc.get("config_name", ""),
        "best_average_c_config": best_avg.get("config_name", ""),
        "method_framing": "encoder initialization plus prefix-only refinement",
    }


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "config_name", "initialization", "optimizer", "learning_rate", "refinement_steps", "selection", "c_rmse", "future_1_period_mse", "future_2_period_mse", "future_5_period_mse", "future_10_period_mse", "runtime_per_trajectory_seconds"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Encoder Initialized Gradient Refinement</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Encoder-initialized Gradient Refinement</h1>",
        "<p>The network predicts an initializer only. Each trajectory is then refined with GD/Adam on prefix recurrence consistency. Future rollout is evaluation-only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 93000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 93100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 93200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    print("phase 1 training c_init encoder", flush=True)
    model, training_history, encoder_rows = _train_encoder(train, val, test)
    print("phase 2 prefix-only gradient refinement", flush=True)
    refinement_rows, init_baselines = _refinement_grid(model, train, test)
    baselines = _baseline_rows(test) + init_baselines
    summary = baselines + encoder_rows + refinement_rows
    return {
        "summary": summary,
        "refinement_results": refinement_rows,
        "training_history": training_history,
        "baseline_comparison": baselines + encoder_rows,
        "full_run_table": summary + teacher_history + training_history,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("refinement_results", "refinement_results.csv"),
        ("training_history", "training_history.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs["summary"])
    setup = {
        "diagnostic": "stage1B_encoder_initialized_gradient_refinement",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "encoder_loss": "standardized c MSE + lambda_abs * raw c MSE",
        "lambda_abs": LAMBDA_ABS,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "refinement_optimizers": ["gd", "adam"],
        "refinement_learning_rates": REFINE_LRS,
        "refinement_steps": REFINE_STEPS,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_simulator_c_labels_in_training": False,
        "uses_future_data_during_refinement": False,
        "uses_LS_inside_encoder": False,
        "uses_closed_form_c_estimator": False,
        "uses_hessian_or_newton": False,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_steps(outputs["refinement_results"], "future_10_period_mse", "rollout_mse_vs_refinement_steps.png")
    _plot_steps(outputs["refinement_results"], "c_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_encoder_vs_average(outputs["refinement_results"])
    _plot_runtime(outputs["refinement_results"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **_interpret(outputs["summary"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
