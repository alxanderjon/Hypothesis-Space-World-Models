"""Targeted Stage 1B large-spaced c_i amortization diagnostic.

This isolates the first widened-dynamics stage: can a TCN encoder learn
prefix -> c_i when the trajectory constants are deliberately far apart?
The teacher is the self-supervised free-c prefix optimizer; true c is used
only for data generation and diagnostic attribution.
"""

from __future__ import annotations

import csv
import json
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

from experiments.sho.run_stage1b_widened_dynamics_curriculum_c_prediction import (
    BATCH_SIZE,
    DTYPE,
    MAX_STEPS_PER_STAGE,
    PATIENCE,
    EVAL_INTERVAL,
    LR_VALUES,
    SEED,
    TCNEncoder,
    TeacherSet,
    _eval,
    _loss_parts,
    _make_stage_teachers,
    _z,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    TRAIN_DATA_SEED,
    _write_csv,
    _write_json,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/widened_dynamics_large_spacing_targeted")
STAGE_NAME = "A_very_wide"
C_RANGE = (0.5, 1.95)
C_RANGE_LABEL = "c in [0.5,1.95]"
LOSS_VARIANTS = [part for part in os.environ.get("LARGE_SPACING_LOSS_VARIANTS", "combined,mse").split(",") if part]
RUN_LR_VALUES = [
    float(part)
    for part in os.environ.get("LARGE_SPACING_LR_VALUES", ",".join(f"{v:g}" for v in LR_VALUES[:2])).split(",")
    if part.strip()
]
CACHE_PATH = OUTPUT_DIR / "stage_A_teacher_cache.pt"


def _teacher_sets() -> tuple[TeacherSet, TeacherSet, TeacherSet, list[dict[str, Any]]]:
    if CACHE_PATH.exists():
        cached = torch.load(CACHE_PATH, weights_only=False)
        return cached["train"], cached["val"], cached["test"], cached["teacher_rows"]
    train, val, test, teacher_rows = _make_stage_teachers(STAGE_NAME, C_RANGE, TRAIN_DATA_SEED + 30000)
    torch.save({"train": train, "val": val, "test": test, "teacher_rows": teacher_rows}, CACHE_PATH)
    return train, val, test, teacher_rows


@torch.no_grad()
def _cheap_c_metrics(model: torch.nn.Module, dataset: TeacherSet, config_name: str, extra: dict[str, Any]) -> dict[str, Any]:
    z_pred = model(dataset.data.positions[:, :128])
    z_true = _z(dataset)
    c_pred = dataset.mu + dataset.sigma * z_pred
    c_true = dataset.c_free
    rmse = torch.sqrt(torch.mean((c_pred - c_true) ** 2))
    mae = torch.mean(torch.abs(c_pred - c_true))
    c_std = torch.std(c_true).clamp_min(1e-12)
    ss_res = torch.sum((c_pred - c_true) ** 2)
    ss_tot = torch.sum((c_true - c_true.mean()) ** 2).clamp_min(1e-30)
    r2 = 1.0 - ss_res / ss_tot
    xp = c_pred - c_pred.mean()
    xt = c_true - c_true.mean()
    corr = torch.sum(xp * xt) / torch.sqrt(torch.sum(xp**2) * torch.sum(xt**2)).clamp_min(1e-30)
    row: dict[str, Any] = {
        "stage": "large_spacing_history",
        "config_name": config_name,
        "c_rmse": float(rmse.item()),
        "c_mae": float(mae.item()),
        "diagnostic_c_r2": float(r2.item()),
        "correlation": float(corr.item()),
        "collapse_ratio": float((torch.std(c_pred) / c_std).item()),
        "predicted_c_mean": float(c_pred.mean().item()),
        "predicted_c_std": float(c_pred.std().item()),
        "teacher_c_mean": float(c_true.mean().item()),
        "teacher_c_std": float(c_true.std().item()),
    }
    row.update(extra)
    return row


def _train_large_spacing(
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    variant: str,
    lr: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    model = TCNEncoder().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_score = float("inf")
    stale = 0
    gen = torch.Generator().manual_seed(SEED + int(lr * 1e9) + (0 if variant == "combined" else 100000))
    n = train.data.positions.shape[0]
    config_name = f"{STAGE_NAME}_{variant}_lr_{lr:g}"
    z_train = _z(train)
    for step in range(1, MAX_STEPS_PER_STAGE + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        opt.zero_grad()
        loss, parts = _loss_parts(model(train.data.positions[idx, :128]), z_train[idx], variant)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS_PER_STAGE:
            row = _cheap_c_metrics(
                model,
                val,
                config_name,
                {
                    "step": step,
                    "loss_variant": variant,
                    "learning_rate": lr,
                    "train_loss": float(loss.item()),
                    **parts,
                },
            )
            history.append(row)
            _write_csv(OUTPUT_DIR / "training_history.csv", history)
            print(
                f"    step={step} val_c_rmse={row['c_rmse']:.4g} "
                f"r2={row['diagnostic_c_r2']:.4g} collapse={row['collapse_ratio']:.4g}",
                flush=True,
            )
            score = float(row["c_rmse"])
            if score < best_score:
                best_score = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return _eval(
        model,
        test,
        "large_spacing_targeted",
        config_name,
        {
            "curriculum_stage": STAGE_NAME,
            "c_i_range": C_RANGE_LABEL,
            "encoder_type": "tcn",
            "loss_variant": variant,
            "learning_rate": lr,
            "best_val_c_rmse": best_score,
            "max_steps": MAX_STEPS_PER_STAGE,
            "patience": PATIENCE,
        },
    )


def _plot_history(history: list[dict[str, Any]], metric: str, filename: str) -> None:
    if not any(metric in r for r in history):
        return
    plt.figure(figsize=(8, 5))
    for label in sorted({r["config_name"] for r in history}):
        rows = [r for r in history if r["config_name"] == label and metric in r]
        rows = sorted(rows, key=lambda r: int(r["step"]))
        if not rows:
            continue
        plt.plot(
            [int(r["step"]) for r in rows],
            [float(r[metric]) for r in rows],
            marker="o",
            markersize=2,
            linewidth=1.2,
            label=label,
        )
    if "mse" in metric or "rmse" in metric:
        plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel(metric)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_predicted_vs_teacher(rows: list[dict[str, Any]]) -> None:
    labels = [str(r["config_name"]) for r in rows]
    pred_std = [float(r["predicted_c_std"]) for r in rows]
    teacher_std = [float(r["teacher_c_std"]) for r in rows]
    x = list(range(len(rows)))
    width = 0.38
    plt.figure(figsize=(8, 5))
    plt.bar([v - width / 2 for v in x], pred_std, width=width, label="predicted c std")
    plt.bar([v + width / 2 for v in x], teacher_std, width=width, label="teacher c std")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("standard deviation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_spread.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = [
        "config_name",
        "loss_variant",
        "learning_rate",
        "c_rmse",
        "c_mae",
        "c_max_abs_error",
        "diagnostic_c_r2",
        "correlation",
        "collapse_ratio",
        "predicted_c_std",
        "teacher_c_std",
        "state_given_1_period_mse",
        "state_given_2_period_mse",
        "state_given_5_period_mse",
        "state_given_10_period_mse",
    ]
    rows_html = []
    for r in outputs["summary"]:
        rows_html.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Large-Spaced c_i Targeted Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}code{background:#f6f8fa;padding:2px 4px;border-radius:4px}</style>",
        "</head><body><h1>Stage 1B Large-Spaced c_i Targeted Diagnostic</h1>",
        "<p>This run isolates Stage A only: <code>c_i in [0.5, 1.95]</code>. The encoder trains on standardized self-supervised free-c teachers from position-only prefixes.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>",
        "\n".join(rows_html),
        "</tbody></table>",
        "<h2>Plots</h2><ul>",
        "<li><a href='c_rmse_vs_step.png'>c RMSE vs step</a></li>",
        "<li><a href='c_r2_vs_step.png'>c R² vs step</a></li>",
        "<li><a href='collapse_ratio_vs_step.png'>collapse ratio vs step</a></li>",
        "<li><a href='rollout_mse_vs_step.png'>10-period rollout MSE vs step</a></li>",
        "<li><a href='predicted_vs_teacher_spread.png'>predicted vs teacher spread</a></li>",
        "</ul></body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _interpret(summary: list[dict[str, Any]], teacher_rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(summary, key=lambda r: float(r["c_rmse"]))
    best_rollout = min(summary, key=lambda r: float(r["state_given_10_period_mse"]))
    teacher_test = next(r for r in teacher_rows if r["split"] == "test")
    return {
        "stage": STAGE_NAME,
        "c_range": C_RANGE_LABEL,
        "teacher_test_c_std": teacher_test["teacher_c_std"],
        "teacher_test_10_period_mse": teacher_test["state_given_10_period_mse"],
        "best_c_rmse_config": best["config_name"],
        "best_c_rmse": best["c_rmse"],
        "best_c_r2": best["diagnostic_c_r2"],
        "best_correlation": best["correlation"],
        "best_collapse_ratio": best["collapse_ratio"],
        "best_rollout_config": best_rollout["config_name"],
        "best_10_period_rollout_mse": best_rollout["state_given_10_period_mse"],
        "learned_large_spacing": float(best["diagnostic_c_r2"]) > 0.99 and float(best["collapse_ratio"]) > 0.8,
        "pure_amortized_reaches_1e_6_rollout": float(best_rollout["state_given_10_period_mse"]) < 1e-6,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    print(f"building teachers for {STAGE_NAME}", flush=True)
    train, val, test, teacher_rows = _teacher_sets()
    _write_csv(OUTPUT_DIR / "teacher_results.csv", teacher_rows)

    summary: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for lr in RUN_LR_VALUES:
        for variant in LOSS_VARIANTS:
            print(f"training large-spacing encoder loss={variant} lr={lr:g}", flush=True)
            final = _train_large_spacing(train, val, test, variant, lr, history)
            summary.append(final)
            _write_csv(OUTPUT_DIR / "training_history.csv", history)
            _write_csv(OUTPUT_DIR / "summary.csv", summary)

    interpretation = _interpret(summary, teacher_rows)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_widened_dynamics_large_spacing_targeted",
            "dtype": str(DTYPE),
            "stage": STAGE_NAME,
            "c_range": list(C_RANGE),
            "prefix_only": True,
            "uses_velocity": False,
            "uses_omega_labels": False,
            "uses_coefficient_labels": False,
            "uses_true_c_labels_in_training_loss": False,
            "teacher": "self_supervised_free_c_from_prefix",
            "max_steps_per_stage": MAX_STEPS_PER_STAGE,
            "eval_interval": EVAL_INTERVAL,
            "patience": PATIENCE,
            "learning_rates": RUN_LR_VALUES,
            "loss_variants": LOSS_VARIANTS,
        },
    )
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", {"summary": summary, "teacher_results": teacher_rows, "history": history})
    _plot_history(history, "c_rmse", "c_rmse_vs_step.png")
    _plot_history(history, "diagnostic_c_r2", "c_r2_vs_step.png")
    _plot_history(history, "collapse_ratio", "collapse_ratio_vs_step.png")
    _plot_history(history, "state_given_10_period_mse", "rollout_mse_vs_step.png")
    _plot_predicted_vs_teacher(summary)
    _write_html({"summary": summary}, interpretation)


if __name__ == "__main__":
    main()
