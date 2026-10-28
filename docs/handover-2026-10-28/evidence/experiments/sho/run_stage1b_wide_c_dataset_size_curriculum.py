"""Stage 1B dataset-size curriculum for wide-spaced c_i prediction.

Before narrowing the c_i range, this tests whether a TCN encoder can scale
from tiny exact overfit to larger wide-c datasets while preserving precision.
Teachers are free scalar c_i values fitted from the prefix only.
"""

from __future__ import annotations

import csv
import json
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, _future_rollout_mse
from experiments.sho.run_stage1b_large_spacing_exact_overfit import (
    C_RANGE,
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    TCNEncoder,
    TeacherSplit,
    _corr,
    _loss_spec,
    _predict_c,
    _scalar_r2,
    _training_loss,
)
from experiments.sho.run_stage1b_widened_dynamics_curriculum_c_prediction import (
    SHOSet,
    _free_teacher_multistart,
    _make_c_range_data,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import _write_csv, _write_json


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/wide_c_dataset_size_curriculum")
C_RANGE_LABEL = "c in [0.5,1.95]"


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


def _int_list_env(name: str, default: list[int]) -> list[int]:
    return [int(part) for part in _list_env(name, [str(v) for v in default])]


def _float_list_env(name: str, default: list[float]) -> list[float]:
    return [float(part) for part in _list_env(name, [f"{v:g}" for v in default])]


STAGE_SIZES = _int_list_env("SIZE_CURRICULUM_SIZES", [8, 32, 128, 512])
LOSS_SPECS = _list_env("SIZE_CURRICULUM_LOSSES", ["standardized", "combined_1"])
LR_VALUES = _float_list_env("SIZE_CURRICULUM_LRS", [3e-4])
VAL_N = int(os.environ.get("SIZE_CURRICULUM_VAL_N", "128"))
TEST_N = int(os.environ.get("SIZE_CURRICULUM_TEST_N", "128"))
EVAL_INTERVAL = int(os.environ.get("SIZE_CURRICULUM_EVAL_INTERVAL", "500"))
PATIENCE = int(os.environ.get("SIZE_CURRICULUM_PATIENCE", "5000"))
EXACT_STOP_RMSE = float(os.environ.get("SIZE_CURRICULUM_EXACT_STOP_RMSE", "1e-8"))


def _max_steps(n: int) -> int:
    override = os.environ.get("SIZE_CURRICULUM_MAX_STEPS")
    if override:
        return int(override)
    return 100_000 if n <= 32 else 50_000


def _cache_path(name: str, n: int, seed: int) -> Path:
    lo, hi = C_RANGE
    return OUTPUT_DIR / "teacher_cache" / f"{name}_n{n}_seed{seed}_c_{lo}_{hi}.pt"


def _make_teacher_split(name: str, n: int, seed: int, mu: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> TeacherSplit:
    path = _cache_path(name, n, seed)
    if path.exists():
        cached = torch.load(path, weights_only=False)
        data = cached["data"]
        c_teacher = cached["c_teacher"]
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _make_c_range_data(n, seed, C_RANGE)
        c_teacher = _free_teacher_multistart(data)
        torch.save({"data": data, "c_teacher": c_teacher}, path)
    if mu is None:
        mu = c_teacher.mean()
    if sigma is None:
        sigma = c_teacher.std().clamp_min(1e-12)
    return TeacherSplit(name=name, data=data, c_teacher=c_teacher, mu=mu, sigma=sigma)


@torch.no_grad()
def _teacher_row(split: TeacherSplit, stage_size: int | str) -> dict[str, Any]:
    mse10, maxerr, finalerr = _future_rollout_mse(split.c_teacher, split.data, 10.0)
    return {
        "stage_size": stage_size,
        "split": split.name,
        "num_examples": split.c_teacher.numel(),
        "teacher_c_mean": float(split.c_teacher.mean().item()),
        "teacher_c_std": float(split.c_teacher.std().item()),
        "teacher_10_period_mse": mse10,
        "teacher_max_trajectory_error": maxerr,
        "teacher_final_time_absolute_error": finalerr,
        "teacher_reaches_numerical_floor": mse10 < 1e-20,
    }


@torch.no_grad()
def _metrics(
    model: nn.Module,
    split: TeacherSplit,
    loss_kind: str,
    stage: str,
    config_name: str,
    stage_size: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    c_pred = _predict_c(model(split.data.positions[:, :P]), split, loss_kind)
    err = c_pred - split.c_teacher
    mse10, maxerr, finalerr = _future_rollout_mse(c_pred, split.data, 10.0)
    rmse = torch.sqrt(torch.mean(err**2))
    row: dict[str, Any] = {
        "stage": stage,
        "config_name": config_name,
        "stage_size": stage_size,
        "split": split.name,
        "num_examples": split.c_teacher.numel(),
        "c_rmse": float(rmse.item()),
        "c_mae": float(torch.mean(torch.abs(err)).item()),
        "c_max_abs_error": float(torch.max(torch.abs(err)).item()),
        "c_bias": float(torch.mean(err).item()),
        "c_r2": _scalar_r2(c_pred, split.c_teacher),
        "correlation": _corr(c_pred, split.c_teacher),
        "collapse_ratio": float((torch.std(c_pred) / torch.std(split.c_teacher).clamp_min(1e-12)).item()),
        "predicted_c_mean": float(c_pred.mean().item()),
        "predicted_c_std": float(c_pred.std().item()),
        "teacher_c_mean": float(split.c_teacher.mean().item()),
        "teacher_c_std": float(split.c_teacher.std().item()),
        "state_given_10_period_mse": mse10,
        "max_trajectory_error": maxerr,
        "final_time_absolute_error": finalerr,
        "crosses_c_rmse_1e-5": float(rmse.item()) < 1e-5,
        "crosses_c_rmse_1e-7": float(rmse.item()) < 1e-7,
        "crosses_c_rmse_1e-8": float(rmse.item()) < 1e-8,
        "crosses_rollout_mse_1e-6": mse10 < 1e-6,
        "crosses_rollout_mse_1e-9": mse10 < 1e-9,
    }
    if extra:
        row.update(extra)
    return row


def _train_stage(
    model: nn.Module,
    train: TeacherSplit,
    val: TeacherSplit,
    test: TeacherSplit,
    retention_splits: dict[int, TeacherSplit],
    loss_spec: str,
    lr: float,
    config_name: str,
    stage_size: int,
    history: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], list[dict[str, Any]]]:
    loss_kind, lambda_abs = _loss_spec(loss_spec)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    best_step = 0
    stale = 0
    max_steps = _max_steps(stage_size)
    stage_start = time.time()
    final_rows: list[dict[str, Any]] = []
    retention_rows: list[dict[str, Any]] = []
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        loss = _training_loss(model(train.data.positions[:, :P]), train, loss_kind, lambda_abs)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_steps:
            train_row = _metrics(
                model,
                train,
                loss_kind,
                "history",
                config_name,
                stage_size,
                {
                    "loss_spec": loss_spec,
                    "loss_kind": loss_kind,
                    "lambda_abs": lambda_abs,
                    "learning_rate": lr,
                    "step": step,
                    "train_loss": float(loss.item()),
                },
            )
            val_row = _metrics(
                model,
                val,
                loss_kind,
                "history",
                config_name,
                stage_size,
                {
                    "loss_spec": loss_spec,
                    "loss_kind": loss_kind,
                    "lambda_abs": lambda_abs,
                    "learning_rate": lr,
                    "step": step,
                    "train_loss": float(loss.item()),
                },
            )
            train_row["history_eval_split"] = "train"
            val_row["history_eval_split"] = "val"
            history.extend([train_row, val_row])
            print(
                f"  {config_name} N={stage_size} step={step} "
                f"train_rmse={train_row['c_rmse']:.4g} val_rmse={val_row['c_rmse']:.4g}",
                flush=True,
            )
            val_rmse = float(val_row["c_rmse"])
            if val_rmse < best_val:
                best_val = val_rmse
                best_step = step
                stale = 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += EVAL_INTERVAL
            if val_rmse < EXACT_STOP_RMSE:
                break
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    common = {
        "loss_spec": loss_spec,
        "loss_kind": loss_kind,
        "lambda_abs": lambda_abs,
        "learning_rate": lr,
        "best_val_c_rmse": best_val,
        "best_step": best_step,
        "max_steps": max_steps,
        "wall_clock_sec": time.time() - stage_start,
    }
    for split in [train, val, test]:
        final_rows.append(_metrics(model, split, loss_kind, "stage_final", config_name, stage_size, common))
    for old_n, split in retention_splits.items():
        retention = _metrics(model, split, loss_kind, "retention", config_name, stage_size, common)
        retention["retention_subset_size"] = old_n
        retention_rows.append(retention)
    return best_state, final_rows, retention_rows


def _plot_metric_by_size(rows: list[dict[str, Any]], metric: str, filename: str) -> None:
    vals = [r for r in rows if r["stage"] == "stage_final" and r["split"].startswith("test")]
    if not vals:
        vals = [r for r in rows if r["stage"] == "stage_final" and r["split"].startswith("val")]
    plt.figure(figsize=(8, 5))
    for label in sorted({r["config_name"] for r in vals}):
        group = sorted([r for r in vals if r["config_name"] == label], key=lambda r: int(r["stage_size"]))
        plt.plot([int(r["stage_size"]) for r in group], [float(r[metric]) for r in group], marker="o", label=label)
    plt.xscale("log", base=2)
    if "mse" in metric or "rmse" in metric:
        plt.yscale("log")
    plt.xlabel("dataset size N")
    plt.ylabel(metric)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_train_vs_val(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for label in sorted({r["config_name"] for r in history}):
        for split_name, linestyle in [("train", "-"), ("val", "--")]:
            vals = [
                r
                for r in history
                if r["config_name"] == label and r.get("history_eval_split") == split_name
            ]
            vals = sorted(vals, key=lambda r: (int(r["stage_size"]), int(r["step"])))
            if vals:
                x = [int(r["stage_size"]) * 1_000_000 + int(r["step"]) for r in vals]
                plt.plot(x, [float(r["c_rmse"]) for r in vals], linestyle=linestyle, linewidth=1.1, label=f"{label} {split_name}")
    plt.yscale("log")
    plt.xlabel("stage/step packed axis")
    plt.ylabel("c RMSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_vs_val_c_rmse.png", dpi=170)
    plt.close()


def _plot_retention(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for old_n in sorted({int(r["retention_subset_size"]) for r in rows}):
        group = sorted([r for r in rows if int(r["retention_subset_size"]) == old_n], key=lambda r: int(r["stage_size"]))
        plt.plot([int(r["stage_size"]) for r in group], [float(r["c_rmse"]) for r in group], marker="o", label=f"retain N={old_n}")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("current curriculum N")
    plt.ylabel("retention c RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "retention_old_subsets.png", dpi=170)
    plt.close()


def _plot_spread(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r["stage"] == "stage_final" and r["split"].startswith("test")]
    if not vals:
        vals = [r for r in rows if r["stage"] == "stage_final" and r["split"].startswith("val")]
    plt.figure(figsize=(9, 5))
    x = list(range(len(vals)))
    labels = [f"{r['config_name']} N={r['stage_size']}" for r in vals]
    width = 0.38
    plt.bar([v - width / 2 for v in x], [float(r["predicted_c_std"]) for r in vals], width=width, label="pred")
    plt.bar([v + width / 2 for v in x], [float(r["teacher_c_std"]) for r in vals], width=width, label="teacher")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=6)
    plt.ylabel("c std")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_by_N.png", dpi=170)
    plt.close()


def _interpret(rows: list[dict[str, Any]], retention: list[dict[str, Any]]) -> dict[str, Any]:
    final = [r for r in rows if r["stage"] == "stage_final" and r["split"].startswith("val")]
    by_n: dict[int, float] = {}
    for r in final:
        n = int(r["stage_size"])
        by_n[n] = min(by_n.get(n, float("inf")), float(r["c_rmse"]))
    break_n = next((n for n in sorted(by_n) if by_n[n] >= 1e-5), "")
    best = min(final, key=lambda r: float(r["c_rmse"])) if final else {}
    retention_bad = any(float(r["c_rmse"]) >= 1e-5 for r in retention if int(r["stage_size"]) > int(r["retention_subset_size"]))
    return {
        "best_validation_config": best.get("config_name", ""),
        "best_validation_stage_size": best.get("stage_size", ""),
        "best_validation_c_rmse": best.get("c_rmse", ""),
        "validation_c_rmse_by_N": by_n,
        "precision_break_N_at_1e-5": break_n,
        "retention_failure_above_1e-5": retention_bad,
        "provisional_decision": "Proceed to range narrowing only if validation/test precision remains below the target threshold at the largest attempted N.",
    }


def _write_html(rows: list[dict[str, Any]], retention: list[dict[str, Any]], teachers: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    cols = [
        "config_name",
        "stage_size",
        "split",
        "loss_spec",
        "learning_rate",
        "c_rmse",
        "c_r2",
        "collapse_ratio",
        "c_max_abs_error",
        "c_bias",
        "predicted_c_std",
        "teacher_c_std",
        "state_given_10_period_mse",
        "crosses_c_rmse_1e-5",
        "crosses_c_rmse_1e-7",
        "crosses_c_rmse_1e-8",
        "crosses_rollout_mse_1e-6",
        "crosses_rollout_mse_1e-9",
    ]
    body = []
    for r in rows:
        if r["stage"] != "stage_final":
            continue
        body.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Wide c Dataset-Size Curriculum</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B Wide c_i Dataset-Size Curriculum</h1>",
        "<p>Wide range only: c_i in [0.5, 1.95]. The model carries the best validation checkpoint from smaller to larger N.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Teacher Gate</h2><pre>" + escape(json.dumps(teachers, indent=2)) + "</pre>",
        "<h2>Stage Results</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>",
        "\n".join(body),
        "</tbody></table>",
        "<h2>Retention</h2><pre>" + escape(json.dumps(retention, indent=2)) + "</pre>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    setup = {
        "diagnostic": "stage1B_wide_c_dataset_size_curriculum",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "c_range": list(C_RANGE),
        "c_range_label": C_RANGE_LABEL,
        "stage_sizes": STAGE_SIZES,
        "loss_specs": LOSS_SPECS,
        "learning_rates": LR_VALUES,
        "val_n": VAL_N,
        "test_n": TEST_N,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_simulator_c_labels_in_training": False,
        "uses_ls_inside_encoder": False,
        "uses_closed_form_c_estimator": False,
        "teacher": "prefix_only_free_c_optimizer",
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    val_seed = TRAIN_DATA_SEED + 52000
    test_seed = TEST_DATA_SEED + 53000
    teacher_rows: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "teacher_results.csv").open())) if (OUTPUT_DIR / "teacher_results.csv").exists() else []
    history_rows: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "training_history.csv").open())) if (OUTPUT_DIR / "training_history.csv").exists() else []
    stage_rows: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "size_curriculum_results.csv").open())) if (OUTPUT_DIR / "size_curriculum_results.csv").exists() else []
    retention_rows: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "retention_results.csv").open())) if (OUTPUT_DIR / "retention_results.csv").exists() else []
    completed = {
        (r.get("config_name"), str(r.get("stage_size")), r.get("split"))
        for r in stage_rows
        if r.get("stage") == "stage_final"
    }

    for lr in LR_VALUES:
        for loss_spec in LOSS_SPECS:
            print(f"starting curriculum loss={loss_spec} lr={lr:g}", flush=True)
            model = TCNEncoder().to(dtype=DTYPE)
            config_name = f"tcn_{loss_spec}_lr_{lr:g}"
            retention_splits: dict[int, TeacherSplit] = {}
            for n in STAGE_SIZES:
                print(f"building teachers N={n}", flush=True)
                train = _make_teacher_split(f"train_N{n}", n, TRAIN_DATA_SEED + 51000 + n)
                val = _make_teacher_split("val_wide", VAL_N, val_seed, train.mu, train.sigma)
                test = _make_teacher_split("test_wide", TEST_N, test_seed, train.mu, train.sigma)
                if not any(r.get("split") == train.name for r in teacher_rows):
                    teacher_rows.append(_teacher_row(train, n))
                if not any(r.get("split") == val.name for r in teacher_rows):
                    teacher_rows.append(_teacher_row(val, "val"))
                if not any(r.get("split") == test.name for r in teacher_rows):
                    teacher_rows.append(_teacher_row(test, "test"))
                if n in {8, 32}:
                    retention_splits[n] = train
                if (config_name, str(n), train.name) in completed:
                    print(f"skipping completed config={config_name} N={n}", flush=True)
                    continue
                _write_csv(OUTPUT_DIR / "teacher_results.csv", teacher_rows)
                _, final_rows, retain = _train_stage(
                    model,
                    train,
                    val,
                    test,
                    retention_splits,
                    loss_spec,
                    lr,
                    config_name,
                    n,
                    history_rows,
                )
                stage_rows.extend(final_rows)
                retention_rows.extend(retain)
                _write_csv(OUTPUT_DIR / "training_history.csv", history_rows)
                _write_csv(OUTPUT_DIR / "size_curriculum_results.csv", stage_rows)
                _write_csv(OUTPUT_DIR / "summary.csv", stage_rows)
                _write_csv(OUTPUT_DIR / "retention_results.csv", retention_rows)

    interpretation = _interpret(stage_rows, retention_rows)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", {"stage_results": stage_rows, "history": history_rows, "retention": retention_rows, "teacher": teacher_rows})
    _plot_metric_by_size(stage_rows, "c_rmse", "c_rmse_vs_dataset_size.png")
    _plot_metric_by_size(stage_rows, "state_given_10_period_mse", "rollout_mse_vs_dataset_size.png")
    _plot_spread(stage_rows)
    _plot_retention(retention_rows)
    _plot_train_vs_val(history_rows)
    _write_html(stage_rows, retention_rows, teacher_rows, interpretation)


if __name__ == "__main__":
    main()
