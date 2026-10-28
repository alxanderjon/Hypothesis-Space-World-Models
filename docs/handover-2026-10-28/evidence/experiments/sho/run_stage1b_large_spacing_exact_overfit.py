"""Stage 1B large-spacing exact-overfit diagnostic for amortized c_i.

This isolates the easiest wide-c setting before any narrowing curriculum:
can an encoder fit prefix-only free-c teachers to high precision when
c_i values are deliberately far apart?
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
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
    P,
    TEST_NUM,
    TRAIN_NUM,
    VAL_NUM,
    TCNBlock,
    _future_rollout_mse,
)
from experiments.sho.run_stage1b_widened_dynamics_curriculum_c_prediction import (
    SHOSet,
    _free_teacher_multistart,
    _make_c_range_data,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/large_spacing_exact_overfit")
C_RANGE = (0.5, 1.95)
C_RANGE_LABEL = "c in [0.5,1.95]"
EXACT_RMSE = 1e-8
EVAL_INTERVAL = int(os.environ.get("EXACT_OVERFIT_EVAL_INTERVAL", "500"))


def _list_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return [part.strip() for part in raw.split(",") if part.strip()]


def _int_list_env(name: str, default: list[int]) -> list[int]:
    return [int(part) for part in _list_env(name, [str(v) for v in default])]


def _float_list_env(name: str, default: list[float]) -> list[float]:
    return [float(part) for part in _list_env(name, [f"{v:g}" for v in default])]


DATASET_KINDS = _list_env("EXACT_OVERFIT_DATASETS", ["tiny8", "small32", "medium128", "full"])
MODEL_KINDS = _list_env("EXACT_OVERFIT_MODELS", ["tcn", "mlp"])
LOSS_SPECS = _list_env(
    "EXACT_OVERFIT_LOSSES",
    ["raw", "standardized", "combined_1", "combined_100", "combined_10000"],
)
LR_VALUES = _float_list_env("EXACT_OVERFIT_LRS", [1e-3, 3e-4, 1e-4])


@dataclass(frozen=True)
class TeacherSplit:
    name: str
    data: SHOSet
    c_teacher: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor

    @property
    def z_teacher(self) -> torch.Tensor:
        return (self.c_teacher - self.mu) / self.sigma.clamp_min(1e-12)


class TCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(torch.nn.functional.silu(self.inp(x.unsqueeze(1))))
        return self.head(y.mean(dim=-1)).squeeze(-1)


class MLPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(P, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _make_model(kind: str) -> nn.Module:
    if kind == "tcn":
        return TCNEncoder().to(dtype=DTYPE)
    if kind == "mlp":
        return MLPEncoder().to(dtype=DTYPE)
    raise ValueError(kind)


def _param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _dataset_size(kind: str) -> int:
    if kind == "tiny8":
        return 8
    if kind == "small32":
        return 32
    if kind == "medium128":
        return 128
    if kind == "full":
        return TRAIN_NUM
    raise ValueError(kind)


def _max_steps(kind: str) -> int:
    override = os.environ.get("EXACT_OVERFIT_MAX_STEPS")
    if override:
        return int(override)
    if kind in {"tiny8", "small32"}:
        return 100_000
    return 50_000


def _cache_path(split: str, n: int, seed: int) -> Path:
    return OUTPUT_DIR / "teacher_cache" / f"{split}_n{n}_seed{seed}_c_{C_RANGE[0]}_{C_RANGE[1]}.pt"


def _make_teacher(split: str, n: int, seed: int, mu: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> TeacherSplit:
    path = _cache_path(split, n, seed)
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
    return TeacherSplit(split, data, c_teacher, mu, sigma)


def _make_splits(kind: str) -> dict[str, TeacherSplit]:
    n = _dataset_size(kind)
    train = _make_teacher("train_" + kind, n, TRAIN_DATA_SEED + 41000 + n)
    out = {"train": train}
    if kind == "full":
        out["val"] = _make_teacher("val_full", VAL_NUM, TRAIN_DATA_SEED + 42000, train.mu, train.sigma)
        out["test"] = _make_teacher("test_full", TEST_NUM, TEST_DATA_SEED + 43000, train.mu, train.sigma)
    return out


def _loss_spec(spec: str) -> tuple[str, float]:
    if spec == "raw":
        return "raw", 0.0
    if spec == "standardized":
        return "standardized", 0.0
    if spec.startswith("combined_"):
        return "combined", float(spec.split("_", 1)[1])
    raise ValueError(spec)


def _predict_c(raw_output: torch.Tensor, split: TeacherSplit, loss_kind: str) -> torch.Tensor:
    if loss_kind == "raw":
        return raw_output
    return split.mu + split.sigma * raw_output


def _training_loss(raw_output: torch.Tensor, split: TeacherSplit, loss_kind: str, lambda_abs: float) -> torch.Tensor:
    if loss_kind == "raw":
        return torch.mean((raw_output - split.c_teacher) ** 2)
    z_mse = torch.mean((raw_output - split.z_teacher) ** 2)
    if loss_kind == "standardized":
        return z_mse
    c_pred = _predict_c(raw_output, split, loss_kind)
    return z_mse + lambda_abs * torch.mean((c_pred - split.c_teacher) ** 2)


def _scalar_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    ss_res = torch.sum((pred - target) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2).clamp_min(1e-30)
    return float((1.0 - ss_res / ss_tot).item())


def _corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    xp = pred - pred.mean()
    xt = target - target.mean()
    denom = torch.sqrt(torch.sum(xp**2) * torch.sum(xt**2)).clamp_min(1e-30)
    return float((torch.sum(xp * xt) / denom).item())


@torch.no_grad()
def _metrics(
    model: nn.Module,
    split: TeacherSplit,
    loss_kind: str,
    stage: str,
    config_name: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    raw = model(split.data.positions[:, :P])
    c_pred = _predict_c(raw, split, loss_kind)
    err = c_pred - split.c_teacher
    mse10, maxerr, finalerr = _future_rollout_mse(c_pred, split.data, 10.0)
    row: dict[str, Any] = {
        "stage": stage,
        "split": split.name,
        "config_name": config_name,
        "c_rmse": float(torch.sqrt(torch.mean(err**2)).item()),
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
    }
    row.update(extra)
    return row


@torch.no_grad()
def _teacher_quality_row(split: TeacherSplit) -> dict[str, Any]:
    mse10, maxerr, finalerr = _future_rollout_mse(split.c_teacher, split.data, 10.0)
    return {
        "stage": "teacher",
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
def _affine_calibration(
    model: nn.Module,
    fit_split: TeacherSplit,
    eval_split: TeacherSplit,
    loss_kind: str,
    config_name: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    fit_pred = _predict_c(model(fit_split.data.positions[:, :P]), fit_split, loss_kind)
    x = fit_pred
    y = fit_split.c_teacher
    x_mean = x.mean()
    y_mean = y.mean()
    var = torch.sum((x - x_mean) ** 2).clamp_min(1e-30)
    a = torch.sum((x - x_mean) * (y - y_mean)) / var
    b = y_mean - a * x_mean
    eval_pred = _predict_c(model(eval_split.data.positions[:, :P]), eval_split, loss_kind)
    cal = a * eval_pred + b
    err = cal - eval_split.c_teacher
    mse10, maxerr, finalerr = _future_rollout_mse(cal, eval_split.data, 10.0)
    row: dict[str, Any] = {
        "config_name": config_name,
        "fit_split": fit_split.name,
        "eval_split": eval_split.name,
        "affine_a": float(a.item()),
        "affine_b": float(b.item()),
        "calibrated_c_rmse": float(torch.sqrt(torch.mean(err**2)).item()),
        "calibrated_c_r2": _scalar_r2(cal, eval_split.c_teacher),
        "calibrated_correlation": _corr(cal, eval_split.c_teacher),
        "calibrated_10_period_mse": mse10,
        "calibrated_max_trajectory_error": maxerr,
        "calibrated_final_time_absolute_error": finalerr,
    }
    row.update(extra)
    return row


def _plot_history(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10, 5))
    for label in sorted({r["config_name"] for r in rows if r.get("split") == "train"}):
        vals = [r for r in rows if r.get("split") == "train" and r["config_name"] == label]
        vals = sorted(vals, key=lambda r: int(r["step"]))
        if vals:
            plt.plot([int(r["step"]) for r in vals], [float(r["c_rmse"]) for r in vals], label=label, linewidth=1.2)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("train c RMSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_c_rmse_vs_steps.png", dpi=170)
    plt.close()


def _plot_pred_vs_teacher(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r["stage"] == "final" and r["split"].startswith("train")]
    plt.figure(figsize=(8, 5))
    labels = [r["config_name"] for r in vals]
    x = range(len(vals))
    plt.bar(x, [float(r["c_rmse"]) for r in vals])
    plt.yscale("log")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=6)
    plt.ylabel("train c RMSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_c.png", dpi=170)
    plt.close()


def _plot_error_vs_rollout(rows: list[dict[str, Any]]) -> None:
    vals = [r for r in rows if r["stage"] == "final"]
    plt.figure(figsize=(6, 5))
    for r in vals:
        plt.scatter(float(r["c_rmse"]), float(r["state_given_10_period_mse"]), label=r["config_name"], s=25)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("c RMSE")
    plt.ylabel("10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "c_error_vs_rollout_mse.png", dpi=170)
    plt.close()


def _plot_calibration(cal_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> None:
    pairs = []
    for c in cal_rows:
        match = next((r for r in final_rows if r["config_name"] == c["config_name"] and r["split"] == c["eval_split"]), None)
        if match:
            pairs.append((c, match))
    if not pairs:
        return
    labels = [c["config_name"] for c, _ in pairs]
    x = list(range(len(pairs)))
    plt.figure(figsize=(10, 5))
    plt.bar([v - 0.18 for v in x], [float(m["c_rmse"]) for _, m in pairs], width=0.36, label="uncalibrated")
    plt.bar([v + 0.18 for v in x], [float(c["calibrated_c_rmse"]) for c, _ in pairs], width=0.36, label="calibrated")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=60, ha="right", fontsize=6)
    plt.ylabel("c RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibrated_vs_uncalibrated.png", dpi=170)
    plt.close()


def _write_html(summary: list[dict[str, Any]], teacher: list[dict[str, Any]], cal: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    cols = [
        "dataset_kind",
        "model_kind",
        "loss_spec",
        "learning_rate",
        "split",
        "c_rmse",
        "c_r2",
        "collapse_ratio",
        "state_given_10_period_mse",
        "c_max_abs_error",
        "predicted_c_std",
        "teacher_c_std",
        "steps_run",
        "exact_stop",
    ]
    body = []
    for r in summary:
        body.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Large-Spacing Exact Overfit</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}code{background:#f6f8fa;padding:2px 4px;border-radius:4px}</style>",
        "</head><body><h1>Stage 1B Large-Spacing Exact-Overfit Diagnostic</h1>",
        "<p>Stage A only: <code>c_i in [0.5, 1.95]</code>. Teachers are free scalar c_i fits from the position-only prefix.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Teacher Gate</h2><pre>" + escape(json.dumps(teacher, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>",
        "\n".join(body),
        "</tbody></table>",
        "<h2>Calibration</h2><pre>" + escape(json.dumps(cal, indent=2)) + "</pre>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_one(dataset_kind: str, model_kind: str, loss_spec: str, lr: float, splits: dict[str, TeacherSplit]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    loss_kind, lambda_abs = _loss_spec(loss_spec)
    model = _make_model(model_kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train = splits["train"]
    max_steps = _max_steps(dataset_kind)
    config = f"{dataset_kind}_{model_kind}_{loss_spec}_lr_{lr:g}"
    history: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    start = time.time()
    exact_stop = False
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_train_rmse = float("inf")
    best_step = 0
    for step in range(1, max_steps + 1):
        opt.zero_grad()
        raw = model(train.data.positions[:, :P])
        loss = _training_loss(raw, train, loss_kind, lambda_abs)
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == max_steps:
            row = _metrics(
                model,
                train,
                loss_kind,
                "history",
                config,
                {
                    "dataset_kind": dataset_kind,
                    "model_kind": model_kind,
                    "loss_spec": loss_spec,
                    "loss_kind": loss_kind,
                    "lambda_abs": lambda_abs,
                    "learning_rate": lr,
                    "step": step,
                    "train_loss": float(loss.item()),
                    "param_count": _param_count(model),
                },
            )
            history.append(row)
            if float(row["c_rmse"]) < best_train_rmse:
                best_train_rmse = float(row["c_rmse"])
                best_step = step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(
                f"  {config} step={step} train_c_rmse={row['c_rmse']:.4g} "
                f"r2={row['c_r2']:.4g} rollout10={row['state_given_10_period_mse']:.4g}",
                flush=True,
            )
            if float(row["c_rmse"]) < EXACT_RMSE:
                exact_stop = True
                break
    model.load_state_dict(best_state)
    for split_name, split in splits.items():
        final_rows.append(
            _metrics(
                model,
                split,
                loss_kind,
                "final",
                config,
                {
                    "dataset_kind": dataset_kind,
                    "model_kind": model_kind,
                    "loss_spec": loss_spec,
                    "loss_kind": loss_kind,
                    "lambda_abs": lambda_abs,
                    "learning_rate": lr,
                    "steps_run": step,
                    "best_step": best_step,
                    "best_train_c_rmse": best_train_rmse,
                    "max_steps": max_steps,
                    "exact_stop": exact_stop,
                    "wall_clock_sec": time.time() - start,
                    "param_count": _param_count(model),
                },
            )
        )
    cal_rows = []
    fit_split = splits.get("val", train)
    for split in splits.values():
        cal_rows.append(
            _affine_calibration(
                model,
                fit_split,
                split,
                loss_kind,
                config,
                {
                    "dataset_kind": dataset_kind,
                    "model_kind": model_kind,
                    "loss_spec": loss_spec,
                    "learning_rate": lr,
                },
            )
        )
    return history, final_rows, cal_rows


def _interpret(summary: list[dict[str, Any]], cal: list[dict[str, Any]]) -> dict[str, Any]:
    def best_train(kind: str) -> dict[str, Any] | None:
        rows = [r for r in summary if r["stage"] == "final" and r["dataset_kind"] == kind and r["split"].startswith("train")]
        return min(rows, key=lambda r: float(r["c_rmse"])) if rows else None

    tiny = best_train("tiny8")
    small = best_train("small32")
    full = best_train("full")
    best = min([r for r in summary if r["stage"] == "final"], key=lambda r: float(r["c_rmse"])) if summary else {}
    return {
        "tiny8_reaches_1e_8": bool(tiny and float(tiny["c_rmse"]) < EXACT_RMSE),
        "tiny8_best_c_rmse": tiny.get("c_rmse", "") if tiny else "",
        "small32_reaches_1e_8": bool(small and float(small["c_rmse"]) < EXACT_RMSE),
        "small32_best_c_rmse": small.get("c_rmse", "") if small else "",
        "full_best_train_c_rmse": full.get("c_rmse", "") if full else "",
        "best_overall_config": best.get("config_name", ""),
        "best_overall_split": best.get("split", ""),
        "best_overall_c_rmse": best.get("c_rmse", ""),
        "affine_calibration_best_rmse": min((float(r["calibrated_c_rmse"]) for r in cal), default=float("inf")),
        "provisional_read": "If tiny/small overfit fails above 1e-8, fix exact overfit or output scaling before widened-to-target curriculum.",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    all_history: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "full_run_table.csv").open())) if (OUTPUT_DIR / "full_run_table.csv").exists() else []
    all_summary: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "summary.csv").open())) if (OUTPUT_DIR / "summary.csv").exists() else []
    all_cal: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "calibration_results.csv").open())) if (OUTPUT_DIR / "calibration_results.csv").exists() else []
    teacher_rows: list[dict[str, Any]] = list(csv.DictReader((OUTPUT_DIR / "teacher_results.csv").open())) if (OUTPUT_DIR / "teacher_results.csv").exists() else []
    completed = {
        (r.get("dataset_kind"), r.get("model_kind"), r.get("loss_spec"), str(r.get("learning_rate")), r.get("split"))
        for r in all_summary
        if r.get("stage") == "final"
    }

    setup = {
        "diagnostic": "stage1B_large_spacing_exact_overfit",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "c_range": list(C_RANGE),
        "clean_data": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_simulator_c_labels_in_training": False,
        "uses_future_data_for_teacher": False,
        "uses_ls_inside_encoder": False,
        "teacher": "self_supervised_free_c_prefix_optimizer",
        "datasets": DATASET_KINDS,
        "models": MODEL_KINDS,
        "loss_specs": LOSS_SPECS,
        "learning_rates": LR_VALUES,
        "eval_interval": EVAL_INTERVAL,
        "exact_stop_train_c_rmse": EXACT_RMSE,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    for dataset_kind in DATASET_KINDS:
        print(f"building teachers dataset={dataset_kind}", flush=True)
        splits = _make_splits(dataset_kind)
        for split in splits.values():
            if not any(r.get("split") == split.name for r in teacher_rows):
                teacher_rows.append(_teacher_quality_row(split))
        _write_csv(OUTPUT_DIR / "teacher_results.csv", teacher_rows)
        for model_kind in MODEL_KINDS:
            for loss_spec in LOSS_SPECS:
                for lr in LR_VALUES:
                    train_key = (dataset_kind, model_kind, loss_spec, str(lr), splits["train"].name)
                    if train_key in completed:
                        print(f"skipping completed dataset={dataset_kind} model={model_kind} loss={loss_spec} lr={lr:g}", flush=True)
                        continue
                    print(f"training dataset={dataset_kind} model={model_kind} loss={loss_spec} lr={lr:g}", flush=True)
                    hist, rows, cal_rows = _run_one(dataset_kind, model_kind, loss_spec, lr, splits)
                    all_history.extend(hist)
                    all_summary.extend(rows)
                    all_cal.extend(cal_rows)
                    _write_csv(OUTPUT_DIR / "full_run_table.csv", all_history + all_summary)
                    _write_csv(OUTPUT_DIR / "overfit_results.csv", all_summary)
                    _write_csv(OUTPUT_DIR / "summary.csv", all_summary)
                    _write_csv(OUTPUT_DIR / "calibration_results.csv", all_cal)

    interpretation = _interpret(all_summary, all_cal)
    _write_json(OUTPUT_DIR / "metrics.json", {"summary": all_summary, "history": all_history, "teacher": teacher_rows, "calibration": all_cal})
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _plot_history(all_history)
    _plot_pred_vs_teacher(all_summary)
    _plot_error_vs_rollout(all_summary)
    _plot_calibration(all_cal, all_summary)
    _write_html(all_summary, teacher_rows, all_cal, interpretation)


if __name__ == "__main__":
    main()
