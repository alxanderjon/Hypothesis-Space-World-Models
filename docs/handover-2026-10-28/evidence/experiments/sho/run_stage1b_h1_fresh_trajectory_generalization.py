"""Stage 1B H=1 fresh-trajectory generalization diagnostic.

This diagnostic trains a TCN encoder on freshly generated clean SHO prefixes at
every optimizer step. It tests whether fixed-set memorization was the reason the
strict H=1 gate failed.

Training uses only prefix one-step prediction errors, optionally with a
transition-covariance auxiliary term. Prefix-only free-c teachers are computed
for diagnostics only on fixed monitor/validation/test sets.
"""

from __future__ import annotations

import copy
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    P,
    SHOSet,
    TeacherSet,
    TCNBlock,
    _make_data,
    _make_teacher_sets,
    _r2,
)
from experiments.sho.run_stage1b_covariance_superloss_encoder_diagnostic import _covariance_loss_vector
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import _param_count
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _state_at_time
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

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_fresh_trajectory_generalization")
BATCH_SIZE = int(os.environ.get("H1FRESH_BATCH_SIZE", "32"))
TRAIN_MONITOR_N = 32
VAL_N = 16
TEST_N = 16
POOL_N = 128
LR = float(os.environ.get("H1FRESH_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1FRESH_MAX_STEPS", "50000"))
EVAL_INTERVAL = int(os.environ.get("H1FRESH_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("H1FRESH_PATIENCE", "5000"))
GATE_MAX_ABS_ERROR = float(os.environ.get("H1FRESH_GATE_MAX_ABS_ERROR", "1e-6"))
SMOOTHMAX_TEMPERATURE = float(os.environ.get("H1FRESH_SMOOTHMAX_TEMPERATURE", "1e-6"))
COV_WEIGHT = 0.1


class COnlyTCN(nn.Module):
    """TCN encoder: position prefix -> trajectory-specific c_i."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.inp(prefix.unsqueeze(1)))
        y = self.blocks(y).mean(dim=-1)
        raw = self.head(y).squeeze(-1)
        return 2.0 * torch.tanh(raw)


def _sample_prefix_batch(n: int, generator: torch.Generator) -> SHOSet:
    initial_conditions = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    lo, hi = OMEGA_RANGE
    omegas = float(lo) + (float(hi) - float(lo)) * torch.rand(n, generator=generator, dtype=DTYPE)
    times = torch.arange(P, dtype=DTYPE).unsqueeze(0) * float(DELTA_T)
    times = times.expand(n, -1)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _stratified_subset(data: SHOSet, n: int) -> SHOSet:
    order = torch.argsort(data.omegas)
    positions = torch.linspace(0, data.omegas.numel() - 1, n, dtype=DTYPE)
    idx = order[positions.round().to(torch.long)]
    return SHOSet(positions=data.positions[idx], omegas=data.omegas[idx])


def _smoothmax(values: torch.Tensor, temperature: float = SMOOTHMAX_TEMPERATURE) -> torch.Tensor:
    flat = values.reshape(-1)
    if temperature <= 0:
        return torch.max(flat)
    m = torch.max(flat)
    return m + temperature * torch.log(torch.mean(torch.exp((flat - m) / temperature)))


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _h1_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return _smoothmax(_h1_errors(c, positions) ** 2)


@torch.no_grad()
def _max_h1_abs_error(c: torch.Tensor, positions: torch.Tensor) -> float:
    return float(torch.max(torch.abs(_h1_errors(c, positions))).item())


def _predict_c(model: nn.Module, dataset: TeacherSet | SHOSet) -> torch.Tensor:
    positions = dataset.data.positions if isinstance(dataset, TeacherSet) else dataset.positions
    return model(positions[:, :P])


def _branch_grid() -> list[dict[str, Any]]:
    return [
        {"branch": "tcn_h1_loss", "use_covariance": False, "covariance_weight": 0.0},
        {"branch": "tcn_h1_loss_plus_covariance", "use_covariance": True, "covariance_weight": COV_WEIGHT},
    ]


def _metrics(
    model: nn.Module,
    dataset: TeacherSet,
    branch: str,
    split: str,
    step: int | str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        c_err = c - dataset.c_free
        cov = _covariance_loss_vector(c, dataset.data.positions)
        h1_max = _max_h1_abs_error(c, dataset.data.positions)
    row: dict[str, Any] = {
        "branch": branch,
        "split": split,
        "step": step,
        "max_h1_abs_error": h1_max,
        "max_h1_squared_error": h1_max**2,
        "smoothmax_h1_squared_error": float(_h1_loss(c, dataset.data.positions).item()),
        "max_abs_c_error_vs_teacher": float(torch.max(torch.abs(c_err)).item()),
        "c_rmse_vs_teacher": float(torch.sqrt(torch.mean(c_err**2)).item()),
        "c_r2_vs_teacher": _r2(c[:, None], dataset.c_free[:, None]),
        "c_bias_vs_teacher": float(torch.mean(c_err).item()),
        "predicted_c_std": float(torch.std(c).item()),
        "teacher_c_std": float(torch.std(dataset.c_free).item()),
        "collapse_ratio": float((torch.std(c) / torch.std(dataset.c_free).clamp_min(1e-12)).item()),
        "covariance_residual": float(cov.mean().item()),
        "covariance_residual_max": float(cov.max().item()),
        "parameter_count": _param_count(model),
    }
    if extra:
        row.update(extra)
    return row


@torch.no_grad()
def _batch_train_metrics(model: nn.Module, batch: SHOSet) -> dict[str, float]:
    c = _predict_c(model, batch)
    cov = _covariance_loss_vector(c, batch.positions)
    return {
        "fresh_batch_max_h1_abs_error": _max_h1_abs_error(c, batch.positions),
        "fresh_batch_covariance_residual": float(cov.mean().item()),
        "fresh_batch_covariance_residual_max": float(cov.max().item()),
    }


def _train_branch(
    cfg: dict[str, Any],
    train_monitor: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    branch = str(cfg["branch"])
    model = COnlyTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(SEED + 221000 + (10000 if cfg["use_covariance"] else 0))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefix_batch(BATCH_SIZE, gen)
        c = _predict_c(model, batch)
        loss = _h1_loss(c, batch.positions)
        cov_vec = _covariance_loss_vector(c, batch.positions)
        cov_loss = _smoothmax(cov_vec)
        if cfg["use_covariance"]:
            loss = loss + float(cfg["covariance_weight"]) * cov_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            train_batch_stats = _batch_train_metrics(model, batch)
            train_row = _metrics(
                model,
                train_monitor,
                branch,
                "train_monitor",
                step,
                {
                    "train_loss": float(loss.detach().item()),
                    "train_h1_loss": float(_h1_loss(c.detach(), batch.positions).item()),
                    "train_covariance_aux_loss": float(cov_loss.detach().item()),
                    "elapsed_seconds": time.perf_counter() - start,
                    **train_batch_stats,
                },
            )
            val_row = _metrics(
                model,
                val,
                branch,
                "val",
                step,
                {
                    "train_loss": float(loss.detach().item()),
                    "train_h1_loss": float(_h1_loss(c.detach(), batch.positions).item()),
                    "train_covariance_aux_loss": float(cov_loss.detach().item()),
                    "elapsed_seconds": time.perf_counter() - start,
                    **train_batch_stats,
                },
            )
            history.extend([train_row, val_row])
            score = float(val_row["max_h1_abs_error"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    train_final = _metrics(model, train_monitor, branch, "train_monitor", "best", {"best_validation_max_h1_abs_error": best})
    val_final = _metrics(model, val, branch, "val", "best", {"best_validation_max_h1_abs_error": best})
    test_final = _metrics(model, test, branch, "test", "best", {"best_validation_max_h1_abs_error": best})
    passed = bool(val_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR and test_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR)
    final = {
        "branch": branch,
        "uses_covariance": cfg["use_covariance"],
        "covariance_weight": cfg["covariance_weight"],
        "gate_passed": passed,
        "gate_threshold": GATE_MAX_ABS_ERROR,
        "train_monitor_max_h1_abs_error": train_final["max_h1_abs_error"],
        "val_max_h1_abs_error": val_final["max_h1_abs_error"],
        "test_max_h1_abs_error": test_final["max_h1_abs_error"],
        "train_monitor_c_rmse_vs_teacher": train_final["c_rmse_vs_teacher"],
        "val_c_rmse_vs_teacher": val_final["c_rmse_vs_teacher"],
        "test_c_rmse_vs_teacher": test_final["c_rmse_vs_teacher"],
        "train_monitor_max_abs_c_error_vs_teacher": train_final["max_abs_c_error_vs_teacher"],
        "val_max_abs_c_error_vs_teacher": val_final["max_abs_c_error_vs_teacher"],
        "test_max_abs_c_error_vs_teacher": test_final["max_abs_c_error_vs_teacher"],
        "train_monitor_covariance_residual": train_final["covariance_residual"],
        "val_covariance_residual": val_final["covariance_residual"],
        "test_covariance_residual": test_final["covariance_residual"],
        "parameter_count": test_final["parameter_count"],
        "best_validation_max_h1_abs_error": best,
        "runtime_seconds": time.perf_counter() - start,
    }
    history.extend([train_final, val_final, test_final])
    return model, history, final


def _prediction_rows(models: dict[str, nn.Module], datasets: dict[str, TeacherSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch, model in models.items():
        for split, dataset in datasets.items():
            with torch.no_grad():
                c = _predict_c(model, dataset)
            for idx in range(c.numel()):
                rows.append(
                    {
                        "branch": branch,
                        "split": split,
                        "trajectory_index": idx,
                        "omega": float(dataset.data.omegas[idx].item()),
                        "c_pred": float(c[idx].item()),
                        "c_teacher": float(dataset.c_free[idx].item()),
                        "c_error": float((c[idx] - dataset.c_free[idx]).item()),
                    }
                )
    return rows


def _plot_train_val(history: list[dict[str, Any]]) -> None:
    rows = [r for r in history if str(r["step"]).isdigit() and r["split"] in {"train_monitor", "val"}]
    plt.figure(figsize=(8, 5))
    for branch in sorted({r["branch"] for r in rows}):
        for split, style in [("train_monitor", "-"), ("val", "--")]:
            sub = [r for r in rows if r["branch"] == branch and r["split"] == split]
            plt.plot([int(r["step"]) for r in sub], [float(r["max_h1_abs_error"]) for r in sub], style, label=f"{branch} {split}")
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle=":", linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("maximum H=1 absolute error")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_vs_validation_max_h1_error.png", dpi=170)
    plt.close()


def _plot_pred_vs_teacher(pred_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in pred_rows if r["split"] == "test"]
    plt.figure(figsize=(6, 6))
    for branch in sorted({r["branch"] for r in rows}):
        sub = [r for r in rows if r["branch"] == branch]
        plt.scatter([float(r["c_teacher"]) for r in sub], [float(r["c_pred"]) for r in sub], s=20, alpha=0.85, label=branch)
    vals = [float(r["c_teacher"]) for r in rows] + [float(r["c_pred"]) for r in rows]
    lo, hi = min(vals), max(vals)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("prefix-only free-c teacher")
    plt.ylabel("predicted c")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_ci.png", dpi=170)
    plt.close()


def _plot_cov_comparison(summary: list[dict[str, Any]]) -> None:
    labels = [str(r["branch"]) for r in summary]
    vals = [float(r["test_max_h1_abs_error"]) for r in summary]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(vals)), vals)
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(vals)), labels, rotation=20, ha="right")
    plt.yscale("log")
    plt.ylabel("test maximum H=1 absolute error")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "covariance_vs_no_covariance_comparison.png", dpi=170)
    plt.close()


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    base = next(r for r in summary if r["branch"] == "tcn_h1_loss")
    cov = next(r for r in summary if r["branch"] == "tcn_h1_loss_plus_covariance")
    passed = [r["branch"] for r in summary if r["gate_passed"]]
    best_test = min(summary, key=lambda r: float(r["test_max_h1_abs_error"]))
    return {
        "fresh_trajectory_training_eliminates_fixed_set_memorization": True,
        "any_branch_passed_h1": bool(passed),
        "passed_branches": passed,
        "best_test_branch": best_test["branch"],
        "best_test_max_h1_abs_error": best_test["test_max_h1_abs_error"],
        "covariance_helped_test_h1_error": float(cov["test_max_h1_abs_error"]) < float(base["test_max_h1_abs_error"]),
        "covariance_helped_test_c_rmse": float(cov["test_c_rmse_vs_teacher"]) < float(base["test_c_rmse_vs_teacher"]),
        "recommend_resume_strict_sequential_curriculum": bool(passed),
        "failure_mode": (
            "H=1 passed on validation and test; strict sequential curriculum is now justified"
            if passed
            else "H=1 still fails with fresh trajectories; bottleneck is representation/precision rather than fixed-set memorization"
        ),
    }


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    cols = [
        "branch",
        "uses_covariance",
        "gate_passed",
        "train_monitor_max_h1_abs_error",
        "val_max_h1_abs_error",
        "test_max_h1_abs_error",
        "test_c_rmse_vs_teacher",
        "test_max_abs_c_error_vs_teacher",
        "test_covariance_residual",
        "parameter_count",
    ]
    rows = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in summary)
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>H=1 Fresh-Trajectory Generalization</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B H=1 Fresh-Trajectory Generalization</h1>",
        "<p>The optimizer receives a freshly generated batch of clean position-only prefixes at every step. Fixed train-monitor, validation, and test sets are evaluation only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + rows + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_monitor_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 231000)
    val_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 231100)
    test_pool = _make_data(POOL_N, TEST_DATA_SEED + 231200)
    train_monitor_data = _stratified_subset(train_monitor_pool, TRAIN_MONITOR_N)
    val_data = _stratified_subset(val_pool, VAL_N)
    test_data = _stratified_subset(test_pool, TEST_N)
    print("building prefix-only free-c teachers for diagnostics", flush=True)
    train_monitor, val, test, teacher_history = _make_teacher_sets(train_monitor_data, val_data, test_data)
    models: dict[str, nn.Module] = {}
    all_history: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for cfg in _branch_grid():
        print(f"training {cfg['branch']}", flush=True)
        model, history, final = _train_branch(cfg, train_monitor, val, test)
        models[str(cfg["branch"])] = model
        all_history.extend(history)
        summary.append(final)
    prediction_rows = _prediction_rows(models, {"train_monitor": train_monitor, "val": val, "test": test})
    interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "training_history.csv", all_history)
    _write_csv(OUTPUT_DIR / "teacher_history.csv", teacher_history)
    _write_csv(OUTPUT_DIR / "predictions_vs_teacher.csv", prediction_rows)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_h1_fresh_trajectory_generalization",
            "dtype": str(DTYPE),
            "prefix_length_P": P,
            "fresh_batch_size": BATCH_SIZE,
            "fixed_train_monitor_n": TRAIN_MONITOR_N,
            "val_n": VAL_N,
            "test_n": TEST_N,
            "learning_rate": LR,
            "max_steps": MAX_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "patience": PATIENCE,
            "gate_max_abs_error": GATE_MAX_ABS_ERROR,
            "smoothmax_temperature": SMOOTHMAX_TEMPERATURE,
            "covariance_weight": COV_WEIGHT,
            "uses_superloss": False,
            "uses_velocity": False,
            "uses_labels": False,
            "uses_ls_or_closed_form_estimator": False,
            "uses_future_data_in_training": False,
            "fresh_training_trajectories_every_step": True,
        },
    )
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", {"summary": summary, "training_history": all_history, "predictions_vs_teacher": prediction_rows})
    _plot_train_val(all_history)
    _plot_pred_vs_teacher(prediction_rows)
    _plot_cov_comparison(summary)
    _write_html(summary, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
