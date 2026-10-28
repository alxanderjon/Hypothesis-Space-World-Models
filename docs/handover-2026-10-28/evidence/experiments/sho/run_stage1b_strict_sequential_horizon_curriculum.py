"""Stage 1B strict sequential-horizon curriculum diagnostic.

This diagnostic tests whether a prefix encoder can master one rollout horizon
before advancing to the next. It intentionally avoids SuperLoss and avoids
training all horizons at once.

Training uses only clean position-prefix data. Prefix-only free-c teachers are
computed for diagnostics only.
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    P,
    TeacherSet,
    _make_data,
    _make_teacher_sets,
    _r2,
)
from experiments.sho.run_stage1b_covariance_superloss_encoder_diagnostic import _covariance_loss_vector
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import (
    CInitTCN,
    _param_count,
    _predict_c,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/strict_sequential_horizon_curriculum")
HORIZONS = [1, 2, 4, 8, 16, 32, 64]
TRAIN_N = 32
VAL_N = 16
TEST_N = 16
POOL_N = 128
LR = float(os.environ.get("SSHC_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("SSHC_MAX_STEPS", "20000"))
EVAL_INTERVAL = int(os.environ.get("SSHC_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("SSHC_PATIENCE", "3000"))
GATE_MAX_ABS_ERROR = float(os.environ.get("SSHC_GATE_MAX_ABS_ERROR", "1e-6"))
COV_WEIGHT = 0.1


def _stratified_subset(data: Any, n: int) -> Any:
    order = torch.argsort(data.omegas)
    if n == data.omegas.numel():
        idx = order
    else:
        positions = torch.linspace(0, data.omegas.numel() - 1, n, dtype=torch.float64)
        idx = order[positions.round().to(torch.long)]
    return type(data)(positions=data.positions[idx], omegas=data.omegas[idx])


def _max_error_tensor(c: torch.Tensor, positions: torch.Tensor, horizon: int) -> torch.Tensor:
    x = positions[:, :P]
    starts = torch.arange(1, P - horizon, device=x.device)
    a = x[:, starts - 1]
    b = x[:, starts]
    errs = []
    for step in range(1, horizon + 1):
        nxt = c[:, None] * b - a
        truth = x[:, starts + step]
        errs.append(nxt - truth)
        a, b = b, nxt
    return torch.stack(errs, dim=-1)


def _horizon_max_squared_loss(c: torch.Tensor, positions: torch.Tensor, horizon: int) -> torch.Tensor:
    err = _max_error_tensor(c, positions, horizon)
    return torch.max(err**2)


@torch.no_grad()
def _horizon_max_abs_error(c: torch.Tensor, positions: torch.Tensor, horizon: int) -> float:
    return float(torch.max(torch.abs(_max_error_tensor(c, positions, horizon))).item())


@torch.no_grad()
def _worst_10_period_error(c: torch.Tensor, dataset: TeacherSet) -> tuple[float, float, float]:
    # 10-period future trajectory is diagnostics only.
    from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import _steps

    steps = _steps(10.0)
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
    mse_by_traj = torch.mean((pred - truth) ** 2, dim=1)
    return (
        float(mse_by_traj.max().item()),
        float(mse_by_traj.mean().item()),
        float(torch.max(torch.abs(pred - truth)).item()),
    )


def _eval_model(
    model: CInitTCN,
    dataset: TeacherSet,
    branch: str,
    stage_horizon: int,
    split: str,
    mastered: list[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        c_err = c - dataset.c_free
        cov = _covariance_loss_vector(c, dataset.data.positions)
        worst10, mean10, max_abs10 = _worst_10_period_error(c, dataset)
    row: dict[str, Any] = {
        "branch": branch,
        "split": split,
        "stage_horizon": stage_horizon,
        "mastered_horizons": json.dumps(mastered),
        "stage_max_abs_error": _horizon_max_abs_error(c, dataset.data.positions, stage_horizon),
        "stage_max_squared_error": float(_horizon_max_squared_loss(c, dataset.data.positions, stage_horizon).item()),
        "max_abs_c_error": float(torch.max(torch.abs(c_err)).item()),
        "c_rmse": float(torch.sqrt(torch.mean(c_err**2)).item()),
        "c_r2_against_teacher": _r2(c[:, None], dataset.c_free[:, None]),
        "collapse_ratio": float((torch.std(c) / torch.std(dataset.c_free).clamp_min(1e-12)).item()),
        "covariance_residual": float(cov.mean().item()),
        "covariance_residual_max": float(cov.max().item()),
        "worst_10period_rollout_mse": worst10,
        "mean_10period_rollout_mse": mean10,
        "max_abs_10period_rollout_error": max_abs10,
        "parameter_count": _param_count(model),
    }
    for h in HORIZONS:
        row[f"horizon_{h}_max_abs_error"] = _horizon_max_abs_error(c, dataset.data.positions, h)
    if extra:
        row.update(extra)
    return row


def _gate_passes(row: dict[str, Any], mastered: list[int]) -> tuple[bool, str]:
    current = float(row["stage_max_abs_error"])
    if current > GATE_MAX_ABS_ERROR:
        return False, f"stage max abs error {current:.6g} > {GATE_MAX_ABS_ERROR:g}"
    for h in mastered:
        val = float(row[f"horizon_{h}_max_abs_error"])
        if val > GATE_MAX_ABS_ERROR:
            return False, f"previous horizon {h} regressed to {val:.6g} > {GATE_MAX_ABS_ERROR:g}"
    return True, "passed"


def _train_branch(branch: str, train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    use_cov = branch == "sequential_rollout_plus_covariance"
    model = CInitTCN().to(dtype=DTYPE)
    mastered: list[int] = []
    history: list[dict[str, Any]] = []
    branch_start = time.perf_counter()
    stop_reason = "completed all stages"
    stopped_at: int | None = None
    for stage_index, horizon in enumerate(HORIZONS):
        print(f"  {branch} / H={horizon}", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best_state = copy.deepcopy(model.state_dict())
        best_score = float("inf")
        stale = 0
        stage_start = time.perf_counter()
        for step in range(1, MAX_STEPS + 1):
            c = _predict_c(model, train)
            loss = _horizon_max_squared_loss(c, train.data.positions, horizon)
            cov_max = torch.max(_covariance_loss_vector(c, train.data.positions))
            if use_cov:
                loss = loss + COV_WEIGHT * cov_max
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                row = _eval_model(
                    model,
                    val,
                    branch,
                    horizon,
                    "val",
                    mastered,
                    {
                        "step": step,
                        "stage_index": stage_index,
                        "train_loss": float(loss.detach().item()),
                        "train_stage_max_abs_error": _horizon_max_abs_error(c.detach(), train.data.positions, horizon),
                        "train_covariance_residual_max": float(cov_max.detach().item()),
                        "stage_elapsed_seconds": time.perf_counter() - stage_start,
                        "branch_elapsed_seconds": time.perf_counter() - branch_start,
                        "uses_covariance": use_cov,
                    },
                )
                history.append(row)
                score = float(row["stage_max_abs_error"])
                if score < best_score:
                    best_score = score
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += EVAL_INTERVAL
                if stale >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        gate_row = _eval_model(
            model,
            val,
            branch,
            horizon,
            "val",
            mastered,
            {
                "step": "best",
                "stage_index": stage_index,
                "best_validation_stage_max_abs_error": best_score,
                "uses_covariance": use_cov,
                "gate_threshold": GATE_MAX_ABS_ERROR,
            },
        )
        ok, reason = _gate_passes(gate_row, mastered)
        gate_row["gate_passed"] = ok
        gate_row["gate_reason"] = reason
        gate_row["steps_required"] = min(
            [int(r["step"]) for r in history if r["branch"] == branch and int(r.get("stage_index", -1)) == stage_index and float(r["stage_max_abs_error"]) <= GATE_MAX_ABS_ERROR],
            default="not_passed",
        )
        history.append(gate_row)
        if not ok:
            stop_reason = reason
            stopped_at = horizon
            break
        mastered.append(horizon)
    final_horizon = mastered[-1] if mastered else HORIZONS[0]
    summary = _eval_model(
        model,
        test,
        branch,
        final_horizon,
        "test",
        mastered,
        {
            "completed_all_stages": len(mastered) == len(HORIZONS),
            "stopped_at_horizon": stopped_at if stopped_at is not None else "",
            "stop_reason": stop_reason,
            "uses_covariance": use_cov,
            "num_mastered_horizons": len(mastered),
            "passed_horizons": json.dumps(mastered),
            "total_runtime_seconds": time.perf_counter() - branch_start,
        },
    )
    return history, summary


def _interpret(summaries: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in summaries:
        branch = str(row["branch"])
        out[f"{branch}_passed_horizons"] = row["passed_horizons"]
        out[f"{branch}_stopped_at_horizon"] = row["stopped_at_horizon"]
        out[f"{branch}_stop_reason"] = row["stop_reason"]
        out[f"{branch}_test_worst_10period_mse"] = row["worst_10period_rollout_mse"]
        out[f"{branch}_test_c_rmse"] = row["c_rmse"]
    rollout = next(r for r in summaries if r["branch"] == "sequential_rollout_only")
    cov = next(r for r in summaries if r["branch"] == "sequential_rollout_plus_covariance")
    out["covariance_helped_pass_more_stages"] = int(cov["num_mastered_horizons"]) > int(rollout["num_mastered_horizons"])
    out["covariance_lower_test_c_rmse"] = float(cov["c_rmse"]) < float(rollout["c_rmse"])
    out["stage1A_like_assessment"] = (
        "yes" if all(bool(r["completed_all_stages"]) for r in summaries) else "no: at least one branch failed the strict gate before completing all horizons"
    )
    gates = [r for r in history if r.get("step") == "best"]
    out["gate_results"] = [
        {
            "branch": r["branch"],
            "horizon": r["stage_horizon"],
            "passed": r["gate_passed"],
            "reason": r["gate_reason"],
            "max_abs_error": r["stage_max_abs_error"],
            "max_abs_c_error": r["max_abs_c_error"],
        }
        for r in gates
    ]
    return out


def _plot_stage_progress(history: list[dict[str, Any]]) -> None:
    gate_rows = [r for r in history if r.get("step") == "best"]
    plt.figure(figsize=(8, 5))
    for branch in sorted({r["branch"] for r in gate_rows}):
        rows = [r for r in gate_rows if r["branch"] == branch]
        plt.plot([int(r["stage_horizon"]) for r in rows], [float(r["stage_max_abs_error"]) for r in rows], marker="o", label=branch)
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle="--", linewidth=1)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("curriculum horizon")
    plt.ylabel("validation max abs rollout error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "maximum_rollout_error_vs_curriculum_stage.png", dpi=170)
    plt.close()


def _plot_c_error(history: list[dict[str, Any]]) -> None:
    gate_rows = [r for r in history if r.get("step") == "best"]
    plt.figure(figsize=(8, 5))
    for branch in sorted({r["branch"] for r in gate_rows}):
        rows = [r for r in gate_rows if r["branch"] == branch]
        plt.plot([int(r["stage_horizon"]) for r in rows], [float(r["max_abs_c_error"]) for r in rows], marker="o", label=branch)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("curriculum horizon")
    plt.ylabel("validation max |c_pred - c_teacher|")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "maximum_ci_error_vs_curriculum_stage.png", dpi=170)
    plt.close()


def _plot_stage_progression(summaries: list[dict[str, Any]]) -> None:
    labels = [r["branch"] for r in summaries]
    vals = [int(r["num_mastered_horizons"]) for r in summaries]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=20, ha="right")
    plt.ylabel("number of gated horizons passed")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_only_vs_covariance_stage_progression.png", dpi=170)
    plt.close()


def _write_html(summaries: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    cols = [
        "branch",
        "passed_horizons",
        "stopped_at_horizon",
        "stop_reason",
        "c_rmse",
        "max_abs_c_error",
        "covariance_residual",
        "worst_10period_rollout_mse",
        "num_mastered_horizons",
    ]
    rows = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in summaries)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Strict Sequential Horizon Curriculum</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Strict Sequential-Horizon Curriculum</h1>",
        "<p>Full-batch TCN encoder training on stratified small clean prefixes. Each horizon must pass max absolute validation error <= 1e-6 before advancing.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + rows + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED)
    train_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 201000)
    val_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 201100)
    test_pool = _make_data(POOL_N, TEST_DATA_SEED + 201200)
    train_data = _stratified_subset(train_pool, TRAIN_N)
    val_data = _stratified_subset(val_pool, VAL_N)
    test_data = _stratified_subset(test_pool, TEST_N)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    all_history: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for branch in ["sequential_rollout_only", "sequential_rollout_plus_covariance"]:
        print(f"training {branch}", flush=True)
        history, summary = _train_branch(branch, train, val, test)
        all_history.extend(history)
        summaries.append(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", summaries)
    _write_csv(OUTPUT_DIR / "stage_history.csv", all_history)
    _write_csv(OUTPUT_DIR / "teacher_history.csv", teacher_history)
    interpretation = _interpret(summaries, all_history)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_strict_sequential_horizon_curriculum",
            "dtype": str(DTYPE),
            "prefix_length_P": P,
            "train_n": TRAIN_N,
            "val_n": VAL_N,
            "test_n": TEST_N,
            "stratified_subset": True,
            "horizons": HORIZONS,
            "learning_rate": LR,
            "max_steps_per_stage": MAX_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "patience": PATIENCE,
            "gate_max_abs_error": GATE_MAX_ABS_ERROR,
            "covariance_weight": COV_WEIGHT,
            "uses_superloss": False,
            "trains_all_horizons_simultaneously": False,
            "uses_velocity": False,
            "uses_labels": False,
            "uses_LS_or_closed_form_estimator": False,
            "uses_future_data_in_training": False,
        },
    )
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", {"summary": summaries, "stage_history": all_history})
    _plot_stage_progress(all_history)
    _plot_c_error(all_history)
    _plot_stage_progression(summaries)
    _write_html(summaries, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
