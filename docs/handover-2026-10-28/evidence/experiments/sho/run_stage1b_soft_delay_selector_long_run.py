"""Targeted long-run follow-up for the Stage 1B soft delay selector."""

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
import numpy as np
import torch

from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import (
    DelayPriorModel,
    _fixed_delay_rollout,
    _param_distance,
    _param_snapshot,
    _rollout_metrics,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _evaluate_raw_window_ls,
    _make_data,
    _prefix_pairs,
    _rollout_matrix,
    _rollout_truth,
    _spectral_radius,
    _write_csv,
    _write_json,
)
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/soft_delay_selector_long_run")
M = 8
TRAIN_NUM = 8
TEST_NUM = 6
LAMBDA_RECON = 1e-1
LEARNING_RATE = 1e-3
EVAL_EVERY = 500
MIN_DELTA = 1e-14

CONFIGS = [
    {"config": "A_previous_best", "lambda_entropy": 1e-2, "mu": 1.0},
    {"config": "B_no_entropy_mu1", "lambda_entropy": 0.0, "mu": 1.0},
    {"config": "C_entropy_no_bias", "lambda_entropy": 1e-2, "mu": 0.0},
]

STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
STAGE_BUDGETS: dict[int | str, dict[str, int]] = {
    1: {"max_steps": 5000, "patience": 1000},
    2: {"max_steps": 5000, "patience": 1000},
    4: {"max_steps": 5000, "patience": 1000},
    8: {"max_steps": 5000, "patience": 1000},
    16: {"max_steps": 10000, "patience": 2000},
    32: {"max_steps": 10000, "patience": 2000},
    64: {"max_steps": 10000, "patience": 2000},
    "full_10_period": {"max_steps": 20000, "patience": 5000},
}


def _entropy_and_pi(model: DelayPriorModel) -> tuple[torch.Tensor, torch.Tensor]:
    pi = torch.softmax(model.alpha, dim=0)
    entropy = -torch.sum(pi * torch.log(pi.clamp_min(1e-30)))
    return entropy, pi


def _loss_parts(model: DelayPriorModel, positions: torch.Tensor, stage: int | str) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode_x(model.encode(q_in)) - q_in[..., -1]) ** 2)
    entropy, _ = _entropy_and_pi(model)
    loss = rollout_loss + LAMBDA_RECON * recon_loss + model.lambda_entropy * entropy
    return loss, {
        "rollout_loss": float(rollout_loss.detach().item()),
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }, k_h


def _evaluate_split(
    model: DelayPriorModel,
    positions: torch.Tensor,
    stage: int | str,
    raw_mse: float,
    fixed_mse: float,
) -> dict[str, Any]:
    with torch.no_grad():
        _, parts, k_h = _loss_parts(model, positions, stage)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _rollout_metrics(pred_x, truth)
        entropy, pi_t = _entropy_and_pi(model)
        pi = pi_t.detach().cpu().numpy()
        selected = int(np.argmax(pi)) + 1
    return {
        **metrics,
        **parts,
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
        "gap_to_fixed_delay_floor": metrics["rollout_mse"] - fixed_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "pi_values": json.dumps(pi.tolist()),
        "selected_lag": selected,
        "max_pi": float(np.max(pi)),
        "entropy": float(entropy.item()),
    }


def _is_still_decreasing(loss_trace: list[float], patience: int) -> bool:
    if len(loss_trace) < max(20, patience):
        return True
    recent = np.array(loss_trace[-patience:], dtype=float)
    x = np.arange(recent.shape[0], dtype=float)
    slope = float(np.polyfit(x, np.log(np.maximum(recent, 1e-300)), 1)[0])
    return slope < -1e-5


def _train_config(
    cfg: dict[str, Any],
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    raw_mse: float,
    fixed_mse: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(SEED)
    model = DelayPriorModel(M, "soft_delay_selector", "uniform_alpha", lambda_entropy=float(cfg["lambda_entropy"])).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    previous_best_snapshot: list[torch.Tensor] | None = None
    global_step = 0
    best_mse = float("inf")
    final_row: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    pi_rows: list[dict[str, Any]] = []

    for stage in STAGES:
        budget = STAGE_BUDGETS[stage]
        best_stage_loss = float("inf")
        best_stage_step = 0
        best_stage_snapshot = _param_snapshot(model)
        no_improve = 0
        loss_trace: list[float] = []
        stage_start_step = global_step
        stage_stopped_early = False
        for local_step in range(1, budget["max_steps"] + 1):
            global_step += 1
            optimizer.zero_grad()
            stage_loss, train_parts, _ = _loss_parts(model, train_positions, stage)
            drift = _param_distance(model, previous_best_snapshot)
            bias = float(cfg["mu"]) * drift if previous_best_snapshot is not None else torch.zeros((), dtype=DTYPE)
            loss = stage_loss + bias
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
            stage_loss_value = float(stage_loss.detach().item())
            loss_trace.append(stage_loss_value)
            if stage_loss_value < best_stage_loss - MIN_DELTA:
                best_stage_loss = stage_loss_value
                best_stage_step = local_step
                best_stage_snapshot = _param_snapshot(model)
                no_improve = 0
            else:
                no_improve += 1
            if local_step % EVAL_EVERY == 0 or local_step == budget["max_steps"]:
                test_eval = _evaluate_split(model, test_positions, stage, raw_mse, fixed_mse)
                best_mse = min(best_mse, test_eval["rollout_mse"])
                entropy, pi_t = _entropy_and_pi(model)
                pi = pi_t.detach().cpu().numpy()
                history.append({
                    "config": cfg["config"],
                    "stage": stage,
                    "global_step": global_step,
                    "local_step": local_step,
                    "train_rollout_mse": train_parts["rollout_loss"],
                    "test_rollout_mse": test_eval["rollout_mse"],
                    "best_rollout_mse": best_mse,
                    "one_step_mse": test_eval["one_step_mse"],
                    "two_step_mse": test_eval["two_step_mse"],
                    "current_reconstruction_mse": test_eval["current_reconstruction_mse"],
                    "entropy": test_eval["entropy"],
                    "selected_lag": test_eval["selected_lag"],
                    "max_pi": test_eval["max_pi"],
                    "latent_transition_mse": test_eval["latent_transition_mse"],
                    "ls_condition_number": test_eval["ls_condition_number"],
                    "spectral_radius_k_h": test_eval["spectral_radius_k_h"],
                    "max_abs_trajectory_error": test_eval["max_abs_trajectory_error"],
                    "final_time_abs_error": test_eval["final_time_abs_error"],
                    "biased_regularization_penalty": float(bias.detach().item()),
                    "parameter_drift_from_previous_stage": float(drift.detach().item()),
                })
                pi_rows.append({
                    "config": cfg["config"],
                    "stage": stage,
                    "global_step": global_step,
                    "local_step": local_step,
                    "entropy": float(entropy.detach().item()),
                    "selected_lag": int(np.argmax(pi)) + 1,
                    "max_pi": float(np.max(pi)),
                    **{f"pi_lag_{idx + 1}": float(value) for idx, value in enumerate(pi)},
                })
            if no_improve >= budget["patience"] and not _is_still_decreasing(loss_trace, min(len(loss_trace), budget["patience"])):
                stage_stopped_early = True
                break
        previous_best_snapshot = best_stage_snapshot
        stage_eval = _evaluate_split(model, test_positions, stage, raw_mse, fixed_mse)
        stage_eval.update({
            "config": cfg["config"],
            "lambda_entropy": cfg["lambda_entropy"],
            "mu": cfg["mu"],
            "stage": stage,
            "stage_start_step": stage_start_step,
            "stage_end_step": global_step,
            "stage_steps_run": global_step - stage_start_step,
            "stage_best_train_loss": best_stage_loss,
            "stage_best_step": best_stage_step,
            "stage_stopped_early": stage_stopped_early,
            "stage_still_decreasing": _is_still_decreasing(loss_trace, min(len(loss_trace), max(10, budget["patience"]))),
            "best_rollout_mse": best_mse,
        })
        final_row = stage_eval
    return final_row, history, pi_rows


def _plot_lines(rows: list[dict[str, Any]], ykey: str, filename: str, ylabel: str, xkey: str = "global_step") -> None:
    plt.figure(figsize=(9, 5))
    for cfg in CONFIGS:
        selected = [r for r in rows if r["config"] == cfg["config"]]
        if selected:
            plt.plot([float(r[xkey]) for r in selected], [float(r[ykey]) for r in selected], marker="o", markersize=3, label=cfg["config"])
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_stage(rows: list[dict[str, Any]]) -> None:
    stage_index = {stage: idx for idx, stage in enumerate(STAGES)}
    plt.figure(figsize=(9, 5))
    for cfg in CONFIGS:
        selected = [r for r in rows if r["config"] == cfg["config"]]
        if selected:
            plt.plot([stage_index[r["stage"]] for r in selected], [float(r["rollout_mse"]) for r in selected], marker="o", label=cfg["config"])
    plt.yscale("log")
    plt.xticks(range(len(STAGES)), [str(s) for s in STAGES], rotation=30)
    plt.xlabel("stage")
    plt.ylabel("final stage test rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_vs_stage.png", dpi=160)
    plt.close()


def _plot_pi(pi_rows: list[dict[str, Any]]) -> None:
    for cfg in CONFIGS:
        selected = [r for r in pi_rows if r["config"] == cfg["config"]]
        if not selected:
            continue
        lags = [f"pi_lag_{i}" for i in range(1, M)]
        plt.figure(figsize=(9, 5))
        for lag in lags:
            plt.plot([float(r["global_step"]) for r in selected], [float(r[lag]) for r in selected], label=lag)
        plt.xlabel("global_step")
        plt.ylabel("pi")
        plt.title(cfg["config"])
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        safe = cfg["config"].lower()
        plt.savefig(OUTPUT_DIR / f"pi_values_by_stage_{safe}.png", dpi=160)
        plt.close()
    if pi_rows:
        last_rows = {}
        for row in pi_rows:
            last_rows[row["config"]] = row
        labels = list(last_rows)
        matrix = np.array([[float(last_rows[label][f"pi_lag_{i}"]) for i in range(1, M)] for label in labels])
        plt.figure(figsize=(8, 3))
        plt.imshow(matrix, aspect="auto", cmap="viridis")
        plt.yticks(range(len(labels)), labels)
        plt.xticks(range(M - 1), [str(i) for i in range(1, M)])
        plt.colorbar(label="pi")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pi_values_by_stage.png", dpi=160)
        plt.close()


def _plot_comparison(summary_rows: list[dict[str, Any]], raw_mse: float, fixed_mse: float) -> None:
    previous_path = Path("outputs/sho/stage1B/learned_delay_coordinate_prior_diagnostic/summary.csv")
    prev_best = None
    if previous_path.exists():
        csv.field_size_limit(sys.maxsize)
        with previous_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if (
                    row.get("model_type") == "soft_delay_selector"
                    and row.get("m") == "8"
                    and row.get("variant") == "rollout_recon"
                    and row.get("init_kind") == "uniform_alpha"
                    and row.get("lambda_entropy") == "0.01"
                    and row.get("mu") == "1.0"
                ):
                    value = float(row["rollout_mse"])
                    prev_best = value if prev_best is None else min(prev_best, value)
    labels = ["raw LS", "fixed delay", "previous best", *[r["config"] for r in summary_rows]]
    vals = [raw_mse, fixed_mse, prev_best if prev_best is not None else np.nan, *[float(r["rollout_mse"]) for r in summary_rows]]
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(labels)), vals)
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel("10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_to_previous_runtime_adjusted_result.png", dpi=160)
    plt.close()


def _write_html(summary_rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    cols = [
        "config", "lambda_entropy", "mu", "stage", "rollout_mse", "best_rollout_mse",
        "one_step_mse", "two_step_mse", "current_reconstruction_mse", "entropy",
        "selected_lag", "max_pi", "latent_transition_mse", "ls_condition_number",
        "stage_steps_run", "stage_still_decreasing", "stage_stopped_early",
    ]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in cols) + "</tr>" for row in summary_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Soft Delay Selector Long Run</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}</style></head><body>
<h1>Stage 1B Soft Delay Selector Long Run</h1>
<p>Targeted continuation of the best learned delay-coordinate prior branch. K_h is identified by latent LS; no labels, velocity, analytic recurrence/decoder, raw-window teacher, or MLP-generated K are used.</p>
<table><thead><tr>{''.join(f'<th>{escape(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid"><img src="rollout_mse_vs_stage.png"><img src="rollout_mse_vs_training_steps.png"><img src="entropy_vs_stage.png"><img src="pi_values_by_stage.png"><img src="biased_regularization_drift.png"><img src="comparison_to_previous_runtime_adjusted_result.png"></div>
<pre>{escape(json.dumps(setup, indent=2))}</pre></body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    raw = _evaluate_raw_window_ls(test_positions, M)
    fixed = _fixed_delay_rollout(test_positions, M)
    raw_mse = float(raw["rollout_mse"])
    fixed_mse = float(fixed["rollout_mse"])
    setup = {
        "diagnostic": "stage1B_soft_delay_selector_long_run",
        "configs": CONFIGS,
        "m": M,
        "dtype": str(DTYPE),
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "lambda_recon": LAMBDA_RECON,
        "learning_rate": LEARNING_RATE,
        "eval_every": EVAL_EVERY,
        "stage_budgets": {str(k): v for k, v in STAGE_BUDGETS.items()},
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    summary_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    pi_rows: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        print(f"training {cfg['config']}", flush=True)
        final, history, pi = _train_config(cfg, train_positions, test_positions, raw_mse, fixed_mse)
        summary_rows.append(final)
        history_rows.extend(history)
        pi_rows.extend(pi)
        _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
        _write_csv(OUTPUT_DIR / "full_run_table.csv", summary_rows)
        _write_csv(OUTPUT_DIR / "stage_history.csv", history_rows)
        _write_csv(OUTPUT_DIR / "pi_values_by_stage.csv", pi_rows)
    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows, "stage_history": history_rows, "pi_values": pi_rows})
    _plot_stage(summary_rows)
    _plot_lines(history_rows, "test_rollout_mse", "rollout_mse_vs_training_steps.png", "test rollout MSE")
    _plot_lines(history_rows, "entropy", "entropy_vs_stage.png", "entropy", xkey="global_step")
    _plot_lines(history_rows, "parameter_drift_from_previous_stage", "biased_regularization_drift.png", "parameter drift")
    _plot_pi(pi_rows)
    _plot_comparison(summary_rows, raw_mse, fixed_mse)
    _write_html(summary_rows, setup)


if __name__ == "__main__":
    main()
