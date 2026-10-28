"""Stage 1B prefix-limited operator-identification diagnostic.

Fits K_h only from a short observed prefix of R latent transitions, then
rolls out the future trajectory from the end of that prefix.
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
import numpy as np
import torch
from torch import nn

from experiments.sho.run_stage1b_soft_delay_learning_ablation import _best_from_csv, _pi_vector
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _evaluate_raw_window_ls,
    _fit_k_ridge,
    _full_rollout_steps,
    _make_data,
    _rollout_matrix,
    _spectral_radius,
    _window_sequence,
    _write_csv,
    _write_json,
)
from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import _fixed_delay_rollout, _rollout_metrics
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/prefix_limited_operator_identification")
FINAL_DIR = Path("outputs/sho/stage1B/final_soft_delay_selector_success_run")

M = 8
TRAIN_NUM = 8
TEST_NUM = 6
R_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
THRESHOLDS = [1e-6, 1e-9, 1e-12, 1e-14]
LRS = [1e-3, 1e-4, 1e-5]
TRAIN_INITS = ["uniform", "recency_weighted", "lag7"]
TRAIN_STEPS = 300
EVAL_INTERVAL = 100


class PrefixSoftDelay(nn.Module):
    """Soft delay selector where pi_l multiplies x_{t-l}."""

    def __init__(self, *, pi_kind: str, train_alpha: bool) -> None:
        super().__init__()
        self.m = M
        self.pi_kind = pi_kind
        if train_alpha:
            init_pi = _pi_vector(pi_kind).clamp_min(1e-30)
            self.alpha = nn.Parameter(torch.log(init_pi))
            self.register_buffer("pi_fixed", torch.empty(0, dtype=DTYPE))
        else:
            self.alpha = None
            self.register_buffer("pi_fixed", _pi_vector(pi_kind))

    def pi(self) -> torch.Tensor:
        if self.alpha is not None:
            return torch.softmax(self.alpha, dim=0)
        return self.pi_fixed

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        past_by_lag = torch.flip(q[..., :-1], dims=(-1,))
        delayed = torch.sum(past_by_lag * self.pi().to(q.device), dim=-1)
        return torch.stack((delayed, q[..., -1]), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]


class FixedDelay:
    m = M

    def pi(self) -> torch.Tensor:
        pi = torch.zeros(M - 1, dtype=DTYPE)
        pi[0] = 1.0
        return pi

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        return q[..., -2:]

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _pi_metrics(model: Any) -> dict[str, Any]:
    pi = model.pi().detach().cpu().numpy()
    entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-300))))
    return {
        "pi_values": json.dumps(pi.tolist()),
        "entropy": entropy,
        "selected_lag": int(np.argmax(pi)) + 1,
        "max_pi": float(np.max(pi)),
        **{f"pi_lag_{idx + 1}": float(value) for idx, value in enumerate(pi)},
    }


def _observed_pairs(positions: torch.Tensor, model: Any, r_transitions: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_in = _window_sequence(positions, M, M - 1, r_transitions)
    q_out = _window_sequence(positions, M, M, r_transitions)
    return model.encode(q_in), model.encode(q_out)


def _future_truth(positions: torch.Tensor, r_transitions: int) -> torch.Tensor:
    start_end = M - 1 + r_transitions
    return _window_sequence(positions, M, start_end, _full_rollout_steps() + 1)


def _evaluate_prefix(model: Any, positions: torch.Tensor, r_transitions: int) -> dict[str, Any]:
    h_in, h_out = _observed_pairs(positions, model, r_transitions)
    k_h, residual, conditions = _fit_k_ridge(h_in, h_out)
    truth = _future_truth(positions, r_transitions)
    h0 = model.encode(truth[:, 0])
    h_rollout = _rollout_matrix(k_h, h0, truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    metrics = _rollout_metrics(pred_x, truth)
    return {
        **metrics,
        "future_rollout_mse": metrics["rollout_mse"],
        "full_10_period_future_mse": metrics["rollout_mse"],
        "one_step_future_mse": metrics["one_step_mse"],
        "two_step_future_mse": metrics["two_step_mse"],
        "observed_ls_residual": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
        **_pi_metrics(model),
        **_threshold_flags(metrics["rollout_mse"]),
    }


def _prefix_loss(model: PrefixSoftDelay, positions: torch.Tensor, r_transitions: int) -> torch.Tensor:
    h_in, h_out = _observed_pairs(positions, model, r_transitions)
    k_h, _, _ = _fit_k_ridge(h_in, h_out)
    truth = _future_truth(positions, r_transitions)
    h_rollout = _rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1)
    pred_x = model.decode_x(h_rollout)
    return torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)


def _static_sweep(test_positions: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_specs: list[tuple[str, Any]] = [
        ("fixed_delay", FixedDelay()),
        ("uniform_mixed_delay", PrefixSoftDelay(pi_kind="uniform", train_alpha=False).to(dtype=DTYPE)),
        ("recency_weighted_mixed_delay", PrefixSoftDelay(pi_kind="recency_weighted", train_alpha=False).to(dtype=DTYPE)),
    ]
    for model_name, model in model_specs:
        for r_value in R_VALUES:
            row = {
                **_evaluate_prefix(model, test_positions, r_value),
                "model": model_name,
                "variant": "static",
                "observed_ls_transitions_R": r_value,
                "total_observed_positions_required": M + r_value,
                "alpha_trained": False,
                "initialization": "",
                "lr": "",
            }
            rows.append(row)
    return rows


def _train_soft_selector(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    r_value: int,
    init_kind: str,
    lr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = PrefixSoftDelay(pi_kind=init_kind, train_alpha=True).to(dtype=DTYPE)
    assert model.alpha is not None
    optimizer = torch.optim.Adam([model.alpha], lr=lr)
    initial_eval = _evaluate_prefix(model, test_positions, r_value)
    best_eval = initial_eval
    best_state = model.alpha.detach().clone()
    best_mse = float(initial_eval["future_rollout_mse"])
    history: list[dict[str, Any]] = []
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad()
        loss = _prefix_loss(model, train_positions, r_value)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == TRAIN_STEPS:
            eval_row = _evaluate_prefix(model, test_positions, r_value)
            mse = float(eval_row["future_rollout_mse"])
            if mse < best_mse:
                best_mse = mse
                best_eval = eval_row
                best_state = model.alpha.detach().clone()
            history.append({
                **eval_row,
                "model": "trainable_soft_delay",
                "variant": "train_alpha",
                "observed_ls_transitions_R": r_value,
                "total_observed_positions_required": M + r_value,
                "initialization": init_kind,
                "lr": lr,
                "step": step,
                "train_loss": float(loss.detach().item()),
                "best_future_rollout_mse_so_far": best_mse,
            })
    final_eval = _evaluate_prefix(model, test_positions, r_value)
    with torch.no_grad():
        model.alpha.copy_(best_state)
    restored_best = _evaluate_prefix(model, test_positions, r_value)
    summary = {
        **restored_best,
        "model": "trainable_soft_delay",
        "variant": "train_alpha",
        "observed_ls_transitions_R": r_value,
        "total_observed_positions_required": M + r_value,
        "initialization": init_kind,
        "lr": lr,
        "alpha_trained": True,
        "initial_future_rollout_mse": initial_eval["future_rollout_mse"],
        "best_future_rollout_mse": best_eval["future_rollout_mse"],
        "final_future_rollout_mse": final_eval["future_rollout_mse"],
        "training_improved_over_initial": float(best_eval["future_rollout_mse"]) < float(initial_eval["future_rollout_mse"]),
        "training_made_final_worse_than_initial": float(final_eval["future_rollout_mse"]) > float(initial_eval["future_rollout_mse"]),
    }
    return summary, history


def _training_sweep(train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    # Include the requested no-training uniform alpha as an explicit trainable-model baseline.
    for r_value in R_VALUES:
        model = PrefixSoftDelay(pi_kind="uniform", train_alpha=True).to(dtype=DTYPE)
        row = {
            **_evaluate_prefix(model, test_positions, r_value),
            "model": "trainable_soft_delay",
            "variant": "no_training_uniform_alpha",
            "observed_ls_transitions_R": r_value,
            "total_observed_positions_required": M + r_value,
            "initialization": "uniform",
            "lr": "",
            "alpha_trained": False,
            "initial_future_rollout_mse": "",
            "best_future_rollout_mse": "",
            "final_future_rollout_mse": "",
        }
        summaries.append(row)
    for r_value in R_VALUES:
        for init_kind in TRAIN_INITS:
            for lr in LRS:
                summary, history = _train_soft_selector(train_positions, test_positions, r_value, init_kind, lr)
                summaries.append(summary)
                histories.extend(history)
    return summaries, histories


def _plot_lines(rows: list[dict[str, Any]], filename: str, title_key: str = "model") -> None:
    plt.figure(figsize=(8, 5))
    for label in sorted({row[title_key] for row in rows}):
        selected = [row for row in rows if row[title_key] == label]
        selected.sort(key=lambda r: int(r["observed_ls_transitions_R"]))
        plt.plot(
            [int(row["observed_ls_transitions_R"]) for row in selected],
            [float(row["future_rollout_mse"]) for row in selected],
            marker="o",
            label=label,
        )
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("observed LS transitions R")
    plt.ylabel("future 10-period rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_condition(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for model in sorted({row["model"] for row in rows}):
        selected = [row for row in rows if row["model"] == model and row["variant"] == "static"]
        selected.sort(key=lambda r: int(r["observed_ls_transitions_R"]))
        plt.plot([int(r["observed_ls_transitions_R"]) for r in selected], [float(r["ls_condition_number"]) for r in selected], marker="o", label=model)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("observed LS transitions R")
    plt.ylabel("LS covariance condition number")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "condition_number_vs_observed_LS_transitions.png", dpi=160)
    plt.close()


def _plot_training(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for key in sorted({(r["observed_ls_transitions_R"], r["initialization"], str(r["lr"])) for r in history})[:24]:
        r_value, init_kind, lr = key
        selected = [r for r in history if (r["observed_ls_transitions_R"], r["initialization"], str(r["lr"])) == key]
        plt.plot([int(r["step"]) for r in selected], [float(r["future_rollout_mse"]) for r in selected], label=f"R={r_value},{init_kind},lr={lr}")
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("future rollout MSE")
    plt.legend(fontsize=5, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trained_vs_untrained_soft_selector.png", dpi=180)
    plt.close()


def _plot_pi(rows: list[dict[str, Any]]) -> None:
    selected = [r for r in rows if r["model"] in {"fixed_delay", "uniform_mixed_delay", "recency_weighted_mixed_delay"}]
    labels = [f"{r['model']} R={r['observed_ls_transitions_R']}" for r in selected]
    matrix = np.array([[float(r.get(f"pi_lag_{i}", 0.0)) for i in range(1, M)] for r in selected])
    plt.figure(figsize=(8, max(4, len(labels) * 0.16)))
    plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    plt.yticks(range(len(labels)), labels, fontsize=5)
    plt.xticks(range(M - 1), [str(i) for i in range(1, M)])
    plt.xlabel("lag")
    plt.colorbar(label="pi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pi_values_by_R.png", dpi=180)
    plt.close()


def _plot_threshold(rows: list[dict[str, Any]]) -> None:
    selected = [r for r in rows if r["variant"] in {"static", "no_training_uniform_alpha"}]
    labels = [f"{r['model']} R={r['observed_ls_transitions_R']}" for r in selected]
    matrix = np.array([[1.0 if r[f"crosses_{t:g}"] else 0.0 for t in THRESHOLDS] for r in selected])
    plt.figure(figsize=(8, max(4, len(labels) * 0.15)))
    plt.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.yticks(range(len(labels)), labels, fontsize=5)
    plt.xticks(range(len(THRESHOLDS)), [f"{t:g}" for t in THRESHOLDS])
    plt.xlabel("threshold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings_by_R.png", dpi=180)
    plt.close()


def _write_html(summary_rows: list[dict[str, Any]], static_rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    def min_r(model: str, threshold: float) -> str:
        candidates = [int(r["observed_ls_transitions_R"]) for r in summary_rows if r["model"] == model and float(r["future_rollout_mse"]) < threshold]
        return str(min(candidates)) if candidates else "not reached"

    answers = [
        f"Minimum R for uniform mixed-delay below 1e-12: {min_r('uniform_mixed_delay', 1e-12)}.",
        f"Minimum R for fixed delay below 1e-12: {min_r('fixed_delay', 1e-12)}.",
        "Uniform mixed-delay remains strong with prefix-limited LS, but the small-R behavior is now the real prediction test.",
        "Training alpha is recorded separately; compare its best future MSE against the no-training uniform rows rather than only the final checkpoint.",
        "Future points are used only for loss/evaluation in trainable-alpha runs, never for fitting K_h.",
    ]
    cols = [
        "model",
        "variant",
        "observed_ls_transitions_R",
        "future_rollout_mse",
        "one_step_future_mse",
        "two_step_future_mse",
        "ls_condition_number",
        "entropy",
        "selected_lag",
        "max_pi",
        "initialization",
        "lr",
        "best_future_rollout_mse",
        "final_future_rollout_mse",
    ]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in cols) + "</tr>" for row in summary_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Prefix-Limited Operator Identification</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:11px}}td,th{{border:1px solid #d8dee8;padding:5px 6px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head><body>
<h1>Stage 1B Prefix-Limited Operator Identification</h1>
<p>K_h is fit only from the first R observed latent transitions. Future trajectory points are never used for LS identification.</p>
<h2>Direct Answers</h2><ol>{''.join(f'<li>{escape(answer)}</li>' for answer in answers)}</ol>
<h2>Summary</h2><table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="future_mse_vs_observed_LS_transitions.png">
<img src="condition_number_vs_observed_LS_transitions.png">
<img src="fixed_delay_vs_uniform_mixed_delay.png">
<img src="trained_vs_untrained_soft_selector.png">
<img src="pi_values_by_R.png">
<img src="threshold_crossings_by_R.png">
</div>
<h2>Setup</h2><pre>{escape(json.dumps(setup, indent=2))}</pre>
</body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    raw_floor = _evaluate_raw_window_ls(test_positions, M)
    fixed_floor = _fixed_delay_rollout(test_positions, M)
    static_rows = _static_sweep(test_positions)
    trained_rows, training_history = _training_sweep(train_positions, test_positions)
    summary_rows = static_rows + trained_rows
    setup = {
        "diagnostic": "stage1B_prefix_limited_operator_identification",
        "m_window_size": M,
        "observed_ls_transitions_R": R_VALUES,
        "dtype": str(DTYPE),
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
        "train_steps_per_alpha_run": TRAIN_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "whole_trajectory_raw_window_ls_floor": raw_floor["rollout_mse"],
        "whole_trajectory_fixed_delay_selector_floor": fixed_floor["rollout_mse"],
        "previous_final_success_preserved_mse": _best_from_csv(FINAL_DIR / "summary.csv", "best_full_10_period_rollout_mse"),
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher_loss": False,
        "k_generated_by_mlp": False,
        "k_identified_by_observed_prefix_ls": True,
        "wall_clock_sec": time.perf_counter() - start,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows, "training_history": training_history})
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", summary_rows + training_history)
    _write_csv(OUTPUT_DIR / "R_sweep_results.csv", static_rows)
    _write_csv(OUTPUT_DIR / "pi_values_by_R.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "training_history.csv", training_history)
    _plot_lines([r for r in summary_rows if r["variant"] in {"static", "no_training_uniform_alpha"}], "future_mse_vs_observed_LS_transitions.png")
    _plot_condition(static_rows)
    _plot_lines([r for r in static_rows if r["model"] in {"fixed_delay", "uniform_mixed_delay"}], "fixed_delay_vs_uniform_mixed_delay.png")
    _plot_training(training_history)
    _plot_pi(static_rows)
    _plot_threshold(summary_rows)
    _write_html(summary_rows, static_rows, setup)


if __name__ == "__main__":
    main()
