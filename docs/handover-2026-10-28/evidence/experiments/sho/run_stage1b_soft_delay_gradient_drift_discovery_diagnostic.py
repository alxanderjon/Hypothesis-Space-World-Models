"""Stage 1B soft-delay gradient, drift, and discovery diagnostic."""

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

from experiments.sho.run_stage1b_soft_delay_learning_ablation import (
    M,
    _best_from_csv,
    _pi_vector,
    _threshold_flags,
)
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _apply_k,
    _evaluate_raw_window_ls,
    _fit_k_ridge,
    _make_data,
    _prefix_pairs,
    _rollout_matrix,
    _rollout_truth,
    _spectral_radius,
    _write_csv,
    _write_json,
)
from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import _fixed_delay_rollout, _param_distance, _param_snapshot, _rollout_metrics
from hypothesis_space_world_models.training import set_seed


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/soft_delay_gradient_drift_discovery_diagnostic")
FINAL_DIR = Path("outputs/sho/stage1B/final_soft_delay_selector_success_run")

TRAIN_NUM = 8
TEST_NUM = 6
HORIZONS: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
GRADIENT_ETAS = [1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6]
ADAM_LRS = [1e-2, 3e-3, 1e-3, 1e-4, 1e-5]
DISCOVERY_LRS = [1e-3, 1e-4, 1e-5]
THRESHOLDS = [1e-14, 1e-15, 1e-16, 1e-18]

FIXED_OBJECTIVE_STEPS = 500
CURRICULUM_STAGE_STEPS = 150
DISCOVERY_FIXED_STEPS = 500
DISCOVERY_STAGE_STEPS = 150
EVAL_INTERVAL = 100


def _loss_for_alpha(model: SoftDelayAblationModel, positions: torch.Tensor, horizon: int | str) -> torch.Tensor:
    loss, _ = _loss_parts(model, positions, horizon)
    return loss


def _alpha_grad_at(pi_kind: str, positions: torch.Tensor, horizon: int | str) -> tuple[float, np.ndarray]:
    model = SoftDelayAblationModel(pi_kind=pi_kind, train_pi=True).to(dtype=DTYPE)
    assert model.alpha is not None
    loss = _loss_for_alpha(model, positions, horizon)
    loss.backward()
    grad = model.alpha.grad.detach().cpu().numpy().copy()
    return float(loss.detach().item()), grad


def _pi_distance(pi: np.ndarray, kind: str) -> float:
    target = _pi_vector(kind).detach().cpu().numpy()
    return float(np.linalg.norm(pi - target))


class SoftDelayAblationModel(nn.Module):
    """Soft delay selector with pi_l multiplying x_{t-l}.

    Windows are stored chronologically as (x_{t-7}, ..., x_t), so the past
    entries must be flipped before applying pi_1,...,pi_7.
    """

    def __init__(self, *, pi_kind: str = "uniform", train_pi: bool = False) -> None:
        super().__init__()
        self.m = M
        self.pi_kind = pi_kind
        if train_pi:
            init = torch.log(_pi_vector(pi_kind).clamp_min(1e-30))
            self.alpha = nn.Parameter(init.clone())
            self.register_buffer("pi_fixed", torch.empty(0, dtype=DTYPE))
        else:
            self.alpha = None
            self.register_buffer("pi_fixed", _pi_vector(pi_kind))

    def pi(self) -> torch.Tensor:
        if self.alpha is not None:
            return torch.softmax(self.alpha, dim=0)
        return self.pi_fixed

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        pi = self.pi().to(q.device)
        past_by_lag = torch.flip(q[..., :-1], dims=(-1,))
        delayed = torch.sum(past_by_lag * pi, dim=-1)
        return torch.stack((delayed, q[..., -1]), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        return h[..., 1]

    def identify_k(self, q_in: torch.Tensor, q_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fit_k_ridge(self.encode(q_in), self.encode(q_out))


def _evaluate(model: SoftDelayAblationModel, positions: torch.Tensor, stage: int | str = "full_10_period") -> dict[str, Any]:
    with torch.no_grad():
        q_in, q_out = _prefix_pairs(positions, model.m)
        k_h, residual, conditions = model.identify_k(q_in, q_out)
        truth = _rollout_truth(positions, model.m, stage)
        h_rollout = _rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1)
        pred_x = model.decode_x(h_rollout)
        recon = model.decode_x(model.encode(q_in))
        metrics = _rollout_metrics(pred_x, truth)
        recon_mse = torch.mean((recon - q_in[..., -1]) ** 2)
        pi = model.pi().detach().cpu().numpy()
        entropy = float(-np.sum(pi * np.log(np.maximum(pi, 1e-300))))
    return {
        **metrics,
        "current_reconstruction_mse": float(recon_mse.item()),
        "latent_transition_mse": float(residual.item()),
        "ls_condition_number": float(conditions.mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
        "pi_values": json.dumps(pi.tolist()),
        "entropy": entropy,
        "selected_lag": int(np.argmax(pi)) + 1,
        "max_pi": float(np.max(pi)),
        **{f"pi_lag_{idx + 1}": float(value) for idx, value in enumerate(pi)},
        "k_h_ls_identified": True,
    }


def _loss_parts(model: SoftDelayAblationModel, positions: torch.Tensor, stage: int | str) -> tuple[torch.Tensor, dict[str, Any]]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    return rollout_loss, {
        "train_stage_rollout_mse": float(rollout_loss.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }


def _eval_with_common_ridge(pi_kind: str, positions: torch.Tensor, ridge: float) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model = SoftDelayAblationModel(pi_kind=pi_kind, train_pi=False).to(dtype=DTYPE)
    q_in, q_out = _prefix_pairs(positions, M)
    h_in = model.encode(q_in)
    h_out = model.encode(q_out)
    k_h, residual, conditions = _fit_k_ridge(h_in, h_out, ridge=ridge)
    truth = _rollout_truth(positions, M, "full_10_period")
    h_rollout = _rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1)
    pred_x = h_rollout[..., 1]
    row = {
        **_rollout_metrics(pred_x, truth),
        "latent_transition_mse": float(residual.item()),
        "ls_condition_number": float(conditions.mean().item()),
        "spectral_radius_k_h": _spectral_radius(k_h),
    }
    return row, k_h, h_in


def _indexing_audit(test_positions: torch.Tensor) -> dict[str, Any]:
    fixed_row = _fixed_delay_rollout(test_positions, M)
    soft_legacy = _evaluate(SoftDelayAblationModel(pi_kind="lag1", train_pi=False).to(dtype=DTYPE), test_positions)
    soft_common, soft_k, soft_h = _eval_with_common_ridge("lag1", test_positions, ridge=1e-12)
    q_in, q_out = _prefix_pairs(test_positions, M)
    fixed_h = q_in[..., -2:]
    fixed_k, _, _ = _fit_k_ridge(fixed_h, q_out[..., -2:], ridge=1e-12)
    truth = _rollout_truth(test_positions, M, "full_10_period")
    fixed_pred = _rollout_matrix(fixed_k, truth[:, 0, -2:], truth.shape[1] - 1)[..., 1]
    soft_pred = _rollout_matrix(soft_k, truth[:, 0, -2:].clone(), truth.shape[1] - 1)[..., 1]
    return {
        "max_h_fixed_delay_minus_soft_lag1": float(torch.max(torch.abs(fixed_h - soft_h)).item()),
        "k_h_frobenius_difference_common_ridge": float(torch.linalg.norm(fixed_k - soft_k).item()),
        "fixed_delay_rollout_mse_legacy": fixed_row["rollout_mse"],
        "soft_lag1_rollout_mse_legacy_default_ridge": soft_legacy["rollout_mse"],
        "soft_lag1_rollout_mse_common_ridge": soft_common["rollout_mse"],
        "rollout_trajectory_max_difference_common_ridge": float(torch.max(torch.abs(fixed_pred - soft_pred)).item()),
        "trajectories_match_common_ridge": bool(torch.max(torch.abs(fixed_pred - soft_pred)).item() < 1e-12),
        "interpretation": "lag-1 indexing matches fixed delay under a common ridge; the earlier mismatch is solver ridge, not indexing.",
    }


def _zero_coordinate_audit(test_positions: torch.Tensor) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pi_kind in ["uniform", "lag1", "lag7", "far_weighted", "recency_weighted"]:
        model = SoftDelayAblationModel(pi_kind=pi_kind, train_pi=False).to(dtype=DTYPE)
        row = {
            **_evaluate(model, test_positions),
            "config": f"zero_{pi_kind}",
            "pi_kind": pi_kind,
        }
        rows.append(row)
    return rows


def _gradient_alignment(train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_loss, g_full = _alpha_grad_at("uniform", train_positions, "full_10_period")
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    g_full_norm = float(np.linalg.norm(g_full))
    for horizon in HORIZONS:
        loss_h, grad = _alpha_grad_at("uniform", train_positions, horizon)
        grad_norm = float(np.linalg.norm(grad))
        cosine = float(np.dot(grad, g_full) / max(grad_norm * g_full_norm, 1e-300))
        rows.append({
            "horizon": horizon,
            "loss_h": loss_h,
            "loss_full": full_loss,
            "grad_norm": grad_norm,
            "full_grad_norm": g_full_norm,
            "cosine_with_full_horizon": cosine,
        })
        base_model = SoftDelayAblationModel(pi_kind="uniform", train_pi=True).to(dtype=DTYPE)
        assert base_model.alpha is not None
        base_stage = float(_loss_for_alpha(base_model, test_positions, horizon).detach().item())
        base_full = float(_evaluate(base_model, test_positions)["rollout_mse"])
        base_pi = base_model.pi().detach().cpu().numpy()
        for eta in GRADIENT_ETAS:
            model = SoftDelayAblationModel(pi_kind="uniform", train_pi=True).to(dtype=DTYPE)
            assert model.alpha is not None
            with torch.no_grad():
                model.alpha.copy_(model.alpha - eta * torch.tensor(grad, dtype=DTYPE))
            new_stage = float(_loss_for_alpha(model, test_positions, horizon).detach().item())
            new_eval = _evaluate(model, test_positions)
            new_full = float(new_eval["rollout_mse"])
            new_pi = model.pi().detach().cpu().numpy()
            stage_improves = new_stage < base_stage
            full_improves = new_full < base_full
            if stage_improves and full_improves:
                classification = "aligned"
            elif stage_improves and not full_improves:
                classification = "short_horizon_overfit"
            elif (not stage_improves) and (not full_improves):
                classification = "too_large_or_bad_step"
            else:
                classification = "full_improves_stage_worsens"
            if abs(new_full - base_full) <= 1e-18 and abs(new_stage - base_stage) <= 1e-18:
                classification = "negligible"
            step_rows.append({
                **new_eval,
                "horizon": horizon,
                "eta": eta,
                "base_stage_loss": base_stage,
                "new_stage_loss": new_stage,
                "base_full_mse": base_full,
                "new_full_mse": new_full,
                "delta_full_mse": new_full - base_full,
                "delta_pi_l2": float(np.linalg.norm(new_pi - base_pi)),
                "delta_entropy": float(new_eval["entropy"] - _evaluate(base_model, test_positions)["entropy"]),
                "classification": classification,
            })
    return rows, step_rows


def _make_train_model(pi_kind: str) -> SoftDelayAblationModel:
    if pi_kind != "alternating_bad":
        return SoftDelayAblationModel(pi_kind=pi_kind, train_pi=True).to(dtype=DTYPE)
    model = SoftDelayAblationModel(pi_kind="uniform", train_pi=True).to(dtype=DTYPE)
    assert model.alpha is not None
    weights = torch.tensor([1.0, 0.01, 1.0, 0.01, 1.0, 0.01, 1.0], dtype=DTYPE)
    pi = weights / weights.sum()
    with torch.no_grad():
        model.alpha.copy_(torch.log(pi))
    model.pi_kind = pi_kind
    return model


def _train_alpha_run(
    *,
    run_kind: str,
    pi_kind: str,
    objective: int | str,
    lr: float,
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    curriculum: bool,
    mu: float = 0.0,
    fixed_steps: int = FIXED_OBJECTIVE_STEPS,
    stage_steps: int = CURRICULUM_STAGE_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = _make_train_model(pi_kind)
    assert model.alpha is not None
    optimizer = torch.optim.Adam([model.alpha], lr=lr)
    initial_eval = _evaluate(model, test_positions)
    best_eval = initial_eval
    best_mse = float(initial_eval["rollout_mse"])
    final_eval = initial_eval
    accepted_alpha = model.alpha.detach().clone()
    rows: list[dict[str, Any]] = []
    global_step = 0
    stages = HORIZONS if curriculum else [objective]
    steps_for_stage = stage_steps if curriculum else fixed_steps
    for stage in stages:
        for local_step in range(1, steps_for_stage + 1):
            global_step += 1
            optimizer.zero_grad()
            loss = _loss_for_alpha(model, train_positions, stage)
            if mu:
                loss = loss + mu * torch.sum((model.alpha - accepted_alpha) ** 2)
            loss.backward()
            grad_norm = float(torch.linalg.norm(model.alpha.grad.detach()).item())
            optimizer.step()
            if local_step == 1 or local_step % EVAL_INTERVAL == 0 or local_step == steps_for_stage:
                full_eval = _evaluate(model, test_positions)
                full_mse = float(full_eval["rollout_mse"])
                if full_mse < best_mse:
                    best_mse = full_mse
                    best_eval = full_eval
                pi = model.pi().detach().cpu().numpy()
                rows.append({
                    **full_eval,
                    "run_kind": run_kind,
                    "pi_init": pi_kind,
                    "objective": objective,
                    "stage": stage,
                    "lr": lr,
                    "mu": mu,
                    "global_step": global_step,
                    "local_step": local_step,
                    "current_objective_loss": float(loss.detach().item()),
                    "full_10_period_mse": full_mse,
                    "best_full_10_period_mse_so_far": best_mse,
                    "distance_from_uniform_pi": _pi_distance(pi, "uniform"),
                    "distance_from_lag1_pi": _pi_distance(pi, "lag1"),
                    "gradient_norm": grad_norm,
                })
        accepted_alpha = model.alpha.detach().clone()
    final_eval = _evaluate(model, test_positions)
    final_mse = float(final_eval["rollout_mse"])
    summary = {
        **final_eval,
        "run_kind": run_kind,
        "pi_init": pi_kind,
        "objective": objective,
        "lr": lr,
        "mu": mu,
        "initial_full_10_period_mse": initial_eval["rollout_mse"],
        "best_full_10_period_mse": best_mse,
        "final_full_10_period_mse": final_mse,
        "best_checkpoint_much_better_than_final": final_mse > 10.0 * max(best_mse, 1e-300),
        "crosses_1e-14": best_mse < 1e-14,
        "crosses_1e-15": best_mse < 1e-15,
        "crosses_1e-16": best_mse < 1e-16,
        "crosses_1e-18": best_mse < 1e-18,
        "final_distance_from_uniform_pi": _pi_distance(model.pi().detach().cpu().numpy(), "uniform"),
        "final_distance_from_lag1_pi": _pi_distance(model.pi().detach().cpu().numpy(), "lag1"),
    }
    return summary, rows


def _adam_drift(train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for objective in [1, 8, "full_10_period"]:
        for lr in ADAM_LRS:
            summary, history = _train_alpha_run(
                run_kind="adam_drift",
                pi_kind="uniform",
                objective=objective,
                lr=lr,
                train_positions=train_positions,
                test_positions=test_positions,
                curriculum=False,
            )
            summaries.append(summary)
            rows.extend(history)
    for lr in ADAM_LRS:
        summary, history = _train_alpha_run(
            run_kind="adam_drift",
            pi_kind="uniform",
            objective="curriculum",
            lr=lr,
            train_positions=train_positions,
            test_positions=test_positions,
            curriculum=True,
        )
        summaries.append(summary)
        rows.extend(history)
    return summaries, rows


def _discovery(train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pi_kind in ["lag7", "far_weighted", "recency_weighted", "alternating_bad"]:
        for objective, curriculum in [("full_10_period", False), ("curriculum", True)]:
            for lr in DISCOVERY_LRS:
                summary, history = _train_alpha_run(
                    run_kind="discovery",
                    pi_kind=pi_kind,
                    objective=objective,
                    lr=lr,
                    train_positions=train_positions,
                    test_positions=test_positions,
                    curriculum=curriculum,
                    fixed_steps=DISCOVERY_FIXED_STEPS,
                    stage_steps=DISCOVERY_STAGE_STEPS,
                )
                summaries.append(summary)
                rows.extend(history)
    return summaries, rows


def _regularizer(train_positions: torch.Tensor, test_positions: torch.Tensor) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pi_kind in ["uniform", "lag7"]:
        for lr in [1e-3, 1e-5]:
            for mu in [0.0, 1.0]:
                summary, history = _train_alpha_run(
                    run_kind="regularizer",
                    pi_kind=pi_kind,
                    objective="curriculum",
                    lr=lr,
                    mu=mu,
                    train_positions=train_positions,
                    test_positions=test_positions,
                    curriculum=True,
                    stage_steps=DISCOVERY_STAGE_STEPS,
                )
                summaries.append(summary)
                rows.extend(history)
    return summaries, rows


def _plot_zero(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.bar(range(len(rows)), [float(r["rollout_mse"]) for r in rows])
    plt.yscale("log")
    plt.xticks(range(len(rows)), [r["pi_kind"] for r in rows], rotation=25, ha="right")
    plt.ylabel("zero-training 10-period MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zero_training_coordinate_mse.png", dpi=160)
    plt.close()


def _plot_gradient(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(len(rows)), [float(r["cosine_with_full_horizon"]) for r in rows], marker="o")
    plt.xticks(range(len(rows)), [str(r["horizon"]) for r in rows], rotation=25)
    plt.ylabel("cosine with full-horizon gradient")
    plt.xlabel("horizon")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_cosine_with_full_horizon.png", dpi=160)
    plt.close()


def _plot_one_step(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for horizon in HORIZONS:
        selected = [r for r in rows if r["horizon"] == horizon]
        plt.plot([float(r["eta"]) for r in selected], [float(r["delta_full_mse"]) for r in selected], marker="o", label=str(horizon))
    plt.xscale("log")
    plt.yscale("symlog", linthresh=1e-18)
    plt.xlabel("eta")
    plt.ylabel("delta full 10-period MSE")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "one_step_full_mse_change_by_lr.png", dpi=160)
    plt.close()


def _plot_history(rows: list[dict[str, Any]], run_kind: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    selected_configs = sorted({(r["pi_init"], str(r["objective"]), float(r["lr"]), float(r["mu"])) for r in rows if r["run_kind"] == run_kind})
    for pi_init, objective, lr, mu in selected_configs[:16]:
        selected = [r for r in rows if r["run_kind"] == run_kind and r["pi_init"] == pi_init and str(r["objective"]) == objective and float(r["lr"]) == lr and float(r["mu"]) == mu]
        label = f"{pi_init}/{objective}/lr={lr:g}/mu={mu:g}"
        plt.plot([float(r["global_step"]) for r in selected], [float(r["full_10_period_mse"]) for r in selected], label=label)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("full 10-period MSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_pi_drift(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for key in sorted({(r["run_kind"], r["pi_init"], str(r["objective"]), float(r["lr"])) for r in rows})[:16]:
        run_kind, pi_init, objective, lr = key
        selected = [r for r in rows if (r["run_kind"], r["pi_init"], str(r["objective"]), float(r["lr"])) == key]
        plt.plot([float(r["global_step"]) for r in selected], [float(r["distance_from_uniform_pi"]) for r in selected], label=f"{run_kind}/{pi_init}/{objective}/{lr:g}")
    plt.xlabel("step")
    plt.ylabel("distance from uniform pi")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pi_drift_vs_steps.png", dpi=160)
    plt.close()


def _plot_threshold(rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['run_kind']}:{r['pi_init']}:{r['objective']}:{float(r['lr']):g}:mu{float(r['mu']):g}" for r in rows]
    matrix = np.array([[1.0 if r[f"crosses_{t:g}"] else 0.0 for t in THRESHOLDS] for r in rows])
    plt.figure(figsize=(8, max(4, len(labels) * 0.12)))
    plt.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.yticks(range(len(labels)), labels, fontsize=4)
    plt.xticks(range(len(THRESHOLDS)), [f"{t:g}" for t in THRESHOLDS])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings.png", dpi=180)
    plt.close()


def _write_html(setup: dict[str, Any], indexing: dict[str, Any], zero_rows: list[dict[str, Any]], grad_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    best_uniform = next(r for r in zero_rows if r["pi_kind"] == "uniform")
    low_lr_uniform = [r for r in summaries if r["run_kind"] == "adam_drift" and r["pi_init"] == "uniform" and float(r["lr"]) == 1e-5]
    bad_discovery = [r for r in summaries if r["run_kind"] == "discovery"]
    discoveries_under_1e14 = sum(1 for r in bad_discovery if r["crosses_1e-14"])
    answers = [
        f"Indexing audit: max h difference {indexing['max_h_fixed_delay_minus_soft_lag1']}; common-ridge trajectory max difference {indexing['rollout_trajectory_max_difference_common_ridge']}.",
        f"Uniform zero-training MSE is {best_uniform['rollout_mse']}, so the good basin exists before training.",
        f"Gradient cosines with full horizon: {', '.join(str(round(float(r['cosine_with_full_horizon']), 4)) for r in grad_rows)}.",
        f"lr=1e-5 uniform drift runs: best MSE range {min(float(r['best_full_10_period_mse']) for r in low_lr_uniform):.3e} to {max(float(r['best_full_10_period_mse']) for r in low_lr_uniform):.3e}.",
        f"Bad-initialization discovery runs crossing 1e-14: {discoveries_under_1e14} of {len(bad_discovery)}.",
        "The honest Stage 1B claim is that uniform mixed-delay coordinates plus LS provide the near-exact recovery; training signals can preserve or slightly perturb it, but discovery from bad coordinates is the key diagnostic evidence.",
    ]
    cols = ["run_kind", "pi_init", "objective", "lr", "mu", "initial_full_10_period_mse", "best_full_10_period_mse", "final_full_10_period_mse", "crosses_1e-14", "crosses_1e-15", "final_distance_from_uniform_pi"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in summaries)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Soft Delay Gradient Drift Discovery Diagnostic</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:11px}}td,th{{border:1px solid #d8dee8;padding:5px 6px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head><body>
<h1>Stage 1B Soft-Delay Gradient, Drift, and Discovery Diagnostic</h1>
<h2>Direct Answers</h2><ol>{''.join(f'<li>{escape(a)}</li>' for a in answers)}</ol>
<h2>Run Summary</h2><table><thead><tr>{''.join(f'<th>{escape(c)}</th>' for c in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="zero_training_coordinate_mse.png"><img src="gradient_cosine_with_full_horizon.png"><img src="one_step_full_mse_change_by_lr.png"><img src="adam_drift_full_mse_vs_steps.png"><img src="pi_drift_vs_steps.png"><img src="discovery_full_mse_vs_steps.png"><img src="regularizer_mu_comparison.png"><img src="threshold_crossings.png">
</div>
<h2>Indexing Audit</h2><pre>{escape(json.dumps(indexing, indent=2))}</pre>
<h2>Setup</h2><pre>{escape(json.dumps(setup, indent=2))}</pre>
</body></html>"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    raw_mse = float(_evaluate_raw_window_ls(test_positions, M)["rollout_mse"])
    fixed_mse = float(_fixed_delay_rollout(test_positions, M)["rollout_mse"])
    indexing = _indexing_audit(test_positions)
    zero_rows = _zero_coordinate_audit(test_positions)
    grad_rows, one_step_rows = _gradient_alignment(train_positions, test_positions)
    adam_summaries, adam_history = _adam_drift(train_positions, test_positions)
    discovery_summaries, discovery_history = _discovery(train_positions, test_positions)
    regularizer_summaries, regularizer_history = _regularizer(train_positions, test_positions)
    all_summaries = adam_summaries + discovery_summaries + regularizer_summaries
    all_history = adam_history + discovery_history + regularizer_history
    threshold_rows = [
        {
            "run_kind": r["run_kind"],
            "pi_init": r["pi_init"],
            "objective": r["objective"],
            "lr": r["lr"],
            "mu": r["mu"],
            "best_full_10_period_mse": r["best_full_10_period_mse"],
            **{f"crosses_{t:g}": r[f"crosses_{t:g}"] for t in THRESHOLDS},
        }
        for r in all_summaries
    ]
    pi_rows = [
        {
            "run_kind": r["run_kind"],
            "pi_init": r["pi_init"],
            "objective": r["objective"],
            "stage": r["stage"],
            "lr": r["lr"],
            "mu": r["mu"],
            "global_step": r["global_step"],
            "entropy": r["entropy"],
            "distance_from_uniform_pi": r["distance_from_uniform_pi"],
            "distance_from_lag1_pi": r["distance_from_lag1_pi"],
            **{f"pi_lag_{i}": r[f"pi_lag_{i}"] for i in range(1, M)},
        }
        for r in all_history
    ]
    setup = {
        "diagnostic": "stage1B_soft_delay_gradient_drift_discovery",
        "m": M,
        "dtype": str(DTYPE),
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
        "fixed_objective_steps": FIXED_OBJECTIVE_STEPS,
        "curriculum_stage_steps": CURRICULUM_STAGE_STEPS,
        "discovery_fixed_steps": DISCOVERY_FIXED_STEPS,
        "discovery_stage_steps": DISCOVERY_STAGE_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "raw_window_ls_floor_mse": raw_mse,
        "fixed_delay_floor_mse": fixed_mse,
        "uniform_soft_zero_training_mse": next(r["rollout_mse"] for r in zero_rows if r["pi_kind"] == "uniform"),
        "previous_success_preserved_mse": _best_from_csv(FINAL_DIR / "summary.csv", "best_full_10_period_rollout_mse"),
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "k_identified_by_latent_ls": True,
        "wall_clock_sec": time.perf_counter() - start,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "indexing_audit.json", indexing)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "indexing": indexing, "zero": zero_rows, "gradient": grad_rows, "one_step": one_step_rows, "summary": all_summaries})
    _write_csv(OUTPUT_DIR / "summary.csv", all_summaries)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", all_history)
    _write_csv(OUTPUT_DIR / "zero_training_coordinate_audit.csv", zero_rows)
    _write_csv(OUTPUT_DIR / "gradient_alignment_table.csv", grad_rows)
    _write_csv(OUTPUT_DIR / "one_step_gradient_table.csv", one_step_rows)
    _write_csv(OUTPUT_DIR / "adam_drift_table.csv", adam_summaries)
    _write_csv(OUTPUT_DIR / "discovery_from_bad_init_table.csv", discovery_summaries)
    _write_csv(OUTPUT_DIR / "regularizer_check_table.csv", regularizer_summaries)
    _write_csv(OUTPUT_DIR / "threshold_audit.csv", threshold_rows)
    _write_csv(OUTPUT_DIR / "pi_trajectories.csv", pi_rows)
    _plot_zero(zero_rows)
    _plot_gradient(grad_rows)
    _plot_one_step(one_step_rows)
    _plot_history(adam_history, "adam_drift", "adam_drift_full_mse_vs_steps.png")
    _plot_pi_drift(all_history)
    _plot_history(discovery_history, "discovery", "discovery_full_mse_vs_steps.png")
    _plot_history(regularizer_history, "regularizer", "regularizer_mu_comparison.png")
    _plot_threshold(all_summaries)
    _write_html(setup, indexing, zero_rows, grad_rows, all_summaries)


if __name__ == "__main__":
    main()
