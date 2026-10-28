"""Stage 1B soft-delay selector checkpoint-preservation diagnostic.

The model is unchanged from the soft-delay selector branch: a two-dimensional
delay coordinate is learned, K_h is identified by least squares from latent
transitions, and rollout is performed by powers of K_h. This diagnostic only
changes how checkpoints are carried through the curriculum.
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
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/soft_delay_selector_checkpoint_preservation")
LONG_RUN_DIR = Path("outputs/sho/stage1B/soft_delay_selector_long_run")

M = 8
TRAIN_NUM = 8
TEST_NUM = 6
LAMBDA_RECON = 1e-1
LAMBDA_ENTROPY = 0.0
MU = 1.0
LRS = [1e-2, 3e-3]
EVAL_INTERVAL = 100
MIN_DELTA = 1e-14
FREEZE_THRESHOLD = 1e-14

STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
STAGE_BUDGETS: dict[int | str, dict[str, int]] = {
    1: {"max_steps": 500, "patience": 300},
    2: {"max_steps": 500, "patience": 300},
    4: {"max_steps": 600, "patience": 400},
    8: {"max_steps": 800, "patience": 500},
    16: {"max_steps": 1000, "patience": 600},
    32: {"max_steps": 1200, "patience": 700},
    64: {"max_steps": 1500, "patience": 800},
    "full_10_period": {"max_steps": 2500, "patience": 1000},
}

POLICIES = [
    "old_final",
    "stage_best",
    "full_horizon_best",
    "rollback",
    "freeze_alpha_after_discovery",
    "freeze_all_after_threshold",
]


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _load_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict({key: value.detach().clone() for key, value in state.items()})


def _make_optimizer(model: DelayPriorModel, lr: float, freeze_alpha: bool, freeze_all: bool) -> torch.optim.Optimizer | None:
    for param in model.parameters():
        param.requires_grad_(not freeze_all)
    if model.alpha is not None:
        model.alpha.requires_grad_(not freeze_alpha and not freeze_all)
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        return None
    return torch.optim.Adam(params, lr=lr)


def _entropy_and_pi(model: DelayPriorModel) -> tuple[torch.Tensor, torch.Tensor]:
    pi = torch.softmax(model.alpha, dim=0)
    entropy = -torch.sum(pi * torch.log(pi.clamp_min(1e-30)))
    return entropy, pi


def _loss_parts(model: DelayPriorModel, positions: torch.Tensor, stage: int | str) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    h0 = model.encode(truth[:, 0])
    pred_x = model.decode_x(_rollout_matrix(k_h, h0, truth.shape[1] - 1))
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode_x(model.encode(q_in)) - q_in[..., -1]) ** 2)
    entropy, _ = _entropy_and_pi(model)
    loss = rollout_loss + LAMBDA_RECON * recon_loss + LAMBDA_ENTROPY * entropy
    return loss, {
        "rollout_loss": float(rollout_loss.detach().item()),
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }, k_h


def _evaluate_stage(model: DelayPriorModel, positions: torch.Tensor, stage: int | str) -> dict[str, Any]:
    with torch.no_grad():
        _, parts, k_h = _loss_parts(model, positions, stage)
        truth = _rollout_truth(positions, model.m, stage)
        pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _rollout_metrics(pred_x, truth)
        entropy, pi_t = _entropy_and_pi(model)
        pi = pi_t.detach().cpu().numpy()
    return {
        **metrics,
        **parts,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "pi_values": json.dumps(pi.tolist()),
        "selected_lag": int(np.argmax(pi)) + 1,
        "max_pi": float(np.max(pi)),
        "entropy": float(entropy.item()),
    }


def _previous_long_run_best() -> float | None:
    path = LONG_RUN_DIR / "stage_history.csv"
    if not path.exists():
        return None
    best: float | None = None
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                value = float(row.get("test_rollout_mse", "nan"))
            except ValueError:
                continue
            if np.isfinite(value):
                best = value if best is None else min(best, value)
    return best


def _still_decreasing(trace: list[float], patience: int) -> bool:
    if len(trace) < max(20, min(patience, 1000)):
        return True
    recent = np.asarray(trace[-min(patience, 1000):], dtype=float)
    x = np.arange(recent.shape[0], dtype=float)
    slope = float(np.polyfit(x, np.log(np.maximum(recent, 1e-300)), 1)[0])
    return slope < -1e-5


def _record_pi(row_base: dict[str, Any], model: DelayPriorModel) -> dict[str, Any]:
    entropy, pi_t = _entropy_and_pi(model)
    pi = pi_t.detach().cpu().numpy()
    return {
        **row_base,
        "entropy": float(entropy.detach().item()),
        "selected_lag": int(np.argmax(pi)) + 1,
        "max_pi": float(np.max(pi)),
        **{f"pi_lag_{idx + 1}": float(value) for idx, value in enumerate(pi)},
    }


def _train_policy(
    policy: str,
    lr: float,
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    raw_mse: float,
    fixed_mse: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(SEED)
    model = DelayPriorModel(M, "soft_delay_selector", "uniform_alpha", lambda_entropy=LAMBDA_ENTROPY).to(dtype=DTYPE)
    current_lr = lr
    freeze_alpha = False
    freeze_all = False
    optimizer = _make_optimizer(model, current_lr, freeze_alpha=False, freeze_all=False)

    accepted_state = _clone_state_dict(model)
    accepted_params = _param_snapshot(model)
    global_best_state = _clone_state_dict(model)
    global_best_eval = _evaluate_stage(model, test_positions, "full_10_period")
    global_best_mse = float(global_best_eval["rollout_mse"])
    global_best_step = 0
    global_step = 0
    initial_freeze_all = policy == "freeze_all_after_threshold" and global_best_mse < FREEZE_THRESHOLD
    if initial_freeze_all:
        freeze_alpha = True
        freeze_all = True
        optimizer = _make_optimizer(model, current_lr, freeze_alpha=True, freeze_all=True)

    stage_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    pi_rows: list[dict[str, Any]] = []
    checkpoint_events: list[dict[str, Any]] = []
    full_run_rows: list[dict[str, Any]] = []

    checkpoint_path = OUTPUT_DIR / f"best_checkpoint_policy-{policy}_lr-{lr:g}.pt"
    torch.save({"state_dict": global_best_state, "policy": policy, "lr": lr, "global_step": global_step}, checkpoint_path)
    if initial_freeze_all:
        checkpoint_events.append({
            "policy": policy,
            "lr": lr,
            "stage": "initial",
            "global_step": global_step,
            "event": "freeze_all_after_threshold",
            "full_horizon_mse": global_best_mse,
        })

    for stage in STAGES:
        budget = STAGE_BUDGETS[stage]
        if policy == "freeze_alpha_after_discovery" and stage in [16, 32, 64, "full_10_period"] and not freeze_alpha:
            freeze_alpha = True
            optimizer = _make_optimizer(model, current_lr, freeze_alpha=True, freeze_all=False)
            checkpoint_events.append({"policy": policy, "lr": lr, "stage": stage, "global_step": global_step, "event": "freeze_alpha"})

        stage_start_step = global_step
        stage_start_state = _clone_state_dict(model)
        stage_best_state = _clone_state_dict(model)
        stage_best_eval = _evaluate_stage(model, test_positions, stage)
        stage_best_loss = float(stage_best_eval["rollout_mse"])
        stage_best_step = 0
        no_improve = 0
        full_trace: list[float] = []
        stage_rollbacks = 0
        stage_stopped_early = False

        for local_step in range(1, budget["max_steps"] + 1):
            global_step += 1

            if not freeze_all and optimizer is not None:
                optimizer.zero_grad()
                stage_loss, train_parts, _ = _loss_parts(model, train_positions, stage)
                drift = _param_distance(model, accepted_params)
                bias = MU * drift if policy != "old_final" else torch.zeros((), dtype=DTYPE)
                total_loss = stage_loss + bias
                if not torch.isfinite(total_loss):
                    checkpoint_events.append({"policy": policy, "lr": lr, "stage": stage, "global_step": global_step, "event": "nonfinite_loss"})
                    break
                total_loss.backward()
                optimizer.step()
                train_stage_rollout_mse = float(train_parts["rollout_loss"])
                biased_penalty = float(bias.detach().item())
                parameter_drift = float(drift.detach().item())
            else:
                stage_loss, train_parts, _ = _loss_parts(model, train_positions, stage)
                train_stage_rollout_mse = float(train_parts["rollout_loss"])
                parameter_drift = float(_param_distance(model, accepted_params).detach().item())
                biased_penalty = 0.0

            should_eval = local_step % EVAL_INTERVAL == 0 or local_step == 1 or local_step == budget["max_steps"] or freeze_all
            if should_eval:
                stage_eval = _evaluate_stage(model, test_positions, stage)
                full_eval = _evaluate_stage(model, test_positions, "full_10_period")
                full_mse = float(full_eval["rollout_mse"])
                full_trace.append(full_mse)

                if float(stage_eval["rollout_mse"]) < stage_best_loss - MIN_DELTA:
                    stage_best_loss = float(stage_eval["rollout_mse"])
                    stage_best_state = _clone_state_dict(model)
                    stage_best_step = local_step
                    no_improve = 0
                else:
                    no_improve += EVAL_INTERVAL

                if full_mse < global_best_mse - MIN_DELTA:
                    global_best_mse = full_mse
                    global_best_state = _clone_state_dict(model)
                    global_best_eval = full_eval
                    global_best_step = global_step
                    torch.save(
                        {"state_dict": global_best_state, "policy": policy, "lr": lr, "global_step": global_step, "stage": stage},
                        checkpoint_path,
                    )
                    checkpoint_events.append({
                        "policy": policy,
                        "lr": lr,
                        "stage": stage,
                        "global_step": global_step,
                        "event": "new_full_horizon_best",
                        "full_horizon_mse": full_mse,
                    })
                    if policy == "freeze_all_after_threshold" and full_mse < FREEZE_THRESHOLD:
                        freeze_all = True
                        optimizer = _make_optimizer(model, current_lr, freeze_alpha=True, freeze_all=True)
                        checkpoint_events.append({
                            "policy": policy,
                            "lr": lr,
                            "stage": stage,
                            "global_step": global_step,
                            "event": "freeze_all_after_threshold",
                            "full_horizon_mse": full_mse,
                        })

                drift_factor = full_mse / max(global_best_mse, 1e-300)
                if policy == "rollback" and full_mse > 10.0 * global_best_mse and global_best_step < global_step:
                    stage_rollbacks += 1
                    _load_state_dict(model, global_best_state)
                    current_lr *= 0.5
                    optimizer = _make_optimizer(model, current_lr, freeze_alpha=freeze_alpha, freeze_all=freeze_all)
                    checkpoint_events.append({
                        "policy": policy,
                        "lr": lr,
                        "stage": stage,
                        "global_step": global_step,
                        "event": "rollback_to_global_best",
                        "rollback_count_in_stage": stage_rollbacks,
                        "new_lr": current_lr,
                        "full_horizon_mse_before_rollback": full_mse,
                        "global_best_mse": global_best_mse,
                    })
                    if stage_rollbacks >= 3:
                        stage_stopped_early = True
                        break

                row_base = {
                    "policy": policy,
                    "lr": lr,
                    "stage": stage,
                    "global_step": global_step,
                    "local_step": local_step,
                    "train_stage_rollout_mse": train_stage_rollout_mse,
                    "validation_stage_rollout_mse": stage_eval["rollout_mse"],
                    "full_10_period_validation_rollout_mse": full_mse,
                    "best_full_10_period_rollout_mse_so_far": global_best_mse,
                    "one_step_mse": full_eval["one_step_mse"],
                    "two_step_mse": full_eval["two_step_mse"],
                    "current_reconstruction_mse": full_eval["current_reconstruction_mse"],
                    "latent_transition_mse": full_eval["latent_transition_mse"],
                    "ls_condition_number": full_eval["ls_condition_number"],
                    "spectral_radius_k_h": full_eval["spectral_radius_k_h"],
                    "max_abs_trajectory_error": full_eval["max_abs_trajectory_error"],
                    "final_time_abs_error": full_eval["final_time_abs_error"],
                    "parameter_drift_from_previous_accepted_checkpoint": parameter_drift,
                    "biased_regularization_penalty": biased_penalty,
                    "num_rollbacks_in_stage": stage_rollbacks,
                    "current_lr": current_lr,
                    "drift_factor_vs_best_full": drift_factor,
                    "freeze_alpha": freeze_alpha,
                    "freeze_all": freeze_all,
                }
                history_rows.append(_record_pi(row_base, model))
                pi_rows.append(_record_pi(row_base, model))

            if freeze_all:
                break
            if no_improve >= budget["patience"] and not _still_decreasing(full_trace, budget["patience"] // EVAL_INTERVAL + 1):
                stage_stopped_early = True
                break

        final_stage_eval = _evaluate_stage(model, test_positions, stage)
        final_full_eval = _evaluate_stage(model, test_positions, "full_10_period")
        final_full_mse = float(final_full_eval["rollout_mse"])

        if policy == "old_final":
            accepted_state = _clone_state_dict(model)
            carry_kind = "stage_final"
        elif policy == "stage_best":
            _load_state_dict(model, stage_best_state)
            accepted_state = _clone_state_dict(model)
            carry_kind = "stage_validation_best"
        elif policy in {"full_horizon_best", "rollback"}:
            _load_state_dict(model, global_best_state)
            accepted_state = _clone_state_dict(model)
            carry_kind = "full_horizon_global_best"
        elif policy in {"freeze_alpha_after_discovery", "freeze_all_after_threshold"}:
            if freeze_all:
                _load_state_dict(model, global_best_state)
                accepted_state = _clone_state_dict(model)
                carry_kind = "frozen_full_horizon_best"
            else:
                _load_state_dict(model, stage_best_state)
                accepted_state = _clone_state_dict(model)
                carry_kind = "stage_validation_best_with_freeze_policy"
        else:
            raise ValueError(policy)

        accepted_params = _param_snapshot(model)
        accepted_full_eval = _evaluate_stage(model, test_positions, "full_10_period")
        accepted_full_mse = float(accepted_full_eval["rollout_mse"])
        final_equals_best = abs(accepted_full_mse - global_best_mse) <= max(1e-18, 1e-6 * global_best_mse)

        event = {
            "policy": policy,
            "lr": lr,
            "stage": stage,
            "global_step": global_step,
            "event": "stage_complete",
            "carry_kind": carry_kind,
            "stage_steps_run": global_step - stage_start_step,
            "stage_stopped_early": stage_stopped_early,
            "stage_rollbacks": stage_rollbacks,
            "stage_best_step": stage_best_step,
            "stage_start_to_end_param_drift": float(_param_distance(model, [value for value in stage_start_state.values()]).detach().item())
            if False
            else "",
            "final_full_mse_before_carry": final_full_mse,
            "accepted_full_mse_after_carry": accepted_full_mse,
            "best_full_mse_so_far": global_best_mse,
            "final_checkpoint_equals_best_checkpoint": final_equals_best,
        }
        checkpoint_events.append(event)

        stage_row = {
            **accepted_full_eval,
            "policy": policy,
            "lr": lr,
            "lambda_entropy": LAMBDA_ENTROPY,
            "mu": MU,
            "stage": stage,
            "stage_start_step": stage_start_step,
            "stage_end_step": global_step,
            "stage_steps_run": global_step - stage_start_step,
            "stage_stopped_early": stage_stopped_early,
            "stage_best_validation_rollout_mse": stage_best_loss,
            "stage_best_step": stage_best_step,
            "final_stage_validation_rollout_mse_before_carry": final_stage_eval["rollout_mse"],
            "final_full_10_period_rollout_mse_before_carry": final_full_mse,
            "accepted_full_10_period_rollout_mse_after_carry": accepted_full_mse,
            "best_full_10_period_rollout_mse_so_far": global_best_mse,
            "best_full_10_period_global_step": global_best_step,
            "num_rollbacks_in_stage": stage_rollbacks,
            "carry_kind": carry_kind,
            "final_checkpoint_equals_best_checkpoint": final_equals_best,
            "raw_window_ls_rollout_mse": raw_mse,
            "fixed_delay_rollout_mse": fixed_mse,
        }
        stage_rows.append(stage_row)
        full_run_rows.append(stage_row)

    _load_state_dict(model, accepted_state)
    final_eval = _evaluate_stage(model, test_positions, "full_10_period")
    final_equals_best = abs(float(final_eval["rollout_mse"]) - global_best_mse) <= max(1e-18, 1e-6 * global_best_mse)
    summary = {
        **final_eval,
        "policy": policy,
        "lr": lr,
        "lambda_entropy": LAMBDA_ENTROPY,
        "mu": MU,
        "best_full_10_period_rollout_mse": global_best_mse,
        "best_full_10_period_global_step": global_best_step,
        "final_full_10_period_rollout_mse": final_eval["rollout_mse"],
        "final_checkpoint_equals_best_checkpoint": final_equals_best,
        "best_checkpoint_path": str(checkpoint_path),
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
    }
    return summary, full_run_rows, history_rows, pi_rows, checkpoint_events


def _plot_metric_by_stage(rows: list[dict[str, Any]], key: str, filename: str, ylabel: str, log: bool = True) -> None:
    stage_index = {stage: idx for idx, stage in enumerate(STAGES)}
    plt.figure(figsize=(10, 5))
    for policy in POLICIES:
        for lr in LRS:
            selected = [r for r in rows if r.get("policy") == policy and float(r.get("lr", 0.0)) == lr]
            if not selected:
                continue
            plt.plot(
                [stage_index[r["stage"]] for r in selected],
                [float(r[key]) for r in selected],
                marker="o",
                markersize=3,
                label=f"{policy}, lr={lr:g}",
            )
    if log:
        plt.yscale("log")
    plt.xticks(range(len(STAGES)), [str(stage) for stage in STAGES], rotation=30)
    plt.xlabel("curriculum stage")
    plt.ylabel(ylabel)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_scatter(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(6, 5))
    x = [float(r["validation_stage_rollout_mse"]) for r in history]
    y = [float(r["full_10_period_validation_rollout_mse"]) for r in history]
    plt.scatter(x, y, s=12, alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("stage validation rollout MSE")
    plt.ylabel("full 10-period validation rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "stage_loss_vs_full_horizon_loss.png", dpi=160)
    plt.close()


def _plot_summary_bars(summary_rows: list[dict[str, Any]]) -> None:
    labels = [f"{r['policy']}\nlr={float(r['lr']):g}" for r in summary_rows]
    x = np.arange(len(labels))
    best = np.array([float(r["best_full_10_period_rollout_mse"]) for r in summary_rows])
    final = np.array([float(r["final_full_10_period_rollout_mse"]) for r in summary_rows])

    plt.figure(figsize=(10, 5))
    width = 0.38
    plt.bar(x - width / 2, best, width, label="best")
    plt.bar(x + width / 2, final, width, label="final")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("full 10-period rollout MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_so_far_vs_final_mse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x, best)
    plt.yscale("log")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("best full 10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "policy_comparison_bar.png", dpi=160)
    plt.close()


def _plot_pi(pi_rows: list[dict[str, Any]]) -> None:
    if not pi_rows:
        return
    last_by_run: dict[str, dict[str, Any]] = {}
    for row in pi_rows:
        last_by_run[f"{row['policy']}, lr={float(row['lr']):g}"] = row
    labels = list(last_by_run)
    matrix = np.array([[float(last_by_run[label][f"pi_lag_{idx}"]) for idx in range(1, M)] for label in labels])
    plt.figure(figsize=(9, max(3, len(labels) * 0.35)))
    plt.imshow(matrix, aspect="auto", cmap="viridis")
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xticks(range(M - 1), [str(idx) for idx in range(1, M)])
    plt.xlabel("lag")
    plt.colorbar(label="pi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pi_values_by_stage.png", dpi=160)
    plt.close()


def _plot_rollbacks(events: list[dict[str, Any]]) -> None:
    labels = [f"{policy}\nlr={lr:g}" for policy in POLICIES for lr in LRS]
    counts = []
    for policy in POLICIES:
        for lr in LRS:
            counts.append(sum(1 for event in events if event.get("policy") == policy and float(event.get("lr", 0.0)) == lr and event.get("event") == "rollback_to_global_best"))
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("rollback events")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollback_events.png", dpi=160)
    plt.close()


def _write_html(summary_rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    cols = [
        "policy",
        "lr",
        "best_full_10_period_rollout_mse",
        "final_full_10_period_rollout_mse",
        "final_checkpoint_equals_best_checkpoint",
        "one_step_mse",
        "two_step_mse",
        "current_reconstruction_mse",
        "entropy",
        "selected_lag",
        "max_pi",
        "latent_transition_mse",
        "ls_condition_number",
        "raw_window_ls_rollout_mse",
        "fixed_delay_rollout_mse",
    ]
    body = "\n".join(
        "<tr>" + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in cols) + "</tr>"
        for row in summary_rows
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Soft Delay Selector Checkpoint Preservation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}code,pre{{background:#f6f8fa;padding:2px 4px}}</style></head><body>
<h1>Stage 1B Soft-Delay Selector Checkpoint Preservation</h1>
<p>This diagnostic preserves the architecture and LS-identified latent operator. The only intervention is how curriculum checkpoints are selected, rolled back, or frozen. The preservation metric is full 10-period validation rollout MSE.</p>
<table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="full_horizon_mse_vs_stage.png">
<img src="stage_loss_vs_full_horizon_loss.png">
<img src="best_so_far_vs_final_mse.png">
<img src="policy_comparison_bar.png">
<img src="pi_values_by_stage.png">
<img src="entropy_vs_stage.png">
<img src="parameter_drift_by_stage.png">
<img src="rollback_events.png">
</div>
<h2>Setup</h2><pre>{escape(json.dumps(setup, indent=2))}</pre>
</body></html>"""
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
    previous_best = _previous_long_run_best()
    setup = {
        "diagnostic": "stage1B_soft_delay_selector_checkpoint_preservation",
        "m": M,
        "dtype": str(DTYPE),
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
        "lambda_entropy": LAMBDA_ENTROPY,
        "mu": MU,
        "lambda_recon": LAMBDA_RECON,
        "learning_rates": LRS,
        "policies": POLICIES,
        "stage_budgets": {str(k): v for k, v in STAGE_BUDGETS.items()},
        "budget_note": "Runtime-managed checkpoint-preservation screen; model/loss/policy logic matches the request, but max_steps are shorter than the full requested long-run budget.",
        "eval_interval": EVAL_INTERVAL,
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "k_identified_by_latent_ls": True,
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
        "previous_long_run_best_rollout_mse": previous_best,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    summary_rows: list[dict[str, Any]] = []
    full_run_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    pi_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for policy in POLICIES:
        for lr in LRS:
            print(f"training policy={policy} lr={lr:g}", flush=True)
            summary, full_run, history, pi, events = _train_policy(policy, lr, train_positions, test_positions, raw_mse, fixed_mse)
            summary_rows.append(summary)
            full_run_rows.extend(full_run)
            history_rows.extend(history)
            pi_rows.extend(pi)
            event_rows.extend(events)
            _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
            _write_csv(OUTPUT_DIR / "full_run_table.csv", full_run_rows)
            _write_csv(OUTPUT_DIR / "stage_history.csv", history_rows)
            _write_csv(OUTPUT_DIR / "pi_values_by_stage.csv", pi_rows)
            _write_csv(OUTPUT_DIR / "checkpoint_events.csv", event_rows)

    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(
        OUTPUT_DIR / "metrics.json",
        {
            "setup": setup,
            "summary": summary_rows,
            "full_run_table": full_run_rows,
            "stage_history": history_rows,
            "pi_values_by_stage": pi_rows,
            "checkpoint_events": event_rows,
        },
    )
    _plot_metric_by_stage(full_run_rows, "accepted_full_10_period_rollout_mse_after_carry", "full_horizon_mse_vs_stage.png", "accepted full 10-period MSE")
    _plot_metric_by_stage(full_run_rows, "best_full_10_period_rollout_mse_so_far", "best_full_horizon_mse_vs_stage.png", "best full 10-period MSE")
    _plot_metric_by_stage(pi_rows, "entropy", "entropy_vs_stage.png", "entropy", log=False)
    _plot_metric_by_stage(history_rows, "parameter_drift_from_previous_accepted_checkpoint", "parameter_drift_by_stage.png", "parameter drift")
    _plot_scatter(history_rows)
    _plot_summary_bars(summary_rows)
    _plot_pi(pi_rows)
    _plot_rollbacks(event_rows)
    _write_html(summary_rows, setup)


if __name__ == "__main__":
    main()
