"""Stage 1B soft-delay selector learning ablation.

This diagnostic separates what is learned from what is provided by the
soft-delay coordinate family plus least-squares operator identification.
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

from experiments.sho.run_stage1b_learned_delay_coordinate_prior_diagnostic import (
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
    _fit_k_ridge,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/soft_delay_learning_ablation")
FINAL_DIR = Path("outputs/sho/stage1B/final_soft_delay_selector_success_run")
LONG_RUN_DIR = Path("outputs/sho/stage1B/soft_delay_selector_long_run")

M = 8
TRAIN_NUM = 8
TEST_NUM = 6
LR = 3e-3
LAMBDA_RECON = 1e-1
MU = 1.0
EVAL_INTERVAL = 100
STAGES: list[int | str] = [1, 2, 4, 8, 16, 32, 64, "full_10_period"]
STAGE_STEPS = 1000
THRESHOLDS = [1e-14, 1e-15, 1e-16, 1e-18]


def _best_from_csv(path: Path, column: str) -> float | None:
    if not path.exists():
        return None
    best: float | None = None
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                value = float(row.get(column, "nan"))
            except ValueError:
                continue
            if np.isfinite(value):
                best = value if best is None else min(best, value)
    return best


def _pi_vector(kind: str) -> torch.Tensor:
    if kind == "uniform":
        return torch.full((M - 1,), 1.0 / (M - 1), dtype=DTYPE)
    if kind == "lag1":
        pi = torch.zeros(M - 1, dtype=DTYPE)
        pi[0] = 1.0
        return pi
    if kind == "lag7":
        pi = torch.zeros(M - 1, dtype=DTYPE)
        pi[-1] = 1.0
        return pi
    idx = torch.arange(M - 1, dtype=DTYPE)
    beta = 0.5
    if kind == "recency_weighted":
        weights = torch.exp(-beta * idx)
        return weights / weights.sum()
    if kind == "far_weighted":
        weights = torch.exp(beta * idx)
        return weights / weights.sum()
    raise ValueError(f"unknown pi kind={kind}")


class SoftDelayAblationModel(nn.Module):
    def __init__(
        self,
        *,
        pi_kind: str = "uniform",
        train_pi: bool = False,
        train_decoder: bool = False,
        fixed_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.m = M
        self.pi_kind = pi_kind
        self.train_pi = train_pi
        self.train_decoder = train_decoder
        self.fixed_decoder = fixed_decoder
        if train_pi:
            init = torch.log(_pi_vector(pi_kind).clamp_min(1e-30))
            self.alpha = nn.Parameter(init.clone())
            self.register_buffer("pi_fixed", torch.empty(0, dtype=DTYPE))
        else:
            self.alpha = None
            self.register_buffer("pi_fixed", _pi_vector(pi_kind))
        self.decoder = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.decoder.weight.zero_()
            self.decoder.weight[0, 1] = 1.0
        for param in self.decoder.parameters():
            param.requires_grad_(train_decoder and not fixed_decoder)

    def pi(self) -> torch.Tensor:
        if self.alpha is not None:
            return torch.softmax(self.alpha, dim=0)
        return self.pi_fixed

    def encode(self, q: torch.Tensor) -> torch.Tensor:
        pi = self.pi().to(q.device)
        delayed = torch.sum(q[..., :-1] * pi, dim=-1)
        return torch.stack((delayed, q[..., -1]), dim=-1)

    def decode_x(self, h: torch.Tensor) -> torch.Tensor:
        if self.fixed_decoder:
            return h[..., 1]
        return self.decoder(h).squeeze(-1)

    def identify_k(self, q_in: torch.Tensor, q_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fit_k_ridge(self.encode(q_in), self.encode(q_out))

    def decoder_parameters_json(self) -> str:
        if self.fixed_decoder:
            return json.dumps({"fixed": "x_hat=h[1]", "weight": [0.0, 1.0]})
        weight = self.decoder.weight.detach().cpu().reshape(-1).tolist()
        return json.dumps({"linear_weight": weight, "bias": False})


def _fit_closed_form_decoder(model: SoftDelayAblationModel, positions: torch.Tensor) -> torch.Tensor:
    q_in, _ = _prefix_pairs(positions, model.m)
    h = model.encode(q_in).reshape(-1, 2).detach()
    x = q_in[..., -1].reshape(-1, 1).detach()
    coef = torch.linalg.lstsq(h, x).solution.reshape(2)
    return coef


def _evaluate(
    model: SoftDelayAblationModel,
    positions: torch.Tensor,
    stage: int | str = "full_10_period",
    decoder_weight: torch.Tensor | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        q_in, q_out = _prefix_pairs(positions, model.m)
        k_h, residual, conditions = model.identify_k(q_in, q_out)
        truth = _rollout_truth(positions, model.m, stage)
        h_rollout = _rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1)
        if decoder_weight is None:
            pred_x = model.decode_x(h_rollout)
            recon = model.decode_x(model.encode(q_in))
        else:
            pred_x = torch.einsum("btd,d->bt", h_rollout, decoder_weight.to(h_rollout.device))
            recon = torch.einsum("btd,d->bt", model.encode(q_in), decoder_weight.to(q_in.device))
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
        "decoder_parameters": model.decoder_parameters_json()
        if decoder_weight is None
        else json.dumps({"closed_form_weight": decoder_weight.detach().cpu().tolist(), "bias": False}),
        "k_h_ls_identified": True,
    }


def _threshold_flags(value: float) -> dict[str, bool]:
    return {f"crosses_{threshold:g}": value < threshold for threshold in THRESHOLDS}


def _loss_parts(model: SoftDelayAblationModel, positions: torch.Tensor, stage: int | str, include_recon: bool) -> tuple[torch.Tensor, dict[str, Any]]:
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, residual, conditions = model.identify_k(q_in, q_out)
    truth = _rollout_truth(positions, model.m, stage)
    pred_x = model.decode_x(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
    rollout_loss = torch.mean((pred_x[:, 1:] - truth[:, 1:, -1]) ** 2)
    recon_loss = torch.mean((model.decode_x(model.encode(q_in)) - q_in[..., -1]) ** 2)
    loss = rollout_loss + (LAMBDA_RECON * recon_loss if include_recon else torch.zeros((), dtype=DTYPE))
    return loss, {
        "train_stage_rollout_mse": float(rollout_loss.detach().item()),
        "current_reconstruction_mse": float(recon_loss.detach().item()),
        "latent_transition_mse": float(residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }


def _train_case(
    *,
    case_name: str,
    train_pi: bool,
    train_decoder: bool,
    fixed_decoder: bool,
    lr: float,
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    carry_best: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = SoftDelayAblationModel(pi_kind="uniform", train_pi=train_pi, train_decoder=train_decoder, fixed_decoder=fixed_decoder).to(dtype=DTYPE)
    zero_eval = _evaluate(model, test_positions)
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr) if params else None
    accepted_snapshot = _param_snapshot(model)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_eval = zero_eval
    best_mse = float(zero_eval["rollout_mse"])
    global_step = 0
    history: list[dict[str, Any]] = []
    final_eval = zero_eval
    include_recon = train_decoder

    for stage in STAGES:
        if optimizer is None:
            stage_eval = _evaluate(model, test_positions, stage)
            full_eval = _evaluate(model, test_positions)
            history.append({
                **full_eval,
                "config": case_name,
                "stage": stage,
                "global_step": global_step,
                "local_step": 0,
                "train_stage_rollout_mse": "",
                "validation_stage_rollout_mse": stage_eval["rollout_mse"],
                "full_10_period_mse": full_eval["rollout_mse"],
                "best_full_10_period_mse_so_far": best_mse,
                "parameter_drift_from_best_to_final": 0.0,
            })
            continue
        for local_step in range(1, STAGE_STEPS + 1):
            global_step += 1
            optimizer.zero_grad()
            loss, train_parts = _loss_parts(model, train_positions, stage, include_recon)
            bias = MU * _param_distance(model, accepted_snapshot) if carry_best else torch.zeros((), dtype=DTYPE)
            total = loss + bias
            if not torch.isfinite(total):
                break
            total.backward()
            optimizer.step()
            if local_step % EVAL_INTERVAL == 0 or local_step == 1 or local_step == STAGE_STEPS:
                stage_eval = _evaluate(model, test_positions, stage)
                full_eval = _evaluate(model, test_positions)
                full_mse = float(full_eval["rollout_mse"])
                if full_mse < best_mse:
                    best_mse = full_mse
                    best_eval = full_eval
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                history.append({
                    **full_eval,
                    "config": case_name,
                    "stage": stage,
                    "global_step": global_step,
                    "local_step": local_step,
                    "train_stage_rollout_mse": train_parts["train_stage_rollout_mse"],
                    "validation_stage_rollout_mse": stage_eval["rollout_mse"],
                    "full_10_period_mse": full_mse,
                    "best_full_10_period_mse_so_far": best_mse,
                    "biased_regularization_penalty": float(bias.detach().item()),
                })
        if carry_best:
            model.load_state_dict(best_state)
            accepted_snapshot = _param_snapshot(model)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    final_eval = _evaluate(model, test_positions)
    drift = float(_param_distance(model, [v for v in best_state.values()]).detach().item()) if False else 0.0
    summary = {
        **final_eval,
        "config": case_name,
        "lr": lr,
        "zero_training_full_10_period_mse": zero_eval["rollout_mse"],
        "final_full_10_period_mse": final_eval["rollout_mse"],
        "best_full_10_period_mse": best_mse,
        "best_checkpoint_mse": best_eval["rollout_mse"],
        "pi_trained": train_pi,
        "decoder_trained": train_decoder,
        "fixed_decoder": fixed_decoder,
        "training_improved_vs_zero": best_mse < float(zero_eval["rollout_mse"]),
        "training_worsened_final_vs_zero": float(final_eval["rollout_mse"]) > float(zero_eval["rollout_mse"]),
        "parameter_drift_from_best_to_final": drift,
        **_threshold_flags(best_mse),
    }
    return summary, history


def _plot_bars(rows: list[dict[str, Any]], key: str, filename: str, ylabel: str) -> None:
    labels = [row["config"] for row in rows]
    values = [float(row[key]) for row in rows]
    plt.figure(figsize=(max(7, len(labels) * 0.8), 4.8))
    plt.bar(range(len(labels)), values)
    plt.yscale("log")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_lr(history: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for config in sorted({row["config"] for row in history}):
        if not config.startswith("lr_drift"):
            continue
        selected = [row for row in history if row["config"] == config]
        plt.plot([float(row["global_step"]) for row in selected], [float(row["full_10_period_mse"]) for row in selected], label=config)
    plt.yscale("log")
    plt.xlabel("global step")
    plt.ylabel("full 10-period MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lr_drift_full_horizon_mse.png", dpi=160)
    plt.close()


def _plot_pi(rows: list[dict[str, Any]]) -> None:
    labels = [row["config"] for row in rows]
    matrix = np.array([[float(row[f"pi_lag_{idx}"]) for idx in range(1, M)] for row in rows])
    plt.figure(figsize=(max(7, len(labels) * 0.45), 5))
    plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xticks(range(M - 1), [str(idx) for idx in range(1, M)])
    plt.xlabel("lag")
    plt.colorbar(label="pi")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pi_values_by_config.png", dpi=160)
    plt.close()


def _plot_thresholds(rows: list[dict[str, Any]]) -> None:
    labels = [row["config"] for row in rows]
    matrix = np.array([[1.0 if row[f"crosses_{threshold:g}"] else 0.0 for threshold in THRESHOLDS] for row in rows])
    plt.figure(figsize=(8, max(3, len(labels) * 0.3)))
    plt.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xticks(range(len(THRESHOLDS)), [f"{threshold:g}" for threshold in THRESHOLDS])
    plt.xlabel("threshold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_crossings.png", dpi=160)
    plt.close()


def _write_html(summary_rows: list[dict[str, Any]], zero_rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    by_config = {row["config"]: row for row in summary_rows}
    uniform_fixed = by_config.get("zero_uniform_fixed_current_decoder", {})
    train_pi = by_config.get("train_pi_fixed_decoder", {})
    train_decoder = by_config.get("freeze_pi_train_decoder", {})
    train_both = by_config.get("train_pi_and_decoder", {})
    answers = [
        f"Uniform soft selector with fixed decoder zero-training MSE: {uniform_fixed.get('best_full_10_period_mse', uniform_fixed.get('rollout_mse', ''))}.",
        f"Raw-window LS floor: {setup['raw_window_ls_rollout_mse']}; fixed-delay floor: {setup['fixed_delay_rollout_mse']}; previous soft best: {setup['previous_long_run_best_rollout_mse']}.",
        f"Training pi with fixed decoder best MSE: {train_pi.get('best_full_10_period_mse', '')}; final MSE: {train_pi.get('final_full_10_period_mse', '')}.",
        f"Training decoder with frozen pi best MSE: {train_decoder.get('best_full_10_period_mse', '')}; final MSE: {train_decoder.get('final_full_10_period_mse', '')}.",
        f"Training pi and decoder best MSE: {train_both.get('best_full_10_period_mse', '')}; final MSE: {train_both.get('final_full_10_period_mse', '')}.",
        "The honest Stage 1B claim is that the soft-delay coordinate family plus LS operator identification is doing the core work; training can preserve, slightly tune, or drift depending on what is allowed to move.",
    ]
    cols = ["config", "zero_training_full_10_period_mse", "best_full_10_period_mse", "final_full_10_period_mse", "entropy", "selected_lag", "max_pi", "pi_trained", "decoder_trained", "crosses_1e-14", "crosses_1e-15", "crosses_1e-16", "crosses_1e-18"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(row.get(col, '')))}</td>" for col in cols) + "</tr>" for row in summary_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Soft Delay Learning Ablation</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #d8dee8;padding:6px 8px;text-align:right}}th{{background:#eef2f7}}img{{max-width:100%;border:1px solid #d8dee8}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}}pre{{background:#f6f8fa;padding:12px;overflow:auto}}</style></head><body>
<h1>Stage 1B Soft-Delay Learning Ablation</h1>
<p>This diagnostic separates learned selector weights, learned decoder, soft-delay coordinate prior, and LS operator identification.</p>
<h2>Direct Answers</h2><ol>{''.join(f'<li>{escape(answer)}</li>' for answer in answers)}</ol>
<h2>Summary</h2><table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">
<img src="zero_training_mse_by_pi.png">
<img src="zero_training_mse_by_decoder.png">
<img src="trained_vs_untrained_mse.png">
<img src="lr_drift_full_horizon_mse.png">
<img src="threshold_crossings.png">
<img src="pi_values_by_config.png">
</div>
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
    setup = {
        "diagnostic": "stage1B_soft_delay_learning_ablation",
        "m": M,
        "dtype": str(DTYPE),
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "rollout_periods": ROLLOUT_PERIODS,
        "raw_window_ls_rollout_mse": raw_mse,
        "fixed_delay_rollout_mse": fixed_mse,
        "previous_final_success_preserved_mse": _best_from_csv(FINAL_DIR / "summary.csv", "best_full_10_period_rollout_mse"),
        "previous_long_run_best_rollout_mse": _best_from_csv(LONG_RUN_DIR / "stage_history.csv", "test_rollout_mse"),
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence": False,
        "uses_raw_window_ls_teacher": False,
        "k_generated_by_mlp": False,
        "k_identified_by_latent_ls": True,
        "stage_steps": STAGE_STEPS,
        "eval_interval": EVAL_INTERVAL,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    zero_rows: list[dict[str, Any]] = []
    for pi_kind in ["uniform", "lag1", "lag7", "recency_weighted", "far_weighted"]:
        for decoder_kind in ["fixed_current_decoder", "initialized_linear_decoder", "closed_form_linear_readout"]:
            fixed = decoder_kind == "fixed_current_decoder"
            model = SoftDelayAblationModel(pi_kind=pi_kind, train_pi=False, train_decoder=False, fixed_decoder=fixed).to(dtype=DTYPE)
            decoder_weight = _fit_closed_form_decoder(model, train_positions) if decoder_kind == "closed_form_linear_readout" else None
            eval_row = _evaluate(model, test_positions, decoder_weight=decoder_weight)
            mse = float(eval_row["rollout_mse"])
            row = {
                **eval_row,
                "config": f"zero_{pi_kind}_{decoder_kind}",
                "pi_kind": pi_kind,
                "decoder_kind": decoder_kind,
                "zero_training_full_10_period_mse": mse,
                "best_full_10_period_mse": mse,
                "final_full_10_period_mse": mse,
                "pi_trained": False,
                "decoder_trained": False,
                **_threshold_flags(mse),
            }
            zero_rows.append(row)

    trained_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    train_specs = [
        ("freeze_pi_train_decoder", False, True, False, LR, False),
        ("train_pi_fixed_decoder", True, False, True, LR, False),
        ("train_pi_and_decoder", True, True, False, LR, False),
        ("freeze_pi_and_decoder", False, False, True, LR, False),
    ]
    for case_name, train_pi, train_decoder, fixed_decoder, lr, carry_best in train_specs:
        print(f"training {case_name}", flush=True)
        summary, history = _train_case(
            case_name=case_name,
            train_pi=train_pi,
            train_decoder=train_decoder,
            fixed_decoder=fixed_decoder,
            lr=lr,
            train_positions=train_positions,
            test_positions=test_positions,
            carry_best=carry_best,
        )
        trained_rows.append(summary)
        history_rows.extend(history)

    lr_rows: list[dict[str, Any]] = []
    for lr in [1e-2, 3e-3, 1e-3]:
        case_name = f"lr_drift_train_pi_and_decoder_lr_{lr:g}"
        print(f"training {case_name}", flush=True)
        summary, history = _train_case(
            case_name=case_name,
            train_pi=True,
            train_decoder=True,
            fixed_decoder=False,
            lr=lr,
            train_positions=train_positions,
            test_positions=test_positions,
            carry_best=True,
        )
        lr_rows.append(summary)
        history_rows.extend(history)

    summary_rows = zero_rows + trained_rows + lr_rows
    threshold_rows = [{"config": row["config"], "best_full_10_period_mse": row["best_full_10_period_mse"], **{f"crosses_{t:g}": row[f"crosses_{t:g}"] for t in THRESHOLDS}} for row in summary_rows]
    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": summary_rows, "history": history_rows})
    _write_csv(OUTPUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", history_rows)
    _write_csv(OUTPUT_DIR / "zero_training_ablation.csv", zero_rows)
    _write_csv(OUTPUT_DIR / "threshold_audit.csv", threshold_rows)
    _write_csv(OUTPUT_DIR / "lr_drift_table.csv", lr_rows)
    _write_csv(OUTPUT_DIR / "pi_values_by_config.csv", summary_rows)

    _plot_bars([row for row in zero_rows if row["decoder_kind"] == "fixed_current_decoder"], "best_full_10_period_mse", "zero_training_mse_by_pi.png", "zero-training full 10-period MSE")
    _plot_bars([row for row in zero_rows if row["pi_kind"] == "uniform"], "best_full_10_period_mse", "zero_training_mse_by_decoder.png", "uniform pi zero-training MSE")
    _plot_bars(trained_rows, "best_full_10_period_mse", "trained_vs_untrained_mse.png", "best full 10-period MSE")
    _plot_lr(history_rows)
    _plot_thresholds(summary_rows)
    _plot_pi(summary_rows)
    _write_html(summary_rows, zero_rows, setup)


if __name__ == "__main__":
    main()
