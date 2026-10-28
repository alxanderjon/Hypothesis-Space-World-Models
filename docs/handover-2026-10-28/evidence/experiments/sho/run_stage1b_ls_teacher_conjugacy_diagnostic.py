"""Stage 1B LS-teacher / conjugacy diagnostic.

The learned model identifies K_h by least squares in latent space. Raw-window
LS is used only as a data-derived teacher/baseline, never for learned rollout
evaluation.
"""

from __future__ import annotations

import csv
import json
import os
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

from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    CURRICULUM_STAGES,
    DTYPE,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    RIDGE,
    ROLLOUT_PERIODS,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    WindowLSModel,
    _apply_k,
    _decoder_amplification,
    _evaluate_raw_window_ls,
    _fit_k_ridge,
    _linear_r2,
    _make_data,
    _metrics_from_rollout,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/ls_teacher_conjugacy_diagnostic")

WINDOW_SIZES = [2, 8, 32]
H_DIMS = [2, 4]
TRAIN_NUM = 20
TEST_NUM = 12
TEACHER_STEPS = [1, 2, 4, 8, 16]
STAGE_STEPS = {
    1: 30,
    2: 30,
    4: 30,
    8: 30,
    16: 25,
    32: 15,
    64: 8,
    "full_10_period": 2,
}
LAMBDA_RECON = 1e-1
LAMBDA_TEACHER = 1e-1
LAMBDA_CONJ = 1e-1

VARIANTS: dict[str, dict[str, Any]] = {
    "rollout_only": {"teacher": "none", "multi": False, "recon": False},
    "rollout_recon": {"teacher": "none", "multi": False, "recon": True},
    "teacher_dec_1": {"teacher": "decoder", "multi": False, "recon": False},
    "teacher_dec_1_recon": {"teacher": "decoder", "multi": False, "recon": True},
    "teacher_dec_multi": {"teacher": "decoder", "multi": True, "recon": False},
    "teacher_dec_multi_recon": {"teacher": "decoder", "multi": True, "recon": True},
    "conj_1": {"teacher": "conjugacy", "multi": False, "recon": False},
    "conj_1_recon": {"teacher": "conjugacy", "multi": False, "recon": True},
    "conj_multi": {"teacher": "conjugacy", "multi": True, "recon": False},
    "conj_multi_recon": {"teacher": "conjugacy", "multi": True, "recon": True},
}


def _fit_raw_teacher(positions: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_in, q_out = _prefix_pairs(positions, m)
    return _fit_k_ridge(q_in, q_out, ridge=1e-12)


def _apply_power(k_matrix: torch.Tensor, states: torch.Tensor, power: int) -> torch.Tensor:
    result = states
    for _ in range(power):
        result = _apply_k(k_matrix, result)
    return result


def _teacher_losses(
    model: WindowLSModel,
    q_in: torch.Tensor,
    q_out: torch.Tensor,
    k_h: torch.Tensor,
    k_q: torch.Tensor,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = torch.mean((model.decode(model.encode(q_in)) - q_in) ** 2)
    teacher_loss = torch.zeros((), dtype=DTYPE, device=q_in.device)
    conj_loss = torch.zeros((), dtype=DTYPE, device=q_in.device)
    steps = TEACHER_STEPS if cfg["multi"] else [1]

    if cfg["teacher"] == "decoder":
        pieces = []
        h_in = model.encode(q_in)
        for step in steps:
            decoded_student = model.decode(_apply_power(k_h, h_in, step))
            raw_teacher = _apply_power(k_q, q_in, step)
            pieces.append(torch.mean((decoded_student - raw_teacher) ** 2))
        teacher_loss = torch.stack(pieces).mean()

    if cfg["teacher"] == "conjugacy":
        pieces = []
        h_in = model.encode(q_in)
        for step in steps:
            student = _apply_power(k_h, h_in, step)
            teacher_encoded = model.encode(_apply_power(k_q, q_in, step))
            pieces.append(torch.mean((student - teacher_encoded) ** 2))
        conj_loss = torch.stack(pieces).mean()

    return recon_loss, teacher_loss, conj_loss


def _loss_parts(
    model: WindowLSModel,
    positions: torch.Tensor,
    stage: int | str,
    variant: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor]:
    cfg = VARIANTS[variant]
    q_in, q_out = _prefix_pairs(positions, model.m)
    k_h, latent_residual, conditions = model.identify_k(q_in, q_out)
    k_q, _, _ = _fit_raw_teacher(positions, model.m)

    truth = _rollout_truth(positions, model.m, stage)
    h0 = model.encode(truth[:, 0])
    decoded = model.decode(_rollout_matrix(k_h, h0, truth.shape[1] - 1))
    rollout_loss = torch.mean((decoded[:, 1:, -1] - truth[:, 1:, -1]) ** 2)

    recon_loss, teacher_loss, conj_loss = _teacher_losses(model, q_in, q_out, k_h, k_q, cfg)
    loss = rollout_loss
    if cfg["recon"]:
        loss = loss + LAMBDA_RECON * recon_loss
    if cfg["teacher"] == "decoder":
        loss = loss + LAMBDA_TEACHER * teacher_loss
    if cfg["teacher"] == "conjugacy":
        loss = loss + LAMBDA_CONJ * conj_loss

    parts = {
        "window_reconstruction_mse": float(recon_loss.detach().item()),
        "teacher_loss": float(teacher_loss.detach().item()),
        "conjugacy_loss": float(conj_loss.detach().item()),
        "latent_transition_mse": float(latent_residual.detach().item()),
        "ls_condition_number": float(conditions.detach().mean().item()),
    }
    return loss, parts, k_h, k_q


def _evaluate_learned(
    model: WindowLSModel,
    positions: torch.Tensor,
    variant: str,
    raw_mse: float,
    baseline_mse: float | None,
    best_mse: float,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        _, parts, k_h, _ = _loss_parts(model, positions, "full_10_period", variant)
        truth = _rollout_truth(positions, model.m, "full_10_period")
        decoded = model.decode(_rollout_matrix(k_h, model.encode(truth[:, 0]), truth.shape[1] - 1))
        metrics = _metrics_from_rollout(decoded, truth)
        q_in, _ = _prefix_pairs(positions, model.m)
        h_flat = model.encode(q_in).reshape(-1, model.h_dim).detach().cpu().numpy()
        q_flat = q_in.reshape(-1, model.m).detach().cpu().numpy()
        two_state = q_in[..., -2:].reshape(-1, 2).detach().cpu().numpy()

    row = {
        "model_type": "learned_ls_teacher_conjugacy",
        "m": model.m,
        "h_dim": model.h_dim,
        "variant": variant,
        "parameter_count": sum(param.numel() for param in model.parameters()),
        "raw_window_ls_rollout_mse": raw_mse,
        "baseline_learned_window_ls_rollout_mse": "" if baseline_mse is None else baseline_mse,
        "gap_to_raw_ls_floor": metrics["rollout_mse"] - raw_mse,
        "best_10_period_rollout_mse": best_mse,
        "spectral_radius_k_h": _spectral_radius(k_h),
        "probe_h_to_q_r2": _linear_r2(h_flat, q_flat),
        "probe_h_to_two_state_r2": _linear_r2(h_flat, two_state),
        "decoder_amplification": _decoder_amplification(model, q_in),
        **parts,
        **metrics,
    }
    return row


def _train_config(
    train_positions: torch.Tensor,
    test_positions: torch.Tensor,
    m: int,
    h_dim: int,
    variant: str,
    raw_mse: float,
    baseline_mse: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed(SEED)
    model = WindowLSModel(m, h_dim).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    curves: list[dict[str, Any]] = []
    best_mse = float("inf")
    global_step = 0
    for stage in CURRICULUM_STAGES:
        for _ in range(STAGE_STEPS.get(stage, 10)):
            global_step += 1
            optimizer.zero_grad()
            loss, _, _, _ = _loss_parts(model, train_positions, stage, variant)
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
        eval_row = _evaluate_learned(model, test_positions, variant, raw_mse, baseline_mse, best_mse)
        best_mse = min(best_mse, eval_row["rollout_mse"])
        eval_row["best_10_period_rollout_mse"] = best_mse
        curves.append(
            {
                "m": m,
                "h_dim": h_dim,
                "variant": variant,
                "stage": stage,
                "global_step": global_step,
                "rollout_mse": eval_row["rollout_mse"],
                "best_10_period_rollout_mse": best_mse,
                "teacher_loss": eval_row["teacher_loss"],
                "conjugacy_loss": eval_row["conjugacy_loss"],
            }
        )
    return _evaluate_learned(model, test_positions, variant, raw_mse, baseline_mse, best_mse), curves


def _plot_by_variant(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["model_type"] != "raw_window_ls"]
    variants = list(VARIANTS)
    values = []
    for variant in variants:
        vals = [float(row["rollout_mse"]) for row in selected if row["variant"] == variant]
        values.append(float(np.nanmedian(vals)) if vals else np.nan)
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(variants)), values)
    plt.yscale("log")
    plt.xticks(range(len(variants)), variants, rotation=70, ha="right", fontsize=7)
    plt.ylabel("median 10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rollout_mse_by_variant.png", dpi=160)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], x_key: str, filename: str, xlabel: str) -> None:
    selected = [row for row in rows if row["model_type"] != "raw_window_ls" and float(row.get(x_key, 0.0)) > 0]
    plt.figure(figsize=(6.8, 5.0))
    if selected:
        plt.scatter([float(row[x_key]) for row in selected], [float(row["rollout_mse"]) for row in selected], s=28)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("10-period rollout MSE")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def _plot_gap(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["model_type"] != "raw_window_ls"]
    plt.figure(figsize=(12, 5))
    labels = [f"m{row['m']}-h{row['h_dim']}-{row['variant']}" for row in selected]
    vals = [max(float(row["gap_to_raw_ls_floor"]), 1e-30) for row in selected]
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=80, ha="right", fontsize=5)
    plt.ylabel("rollout MSE gap to raw LS floor")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gap_to_raw_ls_floor.png", dpi=160)
    plt.close()


def _plot_probe(rows: list[dict[str, Any]]) -> None:
    selected = [row for row in rows if row["model_type"] != "raw_window_ls"]
    keys = ["probe_h_to_q_r2", "probe_h_to_two_state_r2"]
    labels = ["h -> q", "h -> (x[t-1], x[t])"]
    vals = [np.nanmedian([float(row[key]) for row in selected]) for key in keys]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylim(-0.05, 1.05)
    plt.ylabel("linear probe R2")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "latent_probe_scores.png", dpi=160)
    plt.close()


def _plot_curves(curves: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(10, 5.5))
    for variant in ["rollout_only", "teacher_dec_multi_recon", "conj_multi_recon"]:
        selected = [row for row in curves if row["m"] == 8 and row["h_dim"] == 4 and row["variant"] == variant]
        if selected:
            plt.plot(range(len(selected)), [row["rollout_mse"] for row in selected], marker="o", label=variant)
    plt.yscale("log")
    plt.xlabel("curriculum stage index")
    plt.ylabel("10-period rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "curriculum_stage_curves.png", dpi=160)
    plt.close()


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return escape(str(value))


def _write_html(rows: list[dict[str, Any]], setup: dict[str, Any]) -> None:
    cols = [
        "model_type",
        "m",
        "h_dim",
        "variant",
        "rollout_mse",
        "raw_window_ls_rollout_mse",
        "baseline_learned_window_ls_rollout_mse",
        "gap_to_raw_ls_floor",
        "one_step_mse",
        "two_step_mse",
        "window_reconstruction_mse",
        "teacher_loss",
        "conjugacy_loss",
        "latent_transition_mse",
        "probe_h_to_q_r2",
        "probe_h_to_two_state_r2",
    ]
    top = sorted(rows, key=lambda row: float(row["rollout_mse"]))[:24]
    body = "\n".join("<tr>" + "".join(f"<td>{_fmt(row.get(col, ''))}</td>" for col in cols) + "</tr>" for row in top)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stage 1B LS-Teacher / Conjugacy Diagnostic</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #172033; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 12px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
    img {{ max-width: 100%; border: 1px solid #d8dee8; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
  <h1>Stage 1B LS-Teacher / Conjugacy Diagnostic</h1>
  <p>Runtime-adjusted screen: m={escape(str(WINDOW_SIZES))}, h_dim={escape(str(H_DIMS))}, train/test={TRAIN_NUM}/{TEST_NUM}. Evaluation always uses latent LS K_h; K_q appears only as a raw-window LS baseline and data-derived teacher signal.</p>
  <p>The teacher signal is not omega supervision and does not use coefficients, velocity, analytic SHO recurrence, or an analytic decoder. It does, however, expose the learned representation to transition structure extracted from observed windows, so it should be treated as a stronger data-derived prior than pure rollout reconstruction.</p>
  <h2>Top Rows</h2>
  <table><thead><tr>{''.join(f'<th>{escape(col)}</th>' for col in cols)}</tr></thead><tbody>{body}</tbody></table>
  <div class="grid">
    <img src="rollout_mse_by_variant.png">
    <img src="teacher_loss_vs_rollout_mse.png">
    <img src="conjugacy_loss_vs_rollout_mse.png">
    <img src="gap_to_raw_ls_floor.png">
    <img src="latent_probe_scores.png">
    <img src="curriculum_stage_curves.png">
  </div>
  <h2>Configuration</h2>
  <pre>{escape(json.dumps(setup, indent=2))}</pre>
</body>
</html>
"""
    (OUTPUT_DIR / "summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_positions, _, _ = _make_data(TRAIN_NUM, seed=TRAIN_DATA_SEED)
    test_positions, _, _ = _make_data(TEST_NUM, seed=TEST_DATA_SEED)
    setup = {
        "diagnostic": "stage1B_ls_teacher_conjugacy_diagnostic",
        "runtime_adjusted": True,
        "requested_window_sizes": [2, 8, 16, 32],
        "used_window_sizes": WINDOW_SIZES,
        "requested_h_dims": [2, 4],
        "used_h_dims": H_DIMS,
        "dtype": str(DTYPE),
        "hidden_width": HIDDEN_WIDTH,
        "position_only": True,
        "uses_velocity": False,
        "uses_omega_labels_for_training": False,
        "uses_coefficient_labels": False,
        "uses_analytic_decoder": False,
        "uses_analytic_sho_recurrence_for_training": False,
        "k_generated_by_mlp": False,
        "learned_rollout_uses_k_h_not_k_q": True,
        "raw_window_ls_teacher_from_observed_windows_only": True,
        "lambda_recon": LAMBDA_RECON,
        "lambda_teacher": LAMBDA_TEACHER,
        "lambda_conj": LAMBDA_CONJ,
        "teacher_steps": TEACHER_STEPS,
        "rollout_periods": ROLLOUT_PERIODS,
        "ridge": RIDGE,
        "train_num": TRAIN_NUM,
        "test_num": TEST_NUM,
        "stage_steps": {str(key): value for key, value in STAGE_STEPS.items()},
        "variants": VARIANTS,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)

    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    raw_by_m: dict[int, float] = {}
    baseline_by_mh: dict[tuple[int, int], float] = {}
    for m in WINDOW_SIZES:
        raw_row = _evaluate_raw_window_ls(test_positions, m)
        raw_by_m[m] = raw_row["rollout_mse"]
        rows.append(raw_row)

    for m in WINDOW_SIZES:
        for h_dim in H_DIMS:
            for variant in VARIANTS:
                print(f"training ls_teacher_conjugacy m={m} h_dim={h_dim} variant={variant}", flush=True)
                baseline = baseline_by_mh.get((m, h_dim))
                final_row, curve_rows = _train_config(train_positions, test_positions, m, h_dim, variant, raw_by_m[m], baseline)
                if variant == "rollout_only":
                    baseline_by_mh[(m, h_dim)] = final_row["rollout_mse"]
                    final_row["baseline_learned_window_ls_rollout_mse"] = final_row["rollout_mse"]
                rows.append(final_row)
                curves.extend(curve_rows)
                _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
                _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curves)

    compact = rows
    setup["wall_clock_sec"] = time.perf_counter() - start
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_csv(OUTPUT_DIR / "summary.csv", compact)
    _write_csv(OUTPUT_DIR / "full_run_table.csv", rows)
    _write_csv(OUTPUT_DIR / "curriculum_stage_curves.csv", curves)
    _write_json(OUTPUT_DIR / "metrics.json", {"setup": setup, "summary": compact})

    _plot_by_variant(rows)
    _plot_scatter(rows, "teacher_loss", "teacher_loss_vs_rollout_mse.png", "teacher loss")
    _plot_scatter(rows, "conjugacy_loss", "conjugacy_loss_vs_rollout_mse.png", "conjugacy loss")
    _plot_gap(rows)
    _plot_probe(rows)
    _plot_curves(curves)
    _write_html(rows, setup)


if __name__ == "__main__":
    main()
