"""Stage 1B covariance + SuperLoss encoder diagnostic.

Tests pure encoder prediction of trajectory-specific c_i with prefix-only
objectives:

  A. L_multi
  B. L_multi + alpha L_cov
  C. SuperLoss(L_multi)
  D. SuperLoss(L_multi + alpha L_cov)

Future rollout is evaluation-only. No velocity, labels, LS, closed-form c
estimator, Hessian, or Newton is used in training.
"""

from __future__ import annotations

import copy
import csv
import json
import math
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
    DRIFT_HEAVY,
    J_ALL,
    P,
    TEST_NUM,
    TRAIN_NUM,
    VAL_NUM,
    TeacherSet,
    _make_data,
    _make_teacher_sets,
)
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import (
    CInitTCN,
    _metrics,
    _param_count,
    _predict_c,
    _prefix_loss_vector,
    _refine_c,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/covariance_superloss_encoder_diagnostic")
BATCH_SIZE = int(os.environ.get("COVSUPER_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("COVSUPER_MAX_STEPS", "50000"))
EVAL_INTERVAL = int(os.environ.get("COVSUPER_EVAL_INTERVAL", "1000"))
PATIENCE = int(os.environ.get("COVSUPER_PATIENCE", "50000"))
LEARNING_RATE = float(os.environ.get("COVSUPER_LR", "1e-4"))
SUPERLOSS_FRACTION = float(os.environ.get("COVSUPER_SUPERLOSS_FRACTION", "0.7"))
TAU_EMA = float(os.environ.get("COVSUPER_TAU_EMA", "0.98"))
REFINE_STEPS = [0, 20, 50, 100]
REFINE_LR = 1e-3
ALPHAS = [0.1, 1.0]
SUPER_LAMBDAS = [0.1, 1.0]


def _variant_grid() -> list[dict[str, Any]]:
    requested = os.environ.get("COVSUPER_VARIANTS", "").strip()
    if requested:
        names = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        names = ["multi"] + [f"multi_cov_a{a:g}" for a in ALPHAS] + [f"super_l{l:g}" for l in SUPER_LAMBDAS] + [f"super_cov_a{a:g}_l{l:g}" for a in ALPHAS for l in SUPER_LAMBDAS]
    out = []
    for i, name in enumerate(names):
        parts = name.split("_")
        use_cov = "cov" in parts
        use_super = parts[0] == "super"
        alpha = 0.0
        lam = 0.0
        for part in parts:
            if part.startswith("a"):
                alpha = float(part[1:])
            if part.startswith("l"):
                lam = float(part[1:])
        out.append({"config_index": i, "config_name": name, "use_covariance": use_cov, "alpha_cov": alpha, "use_superloss": use_super, "superloss_lambda": lam})
    return out


def _covariance_loss_vector(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    h_t = torch.stack([x[:, :-2], x[:, 1:-1]], dim=-1)
    h_next = torch.stack([x[:, 1:-1], x[:, 2:]], dim=-1)
    c00 = torch.einsum("nti,ntj->nij", h_t, h_t) / h_t.shape[1]
    c10 = torch.einsum("nti,ntj->nij", h_next, h_t) / h_t.shape[1]
    k = torch.zeros(c.shape[0], 2, 2, dtype=DTYPE, device=c.device)
    k[:, 0, 1] = 1.0
    k[:, 1, 0] = -1.0
    k[:, 1, 1] = c
    residual = torch.bmm(k, c00) - c10
    denom = torch.sum(c10**2, dim=(1, 2)).clamp_min(1e-12)
    return torch.sum(residual**2, dim=(1, 2)) / denom


def _objective_vector(c: torch.Tensor, positions: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    multi = _prefix_loss_vector(c, positions)
    cov = _covariance_loss_vector(c, positions)
    total = multi + float(cfg["alpha_cov"]) * cov if cfg["use_covariance"] else multi
    return total, multi, cov


def _lambertw_principal(z: torch.Tensor, iters: int = 20) -> torch.Tensor:
    z = z.clamp_min(-1.0 / math.e + 1e-12)
    w = torch.where(z < 1.0, z, torch.log1p(z))
    for _ in range(iters):
        ew = torch.exp(w)
        f = w * ew - z
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0).clamp_min(1e-12)
        w = w - f / denom.clamp_min(1e-12)
    return w


def _superloss_weights(losses: torch.Tensor, tau: torch.Tensor, lam: float) -> torch.Tensor:
    # Paper objective: (L_i - tau) sigma_i + lambda (log sigma_i)^2.
    # Optimal sigma_i = exp(W((tau - L_i)/(2 lambda))).
    z = (tau - losses.detach()) / (2.0 * float(lam))
    return torch.exp(_lambertw_principal(z)).detach()


def _eval_encoder(model: CInitTCN, dataset: TeacherSet, cfg: dict[str, Any], stage: str, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        total, multi, cov = _objective_vector(c, dataset.data.positions, cfg)
    row = _metrics(stage, str(cfg["config_name"]), c, dataset, {"split": split, "covariance_residual": float(cov.mean().item()), "multi_prefix_loss": float(multi.mean().item()), "training_objective": float(total.mean().item()), "parameter_count": _param_count(model)})
    if extra:
        row.update(extra)
    return row


def _train_variant(train: TeacherSet, val: TeacherSet, test: TeacherSet, cfg: dict[str, Any]) -> tuple[CInitTCN, list[dict[str, Any]], dict[str, Any]]:
    model = CInitTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 161000 + int(cfg["config_index"]) * 1000)
    n = train.data.positions.shape[0]
    tau = torch.tensor(1.0, dtype=DTYPE)
    super_end = int(MAX_STEPS * SUPERLOSS_FRACTION)
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        c = _predict_c(model, TeacherSet(type(train.data)(positions=pos, omegas=train.data.omegas[idx]), train.c_free[idx], train.mu, train.sigma))
        losses, multi, cov = _objective_vector(c, pos, cfg)
        use_super_now = bool(cfg["use_superloss"]) and step <= super_end
        if use_super_now:
            tau = TAU_EMA * tau + (1.0 - TAU_EMA) * losses.detach().mean()
            weights = _superloss_weights(losses, tau, float(cfg["superloss_lambda"]))
            loss = torch.mean(weights * losses)
            weight_mean = float(weights.mean().item())
            weight_min = float(weights.min().item())
            frac_down = float((weights < 0.5).to(DTYPE).mean().item())
        else:
            loss = torch.mean(losses)
            weight_mean = 1.0
            weight_min = 1.0
            frac_down = 0.0
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_row = _eval_encoder(
                model,
                val,
                cfg,
                "training_history",
                "val",
                {
                    "step": step,
                    "elapsed_seconds": time.perf_counter() - start,
                    "train_loss": float(loss.item()),
                    "train_multi_prefix_loss": float(multi.mean().detach().item()),
                    "train_covariance_residual": float(cov.mean().detach().item()),
                    "superloss_active": use_super_now,
                    "superloss_tau": float(tau.item()),
                    "superloss_weight_mean": weight_mean,
                    "superloss_weight_min": weight_min,
                    "fraction_heavily_downweighted": frac_down,
                },
            )
            history.append(val_row)
            score = float(val_row["c_rmse"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final = _eval_encoder(model, test, cfg, "encoder_final", "test", {"best_validation_c_rmse": best})
    return model, history, final


def _refinement_rows(model: CInitTCN, dataset: TeacherSet, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    with torch.no_grad():
        init = _predict_c(model, dataset)
    rows = []
    for steps in REFINE_STEPS:
        _current, best, runtime = _refine_c(init, dataset, "adam", REFINE_LR, steps)
        row = _metrics(
            "secondary_adam_refinement",
            f"{cfg['config_name']}_adam_refine_steps_{steps}",
            best,
            dataset,
            {
                "variant": cfg["config_name"],
                "refinement_steps": steps,
                "optimizer": "adam",
                "learning_rate": REFINE_LR,
                "runtime_seconds": runtime,
                "runtime_per_trajectory_seconds": runtime / max(1, dataset.data.positions.shape[0]),
            },
        )
        with torch.no_grad():
            row["covariance_residual"] = float(_covariance_loss_vector(best, dataset.data.positions).mean().item())
        rows.append(row)
    return rows


def _threshold_steps(rows: list[dict[str, Any]], variant: str, threshold: float) -> int | None:
    vals = [int(r["refinement_steps"]) for r in rows if r.get("variant") == variant and float(r["future_10_period_mse"]) < threshold]
    return min(vals) if vals else None


def _interpret(variants: list[dict[str, Any]], refinement: list[dict[str, Any]], history: list[dict[str, Any]]) -> dict[str, Any]:
    best = min(variants, key=lambda r: float(r["future_10_period_mse"]), default={})
    multi = next((r for r in variants if r["config_name"] == "multi"), {})
    cov = min([r for r in variants if r.get("alpha_cov", 0) not in ("", 0, "0") and not str(r["config_name"]).startswith("super")], key=lambda r: float(r["future_10_period_mse"]), default={})
    super_only = min([r for r in variants if str(r["config_name"]).startswith("super_l")], key=lambda r: float(r["future_10_period_mse"]), default={})
    super_cov = min([r for r in variants if str(r["config_name"]).startswith("super_cov")], key=lambda r: float(r["future_10_period_mse"]), default={})
    raw_phase = [r for r in history if str(r.get("superloss_active")) == "False"]
    super_phase = [r for r in history if str(r.get("superloss_active")) == "True"]
    return {
        "best_encoder_only_variant": best.get("config_name", ""),
        "best_encoder_only_10_period_mse": best.get("future_10_period_mse", ""),
        "best_encoder_only_c_rmse": best.get("c_rmse", ""),
        "covariance_improves_over_multi": float(cov.get("future_10_period_mse", float("inf"))) < float(multi.get("future_10_period_mse", float("inf"))),
        "superloss_improves_over_multi": float(super_only.get("future_10_period_mse", float("inf"))) < float(multi.get("future_10_period_mse", float("inf"))),
        "covariance_plus_superloss_best": super_cov.get("config_name", ""),
        "covariance_plus_superloss_outperforms_either_alone": float(super_cov.get("future_10_period_mse", float("inf"))) < min(float(cov.get("future_10_period_mse", float("inf"))), float(super_only.get("future_10_period_mse", float("inf")))),
        "final_raw_loss_phase_present": bool(raw_phase),
        "superloss_phase_present": bool(super_phase),
        "min_encoder_refinement_steps_to_1e_6": min([s for r in variants if (s := _threshold_steps(refinement, str(r["config_name"]), 1e-6)) is not None], default=None),
        "min_encoder_refinement_steps_to_1e_9": min([s for r in variants if (s := _threshold_steps(refinement, str(r["config_name"]), 1e-9)) is not None], default=None),
        "replacement_assessment": "pure encoder replacement" if float(best.get("future_10_period_mse", float("inf"))) < 1e-6 else "better initializer only",
    }


def _plot_history(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(9, 5))
    for name in sorted({r["config_name"] for r in rows}):
        vals = sorted([r for r in rows if r["config_name"] == name], key=lambda r: int(r["step"]))
        plt.plot([int(r["step"]) for r in vals], [float(r["c_rmse"]) for r in vals], label=name, linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("encoder-only c RMSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoder_only_c_rmse_vs_training_step.png", dpi=170)
    plt.close()


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals = [float(r[key]) for r in rows]
    labels = [str(r["config_name"]) for r in rows]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(vals)), vals)
    plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=55, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], xkey: str, ykey: str, filename: str) -> None:
    vals = [r for r in rows if r.get(xkey, "") != "" and r.get(ykey, "") != ""]
    if not vals:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter([float(r[xkey]) for r in vals], [float(r[ykey]) for r in vals], s=24)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_refinement(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for name in sorted({r["variant"] for r in rows}):
        vals = sorted([r for r in rows if r["variant"] == name], key=lambda r: int(r["refinement_steps"]))
        plt.plot([int(r["refinement_steps"]) for r in vals], [float(r["future_10_period_mse"]) for r in vals], marker="o", label=name, linewidth=1)
    plt.yscale("log")
    plt.xlabel("Adam refinement steps")
    plt.ylabel("10-period MSE")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "refinement_steps_vs_rollout_mse.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["config_name", "use_covariance", "alpha_cov", "use_superloss", "superloss_lambda", "c_rmse", "c_r2_against_teacher", "collapse_ratio", "covariance_residual", "future_10_period_mse"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["variant_comparison"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Covariance SuperLoss Encoder Diagnostic</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Covariance + SuperLoss Encoder Diagnostic</h1>",
        "<p>Pure TCN encoder prediction of c_i from clean position-only prefixes. Future rollout is evaluation-only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Variant Comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 171000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 171100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 171200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, _teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    history: list[dict[str, Any]] = []
    variants: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    for cfg in _variant_grid():
        print(f"training {cfg['config_name']}", flush=True)
        model, hist, final = _train_variant(train, val, test, cfg)
        final.update(cfg)
        history.extend(hist)
        variants.append(final)
        refinement.extend(_refinement_rows(model, test, cfg))
    summary = variants + refinement
    return {"summary": summary, "training_history": history, "variant_comparison": variants, "refinement_results": refinement}


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    _write_csv(OUTPUT_DIR / "variant_comparison.csv", outputs["variant_comparison"])
    _write_csv(OUTPUT_DIR / "refinement_results.csv", outputs["refinement_results"])
    interpretation = _interpret(outputs["variant_comparison"], outputs["refinement_results"], outputs["training_history"])
    setup = {
        "diagnostic": "stage1B_covariance_superloss_encoder",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "operator": "K(c)=[[0,1],[-1,c]]",
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "alpha_sweep": ALPHAS,
        "superloss_lambda_sweep": SUPER_LAMBDAS,
        "superloss_fraction": SUPERLOSS_FRACTION,
        "tau_ema": TAU_EMA,
        "uses_velocity": False,
        "uses_labels": False,
        "uses_LS_or_closed_form_estimator": False,
        "uses_future_data_in_training": False,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_history(outputs["training_history"])
    _plot_bar(outputs["variant_comparison"], "future_10_period_mse", "encoder_only_10_period_mse_by_variant.png")
    _plot_scatter(outputs["training_history"], "train_loss", "superloss_weight_mean", "superloss_weight_vs_trajectory_loss.png")
    _plot_scatter(outputs["variant_comparison"], "covariance_residual", "future_10_period_mse", "covariance_residual_vs_rollout_mse.png")
    _plot_refinement(outputs["refinement_results"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **_interpret(outputs["variant_comparison"], outputs["refinement_results"], outputs["training_history"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
