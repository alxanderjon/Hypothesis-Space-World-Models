"""Stage 1B unrolled c_i refinement training diagnostic.

Train a prefix encoder so its c_i initialization becomes accurate after a small
fixed number of differentiable prefix-only refinement steps. Future trajectory
points are evaluation-only.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys
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
    DRIFT_HEAVY,
    J_ALL,
    P,
    TEST_NUM,
    THRESHOLDS,
    TRAIN_NUM,
    VAL_NUM,
    _baselines,
    _corr,
    _future_rollout_mse,
    _make_data,
    _make_teacher_sets,
    _r2,
    _sim_c,
    _threshold_flags,
)
from experiments.sho.run_stage1b_c_precision_refinement_diagnostic import (
    FeatureTCNEncoder,
    TeacherSet,
    _precision_metrics,
    _train_previous_references,
)
from experiments.sho.run_stage1b_c_refinement_optimizer_diagnostic import (
    _adam_refine,
    _newton_refine,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/unrolled_c_refinement_training_diagnostic")
BATCH_SIZE = 8
MAX_STEPS = 150
EVAL_INTERVAL = 50
PATIENCE = 150


def _prefix_loss_vector(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    losses = torch.zeros(c.shape[0], dtype=DTYPE, device=c.device)
    for j in J_ALL:
        starts = torch.arange(1, P - j, device=c.device)
        a = x[:, starts - 1]
        b = x[:, starts]
        preds = []
        for _ in range(j):
            nxt = c[:, None] * b - a
            preds.append(nxt)
            a, b = b, nxt
        pred = torch.stack(preds, dim=-1)
        truth = torch.stack([x[:, starts + k] for k in range(1, j + 1)], dim=-1)
        losses = losses + DRIFT_HEAVY[j] * torch.mean((pred - truth) ** 2, dim=(1, 2))
    return losses


def _prefix_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.mean(_prefix_loss_vector(c, positions))


def _c_from_raw(raw: torch.Tensor, parameterization: str) -> torch.Tensor:
    if parameterization == "constrained":
        return 2.0 * torch.tanh(raw)
    if parameterization == "unconstrained":
        return raw
    raise ValueError(parameterization)


def _raw_from_c(c: torch.Tensor, parameterization: str) -> torch.Tensor:
    if parameterization == "constrained":
        return torch.atanh((c / 2.0).clamp(-0.999999999, 0.999999999))
    if parameterization == "unconstrained":
        return c
    raise ValueError(parameterization)


def _newton_unroll(
    raw_init: torch.Tensor,
    positions: torch.Tensor,
    k_steps: int,
    parameterization: str,
    eta: float,
    damping: float,
    stop_gradient: bool,
) -> torch.Tensor:
    raw = raw_init
    for _ in range(k_steps):
        raw = raw.detach().requires_grad_(True) if stop_gradient else raw
        c = _c_from_raw(raw, parameterization)
        losses = _prefix_loss_vector(c, positions)
        grad = torch.autograd.grad(torch.sum(losses), raw, create_graph=True)[0]
        hdiag = []
        for i in range(raw.shape[0]):
            h = torch.autograd.grad(grad[i], raw, retain_graph=True, create_graph=not stop_gradient)[0][i]
            hdiag.append(h)
        hdiag_t = torch.stack(hdiag)
        step = eta * grad / (hdiag_t + damping)
        if stop_gradient:
            step = step.detach()
        raw = raw - step
        if parameterization == "unconstrained":
            raw = raw.clamp(-2.1, 2.1)
    return _c_from_raw(raw, parameterization)


def _adam_style_unroll(
    raw_init: torch.Tensor,
    positions: torch.Tensor,
    k_steps: int,
    parameterization: str,
    lr_refine: float,
    stop_gradient: bool,
) -> torch.Tensor:
    raw = raw_init
    m = torch.zeros_like(raw)
    v = torch.zeros_like(raw)
    beta1, beta2 = 0.9, 0.999
    for step in range(1, k_steps + 1):
        raw = raw.detach().requires_grad_(True) if stop_gradient else raw
        loss = _prefix_loss(_c_from_raw(raw, parameterization), positions)
        grad = torch.autograd.grad(loss, raw, create_graph=not stop_gradient)[0]
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad**2)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        raw = raw - lr_refine * m_hat / (torch.sqrt(v_hat) + 1e-8)
        if parameterization == "unconstrained":
            raw = raw.clamp(-2.1, 2.1)
    return _c_from_raw(raw, parameterization)


def _refine(
    raw_init: torch.Tensor,
    positions: torch.Tensor,
    method: str,
    k_steps: int,
    parameterization: str,
    stop_gradient: bool,
    eta: float = 1.0,
    damping: float = 1e-4,
    lr_refine: float = 1e-2,
) -> torch.Tensor:
    if k_steps == 0:
        return _c_from_raw(raw_init, parameterization)
    if method == "newton":
        return _newton_unroll(raw_init, positions, k_steps, parameterization, eta, damping, stop_gradient)
    if method == "adam_style":
        return _adam_style_unroll(raw_init, positions, k_steps, parameterization, lr_refine, stop_gradient)
    raise ValueError(method)


@torch.no_grad()
def _eval_model(model: nn.Module, dataset: TeacherSet, config: dict[str, Any], stage: str, name: str) -> dict[str, Any]:
    x = dataset.data.positions[:, :P]
    raw = model(x)
    c_init = _c_from_raw(raw, str(config["parameterization"]))
    # Evaluation refinement is allowed to build a graph locally; no future data is
    # used and no parameter gradients are retained.
    with torch.enable_grad():
        c_ref = _refine(
            raw.detach().clone().requires_grad_(True),
            dataset.data.positions,
            str(config["method"]),
            int(config["k_steps"]),
            str(config["parameterization"]),
            bool(config["stop_gradient"]),
            eta=float(config.get("eta", 1.0)),
            damping=float(config.get("damping", 1e-4)),
            lr_refine=float(config.get("lr_refine", 1e-2)),
        ).detach()
    row = _precision_metrics(stage, name, c_ref, dataset, dict(config))
    row.update(
        {
            "encoder_initialization_type": config.get("encoder_initialization_type", ""),
            "refinement_method": config["method"],
            "refinement_step_count_K": config["k_steps"],
            "gradients_through_refinement": not bool(config["stop_gradient"]),
            "beta_init": config["beta_init"],
            "c_init_rmse": float(torch.sqrt(torch.mean((c_init - dataset.c_free) ** 2)).item()),
            "c_refined_rmse": row["c_rmse"],
            "c_init_r2": _r2(c_init[:, None], dataset.c_free[:, None]),
            "c_refined_r2": row["c_r2_against_teacher"],
            "c_init_collapse_ratio": float((torch.std(c_init) / torch.std(dataset.c_free).clamp_min(1e-12)).item()),
            "c_refined_collapse_ratio": row["collapse_ratio"],
            "c_init_10_period_mse": _future_rollout_mse(c_init, dataset.data, 10.0)[0],
            "c_refined_10_period_mse": row["state_given_10_period_mse"],
            "validation_prefix_loss": float(_prefix_loss(c_ref, dataset.data.positions).item()),
        }
    )
    return row


def _train_unrolled(
    start_model: FeatureTCNEncoder,
    train: TeacherSet,
    val: TeacherSet,
    test: TeacherSet,
    config: dict[str, Any],
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = copy.deepcopy(start_model)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["lr_encoder"]))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + int(config["config_index"]) * 1000)
    n = train.data.positions.shape[0]
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        raw = model(pos[:, :P])
        c_init = _c_from_raw(raw, str(config["parameterization"]))
        c_ref = _refine(
            raw,
            pos,
            str(config["method"]),
            int(config["k_steps"]),
            str(config["parameterization"]),
            bool(config["stop_gradient"]),
            eta=float(config.get("eta", 1.0)),
            damping=float(config.get("damping", 1e-4)),
            lr_refine=float(config.get("lr_refine", 1e-2)),
        )
        loss_ref = _prefix_loss(c_ref, pos)
        loss_init = _prefix_loss(c_init, pos)
        loss = loss_ref + float(config["beta_init"]) * loss_init
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            row = _eval_model(model, val, config, "training_history", str(config["config_name"]))
            row.update({"step": step, "train_prefix_loss": float(loss.item()), "train_refined_prefix_loss": float(loss_ref.item()), "train_init_prefix_loss": float(loss_init.item())})
            history.append(row)
            score = float(row["validation_prefix_loss"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    final = _eval_model(model, test, config, "unrolled_training", str(config["config_name"]))
    final["best_validation_prefix_loss"] = best
    return model, history, final


def _teacher_row(dataset: TeacherSet) -> dict[str, Any]:
    row = _precision_metrics("baseline", "free_c_teacher", dataset.c_free, dataset, {"model_type": "free_c_teacher"})
    row.update({"c_init_rmse": 0.0, "c_refined_rmse": 0.0, "c_init_10_period_mse": row["state_given_10_period_mse"], "c_refined_10_period_mse": row["state_given_10_period_mse"]})
    return row


def _previous_baselines(test: TeacherSet) -> list[dict[str, Any]]:
    rows = [
        _teacher_row(test),
        {"stage": "baseline", "config_name": "previous_pure_amortized_tcn", "c_refined_rmse": 1.28e-3, "state_given_10_period_mse": 3.78e-3},
        {"stage": "baseline", "config_name": "previous_50_step_prefix_refinement", "c_refined_rmse": 1.66e-4, "state_given_10_period_mse": 6.10e-5},
        {"stage": "baseline", "config_name": "previous_best_refinement", "c_refined_rmse": 1.0e-12, "state_given_10_period_mse": 2.0e-29},
    ]
    rows.extend(_baselines(test))
    return rows


def _external_init_baselines(test: TeacherSet, train: TeacherSet, k_values: list[int]) -> list[dict[str, Any]]:
    gen = torch.Generator().manual_seed(SEED + 80808)
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    rand = avg + 0.01 * torch.randn(avg.shape, generator=gen, dtype=DTYPE)
    rows: list[dict[str, Any]] = []
    for init_name, c0 in [("average_c", avg), ("random_near_mean", rand)]:
        for k in k_values:
            if k == 0:
                c = c0
            else:
                refined = _newton_refine(c0, test, 1.0, 1e-4, "constrained", J_ALL, init_name)
                matches = [r for r in refined if r.get("selection") == "current" and int(r.get("refinement_steps", -1)) == k]
                if not matches:
                    continue
                row = dict(matches[0])
                row["stage"] = "baseline_external_initialization"
                rows.append(row)
                continue
            row = _precision_metrics("baseline_external_initialization", f"{init_name}_newton_K_{k}", c, test, {"initialization_type": init_name, "refinement_step_count_K": k})
            rows.append(row)
    return rows


def _config_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    idx = 0
    for k in [1, 2]:
        for beta in [0.0, 0.01]:
            for stop in [False, True]:
                configs.append(
                    {
                        "config_index": idx,
                        "config_name": f"newton_K{k}_beta{beta:g}_{'stopgrad' if stop else 'through'}",
                        "encoder_initialization_type": "reconstructed_previous_tcn",
                        "method": "newton",
                        "k_steps": k,
                        "parameterization": "constrained",
                        "eta": 1.0,
                        "damping": 1e-4,
                        "beta_init": beta,
                        "stop_gradient": stop,
                        "lr_encoder": 3e-4,
                    }
                )
                idx += 1
    for k in [2, 5, 10]:
        for lr_refine in [1e-2, 3e-3]:
            configs.append(
                {
                    "config_index": idx,
                    "config_name": f"adamstyle_K{k}_lrref{lr_refine:g}_through",
                    "encoder_initialization_type": "reconstructed_previous_tcn",
                    "method": "adam_style",
                    "k_steps": k,
                    "parameterization": "constrained",
                    "lr_refine": lr_refine,
                    "beta_init": 0.01,
                    "stop_gradient": False,
                    "lr_encoder": 3e-4,
                }
            )
            idx += 1
    # Pure amortized control.
    configs.append(
        {
            "config_index": idx,
            "config_name": "pure_amortized_K0_prefix_loss",
            "encoder_initialization_type": "reconstructed_previous_tcn",
            "method": "newton",
            "k_steps": 0,
            "parameterization": "constrained",
            "eta": 1.0,
            "damping": 1e-4,
            "beta_init": 0.1,
            "stop_gradient": False,
            "lr_encoder": 3e-4,
        }
    )
    return configs


def _threshold_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in THRESHOLDS:
        eligible = [r for r in rows if str(r.get(f"crosses_{threshold:g}", "")).lower() == "true"]
        best = min(rows, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))
        out.append(
            {
                "threshold": threshold,
                "num_crossing": len(eligible),
                "best_crossing_config": min(eligible, key=lambda r: float(r.get("state_given_10_period_mse", float("inf"))))["config_name"] if eligible else "",
                "best_overall_config": best["config_name"],
                "best_overall_mse": best["state_given_10_period_mse"],
            }
        )
    return out


def _plot_bar(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    vals, labels = [], []
    for r in rows:
        try:
            vals.append(float(r[key]))
            labels.append(str(r.get("config_name", "")))
        except Exception:
            pass
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(vals)), vals)
    if "mse" in key or "rmse" in key:
        plt.yscale("log")
    plt.xticks(range(len(vals)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel(key)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_by_k(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(7, 5))
    for method in ["newton", "adam_style"]:
        vals = [r for r in rows if r.get("refinement_method") == method and not bool(r.get("stop_gradient", False))]
        if vals:
            by_k: dict[int, float] = {}
            for r in vals:
                k = int(r["refinement_step_count_K"])
                by_k[k] = min(by_k.get(k, float("inf")), float(r[key]))
            plt.plot(sorted(by_k), [by_k[k] for k in sorted(by_k)], marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("K refinement steps")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "config_name", "refinement_method", "refinement_step_count_K", "gradients_through_refinement", "beta_init", "c_init_rmse", "c_refined_rmse", "c_refined_10_period_mse", "state_given_10_period_mse"]
    body = []
    for r in outputs["summary"]:
        body.append("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Unrolled c Refinement Training</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f3f3f3}</style>",
        "</head><body><h1>Stage 1B Unrolled c_i Refinement Training Diagnostic</h1>",
        "<p>Training loss is prefix-only recurrence consistency after unrolled refinement. Future rollout is evaluation only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 10100)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 10200)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 10300)
    print("building prefix-only free-c teachers", flush=True)
    train_t, val_t, test_t, teacher_hist = _make_teacher_sets(train_data, val_data, test_data)

    print("reconstructing previous TCN initialization", flush=True)
    base, finetuned, base_hist, audit_rows = _train_previous_references(train_t, val_t, test_t)
    start = finetuned

    baselines = _previous_baselines(test_t)
    external = _external_init_baselines(test_t, train_t, [0, 1, 2, 5])

    configs = _config_grid()
    results: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"training {cfg['config_name']}", flush=True)
        _, hist, row = _train_unrolled(start, train_t, val_t, test_t, cfg)
        histories.extend(hist)
        results.append(row)

    summary = baselines + external + results
    return {
        "summary": summary,
        "full_run_table": summary + teacher_hist + base_hist + audit_rows + histories,
        "unrolled_training_results": results,
        "refinement_step_results": results + external,
        "stop_gradient_comparison": [r for r in results if "stopgrad" in str(r.get("config_name", "")) or "through" in str(r.get("config_name", ""))],
        "baseline_comparison": baselines + external,
        "threshold_crossings": _threshold_summary(summary),
        "training_history": histories,
    }


def _interpret(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    results = outputs["unrolled_training_results"]
    best = min(results, key=lambda r: float(r["state_given_10_period_mse"]))
    through = [r for r in results if r.get("gradients_through_refinement")]
    stop = [r for r in results if not r.get("gradients_through_refinement")]
    best_through = min(through, key=lambda r: float(r["state_given_10_period_mse"])) if through else {}
    best_stop = min(stop, key=lambda r: float(r["state_given_10_period_mse"])) if stop else {}
    k_to_1e6 = min((int(r["refinement_step_count_K"]) for r in results if float(r["state_given_10_period_mse"]) < 1e-6), default=None)
    k_to_1e9 = min((int(r["refinement_step_count_K"]) for r in results if float(r["state_given_10_period_mse"]) < 1e-9), default=None)
    beta0 = min([r for r in results if float(r.get("beta_init", 0.0)) == 0.0], key=lambda r: float(r["state_given_10_period_mse"]), default={})
    beta001 = min([r for r in results if float(r.get("beta_init", -1.0)) == 0.01], key=lambda r: float(r["state_given_10_period_mse"]), default={})
    return {
        "training_through_refinement_improves": float(best_through.get("state_given_10_period_mse", float("inf"))) < float(best_stop.get("state_given_10_period_mse", float("inf"))),
        "k_needed_for_1e_6": k_to_1e6,
        "k_needed_for_1e_9": k_to_1e9,
        "best_method": best["config_name"],
        "best_10_period_mse": best["state_given_10_period_mse"],
        "best_c_refined_rmse": best["c_refined_rmse"],
        "best_through_refinement": best_through.get("config_name", ""),
        "best_stop_gradient": best_stop.get("config_name", ""),
        "beta_init_0_best_mse": beta0.get("state_given_10_period_mse", ""),
        "beta_init_0p01_best_mse": beta001.get("state_given_10_period_mse", ""),
        "preferred_method": "amortized_c_initialization_plus_small_fixed_prefix_refinement",
        "state_learning_gate": float(best["state_given_10_period_mse"]) < 1e-6,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, filename in [
        ("summary", "summary.csv"),
        ("full_run_table", "full_run_table.csv"),
        ("unrolled_training_results", "unrolled_training_results.csv"),
        ("refinement_step_results", "refinement_step_results.csv"),
        ("stop_gradient_comparison", "stop_gradient_comparison.csv"),
        ("baseline_comparison", "baseline_comparison.csv"),
        ("threshold_crossings", "threshold_crossings.csv"),
        ("training_history", "training_history.csv"),
    ]:
        _write_csv(OUTPUT_DIR / filename, outputs[key])
    interpretation = _interpret(outputs)
    setup = {
        "diagnostic": "stage1B_unrolled_c_refinement_training",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_true_c_labels_in_training": False,
        "uses_future_data_during_refinement": False,
        "uses_LS_inside_learned_encoder": False,
        "uses_closed_form_c_estimator": False,
        "main_training_loss": "prefix_multistep_recurrence_loss_after_unrolled_refinement",
        "runtime_note": "Previous checkpoint was unavailable; TCN initialization was reconstructed on the same split.",
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _plot_by_k(outputs["unrolled_training_results"], "state_given_10_period_mse", "mse_vs_refinement_steps.png")
    _plot_by_k(outputs["unrolled_training_results"], "c_refined_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_bar(outputs["stop_gradient_comparison"], "state_given_10_period_mse", "unrolled_vs_stop_gradient.png")
    _plot_bar(outputs["unrolled_training_results"], "c_init_rmse", "encoder_init_vs_refined_c.png")
    _plot_by_k(outputs["unrolled_training_results"], "state_given_10_period_mse", "future_rollout_mse_by_K.png")
    _plot_bar(outputs["threshold_crossings"], "num_crossing", "threshold_crossings_by_K.png")
    _plot_bar(outputs["baseline_comparison"], "state_given_10_period_mse", "baseline_comparison.png")
    _plot_bar(outputs["unrolled_training_results"], "max_trajectory_error", "worst_trajectory_examples.png")
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)


if __name__ == "__main__":
    main()
