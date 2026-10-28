"""Stage 1B covariance-first, precision-second curriculum diagnostic.

Focused follow-up to the covariance + SuperLoss encoder diagnostic.

Compares:

  A. structural -> structural raw -> precision
  B. structural only
  C. precision only

The model is the same TCN prefix encoder:

    prefix -> c_pred
    K(c) = [[0, 1], [-1, c]]

Training is prefix-only. Future rollout is evaluation-only.
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
from experiments.sho.run_stage1b_covariance_superloss_encoder_diagnostic import (
    _covariance_loss_vector,
    _superloss_weights,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/covariance_first_precision_second_curriculum")
BATCH_SIZE = int(os.environ.get("CFPS_BATCH_SIZE", "32"))
EVAL_INTERVAL = int(os.environ.get("CFPS_EVAL_INTERVAL", "1000"))
PATIENCE = int(os.environ.get("CFPS_PATIENCE", "1000000000"))
ALPHA_COV = float(os.environ.get("CFPS_ALPHA_COV", "0.1"))
SUPERLOSS_LAMBDA = float(os.environ.get("CFPS_SUPERLOSS_LAMBDA", "0.1"))
TAU_EMA = float(os.environ.get("CFPS_TAU_EMA", "0.98"))
REFINE_STEPS = [0, 20, 50, 100]
REFINE_LR = 1e-3


def _curricula() -> list[dict[str, Any]]:
    return [
        {
            "name": "structural_to_precision",
            "stages": [
                {"stage_name": "stage1_super_structural", "loss_kind": "structural", "use_superloss": True, "lr": 1e-4, "max_steps": 30000},
                {"stage_name": "stage2_raw_structural", "loss_kind": "structural", "use_superloss": False, "lr": 1e-4, "max_steps": 30000},
                {"stage_name": "stage3_precision_lr3e-5", "loss_kind": "precision", "use_superloss": False, "lr": 3e-5, "max_steps": 50000},
                {"stage_name": "stage3_precision_lr1e-5", "loss_kind": "precision", "use_superloss": False, "lr": 1e-5, "max_steps": 50000},
            ],
        },
        {
            "name": "structural_only",
            "stages": [
                {"stage_name": "structural_only_part1", "loss_kind": "structural", "use_superloss": False, "lr": 1e-4, "max_steps": 30000},
                {"stage_name": "structural_only_part2", "loss_kind": "structural", "use_superloss": False, "lr": 1e-4, "max_steps": 30000},
                {"stage_name": "structural_only_part3", "loss_kind": "structural", "use_superloss": False, "lr": 1e-4, "max_steps": 50000},
                {"stage_name": "structural_only_part4", "loss_kind": "structural", "use_superloss": False, "lr": 1e-4, "max_steps": 50000},
            ],
        },
        {
            "name": "precision_only",
            "stages": [
                {"stage_name": "precision_only_lr3e-5", "loss_kind": "precision", "use_superloss": False, "lr": 3e-5, "max_steps": 50000},
                {"stage_name": "precision_only_lr1e-5", "loss_kind": "precision", "use_superloss": False, "lr": 1e-5, "max_steps": 50000},
            ],
        },
    ]


def _batch_dataset(dataset: TeacherSet, idx: torch.Tensor) -> TeacherSet:
    data = type(dataset.data)(
        positions=dataset.data.positions[idx],
        omegas=dataset.data.omegas[idx],
    )
    return TeacherSet(data, dataset.c_free[idx], dataset.mu, dataset.sigma)


def _precision_loss_vector(c: torch.Tensor, dataset: TeacherSet) -> torch.Tensor:
    z_pred = (c - dataset.mu) / dataset.sigma
    return (z_pred - dataset.z) ** 2 + (c - dataset.c_free) ** 2


def _loss_vector(model: CInitTCN, dataset: TeacherSet, loss_kind: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c = _predict_c(model, dataset)
    multi = _prefix_loss_vector(c, dataset.data.positions)
    cov = _covariance_loss_vector(c, dataset.data.positions)
    precision = _precision_loss_vector(c, dataset)
    if loss_kind == "structural":
        total = multi + ALPHA_COV * cov
    elif loss_kind == "precision":
        total = precision
    else:
        raise ValueError(f"unknown loss kind: {loss_kind}")
    return total, multi, cov, precision


def _eval_model(model: CInitTCN, dataset: TeacherSet, curriculum: str, stage_name: str, split: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        total_struct = _prefix_loss_vector(c, dataset.data.positions) + ALPHA_COV * _covariance_loss_vector(c, dataset.data.positions)
        precision = _precision_loss_vector(c, dataset)
        cov = _covariance_loss_vector(c, dataset.data.positions)
        multi = _prefix_loss_vector(c, dataset.data.positions)
    row = _metrics(
        "stage_eval",
        curriculum,
        c,
        dataset,
        {
            "curriculum": curriculum,
            "stage_name": stage_name,
            "split": split,
            "covariance_residual": float(cov.mean().item()),
            "multi_prefix_loss": float(multi.mean().item()),
            "structural_objective": float(total_struct.mean().item()),
            "precision_objective": float(precision.mean().item()),
            "parameter_count": _param_count(model),
        },
    )
    if extra:
        row.update(extra)
    return row


def _run_stage(
    model: CInitTCN,
    train: TeacherSet,
    val: TeacherSet,
    curriculum: str,
    stage_cfg: dict[str, Any],
    seed_offset: int,
) -> tuple[CInitTCN, list[dict[str, Any]], dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=float(stage_cfg["lr"]))
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 211000 + seed_offset)
    n = train.data.positions.shape[0]
    tau = torch.tensor(1.0, dtype=DTYPE)
    start = time.perf_counter()
    for step in range(1, int(stage_cfg["max_steps"]) + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        batch = _batch_dataset(train, idx)
        losses, multi, cov, precision = _loss_vector(model, batch, str(stage_cfg["loss_kind"]))
        if bool(stage_cfg["use_superloss"]):
            tau = TAU_EMA * tau + (1.0 - TAU_EMA) * losses.detach().mean()
            weights = _superloss_weights(losses, tau, SUPERLOSS_LAMBDA)
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
        if step == 1 or step % EVAL_INTERVAL == 0 or step == int(stage_cfg["max_steps"]):
            row = _eval_model(
                model,
                val,
                curriculum,
                str(stage_cfg["stage_name"]),
                "val",
                {
                    "step": step,
                    "stage_elapsed_seconds": time.perf_counter() - start,
                    "train_loss": float(loss.item()),
                    "train_multi_prefix_loss": float(multi.mean().detach().item()),
                    "train_covariance_residual": float(cov.mean().detach().item()),
                    "train_precision_objective": float(precision.mean().detach().item()),
                    "loss_kind": stage_cfg["loss_kind"],
                    "use_superloss": bool(stage_cfg["use_superloss"]),
                    "superloss_tau": float(tau.item()),
                    "superloss_weight_mean": weight_mean,
                    "superloss_weight_min": weight_min,
                    "fraction_heavily_downweighted": frac_down,
                    "learning_rate": stage_cfg["lr"],
                    "stage_max_steps": stage_cfg["max_steps"],
                },
            )
            history.append(row)
            score = float(row["c_rmse"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    best_row = _eval_model(
        model,
        val,
        curriculum,
        str(stage_cfg["stage_name"]),
        "val",
        {
            "best_validation_c_rmse": best,
            "loss_kind": stage_cfg["loss_kind"],
            "use_superloss": bool(stage_cfg["use_superloss"]),
            "learning_rate": stage_cfg["lr"],
            "stage_max_steps": stage_cfg["max_steps"],
        },
    )
    return model, history, best_row


def _train_curriculum(curriculum_cfg: dict[str, Any], train: TeacherSet, val: TeacherSet, test: TeacherSet, index: int) -> tuple[CInitTCN, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = CInitTCN().to(dtype=DTYPE)
    history: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for j, stage_cfg in enumerate(curriculum_cfg["stages"]):
        print(f"  {curriculum_cfg['name']} / {stage_cfg['stage_name']}", flush=True)
        model, hist, stage_row = _run_stage(model, train, val, str(curriculum_cfg["name"]), stage_cfg, index * 10000 + j * 1000)
        history.extend(hist)
        stage_rows.append(stage_row)
    final = _eval_model(model, test, str(curriculum_cfg["name"]), "final", "test", {"stage": "encoder_final"})
    return model, history, stage_rows, final


def _refinement_rows(model: CInitTCN, dataset: TeacherSet, curriculum: str) -> list[dict[str, Any]]:
    with torch.no_grad():
        init = _predict_c(model, dataset)
    rows = []
    for steps in REFINE_STEPS:
        _current, best, runtime = _refine_c(init, dataset, "adam", REFINE_LR, steps)
        row = _metrics(
            "secondary_adam_refinement",
            f"{curriculum}_adam_refine_steps_{steps}",
            best,
            dataset,
            {
                "curriculum": curriculum,
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


def _threshold_steps(rows: list[dict[str, Any]], curriculum: str, threshold: float) -> int | None:
    vals = [
        int(r["refinement_steps"])
        for r in rows
        if r.get("curriculum") == curriculum and float(r["future_10_period_mse"]) < threshold
    ]
    return min(vals) if vals else None


def _interpret(finals: list[dict[str, Any]], stage_rows: list[dict[str, Any]], refinement: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(r["curriculum"]): r for r in finals}
    best = min(finals, key=lambda r: float(r["future_10_period_mse"]))
    structural = by_name.get("structural_only", {})
    precision = by_name.get("precision_only", {})
    curriculum = by_name.get("structural_to_precision", {})
    return {
        "best_encoder_only_curriculum": best.get("curriculum", ""),
        "best_encoder_only_10_period_mse": best.get("future_10_period_mse", ""),
        "best_encoder_only_c_rmse": best.get("c_rmse", ""),
        "structural_pretraining_improves_final_precision_vs_precision_only": float(curriculum.get("c_rmse", float("inf"))) < float(precision.get("c_rmse", float("inf"))),
        "final_precision_stage_outperforms_structural_only": float(curriculum.get("future_10_period_mse", float("inf"))) < float(structural.get("future_10_period_mse", float("inf"))),
        "final_precision_stage_outperforms_precision_only": float(curriculum.get("future_10_period_mse", float("inf"))) < float(precision.get("future_10_period_mse", float("inf"))),
        "min_refinement_steps_to_1e_6": {name: _threshold_steps(refinement, name, 1e-6) for name in by_name},
        "min_refinement_steps_to_1e_9": {name: _threshold_steps(refinement, name, 1e-9) for name in by_name},
        "min_refinement_steps_to_1e_12": {name: _threshold_steps(refinement, name, 1e-12) for name in by_name},
        "stage1_superloss_present": any(str(r.get("stage_name")) == "stage1_super_structural" for r in stage_rows),
        "precision_loss_note": "implemented as standardized c-teacher MSE plus raw c-teacher MSE; the literal minus sign would make the loss unbounded in the wrong direction",
    }


def _plot_stage(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(9, 5))
    for name in sorted({str(r["curriculum"]) for r in rows}):
        vals = [r for r in rows if str(r["curriculum"]) == name]
        plt.plot(range(len(vals)), [float(r[key]) for r in vals], marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("stage index")
    plt.ylabel(key)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_scatter(rows: list[dict[str, Any]], xkey: str, ykey: str, filename: str) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter([float(r[xkey]) for r in rows], [float(r[ykey]) for r in rows], s=40)
    for r in rows:
        plt.annotate(str(r["curriculum"]), (float(r[xkey]), float(r[ykey])), fontsize=7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_refinement(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for name in sorted({str(r["curriculum"]) for r in rows}):
        vals = sorted([r for r in rows if str(r["curriculum"]) == name], key=lambda r: int(r["refinement_steps"]))
        plt.plot([int(r["refinement_steps"]) for r in vals], [float(r["future_10_period_mse"]) for r in vals], marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("Adam refinement steps")
    plt.ylabel("10-period rollout MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "refinement_steps_comparison.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["curriculum", "c_rmse", "c_r2_against_teacher", "collapse_ratio", "covariance_residual", "future_10_period_mse", "parameter_count"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Covariance-First Precision-Second Curriculum</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Covariance-First, Precision-Second Curriculum</h1>",
        "<p>TCN prefix encoder predicts trajectory-specific c_i from clean position-only prefixes. Future rollout is evaluation-only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Encoder-Only Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 181000)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 181100)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 181200)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, _teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    history: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    for i, curriculum_cfg in enumerate(_curricula()):
        print(f"training {curriculum_cfg['name']}", flush=True)
        model, hist, stages, final = _train_curriculum(curriculum_cfg, train, val, test, i)
        history.extend(hist)
        stage_rows.extend(stages)
        finals.append(final)
        refinement.extend(_refinement_rows(model, test, str(curriculum_cfg["name"])))
    return {
        "summary": finals,
        "stage_results": stage_rows,
        "refinement_results": refinement,
        "training_history": history,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "stage_results.csv", outputs["stage_results"])
    _write_csv(OUTPUT_DIR / "refinement_results.csv", outputs["refinement_results"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    interpretation = _interpret(outputs["summary"], outputs["stage_results"], outputs["refinement_results"])
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_covariance_first_precision_second_curriculum",
            "dtype": str(DTYPE),
            "prefix_length_P": P,
            "operator": "K(c)=[[0,1],[-1,c]]",
            "J": J_ALL,
            "J_weights": DRIFT_HEAVY,
            "alpha_cov": ALPHA_COV,
            "superloss_lambda": SUPERLOSS_LAMBDA,
            "precision_loss": "MSE(z_pred,z_teacher)+MSE(c_pred,c_teacher)",
            "uses_velocity": False,
            "uses_labels": False,
            "uses_LS_or_closed_form_estimator": False,
            "uses_future_data_in_training": False,
        },
    )
    _plot_stage(outputs["stage_results"], "c_rmse", "encoder_c_rmse_by_stage.png")
    _plot_stage(outputs["stage_results"], "future_10_period_mse", "encoder_rollout_mse_by_stage.png")
    _plot_scatter(outputs["summary"], "covariance_residual", "c_rmse", "covariance_residual_vs_c_rmse.png")
    _plot_refinement(outputs["refinement_results"])
    _write_html(outputs, interpretation)
    return interpretation


def main() -> None:
    outputs = _run_all()
    interpretation = _write_outputs(outputs)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
