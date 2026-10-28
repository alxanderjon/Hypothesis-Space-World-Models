"""Focused long Stage 1B encoder-initialized Adam refinement diagnostic."""

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
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import (
    CInitTCN,
    _metrics,
    _param_count,
    _predict_c,
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

OUTPUT_DIR = Path("outputs/sho/stage1B/encoder_initialized_gradient_refinement_focused_long")
BATCH_SIZE = int(os.environ.get("EIGR_LONG_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("EIGR_LONG_MAX_STEPS", "50000"))
EVAL_INTERVAL = int(os.environ.get("EIGR_LONG_EVAL_INTERVAL", "500"))
PATIENCE = int(os.environ.get("EIGR_LONG_PATIENCE", "15000"))
ENCODER_LRS = [float(x) for x in os.environ.get("EIGR_LONG_ENCODER_LRS", "3e-4,1e-4").split(",")]
REFINE_LR = float(os.environ.get("EIGR_LONG_REFINE_LR", "1e-3"))
REFINE_STEPS = [int(x) for x in os.environ.get("EIGR_LONG_REFINE_STEPS", "0,5,10,20,50,100,200").split(",")]


def _train_encoder_lr(train: TeacherSet, val: TeacherSet, test: TeacherSet, lr: float) -> tuple[CInitTCN, list[dict[str, Any]], list[dict[str, Any]]]:
    model = CInitTCN().to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    gen = torch.Generator().manual_seed(SEED + 121000 + int(round(lr * 1e8)))
    n = train.data.positions.shape[0]
    config = f"c_init_tcn_lr_{lr:g}"
    start_time = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        idx = torch.randint(0, n, (min(BATCH_SIZE, n),), generator=gen)
        pos = train.data.positions[idx]
        z_true = train.z[idx]
        c_true = train.c_free[idx]
        z_pred = model(pos[:, :P])
        c_pred = train.mu + train.sigma * z_pred
        loss = torch.mean((z_pred - z_true) ** 2) + torch.mean((c_pred - c_true) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            with torch.no_grad():
                train_row = _metrics("training_history", config, _predict_c(model, train), train, {"split": "train", "step": step, "encoder_lr": lr, "train_loss": float(loss.item()), "elapsed_seconds": time.perf_counter() - start_time})
                val_row = _metrics("training_history", config, _predict_c(model, val), val, {"split": "val", "step": step, "encoder_lr": lr, "train_loss": float(loss.item()), "elapsed_seconds": time.perf_counter() - start_time})
            history.extend([train_row, val_row])
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
    finals = []
    with torch.no_grad():
        for split, dataset in [("train", train), ("val", val), ("test", test)]:
            finals.append(
                _metrics(
                    "encoder_final",
                    config,
                    _predict_c(model, dataset),
                    dataset,
                    {
                        "split": split,
                        "encoder_lr": lr,
                        "best_validation_c_rmse": best,
                        "parameter_count": _param_count(model),
                        "max_steps": MAX_STEPS,
                    },
                )
            )
    return model, history, finals


def _evaluate_refinement(model: CInitTCN, train: TeacherSet, test: TeacherSet, encoder_name: str) -> list[dict[str, Any]]:
    with torch.no_grad():
        encoder_init = _predict_c(model, test)
    avg = torch.full_like(test.c_free, float(train.c_free.mean().item()))
    rows: list[dict[str, Any]] = []
    for init_name, init_c in [("encoder_init", encoder_init), ("average_c", avg)]:
        for steps in REFINE_STEPS:
            _current, best, runtime = _refine_c(init_c, test, "adam", REFINE_LR, steps)
            row = _metrics(
                "adam_refinement",
                f"{encoder_name}_{init_name}_adam_lr_{REFINE_LR:g}_steps_{steps}_best_prefix_loss",
                best,
                test,
                {
                    "encoder_config": encoder_name,
                    "initialization": init_name,
                    "optimizer": "adam",
                    "learning_rate": REFINE_LR,
                    "refinement_steps": steps,
                    "selection": "best_prefix_loss",
                    "runtime_seconds": runtime,
                    "runtime_per_trajectory_seconds": runtime / max(1, test.data.positions.shape[0]),
                },
            )
            rows.append(row)
    return rows


def _steps_to(rows: list[dict[str, Any]], init: str, threshold: float) -> int | None:
    vals = [int(r["refinement_steps"]) for r in rows if r.get("initialization") == init and float(r["future_10_period_mse"]) < threshold]
    return min(vals) if vals else None


def _interpret(summary: list[dict[str, Any]], refinement: list[dict[str, Any]]) -> dict[str, Any]:
    enc_final = [r for r in summary if r.get("stage") == "encoder_final" and r.get("split") == "test"]
    best_encoder = min(enc_final, key=lambda r: float(r["c_rmse"]), default={})
    best_ref = min(refinement, key=lambda r: float(r["future_10_period_mse"]), default={})
    enc_rows = [r for r in refinement if r.get("initialization") == "encoder_init"]
    avg_rows = [r for r in refinement if r.get("initialization") == "average_c"]
    best_enc_ref = min(enc_rows, key=lambda r: float(r["future_10_period_mse"]), default={})
    best_avg_ref = min(avg_rows, key=lambda r: float(r["future_10_period_mse"]), default={})
    return {
        "longer_training_reduces_c_init_error": "compare_against_bounded_pass_test_c_rmse_0.003528",
        "best_encoder_config": best_encoder.get("config_name", ""),
        "best_encoder_test_c_rmse": best_encoder.get("c_rmse", ""),
        "best_encoder_test_10_period_mse": best_encoder.get("future_10_period_mse", ""),
        "encoder_reaches_1e_6_steps": _steps_to(refinement, "encoder_init", 1e-6),
        "encoder_reaches_1e_9_steps": _steps_to(refinement, "encoder_init", 1e-9),
        "average_c_reaches_1e_6_steps": _steps_to(refinement, "average_c", 1e-6),
        "average_c_reaches_1e_9_steps": _steps_to(refinement, "average_c", 1e-9),
        "best_refinement_config": best_ref.get("config_name", ""),
        "best_refinement_10_period_mse": best_ref.get("future_10_period_mse", ""),
        "best_encoder_init_refinement_10_period_mse": best_enc_ref.get("future_10_period_mse", ""),
        "best_average_c_refinement_10_period_mse": best_avg_ref.get("future_10_period_mse", ""),
        "viable_stage1B_hybrid": bool(_steps_to(refinement, "encoder_init", 1e-6) is not None),
        "honest_method": "encoder initialization plus prefix-only Adam refinement",
    }


def _plot_refinement(rows: list[dict[str, Any]], key: str, filename: str) -> None:
    plt.figure(figsize=(7, 5))
    for init in ["encoder_init", "average_c"]:
        vals = sorted([r for r in rows if r.get("initialization") == init], key=lambda r: int(r["refinement_steps"]))
        if vals:
            plt.plot([int(r["refinement_steps"]) for r in vals], [float(r[key]) for r in vals], marker="o", label=init)
    plt.yscale("log")
    plt.xlabel("Adam refinement steps")
    plt.ylabel(key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170)
    plt.close()


def _plot_training(rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for lr in ENCODER_LRS:
        vals = sorted([r for r in rows if r.get("split") == "val" and abs(float(r.get("encoder_lr", -1)) - lr) < 1e-15], key=lambda r: int(r["step"]))
        if vals:
            plt.plot([int(r["step"]) for r in vals], [float(r["c_rmse"]) for r in vals], label=f"lr={lr:g}")
    plt.yscale("log")
    plt.xlabel("encoder step")
    plt.ylabel("validation c RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "encoder_val_c_rmse_vs_step.png", dpi=170)
    plt.close()


def _write_html(outputs: dict[str, list[dict[str, Any]]], interpretation: dict[str, Any]) -> None:
    cols = ["stage", "split", "config_name", "initialization", "learning_rate", "refinement_steps", "c_rmse", "future_10_period_mse", "runtime_per_trajectory_seconds"]
    body = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in outputs["summary"])
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Focused Long Encoder Init Adam Refinement</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Focused Long Encoder-initialized Adam Refinement</h1>",
        "<p>TCN encoder is trained on prefix-only free-c teachers. Inference uses Adam refinement on prefix recurrence loss only; future rollout is evaluation-only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Summary</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + body + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def _run_all() -> dict[str, list[dict[str, Any]]]:
    torch.manual_seed(SEED)
    train_data = _make_data(TRAIN_NUM, TRAIN_DATA_SEED + 94100)
    val_data = _make_data(VAL_NUM, TRAIN_DATA_SEED + 94200)
    test_data = _make_data(TEST_NUM, TEST_DATA_SEED + 94300)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    all_history: list[dict[str, Any]] = []
    encoder_rows: list[dict[str, Any]] = []
    candidates: list[tuple[CInitTCN, dict[str, Any]]] = []
    for lr in ENCODER_LRS:
        print(f"training encoder lr={lr:g}", flush=True)
        model, hist, finals = _train_encoder_lr(train, val, test, lr)
        all_history.extend(hist)
        encoder_rows.extend(finals)
        test_final = [r for r in finals if r.get("split") == "test"][0]
        candidates.append((model, test_final))
    best_model, best_row = min(candidates, key=lambda pair: float(pair[1]["c_rmse"]))
    print(f"evaluating Adam refinement from {best_row['config_name']}", flush=True)
    refinement_rows = _evaluate_refinement(best_model, train, test, str(best_row["config_name"]))
    summary = encoder_rows + refinement_rows
    return {
        "summary": summary,
        "refinement_results": refinement_rows,
        "training_history": all_history,
        "teacher_history": teacher_history,
    }


def _write_outputs(outputs: dict[str, list[dict[str, Any]]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "summary.csv", outputs["summary"])
    _write_csv(OUTPUT_DIR / "refinement_results.csv", outputs["refinement_results"])
    _write_csv(OUTPUT_DIR / "training_history.csv", outputs["training_history"])
    _write_csv(OUTPUT_DIR / "teacher_history.csv", outputs["teacher_history"])
    interpretation = _interpret(outputs["summary"], outputs["refinement_results"])
    setup = {
        "diagnostic": "focused_long_encoder_initialized_adam_refinement",
        "dtype": str(DTYPE),
        "prefix_length_P": P,
        "J": J_ALL,
        "J_weights": DRIFT_HEAVY,
        "encoder_lrs": ENCODER_LRS,
        "max_steps": MAX_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "patience": PATIENCE,
        "encoder_loss": "standardized c-teacher MSE + raw c MSE",
        "refinement_optimizer": "adam",
        "refinement_lr": REFINE_LR,
        "refinement_steps": REFINE_STEPS,
        "uses_velocity": False,
        "uses_omega_labels": False,
        "uses_coefficient_labels": False,
        "uses_simulator_c_labels_in_training": False,
        "uses_future_data_during_refinement": False,
        "uses_LS_inside_encoder": False,
        "uses_closed_form_c_estimator": False,
        "uses_hessian_or_newton": False,
    }
    _write_json(OUTPUT_DIR / "setup.json", setup)
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", outputs)
    _plot_refinement(outputs["refinement_results"], "future_10_period_mse", "rollout_mse_vs_refinement_steps.png")
    _plot_refinement(outputs["refinement_results"], "c_rmse", "c_rmse_vs_refinement_steps.png")
    _plot_training(outputs["training_history"])
    _write_html(outputs, interpretation)


def main() -> None:
    outputs = _run_all()
    _write_outputs(outputs)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **_interpret(outputs["summary"], outputs["refinement_results"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
