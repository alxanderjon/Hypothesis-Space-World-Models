"""Stage 1B strict H=1 local-transition discovery diagnostic.

This diagnostic asks whether a prefix encoder can discover a trajectory-level
one-step recurrence constant from clean position-only prefixes before any
longer-horizon curriculum is attempted.

Training uses only prefix one-step prediction errors. Prefix-only free-c
teachers are computed for diagnostics only.
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
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    P,
    TeacherSet,
    TCNBlock,
    _make_data,
    _make_teacher_sets,
    _r2,
)
from experiments.sho.run_stage1b_covariance_superloss_encoder_diagnostic import _covariance_loss_vector
from experiments.sho.run_stage1b_encoder_initialized_gradient_refinement import _param_count
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

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_local_transition_discovery")
TRAIN_N = 32
VAL_N = 16
TEST_N = 16
POOL_N = 128
LR = float(os.environ.get("H1DISC_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1DISC_MAX_STEPS", "30000"))
EVAL_INTERVAL = int(os.environ.get("H1DISC_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("H1DISC_PATIENCE", "5000"))
GATE_MAX_ABS_ERROR = float(os.environ.get("H1DISC_GATE_MAX_ABS_ERROR", "1e-6"))
SMOOTHMAX_TEMPERATURE = float(os.environ.get("H1DISC_SMOOTHMAX_TEMPERATURE", "1e-6"))


class WholePrefixTCN(nn.Module):
    """Whole-prefix baseline: prefix -> c_i."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, d) for d in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.silu(self.inp(prefix.unsqueeze(1)))
        y = self.blocks(y).mean(dim=-1)
        raw = self.head(y).squeeze(-1)
        return 2.0 * torch.tanh(raw)


class LocalTransitionAggregator(nn.Module):
    """Shared triplet encoder with summed evidence over prefix transitions."""

    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        triplets = torch.stack([prefix[:, :-2], prefix[:, 1:-1], prefix[:, 2:]], dim=-1)
        evidence = self.phi(triplets)
        z = evidence.sum(dim=1) / math.sqrt(float(triplets.shape[1]))
        raw = self.head(z).squeeze(-1)
        return 2.0 * torch.tanh(raw)


class CovarianceFeatureEncoder(nn.Module):
    """Small MLP on flattened C00 and C10 transition covariance features."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        h_t = torch.stack([prefix[:, :-2], prefix[:, 1:-1]], dim=-1)
        h_next = torch.stack([prefix[:, 1:-1], prefix[:, 2:]], dim=-1)
        c00 = torch.einsum("nti,ntj->nij", h_t, h_t)
        c10 = torch.einsum("nti,ntj->nij", h_next, h_t)
        scale = float(h_t.shape[1])
        features = torch.cat([(c00 / scale).reshape(prefix.shape[0], -1), (c10 / scale).reshape(prefix.shape[0], -1)], dim=1)
        raw = self.net(features).squeeze(-1)
        return 2.0 * torch.tanh(raw)


def _make_encoder(name: str) -> nn.Module:
    if name == "whole_prefix_tcn":
        return WholePrefixTCN().to(dtype=DTYPE)
    if name == "local_transition_aggregation":
        return LocalTransitionAggregator().to(dtype=DTYPE)
    if name == "covariance_feature_encoder":
        return CovarianceFeatureEncoder().to(dtype=DTYPE)
    raise ValueError(name)


def _stratified_subset(data: Any, n: int) -> Any:
    order = torch.argsort(data.omegas)
    positions = torch.linspace(0, data.omegas.numel() - 1, n, dtype=DTYPE)
    idx = order[positions.round().to(torch.long)]
    return type(data)(positions=data.positions[idx], omegas=data.omegas[idx])


def _predict_c(model: nn.Module, dataset: TeacherSet) -> torch.Tensor:
    return model(dataset.data.positions[:, :P])


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _smoothmax(values: torch.Tensor, temperature: float = SMOOTHMAX_TEMPERATURE) -> torch.Tensor:
    flat = values.reshape(-1)
    if temperature <= 0:
        return torch.max(flat)
    m = torch.max(flat)
    return m + temperature * torch.log(torch.mean(torch.exp((flat - m) / temperature)))


def _h1_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return _smoothmax(_h1_errors(c, positions) ** 2)


@torch.no_grad()
def _max_h1_abs_error(c: torch.Tensor, positions: torch.Tensor) -> float:
    return float(torch.max(torch.abs(_h1_errors(c, positions))).item())


def _metrics(model: nn.Module, dataset: TeacherSet, encoder_name: str, split: str, step: int | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    with torch.no_grad():
        c = _predict_c(model, dataset)
        c_err = c - dataset.c_free
        cov = _covariance_loss_vector(c, dataset.data.positions)
        h1_max = _max_h1_abs_error(c, dataset.data.positions)
    row: dict[str, Any] = {
        "encoder": encoder_name,
        "split": split,
        "step": step,
        "max_h1_abs_error": h1_max,
        "max_h1_squared_error": h1_max**2,
        "smoothmax_h1_squared_error": float(_h1_loss(c, dataset.data.positions).item()),
        "max_abs_c_error_vs_teacher": float(torch.max(torch.abs(c_err)).item()),
        "c_rmse_vs_teacher": float(torch.sqrt(torch.mean(c_err**2)).item()),
        "c_r2_vs_teacher": _r2(c[:, None], dataset.c_free[:, None]),
        "c_bias_vs_teacher": float(torch.mean(c_err).item()),
        "predicted_c_std": float(torch.std(c).item()),
        "teacher_c_std": float(torch.std(dataset.c_free).item()),
        "collapse_ratio": float((torch.std(c) / torch.std(dataset.c_free).clamp_min(1e-12)).item()),
        "covariance_residual": float(cov.mean().item()),
        "covariance_residual_max": float(cov.max().item()),
        "parameter_count": _param_count(model),
    }
    if extra:
        row.update(extra)
    return row


def _train_encoder(name: str, train: TeacherSet, val: TeacherSet, test: TeacherSet) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = _make_encoder(name)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        c = _predict_c(model, train)
        loss = _h1_loss(c, train.data.positions)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            train_row = _metrics(
                model,
                train,
                name,
                "train",
                step,
                {"train_loss": float(loss.detach().item()), "elapsed_seconds": time.perf_counter() - start},
            )
            val_row = _metrics(
                model,
                val,
                name,
                "val",
                step,
                {"train_loss": float(loss.detach().item()), "elapsed_seconds": time.perf_counter() - start},
            )
            history.extend([train_row, val_row])
            score = float(val_row["max_h1_abs_error"])
            if score < best:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    train_final = _metrics(model, train, name, "train", "best", {"best_validation_max_h1_abs_error": best})
    val_final = _metrics(model, val, name, "val", "best", {"best_validation_max_h1_abs_error": best})
    test_final = _metrics(model, test, name, "test", "best", {"best_validation_max_h1_abs_error": best})
    passed = bool(val_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR and test_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR)
    final = {
        "encoder": name,
        "gate_passed": passed,
        "gate_threshold": GATE_MAX_ABS_ERROR,
        "train_max_h1_abs_error": train_final["max_h1_abs_error"],
        "val_max_h1_abs_error": val_final["max_h1_abs_error"],
        "test_max_h1_abs_error": test_final["max_h1_abs_error"],
        "train_c_rmse_vs_teacher": train_final["c_rmse_vs_teacher"],
        "val_c_rmse_vs_teacher": val_final["c_rmse_vs_teacher"],
        "test_c_rmse_vs_teacher": test_final["c_rmse_vs_teacher"],
        "train_max_abs_c_error_vs_teacher": train_final["max_abs_c_error_vs_teacher"],
        "val_max_abs_c_error_vs_teacher": val_final["max_abs_c_error_vs_teacher"],
        "test_max_abs_c_error_vs_teacher": test_final["max_abs_c_error_vs_teacher"],
        "train_succeeds_but_validation_fails": bool(train_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR and val_final["max_h1_abs_error"] > GATE_MAX_ABS_ERROR),
        "covariance_residual_test": test_final["covariance_residual"],
        "parameter_count": test_final["parameter_count"],
        "best_validation_max_h1_abs_error": best,
        "runtime_seconds": time.perf_counter() - start,
    }
    history.extend([train_final, val_final, test_final])
    return model, history, final


def _prediction_rows(models: dict[str, nn.Module], datasets: dict[str, TeacherSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        for split, dataset in datasets.items():
            with torch.no_grad():
                c = _predict_c(model, dataset)
            for idx in range(c.numel()):
                rows.append(
                    {
                        "encoder": name,
                        "split": split,
                        "trajectory_index": idx,
                        "omega": float(dataset.data.omegas[idx].item()),
                        "c_pred": float(c[idx].item()),
                        "c_teacher": float(dataset.c_free[idx].item()),
                        "c_error": float((c[idx] - dataset.c_free[idx]).item()),
                    }
                )
    return rows


def _plot_history(history: list[dict[str, Any]]) -> None:
    rows = [r for r in history if r["split"] == "val" and str(r["step"]).isdigit()]
    plt.figure(figsize=(8, 5))
    for name in sorted({r["encoder"] for r in rows}):
        sub = [r for r in rows if r["encoder"] == name]
        plt.plot([int(r["step"]) for r in sub], [float(r["max_h1_abs_error"]) for r in sub], label=name)
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("validation maximum H=1 absolute error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "maximum_h1_error_vs_training_step.png", dpi=170)
    plt.close()


def _plot_pred_vs_teacher(pred_rows: list[dict[str, Any]]) -> None:
    rows = [r for r in pred_rows if r["split"] == "test"]
    plt.figure(figsize=(6, 6))
    for name in sorted({r["encoder"] for r in rows}):
        sub = [r for r in rows if r["encoder"] == name]
        plt.scatter([float(r["c_teacher"]) for r in sub], [float(r["c_pred"]) for r in sub], s=18, label=name, alpha=0.8)
    vals = [float(r["c_teacher"]) for r in rows] + [float(r["c_pred"]) for r in rows]
    lo, hi = min(vals), max(vals)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("prefix-only free-c teacher")
    plt.ylabel("predicted c")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_ci.png", dpi=170)
    plt.close()


def _plot_train_vs_val(history: list[dict[str, Any]]) -> None:
    rows = [r for r in history if str(r["step"]).isdigit() and r["split"] in {"train", "val"}]
    plt.figure(figsize=(8, 5))
    for name in sorted({r["encoder"] for r in rows}):
        for split, style in [("train", "-"), ("val", "--")]:
            sub = [r for r in rows if r["encoder"] == name and r["split"] == split]
            plt.plot([int(r["step"]) for r in sub], [float(r["max_h1_abs_error"]) for r in sub], style, label=f"{name} {split}")
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle=":", linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("maximum H=1 absolute error")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "train_vs_validation_maximum_error.png", dpi=170)
    plt.close()


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    cols = [
        "encoder",
        "gate_passed",
        "train_max_h1_abs_error",
        "val_max_h1_abs_error",
        "test_max_h1_abs_error",
        "test_c_rmse_vs_teacher",
        "test_max_abs_c_error_vs_teacher",
        "covariance_residual_test",
        "train_succeeds_but_validation_fails",
        "parameter_count",
    ]
    rows = "\n".join("<tr>" + "".join(f"<td>{escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in summary)
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>H=1 Local Transition Discovery</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px;border-radius:6px}</style>",
        "</head><body><h1>Stage 1B Strict H=1 Local-Transition Discovery</h1>",
        "<p>Encoders were trained only with a smooth maximum over prefix one-step squared errors. Free-c teachers are diagnostics only.</p>",
        "<h2>Interpretation</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>",
        "<h2>Encoder Comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(c)}</th>" for c in cols) + "</tr></thead><tbody>" + rows + "</tbody></table>",
        "</body></html>",
    ]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(body), encoding="utf-8")


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    best_val = min(summary, key=lambda r: float(r["val_max_h1_abs_error"]))
    best_test = min(summary, key=lambda r: float(r["test_max_h1_abs_error"]))
    passed = [r["encoder"] for r in summary if r["gate_passed"]]
    tcn = next(r for r in summary if r["encoder"] == "whole_prefix_tcn")
    local = next(r for r in summary if r["encoder"] == "local_transition_aggregation")
    cov = next(r for r in summary if r["encoder"] == "covariance_feature_encoder")
    return {
        "any_architecture_passed_h1": bool(passed),
        "passed_encoders": passed,
        "best_validation_encoder": best_val["encoder"],
        "best_validation_max_h1_abs_error": best_val["val_max_h1_abs_error"],
        "best_test_encoder": best_test["encoder"],
        "best_test_max_h1_abs_error": best_test["test_max_h1_abs_error"],
        "local_transition_aggregation_generalizes_better_than_tcn": float(local["test_max_h1_abs_error"]) < float(tcn["test_max_h1_abs_error"]),
        "covariance_features_generalize_better_than_tcn": float(cov["test_max_h1_abs_error"]) < float(tcn["test_max_h1_abs_error"]),
        "covariance_features_generalize_better_than_local_aggregation": float(cov["test_max_h1_abs_error"]) < float(local["test_max_h1_abs_error"]),
        "failure_mode": (
            "at least one encoder passes strict validation and test gate"
            if passed
            else "no encoder reaches the 1e-6 max-error gate; local one-step discovery remains a precision/generalization bottleneck"
        ),
    }


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 211000)
    val_pool = _make_data(POOL_N, TRAIN_DATA_SEED + 211100)
    test_pool = _make_data(POOL_N, TEST_DATA_SEED + 211200)
    train_data = _stratified_subset(train_pool, TRAIN_N)
    val_data = _stratified_subset(val_pool, VAL_N)
    test_data = _stratified_subset(test_pool, TEST_N)
    print("building prefix-only free-c teachers", flush=True)
    train, val, test, teacher_history = _make_teacher_sets(train_data, val_data, test_data)
    all_history: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    best_models: dict[str, nn.Module] = {}
    for name in ["whole_prefix_tcn", "local_transition_aggregation", "covariance_feature_encoder"]:
        print(f"training {name}", flush=True)
        model, history, final = _train_encoder(name, train, val, test)
        all_history.extend(history)
        summary.append(final)
        best_models[name] = model
    prediction_rows = _prediction_rows(best_models, {"train": train, "val": val, "test": test})
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "encoder_comparison.csv", summary)
    _write_csv(OUTPUT_DIR / "training_history.csv", all_history)
    _write_csv(OUTPUT_DIR / "predictions_vs_teacher.csv", prediction_rows)
    _write_csv(OUTPUT_DIR / "teacher_history.csv", teacher_history)
    interpretation = _interpret(summary)
    _write_json(
        OUTPUT_DIR / "setup.json",
        {
            "diagnostic": "stage1B_h1_local_transition_discovery",
            "dtype": str(DTYPE),
            "prefix_length_P": P,
            "train_n": TRAIN_N,
            "val_n": VAL_N,
            "test_n": TEST_N,
            "h1_only": True,
            "learning_rate": LR,
            "max_steps": MAX_STEPS,
            "eval_interval": EVAL_INTERVAL,
            "patience": PATIENCE,
            "gate_max_abs_error": GATE_MAX_ABS_ERROR,
            "smoothmax_temperature": SMOOTHMAX_TEMPERATURE,
            "uses_superloss": False,
            "uses_velocity": False,
            "uses_labels": False,
            "uses_ls_or_closed_form_estimator": False,
            "uses_future_data_in_training": False,
        },
    )
    _write_json(OUTPUT_DIR / "interpretation.json", interpretation)
    _write_json(OUTPUT_DIR / "metrics.json", {"summary": summary, "training_history": all_history, "predictions_vs_teacher": prediction_rows})
    _plot_history(all_history)
    _plot_train_vs_val(all_history)
    _plot_pred_vs_teacher(prediction_rows)
    _write_html(summary, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
