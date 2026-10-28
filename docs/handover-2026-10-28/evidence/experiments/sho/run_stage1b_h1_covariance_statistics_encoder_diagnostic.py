"""Stage 1B H=1 covariance-statistics encoder diagnostic.

Compare a raw-prefix TCN with MLPs receiving the sufficient-looking
one-step transition statistics C00 and C10. Training samples a fresh clean
position-only trajectory batch at every step. Fixed sets and prefix-only
free-c teachers are used only for evaluation and diagnostics.
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
    SHOSet,
    TCNBlock,
    TeacherSet,
    _make_data,
    _make_teacher_sets,
    _r2,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
csv.field_size_limit(sys.maxsize)

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_covariance_statistics_encoder_diagnostic")
BATCH_SIZE = int(os.environ.get("H1COV_BATCH_SIZE", "32"))
VAL_N = int(os.environ.get("H1COV_VAL_N", "128"))
TEST_N = int(os.environ.get("H1COV_TEST_N", "128"))
LR = float(os.environ.get("H1COV_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1COV_MAX_STEPS", "30000"))
EVAL_INTERVAL = int(os.environ.get("H1COV_EVAL_INTERVAL", "200"))
PATIENCE = int(os.environ.get("H1COV_PATIENCE", "5000"))
GATE_MAX_ABS_ERROR = float(os.environ.get("H1COV_GATE_MAX_ABS_ERROR", "1e-6"))
SMOOTHMAX_TEMPERATURE = float(os.environ.get("H1COV_SMOOTHMAX_TEMPERATURE", "1e-6"))
COVARIANCE_EPSILON = float(os.environ.get("H1COV_EPSILON", "1e-12"))


class RawPrefixTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        return 2.0 * torch.tanh(self.head(features).squeeze(-1))


def _covariance_features(prefix: torch.Tensor, normalized: bool) -> torch.Tensor:
    h_t = torch.stack([prefix[:, :-2], prefix[:, 1:-1]], dim=-1)
    h_next = torch.stack([prefix[:, 1:-1], prefix[:, 2:]], dim=-1)
    c00 = torch.einsum("nti,ntj->nij", h_t, h_t)
    c10 = torch.einsum("nti,ntj->nij", h_next, h_t)
    if normalized:
        c00 = c00 / (torch.linalg.vector_norm(c00.reshape(c00.shape[0], -1), dim=1).reshape(-1, 1, 1) + COVARIANCE_EPSILON)
        c10 = c10 / (torch.linalg.vector_norm(c10.reshape(c10.shape[0], -1), dim=1).reshape(-1, 1, 1) + COVARIANCE_EPSILON)
    return torch.cat([c00.reshape(prefix.shape[0], -1), c10.reshape(prefix.shape[0], -1)], dim=1)


class CovarianceMLP(nn.Module):
    def __init__(self, normalized: bool) -> None:
        super().__init__()
        self.normalized = normalized
        self.net = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(self.net(_covariance_features(prefix, self.normalized)).squeeze(-1))


def _sample_prefix_batch(n: int, generator: torch.Generator) -> SHOSet:
    initial = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    lo, hi = OMEGA_RANGE
    omegas = float(lo) + (float(hi) - float(lo)) * torch.rand(n, generator=generator, dtype=DTYPE)
    times = torch.arange(P, dtype=DTYPE).unsqueeze(0).expand(n, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _stratified_subset(data: SHOSet, n: int) -> SHOSet:
    order = torch.argsort(data.omegas)
    points = torch.linspace(0, data.omegas.numel() - 1, n, dtype=DTYPE)
    indices = order[points.round().to(torch.long)]
    return SHOSet(positions=data.positions[indices], omegas=data.omegas[indices])


def _smoothmax(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1)
    if SMOOTHMAX_TEMPERATURE <= 0:
        return flat.max()
    maximum = flat.max()
    return maximum + SMOOTHMAX_TEMPERATURE * torch.log(torch.mean(torch.exp((flat - maximum) / SMOOTHMAX_TEMPERATURE)))


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _h1_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return _smoothmax(_h1_errors(c, positions) ** 2)


def _model(name: str) -> nn.Module:
    if name == "raw_prefix_tcn":
        return RawPrefixTCN().to(dtype=DTYPE)
    if name == "covariance_statistics_mlp":
        return CovarianceMLP(normalized=False).to(dtype=DTYPE)
    if name == "normalized_covariance_statistics_mlp":
        return CovarianceMLP(normalized=True).to(dtype=DTYPE)
    raise ValueError(name)


@torch.no_grad()
def _metrics(model: nn.Module, dataset: TeacherSet, name: str, split: str, step: int | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    c = model(dataset.data.positions[:, :P])
    c_error = c - dataset.c_free
    h1 = _h1_errors(c, dataset.data.positions)
    row: dict[str, Any] = {
        "encoder": name,
        "split": split,
        "step": step,
        "max_h1_abs_error": float(torch.abs(h1).max().item()),
        "rmse_h1_error": float(torch.sqrt(torch.mean(h1**2)).item()),
        "smoothmax_h1_squared_error": float(_h1_loss(c, dataset.data.positions).item()),
        "max_abs_c_error_vs_teacher": float(torch.abs(c_error).max().item()),
        "c_rmse_vs_teacher": float(torch.sqrt(torch.mean(c_error**2)).item()),
        "c_r2_vs_teacher": _r2(c[:, None], dataset.c_free[:, None]),
        "c_predicted_min": float(c.min().item()),
        "c_predicted_max": float(c.max().item()),
    }
    if extra:
        row.update(extra)
    return row


def _train(name: str, val: TeacherSet, test: TeacherSet) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    model = _model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + 417000 + list(MODELS).index(name) * 1000)
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefix_batch(BATCH_SIZE, generator)
        prediction = model(batch.positions)
        loss = _h1_loss(prediction, batch.positions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            common = {"train_loss": float(loss.detach().item()), "elapsed_seconds": time.perf_counter() - start}
            history.extend([
                _metrics(model, _TRAIN_MONITOR, name, "train_fresh_monitor", step, common),
                _metrics(model, val, name, "val", step, common),
            ])
            score = history[-1]["max_h1_abs_error"]
            if score < best:
                best = float(score)
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    finals = [_metrics(model, data, name, split, "best", {"best_validation_max_h1_abs_error": best}) for split, data in [("train_fresh_monitor", _TRAIN_MONITOR), ("val", val), ("test", test)]]
    train_final, val_final, test_final = finals
    summary = {
        "encoder": name,
        "gate_passed": bool(val_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR and test_final["max_h1_abs_error"] <= GATE_MAX_ABS_ERROR),
        "gate_threshold": GATE_MAX_ABS_ERROR,
        "train_max_h1_error": train_final["max_h1_abs_error"],
        "val_max_h1_error": val_final["max_h1_abs_error"],
        "test_max_h1_error": test_final["max_h1_abs_error"],
        "val_rmse_h1_error": val_final["rmse_h1_error"],
        "test_rmse_h1_error": test_final["rmse_h1_error"],
        "val_c_rmse_vs_teacher": val_final["c_rmse_vs_teacher"],
        "test_c_rmse_vs_teacher": test_final["c_rmse_vs_teacher"],
        "val_max_abs_c_error_vs_teacher": val_final["max_abs_c_error_vs_teacher"],
        "test_max_abs_c_error_vs_teacher": test_final["max_abs_c_error_vs_teacher"],
        "best_validation_max_h1_error": best,
        "runtime_seconds": time.perf_counter() - start,
    }
    predictions = []
    for split, data in [("train_fresh_monitor", _TRAIN_MONITOR), ("val", val), ("test", test)]:
        with torch.no_grad():
            predicted = model(data.data.positions[:, :P])
        for index in range(predicted.numel()):
            predictions.append({"row_type": "prediction", "encoder": name, "split": split, "trajectory_index": index, "c_pred": float(predicted[index].item()), "c_teacher": float(data.c_free[index].item()), "c_error": float((predicted[index] - data.c_free[index]).item())})
    return model, history + finals, summary, predictions


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["encoder"]: row for row in summary}
    raw = by_name["raw_prefix_tcn"]
    covariance = by_name["covariance_statistics_mlp"]
    normalized = by_name["normalized_covariance_statistics_mlp"]
    covariance_better = (
        float(covariance["val_max_h1_error"]) < float(raw["val_max_h1_error"])
        and float(covariance["test_max_h1_error"]) < float(raw["test_max_h1_error"])
    )
    normalization_helps = (
        float(normalized["val_max_h1_error"]) < float(covariance["val_max_h1_error"])
        and float(normalized["test_max_h1_error"]) < float(covariance["test_max_h1_error"])
    )
    covariance_passes = bool(covariance["gate_passed"] or normalized["gate_passed"])
    raw_feature_extraction_bottleneck = covariance_passes and not bool(raw["gate_passed"])
    amortized_scalar_bottleneck = not covariance_passes
    any_pass = any(row["gate_passed"] for row in summary)
    return {
        "covariance_statistics_passes_h1": bool(covariance["gate_passed"]),
        "normalized_covariance_statistics_passes_h1": bool(normalized["gate_passed"]),
        "raw_prefix_tcn_passes_h1": bool(raw["gate_passed"]),
        "normalization_helps": normalization_helps,
        "covariance_generalizes_better_than_raw_prefix": covariance_better,
        "raw_prefix_bottleneck_mainly_feature_extraction": raw_feature_extraction_bottleneck,
        "amortized_scalar_prediction_remaining_bottleneck": amortized_scalar_bottleneck,
        "passed_encoders": [row["encoder"] for row in summary if row["gate_passed"]],
        "answer": "At least one encoder passes the strict H=1 gate." if any_pass else "No encoder passes the strict H=1 gate; amortized scalar prediction remains a bottleneck.",
    }


def _plot_history(history: list[dict[str, Any]]) -> None:
    rows = [row for row in history if row["split"] == "val" and str(row["step"]).isdigit()]
    plt.figure(figsize=(8, 5))
    for name in MODELS:
        subset = [row for row in rows if row["encoder"] == name]
        plt.plot([int(row["step"]) for row in subset], [float(row["max_h1_abs_error"]) for row in subset], label=name)
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle="--", linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("validation maximum H=1 absolute error")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "max_h1_error_vs_training_step.png", dpi=170)
    plt.close()


def _plot_predictions(predictions: list[dict[str, Any]]) -> None:
    rows = [row for row in predictions if row["split"] == "test"]
    plt.figure(figsize=(6, 6))
    for name in MODELS:
        subset = [row for row in rows if row["encoder"] == name]
        plt.scatter([row["c_teacher"] for row in subset], [row["c_pred"] for row in subset], s=12, label=name)
    values = [row["c_teacher"] for row in rows] + [row["c_pred"] for row in rows]
    lower, upper = min(values), max(values)
    plt.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    plt.xlabel("prefix-only free-c teacher")
    plt.ylabel("predicted c")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_c.png", dpi=170)
    plt.close()


def _plot_comparison(summary: list[dict[str, Any]]) -> None:
    values = [float(row["test_max_h1_error"]) for row in summary]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(values)), values)
    plt.axhline(GATE_MAX_ABS_ERROR, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(values)), [row["encoder"] for row in summary], rotation=18, ha="right")
    plt.yscale("log")
    plt.ylabel("test maximum H=1 absolute error")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tcn_vs_covariance_statistics.png", dpi=170)
    plt.close()


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    columns = ["encoder", "gate_passed", "train_max_h1_error", "val_max_h1_error", "test_max_h1_error", "test_c_rmse_vs_teacher", "test_max_abs_c_error_vs_teacher"]
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>" for row in summary)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>H=1 covariance statistics encoder diagnostic</title>", "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px}</style>", "</head><body><h1>Stage 1B H=1 Covariance-Statistics Encoder Diagnostic</h1>", "<p>Fresh clean position-only batches train all encoders. Validation and test sets are fixed; free-c teachers are diagnostic-only.</p>", "<h2>Answers</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>", "<h2>Encoder comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + rows + "</tbody></table></body></html>"]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


MODELS = ("raw_prefix_tcn", "covariance_statistics_mlp", "normalized_covariance_statistics_mlp")
_TRAIN_MONITOR: TeacherSet


def main() -> None:
    global _TRAIN_MONITOR
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    val_data = _stratified_subset(_make_data(VAL_N, TRAIN_DATA_SEED + 417100), VAL_N)
    test_data = _stratified_subset(_make_data(TEST_N, TEST_DATA_SEED + 417200), TEST_N)
    monitor_data = _stratified_subset(_make_data(VAL_N, TRAIN_DATA_SEED + 417300), VAL_N)
    _TRAIN_MONITOR, val, test, _ = _make_teacher_sets(monitor_data, val_data, test_data)
    summary: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for name in MODELS:
        print(f"training {name}", flush=True)
        _, model_history, final, model_predictions = _train(name, val, test)
        summary.append(final)
        history.extend(model_history)
        predictions.extend(model_predictions)
    interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "training_history.csv", history)
    _write_csv(OUTPUT_DIR / "encoder_comparison.csv", summary + predictions)
    _write_html(summary, interpretation)
    _plot_history(history)
    _plot_predictions(predictions)
    _plot_comparison(summary)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
