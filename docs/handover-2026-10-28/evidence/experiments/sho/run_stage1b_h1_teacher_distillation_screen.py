"""Quick Stage 1B H=1 prefix-derived free-c teacher-distillation screen."""

from __future__ import annotations

import copy
import csv
import json
import math
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
import torch
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, SHOSet, TCNBlock, _r2
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, OMEGA_RANGE, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_teacher_distillation_screen")
BATCH_SIZE = int(os.environ.get("H1DISTILL_BATCH_SIZE", "32"))
MONITOR_N = int(os.environ.get("H1DISTILL_MONITOR_N", "64"))
VAL_N = int(os.environ.get("H1DISTILL_VAL_N", "128"))
TEST_N = int(os.environ.get("H1DISTILL_TEST_N", "128"))
LR = float(os.environ.get("H1DISTILL_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1DISTILL_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("H1DISTILL_EVAL_INTERVAL", "200"))
TEACHER_STEPS = int(os.environ.get("H1DISTILL_TEACHER_STEPS", "450"))
TEACHER_LR = float(os.environ.get("H1DISTILL_TEACHER_LR", "1e-2"))
TEACHER_INITIAL_C = float(os.environ.get("H1DISTILL_TEACHER_INITIAL_C", "1.95"))


class TCNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        return self.head(features).squeeze(-1)


class LocalTransitionSetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        transitions = torch.stack((prefix[:, :-2], prefix[:, 1:-1], prefix[:, 2:]), dim=-1)
        # Sum aggregation requested by the diagnostic; the fixed scale keeps its
        # magnitude stable without changing the set representation.
        z = self.phi(transitions).sum(dim=1) / math.sqrt(float(transitions.shape[1]))
        return self.head(z).squeeze(-1)


def _sample_prefixes(n: int, generator: torch.Generator) -> SHOSet:
    initial = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    lo, hi = OMEGA_RANGE
    omegas = float(lo) + (float(hi) - float(lo)) * torch.rand(n, generator=generator, dtype=DTYPE)
    times = torch.arange(P, dtype=DTYPE).unsqueeze(0).expand(n, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial, omegas, times)
    return SHOSet(positions=positions, omegas=omegas)


def _fixed_prefixes(n: int, seed: int) -> SHOSet:
    return _sample_prefixes(n, torch.Generator().manual_seed(seed))


def _free_c_teacher(positions: torch.Tensor) -> torch.Tensor:
    """Optimize independent c values using only the supplied observed prefix."""
    raw = nn.Parameter(torch.full((positions.shape[0],), TEACHER_INITIAL_C, dtype=DTYPE))
    optimizer = torch.optim.Adam([raw], lr=TEACHER_LR)
    best = raw.detach().clone()
    best_loss = float("inf")
    for _ in range(TEACHER_STEPS):
        optimizer.zero_grad()
        # H=1 prefix residual only. This is deliberately optimized rather than
        # solved by LS or a closed-form recurrence estimator.
        x = positions[:, :P]
        residual = raw[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]
        loss = torch.mean(residual**2)
        loss.backward()
        optimizer.step()
        score = float(loss.detach().item())
        if score < best_loss:
            best_loss = score
            best = raw.detach().clone()
    return best


def _predict_c(model: nn.Module, positions: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return mean + scale * model(positions[:, :P])


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


@torch.no_grad()
def _metrics(model: nn.Module, data: SHOSet, teacher: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor, encoder: str, split: str, step: int | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction = _predict_c(model, data.positions, mean, scale)
    c_error = prediction - teacher
    h1 = _h1_errors(prediction, data.positions)
    row: dict[str, Any] = {
        "encoder": encoder,
        "split": split,
        "step": step,
        "c_rmse": float(torch.sqrt(torch.mean(c_error**2)).item()),
        "max_abs_c_error": float(torch.abs(c_error).max().item()),
        "c_r2": _r2(prediction[:, None], teacher[:, None]),
        "max_h1_abs_error": float(torch.abs(h1).max().item()),
        "rmse_h1_error": float(torch.sqrt(torch.mean(h1**2)).item()),
    }
    if extra:
        row.update(extra)
    return row


def _make_model(name: str) -> nn.Module:
    if name == "tcn_encoder":
        return TCNEncoder().to(dtype=DTYPE)
    if name == "local_transition_set_encoder":
        return LocalTransitionSetEncoder().to(dtype=DTYPE)
    raise ValueError(name)


def _train(name: str, monitor: tuple[SHOSet, torch.Tensor], val: tuple[SHOSet, torch.Tensor], test: tuple[SHOSet, torch.Tensor], mean: torch.Tensor, scale: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = _make_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + 631000 + MODELS.index(name) * 1000)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefixes(BATCH_SIZE, generator)
        teacher = _free_c_teacher(batch.positions)
        target_z = (teacher - mean) / scale
        predicted_z = model(batch.positions)
        loss = torch.mean((predicted_z - target_z) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            common = {"train_batch_standardized_c_mse": float(loss.detach().item()), "elapsed_seconds": time.perf_counter() - start}
            train_row = _metrics(model, *monitor, mean, scale, name, "train_monitor", step, common)
            val_row = _metrics(model, *val, mean, scale, name, "val", step, common)
            history.extend((train_row, val_row))
            if float(val_row["c_rmse"]) < best_score:
                best_score = float(val_row["c_rmse"])
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    finals = [_metrics(model, *dataset, mean, scale, name, split, "best", {"best_step": best_step}) for split, dataset in (("train_monitor", monitor), ("val", val), ("test", test))]
    history.extend(finals)
    train_final, val_final, test_final = finals
    summary = {
        "encoder": name,
        "best_step": best_step,
        "train_c_rmse": train_final["c_rmse"],
        "val_c_rmse": val_final["c_rmse"],
        "test_c_rmse": test_final["c_rmse"],
        "val_max_abs_c_error": val_final["max_abs_c_error"],
        "test_max_abs_c_error": test_final["max_abs_c_error"],
        "val_max_h1_abs_error": val_final["max_h1_abs_error"],
        "test_max_h1_abs_error": test_final["max_h1_abs_error"],
        "val_c_r2": val_final["c_r2"],
        "test_c_r2": test_final["c_r2"],
        "train_to_val_c_rmse_ratio": float(val_final["c_rmse"]) / max(float(train_final["c_rmse"]), 1e-30),
        "runtime_seconds": time.perf_counter() - start,
    }
    return history, summary


def _plot(history: list[dict[str, Any]]) -> None:
    rows = [row for row in history if row["split"] == "val" and str(row["step"]).isdigit()]
    plt.figure(figsize=(8, 5))
    for name in MODELS:
        subset = [row for row in rows if row["encoder"] == name]
        plt.plot([int(row["step"]) for row in subset], [float(row["c_rmse"]) for row in subset], label=name)
    plt.axhline(1e-3, color="gray", linestyle="--", linewidth=1, label="previous ~1e-3 level")
    plt.axhline(1e-6, color="black", linestyle=":", linewidth=1, label="1e-6 target")
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("validation c RMSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_c_rmse_vs_training_step.png", dpi=170)
    plt.close()


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["encoder"]: row for row in summary}
    tcn, local = by_name["tcn_encoder"], by_name["local_transition_set_encoder"]
    best = min(summary, key=lambda row: float(row["val_c_rmse"]))
    substantial = float(best["val_c_rmse"]) < 5e-4 and float(best["test_c_rmse"]) < 5e-4
    return {
        "direct_teacher_distillation_beats_previous_1e-3_level": substantial,
        "local_transition_encoder_beats_tcn_on_validation_and_test": float(local["val_c_rmse"]) < float(tcn["val_c_rmse"]) and float(local["test_c_rmse"]) < float(tcn["test_c_rmse"]),
        "best_encoder": best["encoder"],
        "best_validation_c_rmse": best["val_c_rmse"],
        "precision_regime": "moving_toward_1e-6" if float(best["val_c_rmse"]) < 1e-4 else "plateauing_near_or_above_1e-3",
        "longer_run_justified": substantial,
    }


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    columns = ["encoder", "best_step", "train_c_rmse", "val_c_rmse", "test_c_rmse", "val_max_abs_c_error", "test_max_abs_c_error", "val_max_h1_abs_error", "test_max_h1_abs_error", "val_c_r2", "test_c_r2"]
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns) + "</tr>" for row in summary)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>H=1 teacher distillation screen</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px}</style></head><body>", "<h1>Stage 1B H=1 Teacher-Distillation Screen</h1>", "<p>Fresh trajectories and fresh prefix-only optimized free-c teachers were used every training batch. Teacher labels were standardized using the fixed train-monitor teacher distribution.</p>", "<h2>Answers</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>", "<h2>Results</h2><table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + rows + "</tbody></table>", "</body></html>"]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


MODELS = ("tcn_encoder", "local_transition_set_encoder")


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building fixed prefix-only teachers", flush=True)
    monitor_data = _fixed_prefixes(MONITOR_N, TRAIN_DATA_SEED + 631100)
    val_data = _fixed_prefixes(VAL_N, TRAIN_DATA_SEED + 631200)
    test_data = _fixed_prefixes(TEST_N, TEST_DATA_SEED + 631300)
    monitor = (monitor_data, _free_c_teacher(monitor_data.positions))
    val = (val_data, _free_c_teacher(val_data.positions))
    test = (test_data, _free_c_teacher(test_data.positions))
    mean = monitor[1].mean()
    scale = monitor[1].std().clamp_min(1e-12)
    history: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name in MODELS:
        print(f"training {name}", flush=True)
        model_history, model_summary = _train(name, monitor, val, test, mean, scale)
        history.extend(model_history)
        summary.append(model_summary)
    interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary)
    _write_csv(OUTPUT_DIR / "training_history.csv", history)
    _plot(history)
    _write_html(summary, interpretation)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
