"""Focused H=1 exact-versus-learned transition-statistic diagnostic."""

from __future__ import annotations

import copy
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
import torch
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, TCNBlock, _r2
from experiments.sho.run_stage1b_h1_teacher_distillation_screen import _fixed_prefixes, _free_c_teacher, _sample_prefixes
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_transition_statistic_diagnostic")
PREVIOUS_SUMMARY = Path("outputs/sho/stage1B/h1_teacher_distillation_screen/summary.csv")
BATCH_SIZE = int(os.environ.get("H1STAT_BATCH_SIZE", "32"))
MONITOR_N = int(os.environ.get("H1STAT_MONITOR_N", "64"))
VAL_N = int(os.environ.get("H1STAT_VAL_N", "128"))
TEST_N = int(os.environ.get("H1STAT_TEST_N", "128"))
LR = float(os.environ.get("H1STAT_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1STAT_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("H1STAT_EVAL_INTERVAL", "200"))
SMOOTHMAX_TEMPERATURE = float(os.environ.get("H1STAT_SMOOTHMAX_TEMPERATURE", "1e-4"))
AUX_WEIGHT = float(os.environ.get("H1STAT_AUX_WEIGHT", "1.0"))
GATE = 1e-6


def _statistics(prefix: torch.Tensor) -> torch.Tensor:
    x_prev, x, x_next = prefix[:, :-2], prefix[:, 1:-1], prefix[:, 2:]
    a = torch.sum(x * (x_prev + x_next), dim=1)
    b = torch.sum(x**2, dim=1)
    return torch.stack((a, b), dim=1)


def _smoothmax(values: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(-1)
    maximum = flat.max()
    if SMOOTHMAX_TEMPERATURE <= 0:
        return maximum
    return maximum + SMOOTHMAX_TEMPERATURE * torch.log(
        torch.mean(torch.exp((flat - maximum) / SMOOTHMAX_TEMPERATURE))
    )


class CHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, standardized_statistics: torch.Tensor) -> torch.Tensor:
        return self.net(standardized_statistics).squeeze(-1)


class ExactStatisticsMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_head = CHead()

    def forward(self, prefix: torch.Tensor, stat_mean: torch.Tensor, stat_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        true_z = (_statistics(prefix) - stat_mean) / stat_scale
        return self.c_head(true_z), true_z


class LearnedStatisticsTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.statistics_head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 2))
        self.c_head = CHead()

    def forward(self, prefix: torch.Tensor, stat_mean: torch.Tensor, stat_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        predicted_z = self.statistics_head(features)
        return self.c_head(predicted_z), predicted_z


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _predict(model: nn.Module, positions: torch.Tensor, stat_mean: torch.Tensor, stat_scale: torch.Tensor, c_mean: torch.Tensor, c_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_c_z, predicted_stat_z = model(positions[:, :P], stat_mean, stat_scale)
    return c_mean + c_scale * predicted_c_z, predicted_stat_z


@torch.no_grad()
def _metrics(model: nn.Module, data: Any, teacher: torch.Tensor, stat_mean: torch.Tensor, stat_scale: torch.Tensor, c_mean: torch.Tensor, c_scale: torch.Tensor, encoder: str, split: str, step: int | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction, predicted_stat_z = _predict(model, data.positions, stat_mean, stat_scale, c_mean, c_scale)
    true_statistics = _statistics(data.positions[:, :P])
    predicted_statistics = stat_mean + stat_scale * predicted_stat_z
    stat_error = predicted_statistics - true_statistics
    c_error = prediction - teacher
    h1 = _h1_errors(prediction, data.positions)
    row: dict[str, Any] = {
        "encoder": encoder,
        "split": split,
        "step": step,
        "max_abs_c_error": float(torch.abs(c_error).max().item()),
        "c_rmse": float(torch.sqrt(torch.mean(c_error**2)).item()),
        "c_r2": _r2(prediction[:, None], teacher[:, None]),
        "max_h1_abs_error": float(torch.abs(h1).max().item()),
        "rmse_h1_error": float(torch.sqrt(torch.mean(h1**2)).item()),
        "a_rmse": float(torch.sqrt(torch.mean(stat_error[:, 0] ** 2)).item()),
        "b_rmse": float(torch.sqrt(torch.mean(stat_error[:, 1] ** 2)).item()),
        "standardized_statistics_rmse": float(torch.sqrt(torch.mean((predicted_stat_z - (true_statistics - stat_mean) / stat_scale) ** 2)).item()),
    }
    if extra:
        row.update(extra)
    return row


def _previous_baseline() -> dict[str, Any]:
    with PREVIOUS_SUMMARY.open(newline="", encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if row["encoder"] == "tcn_encoder")
    return {
        "encoder": "previous_raw_prefix_tcn",
        "best_step": row["best_step"],
        "gate_passed": False,
        "val_max_abs_c_error": row["val_max_abs_c_error"],
        "test_max_abs_c_error": row["test_max_abs_c_error"],
        "val_c_rmse": row["val_c_rmse"],
        "test_c_rmse": row["test_c_rmse"],
        "val_max_h1_abs_error": row["val_max_h1_abs_error"],
        "test_max_h1_abs_error": row["test_max_h1_abs_error"],
        "val_c_r2": row["val_c_r2"],
        "test_c_r2": row["test_c_r2"],
        "val_a_rmse": "",
        "test_a_rmse": "",
        "val_b_rmse": "",
        "test_b_rmse": "",
        "source": str(PREVIOUS_SUMMARY),
    }


def _train(monitor: tuple[Any, torch.Tensor], val: tuple[Any, torch.Tensor], test: tuple[Any, torch.Tensor], stat_mean: torch.Tensor, stat_scale: torch.Tensor, c_mean: torch.Tensor, c_scale: torch.Tensor) -> tuple[dict[str, nn.Module], list[dict[str, Any]], list[dict[str, Any]]]:
    models: dict[str, nn.Module] = {
        "exact_statistics_mlp": ExactStatisticsMLP().to(dtype=DTYPE),
        "learned_statistics_tcn": LearnedStatisticsTCN().to(dtype=DTYPE),
    }
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=LR) for name, model in models.items()}
    best_states = {name: copy.deepcopy(model.state_dict()) for name, model in models.items()}
    best_scores = {name: float("inf") for name in models}
    best_steps = {name: 0 for name in models}
    generator = torch.Generator().manual_seed(SEED + 742000)
    history: list[dict[str, Any]] = []
    starts = {name: time.perf_counter() for name in models}
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefixes(BATCH_SIZE, generator)
        teacher = _free_c_teacher(batch.positions)
        true_stat_z = (_statistics(batch.positions) - stat_mean) / stat_scale
        target_c_z = (teacher - c_mean) / c_scale
        for name, model in models.items():
            predicted_c_z, predicted_stat_z = model(batch.positions, stat_mean, stat_scale)
            c_loss = _smoothmax((predicted_c_z - target_c_z) ** 2)
            aux_loss = torch.mean((predicted_stat_z - true_stat_z) ** 2) if name == "learned_statistics_tcn" else torch.zeros((), dtype=DTYPE)
            loss = c_loss + AUX_WEIGHT * aux_loss
            optimizers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizers[name].step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                common = {"train_c_smoothmax": float(c_loss.detach().item()), "train_statistics_aux_mse": float(aux_loss.detach().item()), "train_total_loss": float(loss.detach().item()), "elapsed_seconds": time.perf_counter() - starts[name]}
                history.append(_metrics(model, *monitor, stat_mean, stat_scale, c_mean, c_scale, name, "train_monitor", step, common))
                val_row = _metrics(model, *val, stat_mean, stat_scale, c_mean, c_scale, name, "val", step, common)
                history.append(val_row)
                score = float(val_row["max_h1_abs_error"])
                if score < best_scores[name]:
                    best_scores[name] = score
                    best_steps[name] = step
                    best_states[name] = copy.deepcopy(model.state_dict())
    summary: list[dict[str, Any]] = []
    for name, model in models.items():
        model.load_state_dict(best_states[name])
        finals = [_metrics(model, *dataset, stat_mean, stat_scale, c_mean, c_scale, name, split, "best", {"best_step": best_steps[name]}) for split, dataset in (("train_monitor", monitor), ("val", val), ("test", test))]
        history.extend(finals)
        _, val_final, test_final = finals
        summary.append({
            "encoder": name,
            "best_step": best_steps[name],
            "gate_passed": bool(val_final["max_h1_abs_error"] <= GATE and test_final["max_h1_abs_error"] <= GATE),
            "val_max_abs_c_error": val_final["max_abs_c_error"], "test_max_abs_c_error": test_final["max_abs_c_error"],
            "val_c_rmse": val_final["c_rmse"], "test_c_rmse": test_final["c_rmse"],
            "val_max_h1_abs_error": val_final["max_h1_abs_error"], "test_max_h1_abs_error": test_final["max_h1_abs_error"],
            "val_c_r2": val_final["c_r2"], "test_c_r2": test_final["c_r2"],
            "val_a_rmse": val_final["a_rmse"], "test_a_rmse": test_final["a_rmse"],
            "val_b_rmse": val_final["b_rmse"], "test_b_rmse": test_final["b_rmse"],
            "source": "current_run",
        })
    return models, history, summary


def _plot_history(history: list[dict[str, Any]]) -> None:
    rows = [row for row in history if row["split"] == "val" and str(row["step"]).isdigit()]
    plt.figure(figsize=(8, 5))
    for name in ("exact_statistics_mlp", "learned_statistics_tcn"):
        subset = [row for row in rows if row["encoder"] == name]
        plt.plot([int(row["step"]) for row in subset], [float(row["max_h1_abs_error"]) for row in subset], label=name)
    plt.axhline(GATE, color="black", linestyle="--", linewidth=1)
    plt.yscale("log"); plt.xlabel("training step"); plt.ylabel("validation max H=1 absolute error"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "max_h1_error_vs_training_step.png", dpi=170); plt.close()


def _plot_predictions(models: dict[str, nn.Module], test: tuple[Any, torch.Tensor], stat_mean: torch.Tensor, stat_scale: torch.Tensor, c_mean: torch.Tensor, c_scale: torch.Tensor) -> None:
    plt.figure(figsize=(6, 6)); values: list[float] = []
    for name, model in models.items():
        with torch.no_grad(): prediction, _ = _predict(model, test[0].positions, stat_mean, stat_scale, c_mean, c_scale)
        teacher = test[1]; values.extend(teacher.tolist()); values.extend(prediction.tolist())
        plt.scatter(teacher, prediction, s=12, alpha=.75, label=name)
    lower, upper = min(values), max(values); plt.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    plt.xlabel("prefix-only free-c teacher"); plt.ylabel("predicted c"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_teacher_c.png", dpi=170); plt.close()


def _plot_comparison(summary: list[dict[str, Any]]) -> None:
    values = [float(row["test_max_h1_abs_error"]) for row in summary]
    plt.figure(figsize=(7, 4)); plt.bar(range(len(summary)), values); plt.axhline(GATE, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(summary)), [row["encoder"] for row in summary], rotation=18, ha="right"); plt.yscale("log"); plt.ylabel("test max H=1 absolute error"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transition_statistics_vs_previous_tcn.png", dpi=170); plt.close()


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["encoder"]: row for row in summary}; exact = by_name["exact_statistics_mlp"]; learned = by_name["learned_statistics_tcn"]
    exact_passes = bool(exact["gate_passed"]); learned_passes = bool(learned["gate_passed"])
    return {
        "exact_statistics_mlp_passes_h1": exact_passes,
        "prefix_feature_extraction_is_main_bottleneck": exact_passes and not learned_passes,
        "tcn_learns_statistics_accurately_enough": learned_passes,
        "high_precision_scalar_arithmetic_is_bottleneck": not exact_passes,
        "clear_precision_breakthrough": exact_passes or learned_passes,
        "longer_run_started": False,
    }


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    columns = ["encoder", "best_step", "gate_passed", "val_max_abs_c_error", "test_max_abs_c_error", "val_c_rmse", "test_c_rmse", "val_max_h1_abs_error", "test_max_h1_abs_error", "val_a_rmse", "test_a_rmse", "val_b_rmse", "test_b_rmse"]
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(row.get(column, '')))}</td>" for column in columns) + "</tr>" for row in summary)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>H=1 transition-statistic diagnostic</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px}</style></head><body>", "<h1>Stage 1B H=1 Transition-Statistic Diagnostic</h1>", "<p>A and B are derived only from clean observed prefixes. No analytic A/B-to-c computation is used.</p>", "<h2>Answers</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>", "<h2>Comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + rows + "</tbody></table></body></html>"]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building fixed prefix-only teachers and statistic scalers", flush=True)
    monitor_data = _fixed_prefixes(MONITOR_N, TRAIN_DATA_SEED + 742100); val_data = _fixed_prefixes(VAL_N, TRAIN_DATA_SEED + 742200); test_data = _fixed_prefixes(TEST_N, TEST_DATA_SEED + 742300)
    monitor = (monitor_data, _free_c_teacher(monitor_data.positions)); val = (val_data, _free_c_teacher(val_data.positions)); test = (test_data, _free_c_teacher(test_data.positions))
    monitor_statistics = _statistics(monitor_data.positions); stat_mean = monitor_statistics.mean(0); stat_scale = monitor_statistics.std(0).clamp_min(1e-12)
    c_mean = monitor[1].mean(); c_scale = monitor[1].std().clamp_min(1e-12)
    print("training paired exact- and learned-statistics models", flush=True)
    models, history, current_summary = _train(monitor, val, test, stat_mean, stat_scale, c_mean, c_scale)
    summary = current_summary + [_previous_baseline()]; interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "training_history.csv", history); _write_html(summary, interpretation)
    _plot_history(history); _plot_predictions(models, test, stat_mean, stat_scale, c_mean, c_scale); _plot_comparison(summary)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
