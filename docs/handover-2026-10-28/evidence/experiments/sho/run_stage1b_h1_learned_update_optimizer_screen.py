"""Quick Stage 1B learned-update optimizer screen at H=1 only."""

from __future__ import annotations

import copy
import json
import os
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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, TCNBlock
from experiments.sho.run_stage1b_h1_teacher_distillation_screen import _fixed_prefixes, _free_c_teacher, _sample_prefixes
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_learned_update_optimizer_screen")
BATCH_SIZE = int(os.environ.get("H1UPDATE_BATCH_SIZE", "32"))
MONITOR_N = int(os.environ.get("H1UPDATE_MONITOR_N", "64"))
VAL_N = int(os.environ.get("H1UPDATE_VAL_N", "128"))
TEST_N = int(os.environ.get("H1UPDATE_TEST_N", "128"))
LR = float(os.environ.get("H1UPDATE_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1UPDATE_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("H1UPDATE_EVAL_INTERVAL", "200"))
SMOOTHMAX_TEMPERATURE = float(os.environ.get("H1UPDATE_SMOOTHMAX_TEMPERATURE", "1e-6"))
ADAM_REFINE_LR = float(os.environ.get("H1UPDATE_ADAM_REFINE_LR", "1e-2"))
MAX_UPDATES = 5
REPORT_STEPS = (0, 1, 2, 5)
GATE = 1e-6


class InitializerTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        return 2.0 * torch.tanh(self.head(features).squeeze(-1))


class UpdateHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 0.1 * torch.tanh(self.net(features).squeeze(-1))


class LearnedOptimizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.initializer = InitializerTCN()
        self.update = UpdateHead()


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _prefix_loss_vector(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    squared = _h1_errors(c, positions) ** 2
    maximum = squared.max(dim=1).values
    if SMOOTHMAX_TEMPERATURE <= 0:
        return maximum
    return maximum + SMOOTHMAX_TEMPERATURE * torch.log(
        torch.mean(torch.exp((squared - maximum[:, None]) / SMOOTHMAX_TEMPERATURE), dim=1)
    )


def _learned_trajectory(model: LearnedOptimizer, positions: torch.Tensor, create_graph: bool) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    c = model.initializer(positions[:, :P])
    values = [c]
    losses: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    previous_delta = torch.zeros_like(c)
    for _ in range(MAX_UPDATES):
        loss_vector = _prefix_loss_vector(c, positions)
        gradient = torch.autograd.grad(loss_vector.sum(), c, create_graph=create_graph)[0]
        delta = model.update(torch.stack((c, gradient, loss_vector, previous_delta), dim=1))
        c = c + delta
        losses.append(loss_vector)
        deltas.append(delta)
        values.append(c)
        previous_delta = delta
    return values, losses, deltas


def _learned_values(model: LearnedOptimizer, positions: torch.Tensor, training: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    with torch.enable_grad():
        values, _, deltas = _learned_trajectory(model, positions, create_graph=training)
    if training:
        return values, deltas
    return [value.detach() for value in values], [delta.detach() for delta in deltas]


def _adam_values(c0: torch.Tensor, positions: torch.Tensor) -> list[torch.Tensor]:
    c = nn.Parameter(c0.detach().clone())
    optimizer = torch.optim.Adam([c], lr=ADAM_REFINE_LR)
    values = [c.detach().clone()]
    for _ in range(MAX_UPDATES):
        optimizer.zero_grad()
        _prefix_loss_vector(c, positions).sum().backward()
        optimizer.step()
        values.append(c.detach().clone())
    return values


def _metrics(c: torch.Tensor, positions: torch.Tensor, teacher: torch.Tensor, method: str, split: str, refinement_step: int, training_step: int | str, delta: torch.Tensor | None = None) -> dict[str, Any]:
    c_error = c - teacher
    h1 = _h1_errors(c, positions)
    return {
        "method": method,
        "split": split,
        "training_step": training_step,
        "refinement_step": refinement_step,
        "max_h1_abs_error": float(torch.abs(h1).max().item()),
        "rmse_h1_error": float(torch.sqrt(torch.mean(h1**2)).item()),
        "c_rmse_vs_teacher": float(torch.sqrt(torch.mean(c_error**2)).item()),
        "max_abs_c_error_vs_teacher": float(torch.abs(c_error).max().item()),
        "mean_abs_delta": 0.0 if delta is None else float(torch.abs(delta).mean().item()),
        "max_abs_delta": 0.0 if delta is None else float(torch.abs(delta).max().item()),
    }


def _evaluate_learned(model: LearnedOptimizer, dataset: tuple[Any, torch.Tensor], split: str, training_step: int | str) -> list[dict[str, Any]]:
    data, teacher = dataset
    values, deltas = _learned_values(model, data.positions, training=False)
    return [_metrics(values[k], data.positions, teacher, "encoder_only" if k == 0 else "learned_updates", split, k, training_step, None if k == 0 else deltas[k - 1]) for k in REPORT_STEPS]


def _train(monitor: tuple[Any, torch.Tensor], val: tuple[Any, torch.Tensor]) -> tuple[LearnedOptimizer, list[dict[str, Any]], int]:
    model = LearnedOptimizer().to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + 964000)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    history: list[dict[str, Any]] = []
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefixes(BATCH_SIZE, generator)
        values, _ = _learned_values(model, batch.positions, training=True)
        # Deep supervision makes K=0,1,2,5 meaningful truncations of one shared
        # learned optimizer while backpropagating through every update path.
        loss = sum(_prefix_loss_vector(values[k], batch.positions).mean() for k in REPORT_STEPS)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            monitor_rows = _evaluate_learned(model, monitor, "train_monitor", step)
            val_rows = _evaluate_learned(model, val, "val", step)
            for row in monitor_rows + val_rows:
                row["train_loss"] = float(loss.detach().item())
            history.extend(monitor_rows + val_rows)
            score = next(float(row["max_h1_abs_error"]) for row in val_rows if row["refinement_step"] == MAX_UPDATES)
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, best_step


def _final_rows(model: LearnedOptimizer, datasets: dict[str, tuple[Any, torch.Tensor]], best_step: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    learned_rows: list[dict[str, Any]] = []
    adam_rows: list[dict[str, Any]] = []
    for split, dataset in datasets.items():
        rows = _evaluate_learned(model, dataset, split, "best")
        for row in rows:
            row["best_training_step"] = best_step
        learned_rows.extend(rows)
        c0 = next_c0 = None
        values, _ = _learned_values(model, dataset[0].positions, training=False)
        c0 = values[0]
        adam = _adam_values(c0, dataset[0].positions)
        for k in REPORT_STEPS:
            row = _metrics(adam[k], dataset[0].positions, dataset[1], "adam_refinement", split, k, "best", None if k == 0 else adam[k] - adam[k - 1])
            row["best_training_step"] = best_step
            adam_rows.append(row)
    return learned_rows, adam_rows


def _summary(learned_rows: list[dict[str, Any]], adam_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, source in (("learned_updates", learned_rows), ("adam_refinement", adam_rows)):
        steps = REPORT_STEPS if method == "adam_refinement" else REPORT_STEPS[1:]
        for k in steps:
            val = next(row for row in source if row["split"] == "val" and row["refinement_step"] == k)
            test = next(row for row in source if row["split"] == "test" and row["refinement_step"] == k)
            rows.append({"method": method, "refinement_steps": k, "gate_passed": bool(val["max_h1_abs_error"] <= GATE and test["max_h1_abs_error"] <= GATE), "val_max_h1_abs_error": val["max_h1_abs_error"], "test_max_h1_abs_error": test["max_h1_abs_error"], "val_c_rmse_vs_teacher": val["c_rmse_vs_teacher"], "test_c_rmse_vs_teacher": test["c_rmse_vs_teacher"], "val_mean_abs_delta": val["mean_abs_delta"], "test_mean_abs_delta": test["mean_abs_delta"]})
    val0 = next(row for row in learned_rows if row["split"] == "val" and row["refinement_step"] == 0)
    test0 = next(row for row in learned_rows if row["split"] == "test" and row["refinement_step"] == 0)
    rows.insert(0, {"method": "encoder_only", "refinement_steps": 0, "gate_passed": bool(val0["max_h1_abs_error"] <= GATE and test0["max_h1_abs_error"] <= GATE), "val_max_h1_abs_error": val0["max_h1_abs_error"], "test_max_h1_abs_error": test0["max_h1_abs_error"], "val_c_rmse_vs_teacher": val0["c_rmse_vs_teacher"], "test_c_rmse_vs_teacher": test0["c_rmse_vs_teacher"], "val_mean_abs_delta": 0.0, "test_mean_abs_delta": 0.0})
    return rows


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    encoder = next(row for row in summary if row["method"] == "encoder_only")
    learned = [row for row in summary if row["method"] == "learned_updates"]
    adam = {int(row["refinement_steps"]): row for row in summary if row["method"] == "adam_refinement"}
    improved = [row for row in learned if float(row["val_max_h1_abs_error"]) < float(encoder["val_max_h1_abs_error"]) and float(row["test_max_h1_abs_error"]) < float(encoder["test_max_h1_abs_error"])]
    passed = [int(row["refinement_steps"]) for row in learned if row["gate_passed"]]
    beats_adam = [int(row["refinement_steps"]) for row in learned if float(row["val_max_h1_abs_error"]) < float(adam[int(row["refinement_steps"])]["val_max_h1_abs_error"]) and float(row["test_max_h1_abs_error"]) < float(adam[int(row["refinement_steps"])]["test_max_h1_abs_error"])]
    return {"learned_updates_improve_encoder_only": bool(improved), "improving_learned_update_counts": [int(row["refinement_steps"]) for row in improved], "learned_update_counts_passing_h1": passed, "learned_update_counts_beating_adam": beats_adam, "update_rule_generalizes_to_unseen_trajectories": bool(improved), "longer_run_started": False}


def _plot(learned_rows: list[dict[str, Any]], adam_rows: list[dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 5))
    for split, linestyle in (("val", "--"), ("test", "-")):
        learned = [row for row in learned_rows if row["split"] == split]
        adam = [row for row in adam_rows if row["split"] == split]
        plt.plot([row["refinement_step"] for row in learned], [row["max_h1_abs_error"] for row in learned], marker="o", linestyle=linestyle, label=f"learned {split}")
        plt.plot([row["refinement_step"] for row in adam], [row["max_h1_abs_error"] for row in adam], marker="s", linestyle=linestyle, label=f"Adam {split}")
    plt.axhline(GATE, color="black", linestyle=":", linewidth=1); plt.yscale("log"); plt.xlabel("refinement step"); plt.ylabel("max H=1 absolute error"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "h1_error_vs_refinement_step.png", dpi=170); plt.close()


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    columns = ["method", "refinement_steps", "gate_passed", "val_max_h1_abs_error", "test_max_h1_abs_error", "val_c_rmse_vs_teacher", "test_c_rmse_vs_teacher", "val_mean_abs_delta", "test_mean_abs_delta"]
    table_rows = "".join("<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns) + "</tr>" for row in summary)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>H=1 learned-update optimizer screen</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px}</style></head><body>", "<h1>Stage 1B H=1 Learned-Update Optimizer Screen</h1>", "<p>One shared learned update head is unrolled to five steps; K=0,1,2,5 are truncations of the same restored model. Adam starts from the identical c0. Only clean observed prefixes enter refinement.</p>", "<h2>Answers</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>", "<h2>Comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + table_rows + "</tbody></table></body></html>"]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building fixed prefix-only diagnostic teachers", flush=True)
    monitor_data = _fixed_prefixes(MONITOR_N, TRAIN_DATA_SEED + 964100); val_data = _fixed_prefixes(VAL_N, TRAIN_DATA_SEED + 964200); test_data = _fixed_prefixes(TEST_N, TEST_DATA_SEED + 964300)
    monitor = (monitor_data, _free_c_teacher(monitor_data.positions)); val = (val_data, _free_c_teacher(val_data.positions)); test = (test_data, _free_c_teacher(test_data.positions))
    print("training five-step learned optimizer", flush=True)
    model, history, best_step = _train(monitor, val)
    learned_rows, adam_rows = _final_rows(model, {"train_monitor": monitor, "val": val, "test": test}, best_step)
    summary = _summary(learned_rows, adam_rows); interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "training_history.csv", history + learned_rows + adam_rows); _write_html(summary, interpretation); _plot(learned_rows, adam_rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "best_training_step": best_step, **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
