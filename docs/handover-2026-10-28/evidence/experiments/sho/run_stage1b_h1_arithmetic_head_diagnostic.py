"""Stage 1B H=1 arithmetic-head diagnostic on prefix transition statistics."""

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

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import P, _r2
from experiments.sho.run_stage1b_h1_teacher_distillation_screen import _fixed_prefixes, _free_c_teacher, _sample_prefixes
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/h1_arithmetic_head_diagnostic")
BATCH_SIZE = int(os.environ.get("H1ARITH_BATCH_SIZE", "32"))
MONITOR_N = int(os.environ.get("H1ARITH_MONITOR_N", "64"))
VAL_N = int(os.environ.get("H1ARITH_VAL_N", "128"))
TEST_N = int(os.environ.get("H1ARITH_TEST_N", "128"))
LR = float(os.environ.get("H1ARITH_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("H1ARITH_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("H1ARITH_EVAL_INTERVAL", "200"))
RATIONAL_EPSILON = float(os.environ.get("H1ARITH_RATIONAL_EPSILON", "1e-8"))
NALU_EPSILON = float(os.environ.get("H1ARITH_NALU_EPSILON", "1e-12"))
GATE = 1e-6
MODELS = ("mlp_head", "learned_rational_head", "neural_arithmetic_head")


def _statistics(prefix: torch.Tensor) -> torch.Tensor:
    x_prev, x, x_next = prefix[:, :-2], prefix[:, 1:-1], prefix[:, 2:]
    return torch.stack((torch.sum(x * (x_prev + x_next), dim=1), torch.sum(x**2, dim=1)), dim=1)


class MLPHead(nn.Module):
    def __init__(self, initial_c: float) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.constant_(self.net[-1].bias, initial_c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


class LearnedRationalHead(nn.Module):
    def __init__(self, initial_c: float) -> None:
        super().__init__()
        self.numerator = nn.Linear(2, 1)
        self.denominator = nn.Linear(2, 1)
        nn.init.normal_(self.numerator.weight, std=1e-2)
        nn.init.normal_(self.denominator.weight, std=1e-2)
        nn.init.constant_(self.numerator.bias, initial_c)
        nn.init.ones_(self.denominator.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        numerator = self.numerator(inputs).squeeze(-1)
        denominator = self.denominator(inputs).squeeze(-1)
        return numerator / (denominator + RATIONAL_EPSILON)


class NALULayer(nn.Module):
    """Small NALU-style layer with learned additive/multiplicative paths."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.w_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.m_hat = nn.Parameter(torch.empty(out_features, in_features))
        self.gate = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.w_hat)
        nn.init.xavier_uniform_(self.m_hat)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.tanh(self.w_hat) * torch.sigmoid(self.m_hat)
        additive = torch.nn.functional.linear(inputs, weights)
        log_inputs = torch.log(torch.abs(inputs) + NALU_EPSILON)
        multiplicative = torch.exp(torch.clamp(torch.nn.functional.linear(log_inputs, weights), -20.0, 20.0))
        gate = torch.sigmoid(self.gate(inputs))
        return gate * additive + (1.0 - gate) * multiplicative


class NeuralArithmeticHead(nn.Module):
    def __init__(self, initial_c: float) -> None:
        super().__init__()
        self.nalu = NALULayer(2, 16)
        self.output = nn.Linear(16, 1)
        nn.init.constant_(self.output.bias, initial_c)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.nalu(inputs)).squeeze(-1)


def _make_models(initial_c: float) -> dict[str, nn.Module]:
    return {
        "mlp_head": MLPHead(initial_c).to(dtype=DTYPE),
        "learned_rational_head": LearnedRationalHead(initial_c).to(dtype=DTYPE),
        "neural_arithmetic_head": NeuralArithmeticHead(initial_c).to(dtype=DTYPE),
    }


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _predict(model: nn.Module, positions: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    return model(_statistics(positions[:, :P]) / input_scale)


@torch.no_grad()
def _metrics(model: nn.Module, data: Any, teacher: torch.Tensor, input_scale: torch.Tensor, encoder: str, split: str, step: int | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    prediction = _predict(model, data.positions, input_scale)
    c_error = prediction - teacher
    h1_error = _h1_errors(prediction, data.positions)
    row: dict[str, Any] = {
        "encoder": encoder,
        "split": split,
        "step": step,
        "c_rmse": float(torch.sqrt(torch.mean(c_error**2)).item()),
        "max_abs_c_error": float(torch.abs(c_error).max().item()),
        "c_r2": _r2(prediction[:, None], teacher[:, None]),
        "max_h1_abs_error": float(torch.abs(h1_error).max().item()),
        "rmse_h1_error": float(torch.sqrt(torch.mean(h1_error**2)).item()),
        "predicted_c_min": float(prediction.min().item()),
        "predicted_c_max": float(prediction.max().item()),
    }
    if extra:
        row.update(extra)
    return row


def _train(monitor: tuple[Any, torch.Tensor], val: tuple[Any, torch.Tensor], test: tuple[Any, torch.Tensor], input_scale: torch.Tensor) -> tuple[dict[str, nn.Module], list[dict[str, Any]], list[dict[str, Any]]]:
    models = _make_models(float(monitor[1].mean().item()))
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=LR) for name, model in models.items()}
    best_states = {name: copy.deepcopy(model.state_dict()) for name, model in models.items()}
    best_scores = {name: float("inf") for name in models}
    best_steps = {name: 0 for name in models}
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(SEED + 853000)
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_prefixes(BATCH_SIZE, generator)
        teacher = _free_c_teacher(batch.positions)
        inputs = _statistics(batch.positions) / input_scale
        for name, model in models.items():
            prediction = model(inputs)
            loss = torch.mean((prediction - teacher) ** 2)
            optimizers[name].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizers[name].step()
            if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
                common = {"train_batch_c_mse": float(loss.detach().item())}
                history.append(_metrics(model, *monitor, input_scale, name, "train_monitor", step, common))
                val_row = _metrics(model, *val, input_scale, name, "val", step, common)
                history.append(val_row)
                score = float(val_row["max_h1_abs_error"])
                if score < best_scores[name]:
                    best_scores[name] = score
                    best_steps[name] = step
                    best_states[name] = copy.deepcopy(model.state_dict())
    summary: list[dict[str, Any]] = []
    for name, model in models.items():
        model.load_state_dict(best_states[name])
        finals = [_metrics(model, *dataset, input_scale, name, split, "best", {"best_step": best_steps[name]}) for split, dataset in (("train_monitor", monitor), ("val", val), ("test", test))]
        history.extend(finals)
        train_final, val_final, test_final = finals
        summary.append({
            "encoder": name,
            "best_step": best_steps[name],
            "gate_passed": bool(val_final["max_h1_abs_error"] <= GATE and test_final["max_h1_abs_error"] <= GATE),
            "train_c_rmse": train_final["c_rmse"], "val_c_rmse": val_final["c_rmse"], "test_c_rmse": test_final["c_rmse"],
            "val_max_abs_c_error": val_final["max_abs_c_error"], "test_max_abs_c_error": test_final["max_abs_c_error"],
            "val_max_h1_abs_error": val_final["max_h1_abs_error"], "test_max_h1_abs_error": test_final["max_h1_abs_error"],
            "val_c_r2": val_final["c_r2"], "test_c_r2": test_final["c_r2"],
        })
    return models, history, summary


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {row["encoder"]: row for row in summary}
    mlp, rational, arithmetic = rows["mlp_head"], rows["learned_rational_head"], rows["neural_arithmetic_head"]
    rational_better = float(rational["val_max_h1_abs_error"]) < float(mlp["val_max_h1_abs_error"]) and float(rational["test_max_h1_abs_error"]) < float(mlp["test_max_h1_abs_error"])
    arithmetic_best = all(float(arithmetic[f"{split}_max_h1_abs_error"]) < float(other[f"{split}_max_h1_abs_error"]) for split in ("val", "test") for other in (mlp, rational))
    passed = [row["encoder"] for row in summary if row["gate_passed"]]
    return {
        "rational_head_outperforms_mlp": rational_better,
        "arithmetic_head_outperforms_both": arithmetic_best,
        "either_moves_to_1e-6_gate": bool(passed),
        "passed_heads": passed,
        "arithmetic_structure_is_missing_inductive_bias": bool(passed and any(name != "mlp_head" for name in passed) and not mlp["gate_passed"]),
        "longer_run_started": False,
    }


def _comparison_plot(models: dict[str, nn.Module], summary: list[dict[str, Any]], test: tuple[Any, torch.Tensor], input_scale: torch.Tensor) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    values = [float(row["test_max_h1_abs_error"]) for row in summary]
    axes[0].bar(range(len(summary)), values)
    axes[0].axhline(GATE, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(range(len(summary)), [row["encoder"] for row in summary], rotation=18, ha="right")
    axes[0].set_yscale("log"); axes[0].set_ylabel("test max H=1 absolute error")
    all_values: list[float] = []
    for name, model in models.items():
        with torch.no_grad(): prediction = _predict(model, test[0].positions, input_scale)
        axes[1].scatter(test[1], prediction, s=11, alpha=.7, label=name)
        all_values.extend(test[1].tolist()); all_values.extend(prediction.tolist())
    lower, upper = min(all_values), max(all_values)
    axes[1].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axes[1].set_xlabel("prefix-only free-c teacher"); axes[1].set_ylabel("predicted c"); axes[1].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(OUTPUT_DIR / "arithmetic_head_comparison.png", dpi=170); plt.close(figure)


def _write_html(summary: list[dict[str, Any]], interpretation: dict[str, Any]) -> None:
    columns = ["encoder", "best_step", "gate_passed", "train_c_rmse", "val_c_rmse", "test_c_rmse", "val_max_abs_c_error", "test_max_abs_c_error", "val_max_h1_abs_error", "test_max_h1_abs_error", "val_c_r2", "test_c_r2"]
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(row[column]))}</td>" for column in columns) + "</tr>" for row in summary)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>H=1 arithmetic-head diagnostic</title><style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.45}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:5px 7px}th{background:#f4f4f4}pre{background:#f7f7f7;padding:12px}</style></head><body>", "<h1>Stage 1B H=1 Arithmetic-Head Diagnostic</h1>", "<p>All heads receive the same scaled prefix-derived A and B. No analytic ratio, LS, labels, velocities, or future samples are used.</p>", "<h2>Answers</h2><pre>" + escape(json.dumps(interpretation, indent=2)) + "</pre>", "<h2>Comparison</h2><table><thead><tr>" + "".join(f"<th>{escape(column)}</th>" for column in columns) + "</tr></thead><tbody>" + rows + "</tbody></table></body></html>"]
    (OUTPUT_DIR / "summary.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("building fixed prefix-only teachers and A/B scaler", flush=True)
    monitor_data = _fixed_prefixes(MONITOR_N, TRAIN_DATA_SEED + 853100); val_data = _fixed_prefixes(VAL_N, TRAIN_DATA_SEED + 853200); test_data = _fixed_prefixes(TEST_N, TEST_DATA_SEED + 853300)
    monitor = (monitor_data, _free_c_teacher(monitor_data.positions)); val = (val_data, _free_c_teacher(val_data.positions)); test = (test_data, _free_c_teacher(test_data.positions))
    input_scale = torch.sqrt(torch.mean(_statistics(monitor_data.positions) ** 2, dim=0)).clamp_min(1e-12)
    print("training paired arithmetic heads", flush=True)
    models, history, summary = _train(monitor, val, test, input_scale)
    interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "training_history.csv", history); _write_html(summary, interpretation); _comparison_plot(models, summary, test, input_scale)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
