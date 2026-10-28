"""Stage 1B one-frequency to two-frequency H=1 curriculum diagnostic."""

from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import dataclass
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
from experiments.sho.run_stage1b_h1_teacher_distillation_screen import _free_c_teacher
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/two_frequency_curriculum_diagnostic")
BATCH_SIZE = int(os.environ.get("TWOFREQ_BATCH_SIZE", "32"))
VAL_PER_FREQUENCY = int(os.environ.get("TWOFREQ_VAL_PER_FREQUENCY", "64"))
TEST_PER_FREQUENCY = int(os.environ.get("TWOFREQ_TEST_PER_FREQUENCY", "64"))
LR = float(os.environ.get("TWOFREQ_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("TWOFREQ_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("TWOFREQ_EVAL_INTERVAL", "200"))
GATE = 1e-6
OMEGA_2 = 2.0
OMEGA_4 = 4.0


@dataclass(frozen=True)
class FrequencySet:
    positions: torch.Tensor
    groups: torch.Tensor


class CEncoderTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        return 2.0 * torch.tanh(self.head(features).squeeze(-1))


def _positions(initial: torch.Tensor, omegas: torch.Tensor) -> torch.Tensor:
    times = torch.arange(P, dtype=DTYPE).unsqueeze(0).expand(initial.shape[0], -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial, omegas, times)
    return positions


def _sample_single(n: int, omega: float, generator: torch.Generator) -> FrequencySet:
    initial = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    omegas = torch.full((n,), omega, dtype=DTYPE)
    return FrequencySet(_positions(initial, omegas), omegas)


def _sample_mixture(n: int, generator: torch.Generator) -> FrequencySet:
    if n % 2:
        raise ValueError("50/50 batches require an even batch size")
    half = n // 2
    initial = 2.0 * torch.rand(n, 2, generator=generator, dtype=DTYPE) - 1.0
    groups = torch.cat((torch.full((half,), OMEGA_2, dtype=DTYPE), torch.full((half,), OMEGA_4, dtype=DTYPE)))
    permutation = torch.randperm(n, generator=generator)
    groups = groups[permutation]
    return FrequencySet(_positions(initial, groups), groups)


def _fixed_grouped(per_frequency: int, seed: int, include_omega4: bool) -> FrequencySet:
    generator = torch.Generator().manual_seed(seed)
    if not include_omega4:
        return _sample_single(per_frequency, OMEGA_2, generator)
    return _sample_mixture(2 * per_frequency, generator)


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _training_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.mean(_h1_errors(c, positions) ** 2)


@torch.no_grad()
def _evaluation_rows(model: nn.Module, data: FrequencySet, teachers: torch.Tensor, stage: str, split: str, step: int | str, train_loss: float | None = None) -> list[dict[str, Any]]:
    prediction = model(data.positions)
    rows: list[dict[str, Any]] = []
    for group_name, mask in [("all", torch.ones_like(data.groups, dtype=torch.bool)), ("omega_2", data.groups == OMEGA_2), ("omega_4", data.groups == OMEGA_4)]:
        if not bool(mask.any()):
            continue
        c = prediction[mask]
        teacher = teachers[mask]
        error = _h1_errors(c, data.positions[mask])
        rows.append({
            "stage": stage, "split": split, "frequency_group": group_name, "step": step,
            "max_h1_abs_error": float(torch.abs(error).max().item()),
            "rmse_h1_error": float(torch.sqrt(torch.mean(error**2)).item()),
            "c_rmse_vs_prefix_teacher": float(torch.sqrt(torch.mean((c - teacher) ** 2)).item()),
            "max_abs_c_error_vs_prefix_teacher": float(torch.abs(c - teacher).max().item()),
            "predicted_c_mean": float(c.mean().item()), "predicted_c_std": float(c.std().item()),
            "predicted_c_min": float(c.min().item()), "predicted_c_max": float(c.max().item()),
            "train_loss": "" if train_loss is None else train_loss,
        })
    return rows


def _train_stage(model: nn.Module, stage: str, val: tuple[FrequencySet, torch.Tensor], batch_fn: Any, stop_on_validation_gate: bool) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], int, bool]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + (1075000 if stage == "stage1_one_frequency" else 1076000))
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    passed = False
    history: list[dict[str, Any]] = []
    for step in range(1, MAX_STEPS + 1):
        batch = batch_fn(BATCH_SIZE, generator)
        prediction = model(batch.positions)
        loss = _training_loss(prediction, batch.positions)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            val_rows = _evaluation_rows(model, *val, stage, "val", step, float(loss.detach().item()))
            history.extend(val_rows)
            score = next(float(row["max_h1_abs_error"]) for row in val_rows if row["frequency_group"] == "all")
            if score < best_score:
                best_score = score; best_step = step; best_state = copy.deepcopy(model.state_dict())
            if stop_on_validation_gate and score <= GATE:
                passed = True
                break
    model.load_state_dict(best_state)
    if not stop_on_validation_gate:
        passed = best_score <= GATE
    return history, best_state, best_step, passed


def _final_rows(model: nn.Module, stage: str, datasets: dict[str, tuple[FrequencySet, torch.Tensor]], best_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, dataset in datasets.items():
        split_rows = _evaluation_rows(model, *dataset, stage, split, "best")
        for row in split_rows:
            row["best_step"] = best_step
        rows.extend(split_rows)
    return rows


def _summaries(final_rows: list[dict[str, Any]], stage1_proceeded: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in ("stage1_one_frequency", "stage2_two_frequencies"):
        stage_rows = [row for row in final_rows if row["stage"] == stage]
        if not stage_rows:
            continue
        for group in ("all", "omega_2", "omega_4"):
            val = next((row for row in stage_rows if row["split"] == "val" and row["frequency_group"] == group), None)
            test = next((row for row in stage_rows if row["split"] == "test" and row["frequency_group"] == group), None)
            if val is None or test is None:
                continue
            rows.append({
                "stage": stage, "frequency_group": group, "stage1_proceeded": stage1_proceeded,
                "best_step": val["best_step"], "gate_passed": bool(val["max_h1_abs_error"] <= GATE and test["max_h1_abs_error"] <= GATE),
                "val_max_h1_abs_error": val["max_h1_abs_error"], "test_max_h1_abs_error": test["max_h1_abs_error"],
                "val_max_abs_c_error": val["max_abs_c_error_vs_prefix_teacher"], "test_max_abs_c_error": test["max_abs_c_error_vs_prefix_teacher"],
                "test_predicted_c_mean": test["predicted_c_mean"], "test_predicted_c_std": test["predicted_c_std"], "test_predicted_c_min": test["predicted_c_min"], "test_predicted_c_max": test["predicted_c_max"],
            })
    return rows


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    stage1 = next(row for row in summary if row["stage"] == "stage1_one_frequency" and row["frequency_group"] == "omega_2")
    stage2_rows = [row for row in summary if row["stage"] == "stage2_two_frequencies"]
    if not stage2_rows:
        return {"stage1_mastered": bool(stage1["gate_passed"]), "stage2_ran": False, "recommendation": "stop: Stage 1 did not pass"}
    all_row = next(row for row in stage2_rows if row["frequency_group"] == "all")
    group2 = next(row for row in stage2_rows if row["frequency_group"] == "omega_2")
    group4 = next(row for row in stage2_rows if row["frequency_group"] == "omega_4")
    distinct = float(group2["test_predicted_c_min"]) > float(group4["test_predicted_c_max"]) or float(group4["test_predicted_c_min"]) > float(group2["test_predicted_c_max"])
    both_groups = bool(group2["gate_passed"] and group4["gate_passed"])
    passed = bool(all_row["gate_passed"] and both_groups)
    return {
        "stage1_mastered": bool(stage1["gate_passed"]), "stage2_ran": True,
        "stage2_passed_overall_and_per_group": passed,
        "distinguishes_frequencies_as_nonoverlapping_clusters": distinct,
        "omega2_preserved_after_stage2": bool(group2["gate_passed"]),
        "clusters_are_precise_not_merely_separated": bool(distinct and both_groups),
        "recommendation": "next: 2 frequencies -> 4 frequencies" if passed else "stop at two frequencies",
    }


def _plot(model: nn.Module, test: tuple[FrequencySet, torch.Tensor]) -> None:
    with torch.no_grad(): prediction = model(test[0].positions)
    plt.figure(figsize=(8, 5))
    for omega, label in ((OMEGA_2, "omega=2"), (OMEGA_4, "omega=4")):
        values = prediction[test[0].groups == omega].cpu().numpy()
        plt.hist(values, bins=24, alpha=.6, label=label)
    plt.xlabel("predicted c"); plt.ylabel("trajectory count"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "predicted_c_distributions.png", dpi=170); plt.close()


def main() -> None:
    if BATCH_SIZE % 2:
        raise ValueError("TWOFREQ_BATCH_SIZE must be even")
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stage1_val_data = _fixed_grouped(VAL_PER_FREQUENCY, TRAIN_DATA_SEED + 1075100, False); stage1_test_data = _fixed_grouped(TEST_PER_FREQUENCY, TEST_DATA_SEED + 1075200, False)
    stage2_val_data = _fixed_grouped(VAL_PER_FREQUENCY, TRAIN_DATA_SEED + 1076100, True); stage2_test_data = _fixed_grouped(TEST_PER_FREQUENCY, TEST_DATA_SEED + 1076200, True)
    print("building fixed prefix-only diagnostic teachers", flush=True)
    stage1_val = (stage1_val_data, _free_c_teacher(stage1_val_data.positions)); stage1_test = (stage1_test_data, _free_c_teacher(stage1_test_data.positions))
    stage2_val = (stage2_val_data, _free_c_teacher(stage2_val_data.positions)); stage2_test = (stage2_test_data, _free_c_teacher(stage2_test_data.positions))
    model = CEncoderTCN().to(dtype=DTYPE)
    print("training Stage 1: omega=2", flush=True)
    history1, _, best_step1, stage1_val_passed = _train_stage(model, "stage1_one_frequency", stage1_val, lambda n, gen: _sample_single(n, OMEGA_2, gen), True)
    final_rows = _final_rows(model, "stage1_one_frequency", {"val": stage1_val, "test": stage1_test}, best_step1)
    history = history1
    if stage1_val_passed:
        print("Stage 1 validation gate passed; training Stage 2: omega=2/4", flush=True)
        history2, _, best_step2, _ = _train_stage(model, "stage2_two_frequencies", stage2_val, _sample_mixture, False)
        history.extend(history2); final_rows.extend(_final_rows(model, "stage2_two_frequencies", {"val": stage2_val, "test": stage2_test}, best_step2)); _plot(model, stage2_test)
    else:
        print("Stage 1 validation gate failed; stopping before Stage 2", flush=True)
        _plot(model, (FrequencySet(torch.cat((stage1_test_data.positions, stage2_test_data.positions[stage2_test_data.groups == OMEGA_4])), torch.cat((stage1_test_data.groups, stage2_test_data.groups[stage2_test_data.groups == OMEGA_4]))), torch.cat((stage1_test[1], stage2_test[1][stage2_test_data.groups == OMEGA_4]))))
    summary = _summaries(final_rows, stage1_val_passed); interpretation = _interpret(summary)
    _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "training_history.csv", history + final_rows)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
