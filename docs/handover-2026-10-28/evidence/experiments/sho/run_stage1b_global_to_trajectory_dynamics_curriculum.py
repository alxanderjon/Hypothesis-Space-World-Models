"""Stage 1B global-to-trajectory-specific dynamics curriculum at H=1."""

from __future__ import annotations

import copy
import json
import os
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
from experiments.sho.run_stage1b_two_frequency_curriculum_diagnostic import FrequencySet, OMEGA_2, OMEGA_4, _fixed_grouped, _sample_mixture, _sample_single
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE, SEED, TEST_DATA_SEED, TRAIN_DATA_SEED, _write_csv


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/global_to_trajectory_dynamics_curriculum")
BATCH_SIZE = int(os.environ.get("GLOBALCURR_BATCH_SIZE", "32"))
VAL_PER_FREQUENCY = int(os.environ.get("GLOBALCURR_VAL_PER_FREQUENCY", "64"))
TEST_PER_FREQUENCY = int(os.environ.get("GLOBALCURR_TEST_PER_FREQUENCY", "64"))
LR = float(os.environ.get("GLOBALCURR_LR", "1e-4"))
MAX_STEPS = int(os.environ.get("GLOBALCURR_MAX_STEPS", "5000"))
EVAL_INTERVAL = int(os.environ.get("GLOBALCURR_EVAL_INTERVAL", "200"))
GLOBAL_INITIAL_C = float(os.environ.get("GLOBALCURR_INITIAL_C", "1.7"))
RESIDUAL_TOLERANCE = float(os.environ.get("GLOBALCURR_RESIDUAL_TOLERANCE", "1e-6"))
GATE = 1e-6


class ResidualTCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(*(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16]))
        self.head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(self.input(prefix.unsqueeze(1)))
        features = self.blocks(features).mean(dim=-1)
        return self.head(features).squeeze(-1)


class GlobalResidualModel(nn.Module):
    def __init__(self, c_global: float, with_residual: bool) -> None:
        super().__init__()
        initial = torch.tensor(c_global / 2.0, dtype=DTYPE).clamp(-0.999999, 0.999999)
        self.raw_global = nn.Parameter(torch.atanh(initial))
        self.residual = ResidualTCN() if with_residual else None

    @property
    def c_global(self) -> torch.Tensor:
        return 2.0 * torch.tanh(self.raw_global)

    def components(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = torch.zeros(positions.shape[0], dtype=positions.dtype, device=positions.device) if self.residual is None else self.residual(positions[:, :P])
        return self.c_global + residual, residual

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.components(positions)[0]


def _h1_errors(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = positions[:, :P]
    return c[:, None] * x[:, 1:-1] - x[:, :-2] - x[:, 2:]


def _training_loss(c: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return torch.mean(_h1_errors(c, positions) ** 2)


@torch.no_grad()
def _evaluation_rows(model: GlobalResidualModel, data: FrequencySet, stage: str, split: str, step: int | str, train_loss: float | str = "") -> list[dict[str, Any]]:
    prediction, residual = model.components(data.positions)
    rows: list[dict[str, Any]] = []
    groups = (("all", torch.ones_like(data.groups, dtype=torch.bool)), ("omega_2", data.groups == OMEGA_2), ("omega_4", data.groups == OMEGA_4))
    for group_name, mask in groups:
        if not bool(mask.any()):
            continue
        c_group, r_group = prediction[mask], residual[mask]
        error = _h1_errors(c_group, data.positions[mask])
        rows.append({
            "stage": stage, "split": split, "frequency_group": group_name, "step": step,
            "max_h1_abs_error": float(torch.abs(error).max().item()), "rmse_h1_error": float(torch.sqrt(torch.mean(error**2)).item()),
            "c_global": float(model.c_global.item()), "predicted_c_mean": float(c_group.mean().item()), "predicted_c_std": float(c_group.std().item()),
            "predicted_c_min": float(c_group.min().item()), "predicted_c_max": float(c_group.max().item()),
            "residual_mean": float(r_group.mean().item()), "residual_std": float(r_group.std().item()), "max_abs_residual": float(torch.abs(r_group).max().item()),
            "train_loss": train_loss,
        })
    return rows


def _score(rows: list[dict[str, Any]]) -> float:
    return next(float(row["max_h1_abs_error"]) for row in rows if row["frequency_group"] == "all")


def _train_global(model: GlobalResidualModel, val: FrequencySet) -> tuple[list[dict[str, Any]], int, bool]:
    optimizer = torch.optim.Adam([model.raw_global], lr=LR)
    generator = torch.Generator().manual_seed(SEED + 1185000)
    best_state = copy.deepcopy(model.state_dict()); best_score = float("inf"); best_step = 0; history: list[dict[str, Any]] = []
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_single(BATCH_SIZE, OMEGA_2, generator)
        loss = _training_loss(model(batch.positions), batch.positions)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            rows = _evaluation_rows(model, val, "stage1_global_scalar", "val", step, float(loss.detach().item())); history.extend(rows)
            score = _score(rows)
            if score < best_score:
                best_score = score; best_step = step; best_state = copy.deepcopy(model.state_dict())
            if score <= GATE:
                break
    model.load_state_dict(best_state)
    return history, best_step, best_score <= GATE


def _train_residual_preservation(model: GlobalResidualModel, val: FrequencySet) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + 1186000)
    initial_rows = _evaluation_rows(model, val, "stage2_zero_residual", "val", 0, "")
    history = list(initial_rows); best_state = copy.deepcopy(model.state_dict()); best_score = float("inf"); best_step = 0
    # At least one joint optimization update verifies that both the global
    # scalar and newly attached zero-output branch participate in training.
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_single(BATCH_SIZE, OMEGA_2, generator)
        loss = _training_loss(model(batch.positions), batch.positions)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            rows = _evaluation_rows(model, val, "stage2_zero_residual", "val", step, float(loss.detach().item())); history.extend(rows)
            score = _score(rows)
            if score < best_score:
                best_score = score; best_step = step; best_state = copy.deepcopy(model.state_dict())
            current_residual = next(float(row["max_abs_residual"]) for row in rows if row["frequency_group"] == "all")
            if score <= GATE and current_residual <= RESIDUAL_TOLERANCE:
                break
    model.load_state_dict(best_state)
    return history, best_step


def _train_two_dynamics(model: GlobalResidualModel, val: FrequencySet) -> tuple[list[dict[str, Any]], int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(SEED + 1187000)
    best_state = copy.deepcopy(model.state_dict()); best_score = float("inf"); best_step = 0; history: list[dict[str, Any]] = []
    for step in range(1, MAX_STEPS + 1):
        batch = _sample_mixture(BATCH_SIZE, generator)
        loss = _training_loss(model(batch.positions), batch.positions)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            rows = _evaluation_rows(model, val, "stage3_two_dynamics", "val", step, float(loss.detach().item())); history.extend(rows)
            score = _score(rows)
            if score < best_score:
                best_score = score; best_step = step; best_state = copy.deepcopy(model.state_dict())
            group_scores = [float(row["max_h1_abs_error"]) for row in rows if row["frequency_group"] in {"omega_2", "omega_4"}]
            if group_scores and max(group_scores) <= GATE:
                break
    model.load_state_dict(best_state)
    return history, best_step


def _final_rows(model: GlobalResidualModel, stage: str, val: FrequencySet, test: FrequencySet, best_step: int) -> list[dict[str, Any]]:
    rows = _evaluation_rows(model, val, stage, "val", "best") + _evaluation_rows(model, test, stage, "test", "best")
    for row in rows:
        row["best_step"] = best_step
    return rows


def _stage_passes(rows: list[dict[str, Any]], residual_required: bool) -> bool:
    relevant = [row for row in rows if row["frequency_group"] == "all"]
    error_pass = len(relevant) == 2 and all(float(row["max_h1_abs_error"]) <= GATE for row in relevant)
    residual_pass = not residual_required or all(float(row["max_abs_residual"]) <= RESIDUAL_TOLERANCE for row in relevant)
    return error_pass and residual_pass


def _summary(final_rows: list[dict[str, Any]], stage_pass: dict[str, bool]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for stage in ("stage1_global_scalar", "stage2_zero_residual", "stage3_two_dynamics"):
        rows = [row for row in final_rows if row["stage"] == stage]
        for group in ("all", "omega_2", "omega_4"):
            val = next((row for row in rows if row["split"] == "val" and row["frequency_group"] == group), None); test = next((row for row in rows if row["split"] == "test" and row["frequency_group"] == group), None)
            if val is None or test is None:
                continue
            summary.append({
                "stage": stage, "frequency_group": group, "stage_passed": stage_pass[stage], "best_step": val["best_step"],
                "val_max_h1_abs_error": val["max_h1_abs_error"], "test_max_h1_abs_error": test["max_h1_abs_error"], "c_global": test["c_global"],
                "test_predicted_c_mean": test["predicted_c_mean"], "test_predicted_c_std": test["predicted_c_std"], "test_predicted_c_min": test["predicted_c_min"], "test_predicted_c_max": test["predicted_c_max"],
                "test_residual_mean": test["residual_mean"], "test_residual_std": test["residual_std"], "test_max_abs_residual": test["max_abs_residual"],
            })
    return summary


def _interpret(summary: list[dict[str, Any]]) -> dict[str, Any]:
    stage1 = next(row for row in summary if row["stage"] == "stage1_global_scalar" and row["frequency_group"] == "all")
    stage2 = next((row for row in summary if row["stage"] == "stage2_zero_residual" and row["frequency_group"] == "all"), None)
    stage3 = [row for row in summary if row["stage"] == "stage3_two_dynamics"]
    if stage2 is None:
        return {"global_scalar_passes": False, "zero_residual_branch_preserves_solution": False, "two_dynamics_ran": False, "recommendation": "stop after failed Stage 1"}
    if not stage2["stage_passed"]:
        return {"global_scalar_passes": bool(stage1["stage_passed"]), "zero_residual_branch_preserves_solution": False, "two_dynamics_ran": False, "recommendation": "stop after failed Stage 2"}
    all3 = next(row for row in stage3 if row["frequency_group"] == "all"); group2 = next(row for row in stage3 if row["frequency_group"] == "omega_2"); group4 = next(row for row in stage3 if row["frequency_group"] == "omega_4")
    distinct = float(group2["test_predicted_c_min"]) > float(group4["test_predicted_c_max"]) or float(group4["test_predicted_c_min"]) > float(group2["test_predicted_c_max"])
    return {"global_scalar_passes": bool(stage1["stage_passed"]), "zero_residual_branch_preserves_solution": bool(stage2["stage_passed"]), "two_dynamics_ran": True, "two_dynamics_passes": bool(all3["stage_passed"]), "omega2_preserved": float(group2["val_max_h1_abs_error"]) <= GATE and float(group2["test_max_h1_abs_error"]) <= GATE, "clusters_nonoverlapping": distinct, "recommendation": "next: 2 -> 4 dynamics at H=1" if all3["stage_passed"] else "stop at two dynamics"}


def _plot(model: GlobalResidualModel, data: FrequencySet) -> None:
    with torch.no_grad(): prediction, _ = model.components(data.positions)
    plt.figure(figsize=(8, 5))
    for omega, label in ((OMEGA_2, "omega=2"), (OMEGA_4, "omega=4")):
        plt.hist(prediction[data.groups == omega].cpu().numpy(), bins=24, alpha=.6, label=label)
    plt.xlabel("predicted c"); plt.ylabel("trajectory count"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "predicted_c_distributions.png", dpi=170); plt.close()


def main() -> None:
    if BATCH_SIZE % 2:
        raise ValueError("GLOBALCURR_BATCH_SIZE must be even")
    torch.manual_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    single_val = _fixed_grouped(VAL_PER_FREQUENCY, TRAIN_DATA_SEED + 1185100, False); single_test = _fixed_grouped(TEST_PER_FREQUENCY, TEST_DATA_SEED + 1185200, False)
    mixed_val = _fixed_grouped(VAL_PER_FREQUENCY, TRAIN_DATA_SEED + 1186100, True); mixed_test = _fixed_grouped(TEST_PER_FREQUENCY, TEST_DATA_SEED + 1186200, True)
    history: list[dict[str, Any]] = []; finals: list[dict[str, Any]] = []; passes = {stage: False for stage in ("stage1_global_scalar", "stage2_zero_residual", "stage3_two_dynamics")}
    print("Stage 1: training global scalar", flush=True)
    global_model = GlobalResidualModel(GLOBAL_INITIAL_C, with_residual=False).to(dtype=DTYPE)
    h1, best1, _ = _train_global(global_model, single_val); history.extend(h1); f1 = _final_rows(global_model, "stage1_global_scalar", single_val, single_test, best1); finals.extend(f1); passes["stage1_global_scalar"] = _stage_passes(f1, False)
    final_model: GlobalResidualModel = global_model
    if passes["stage1_global_scalar"]:
        print("Stage 1 passed; Stage 2: attaching zero-initialized residual", flush=True)
        residual_model = GlobalResidualModel(float(global_model.c_global.item()), with_residual=True).to(dtype=DTYPE)
        h2, best2 = _train_residual_preservation(residual_model, single_val); history.extend(h2); f2 = _final_rows(residual_model, "stage2_zero_residual", single_val, single_test, best2); finals.extend(f2); passes["stage2_zero_residual"] = _stage_passes(f2, True); final_model = residual_model
        if passes["stage2_zero_residual"]:
            print("Stage 2 passed; Stage 3: training balanced omega=2/4 dynamics", flush=True)
            h3, best3 = _train_two_dynamics(residual_model, mixed_val); history.extend(h3); f3 = _final_rows(residual_model, "stage3_two_dynamics", mixed_val, mixed_test, best3); finals.extend(f3); passes["stage3_two_dynamics"] = _stage_passes(f3, False) and all(float(row["max_h1_abs_error"]) <= GATE for row in f3 if row["frequency_group"] in {"omega_2", "omega_4"}); final_model = residual_model
        else:
            print("Stage 2 failed; stopping before Stage 3", flush=True)
    else:
        print("Stage 1 failed; stopping", flush=True)
    summary = _summary(finals, passes); interpretation = _interpret(summary)
    plot_data = mixed_test if passes["stage2_zero_residual"] else FrequencySet(torch.cat((single_test.positions, mixed_test.positions[mixed_test.groups == OMEGA_4])), torch.cat((single_test.groups, mixed_test.groups[mixed_test.groups == OMEGA_4])))
    _write_csv(OUTPUT_DIR / "summary.csv", summary); _write_csv(OUTPUT_DIR / "training_history.csv", history + finals); _plot(final_model, plot_data)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), **interpretation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
