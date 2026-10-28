"""Learn where to search for continuous trajectory-specific SHO dynamics."""

from __future__ import annotations

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch import nn

from experiments.sho.run_stage1b_amortized_c_teacher_anticollapse_diagnostic import (
    P,
    TCNBlock,
)
from experiments.sho.run_stage1b_continuous_c_search_comparison import (
    LOWER,
    UPPER,
    _losses,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    TEST_DATA_SEED,
    TRAIN_DATA_SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path("outputs/sho/stage1B/learned_continuous_c_localization")
REGION_COUNT = 32
TRAIN_N = 4_096
VALIDATION_N = 512
TEST_N = 512
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_STEPS = 5_000
EVAL_INTERVAL = 200
PATIENCE = 2_000
BRENT_XATOL = 1e-14
BRENT_MAXITER = 1_000
NEWTON_MAX_BACKTRACKS = 60
MODEL_SEED = SEED + 1_810_000


@dataclass(frozen=True)
class PrefixData:
    positions: torch.Tensor


@dataclass(frozen=True)
class TeacherData:
    data: PrefixData
    c_star: torch.Tensor
    teacher_prefix_loss: torch.Tensor
    teacher_brent_evaluations: torch.Tensor


class TCNHead(nn.Module):
    """Existing five-block Stage 1B TCN with a configurable output head."""

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.input = nn.Conv1d(1, 32, 3, padding=1)
        self.blocks = nn.Sequential(
            *(TCNBlock(32, dilation) for dilation in [1, 2, 4, 8, 16])
        )
        self.head = nn.Sequential(
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, output_size)
        )

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.silu(
            self.input(prefix.unsqueeze(1))
        )
        features = self.blocks(features).mean(dim=-1)
        return self.head(features)


def _region_edges() -> torch.Tensor:
    return torch.linspace(LOWER, UPPER, REGION_COUNT + 1, dtype=DTYPE)


def _region_labels(c: torch.Tensor) -> torch.Tensor:
    width = (UPPER - LOWER) / REGION_COUNT
    labels = torch.floor((c - LOWER) / width).to(torch.long)
    return labels.clamp(0, REGION_COUNT - 1)


def _sample_unique_continuous_data(n: int, seed: int) -> PrefixData:
    """Stratified continuous draws; hidden c is used only by the simulator."""
    generator = torch.Generator().manual_seed(seed)
    offsets = torch.rand(n, generator=generator, dtype=DTYPE)
    unit = (torch.arange(n, dtype=DTYPE) + offsets) / float(n)
    hidden_c = LOWER + (UPPER - LOWER) * unit
    hidden_c = hidden_c[torch.randperm(n, generator=generator)]
    omegas = torch.acos(hidden_c / 2.0) / float(DELTA_T)
    initial_conditions = 2.0 * torch.rand(
        n, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    minimum_omega = math.acos(UPPER / 2.0) / float(DELTA_T)
    future_steps = math.ceil(
        10.0 * math.tau / (minimum_omega * float(DELTA_T))
    )
    times = torch.arange(P + future_steps + 1, dtype=DTYPE).unsqueeze(0)
    times = times.expand(n, -1) * float(DELTA_T)
    positions, _ = _state_at_time(initial_conditions, omegas, times)
    return PrefixData(positions=positions)


def _trajectory_loss(value: float, positions: torch.Tensor) -> float:
    c = torch.tensor([value], dtype=DTYPE)
    return float(_losses(c, positions[:, :P])[0].item())


def _one_safeguarded_newton_step(
    initial: float,
    positions: torch.Tensor,
    bounds: tuple[float, float],
) -> tuple[float, float, bool, int, str]:
    parameter = torch.tensor(initial, dtype=DTYPE, requires_grad=True)
    objective = _losses(parameter.reshape(1), positions[:, :P])[0]
    gradient = torch.autograd.grad(objective, parameter, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient, parameter)[0]
    current_objective = float(objective.detach().item())
    gradient_value = float(gradient.detach().item())
    hessian_value = float(hessian.detach().item())
    if not math.isfinite(hessian_value) or hessian_value <= 0.0:
        return (
            initial,
            current_objective,
            False,
            0,
            "nonpositive_or_invalid_hessian",
        )
    proposed = -gradient_value / hessian_value
    if not math.isfinite(proposed):
        return initial, current_objective, False, 0, "invalid_newton_step"
    lower, upper = bounds
    scale = 1.0
    for backtracks in range(NEWTON_MAX_BACKTRACKS + 1):
        trial = initial + scale * proposed
        if lower <= trial <= upper and trial != initial:
            trial_objective = _trajectory_loss(trial, positions)
            if trial_objective < current_objective:
                return trial, trial_objective, True, backtracks, "accepted"
        scale *= 0.5
    return (
        initial,
        current_objective,
        False,
        NEWTON_MAX_BACKTRACKS,
        "no_float64_decreasing_step",
    )


def _bounded_brent_newton(
    positions: torch.Tensor,
    bounds: tuple[float, float],
) -> tuple[float, float, int, bool, int, str, bool, str]:
    evaluations = 0

    def objective(value: float) -> float:
        nonlocal evaluations
        evaluations += 1
        return _trajectory_loss(value, positions)

    result = minimize_scalar(
        objective,
        bounds=bounds,
        method="bounded",
        options={"xatol": BRENT_XATOL, "maxiter": BRENT_MAXITER},
    )
    (
        polished_c,
        polished_loss,
        newton_accepted,
        newton_backtracks,
        newton_status,
    ) = _one_safeguarded_newton_step(float(result.x), positions, bounds)
    return (
        polished_c,
        polished_loss,
        evaluations,
        newton_accepted,
        newton_backtracks,
        newton_status,
        bool(result.success),
        str(result.message),
    )


def _make_teachers(data: PrefixData, split: str) -> TeacherData:
    n = data.positions.shape[0]
    c_star = torch.empty(n, dtype=DTYPE)
    prefix_loss = torch.empty(n, dtype=DTYPE)
    evaluations = torch.empty(n, dtype=torch.long)
    start = time.perf_counter()
    for trajectory in range(n):
        (
            c_value,
            loss,
            brent_evaluations,
            _,
            _,
            _,
            success,
            message,
        ) = _bounded_brent_newton(
            data.positions[trajectory : trajectory + 1], (LOWER, UPPER)
        )
        if not success:
            raise RuntimeError(
                f"Teacher Brent failed on {split} trajectory {trajectory}: {message}"
            )
        c_star[trajectory] = c_value
        prefix_loss[trajectory] = loss
        evaluations[trajectory] = brent_evaluations
        if (trajectory + 1) % 256 == 0 or trajectory + 1 == n:
            print(
                f"  {split} teachers {trajectory + 1}/{n} "
                f"({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    return TeacherData(data, c_star, prefix_loss, evaluations)


@torch.no_grad()
def _localization_metrics(
    model: nn.Module, teacher: TeacherData
) -> dict[str, float]:
    logits = model(teacher.data.positions[:, :P])
    labels = _region_labels(teacher.c_star)
    top = torch.topk(logits, k=3, dim=1).indices
    return {
        "cross_entropy": float(
            torch.nn.functional.cross_entropy(logits, labels).item()
        ),
        "top1": float((top[:, 0] == labels).to(DTYPE).mean().item()),
        "top2": float(
            (top[:, :2] == labels[:, None]).any(dim=1).to(DTYPE).mean().item()
        ),
        "top3": float(
            (top == labels[:, None]).any(dim=1).to(DTYPE).mean().item()
        ),
    }


@torch.no_grad()
def _regression_metrics(
    model: nn.Module,
    teacher: TeacherData,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, float]:
    prediction = mean + scale * model(teacher.data.positions[:, :P]).squeeze(-1)
    error = torch.abs(prediction - teacher.c_star)
    return {
        "rmse": float(
            torch.sqrt(torch.mean((prediction - teacher.c_star) ** 2)).item()
        ),
        "median": float(torch.median(error).item()),
        "worst": float(error.max().item()),
    }


def _train_localizer(
    train: TeacherData, validation: TeacherData
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = TCNHead(REGION_COUNT).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    labels = _region_labels(train.c_star)
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    stale = 0
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        indices = torch.randint(
            0, train.data.positions.shape[0], (BATCH_SIZE,), generator=generator
        )
        logits = model(train.data.positions[indices, :P])
        loss = torch.nn.functional.cross_entropy(logits, labels[indices])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            metrics = _localization_metrics(model, validation)
            print(
                f"  localizer step={step}: val CE={metrics['cross_entropy']:.4e}, "
                f"top1={metrics['top1']:.4f}, top3={metrics['top3']:.4f}",
                flush=True,
            )
            if metrics["cross_entropy"] < best_score:
                best_score = metrics["cross_entropy"]
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "validation_cross_entropy": best_score,
                        "regions": REGION_COUNT,
                        "model_seed": MODEL_SEED,
                        "shuffle_seed": MODEL_SEED + 1,
                        "learning_rate": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                    },
                    OUTPUT_DIR / "best_localizer_validation_checkpoint.pt",
                )
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_step": best_step,
        "best_validation_cross_entropy": best_score,
        "training_runtime_seconds": time.perf_counter() - start,
    }


def _train_regressor(
    train: TeacherData,
    validation: TeacherData,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    model = TCNHead(1).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    targets = (train.c_star - mean) / scale
    generator = torch.Generator().manual_seed(MODEL_SEED + 1)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    best_step = 0
    stale = 0
    start = time.perf_counter()
    for step in range(1, MAX_STEPS + 1):
        indices = torch.randint(
            0, train.data.positions.shape[0], (BATCH_SIZE,), generator=generator
        )
        prediction = model(train.data.positions[indices, :P]).squeeze(-1)
        loss = torch.mean((prediction - targets[indices]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            metrics = _regression_metrics(model, validation, mean, scale)
            print(
                f"  regressor step={step}: val RMSE={metrics['rmse']:.4e}, "
                f"median={metrics['median']:.4e}",
                flush=True,
            )
            if metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "validation_rmse": best_score,
                        "output_size": 1,
                        "model_seed": MODEL_SEED,
                        "shuffle_seed": MODEL_SEED + 1,
                        "learning_rate": LEARNING_RATE,
                        "batch_size": BATCH_SIZE,
                    },
                    OUTPUT_DIR / "best_regressor_validation_checkpoint.pt",
                )
                stale = 0
            else:
                stale += EVAL_INTERVAL
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {
        "best_step": best_step,
        "best_validation_c_rmse": best_score,
        "training_runtime_seconds": time.perf_counter() - start,
    }


def _future_metrics(
    c: float, positions: torch.Tensor, c_star: float
) -> tuple[float, float, float]:
    omega = math.acos(max(-1.0, min(1.0, c_star / 2.0))) / float(DELTA_T)
    steps = math.ceil(10.0 * math.tau / (omega * float(DELTA_T)))
    target = positions[P : P + steps]
    h1_error = (
        c * positions[P - 1 : P + steps - 1]
        - positions[P - 2 : P + steps - 2]
        - target
    )
    previous, current = positions[P - 2], positions[P - 1]
    predictions = []
    for _ in range(steps):
        following = c * current - previous
        predictions.append(following)
        previous, current = current, following
    rollout_error = torch.stack(predictions) - target
    return (
        float(torch.abs(h1_error).max().item()),
        float(torch.mean(rollout_error**2).item()),
        float(torch.abs(rollout_error).max().item()),
    )


def _evaluate_test(
    localizer: nn.Module,
    regressor: nn.Module,
    test: TeacherData,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    with torch.no_grad():
        logits = localizer(test.data.positions[:, :P])
        probabilities = torch.softmax(logits, dim=1)
        top = torch.topk(probabilities, k=3, dim=1)
        regression_c = (
            mean
            + scale * regressor(test.data.positions[:, :P]).squeeze(-1)
        )
    labels = _region_labels(test.c_star)
    edges = _region_edges()
    rows: list[dict[str, Any]] = []
    for trajectory in range(test.data.positions.shape[0]):
        c_star = float(test.c_star[trajectory].item())
        true_region = int(labels[trajectory].item())
        predicted_region = int(top.indices[trajectory, 0].item())
        lower = float(edges[predicted_region].item())
        upper = float(edges[predicted_region + 1].item())
        (
            refined_c,
            refined_loss,
            brent_evaluations,
            newton_accepted,
            newton_backtracks,
            newton_status,
            brent_success,
            brent_message,
        ) = _bounded_brent_newton(
            test.data.positions[trajectory : trajectory + 1], (lower, upper)
        )
        localization_correct = predicted_region == true_region
        top2 = bool((top.indices[trajectory, :2] == true_region).any().item())
        top3 = bool((top.indices[trajectory, :3] == true_region).any().item())
        h1, rollout_mse, rollout_error = _future_metrics(
            refined_c, test.data.positions[trajectory], c_star
        )
        regression_value = float(regression_c[trajectory].item())
        regression_h1, regression_mse, regression_rollout = _future_metrics(
            regression_value, test.data.positions[trajectory], c_star
        )
        rows.append(
            {
                "trajectory_index": trajectory,
                "c_star": c_star,
                "teacher_prefix_loss": float(
                    test.teacher_prefix_loss[trajectory].item()
                ),
                "teacher_brent_evaluations": int(
                    test.teacher_brent_evaluations[trajectory].item()
                ),
                "true_region": true_region + 1,
                "predicted_region": predicted_region + 1,
                "predicted_region_lower": lower,
                "predicted_region_upper": upper,
                "localization_correct": localization_correct,
                "true_c_outside_predicted_region": not localization_correct,
                "top1_probability": float(top.values[trajectory, 0].item()),
                "top2_region": int(top.indices[trajectory, 1].item()) + 1,
                "top2_contains_true_region": top2,
                "top3_region": int(top.indices[trajectory, 2].item()) + 1,
                "top3_contains_true_region": top3,
                "refined_c": refined_c,
                "refined_absolute_c_error": abs(refined_c - c_star),
                "refined_prefix_loss": refined_loss,
                "refined_max_h1_error": h1,
                "refined_10_period_mse": rollout_mse,
                "refined_max_rollout_error": rollout_error,
                "inference_brent_evaluations": brent_evaluations,
                "inference_brent_success": brent_success,
                "inference_brent_message": brent_message,
                "inference_newton_accepted": newton_accepted,
                "inference_newton_backtracks": newton_backtracks,
                "inference_newton_status": newton_status,
                "direct_regression_c": regression_value,
                "direct_regression_absolute_c_error": abs(
                    regression_value - c_star
                ),
                "direct_regression_max_h1_error": regression_h1,
                "direct_regression_10_period_mse": regression_mse,
                "direct_regression_max_rollout_error": regression_rollout,
            }
        )

    correct_rows = [row for row in rows if row["localization_correct"]]
    refined_summary: dict[str, Any] = {
        "method": "tcn_region_localization_then_brent_newton",
        "test_trajectories": len(rows),
        "top1_region_accuracy": sum(
            bool(row["localization_correct"]) for row in rows
        )
        / len(rows),
        "top2_region_coverage": sum(
            bool(row["top2_contains_true_region"]) for row in rows
        )
        / len(rows),
        "top3_region_coverage": sum(
            bool(row["top3_contains_true_region"]) for row in rows
        )
        / len(rows),
        "localization_failure_count": sum(
            bool(row["true_c_outside_predicted_region"]) for row in rows
        ),
        "worst_refined_absolute_c_error": max(
            float(row["refined_absolute_c_error"]) for row in rows
        ),
        "median_refined_absolute_c_error": float(
            np.median([row["refined_absolute_c_error"] for row in rows])
        ),
        "worst_refined_prefix_loss": max(
            float(row["refined_prefix_loss"]) for row in rows
        ),
        "worst_refined_h1_error": max(
            float(row["refined_max_h1_error"]) for row in rows
        ),
        "worst_refined_10_period_mse": max(
            float(row["refined_10_period_mse"]) for row in rows
        ),
        "worst_refined_rollout_error": max(
            float(row["refined_max_rollout_error"]) for row in rows
        ),
        "median_inference_brent_evaluations": float(
            np.median([row["inference_brent_evaluations"] for row in rows])
        ),
        "maximum_inference_brent_evaluations": max(
            int(row["inference_brent_evaluations"]) for row in rows
        ),
        "correct_localization_count": len(correct_rows),
        "correct_localization_worst_c_error": max(
            float(row["refined_absolute_c_error"]) for row in correct_rows
        )
        if correct_rows
        else float("nan"),
        "correct_localization_worst_prefix_loss": max(
            float(row["refined_prefix_loss"]) for row in correct_rows
        )
        if correct_rows
        else float("nan"),
        "correct_localization_worst_10_period_mse": max(
            float(row["refined_10_period_mse"]) for row in correct_rows
        )
        if correct_rows
        else float("nan"),
        "correct_localization_newton_failure_count": sum(
            row["inference_newton_status"]
            not in {"accepted", "no_float64_decreasing_step"}
            for row in correct_rows
        ),
    }
    regression_errors = [
        float(row["direct_regression_absolute_c_error"]) for row in rows
    ]
    regression_summary: dict[str, Any] = {
        "method": "tcn_direct_c_regression",
        "test_trajectories": len(rows),
        "worst_direct_regression_c_error": max(regression_errors),
        "median_direct_regression_c_error": float(
            np.median(regression_errors)
        ),
        "direct_regression_c_rmse": float(
            math.sqrt(np.mean(np.square(regression_errors)))
        ),
        "worst_direct_regression_10_period_mse": max(
            float(row["direct_regression_10_period_mse"]) for row in rows
        ),
        "median_direct_regression_10_period_mse": float(
            np.median(
                [row["direct_regression_10_period_mse"] for row in rows]
            )
        ),
        "worst_direct_regression_h1_error": max(
            float(row["direct_regression_max_h1_error"]) for row in rows
        ),
        "worst_direct_regression_rollout_error": max(
            float(row["direct_regression_max_rollout_error"]) for row in rows
        ),
    }
    return rows, refined_summary, regression_summary


def _plot_test(rows: list[dict[str, Any]]) -> None:
    true_c = np.asarray([float(row["c_star"]) for row in rows])
    predicted = np.asarray([int(row["predicted_region"]) for row in rows])
    correct = np.asarray([bool(row["localization_correct"]) for row in rows])
    true_region = np.asarray([int(row["true_region"]) for row in rows])
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.scatter(
        true_c[correct],
        predicted[correct],
        s=13,
        alpha=0.6,
        color="tab:blue",
        label="correct region",
    )
    axis.scatter(
        true_c[~correct],
        predicted[~correct],
        s=25,
        alpha=0.9,
        color="tab:red",
        marker="x",
        label="localization failure",
    )
    order = np.argsort(true_c)
    axis.step(
        true_c[order],
        true_region[order],
        where="mid",
        color="black",
        linewidth=1,
        alpha=0.55,
        label="true region",
    )
    axis.set_xlim(LOWER, UPPER)
    axis.set_ylim(0.5, REGION_COUNT + 0.5)
    axis.set_xlabel(r"teacher $c^*$")
    axis.set_ylabel("predicted search region")
    axis.set_yticks(range(1, REGION_COUNT + 1, 2))
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "true_c_vs_predicted_search_region.png", dpi=180)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating unique continuous-c splits", flush=True)
    train_data = _sample_unique_continuous_data(
        TRAIN_N, TRAIN_DATA_SEED + 1_810_100
    )
    validation_data = _sample_unique_continuous_data(
        VALIDATION_N, TRAIN_DATA_SEED + 1_810_200
    )
    test_data = _sample_unique_continuous_data(
        TEST_N, TEST_DATA_SEED + 1_810_300
    )
    print("Building exact prefix-search teachers", flush=True)
    train = _make_teachers(train_data, "train")
    validation = _make_teachers(validation_data, "validation")
    test = _make_teachers(test_data, "test")
    mean = train.c_star.mean()
    scale = train.c_star.std().clamp_min(1e-12)

    print("Training 32-region TCN localizer", flush=True)
    localizer, localizer_training = _train_localizer(train, validation)
    print("Training matched direct-regression TCN", flush=True)
    regressor, regression_training = _train_regressor(
        train, validation, mean, scale
    )

    validation_localization = _localization_metrics(localizer, validation)
    test_localization = _localization_metrics(localizer, test)
    validation_regression = _regression_metrics(
        regressor, validation, mean, scale
    )
    test_regression = _regression_metrics(regressor, test, mean, scale)
    print("Running region-restricted test refinement", flush=True)
    rows, localization_summary, regression_summary = _evaluate_test(
        localizer, regressor, test, mean, scale
    )
    common = {
        "regions": REGION_COUNT,
        "train_trajectories": TRAIN_N,
        "validation_trajectories": VALIDATION_N,
        "test_trajectories": TEST_N,
        "training_steps_maximum": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "teacher_worst_prefix_loss": float(
            torch.cat(
                (
                    train.teacher_prefix_loss,
                    validation.teacher_prefix_loss,
                    test.teacher_prefix_loss,
                )
            ).max().item()
        ),
    }
    localization_summary.update(
        {
            **common,
            "best_training_step": localizer_training["best_step"],
            "training_runtime_seconds": localizer_training[
                "training_runtime_seconds"
            ],
            "validation_top1_region_accuracy": validation_localization[
                "top1"
            ],
            "validation_top2_region_coverage": validation_localization[
                "top2"
            ],
            "validation_top3_region_coverage": validation_localization[
                "top3"
            ],
            "test_cross_entropy": test_localization["cross_entropy"],
        }
    )
    regression_summary.update(
        {
            **common,
            "best_training_step": regression_training["best_step"],
            "training_runtime_seconds": regression_training[
                "training_runtime_seconds"
            ],
            "validation_direct_regression_c_rmse": validation_regression[
                "rmse"
            ],
            "validation_direct_regression_median_c_error": (
                validation_regression["median"]
            ),
            "validation_direct_regression_worst_c_error": (
                validation_regression["worst"]
            ),
            "test_direct_regression_validation_style_rmse": test_regression[
                "rmse"
            ],
        }
    )
    _write_csv(
        OUTPUT_DIR / "summary.csv",
        [localization_summary, regression_summary],
    )
    _write_csv(OUTPUT_DIR / "per_test_results.csv", rows)
    _plot_test(rows)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "top1": localization_summary["top1_region_accuracy"],
            "top2": localization_summary["top2_region_coverage"],
            "top3": localization_summary["top3_region_coverage"],
            "failures": localization_summary["localization_failure_count"],
            "correct_region_worst_c_error": localization_summary[
                "correct_localization_worst_c_error"
            ],
            "direct_regression_worst_c_error": regression_summary[
                "worst_direct_regression_c_error"
            ],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
