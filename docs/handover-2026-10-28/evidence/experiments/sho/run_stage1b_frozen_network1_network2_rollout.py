"""Frozen Network 1 -> Network 2 end-to-end rollout evaluation."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import torch

import experiments.sho.run_stage1b_network2_powell_like_local_search as network2
import experiments.sho.run_stage1b_region_learned_search as network1
import experiments.sho.run_stage1b_tcn_localization_classifier_diagnostic as data_source
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import (
    DTYPE,
    SEED,
    _write_csv,
)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUTPUT_DIR = Path(
    "outputs/sho/stage1B/frozen_network1_network2_end_to_end_rollout"
)
NETWORK1_CHECKPOINT = network2.FROZEN_ROOT / "best_checkpoint.pt"
NETWORK2_CHECKPOINT = network2.CHECKPOINT_PATH
TEST_TRAJECTORIES = 512
FRESH_TEST_SEED = SEED + 2_160_000
NETWORK2_INFERENCE_SEED = SEED + 2_170_000
STOP_THRESHOLDS = (1e-12, 1e-16, 1e-20)
PERIOD_GRID = (1, 2, 5, 10, 20, 40, 60, 80, 100)
MAX_ABSOLUTE_STATE_FOR_STABILITY = 1e6


def _load_frozen_models() -> tuple[network1.RegionPolicy, network2.ExploreExploitController, dict[str, Any]]:
    if not NETWORK1_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing frozen Network 1: {NETWORK1_CHECKPOINT}")
    if not NETWORK2_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing frozen Network 2: {NETWORK2_CHECKPOINT}")

    network1_payload = torch.load(
        NETWORK1_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if network1_payload.get("horizon") != 20:
        raise RuntimeError("Network 1 checkpoint is not the frozen 20-query model")
    network1_model = network1.RegionPolicy().to(dtype=DTYPE)
    network1_model.load_state_dict(network1_payload["state_dict"])
    network1_model.eval()

    network2_payload = torch.load(
        NETWORK2_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if network2_payload.get("status") != "permanently_frozen_replay_verified":
        raise RuntimeError("Network 2 checkpoint has not passed replay verification")
    verification = network2_payload.get("replay_verification") or {}
    if not bool(verification.get("all_passed", False)):
        raise RuntimeError("Network 2 replay verification is absent or failed")
    if network2_payload.get("model_class") != "ExploreExploitController":
        raise RuntimeError("Unexpected Network 2 checkpoint model class")
    network2_model = network2.ExploreExploitController().to(dtype=DTYPE)
    network2_model.load_state_dict(network2_payload["state_dict"])
    network2_model.eval()
    return network1_model, network2_model, network2_payload


@torch.no_grad()
def _first_crossings(
    rollout: network2.SearchRollout,
) -> dict[float, dict[str, torch.Tensor]]:
    crossings: dict[float, dict[str, torch.Tensor]] = {}
    for threshold in STOP_THRESHOLDS:
        history = rollout.best_loss_history < threshold
        reached = history.any(dim=1)
        first_index = torch.argmax(history.to(torch.long), dim=1)
        c_hat = rollout.best_c_history.gather(
            1, first_index[:, None]
        ).squeeze(1)
        prefix_loss = rollout.best_loss_history.gather(
            1, first_index[:, None]
        ).squeeze(1)
        evaluations = network2.INITIAL_EVALUATIONS + first_index
        crossings[threshold] = {
            "reached": reached,
            "history_index": first_index,
            "evaluations": evaluations,
            "c_hat": c_hat,
            "prefix_loss": prefix_loss,
        }
    return crossings


def _steps_for_periods(c_star: torch.Tensor, periods: float) -> torch.Tensor:
    phase_per_step = torch.acos(torch.clamp(c_star / 2.0, -1.0, 1.0))
    return torch.ceil(periods * math.tau / phase_per_step).to(torch.long)


@torch.no_grad()
def _recurrence_rollout(
    c: torch.Tensor, prefix: torch.Tensor, steps: int
) -> torch.Tensor:
    previous = prefix[:, -2]
    current = prefix[:, -1]
    values: list[torch.Tensor] = []
    for _ in range(steps):
        following = c * current - previous
        values.append(following)
        previous, current = current, following
    return torch.stack(values, dim=1)


def _prefix_max_h1_error(c_hat: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    prediction = c_hat[:, None] * prefix[:, 1:-1] - prefix[:, :-2]
    return torch.abs(prediction - prefix[:, 2:]).max(dim=1).values


def _masked_mse(error: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    cumulative = torch.cumsum(error.square(), dim=1)
    index = (steps - 1).clamp_min(0)
    return cumulative.gather(1, index[:, None]).squeeze(1) / steps.to(DTYPE)


def _masked_max_absolute(error: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(error.shape[1], dtype=torch.long)[None, :]
    valid = indices < steps[:, None]
    return torch.where(valid, torch.abs(error), torch.zeros_like(error)).max(
        dim=1
    ).values


@torch.no_grad()
def _rollout_metrics(
    c_hat: torch.Tensor,
    prefix: torch.Tensor,
    c_star: torch.Tensor,
    reached: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[int, dict[str, float]]]:
    steps_10 = _steps_for_periods(c_star, 10.0)
    steps_100 = _steps_for_periods(c_star, 100.0)
    maximum_steps = int(steps_100.max().item())
    truth = _recurrence_rollout(c_star, prefix, maximum_steps)
    prediction = _recurrence_rollout(c_hat, prefix, maximum_steps)
    error = prediction - truth
    mse_10 = _masked_mse(error, steps_10)
    mse_100 = _masked_mse(error, steps_100)
    maximum_10 = _masked_max_absolute(error, steps_10)
    maximum_100 = _masked_max_absolute(error, steps_100)
    prediction_maximum_100 = _masked_max_absolute(prediction, steps_100)
    finite = torch.isfinite(prediction).all(dim=1)
    unstable = (~finite) | (prediction_maximum_100 > MAX_ABSOLUTE_STATE_FOR_STABILITY)
    metrics = {
        "c_error": torch.abs(c_hat - c_star),
        "h1_error": _prefix_max_h1_error(c_hat, prefix),
        "mse_10": mse_10,
        "mse_100": mse_100,
        "maximum_10": maximum_10,
        "maximum_100": maximum_100,
        "unstable": unstable,
        "steps_10": steps_10,
        "steps_100": steps_100,
    }
    progress: dict[int, dict[str, float]] = {}
    for periods in PERIOD_GRID:
        steps = _steps_for_periods(c_star, float(periods))
        maximum = _masked_max_absolute(error, steps)
        valid_values = maximum[reached]
        progress[periods] = {
            "median": float(torch.median(valid_values).item()),
            "worst": float(valid_values.max().item()),
        }
    return metrics, progress


def _label(threshold: float) -> str:
    return f"1e_{int(round(-math.log10(threshold)))}"


def _summary_row(
    threshold: float,
    crossing: dict[str, torch.Tensor],
    metrics: dict[str, torch.Tensor],
    network1_coverage: float,
    runtime_seconds: float,
) -> dict[str, Any]:
    reached = crossing["reached"]
    count = int(reached.sum().item())
    evaluations = crossing["evaluations"][reached].to(DTYPE)
    row: dict[str, Any] = {
        "stopping_threshold": threshold,
        "test_trajectories": int(reached.shape[0]),
        "success_count": count,
        "success_rate": float(reached.to(DTYPE).mean().item()),
        "network1_safe_region_coverage_posthoc": network1_coverage,
        "median_c_error_posthoc": float(torch.median(metrics["c_error"][reached]).item()),
        "worst_c_error_posthoc": float(metrics["c_error"][reached].max().item()),
        "worst_h1_error": float(metrics["h1_error"][reached].max().item()),
        "median_10_period_mse": float(torch.median(metrics["mse_10"][reached]).item()),
        "worst_10_period_mse": float(metrics["mse_10"][reached].max().item()),
        "median_100_period_mse": float(torch.median(metrics["mse_100"][reached]).item()),
        "worst_100_period_mse": float(metrics["mse_100"][reached].max().item()),
        "worst_absolute_error_10_periods": float(metrics["maximum_10"][reached].max().item()),
        "worst_absolute_error_100_periods": float(metrics["maximum_100"][reached].max().item()),
        "unstable_or_diverging_trajectories": int(metrics["unstable"][reached].sum().item()),
        "median_network2_evaluations": float(torch.median(evaluations).item()),
        "mean_network2_evaluations": float(evaluations.mean().item()),
        "worst_network2_evaluations": float(evaluations.max().item()),
        "fresh_test_seed": FRESH_TEST_SEED,
        "network2_inference_seed": NETWORK2_INFERENCE_SEED,
        "period_definition": "trajectory_specific_posthoc_reference_period",
        "network1_checkpoint": str(NETWORK1_CHECKPOINT),
        "network2_checkpoint": str(NETWORK2_CHECKPOINT),
        "total_runtime_seconds": runtime_seconds,
    }
    row["median_mse_growth_100_over_10"] = (
        row["median_100_period_mse"] / max(row["median_10_period_mse"], 1e-300)
    )
    row["worst_mse_growth_100_over_10"] = (
        row["worst_100_period_mse"] / max(row["worst_10_period_mse"], 1e-300)
    )
    row["worst_absolute_error_growth_100_over_10"] = (
        row["worst_absolute_error_100_periods"]
        / max(row["worst_absolute_error_10_periods"], 1e-300)
    )
    return row


def _trajectory_rows(
    c_star: torch.Tensor,
    network1_rollout: network1.Rollout,
    crossings: dict[float, dict[str, torch.Tensor]],
    metrics_by_threshold: dict[float, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    lower = network1_rollout.candidate_lower[:, -1]
    upper = network1_rollout.candidate_upper[:, -1]
    rows: list[dict[str, Any]] = []
    for trajectory in range(c_star.shape[0]):
        row: dict[str, Any] = {
            "trajectory_index": trajectory,
            "reference_c_star_posthoc": float(c_star[trajectory].item()),
            "network1_safe_lower": float(lower[trajectory].item()),
            "network1_safe_upper": float(upper[trajectory].item()),
            "network1_safe_width": float((upper[trajectory] - lower[trajectory]).item()),
            "network1_safe_region_contains_c_star_posthoc": bool(
                lower[trajectory] <= c_star[trajectory] <= upper[trajectory]
            ),
            "network1_best_observed_c": float(network1_rollout.best_c[trajectory, -1].item()),
            "network1_best_observed_prefix_loss": float(network1_rollout.best_loss[trajectory, -1].item()),
            "rollout_steps_10_periods": int(
                metrics_by_threshold[STOP_THRESHOLDS[0]]["steps_10"][trajectory].item()
            ),
            "rollout_steps_100_periods": int(
                metrics_by_threshold[STOP_THRESHOLDS[0]]["steps_100"][trajectory].item()
            ),
        }
        for threshold in STOP_THRESHOLDS:
            label = _label(threshold)
            crossing = crossings[threshold]
            metrics = metrics_by_threshold[threshold]
            reached = bool(crossing["reached"][trajectory])
            row[f"reached_{label}"] = reached
            row[f"evaluations_to_{label}"] = (
                int(crossing["evaluations"][trajectory].item()) if reached else ""
            )
            row[f"c_hat_at_{label}"] = float(crossing["c_hat"][trajectory].item())
            row[f"prefix_loss_at_{label}"] = float(crossing["prefix_loss"][trajectory].item())
            row[f"c_error_at_{label}_posthoc"] = float(metrics["c_error"][trajectory].item())
            row[f"max_h1_error_at_{label}"] = float(metrics["h1_error"][trajectory].item())
            row[f"mse_10_periods_at_{label}"] = float(metrics["mse_10"][trajectory].item())
            row[f"mse_100_periods_at_{label}"] = float(metrics["mse_100"][trajectory].item())
            row[f"max_abs_error_10_periods_at_{label}"] = float(metrics["maximum_10"][trajectory].item())
            row[f"max_abs_error_100_periods_at_{label}"] = float(metrics["maximum_100"][trajectory].item())
            row[f"unstable_or_diverging_at_{label}"] = bool(metrics["unstable"][trajectory])
        rows.append(row)
    return rows


def _plot(progress_by_threshold: dict[float, dict[int, dict[str, float]]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for threshold in STOP_THRESHOLDS:
        progress = progress_by_threshold[threshold]
        periods = list(PERIOD_GRID)
        label = f"stop below {threshold:g}"
        axes[0].plot(periods, [progress[p]["median"] for p in periods], marker="o", label=label)
        axes[1].plot(periods, [progress[p]["worst"] for p in periods], marker="o", label=label)
    for axis, title in zip(axes, ("Median", "Worst")):
        axis.set_yscale("log")
        axis.set_ylabel(f"{title} cumulative absolute error")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_xlabel("trajectory periods")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "rollout_error_vs_horizon.png", dpi=180)
    plt.close(figure)


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    network1_model, network2_model, checkpoint = _load_frozen_models()
    prefix, _unused_labels, hidden_c = data_source._make_exact_labeled_split(
        TEST_TRAJECTORIES, FRESH_TEST_SEED
    )
    del _unused_labels

    print("running frozen Network 1 on fresh unseen prefixes", flush=True)
    network1_rollout = network1._evaluate_rollout(
        network1_model, prefix, int(checkpoint["frozen_network1_horizon"])
    )
    lower = network1_rollout.candidate_lower[:, -1]
    upper = network1_rollout.candidate_upper[:, -1]
    best_c = network1_rollout.best_c[:, -1]
    best_loss = network1_rollout.best_loss[:, -1]
    initial_u, initial_loss = network2._initial_history(
        prefix, lower, upper, best_c, best_loss
    )

    print("running frozen Network 2 with actual prefix-loss evaluations", flush=True)
    network2_rollout = network2._evaluate(
        network2_model,
        prefix,
        lower,
        upper,
        initial_u,
        initial_loss,
        network2.MAX_LEARNED_QUERIES,
        NETWORK2_INFERENCE_SEED,
    )
    crossings = _first_crossings(network2_rollout)

    # Hidden simulator c is opened only after both frozen searches and stopping
    # decisions are complete; it is used solely for post-hoc rollout scoring.
    c_star = hidden_c
    network1_covered = (c_star >= lower) & (c_star <= upper)
    network1_coverage = float(network1_covered.to(DTYPE).mean().item())
    metrics_by_threshold: dict[float, dict[str, torch.Tensor]] = {}
    progress_by_threshold: dict[float, dict[int, dict[str, float]]] = {}
    for threshold in STOP_THRESHOLDS:
        metrics, progress = _rollout_metrics(
            crossings[threshold]["c_hat"],
            prefix,
            c_star,
            crossings[threshold]["reached"],
        )
        metrics_by_threshold[threshold] = metrics
        progress_by_threshold[threshold] = progress

    runtime = time.perf_counter() - start
    summary_rows = [
        _summary_row(
            threshold,
            crossings[threshold],
            metrics_by_threshold[threshold],
            network1_coverage,
            runtime,
        )
        for threshold in STOP_THRESHOLDS
    ]
    _write_csv(OUTPUT_DIR / "end_to_end_rollout_summary.csv", summary_rows)
    _write_csv(
        OUTPUT_DIR / "per_trajectory_rollout.csv",
        _trajectory_rows(
            c_star, network1_rollout, crossings, metrics_by_threshold
        ),
    )
    _plot(progress_by_threshold)
    print(
        {
            "output_dir": str(OUTPUT_DIR),
            "network1_coverage": network1_coverage,
            "success_1e_20": float(
                crossings[1e-20]["reached"].to(DTYPE).mean().item()
            ),
            "runtime_seconds": runtime,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
