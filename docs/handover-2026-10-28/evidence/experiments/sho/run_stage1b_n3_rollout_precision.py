#!/usr/bin/env python3
"""Frozen long-horizon rollout audit for N2, N3, and reference operators."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.sho.run_stage1b_edirect_vs_direct import _write_csv
from experiments.sho.run_stage1b_pure_direct_unique_scaling import (
    FRESH_TEST_SEED,
    _make_exact_labeled_split,
)
from experiments.sho.run_stage1b_two_parameter_learned_search import (
    P,
    _posthoc_reference,
)
from experiments.sho.run_stage1b_two_state_omega import DELTA_T, _state_at_time
from experiments.sho.run_stage1b_window_operator_prior_diagnostic import DTYPE


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

N2_ORIGINAL = Path(
    "outputs/sho/stage1B/learned_powell_2d/per_trajectory_results.csv"
)
N2_CONTINUATION = Path(
    "outputs/sho/stage1B/learned_powell_2d_continuation/continuation_per_trajectory.csv"
)
N3_RESULTS = Path(
    "outputs/sho/stage1B/n3_gradient_polishing/n3_per_trajectory.csv"
)
OUTPUT_DIR = Path("outputs/sho/stage1B/n3_rollout_precision")
TRAJECTORY_COUNT = 256
PERIODS = (10, 100, 500, 1000)
METHODS = ("n2", "n3", "reference")
CHUNK_SIZE = 32
UNSTABLE_ABSOLUTE_STATE = 1.0e6
ROLLOUT_CONVENTION = (
    "all operators start from exact observed prefix-end state (x_126,x_127); "
    "first prediction is x_128; horizon steps=ceil(periods*2pi/acos(c_ref/2)); "
    "errors are against direct float64 analytic SHO samples"
)


@torch.jit.script
def _recurrence_rollout(
    parameter_a: torch.Tensor,
    parameter_b: torch.Tensor,
    previous: torch.Tensor,
    current: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    output = torch.empty(
        (parameter_a.shape[0], steps), dtype=parameter_a.dtype, device=parameter_a.device
    )
    for index in range(steps):
        following = parameter_a * previous + parameter_b * current
        output[:, index] = following
        previous = current
        current = following
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_parameters() -> tuple[torch.Tensor, torch.Tensor]:
    original = _read_csv(N2_ORIGINAL)
    continuation = {
        int(row["trajectory_id"]): row
        for row in _read_csv(N2_CONTINUATION)
        if int(row["checkpoint_budget"]) == 1500
    }
    n3 = {
        int(row["trajectory_id"]): row
        for row in _read_csv(N3_RESULTS)
        if int(row["requested_accepted_n3_steps"]) == 20
    }
    if len(original) != TRAJECTORY_COUNT or len(continuation) != 70 or len(n3) != TRAJECTORY_COUNT:
        raise RuntimeError("Persisted N2/N3 endpoint counts are incomplete")
    n2_values: list[tuple[float, float]] = []
    n3_values: list[tuple[float, float]] = []
    for trajectory_id, original_row in enumerate(original):
        if trajectory_id in continuation:
            row = continuation[trajectory_id]
            n2_values.append((float(row["best_a"]), float(row["best_b"])))
        else:
            n2_values.append(
                (float(original_row["final_a"]), float(original_row["final_b"]))
            )
        n3_row = n3[trajectory_id]
        n3_values.append((float(n3_row["best_a"]), float(n3_row["best_b"])))
    return (
        torch.tensor(n2_values, dtype=DTYPE),
        torch.tensor(n3_values, dtype=DTYPE),
    )


def _reconstruct_exact_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, _, hidden_c = _make_exact_labeled_split(
        TRAJECTORY_COUNT, FRESH_TEST_SEED
    )
    generator = torch.Generator().manual_seed(FRESH_TEST_SEED)
    torch.rand(TRAJECTORY_COUNT, generator=generator, dtype=DTYPE)
    torch.randperm(TRAJECTORY_COUNT, generator=generator)
    initial_conditions = 2.0 * torch.rand(
        TRAJECTORY_COUNT, 2, generator=generator, dtype=DTYPE
    ) - 1.0
    omegas = torch.acos(hidden_c / 2.0) / float(DELTA_T)
    prefix_times = (
        torch.arange(P, dtype=DTYPE)[None, :].expand(TRAJECTORY_COUNT, -1)
        * float(DELTA_T)
    )
    audited_positions, _ = _state_at_time(initial_conditions, omegas, prefix_times)
    if not bool(torch.equal(audited_positions, positions)):
        maximum = float(torch.max(torch.abs(audited_positions - positions)))
        raise RuntimeError(f"Exact data reconstruction mismatch: {maximum}")
    return positions, hidden_c, initial_conditions, omegas


def _steps_for_periods(hidden_c: torch.Tensor) -> torch.Tensor:
    phase_per_step = torch.acos(torch.clamp(hidden_c / 2.0, -1.0, 1.0))
    return torch.stack(
        [
            torch.ceil(float(periods) * math.tau / phase_per_step).to(torch.long)
            for periods in PERIODS
        ],
        dim=1,
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _evaluate_rollouts(
    positions: torch.Tensor,
    hidden_c: torch.Tensor,
    initial_conditions: torch.Tensor,
    omegas: torch.Tensor,
    parameters: dict[str, torch.Tensor],
) -> list[dict[str, object]]:
    steps = _steps_for_periods(hidden_c)
    metrics: dict[str, dict[str, np.ndarray]] = {
        method: {
            name: np.zeros((TRAJECTORY_COUNT, len(PERIODS)), dtype=np.float64)
            for name in ("mse", "max_abs_error", "final_abs_error", "unstable")
        }
        for method in METHODS
    }
    gaps = {
        name: np.zeros((TRAJECTORY_COUNT, len(PERIODS)), dtype=np.float64)
        for name in ("mse", "max_abs", "final_abs")
    }

    for start in range(0, TRAJECTORY_COUNT, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, TRAJECTORY_COUNT)
        chunk_steps = steps[start:stop]
        maximum_steps = int(torch.max(chunk_steps[:, -1]))
        future_times = (
            torch.arange(P, P + maximum_steps, dtype=DTYPE)[None, :]
            * float(DELTA_T)
        )
        x0 = initial_conditions[start:stop, 0:1]
        v0 = initial_conditions[start:stop, 1:2]
        omega = omegas[start:stop, None]
        phases = omega * future_times
        truth = x0 * torch.cos(phases) + (v0 / omega) * torch.sin(phases)
        previous = positions[start:stop, P - 2]
        current = positions[start:stop, P - 1]
        predictions: dict[str, torch.Tensor] = {}
        for method in METHODS:
            theta = parameters[method][start:stop]
            prediction = _recurrence_rollout(
                theta[:, 0], theta[:, 1], previous, current, maximum_steps
            )
            predictions[method] = prediction
            error = prediction - truth
            for local_index in range(stop - start):
                trajectory_id = start + local_index
                for horizon_index in range(len(PERIODS)):
                    count = int(chunk_steps[local_index, horizon_index])
                    trajectory_prediction = prediction[local_index, :count]
                    trajectory_error = error[local_index, :count]
                    metrics[method]["mse"][trajectory_id, horizon_index] = float(
                        torch.mean(trajectory_error.square())
                    )
                    metrics[method]["max_abs_error"][trajectory_id, horizon_index] = float(
                        torch.max(torch.abs(trajectory_error))
                    )
                    metrics[method]["final_abs_error"][trajectory_id, horizon_index] = float(
                        torch.abs(trajectory_error[-1])
                    )
                    metrics[method]["unstable"][trajectory_id, horizon_index] = int(
                        (not bool(torch.isfinite(trajectory_prediction).all()))
                        or (
                            float(torch.max(torch.abs(trajectory_prediction)))
                            > UNSTABLE_ABSOLUTE_STATE
                        )
                    )

        gap = predictions["n3"] - predictions["reference"]
        for local_index in range(stop - start):
            trajectory_id = start + local_index
            for horizon_index in range(len(PERIODS)):
                count = int(chunk_steps[local_index, horizon_index])
                trajectory_gap = gap[local_index, :count]
                gaps["mse"][trajectory_id, horizon_index] = float(
                    torch.mean(trajectory_gap.square())
                )
                gaps["max_abs"][trajectory_id, horizon_index] = float(
                    torch.max(torch.abs(trajectory_gap))
                )
                gaps["final_abs"][trajectory_id, horizon_index] = float(
                    torch.abs(trajectory_gap[-1])
                )
        print(f"rollout trajectories {stop}/{TRAJECTORY_COUNT}", flush=True)

    references = parameters["reference"].numpy()
    parameter_errors = {
        method: np.linalg.norm(parameters[method].numpy() - references, axis=1)
        for method in METHODS
    }
    rows: list[dict[str, object]] = []
    for trajectory_id in range(TRAJECTORY_COUNT):
        for method in METHODS:
            previous_values: dict[str, float] | None = None
            for horizon_index, periods in enumerate(PERIODS):
                current_values = {
                    name: float(metrics[method][name][trajectory_id, horizon_index])
                    for name in ("mse", "max_abs_error", "final_abs_error")
                }
                row: dict[str, object] = {
                    "trajectory_id": trajectory_id,
                    "method": method,
                    "periods": periods,
                    "rollout_steps": int(steps[trajectory_id, horizon_index]),
                    "trajectory_mse": current_values["mse"],
                    "maximum_absolute_trajectory_error": current_values["max_abs_error"],
                    "final_time_absolute_error": current_values["final_abs_error"],
                    "unstable_or_diverging": int(
                        metrics[method]["unstable"][trajectory_id, horizon_index]
                    ),
                    "parameter_error_posthoc": parameter_errors[method][trajectory_id],
                    "mse_growth_vs_previous_horizon": (
                        ""
                        if previous_values is None
                        else _ratio(current_values["mse"], previous_values["mse"])
                    ),
                    "max_error_growth_vs_previous_horizon": (
                        ""
                        if previous_values is None
                        else _ratio(
                            current_values["max_abs_error"],
                            previous_values["max_abs_error"],
                        )
                    ),
                    "final_error_growth_vs_previous_horizon": (
                        ""
                        if previous_values is None
                        else _ratio(
                            current_values["final_abs_error"],
                            previous_values["final_abs_error"],
                        )
                    ),
                    "n3_to_reference_mse_ratio": "",
                    "n3_to_reference_max_error_ratio": "",
                    "n3_to_reference_final_error_ratio": "",
                    "n3_reference_gap_mse": "",
                    "n3_reference_max_absolute_gap": "",
                    "n3_reference_final_absolute_gap": "",
                    "n2_to_n3_mse_improvement_factor": "",
                    "n2_to_n3_max_error_improvement_factor": "",
                    "n2_to_n3_final_error_improvement_factor": "",
                    "rollout_convention": ROLLOUT_CONVENTION,
                }
                if method == "n3":
                    reference_values = {
                        name: float(
                            metrics["reference"][name][trajectory_id, horizon_index]
                        )
                        for name in ("mse", "max_abs_error", "final_abs_error")
                    }
                    n2_values = {
                        name: float(metrics["n2"][name][trajectory_id, horizon_index])
                        for name in ("mse", "max_abs_error", "final_abs_error")
                    }
                    row.update(
                        {
                            "n3_to_reference_mse_ratio": _ratio(
                                current_values["mse"], reference_values["mse"]
                            ),
                            "n3_to_reference_max_error_ratio": _ratio(
                                current_values["max_abs_error"],
                                reference_values["max_abs_error"],
                            ),
                            "n3_to_reference_final_error_ratio": _ratio(
                                current_values["final_abs_error"],
                                reference_values["final_abs_error"],
                            ),
                            "n3_reference_gap_mse": gaps["mse"][trajectory_id, horizon_index],
                            "n3_reference_max_absolute_gap": gaps["max_abs"][trajectory_id, horizon_index],
                            "n3_reference_final_absolute_gap": gaps["final_abs"][trajectory_id, horizon_index],
                            "n2_to_n3_mse_improvement_factor": _ratio(
                                n2_values["mse"], current_values["mse"]
                            ),
                            "n2_to_n3_max_error_improvement_factor": _ratio(
                                n2_values["max_abs_error"], current_values["max_abs_error"]
                            ),
                            "n2_to_n3_final_error_improvement_factor": _ratio(
                                n2_values["final_abs_error"],
                                current_values["final_abs_error"],
                            ),
                        }
                    )
                rows.append(row)
                previous_values = current_values
    return rows


def _safe_array(values: list[object]) -> np.ndarray:
    maximum = np.finfo(np.float64).max
    return np.asarray(
        [float(value) if math.isfinite(float(value)) else maximum for value in values]
    )


def _summary_rows(per_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    parameter_summary = {
        method: (
            float(
                np.median(
                    [
                        float(row["parameter_error_posthoc"])
                        for row in per_rows
                        if row["method"] == method and int(row["periods"]) == PERIODS[0]
                    ]
                )
            ),
            max(
                float(row["parameter_error_posthoc"])
                for row in per_rows
                if row["method"] == method and int(row["periods"]) == PERIODS[0]
            ),
        )
        for method in METHODS
    }
    for method in METHODS:
        previous_summary: dict[str, float] | None = None
        for periods in PERIODS:
            selected = [
                row
                for row in per_rows
                if row["method"] == method and int(row["periods"]) == periods
            ]
            mse = np.asarray([float(row["trajectory_mse"]) for row in selected])
            maximum = np.asarray(
                [float(row["maximum_absolute_trajectory_error"]) for row in selected]
            )
            final = np.asarray(
                [float(row["final_time_absolute_error"]) for row in selected]
            )
            current_summary = {
                "median_mse": float(np.median(mse)),
                "worst_mse": float(np.max(mse)),
                "median_max": float(np.median(maximum)),
                "worst_max": float(np.max(maximum)),
                "median_final": float(np.median(final)),
                "worst_final": float(np.max(final)),
            }
            row: dict[str, object] = {
                "method": method,
                "periods": periods,
                "trajectories": len(selected),
                "median_trajectory_mse": current_summary["median_mse"],
                "worst_trajectory_mse": current_summary["worst_mse"],
                "median_maximum_absolute_trajectory_error": current_summary["median_max"],
                "worst_maximum_absolute_trajectory_error": current_summary["worst_max"],
                "median_final_time_absolute_error": current_summary["median_final"],
                "worst_final_time_absolute_error": current_summary["worst_final"],
                "unstable_or_diverging_count": sum(
                    int(item["unstable_or_diverging"]) for item in selected
                ),
                "median_parameter_error_posthoc": parameter_summary[method][0],
                "worst_parameter_error_posthoc": parameter_summary[method][1],
                "median_mse_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(current_summary["median_mse"], previous_summary["median_mse"])
                ),
                "worst_mse_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(current_summary["worst_mse"], previous_summary["worst_mse"])
                ),
                "median_max_error_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(current_summary["median_max"], previous_summary["median_max"])
                ),
                "worst_max_error_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(current_summary["worst_max"], previous_summary["worst_max"])
                ),
                "median_final_error_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(
                        current_summary["median_final"],
                        previous_summary["median_final"],
                    )
                ),
                "worst_final_error_growth_vs_previous_horizon": (
                    ""
                    if previous_summary is None
                    else _ratio(
                        current_summary["worst_final"],
                        previous_summary["worst_final"],
                    )
                ),
                "median_n3_to_reference_mse_ratio": "",
                "worst_n3_to_reference_mse_ratio": "",
                "median_n3_to_reference_max_error_ratio": "",
                "worst_n3_to_reference_max_error_ratio": "",
                "median_n3_to_reference_final_error_ratio": "",
                "worst_n3_to_reference_final_error_ratio": "",
                "median_n3_reference_max_absolute_gap": "",
                "worst_n3_reference_max_absolute_gap": "",
                "median_n3_reference_gap_mse": "",
                "worst_n3_reference_gap_mse": "",
                "median_n3_reference_final_absolute_gap": "",
                "worst_n3_reference_final_absolute_gap": "",
                "median_n2_to_n3_mse_improvement_factor": "",
                "worst_n2_to_n3_mse_improvement_factor": "",
                "median_n2_to_n3_max_error_improvement_factor": "",
                "worst_n2_to_n3_max_error_improvement_factor": "",
                "rollout_convention": ROLLOUT_CONVENTION,
            }
            if method == "n3":
                mappings = (
                    ("n3_to_reference_mse_ratio", "median_n3_to_reference_mse_ratio", "worst_n3_to_reference_mse_ratio"),
                    ("n3_to_reference_max_error_ratio", "median_n3_to_reference_max_error_ratio", "worst_n3_to_reference_max_error_ratio"),
                    ("n3_to_reference_final_error_ratio", "median_n3_to_reference_final_error_ratio", "worst_n3_to_reference_final_error_ratio"),
                    ("n3_reference_max_absolute_gap", "median_n3_reference_max_absolute_gap", "worst_n3_reference_max_absolute_gap"),
                    ("n3_reference_gap_mse", "median_n3_reference_gap_mse", "worst_n3_reference_gap_mse"),
                    ("n3_reference_final_absolute_gap", "median_n3_reference_final_absolute_gap", "worst_n3_reference_final_absolute_gap"),
                    ("n2_to_n3_mse_improvement_factor", "median_n2_to_n3_mse_improvement_factor", "worst_n2_to_n3_mse_improvement_factor"),
                    ("n2_to_n3_max_error_improvement_factor", "median_n2_to_n3_max_error_improvement_factor", "worst_n2_to_n3_max_error_improvement_factor"),
                )
                for source_key, median_key, worst_key in mappings:
                    values = _safe_array([item[source_key] for item in selected])
                    row[median_key] = float(np.median(values))
                    row[worst_key] = float(np.max(values))
            output.append(row)
            previous_summary = current_summary
    return output


def _plot(path: Path, summary_rows: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    colors = {"n2": "tab:orange", "n3": "tab:blue", "reference": "tab:green"}
    for method in METHODS:
        selected = [row for row in summary_rows if row["method"] == method]
        periods = [int(row["periods"]) for row in selected]
        axes[0].loglog(
            periods,
            [float(row["median_trajectory_mse"]) for row in selected],
            marker="o",
            color=colors[method],
            label=method,
        )
        axes[1].loglog(
            periods,
            [float(row["worst_maximum_absolute_trajectory_error"]) for row in selected],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[0].set_xlabel("periods")
    axes[0].set_ylabel("median trajectory MSE")
    axes[1].set_xlabel("periods")
    axes[1].set_ylabel("worst maximum absolute error")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
        axis.set_xticks(PERIODS, labels=[str(value) for value in PERIODS])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    for required in (N2_ORIGINAL, N2_CONTINUATION, N3_RESULTS):
        if not required.exists():
            raise FileNotFoundError(required)
    n2, n3 = _load_parameters()
    positions, hidden_c, initial_conditions, omegas = _reconstruct_exact_data()
    reference = _posthoc_reference(hidden_c)
    parameters = {"n2": n2, "n3": n3, "reference": reference}
    per_rows = _evaluate_rollouts(
        positions, hidden_c, initial_conditions, omegas, parameters
    )
    summary_rows = _summary_rows(per_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "rollout_precision_summary.csv", summary_rows)
    _write_csv(OUTPUT_DIR / "rollout_per_trajectory.csv", per_rows)
    _plot(OUTPUT_DIR / "rollout_error_vs_periods.png", summary_rows)
    print(f"wrote frozen rollout audit to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
